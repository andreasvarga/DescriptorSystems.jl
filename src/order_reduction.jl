"""
    gss2ss(sys; Eshape = "ident", atol = 0, atol1 = atol, atol2 = atol, rtol = nϵ) -> (sysr, r)

Convert the descriptor system `sys = (A-λE,B,C,D)` to an input-output equivalent descriptor system realization 
`sysr = (Ar-λEr,Br,Cr,Dr)` without non-dynamic modes and having the same transfer function matrix.
The resulting `Er` is in the SVD-like form `Er = blockdiag(E1,0)`, with `E1` an `r × r` nonsingular matrix, 
where `r` is the rank of `E`.

The keyword argument `Eshape` specifies the shape of `E1` as follows:

if `Eshape = "ident"` (the default option), `E1` is an identity matrix of order `r` and if `E` is nonsingular, 
then the resulting system `sysr` is a standard state-space system;

if `Eshape = "diag"`, `E1` is a diagonal matrix of order `r`, where the diagonal elements are the 
decreasingly ordered nonzero singular values of `E`;

if `Eshape = "triu"`, `E1` is an upper triangular nonsingular matrix of order `r`. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 

If `Eshape = "triu"`, the reductions of `E` and `A` are performed using rank decisions based on rank revealing 
QR-decompositions with column pivoting.  If `Eshape = "ident"` or `Eshape = "diag"` the reductions are performed
using the more reliable SVD-decompositions.
"""
function gss2ss(sys::DescriptorStateSpace{T}; Eshape = "ident", atol::Real = zero(real(T)), atol1::Real = atol,  
                atol2::Real = atol, rtol::Real =  sys.nx*eps(real(float(one(real(T)))))*iszero(max(atol1,atol2))) where T

    # finish for a standard state space system or pure gain
    n = sys.nx
    (n == 0 || sys.E == I) && (return sys, n) 

    T <: BlasFloat ? T1 = T : T1 = promote_type(Float64,T)

    Ar, Er, Br, Cr, Dr = dssdata(T1,sys) 
        
    # exploit the upper triangular form of a nonsingular E
    epsm = eps(real(T1))
    if  istriu(Er) && rcond(UpperTriangular(Er),atol2) > n*epsm
        Eshape == "triu" && (return sys, n) 
        if Eshape == "ident"
            eltype(Er) <: Complex && 
               (return dss(ldiv!(UpperTriangular(Er),Ar), ldiv!(UpperTriangular(Er),Br), Cr, Dr, Ts = sys.Ts), n)
            # make diagonal elements of E positive
            indneg = (diag(Er) .< 0)
            if any(indneg)
                Ar[indneg,:] = -Ar[indneg,:]
                Er[indneg,:] = -Er[indneg,:]
                Br[indneg,:] = -Br[indneg,:]
            end
            e2 = UpperTriangular(sqrt(Er)) 
            ldiv!(e2,Ar)
            rdiv!(Ar,e2)
            ldiv!(e2,Br)
            rdiv!(Cr,e2)
            return dss(Ar, Br, Cr, Dr, Ts = sys.Ts), n
        end
    end
        
    # Using orthogonal/unitary transformation matrices Q and Z, reduce the 
    # matrices A, E, B and C to the forms
    # 
    #               [ At11  At12 At13 ]                  [ Et11  0  0 ]   
    # Ar = Q'*A*Z = [ At21  At22  0   ] ,  Er = Q'*E*Z = [  0    0  0 ] , 
    #               [ At31   0    0   ]                  [  0    0  0 ]
    #
    #             [ Bt1 ] 
    # Br = Q'*B = [ Bt2 ] ,  Cr = C*Z = [ Ct1  Ct2  Ct3 ]
    #             [ Bt3 ]
    #
    fast = (Eshape == "triu")
    rE, rA22  = _svdlikeAE!(Ar, Er, nothing, nothing, Br, Cr, 
                     fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = false, withZ = false)
    # employ state residualzation formulas to remove non-dynamic modes
    if rA22 > 0
        i1 = 1:rE
        i2 = rE+1:rE+rA22
        # make At22 = I
        fast ? (A22 = UpperTriangular(Ar[i2,i2])) : (A22 = Diagonal(Ar[i2,i2]))
        ldiv!(A22,view(Ar,i2,i1))
        ldiv!(A22,view(Br,i2,:))
        # apply simplified residualization formulas
        Dr -= Cr[:,i2]*Br[i2,:]
        Br[i1,:] -= Ar[i1,i2]*Br[i2,:]
        Cr[:,i1] -= Cr[:,i2]*Ar[i2,i1]
        Ar[i1,i1] -= Ar[i1,i2]*Ar[i2,i1]
        ir = [i1; rE+rA22+1:n]
    else
        i1 = 1:rE
        ir = 1:n
    end
    # bring E to the required shape
    if fast || Eshape == "diag"
       # Eshape == "triu" or Eshape == "diag": we are already done
       return dss(view(Ar,ir,ir),view(Er,ir,ir),view(Br,ir,:),view(Cr,:,ir),Dr, Ts = sys.Ts), rE
    elseif Eshape == "ident" 
        tid = Diagonal(1 ./sqrt.(diag(Er[i1,i1]))) 
        Er[i1,i1] = eye(T1,rE)
        lmul!(tid,view(Ar,i1,ir))
        rmul!(view(Ar,ir,i1),tid)
        lmul!(tid,view(Br,i1,:))
        rmul!(view(Cr,:,i1),tid)
        return rE+rA22 == n ?  (dss(view(Ar,ir,ir),view(Br,ir,:),view(Cr,:,ir),Dr, Ts = sys.Ts), rE) : 
                               (dss(view(Ar,ir,ir),view(Er,ir,ir),view(Br,ir,:),view(Cr,:,ir),Dr, Ts = sys.Ts), rE)
    else
        error("improper shape option for E")
    end
    
    # end GSS2SS
end
    
    
"""
    sysr = gir(sys; finite = true, infinite = true, contr = true, obs = true, noseig = false,
               fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol = nϵ) 

Compute for a descriptor system `sys = (A-λE,B,C,D)` of order `n` a reduced order descriptor system  
`sysr = (Ar-λEr,Br,Cr,Dr)` of order `nr ≤ n` such that `sys` and `sysr` have the same transfer function matrix, i.e., 

             -1                    -1
     C*(λE-A)  *B + D = Cr*(λEr-Ar)  *Br + Dr .
     
The least possible order `nr` is achieved if `finite = true`, `infinite = true`, `contr = true`, 
`obs = true` and `nseig = true`. Such a realization is called `minimal` and satisfies:

     (1) rank[Br Ar-λEr] = nr for all finite λ (finite controllability)

     (2) rank[Br Er] = nr (infinite controllability)

     (3) rank[Ar-λEr; Cr] = nr for all finite λ (finite observability)

     (4) rank[Er; Cr] = nr (infinite observability)

     (5) Ar-λEr has no simple infinite eigenvalues

A realization satisfying only conditions (1)-(4) is called `irreducible` and is computed by default. 

Some reduction steps can be skipped by appropriately selecting the keyword arguments
`contr`, `obs`, `finite`, `infinite` and `nseig`. 

If `contr = false`, then the controllability conditions (1) and (2) are not enforced. 
If `contr = true` and `finite = true`, then the finite controllability condition (1) is enforced. 
If `contr = true` and `finite = false`, then the finite controllability condition (1) is not enforced. 
If `contr = true` and `infinite = true`, then the infinite controllability condition (2) is enforced. 
If `contr = true` and `infinite = false`, then the infinite controllability condition (2) is not enforced. 

If `obs = false`, then observability condition (3) and (4) are not enforced.
If `obs = true` and `finite = true`, then the finite observability condition (3) is enforced.
If `obs = true` and `finite = false`, then the finite observability condition (3) is not enforced.
If `obs = true` and `infinite = true`, then the infinite observability condition (4) is enforced.
If `obs = true` and `infinite = false`, then the infinite observability condition (4) is not enforced.

If `nseig = true`, then condition (5) on the lack of simple infinite eigenvalues is also enforced. 

To enforce conditions (1)-(4), the `Procedure GIR` in `[1, page 328]` is employed, which performs 
orthogonal similarity transformations on the matrices of the original realization `(A-λE,B,C,D)` 
to obtain an irreducible realization using structured pencil reduction algorithms. 
To enforce condition (5), residualization formulas (see, e.g., `[1, page 329]`) are employed which
involves matrix inversions. 

The underlying pencil manipulation algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 
The default relative tolerance is `nϵ`, where `ϵ` is the working _machine epsilon_ 
and `n` is the order of the system `sys`.  
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 

[1] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function gir(SYS::DescriptorStateSpace{T}; atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
             rtol::Real =  SYS.nx*eps(real(float(one(real(T)))))*iszero(max(atol1,atol2)), 
             fast::Bool = true, finite::Bool = true, infinite::Bool = true, 
             contr::Bool = true, obs::Bool = true, noseig::Bool = false) where T
    if SYS.E == I
        A, B, C = lsminreal(SYS.A, SYS.B, SYS.C; fast = fast, atol = atol1, rtol = rtol, contr = contr, obs = obs) 
        T1 = eltype(A)
        return DescriptorStateSpace{T1}(A, I, B, C, copy_oftype(SYS.D,T1), SYS.Ts)
    else     
        A, E, B, C, D = lsminreal2(SYS.A, SYS.E, SYS.B, SYS.C, SYS.D; 
            fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, finite = finite, infinite = infinite, 
            contr = contr, obs = obs, noseig = noseig) 
        return DescriptorStateSpace{eltype(A)}(A, E, B, C, D, SYS.Ts)
    end
end
"""
    sysr = gminreal(sys; contr = true, obs = true, noseig = true, fast = true, 
                    atol = 0, atol1 = atol, atol2 = atol, rtol = nϵ) 

Compute for a descriptor system `sys = (A-λE,B,C,D)` of order `n` a reduced order descriptor system  
`sysr = (Ar-λEr,Br,Cr,Dr)` of order `nr ≤ n` such that `sys` and `sysr` have the same transfer function matrix, i.e., 

             -1                    -1
     C*(λE-A)  *B + D = Cr*(λEr-Ar)  *Br + Dr .
     
The least possible order `nr` is achieved if `contr = true`, `obs = true` and `nseig = true`. 
Such a realization is called `minimal` and satisfies:

     (1) rank[Br Ar-λEr] = nr for all finite λ (finite controllability)

     (2) rank[Br Er] = nr (infinite controllability)

     (3) rank[Ar-λEr; Cr] = nr for all finite λ (finite observability)

     (4) rank[Er; Cr] = nr (infinite observability)

     (5) Ar-λEr has no simple infinite eigenvalues

A realization satisfying only conditions (1)-(4) is called `irreducible`. 

Some reduction steps can be skipped by appropriately selecting the keyword arguments
`contr`, `obs` and `nseig`. 

If `contr = false`, then the controllability conditions (1) and (2) are not enforced. 

If `obs = false`, then observability condition (3) and (4) are not enforced.

If `nseig = false`, then condition (5) on the lack of simple infinite eigenvalues is not enforced. 

To enforce conditions (1)-(4), orthogonal similarity transformations are performed on 
the matrices of the original realization `(A-λE,B,C,D)` to obtain an irreducible realization using
structured pencil reduction algorithms, as the fast versions of the reduction techniques of the 
full row rank pencil [B A-λE] and full column rank pencil [A-λE;C] proposed in [1]. 
To enforce condition (5), residualization formulas (see, e.g., `[2, page 329]`) are employed which
involves matrix inversions. 

The underlying pencil manipulation algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.
The default relative tolerance is `nϵ`, where `ϵ` is the working _machine epsilon_ 
and `n` is the order of the system `sys`.  
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 

[1] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[2] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function gminreal(SYS::DescriptorStateSpace{T}; atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
    rtol::Real =  SYS.nx*eps(real(float(one(real(T)))))*iszero(max(atol1,atol2)), 
    fast::Bool = true, contr::Bool = true, obs::Bool = true, noseig::Bool = true) where T
    if SYS.E == I
        A, B, C = lsminreal(SYS.A, SYS.B, SYS.C; fast = fast, atol = atol1, rtol = rtol, contr = contr, obs = obs) 
        T1 = eltype(A)
        return DescriptorStateSpace{T1}(A, I, B, C, copy_oftype(SYS.D,T1), SYS.Ts)
    else     
        A, E, B, C, D = lsminreal(SYS.A, SYS.E, SYS.B, SYS.C, SYS.D; 
            fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, 
            contr = contr, obs = obs, noseig = noseig) 
        return DescriptorStateSpace{eltype(A)}(A, E, B, C, D, SYS.Ts)
    end
end
"""
    gbalmr(sys, balance = false, matchdc = false, ord = missing, atolhsv = 0, rtolhsv = nϵ, 
           atolmin = atolhsv, rtolmin = rtolhsv, 
           atol = 0, atol1 = atol, atol2 = atol, rtol, fast = true) -> (sysr, hs)

Compute for a proper and stable descriptor system `sys = (A-λE,B,C,D)` with the transfer function
matrix `G(λ)`, a reduced order realization `sysr = (Ar-λEr,Br,Cr,Dr)` and the vector `hs` of decreasingly 
ordered Hankel singular values of the system `sys`. If `balance = true`, a balancing-based approach
is used to determine a reduced order minimal realization 
of the form `sysr = (Ar-λI,Br,Cr,Dr)`. For a continuous-time system `sys`, the resulting realization `sysr`
is balanced, i.e., the controllability and observability grammians are equal and diagonal. 
If additonally `matchdc = true`, the resulting `sysr` is computed using state rezidualization formulas 
(also known as _singular perturbation approximation_) which additionally preserves the DC-gain of `sys`. 
In this case, the resulting realization `sysr` is balanced (for both continuous- and discrete-time systems).
If `balance = false`, an enhanced accuracy balancing-free approach is used to determine the 
reduced order system `sysr`. 

If `ord = nr`, the resulting order of `sysr` is `min(nr,nrmin)`, where `nrmin` is the order of a minimal  
realization of `sys` determined as the number of Hankel singular values exceeding `max(atolmin,rtolmin*HN)`, with
`HN`, the Hankel norm of `G(λ)`. If `ord = missing`, the resulting order is chosen as the number of Hankel 
singular values exceeding `max(atolhsv,rtolhsv*HN)`. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`,  
the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `nϵ`, where `ϵ` is the working _machine epsilon_ 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol` and `atol2 = atol`. 

If `E` is singular, the uncontrollable infinite eigenvalues of the pair `(A,E)` and
the non-dynamic modes are elliminated using minimal realization techniques.
The rank determinations in the performed reductions
are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`. 

Method:  For the order reduction of a standard system, the balancing-free method of [1] or 
the balancing-based method of [2] are used. For a descriptor system the balancing related order reduction 
methods of [3] are used. To preserve the DC-gain of the original system, the singular perturbation 
approximation method of [4] is used in conjunction with the balancing-based or balancing-free
approach of [5]. 

References

[1] A. Varga. 
    Efficient minimal realization procedure based on balancing.
    In A. El Moudni, P. Borne, and S.G. Tzafestas (Eds.), 
    Prepr. of the IMACS Symp. on Modelling and Control of Technological 
    Systems, Lille, France, vol. 2, pp.42-47, 1991.

[2] M. S. Tombs and I. Postlethwaite. 
    Truncated balanced realization of a stable non-minimal state-space 
    system. Int. J. Control, vol. 46, pp. 1319–1330, 1987.

[3] T. Stykel. 
    Gramian based model reduction for descriptor systems. 
    Mathematics of Control, Signals, and Systems, 16:297–319, 2004.

[4] Y. Liu Y. and B.D.O. Anderson 
    Singular Perturbation Approximation of Balanced Systems,
    Int. J. Control, Vol. 50, pp. 1379-1405, 1989.

[5] Varga A.
    Balancing-free square-root algorithm for computing singular perturbation approximations.
    Proc. 30-th IEEE CDC,  Brighton, Dec. 11-13, 1991, Vol. 2, pp. 1062-1065.
"""   
function gbalmr(sys::DescriptorStateSpace{T}; balance::Bool = false, matchdc::Bool = false, fast::Bool = true, 
    ord::Union{Int,Missing} = missing, atolhsv::Real = zero(real(T)), 
    rtolhsv::Real = sqrt(eps(real(float(one(T)))))*iszero(atolhsv), atolmin::Real = atolhsv, rtolmin::Real = rtolhsv, 
    offset::Real = sqrt(eps(float(real(T)))), atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
    rtol::Real = (size(sys.A,1)*eps(real(float(one(T)))))*iszero(min(atol1,atol2))) where T

    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    s2eps = offset      
    disc = !iszero(sys.Ts)
    standsys = sys.E == I
    ONE = one(T1)
       
    if  standsys
        n = order(sys)
        # for a non-dynamic system, we set the Hankel norm to zero,
        # but the Hankel singular values are empty
        n == 0 && (return sys, zeros(real(T1),0))
        # reduce the system to Schur coordinate form
        SF = schur(sys.A)
        # check stability
        ((disc && maximum(abs.(SF.values)) >= 1-s2eps) || (!disc && maximum(real(SF.values)) >= -s2eps)) &&
              error("The system sys is unstable")
        bs = SF.Z'*sys.B
        cs = sys.C*SF.Z
        S = plyaps(SF.T, bs; disc = disc)
        R = plyaps(SF.T', cs'; disc = disc)
        SV = svd!(R*S); hs = SV.S
    else
        # eliminate uncontrollable infinite eigenvalues and non-dynamic modes if possible
        if LinearAlgebra.LAPACK.gecon!('1', LinearAlgebra.LAPACK.getrf!(copy(sys.E))[1],opnorm(sys.E,1)) < s2eps
           sys = gir(sys, finite = false, noseig = true, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)
           # the resulting sys may still be improper
           LinearAlgebra.LAPACK.gecon!('1', LinearAlgebra.LAPACK.getrf!(copy(sys.E))[1],opnorm(sys.E,1)) < s2eps &&
               error("The system sys is not proper")
        end
        n = order(sys)
        # for a non-dynamic system, we set the Hankel norm to zero,
        # but the Hankel singular values are empty
        n == 0 && (return sys, zeros(real(T1),0))
        # reduce the system to generalized Schur coordinate form
        SF = schur(sys.A,sys.E)
        ((disc && maximum(abs.(SF.values)) >= 1-s2eps) || (!disc && maximum(real(SF.values)) >= -s2eps)) &&
              error("The system sys is unstable") 
        bs = SF.Q'*sys.B
        cs = sys.C*SF.Z
        S = plyaps(SF.S, SF.T, bs; disc = disc)
        R = plyaps(SF.S', SF.T', cs'; disc = disc)
        SV = svd!(R*UpperTriangular(SF.T)*S); hs = SV.S
    end
    # determine the order nrmin of a minimal realization
    atolhsv >= 0 || error("The threshold atolhsv must be non-negative")
    rtolhsv >= 0 || error("The threshold rtolhsv must be non-negative")
    atolmin >= 0 || error("The threshold atolmin must be non-negative")
    rtolmin >= 0 || error("The threshold rtolmin must be non-negative")
    nrmin = count(hs .> max(atolmin,rtolmin*hs[1]))
    # determine the order nr of the reduced model 
    if ismissing(ord)
       nr = min(nrmin, count(hs .> max(atolhsv,rtolhsv*hs[1])))
    else
       ord >= 0 || error("Desired order ord must be non-negative, got instead $ord")
       nr = min(nrmin,ord)
    end
    nr == 0 && (return dss(sys.D,Ts = sys.Ts), hs)
    i1 = 1:nr
    if balance
       # apply balancing-based formulas: resulting Er = I
       if matchdc && nr < nrmin
          indmin = 1:nrmin
          hsi2 = Diagonal(1 ./sqrt.(view(hs,indmin)))
          Tl = lmul!(R',view(SV.U,:,indmin))*hsi2
          Tr = lmul!(S,SV.V[:,indmin])*hsi2 
          standsys ? amin = Tl'*SF.T*Tr : amin = Tl'*SF.S*Tr
          bmin = Tl'*bs
          cmin = cs*Tr
          i2 = nr+1:nrmin
          Ar = view(amin,i1,i1)
          Br = view(bmin,i1,:)
          Cr = view(cmin,:,i1)
          Dr = copy_oftype(sys.D,T1)
          disc && (amin[i2,i2] -= I) 
          LUF = lu!(view(amin,i2,i2))
          ldiv!(LUF,view(amin,i2,i1))
          ldiv!(LUF,view(bmin,i2,:))
          # apply state residualization formulas
          mul!(Dr,view(cmin,:,i2),view(bmin,i2,:),-ONE, ONE)
          mul!(Br,view(amin,i1,i2),view(bmin,i2,:),-ONE, ONE)
          mul!(Cr,view(cmin,:,i2),view(amin,i2,i1),-ONE, ONE)
          mul!(Ar,view(amin,i1,i2),view(amin,i2,i1),-ONE, ONE)
          # return the minimal balanced system
          return dss(Ar, I, Br, Cr, Dr, Ts = sys.Ts), hs
       else
          hsi2 = Diagonal(1 ./sqrt.(view(hs,i1)))
          Tl = lmul!(R',view(SV.U,:,i1))*hsi2
          Tr = lmul!(S,SV.V[:,i1])*hsi2
          # return the minimal balanced system
          return dss(standsys ? Tl'*SF.T*Tr : Tl'*SF.S*Tr, I, Tl'*bs, cs*Tr, sys.D, Ts = sys.Ts), hs
       end
    else
        # apply balancing-free formulas
        if nr < n 
            if matchdc
                i2 = nr+1:nrmin
                Tl = [Matrix(qr!(lmul!(R',view(SV.U,:,i1))).Q) Matrix(qr!(lmul!(R',view(SV.U,:,i2))).Q)]
                Tr = [Matrix(qr!(lmul!(S,SV.V[:,i1])).Q) Matrix(qr!(lmul!(S,SV.V[:,i2])).Q)]
                #standsys ? (amin = Tl'*SF.T*Tr; emin = Tl'*Tr) : (amin = Tl'*SF.S*Tr; emin = Tl'*SF.T*Tr)
                standsys ? amin = Tl'*SF.T*Tr : amin = Tl'*SF.S*Tr
                bmin = Tl'*bs
                cmin = cs*Tr
                Ar = view(amin,i1,i1)
                Er = standsys ? view(Tl,:,i1)'*view(Tr,:,i1) : view(Tl,:,i1)'*UpperTriangular(SF.T)*view(Tr,:,i1)
                Br = view(bmin,i1,:)
                Cr = view(cmin,:,i1)
                Dr = copy_oftype(sys.D,T1)
                disc && (amin[i2,i2] -= standsys ? view(Tl,:,i2)'*view(Tr,:,i2) : view(Tl,:,i2)'*UpperTriangular(SF.T)*view(Tr,:,i2))
                LUF = lu!(view(amin,i2,i2))
                ldiv!(LUF,view(amin,i2,i1))
                ldiv!(LUF,view(bmin,i2,:))
                # apply state residualization formulas
                mul!(Dr,view(cmin,:,i2),view(bmin,i2,:),-ONE, ONE)
                mul!(Br,view(amin,i1,i2),view(bmin,i2,:),-ONE, ONE)
                mul!(Cr,view(cmin,:,i2),view(amin,i2,i1),-ONE, ONE)
                mul!(Ar,view(amin,i1,i2),view(amin,i2,i1),-ONE, ONE)
                standsys || (return dss(Ar, Er, Br, Cr, Dr, Ts = sys.Ts), hs)
                # determine a standard reduced system if sys is a standard system 
                SV = svd!(Er)
                di2 = Diagonal(1 ./sqrt.(SV.S))
                return dss(di2*SV.U'*Ar*SV.Vt'*di2, I, di2*(SV.U'*Br), (Cr*SV.Vt')*di2, Dr, Ts = sys.Ts), hs
            else
                Tl = Matrix(qr!(lmul!(R',view(SV.U,:,i1))).Q)
                Tr = Matrix(qr!(lmul!(S,SV.V[:,i1])).Q)
                # build the minimal system
                standsys || (return dss(Tl'*SF.S*Tr, Tl'*SF.T*Tr, Tl'*bs, cs*Tr, sys.D, Ts = sys.Ts), hs)
                # determine a standard reduced system if sys is a standard system 
                SV = svd!(Tl'*Tr)
                di2 = Diagonal(1 ./sqrt.(SV.S))
                return dss(di2*SV.U'*Tl'*SF.T*Tr*SV.Vt'*di2, I, di2*SV.U'*(Tl'*bs), (cs*Tr)*SV.Vt'*di2, sys.D, Ts = sys.Ts), hs
            end
         else
            return sys, hs  # keep original system if order is preserved
         end
    end
    # end GBALMR
end
