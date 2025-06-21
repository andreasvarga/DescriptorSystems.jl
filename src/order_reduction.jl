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

    T1 = T <: BlasFloat ? T : promote_type(Float64,T)

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

     dss2ss(sys[, x0 = 0]; state-mapping = false, simple_infeigs = true, fast = true, atol1, atol2, rtol) 
               -> (sysr, xr0, Mx, Mu)

Return for a proper descriptor system `sys = (A-λE,B,C,D)` and initial state `x0`, 
the equivalent reduced order standard system `sysr = (Ar-λI,Br,Cr,Dr)` and 
the corresponding reduced consistent initial state `xr0`.

If `state_mapping = true`, the state mapping matrices `Mx` and `Mu` are also determined such that 
the values `x(t)` and `xr(t)` of the state vectors of the systems `sys` and `sysr`, respectively,
and the input vector `u(t)` are related as `x(t) = Mx*xr(t)+Mu*u(t)`.
In this case, higher order uncontrollable infinite eigenvalues can be eliminated if `simple_infeigs = false`.

By default, `state_mapping = false` and `Mx = nothing` and `Mu = nothing`. 
In this case, higher order uncontrollable or unobservable infinite eigenvalues 
can be eliminated if `simple_infeigs = false`. 

By default, `simple_infeigs = true`, and simple infinite eigenvalues for the pair `(A,E)` are assumed and eliminated. 

The underlying pencil manipulation algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true` (default), or the SVD-decomposition,
if `fast = false`. The rank decision based on the SVD-decomposition is generally more reliable, 
but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2` and `rtol` specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `n` is the order of the square matrices `A` and `E`, and  `ϵ` is the working machine epsilon. 
"""
function dss2ss(sys::DescriptorStateSpace{T}, x0::Vector = zeros(T,sys.nx); state_mapping::Bool = false, simple_infeigs::Bool = true,  
                   fast::Bool = true, atol::Real = zero(float(real(T))), atol1::Real = atol, atol2::Real = atol,  
                   rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2))) where T
   T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
   n = sys.nx
   m = sys.nu
   nullx0 = iszero(x0)
   sys.E == I && (return sys, copy_oftype(x0,T1), state_mapping ? I : nothing, state_mapping ? zeros(T1,n,m) : nothing)

   # eliminate uncontrollable/unobservable infinite and nonzero eigenvalues
   simple_infeigs || state_mapping || 
      ((sys, L, Z) = gir_lrtran(sys; fast, rtran = !nullx0, finite = false, noseig = false, atol1, atol2, rtol) )   
       
   A, E, B, C, D = dssdata(T1,sys)
   n1, m = size(B) 
   if n1 < n
      state_mapping && (@warn "state mapping disabled being not feasible: try simple_infeigs = true"; state_mapping = false)
      # adjust initial condition
      nullx0 ? x0 = zeros(T1,n1) : x0 = Z[:,1:n1]'*x0 
   end
   
   ONE = one(T1)
   state_mapping || (Mx = nothing; Mu = nothing)
   if simple_infeigs
      Z = Matrix{T1}(I,n1,n1) 
      n, rA22  = _svdlikeAE!(A, E, nothing, Z, B, C; fast, atol1, atol2, rtol, withQ = false)
      n+rA22 == n1 || error("The system is possibly improper: try with simple_infeigs = false")  
      if rA22 == 0
         # get rid of E matrix, exploit upper triangular or diagonal shape
         fast ? (F = UpperTriangular(E)) : (F = Diagonal(E))
         ldiv!(F,A); ldiv!(F,B)
         state_mapping && (Mx = Z; Mu = zeros(T1,n,m))
         return dss(A, B, C, D, Ts = sys.Ts), Z'*x0, Mx, Mu
      else
         i1 = 1:n
         i2 = n+1:n1
         # adjust initial condition
         Z1 = view(Z,:,i1)
         Z2 = view(Z,:,i2)
         A11 = view(A,i1,i1)
         E11 = view(E,i1,i1)
         A12 = view(A,i1,i2)
         A21 = view(A,i2,i1)
         B1 = view(B,i1,:)
         B2 = view(B,i2,:)
         C1 = view(C,:,i1)
         C2 = view(C,:,i2)
         xt = Z1'*x0 
         # make A22 = I, exploit upper triangular or diagonal shape
         fast ? (A22 = UpperTriangular(A[i2,i2])) : (A22 = Diagonal(A[i2,i2]))
         ldiv!(A22,A21)  # inv(A22)*A21 -> A21
         ldiv!(A22,B2)   # inv(A22)*B2 -> B2
         # apply simplified residualization formulas
         mul!(D, C2, B2, -ONE, ONE)     # D -> D - C2*B2
         mul!(B1, A12, B2, -ONE, ONE)   # B1 -> B1 - A12*B2
         mul!(C1, C2, A21, -ONE, ONE)   # C1 -> C1 - C2*A21
         mul!(A11, A12, A21, -ONE, ONE) # A11 -> A11 - A12*A21
         # state_mapping && rank(B2; atol = atol1, rtol) < rA22 &&  
         #    (@warn "state mapping disabled being not feasible"; state_mapping = false; Mx = nothing; Mu = nothing)
         state_mapping && (mul!(Z1, Z2, A21, ONE, ONE); Mx = copy(Z1); Mu = -Z2*B2)   
         # quick exit in constant case  
         n == 0 && (return dss(A11, B1, C1, D, Ts = sys.Ts), xt, Mx, Mu)
         # get rid of E matrix, exploit upper triangular or diagonal shape
         fast ? (F = UpperTriangular(E11)) : (F = Diagonal(E11))
         ldiv!(F,A11); ldiv!(F,B1)
         return dss(A11, B1, C1, D, Ts = sys.Ts), xt, Mx, Mu
      end   
   else
      # separate infinite eigenvalues in the trailing part; leading E11 is upper triangular
      Z = Matrix{T1}(I,n1,n1) 
      _, blkdims = MatrixPencils.fisplit!(A, E, nothing, Z, B, C; fast, finite_infinite = true, atol1, atol2, rtol, withQ = false) 
      nf, ni = blkdims
      if ni == 0
         # get rid of E matrix, exploit upper triangular shape
         F = UpperTriangular(E)
         ldiv!(F,A); ldiv!(F,B)
         state_mapping && (Mx = Z; Mu = zeros(T1,n,m))
         return dss(A, B, C, D, Ts = sys.Ts), Z'*x0, Mx, Mu
      end
      i1 = 1:nf
      # separate uncontrollable infinite eigenvalues
      i2 = nf+1:nf+ni
      # save trailing matrices of A, E, B and C 
      Ai = A[i2,i2]
      Ei = E[i2,i2]
      Bi = B[i2,:]
      Ci = C[:,i2]
      Qi = nothing
      Zi = Matrix{T}(I,ni,ni)
      Qi, Zi, _, _, niuc = sklf_rightfin!(view(E,i2,i2), view(A,i2,i2), view(B,i2,:), view(C,:,i2);
                                          fast, atol1, atol2, rtol, withQ = false, withZ = true) 
      if niuc == 0
         # restore original matrices if all infinite eigenvalues are controllable
         A[i2,i2] = copy(Ai)
         E[i2,i2] = copy(Ei)
         B[i2,:] = copy(Bi)
         C[:,i2] = copy(Ci)
      else
         # apply the left transformation Zi to Z2, A12 and E12 
         Z[:,i2] = view(Z,:,i2)*Zi
         A[i1,i2] = view(A,i1,i2)*Zi
         E[i1,i2] = view(E,i1,i2)*Zi
      end
      
      # refine structure information
      nic = ni-niuc
      i2 = nf+1:nf+nic
      if norm(view(E,i2,i2),Inf) <= atol2 + rtol*norm(E,Inf)
         # only simple controllable infinite eigenvalues
         Z1 = view(Z,:,i1)
         Z2 = view(Z,:,i2)
         A11 = view(A,i1,i1)
         A12 = view(A,i1,i2)
         B1 = view(B,i1,:)
         B2 = view(B,i2,:)
         C1 = view(C,:,i1)
         C2 = view(C,:,i2)
         E11 = view(E,i1,i1)
         E12 = view(E,i1,i2)
         # xt = Z1'*x0 
         # make A22 = I
         A22 = UpperTriangular(A[i2,i2])
         ldiv!(A22,B2)   # inv(A22)*B2 -> B2
         # make E11 = I
         F = UpperTriangular(E11)
         # inv(E1)[A11 A12] -> [A11 A12]; inv(E1)*E12 -> E12; inv(E1)*B1 -> B1
         ldiv!(F,view(A,i1,:)); ldiv!(F,E12); ldiv!(F,B1)
         # make leading part of E block diagonal (i.e., diag(I,0))
         mul!(C2, C1, E12, -ONE, ONE)   # C2 -> C2 - C1*E12
         mul!(A12, A11, E12, -ONE, ONE)   # A12 -> A12 - A11*E12
         # apply simplified residualization formulas
         mul!(D, C2, B2, -ONE, ONE)     # D -> D - C2*B2
         mul!(B1, A12, B2, -ONE, ONE)   # B1 -> B1 - A12*B2
         xt = Z1'*x0 + E12*(Z2'*x0)     # determine consistent reduced initial state
         mul!(Z2, Z1, E12, -ONE, ONE)   # Z2 -> Z2 - Z1*E12
         state_mapping && (Mx = copy(Z1); Mu = -Z2*B2)   
         return dss(A11, B1, C1, D, Ts = sys.Ts), xt, Mx, Mu
      else
         error("controllable higher order infinite eigenvalues present")
      end
   end
   # end DSS2SS
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
    gir_lrtran(sys; ltran = false, rtran = false, finite = true, infinite = true, contr = true, obs = true, 
             noseig = false, fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol = nϵ) -> (sysr, Q, Z)

This is a special version of the function `gir` to additionally determine the left and right 
transformation matrices `Q = [Q1 Q2]` and `Z = [Z1 Z2]`, respectively, such that the matrices `Ar`, `Er`, `Br` and `Cr` of the resulting
descriptor system `sysr = (Ar-λEr,Br,Cr,Dr)` are given by `Ar = Q1'*A*Z1`, `Er = Q1'*E*Z1`, `Br = Q1'*B`, `Cr = C*Z1`, 
where the number of columns of `Q1` and `Z1` is equal to the order of matrix `Ar`. `Q` and `Z` result orthogonal if `noseig = false`. 
`Q = nothing` if `ltran = false` and `Z = nothing` if `rtran = false`. 
See [`gir`](@ref) for details on the rest of keyword parameters.
"""
function gir_lrtran(SYS::DescriptorStateSpace{T}; ltran::Bool = false, rtran::Bool = false, atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                    rtol::Real =  SYS.nx*eps(real(float(one(real(T)))))*iszero(max(atol1,atol2)), 
                    fast::Bool = true, finite::Bool = true, infinite::Bool = true, 
                    contr::Bool = true, obs::Bool = true, noseig::Bool = false) where T
    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    A, E, B, C, D = dssdata(T1,SYS)

    E == I && (E = eye(T,size(A,1)))
    A, E, B, C, D, Q, Z, = lsminreal2_lrtran(A, E, B, C, D; withQ = ltran, withZ = rtran, 
                                             fast, atol1, atol2, rtol, finite, infinite, contr, obs, noseig) 
    return dss(A, E, B, C, D, Ts = SYS.Ts), Q, Z
end
"""
    sysr = gminreal(sys; contr = true, obs = true, noseig = true, prescale, fast = true, 
                    atol = 0, atol1 = atol, atol2 = atol, rtol = nϵ) 

Compute for a descriptor system `sys = (A-λE,B,C,D)` of order `n` a reduced order descriptor system  
`sysr = (Ar-λEr,Br,Cr,Dr)` of order `nr ≤ n` such that `sys` and `sysr` have the same transfer function matrix, i.e., 

             -1                    -1
     C*(λE-A)  *B + D = Cr*(λEr-Ar)  *Br + Dr .

If `prescale = true`, a preliminary balancing of the descriptor system matrices is performed. 
The default setting is `prescale = gbalqual(sys) > 10000`, where `gbalqual(sys)` is the 
scaling quality of the descriptor system model `sys` (see [`gbalqual`](@ref)). 
     
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
function gminreal(sys::DescriptorStateSpace{T}; prescale = gbalqual(sys) > 10000, atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
    rtol::Real =  100*sys.nx*eps(real(float(one(real(T)))))*iszero(max(atol1,atol2)), 
    fast::Bool = true, contr::Bool = true, obs::Bool = true, noseig::Bool = true) where T
    SYS = prescale ? gprescale(sys)[1] : sys     
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
    gbalmr(sys::DescriptorStateSpace, pext = 0, balance = false, matchdc = false, ord = missing, offset = √ϵ,
           atolhsv = 0, rtolhsv = nϵ, atolmin = atolhsv, rtolmin = rtolhsv, 
           atol = 0, atol1 = atol, atol2 = atol, rtol, fast = true) -> (sysr, hs)

Compute for a dense proper and stable descriptor system `sys = (A-λE,B,C,D)` with the transfer function
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

If the keyword argument `pext` is nonzero, then the trailing `pext` system outputs are not included in the determination of the reduced
order model. However, all state coordinate transformations are also performed on these outputs. 

If `ord = nr`, the resulting order of `sysr` is `min(nr,nrmin)`, where `nrmin` is the order of a minimal  
realization of `sys` determined as the number of Hankel singular values exceeding `max(atolmin,rtolmin*HN)`, with
`HN`, the Hankel norm of `G(λ)`. If `ord = missing`, the resulting order is chosen as the number of Hankel 
singular values exceeding `max(atolhsv,rtolhsv*HN)`. 

To check the stability of the eigenvalues of the pencil `A-λE`, the real parts of eigenvalues must be less than `-β`
for a continuous-time system or 
the moduli of eigenvalues must be less than `1-β` for a discrete-time system, where `β` is the stability domain boundary offset.  
The offset  `β` to be used can be specified via the keyword parameter `offset = β`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

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
function gbalmr(sys::DescriptorStateSpace{T}; balance::Bool = false, matchdc::Bool = false, pext::Int = 0,
    fast::Bool = true, ord::Union{Int,Missing} = missing, atolhsv::Real = zero(real(T)), 
    rtolhsv::Real = sqrt(eps(real(float(one(T)))))*iszero(atolhsv), atolmin::Real = atolhsv, rtolmin::Real = rtolhsv, 
    offset::Real = sqrt(eps(float(real(T)))), atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
    rtol::Real = (size(sys.A,1)*eps(real(float(one(T)))))*iszero(min(atol1,atol2))) where T

    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    s2eps = offset      
    disc = !iszero(sys.Ts)
    standsys = sys.E == I
    ONE = one(T1)
    n = order(sys)
    p = size(sys.D,1)-pext
    p >= 0 || throw(ArgumentError("pext must not exceed the number of system outputs"))
        
    if  standsys
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
        R = plyaps(SF.T', view(cs,1:p,:)'; disc = disc)
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
        R = plyaps(SF.S', SF.T', view(cs,1:p,:)'; disc = disc)
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
"""
    gbalmr(sys::SparseDescriptorStateSpace, pext = 0, balance = false, matchdc = false, ord = missing, 
           atolhsv = 0, rtolhsv = nϵ, atolmin = atolhsv, rtolmin = rtolhsv, abstol = 1e-12, reltol = 0, 
           Trsave = false, Tlsave = false, maxiter = 100, shifts = missing, cyclic = false) -> (sysr, hs, info)

Compute for a sparse proper and stable descriptor system `sys = (A-λE,B,C,D)` with the transfer function
matrix `G(λ)`, a reduced order realization `sysr = (Ar-λEr,Br,Cr,Dr)` and the vector `hs` of decreasingly 
ordered relevant Hankel singular values of the system `sys`. If `balance = true`, a balancing-based approach
is used to determine a reduced order minimal realization 
of the form `sysr = (Ar-λI,Br,Cr,Dr)`. For a continuous-time system `sys`, the resulting realization `sysr`
is balanced, i.e., the controllability and observability grammians are equal and diagonal. 
If additonally `matchdc = true`, the resulting `sysr` is computed using state rezidualization formulas 
(also known as _singular perturbation approximation_) which additionally preserves the DC-gain of `sys`. 
In this case, the resulting realization `sysr` is balanced (for both continuous- and discrete-time systems).
If `balance = false`, an enhanced accuracy balancing-free approach is used to determine the 
reduced order system `sysr`. 

The function `gbalmr` can be also used if `sys` has the type `DescriptorStateSpaceExt`.

If the keyword argument `pext` is nonzero, then the trailing `pext` system outputs are not included in the determination of the reduced
order model. However, all state coordinate transformations are also performed on these outputs. 

If `ord = nr`, the resulting order of `sysr` is `min(nr,nrmin)`, where `nrmin` is the order of a minimal  
realization of `sys` determined as the number of Hankel singular values exceeding `max(atolmin,rtolmin*HN)`, with
`HN`, the Hankel norm of `G(λ)`. If `ord = missing`, the resulting order is chosen as the number of Hankel 
singular values exceeding `max(atolhsv,rtolhsv*HN)`. 

_Method:_  For the order reduction of a standard system, the balancing-free method of [1] or 
the balancing-based method of [2] are used. For a descriptor system the balancing related order reduction 
methods of [3] are used. To preserve the DC-gain of the original system, the singular perturbation 
approximation method of [4] is used in conjunction with the balancing-based or balancing-free
approach of [5]. 

The controlability and observability gramians are determined in factored forms `S*S'` and `R*R'`, respectively,
where `S` and `R` are low rank matrices. The Hankel singular values `hs` are computed as the singular values of the product `R'*E*S`.
For the computation of factors `S` and `R`, the low-rank ADI (LR-ADI) method with enhancements proposed in [6] 
are employed.  
The projection matrices `Tl` and `Tr` used to generate the matrices 
of the reduced order models as `Er = Tl'*E*Tr`, `Ar = Tl'*A*Tr`, `Br = Tl'*B`, `Cr = C*Tr`, 
are computed from the singular value decomposition of the product `R'*E*S`. 
If the keyword arguments `Tlsave = true` and `Trsave = true`, then the named touple `info` contains in `info.Tl` and `info.Tr` 
the respective projection matrices.     

For the convergence tests used in the LR-ADI method, the keyword argument `abstol` (default: `abstol = 1e-12`) 
can be used to specify the tolerance on the normalized residuals, while the keyword argument
`reltol` (default: `reltol = 0`) can be used to specify the tolerance for the relative changes of the solution. 
The keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 100`).
The keyword argument `nshifts` specifies the desired number of shifts to be used in an iteration cycle (default: `nshifts = 6`). 
The keyword argument `shifts` can be used to provide a pre-calculated set of complex conjugated shifts to be used
to start the iterations (default: `shifts = missing`).    
If `cyclic = true`, the cyclic low-rank method of [7] is used, with the
pre-calculated shifts provided in the keyword argument `shifts`. 

_Note:_ There is no check of the stability of the eigenvalues of the pencil `A-λE` implemented. 
For an unstable model the LR-ADI methods does not converge and either an error message is issued or the maximum
number of allowed iterations are reached. 

_References_

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

[6] P. Kürschner. Efficient Low-Rank Solution of Large-Scale Matrix Equations. 
    Dissertation, Otto-von-Guericke-Universität, Magdeburg, Germany, 2016. Shaker Verlag,

[7] T. Penzl, A cyclic low-rank Smith method for large sparse Lyapunov equations, 
    SIAM J. Sci. Comput. 21 (4) (1999) 1401–1418.    
"""   
function gbalmr(sys::Union{DescriptorStateSpaceExt{T},SparseDescriptorStateSpace{T}}; balance::Bool = false, matchdc::Bool = false, pext::Int = 0,
    ord::Union{Int,Missing} = missing, atolhsv::Real = zero(real(T)), 
    rtolhsv::Real = sqrt(eps(real(float(one(T)))))*iszero(atolhsv), atolmin::Real = atolhsv, rtolmin::Real = rtolhsv, 
    abstol = 1e-12, reltol = 0, Trsave = false, Tlsave = false, 
    maxiter = 100, shifts = missing, nshifts = 6, cyclic = false) where {T}
    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 

    disc = !iszero(sys.Ts)
    standsys = sys.E == I
    ONE = one(T1)

    p = size(sys.D,1)-pext
    p >= 0 || throw(ArgumentError("pext must not exceed the number of system outputs"))
       
    n = order(sys); 

    # for a non-dynamic system, we set the Hankel norm to zero,
    # but the Hankel singular values are empty
    n == 0 && (return sys, zeros(real(T1),0))
    # reduce the system to generalized Schur coordinate form
    S = 0; R = 0; info = nothing
    try 
        if sys.Ts != 0
           S, info = plyapdi(sys.A, sys.E, sys.B; abstol, reltol, maxiter, shifts, nshifts, cyclic)
           R = plyapdi(sys.A', sys.E', view(sys.C,1:p,:)'; abstol, reltol, maxiter, cyclic, shifts = cyclic ? shifts : info.used_shifts)[1]
        else
           S, info = plyapci(sys.A, sys.E, sys.B; abstol, reltol, maxiter, shifts, nshifts, cyclic)
           R = plyapci(sys.A', sys.E', view(sys.C,1:p,:)'; abstol, reltol, maxiter, cyclic, shifts = cyclic ? shifts : info.used_shifts)[1]
        end
    catch
       error("the system is possibly unstable")
    end
    SV = svd!(R'*Matrix(sys.E*S)); hs = SV.S

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
    nr == 0 && (return dss(Matrix(sys.D),Ts = sys.Ts), hs, (Tr = nothing, Tl = nothing))
    i1 = 1:nr
    if balance
       # apply balancing-based formulas: resulting Er = I
       if matchdc && nr < nrmin
          indmin = 1:nrmin
          hsi2 = Diagonal(1 ./sqrt.(view(hs,indmin)))
          Tl = (R*view(SV.U,:,indmin))*hsi2
          Tr = (S*view(SV.V,:,indmin))*hsi2
          amin = Tl'*sys.A*Tr
          bmin = Tl'*sys.B
          cmin = sys.C*Tr
          i2 = nr+1:nrmin
          Ar = view(amin,i1,i1)
          Br = view(bmin,i1,:)
          Cr = view(cmin,:,i1)
          Dr = copy_oftype(Matrix(sys.D),T1)
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
          return dss(Ar, I, Br, Cr, Dr, Ts = sys.Ts), hs, (Tr = Trsave ? Tr : nothing, Tl = Tlsave ? Tl : nothing)
       else
          hsi2 = Diagonal(1 ./sqrt.(view(hs,i1)))
          Tl = (R*view(SV.U,:,i1))*hsi2
          Tr = (S*view(SV.V,:,i1))*hsi2
          # return the minimal balanced system
          return dss(Tl'*sys.A*Tr, I, Tl'*sys.B, sys.C*Tr, Matrix(sys.D), Ts = sys.Ts), hs, 
                (Tr = Trsave ? Tr : nothing, Tl = Tlsave ? Tl : nothing, used_shifts = info.used_shifts)
       end
    else
        # apply balancing-free formulas
       if nr < n 
            if matchdc
                i2 = nr+1:nrmin
                Tl = [Matrix(qr!(R*view(SV.U,:,i1)).Q) Matrix(qr!(R*view(SV.U,:,i2)).Q)]
                Tr = [Matrix(qr!(S*SV.V[:,i1]).Q) Matrix(qr!(S*SV.V[:,i2]).Q)]
                #standsys ? (amin = Tl'*SF.T*Tr; emin = Tl'*Tr) : (amin = Tl'*SF.S*Tr; emin = Tl'*SF.T*Tr)
                amin = Tl'*sys.A*Tr
                bmin = Tl'*sys.B
                cmin = sys.C*Tr
                Ar = view(amin,i1,i1)
                Er = standsys ? view(Tl,:,i1)'*view(Tr,:,i1) : view(Tl,:,i1)'*sys.E*view(Tr,:,i1)
                Br = view(bmin,i1,:)
                Cr = view(cmin,:,i1)
                Dr = copy_oftype(Matrix(sys.D),T1)
                disc && (amin[i2,i2] -= standsys ? view(Tl,:,i2)'*view(Tr,:,i2) : view(Tl,:,i2)'*sys.E*view(Tr,:,i2))
                LUF = lu!(view(amin,i2,i2))
                ldiv!(LUF,view(amin,i2,i1))
                ldiv!(LUF,view(bmin,i2,:))
                # apply state residualization formulas
                mul!(Dr,view(cmin,:,i2),view(bmin,i2,:),-ONE, ONE)
                mul!(Br,view(amin,i1,i2),view(bmin,i2,:),-ONE, ONE)
                mul!(Cr,view(cmin,:,i2),view(amin,i2,i1),-ONE, ONE)
                mul!(Ar,view(amin,i1,i2),view(amin,i2,i1),-ONE, ONE)
            else
                Tl = Matrix(qr!(R*view(SV.U,:,i1)).Q)
                Tr = Matrix(qr!(S*SV.V[:,i1]).Q)
                Ar = Tl'*sys.A*Tr; Er = Tl'*sys.E*Tr; Br = Tl'*sys.B; Cr = sys.C*Tr
            end
            # build the minimal system
            SV = svd!(Er)
            if standsys
               # determine a standard reduced system if sys is a standard system 
               di2 = Diagonal(1 ./sqrt.(SV.S))
               return dss(di2*SV.U'*Ar*SV.Vt'*di2, I, di2*SV.U'*Br, Cr*SV.Vt'*di2, Matrix(sys.D), Ts = sys.Ts), hs, 
                          (Tr = Trsave ? view(Tr,:,1:nr)*(SV.Vt'*di2) : nothing, Tl = Tlsave ? view(Tl,:,1:nr)*(SV.U*di2) : nothing, used_shifts = info.used_shifts)
            else
               # determine a descriptor reduced system with diagonal E if sys is a standard system 
               return dss(SV.U'*Ar*SV.Vt', Diagonal(SV.S), SV.U'*Br, Cr*SV.Vt', Matrix(sys.D), Ts = sys.Ts), hs, 
                          (Tr = Trsave ? view(Tr,:,1:nr)*SV.Vt' : nothing, Tl = Tlsave ? view(Tl,:,1:nr)*SV.U : nothing, used_shifts = info.used_shifts)
            end
         else
            return sys, hs, (Tr = nothing, Tl = nothing, used_shifts = missing)  # keep original system if order is preserved
         end
    end
    # end GBALMR
end

function lsminreal2_lrtran(A::AbstractMatrix, E::AbstractMatrix, 
                    B::AbstractVecOrMat, C::AbstractMatrix, D::AbstractVecOrMat; 
                    withQ::Bool = false, withZ::Bool = false, 
                    atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                    rtol::Real =  (size(A,1)+1)*eps(real(float(one(real(eltype(A))))))*iszero(max(atol1,atol2)), 
                    fast::Bool = true, finite::Bool = true, infinite::Bool = true, 
                    contr::Bool = true, obs::Bool = true, noseig::Bool = true)
   #
   # lsminreal2_ltran(A, E, B, C, D; withQ = false, withZ = false, fast = true, atol1 = 0, atol2 = 0, rtol, 
   #                  finite = true, infinite = true, contr = true, obs = true, noseig = true) 
   #                  -> (Ar, Er, Br, Cr, Dr, Q, Z, nuc, nuo, nse)

   # This is a special version of lsminreal2 to also determine the left and right transformation matrices 
   # Q = [Q1 Q2] and Z = [Z1 Z2], respectively, such that the matrices Ar, Er, Br, and Cr of the resulting descriptor system 
   # (Ar-λEr,Br,Cr,Dr) are given by Ar = Q1'*A*Z1, Er = Q1'*E*Z1, Br = Q1'*B, Cr = C*Z1, where the number of columns of Q1 and Z1 
   # is equal to the order of matrix Ar. Q and Z result orthogonal if noseig = false. 
   # Q = nothing if withQ = false and Z = nothing if withZ = false. 
   n = LinearAlgebra.checksquare(A)
   (n,n) != size(E) && throw(DimensionMismatch("A and E must have the same dimensions"))
   p, m = size(D,1), size(D,2)
   n1, m1 = size(B,1), size(B,2)
   (n,m) == (n1, m1) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
   (p,n) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))
   T = promote_type(eltype(A), eltype(E), eltype(B), eltype(C), eltype(D))
   T <: BlasFloat || (T = promote_type(Float64,T)) 
   ONE = one(T)       

   A1 = copy_oftype(A,T)   
   E1 = copy_oftype(E,T)
   B1 = copy_oftype(B,T)
   C1 = copy_oftype(C,T)
   D1 = copy_oftype(D,T)  

   withQ ? Q = Matrix{T}(I,n,n) : Q = nothing
   withZ ? Z = Matrix{T}(I,n,n) : Z = nothing

   n == 0 && (return A1, E1, B1, C1, D1, Q, Z, 0, 0, 0)

   # save system matrices
   Ar = copy(A1)
   Br = copy(B1)
   Cr = copy(C1)
   Dr = copy(D1)
   Er = copy(E1)
   ir = 1:n
   iz1 = ir; iz2 = n+1:n; 
   if finite
      if contr  
         m == 0 &&  (ir = 1:0; return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, Q, Z, n, 0, 0)
         Q1, Z1, _, nr, nfuc = sklf_rightfin!(Ar, Er, Br, Cr; fast, atol1, atol2, rtol, withQ, withZ) 
         if nfuc > 0
            ir = 1:nr
            # save intermediary results
            A1 = Ar[ir,ir]
            E1 = Er[ir,ir]
            B1 = Br[ir,:]
            C1 = Cr[:,ir]
            iz1 = ir; iz2 = nr+1:n
            withQ && (Q = copy(Q1))
            withZ && (Z = copy(Z1))
         else
            # restore original matrices 
            Ar = copy(A1)
            Er = copy(E1)
            Br = copy(B1)
            Cr = copy(C1)
         end
      else
         nfuc = 0
         nr = n
      end
      if obs 
         p == 0 &&  (ir = 1:0; return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, 
                                      withQ ? Q[:,[iz1;iz2]] : Q, withZ ? Z[:,[iz1;iz2]] : Z, nfuc, nr, 0)
         Q1, Z1, _, no, nfuo = sklf_leftfin!(view(Ar,ir,ir), view(Er,ir,ir), view(Cr,:,ir), view(Br,ir,:); 
                                           fast, atol1, atol2, rtol, withQ, withZ) 
         if nfuo > 0
             iz2 = [ir[1:end-no];iz2]
             ir = ir[end-no+1:end]
             # save intermediary results
             A1 = Ar[ir,ir]
             E1 = Er[ir,ir]
             B1 = Br[ir,:]
             C1 = Cr[:,ir]
             withQ && (Q[:,iz1] = Q[:,iz1]*Q1) 
             withZ && (Z[:,iz1] = Z[:,iz1]*Z1)
             iz1 = ir 
          else
             # restore saved matrices
             Ar[ir,ir] = A1
             Er[ir,ir] = E1
             Br[ir,:] = B1
             Cr[:,ir] = C1
         end
      else
         nfuo = 0
      end
   else
      nfuc = 0
      nfuo = 0
   end
   if infinite
      if contr  
         m == 0 &&  (ir = 1:0; return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, 
                                      withQ ? Q[:,[iz1;iz2]] : Q, withZ ? Z[:,[iz1;iz2]] : Z, n, 0, 0)
         Q1, Z1, _, nr, niuc = sklf_rightfin!(view(Er,ir,ir), view(Ar,ir,ir), view(Br,ir,:), view(Cr,:,ir); 
                                           fast, atol1, atol2, rtol, withQ, withZ) 
         if niuc > 0
            iz2 = [ir[nr+1:end];iz2]
            ir = ir[1:nr]
            # save intermediary results
            A1 = Ar[ir,ir]
            E1 = Er[ir,ir]
            B1 = Br[ir,:]
            C1 = Cr[:,ir]
            withQ && (Q[:,iz1] = Q[:,iz1]*Q1) 
            withZ && (Z[:,iz1] = Z[:,iz1]*Z1)
            iz1 = ir 
        else
            # restore original matrices 
            Ar[ir,ir] = A1
            Er[ir,ir] = E1
            Br[ir,:] = B1
            Cr[:,ir] = C1
         end
      else
         niuc = 0
      end
      if obs 
        p == 0 &&  (ir = 1:0; return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, 
                                      withQ ? Q[:,[iz1;iz2]] : Q, withZ ? Z[:,[iz1;iz2]] : Z, niuc, nr, 0)
         Q1, Z1, _, no, niuo = sklf_leftfin!(view(Er,ir,ir), view(Ar,ir,ir), view(Cr,:,ir), view(Br,ir,:); 
                                           fast, atol1, atol2, rtol, withQ, withZ) 
         if niuo > 0
             iz2 = [ir[1:end-no];iz2]
             ir = ir[end-no+1:end]
             # save intermediary results
             A1 = Ar[ir,ir]
             E1 = Er[ir,ir]
             B1 = Br[ir,:]
             C1 = Cr[:,ir]
             withQ && (Q[:,iz1] = Q[:,iz1]*Q1) 
             withZ && (Z[:,iz1] = Z[:,iz1]*Z1)
             iz1 = ir 
         else
             # restore saved matrices
             Ar[ir,ir] = A1
             Er[ir,ir] = E1
             Br[ir,:] = B1
             Cr[:,ir] = C1
         end
      else
          niuo = 0
      end
   else
      niuc = 0
      niuo = 0
   end
   nuc = nfuc+niuc
   nuo = nfuo+niuo
   if noseig
      nm = length(ir)
      withQ ? Q1 = Matrix{T}(I,nm,nm) : Q1 = nothing
      withZ ? Z1 = Matrix{T}(I,nm,nm) : Z1 = nothing
      rE, rA22  = _svdlikeAE!(view(Ar,ir,ir), view(Er,ir,ir), Q1, Z1, view(Br,ir,:), view(Cr,:,ir); 
                              fast, atol1, atol2, rtol, withQ, withZ)
      if rA22 > 0
         withQ && (Q[:,iz1] = Q[:,iz1]*Q1) 
         withZ && (Z[:,iz1] = Z[:,iz1]*Z1)
         i1 = ir[1:rE]
         i2 = ir[rE+1:rE+rA22]
         # make A22 = I
         fast ? (A22 = UpperTriangular(Ar[i2,i2])) : (A22 = Diagonal(Ar[i2,i2]))
         ldiv!(A22,view(Ar,i2,i1))
         ldiv!(A22,view(Br,i2,:))
         withQ && rdiv!(view(Q,:,i2),A22')
         # apply simplified residualization formulas
         mul!(Dr, view(Cr,:,i2), view(Br,i2,:), -ONE, ONE)               # Dr -= Cr[:,i2]*Br[i2,:]
         mul!(view(Br,i1,:), view(Ar,i1,i2), view(Br,i2,:), -ONE, ONE)   # Br[i1,:] -= Ar[i1,i2]*Br[i2,:]
         mul!(view(Cr,:,i1), view(Cr,:,i2), view(Ar,i2,i1), -ONE, ONE)   # Cr[:,i1] -= Cr[:,i2]*Ar[i2,i1]
         mul!(view(Ar,i1,i1), view(Ar,i1,i2), view(Ar,i2,i1), -ONE, ONE) # Ar[i1,i1] -= Ar[i1,i2]*Ar[i2,i1]
         withQ &&  mul!(view(Q,:,i1), view(Q,:,i2), Ar[i1,i2]', -ONE, ONE) # (Q[:,i1] -= Q[:,i2]*Ar[i1,i2]')
         withZ &&  mul!(view(Z,:,i1), view(Z,:,i2), view(Ar,i2,i1), -ONE, ONE) # (Z[:,i1] -= Z[:,i2]*Ar[i2,i1])
         iz2 = [ir[rE+1:rE+rA22];iz2]
         ir =  [i1; ir[rE+rA22+1:end]]
         iz1 = ir
      else
         # restore saved matrices
         Ar[ir,ir] = A1
         Er[ir,ir] = E1
         Br[ir,:] = B1
         Cr[:,ir] = C1
      end
      return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, withQ ? Q[:,[iz1;iz2]] : Q, withZ ? Z[:,[iz1;iz2]] : Z, nuc, nuo, rA22
   else
      return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, withQ ? Q[:,[iz1;iz2]] : Q, withZ ? Z[:,[iz1;iz2]] : Z, nuc, nuo, 0
   end
end
function projection_shifts(adj, A, E, V, W, p_old; num_desired = 6, implicitVtAV = true)
   ## function p = projection_shifts(A, E, V, W, num_desired, p_old) 
   #
   # Computes new shifts by implicitly or explicitly
   # projecting the E and A matrices to the span of V. Note that the
   # width of V must be a multiple of that of W, V is the newest part
   # of the ADI solution factor Z and the old shift
   # vector p_old passed in must have this multiple as its length.
   #
   # Whether or not the projection is computed implicitly from the
   # contents of V or by an explicit projection, is determined via
   # the implicitVtAV keyword argument. 
   #
   
   #
   # This function is based on the function mess_projection_shifts, which is
   # part of the M-M.E.S.S. project  (http://www.mpi-magdeburg.mpg.de/projects/mess).
   # Authors: Jens Saak, Martin Koehler, Peter Benner and others.
   #
   
   ## Check data
   # if not(isfield(opts, 'shifts')) || not(isstruct(opts.shifts))
   #     mess_warn(opts, 'control_data', ...
   #               ['shift parameter control structure missing.', ...
   #                'Switching to default num_desired = 25.']);
   #     opts.shifts.num_desired = 25;
   # else
   #     if not(isfield(opts.shifts, 'num_desired')) || ...
   #        not(isnumeric(opts.shifts.num_desired))
   
   #         mess_warn(opts, 'control_data', ...
   #                   ['Missing or Corrupted opts.shifts.num_desired field.', ...
   #                    'Switching to default: 25']);
   #         opts.shifts.num_desired = 25;
   #     end
   #     if not(isfield(opts.shifts, 'implicitVtAV')) || ...
   #             isempty(opts.shifts.implicitVtAV)
   #         opts.shifts.implicitVtAV = true;
   #     end
   # end
   haveUV = false
   haveE = (E !== I)
   
   
   L = length(p_old)
   cols_V = size(V, 2);
   cols_W = size(W, 2);
   
   if L > 0 && !iszero(p_old)  
       (cols_V / cols_W == L) || error("V and W have inconsistent no. of columns")
   end
   
   # Initialize data
   T1 = eltype(A) 
   if L > 0 && !iszero(p_old)
       T = zeros(T1,L, L)
       K = zeros(T1,1, L)
       D = zeros(T1,0,0)
       Ir = Matrix{T1}(I(cols_W))
       iC = findall(iszero,imag(p_old))
       iCh = iC[1:2:end]
       iR = findall(!iszero,imag(p_old))
       isubdiag = [iR; iCh]
       h = 1
   end
   
   # Process previous shifts
   if L > 0 && !iszero(p_old) && implicitVtAV
       while h <= L
           is = isubdiag[isubdiag .< h]
           K[1, h] = 1
           if isreal(p_old[h]) # real shift
               T[h, h] = p_old[h]
               if !isempty(is)
                   T[h, is] = 2 * p_old[h] * ones(1, length(is))
               end
               D = cat(D, sqrt(-2 * p_old[h]), dims=Val((1,2)))
               h = h + 1;
           else # complex conjugated pair of shifts
               rpc = real(p_old[h])
               ipc = imag(p_old[h])
               beta = rpc / ipc
               T[h:h + 1, h:h + 1] = [3*rpc -ipc
                                      ipc*(1 + 4 * beta^2) -rpc]
               if !isempty(is)
                   T[h:h +  1, is] = [4*rpc; 4*rpc*beta] * ones(1, length(is))
               end
               # D = blkdiag(D, ...
               #             2 * sqrt(-rpc) * [1, 0; beta, sqrt(1 + beta^2)]);
               D = cat(D, 2*sqrt(-rpc)*[1  0; beta sqrt(1 + beta^2)], dims=Val((1,2)))
               h = h + 2;
           end
       end
       S = kron(D \ (T * D), Ir)
       K = kron(K * D, Ir)
   else  # explicit AV (unless already computed in mess_para)
       S = 0;
       K = 1;
       if !iszero(p_old)
           W = A*V
           if haveUV
               if adj #eqn.type == 'T'
                   W = W + eqn.V * (eqn.U' * V);
               else
                   W = W + eqn.U * (eqn.V' * V);
               end
           end
       end
   end
   
   ## Compute projection matrices
   F = eigen(V'*V)
   s = F.values
   v = F.vectors
   r = (s .> eps() * s[end] * cols_V)
   st = v[:,r]*Diagonal(1 ./ s[r].^.5)
   U = V * st
   
   ## Project V and compute Ritz values
   if haveE
       E_V = E*V
       G = U' * E_V;
       H = U' * W * K * st + G * (S * st);
       G = G * st;
       p = eigvals(H, G);
   else
       H = U' * (W * K) * st + U' * (V * (S * st));
       p = eigvals(H);
   end
   
   ## Postprocess new shifts
   
   # remove infinite values
   p = p[isfinite.(p)]
   
   # remove zeros
   p = p[abs.(p) .> eps()]
   
   # make all shifts are stable
   p[real.(p) .> 0] = -p[real.(p) .> 0]
   
   if !isempty(p)
       # remove small imaginary perturbations
       small_imag = findall(abs.(imag.(p)) ./ abs.(p) .< 1e-12)
       p[small_imag] = real(p[small_imag])
   
       # sort (s.t. compl. pairs are together)
       sort!(p,by=real)
       length(p) > num_desired && (p = mess_mnmx(p, num_desired))
   end
   return p
end
function mess_mnmx(rw, num_desired)
   #
   #  Suboptimal solution of the ADI minimax problem. The delivered parameter
   #  set is closed under complex conjugation.
   #
   #  Calling sequence:
   #
   #    p = mess_mnmx(rw,num_desired)
   #
   #  Input:
   #
   #    rw            a vector containing numbers in the open left
   #                  half plane, which approximate the spectrum of
   #                  the corresponding matrix, e.g., a set of Ritz
   #                  values. The set must be closed w.r.t. complex
   #                  conjugation;
   #    num_desired   desired number of shift parameters
   #                  (length(rw) >= num_desired)
   #                  (The algorithm delivers either num_desired or
   #                  num_desired+1 parameters, to ensure closedness
   #                  under complex conjugation!).
   #
   #  Output:
   #
   #    p             a num_desired- or num_desired+1-vector of
   #                  suboptimal ADI parameters;
   #
   
   #
   # This file is part of the M-M.E.S.S. project
   # (http://www.mpi-magdeburg.mpg.de/projects/mess).
   # Copyright (c) 2009-2023 Jens Saak, Martin Koehler, Peter Benner and others.
   # All rights reserved.
   # License: BSD 2-Clause License (see COPYING)
   #
   
   #  Exact copy from
   #
   #  LYAPACK 1.0 (Thilo Penzl, October 1999)
   
   length(rw) < num_desired || throw(ArgumentError("length(rw) must be at least num_desired"))
    
   max_rr = +Inf                       # Choose initial parameter (pair)
   
   p0 = rw[1]
   for i = 1:length(rw)
       max_r = mess_s(rw[i], rw)[1]
       if max_r < max_rr
           p0 = rw[i]
           max_rr = max_r
       end
   end
   
   if !iszero(imag(p0))
       p = [p0; conj(p0)]
   else
       p = p0
   end
   
   i = mess_s(p, rw)[2]         # Choose further parameters.
   
   while size(p, 1) < num_desired   
       p0 = rw[i]
       if !iszero(imag(p0))
           p = [p; p0; conj(p0)]; ##ok<AGROW>
       else
           p = [p; p0]; ##ok<AGROW>
       end  
       i = mess_s(p, rw)[2]   
   end
   return p
end
function mess_s(p, set)
   #
   # Computation of the maximal magnitude of the rational ADI function over
   # a discrete subset of the left complex half plane.
   #
   #   Calling sequence:
   #
   #     [max_r,ind] = mess_s(p,set)
   #
   #   Input:
   #
   #     p        vector of ADI parameters;
   #     set      vector representing the discrete set.
   #
   #   Output:
   #
   #     max_r    maximal magnitude of the rational ADI function over set;
   #     ind      index - maximum is attained for set(ind).
   #
   
   #
   # This file is part of the M-M.E.S.S. project
   # (http://www.mpi-magdeburg.mpg.de/projects/mess).
   # Copyright (c) 2009-2023 Jens Saak, Martin Koehler, Peter Benner and others.
   # All rights reserved.
   # License: BSD 2-Clause License (see COPYING)
   #
   
   #   Exact copy from
   #
   #   LYAPACK 1.0 (Thilo Penzl, Jan 1999)
   #
   max_r = -1
   ind = 0
   
   for i = 1:length(set)  
       x = set[i]  
       rr = 1
       for j = 1:length(p)  
           rr = rr * abs(p[j] - x) / abs(p[j] + x)  
       end  
       if rr > max_r  
           max_r = rr
           ind = i   
       end   
   end
   return max_r, ind
end
   