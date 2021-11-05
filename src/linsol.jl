"""
    glsol(sysg, sysf; poles = missing, sdeg = missing, mindeg = false, solgen = false, minreal = true, fast = true, 
          atol = 0, atol1 = atol, atol2 = atol, rtol, ) -> (sysx, info, sysgen)

Determine for the descriptor systems `sysg = (Ag-λEg,Bg,Cg,Dg)` and 
`sysf = (Af-λEf,Bf,Cf,Df)` with the transfer function matrices `G(λ)` and `F(λ)`, 
respectively, the descriptor system `sysx` with the transfer function matrix `X(λ)` 
such that `X(λ)` is the solution of the linear rational equation

    X(λ)G(λ) = F(λ) .      (1)

If `solgen = true`, the descriptor system `sysgen` is determined representing a generator of 
all solutions of (1). Its transfer function matrix has the form `GEN(λ) = [ X0(λ); XN(λ) ]`, 
such that any `X(λ)` can be generated as

    X(λ) = X0(λ) + Z(λ)*XN(λ) ,

where `X0(λ)` is a particular solution satisfying `X0(λ)G(λ) = F(λ)`, 
`XN(λ)` is a proper rational left nullspace basis of `G(λ)` satisfying `XN(λ)G(λ) = 0`, and 
`Z(λ)` is an arbitrary rational matrix with suitable dimensions. If `solgen = false`, `sysgen` is
set to `nothing`. 

The call with

    glsol(sysgf, pf; poles = missing, sdeg = missing, mindeg = false, solgen = false, minreal = true, fast = true, 
          atol = 0, atol1 = atol, atol2 = atol, rtol, ) -> (sysx, info, sysgen)

uses the compound descriptor system `sysgf = (A-λE,B,[Cg; Cf],[Dg; Df])`, 
where `Cf` has `pf` rows, to define  
the descriptor systems `sysg = (A-λE,B,Cg,Dg)` and `sysf = (A-λE,B,Cf,Df)`
(i.e., `Ag-λEg = Af-λEf = A-λE` and `Bg = Bf = B`). 

The generator `sysgen` has a descriptor system realization `sysgen = (A0-λE0,B0, [C0; CN],[D0; DN])`,
which is usually not minimal  (unobservable and/or non-dynamic modes present), with  

                 ( Ai-λEi    *       *    )  
       A0-λE0  = (   0     Af-λEf    *    ) , 
                 (   0       0     Al-λEl ) 

                 ( *  )
           B0  = ( *  ),   ( C0 )  = ( C1 C2 C3 ) 
                 ( Bl )    ( CN )    ( 0  0  Cl )
  
   
with `El`, `Ef` and `Ai` invertible and upper triangular, `Ei` nillpotent
and upper triangular, and `DN` full column rank. The dimensions of the diagonal blocks of `A0-λE0` are 
returned in the named tuple `info` as the components `info.nf`, `info.ninf` and `info.nl`, respectively. 

A minimal order descriptor system realization of the proper basis `XN(λ)` is `(Al-λEl,Bl,Cl,DN)`, 
where `Cl` and `DN` have `pr` columns (returned in `info.pr`), representing the dimension of the 
left nullspace basis. The normal rank `nrank` of  `G(λ)` is returned in `info.nrank`. 

If `mindeg = false`, the solution `sysx` is determined in the form `sysx = (A0+F*CN-λE0,B0+F*DN,C0,D0)`,
where the matrix `F = 0`, unless a nonzero stabilizing gain is used such that `Al+F*Bl-λEl` has stable eigenvalues. 
The vector `poles` specified as a keyword argument, can be used to specify the desired eigenvalues
alternatively to or jointly with enforcing a desired stability degree `sdeg` of eigenvalues. 
The dimension `nl` of `Al` is the number of freely assignable poles of the solution `X(λ)` and is returned in `info.nl`. 
The eigenvalues of `Af-λEf` contain the finite zeros of `G(λ)`, while
the zeros  of `Ai-λEi` contain the infinite zeros of `G(λ)`.   
The norm of the employed gain `F` is returned in `info.fnorm`. If `G(λ)` has infinite zeros, then the solution
`X(λ)` may have infinite poles. The integer vector `info.rdeg` contains the relative row degrees of 
`X(λ)` (i.e., the numbers of integrators/delays needed to make each row of `X(λ)` proper).  

If `mindeg = true`, a minimum degree solution is determined as `X(λ) = X0(λ) + Z(λ)XN(λ)`, where `Z(λ)` is
determined using order reduction based on a Type 2 minimum dynamic cover. This computation involves using
non-orthogonal transformations whose worst condition number is returned in `info.tcond`, in conjunction with 
using feedback and feedforward gains, whose norms are returned in `info.fnorm`. High values of these quantities
indicate a potential loss of numerical stability of computations. 

If `minreal = true`, the computed realization `sysx` is minimal.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `Ag`, `Bg`, `Cg`, `Dg`, `Af`, `Bf`, `Cf`, `Df`, 
the absolute tolerance for the nonzero elements of `Eg` and `Ef`,  
and the relative tolerance for the nonzero elements of 
   `Ag`, `Bg`, `Cg`, `Dg`, `Af`, `Bf`, `Cf`, `Df`, `Eg` and `Ef`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the maximal order of the systems `sysg` and `sysf`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`. 

The rank determinations in the performed reductions
 are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_  The dual of method of [1] to solve rational systems is used.

_References:_

[1] A. Varga, "Computation of least order solutions of linear rational equations", 
Proc. MTNS'04, Leuven, Belgium, 2004.
"""
function glsol(sysg::DescriptorStateSpace, sysf::DescriptorStateSpace; kwargs...)  
   size(sysg,2) == size(sysf,2) || throw(DimensionMismatch("sysg and sysf must have the same number of inputs"))
   sysx, info1, sysgen = grsol(gdual(sysg), gdual(sysf); kwargs...)
   info = (nrank = info1.nrank, rdeg = info1.rdeg, tcond = info1.tcond, fnorm = info1.fnorm, pl = info1.mr, nl = info1.nr, nf = info1.nf, ninf = info1.ninf)
   return gdual(sysx,rev = true), info, sysgen === nothing ? sysgen : gdual(sysgen,rev = true)
end
function glsol(sysgf::DescriptorStateSpace, pf::Int; kwargs...)  
   p = size(sysgf,1);
   (pf <= p && pf >= 0) || throw(DimensionMismatch("pf must be at most $p, got $pf"))
   sysx, info1, sysgen = grsol(gdual(sysgf,rev = true), pf; kwargs...)
   info = (nrank = info1.nrank, rdeg = info1.rdeg, tcond = info1.tcond, fnorm = info1.fnorm, pl = info1.mr, nl = info1.nr, nf = info1.nf, ninf = info1.ninf)
   return gdual(sysx,rev = true), info, sysgen === nothing ? sysgen : gdual(sysgen,rev = true)
end
"""
    grsol(sysg, sysf; poles = missing, sdeg = missing, mindeg = false, solgen = false, minreal = true, fast = true, 
          atol = 0, atol1 = atol, atol2 = atol, rtol, ) -> (sysx, info, sysgen)

Determine for the descriptor systems `sysg = (Ag-λEg,Bg,Cg,Dg)` and 
`sysf = (Af-λEf,Bf,Cf,Df)` with the transfer function matrices `G(λ)` and `F(λ)`, 
respectively, the descriptor system `sysx` with the transfer function matrix `X(λ)` 
such that `X(λ)` is the solution of the linear rational equation

    G(λ)X(λ) = F(λ) .      (1)

If `solgen = true`, the descriptor system `sysgen` is determined representing a generator of 
all solutions of (1). Its transfer function matrix has the form `GEN(λ) = [ X0(λ) XN(λ) ]`, 
such that any `X(λ)` can be generated as

    X(λ) = X0(λ) + XN(λ)*Z(λ) ,

where `X0(λ)` is a particular solution satisfying `G(λ)X0(λ) = F(λ)`, 
`XN(λ)` is a proper rational right nullspace basis of `G(λ)` satisfying `G(λ)XN(λ) = 0`, and 
`Z(λ)` is an arbitrary rational matrix with suitable dimensions. If `solgen = false`, `sysgen` is
set to `nothing`. 

The call with

    grsol(sysgf, mf; poles = missing, sdeg = missing, mindeg = false, solgen = false, minreal = true, fast = true, 
          atol = 0, atol1 = atol, atol2 = atol, rtol, ) -> (sysx, info, sysgen)

uses the compound descriptor system `sysgf = (A-λE,[Bg Bf],C,[Dg Df])`, 
where `Bf` has `mf` columns, to define  
the descriptor systems `sysg = (A-λE,Bg,C,Dg)` and `sysf = (A-λE,Bf,C,Df)`
(i.e., `Ag-λEg = Af-λEf = A-λE` and `Cg = Cf = C`). 

The generator `sysgen` has a descriptor system realization `sysgen = (A0-λE0,[B0 BN],C0,[D0 DN])`,
which is usually not minimal  (uncontrollable and/or non-dynamic modes present), with  

                   ( Ar-λEr    *       *    )  
       A0-λE0    = (   0     Af-λEf    *    ) , 
                   (   0       0     Ai-λEi ) 

                   ( B1 | Br )
       [B0 | BN] = ( B2 | 0  ),  Cg  =   ( Cr   *    *  ) ,
                   ( B3 | 0  )
   
with `Er`, `Ef` and `Ai` invertible and upper triangular, `Ei` nillpotent
and upper triangular, and `DN` full row rank. The dimensions of the diagonal blocks of `A0-λE0` are 
returned in the named tuple `info` as the components `info.nr`, `info.nf`, and `info.ninf`, respectively. 

A minimal order descriptor system realization of the proper basis `XN(λ)` is `(Ar-λEr,Br,Cr,DN)`, 
where `Br` and `DN` have `mr` columns (returned in `info.mr`), representing the dimension of the 
right nullspace basis. The normal rank `nrank` of  `G(λ)` is returned in `info.nrank`. 

If `mindeg = false`, the solution `sysx` is determined in the form `sysx = (A0+BN*F-λE0,B0,C0+DN*F,D0)`,
where the matrix `F = 0`, unless a nonzero stabilizing gain is used such that `Ar+Br*F-λEr` has stable eigenvalues. 
The vector `poles` specified as a keyword argument, can be used to specify the desired eigenvalues
alternatively to or jointly with enforcing a desired stability degree `sdeg` of eigenvalues. 
The dimension `nr` of `Ar` is the number of freely assignable poles of the solution `X(λ)` and is returned in `info.nr`. 
The eigenvalues of `Af-λEf` contain the finite zeros of `G(λ)`, while
the zeros  of `Ai-λEi` contain the infinite zeros of `G(λ)`.   
The norm of the employed gain `F` is returned in `info.fnorm`. If `G(λ)` has infinite zeros, then the solution
`X(λ)` may have infinite poles. The integer vector `info.rdeg` contains the relative column degrees of 
`X(λ)` (i.e., the numbers of integrators/delays needed to make each column of `X(λ)` proper).  

If `mindeg = true`, a minimum degree solution is determined as `X(λ) = X0(λ) + XN(λ)*Z(λ)`, where `Z(λ)` is
determined using order reduction based on a Type 2 minimum dynamic cover. This computation involves using
non-orthogonal transformations whose worst condition number is returned in `info.tcond`, in conjunction with 
using feedback and feedforward gains, whose norms are returned in `info.fnorm`. High values of these quantities
indicate a potential loss of numerical stability of computations. 

If `minreal = true`, the computed realization `sysx` is minimal.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `Ag`, `Bg`, `Cg`, `Dg`, `Af`, `Bf`, `Cf`, `Df`, 
the absolute tolerance for the nonzero elements of `Eg` and `Ef`,  
and the relative tolerance for the nonzero elements of 
   `Ag`, `Bg`, `Cg`, `Dg`, `Af`, `Bf`, `Cf`, `Df`, `Eg` and `Ef`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the maximal order of the systems `sysg` and `sysf`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`. 

The rank determinations in the performed reductions
 are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_  The method of [1] to solve rational systems is used.

_References:_

[1] A. Varga, "Computation of least order solutions of linear rational equations", 
Proc. MTNS'04, Leuven, Belgium, 2004.
"""
function grsol(sysg::DescriptorStateSpace{T1}, sysf::DescriptorStateSpace{T2}; 
               poles::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, 
               atol::Real = zero(real(T1)), atol1::Real = atol, atol2::Real = atol, 
               rtol::Real = ((size(sysg.A,1)+1)*eps(real(float(one(T1)))))*iszero(max(atol1,atol2)), 
               fast::Bool = true, mindeg::Bool = false, solgen::Bool = false, minreal::Bool = true) where {T1,T2}
   size(sysg,1) == size(sysf,1) || throw(DimensionMismatch("sysg and sysf must have the same number of outputs"))
   return grsol(gir([sysg sysf]; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol), size(sysf,2); 
   poles = poles, sdeg = sdeg, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, mindeg = mindeg, solgen = solgen, minreal = minreal)
end
function grsol(sysgf::DescriptorStateSpace{T}, mf::Int; 
               poles::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, 
               atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
               rtol::Real = ((size(sysgf.A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2)), 
               fast::Bool = true, mindeg::Bool = false, solgen::Bool = false, minreal::Bool = true) where T 

   T1 = T <: BlasFloat ? T : promote_type(Float64,T) 

   p, mgf = size(sysgf);
   (mf <= mgf && mf >= 0) || throw(DimensionMismatch("mf must be at most $mgf, got $mf"))
   m = mgf-mf
   Ts = sysgf.Ts;   

   ONE = one(T1)

   disc = !iszero(Ts) 
   sdeg_nomissing = !ismissing(sdeg)
   poles_nomissing = !ismissing(poles)


   stabilize = sdeg_nomissing || poles_nomissing

   # set default values of sdeg if poles = missing
   # sdeg_nomissing || (sdeg = disc ? real(T1)(0.95) : real(T1)(-0.05))

   # sort desired eigenvalues
   if poles_nomissing 
      tempc = poles[imag.(poles) .> 0]
      if !isempty(tempc)
         tempc1 = conj(poles[imag.(poles) .< 0])
         isequal(tempc[sortperm(real(tempc))],tempc1[sortperm(real(tempc1))]) ||
                 error("poles must be a self-conjugated complex vector")
      end
      # check that all eigenvalues are inside of the stability region
      sdeg_nomissing && ( ((disc && any(abs.(poles) .> sdeg) )  || (!disc && any(real.(poles) .> sdeg)))  &&
            error("The elements of poles must lie in the stability region of interest") )
   end     

   At, Et, Bt, Ct, Dt = dssdata(T1,sysgf);
    
   n = size(At,1)

   Bf = view(Bt,:,m+1:mgf)
   Df = view(Dt,:,m+1:mgf)
   Bg = view(Bt,:,1:m)
   Dg = view(Dt,:,1:m)  

   # compute the Kronecker-like form of the system matrix of G to obtain
   #                 ( Br  Ar - s Er      *             *           *     )
   #   Q'*(A-sE)*Z = (  0     0      Afin - s Efin      *           *     )
   #                 (  0     0           0        Ainf - s Einf    *     )
   #                 (  0     0           0            0         Al - s El)
 
   M, N, Q, Z, νr, μr, νi, nf, νl, μl = sklf(At, Et, Bg, Ct, Dg, 
                                             finite_infinite = true, fast = fast, ut = true,
                                             atol1 = atol1, atol2 = atol2, rtol = rtol) 

   #                        ( B1 ) 
   # compute Q'*[-Bf;-Df] = ( B2 )
   #                        ( B3 ) 
   #                        ( B4 )
   f = Q'*[-Bf;-Df]; 

   # determine the orders of Ar, Ai and Al
   nr = sum(νr); ninf = sum(νi); nl = sum(μl);
   # determine the orders of regular part and of the maximal invertible part
   nreg = nf+ninf; ninv = nr+nreg+nl; 
   mr = n+m-ninv  # number of columns of Br 

   # check compatibility condition, i.e., B4 = 0.
   tola = max(atol1,atol2)
   iszero(tola) && 
      (tola = n*eps(real(T1))*max(opnorm(Et,1), opnorm(At,1), opnorm(Bt,1), opnorm(Ct,Inf))) 
   sum(νl) > 0 && mf > 0 && maximum(abs.(f[nr+nreg+1:end,:])) >= tola &&
         error("System not compatible")

   # form X0 = (A0-sE0,B0,C0,D0) and XN = (Ar-sEr,Br,Cr,DN), where
   #
   #                ( Ar - s*Er     *           *        )        ( B1 )        ( Br )
   #       A0-sE0 = (    0       Af - s*Ef      *        ) , B0 = ( B2 ) , BN = ( 0  )
   #                (    0          0        Ai - s*Einf )        ( B3 )        ( 0  )
   #      
   #           C0 = ( Cr  Cf Ci );  D0 = 0, with
   #
   #             [ 0  Im ]*Z := [ DN Cr  Cf Ci Cl ];
   n0 = nr+nreg;
   i1 = 1:nr; 
   A0 = M[1:n0,mr+1:mr+n0]; E0 = N[1:n0,mr+1:mr+n0]; 
   B0 = f[1:n0,:];    C0 = Z[n+1:n+m,mr+1:mr+n0]; D0 = zeros(T1,m,mf) 
   Br = view(M,i1,1:mr); BN = [Br; zeros(T1,nreg,mr)]; DN = view(Z,n+1:n+m,1:mr); # CN = C0
   Ar = view(A0,i1,i1)
   Er = view(E0,i1,i1)
   Cr = view(C0,:,i1)

   rdeg = zeros(Int,mf)
   if length(νi) > 1
      i2 = nr+nf+1:n0
      # compute relative column degrees as the number of infinite poles 
      # (in excess with 1 with respect to the controllable infinite eigenvalues)
      for i = 1:mf
          _, _, _, ni, = sklf_right!(copy(E0[i2,i2]), copy(A0[i2,i2]), copy(B0[i2,i:i]), copy(C0[:,i2]); 
                                    fast = fast, atol1 = atol1, atol2 = atol2, atol3 = atol1, rtol = rtol, 
                                    withQ = false, withZ = false) 
          ni > 1  && (rdeg[i] = ni-1)
      end
   end  
  
   if mindeg 
      # form the pair of generators if requested
      # copy A0, E0 and C0, to ensure are not later modified (due to bindings)
      solgen ? sysgen = dss(copy(A0),E0,[B0 BN],copy(C0),[D0 DN],Ts = Ts) : sysgen = nothing

      # block-diagonalize E0 to allow working on the proper part
      # exploit Er is upper triangular
      i2 = nr+1:n0;
      Y = -E0[i1,i2]
      ldiv!(UpperTriangular(Er),Y)
      mul!(view(A0,i1,i2), Ar, Y, ONE, ONE)
      mul!(view(C0,:,i2), Cr, Y, ONE, ONE)
      # Y = -E0[i1,i1]\E0[i1,i2];
      # A0[i1,i2] = A0[i1,i2]+A0[i1,i1]*Y;
      # C0[:,i2] = C0[:,i2]+C0[:,i1]*Y;
   
      # form a pair of generators for the proper part 
      sysr = dss(Ar, Er, [B0[i1,:] A0[i1,i2] Br], Cr, [D0 C0[:,i2] DN], Ts = Ts);
   
      # compute minimum order for the proper part
      sysr, _, info2 = grmcover2(sysr,mf+nreg, fast = fast, atol1 = atol1, atol2 = atol2) 
      aa, ee, bb, cc, dd = dssdata(sysr); na = size(aa,1); ee == I && (ee = Matrix{T1}(I,na,na))
      i3 = mf+1:mf+nreg;
      A0 = [aa bb[:,i3]; zeros(T1,nreg,na) A0[i2,i2]]; E0 = blockdiag(ee,E0[i2,i2])
      B0 = [ bb[:,1:mf]; B0[i2,:]]; C0 = [cc dd[:,i3]]; D0 = dd[:,1:mf]; 
      tcond = info2.tcond;
      fnorm = max(info2.fnorm,info2.gnorm);
   else
      fnorm = 0
      tcond = 1
      if stabilize  
         # make spurious poles stable
         if nr > 0
            i1 = 1:nr; 
            F = saloc(Ar, Er, Br; evals = poles, sdeg = sdeg, disc = disc, fast = fast, atol1 = atol1, atol2 = atol2, atol3 = atol1)[1]
            #F = saloc(A0[i1,i1], E0[i1,i1], Br; evals = poles, sdeg = sdeg, disc = disc, fast = fast, atol1 = atol1, atol2 = atol2, atol3 = atol1)[1]
            #A0[i1,i1] = A0[i1,i1]+Br*F; C0[:,i1] = C0[:,i1]+DN*F
            mul!(Ar, Br, F, ONE, ONE)
            mul!(Cr, DN, F, ONE, ONE)
         end
         fnorm = norm(F)
      end
      # form a pair of generators 
      solgen ? sysgen = dss(A0, E0, [B0 BN], C0, [D0 DN], Ts = Ts) : sysgen = nothing
   end

   # eliminate possible uncontrollable eigenvalues and non-dynamic modes
   minreal ? sysx = gminreal(dss(A0, E0, B0, C0, D0, Ts = Ts), obs = false, fast = fast, atol1 = atol1, atol2 = atol2) :
             sysx = dss(A0, E0, B0, C0, D0, Ts = Ts)
   # sysx = gss2ss(sysx,tol,'triu');
   info = (nrank = ninv-n, rdeg = rdeg, tcond = tcond, fnorm = fnorm, mr = mr, nr = nr, nf = nf, ninf = ninf)
   return sysx, info, sysgen
# end GRSOL
end