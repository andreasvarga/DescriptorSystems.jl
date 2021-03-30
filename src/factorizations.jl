"""
     sysf = glsfg(sys, γ; fast = true, stabilize = true, offset = β, 
                  atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) 

Compute for the descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrices `G(λ)`
and ``{\\small γ > \\|G(λ)\\|_∞}``, the minimum-phase right spectral factor `sysf = (Af-λEf,Bf,Cf,Df)`
with the transfer-function matrix `F(λ)`, such that `F(λ)*F(λ)' = γ^2*I-G(λ)*G(λ)'`.
If `stabilize = true` (the default), a preliminary stabilization of `sys` is performed. 
In this case, `sys` must not have poles on the imaginary-axis in the continuous-time case or 
on the unit circle in the discrete-time case.
If `stabilize = false`, then no preliminary stabilization is performed. In this case, `sys` must be stable.

To assess the presence of poles on the boundary of the stability domain `Cs`, a boundary offset  `β` 
can be specified via the keyword parameter `offset = β`. 
Accordingly, for a continuous-time system, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time system, then the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

If `stabilize = true`, a preliminary separation of finite and infinite eigenvalues of `A-λE`is performed 
using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, 
the absolute tolerance for the nonzero elements of `E`, 
the absolute tolerance for the nonzero elements of `C`,  
and the relative tolerance for the nonzero elements of `A`, `E` and `C`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`, `atol3 = atol`.

_Method:_ Extensions of the factorization approaches of [1] are used.

_References:_

[1] K. Zhou, J. C. Doyle, and K. Glover. Robust and Optimal Control. Prentice Hall, 1996.
"""
function glsfg(sys::DescriptorStateSpace{T},γ::Real; stabilize::Bool = true, 
               offset::Real = sqrt(eps(float(real(T)))), atol::Real = zero(real(T)), 
               atol1::Real = atol, atol2::Real = atol, atol3::Real = atol, 
               rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(max(atol1,atol2,atol3)), 
               fast::Bool = true) where T 

   # Compute the RCF with inner denominator if stabilize = true
   stabilize && (sys = grcfid(sys; mininf = true, fast = fast, offset = offset, 
                          atol1 = atol1, atol2 = atol2, atol3 = atol3, rtol = rtol)[1])

   epsm2 = sqrt(eps(float(real(T))))

   if sys.E != I && (norm(sys.E,Inf) < atol2 || rcond(sys.E) < epsm2 )
      sys = gss2ss(sys; atol1 = atol1, atol2 = atol2, rtol = rtol)
      sys.E != I && rcond(sys.E) < epsm2 && error("Improper input system sys")
   end
   a, e, b, c, d = dssdata(sys)
   discr = (sys.Ts != 0)
   
   # Compute the stabilizing solution of the corresponding Riccati equation
   p, n = size(c)
   r = γ*γ*I-d*d' 
   
   if n > 0
      if discr
         xric, _, kric, = gared(a', e', c', -r, b*b', b*d'; rtol = rtol); #fric = -fric; 
         SF = schur(xric)
         rdiag = Diagonal(sqrt.(max.(real.(SF.values),0)))
         SV = svd([d c*SF.Z*rdiag],full=true)
      else
         _, _, kric, = garec(a', e', c', -r, b*b', b*d'; rtol = rtol); #fric = -fric; 
         SV = svd(d,full=true)
      end
   else
      SV = svd(d,full=true)
      if !isempty(SV.S) && SV.S[1] > abs(γ)
         error("The condition γ > ||sys||∞ is not fulfilled")
      end
      kric = zeros(T,p,n)
   end
   
   # compute square-root factor
   s = SV.S
   rsq = [s;zeros(eltype(s),p-length(s))]
   rsqrt = SV.U*Diagonal(sqrt.(max.(γ*γ .- rsq.^2,0)))
   
   # assemble the spectral factor
   sysf = dss(a, e, kric'*rsqrt, c, rsqrt, Ts = sys.Ts)

   # end GLSFG
end
"""
     sysf = grsfg(sys, γ; fast = true, stabilize = true, offset = β, 
                  atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) 

Compute for the descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrices `G(λ)`
and ``{\\small γ > \\|G(λ)\\|_∞}``, the minimum-phase right spectral factor `sysf = (Af-λEf,Bf,Cf,Df)`
with the transfer-function matrix `F(λ)`, such that `F(λ)'*F(λ) = γ^2*I-G(λ)'*G(λ)`.
If `stabilize = true` (the default), a preliminary stabilization of `sys` is performed. 
In this case, `sys` must not have poles on the imaginary-axis in the continuous-time case or 
on the unit circle in the discrete-time case.
If `stabilize = false`, then no preliminary stabilization is performed. In this case, `sys` must be stable.

To assess the presence of poles on the boundary of the stability domain `Cs`, a boundary offset  `β` 
can be specified via the keyword parameter `offset = β`. 
Accordingly, for a continuous-time system, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time system, then the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

If `stabilize = true`, a preliminary separation of finite and infinite eigenvalues of `A-λE`is performed 
using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, 
the absolute tolerance for the nonzero elements of `E`, 
the absolute tolerance for the nonzero elements of `C`,  
and the relative tolerance for the nonzero elements of `A`, `E` and `C`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`, `atol3 = atol`.

_Method:_ Extensions of the factorization approaches of [1] are used.

_References:_

[1] K. Zhou, J. C. Doyle, and K. Glover. Robust and Optimal Control. Prentice Hall, 1996.
"""
function grsfg(sys::DescriptorStateSpace{T},γ::Real; stabilize::Bool = true, 
               offset::Real = sqrt(eps(float(real(T)))), atol::Real = zero(real(T)), 
               atol1::Real = atol, atol2::Real = atol, atol3::Real = atol, 
               rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(max(atol1,atol2,atol3)), 
               fast::Bool = true) where T 

   # Compute the LCF with inner denominator if stabilize = true
   stabilize && (sys = glcfid(sys; mininf = true, fast = fast, offset = offset, 
                          atol1 = atol1, atol2 = atol2, atol3 = atol3, rtol = rtol)[1])

   epsm2 = sqrt(eps(float(real(T))))

   if sys.E != I && (norm(sys.E,Inf) < atol2 || rcond(sys.E) < epsm2 )
      sys = gss2ss(sys; atol1 = atol1, atol2 = atol2, rtol = rtol)
      sys.E != I && rcond(sys.E) < epsm2 && error("Improper input system sys")
   end
   a, e, b, c, d = dssdata(sys)
   discr = (sys.Ts != 0)
   
   # Compute the stabilizing solution of the corresponding Riccati equation
   n, m = size(b)
   r = γ*γ*I-d'*d 
   
   if n > 0
      if discr
         xric, _, fric, = gared(a, e, b, -r, c'*c, c'*d; rtol = rtol); #fric = -fric; 
         SF = schur(xric)
         rdiag = Diagonal(sqrt.(max.(real.(SF.values),0)))
         SV = svd([d;rdiag*SF.Z'*b],full=true)
      else
         _, _, fric, = garec(a, e, b, -r, c'*c, c'*d; rtol = rtol); #fric = -fric; 
         SV = svd(d,full=true)
      end
   else
      SV = svd(d,full=true)
      if !isempty(SV.S) && SV.S[1] > abs(γ)
         error("The condition γ > ||sys||∞ is not fulfilled")
      end
      fric = zeros(T,m,n);
   end
   
   # compute square-root factor
   s = SV.S
   rsq = [s;zeros(eltype(s),m-length(s))]
   rsqrt = Diagonal(sqrt.(max.(γ*γ .- rsq.^2,0)))*SV.Vt
   
   # assemble the spectral factor
   sysf = dss(a, e, b, rsqrt*fric, rsqrt, Ts = sys.Ts)

   # end GRSFG
end
"""
    gnlcf(sys; fast = true, ss = false, 
         atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (sysn, sysm)

Compute for the descriptor system `sys = (A-λE,B,C,D)`, the factors 
`sysn = (An-λEn,Bn,Cn,Dn)` and `sysm = (An-λEn,Bm,Cn,Dm)` of its normalized
right coprime factorization. If `sys`, `sysn` and `sysm`  
have the transfer function matrices `G(λ)`, `N(λ)` and `M(λ)`, respectively, then
`G(λ) = inv(M(λ))*N(λ)`, with `N(λ)` and `M(λ)` proper and stable transfer 
function matrices and `[N(λ) M(λ)]` coinner. The resulting `En = I` if `ss = true`. 

Pencil reduction algorithms are employed which perform rank decisions based on rank 
revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The keyword arguments `atol1`, `atol2` and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C` and `D`,
the absolute tolerance for the nonzero elements of `E`,   
and the relative tolerance for the nonzero elements of `A`, `E`, `B`, `C` and `D`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the machine epsilon of the element type of `A` 
and `n` is the order of the system `sys`.
The keyword argument `atol` can be used to simultaneously set `atol1 = atol`, `atol2 = atol`.

_Method:_ Pencil reduction algorithms are employed to determine the coinner coimage space `R(λ)` of the 
transfer function matrix `[G(λ) I]` using the dual of method described in [1], which is based on 
the reduction algorithm of [2]. Then the factors `N(λ)` and `M(λ)` result from the partitioning of
`R(λ)` as `R(λ) = [N(λ) M(λ)]`. 

_References:_

[1] Varga, A.
    A note on computing the range of rational matrices. 
    arXiv:1707.0048, [https://arxiv.org/abs/1707.0048](https://arxiv.org/abs/1707.004), 2017.

[2] C. Oara.
    Constructive solutions to spectral and inner–outer factorizations 
    respect to the disk. Automatica,  41, pp. 1855–1866, 2005. 
"""
function gnlcf(sys::DescriptorStateSpace{T}; fast::Bool = true, ss::Bool = false,
              atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol,  
              rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(max(atol1,atol2))) where T 

   # compute the coinner coimage of [G(λ) I] 
   p, m = size(sys)
   sysr, = gcrange([sys I], zeros = "none", coinner = true, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)
   
   # compute a standard state-space realization if requested
   ss && (sysr = gss2ss(sysr, Eshape = "ident", atol1 = atol1, atol2 = atol2, rtol = rtol)[1])
   
   # extract the factors N(λ) and M(λ)
   return sysr[1:p,1:m], sysr[1:p,m+1:end]
   
   # end GNRCF
end
"""
    gnrcf(sys; fast = true, ss = false, 
         atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (sysn, sysm)

Compute for the descriptor system `sys = (A-λE,B,C,D)`, the factors 
`sysn = (An-λEn,Bn,Cn,Dn)` and `sysm = (An-λEn,Bn,Cm,Dm)` of its normalized
right coprime factorization. If `sys`, `sysn` and `sysm`  
have the transfer function matrices `G(λ)`, `N(λ)` and `M(λ)`, respectively, then
`G(λ) = N(λ)*inv(M(λ))`, with `N(λ)` and `M(λ)` proper and stable transfer 
function matrices and `[N(λ);M(λ)]` inner. The resulting `En = I` if `ss = true`. 

Pencil reduction algorithms are employed which perform rank decisions based on rank 
revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The keyword arguments `atol1`, `atol2` and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C` and `D`,
the absolute tolerance for the nonzero elements of `E`,   
and the relative tolerance for the nonzero elements of `A`, `E`, `B`, `C` and `D`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the machine epsilon of the element type of `A` 
and `n` is the order of the system `sys`.
The keyword argument `atol` can be used to simultaneously set `atol1 = atol`, `atol2 = atol`.

_Method:_ Pencil reduction algorithms are employed to determine the inner range space `R(λ)` of the 
transfer function matrix `[G(λ); I]` using the method described in [1], which is based on 
the reduction algorithm of [2]. Then the factors `N(λ)` and `M(λ)` result from the partitioning of
`R(λ)` as `R(λ) = [N(λ);M(λ)]`. 

_References:_

[1] Varga, A.
    A note on computing the range of rational matrices. 
    arXiv:1707.0048, [https://arxiv.org/abs/1707.0048](https://arxiv.org/abs/1707.004), 2017.

[2] C. Oara.
    Constructive solutions to spectral and inner–outer factorizations 
    respect to the disk. Automatica,  41, pp. 1855–1866, 2005. 
"""
function gnrcf(sys::DescriptorStateSpace{T}; fast::Bool = true, ss::Bool = false,
              atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol,  
              rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(max(atol1,atol2))) where T 

   # compute the inner range of [G(λ);I] 
   p, m = size(sys)
   sysr, = grange([sys;I], zeros = "none", inner = true, fast = fast, atol1 = atol1,atol2 = atol2, rtol = rtol)
   
   # compute a standard state-space realization if requested
   ss && (sysr = gss2ss(sysr, Eshape = "ident", atol1 = atol1, atol2 = atol2, rtol = rtol)[1])
   
   # extract the factors N(λ) and M(λ)
   return sysr[1:p,1:m], sysr[p+1:end,1:m]
   
   # end GNRCF
end
"""
    goifac(sys; atol = 0, atol1 = atol, atol2 = atol, atol3 = atol, rtol, 
           fast = true, minphase = true, offset = sqrt(ϵ)) -> (sysi, syso, info)

Compute for the descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`, 
the square inner factor `sysi = (Ai-λEi,Bi,Ci,Di)` with the transfer function matrix `Gi(λ)` 
and the minimum-phase quasi-outer factor or the full column rank factor `syso = (Ao-λEo,Bo,Co,Do)` 
with the transfer function matrix `Go(λ)` such that
   
     G(λ) = Go(λ)*Gi[1:r,:](λ)    (*),

where `r` is the normal rank of `G(λ)`. The resulting proper and stable inner factor satisfies 
`Gi'(λ)*Gi(λ) = I`. If `sys` is stable (proper), then the resulting `syso` is stable (proper). 
The resulting factor `Go(λ)` has full column rank `r`. Depending on the selected factorization option,
if `minphase = true`, then `Go(λ)` is minimum phase,  excepting possibly zeros on the 
boundary of the appropriate stability domain `Cs`, or if `minphase = false`, then `Go(λ)` 
contains all zeros of `G(λ)`, in which case (*) is the extended RQ-like factorization of `G(λ)`.
For a continuous-time system `sys`, the stability domain `Cs` is defined as the set of 
complex numbers with real parts at most `-β`, 
while for a discrete-time system `sys`, `Cs` is the set of complex numbers with 
moduli at most `1-β` (i.e., the interior of a disc of radius `1-β` centered in the origin). 
The boundary offset  `β` to be used to assess the stability of zeros and their number 
on the boundary of `Cs` can be specified via the keyword parameter `offset = β`.
Accordingly, for a continuous-time system, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time system, the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The resulting named triple `ìnfo` contains `(nrank, nfuz, niuz) `, where `ìnfo.nrank = r`, 
the normal rank of `G(λ)`, `ìnfo.nfuz` is the number of finite zeros of `syso` on 
the boundary of `Cs`, and `ìnfo.niuz` is the number of infinite zeros of `syso`. 
`ìnfo.nfuz` is set to `missing` if `minphase = false`. 

_Note:_ `syso` may generally contain a _free inner factor_, which can be eliminated by 
removing the finite unobservable eigenvalues. 

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A` and `C`, 
the absolute tolerance for the nonzero elements of `E`, 
the absolute tolerance for the nonzero elements of `B` and `D`,  
and the relative tolerance for the nonzero elements of `A`, `E`, `B`, `C` and `D`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`, `atol3 = atol`. 

For the assessment of zeros, the dual system pencil `transpose([A-λE B; C D])` is reduced to a 
special Kronecker-like form (see [2]). In this reduction, the 
performed rank decisions are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_  For a continuous-time system, the dual system is formed and the factorization algorithm 
of [1] is used, while for a discrete-time system, the factorization algorithm of [1] is used.

_References:_

[1] C. Oara and A. Varga.
    Computation of the general inner-outer and spectral factorizations.
    IEEE Trans. Autom. Control, vol. 45, pp. 2307--2325, 2000.

[2] C. Oara.
    Constructive solutions to spectral and inner–outer factorizations 
    respect to the disk. Automatica,  41, pp. 1855–1866, 2005. 
"""
function goifac(sys::DescriptorStateSpace; kwargs...)  
   sysi, syso, info = giofac(gdual(sys); kwargs...)
   return gdual(sysi), gdual(syso), info
end
"""
    giofac(sys; atol = 0, atol1 = atol, atol2 = atol, atol3 = atol, rtol, 
           fast = true, minphase = true, offset = sqrt(ϵ)) -> (sysi, syso, info)

Compute for the descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`, 
the square inner factor `sysi = (Ai-λEi,Bi,Ci,Di)` with the transfer function matrix `Gi(λ)` 
and the minimum-phase quasi-outer factor or the full row rank factor `syso = (Ao-λEo,Bo,Co,Do)` 
with the transfer function matrix `Go(λ)` such that
   
     G(λ) = Gi[:,1:r](λ)*Go(λ)    (*),

where `r` is the normal rank of `G(λ)`. The resulting proper and stable inner factor satisfies 
`Gi'(λ)*Gi(λ) = I`. If `sys` is stable (proper), then the resulting `syso` is stable (proper). 
The resulting factor `Go(λ)` has full row rank `r`. Depending on the selected factorization option,
if `minphase = true`, then `Go(λ)` is minimum phase,  excepting possibly zeros on the 
boundary of the appropriate stability domain `Cs`, or if `minphase = false`, then `Go(λ)` 
contains all zeros of `G(λ)`, in which case (*) is the extended QR-like factorization of `G(λ)`.
For a continuous-time system `sys`, the stability domain `Cs` is defined as the set of 
complex numbers with real parts at most `-β`, 
while for a discrete-time system `sys`, `Cs` is the set of complex numbers with 
moduli at most `1-β` (i.e., the interior of a disc of radius `1-β` centered in the origin). 
The boundary offset  `β` to be used to assess the stability of zeros and their number 
on the boundary of `Cs` can be specified via the keyword parameter `offset = β`.
Accordingly, for a continuous-time system, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time system, the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The resulting named triple `ìnfo` contains `(nrank, nfuz, niuz) `, where `ìnfo.nrank = r`, 
the normal rank of `G(λ)`, `ìnfo.nfuz` is the number of finite zeros of `syso` on 
the boundary of `Cs`, and `ìnfo.niuz` is the number of infinite zeros of `syso`. 
`ìnfo.nfuz` is set to `missing` if `minphase = false`. 

_Note:_ `syso` may generally contain a _free inner factor_, which can be eliminated by 
removing the finite unobservable eigenvalues. 

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A` and `B`, 
the absolute tolerance for the nonzero elements of `E`, 
the absolute tolerance for the nonzero elements of `C` and `D`,  
and the relative tolerance for the nonzero elements of `A`, `E`, `B`, `C` and `D`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`, `atol3 = atol`. 

For the assessment of zeros, the system pencil `[A-λE B; C D]` is reduced to a 
special Kronecker-like form (see [2]). In this reduction, the 
performed rank decisions are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_  For a continuous-time system, the factorization algorithm of [1] is used, while 
for a discrete-time system, the factorization algorithm of [1] is used.

_References:_

[1] C. Oara and A. Varga.
    Computation of the general inner-outer and spectral factorizations.
    IEEE Trans. Autom. Control, vol. 45, pp. 2307-2325, 2000.

[2] C. Oara.
    Constructive solutions to spectral and inner–outer factorizations 
    respect to the disk. Automatica,  41, pp. 1855–1866, 2005. 
"""
function giofac(sys::DescriptorStateSpace{T}; atol::Real = zero(real(T)), 
             atol1::Real = atol, atol2::Real = atol, atol3::Real = atol, 
             rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(max(atol1,atol2,atol3)), 
             offset::Real = sqrt(eps(float(real(T)))), fast::Bool = true, minphase::Bool = true) where T 

   disc = !iszero(sys.Ts)

   # enforce controllability of sys
   sys = gir(sys, atol1 = atol1, atol2 = atol2, obs = false) 

   #  Reduce the system matrix pencil to the special Kronecker-like form
   #
   #                 ( Arg-λ*Erg     *        *   *   )
   #                 (   0        Abl-λ*Ebl  Bbl  *   )
   #       At-λ*Et = (   0           0        0   Bn  )
   #                 (--------------------------------)
   #                 (   0          Cbl      Dbl  *   )
   #
   # where the subpencil
   #                           ( Abl-λ*Ebl  Bbl )
   #                           (   Cbl      Ddl )
   #
   # is full column rank, the pair (Abl-λ*Ebl,Bbl) is stabilizable, and
   # Abl-λ*Ebl contains either no zeros of sys if 
   # minphase = false, or contains all unstable zeros of sys if minphase = true. 
   
   
   At, Et, _, Z, dimsc, nmszer, nizer = gsklf(dssdata(sys)...; disc = disc, 
                jobopt = minphase ? "unstable" : "none",  
                offset = offset, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, 
                withQ = false, withZ = true)  
   
   n, m = size(sys.B); p = size(sys.D,1);

   # form the reduced system (Abl-λ*Ebl,Bbl,Cbl,Dbl)         
   nr = dimsc[1]; nric = dimsc[2]; mric = dimsc[3]; nsinf = dimsc[4]; 
   il = n-nsinf-nric+1:n-nsinf; jal = nr+1:nr+nric; jbl = nr+nric+1:nr+nric+mric; icl = n+1:n+p
   # A = At[il,jal]; E = Et[il,jal]; B = At[il,jbl]; C = At[icl,jal]; D = At[icl,jbl];
   A = view(At,il,jal); E = view(Et,il,jal); B = view(At,il,jbl); C = view(At,icl,jal); D = view(At,icl,jbl)

   # move unstable zeros to stable positions and compress the reduced system to one
   # with full row rank transfer function matrix
   if disc
      # X, _,F, = gared(A,E,B,D'*D,C'*C,C'*D); 
      X, _,F, = try 
         gared(A,E,B,D'*D,C'*C,C'*D) 
      catch err
         findfirst("dichotomic",string(err)) === nothing ? error("$err") : 
            error("Solution of the DARE failed: Symplectic pencil has eigenvalues on the unit circle")            
      end
   else
      # X, _, F, = garec(A,E,B,D'*D,C'*C,C'*D); 
      X, _, F, = try 
         garec(A,E,B,D'*D,C'*C,C'*D) 
      catch err
         findfirst("dichotomic",string(err)) === nothing ? error("$err") :
            error("Solution of the CARE failed: Hamiltonian pencil has jw-axis eigenvalues") 
      end
   end

   # assemble the inner and outer factors
   if disc
      mm1 = p-mric
      H = cholesky(Hermitian(D'*D+B'*X*B)).U 
      if mm1 > 0
         V2 = svd([A'*X C'; B'*X D'], full = true).V[:,nric+mric+1:nric+p]
         Y = view(V2,1:nric,:)
         W = view(V2,nric+1:nric+p,:)
         U = cholesky(Hermitian(W'*W+Y'*X*Y)).U;
         rdiv!(V2,U)  # compute Y/U and W/U
      else
         Y = zeros(T,nric,mm1); W = zeros(T,p,mm1)
      end
   else
      FQR = qr(D); 
      H = FQR.R[1:mric,:]; W = FQR.Q[:,mric+1:p] 
      rank(X) < nric ? Y = -pinv(X)*((E')\C'*W) : Y = -(E'*X)\C'*W
   end
    
   # construct the square inner factor
   agi = A-B*F; bgi = [B/H Y]; cgi = C-D*F; dgi = [D/H W]; 

   # convert the inner factor to a standard state-space realization if sys
   # is in a standard state-space form
   sys.E == I ? (E = UpperTriangular(E); sysi = dss(rdiv!(agi,E), bgi, rdiv!(cgi,E), dgi, Ts = sys.Ts) ) : 
                 (sysi = dss(agi, copy(E), bgi, cgi, dgi, Ts = sys.Ts))

   # construct the outer factor
   CDt = [H*F H]*Z[:,nr+1:n+m-nsinf]' 
   syso = dss(sys.A, sys.E, sys.B, CDt[:,1:n], CDt[:,n+1:n+m], Ts = sys.Ts);

   info = (nrank = mric, nfuz = nmszer, niuz = nizer)

   return sysi, syso, info

end
"""
    glcf(sys; smarg, sdeg, evals, mindeg = false, mininf = false, fast = true, 
         atol = 0, atol1 = atol, atol2 = atol, atol3 = atol, rtol = n*ϵ) -> (sysn, sysm)

Compute for the descriptor system `sys = (A-λE,B,C,D)`, the factors 
`sysn = (An-λEn,Bn,Cn,Dn)` and `sysm = (Am-λEm,Bm,Cm,Dm)` of its stable and proper
left coprime factorization. If `sys`, `sysn` and `sysm`  
have the transfer function matrices `G(λ)`, `N(λ)` and `M(λ)`, respectively, then
`G(λ) = inv(M(λ))*N(λ)`, with `N(λ)` and `M(λ)` proper and stable transfer 
function matrices. 
The resulting matrix pairs `(An,En)` and `(Am,Em)` are in (generalized) Schur form. 
The stability domain `Cs` of poles is defined by 
the keyword argument `smarg` for the stability margin, as follows: 
for a continuous-time system `sys`, `Cs` is the set of complex numbers 
with real parts at most `smarg`, 
while for a discrete-time system `sys`, `Cs` is the set of complex numbers with 
moduli at most `smarg < 1` (i.e., the interior of a disc of radius `smarg` centered in the origin). 
If `smarg` is missing, then the employed default values are `smarg = -sqrt(eps)` 
for a continuous-time system and `smarg = 1-sqrt(eps)` for a discrete-time system. 

The keyword argument `sdeg` specifies the prescribed stability degree for the 
assigned eigenvalues of the factors. If both `sdeg` and `smarg` are missing, 
then the employed  default values are `sdeg = -0.05` for a continuous-time system and 
`sdeg = 0.95` for a discrete-time system, while if `smarg` is specified, 
then `sdeg = smarg` is used. 

The keyword argument `evals` is a real or complex vector, which contains a set 
of finite desired eigenvalues for the factors. 
For a system with real data, `evals` must be a self-conjugated complex set 
to ensure that the resulting factors are also real. 
   
If `mindeg = false`, both factors `sysn` and `sysm` have descriptor realizations
with the same order and with `An = Am`, `En = Em` and `Cn = Cm`. If `mindeg = true`, 
the realization of `sysm` is minimal. The number of (finite) poles of `sysm` is 
equal to the number of unstable finite poles of `sys`. 

If `mininf = false`, then `An-λEn` and `Am-λEm` may have simple infinite eigenvalues.
If `mininf = true`,  then `An-λEn` and `Am-λEm` have no simple infinite eigenvalues.
Note that the removing of simple infinite eigenvalues involves matrix inversions. 

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, 
the absolute tolerance for the nonzero elements of `E`, 
the absolute tolerance for the nonzero elements of `C`,  
and the relative tolerance for the nonzero elements of `A`, `E` and `C`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the machine epsilon of the element type of `A`
and `n` is the order of the system `sys`. 
The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`, `atol3 = atol`.

The preliminary separation of finite and infinite eigenvalues of `A-λE` is performed 
using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_  The dual of Procedure GRCF from [2] is used, which represents
an extension of the recursive factorization approach of [1] to cope with  
infinite poles. All infinite eigenvalues are assigned to finite real values. 
If `evals` is missing or does not contain a sufficient 
number of real values, then a part or all of infinite eigenvalues of `A-λE` are 
assigned to the value specified by `sdeg`.  
The pairs `(An,En)` and `(Am,Em)`  result in _generalized Schur form_ with 
both `An` and `Am` quasi-upper triangular 
and `En` and `Em` either both upper triangular or both UniformScalings.

_References:_

[1] A. Varga. Computation of coprime factorizations of rational matrices.
    Linear Algebra and Its Applications, vol. 271, pp.88-115, 1998.

[2] A. Varga. On recursive computation of coprime factorizations of rational matrices. 
    arXiv:1703.07307, https://arxiv.org/abs/1703.07307, 2020. (to appear in Linear Algebra and Its Applications)
"""
function glcf(sys::DescriptorStateSpace; kwargs...)  
              sysn, sysm = grcf(gdual(sys,rev = true); kwargs...)
   return gdual(sysn,rev = true), gdual(sysm,rev = true)
end
"""
    grcf(sys; smarg, sdeg, evals, mindeg = false, mininf = false, fast = true, 
         atol = 0, atol1 = atol, atol2 = atol, atol3 = atol, rtol = n*ϵ) -> (sysn, sysm)

Compute for the descriptor system `sys = (A-λE,B,C,D)`, the factors 
`sysn = (An-λEn,Bn,Cn,Dn)` and `sysm = (Am-λEm,Bm,Cm,Dm)` of its stable and proper
right coprime factorization. If `sys`, `sysn` and `sysm`  
have the transfer function matrices `G(λ)`, `N(λ)` and `M(λ)`, respectively, then
`G(λ) = N(λ)*inv(M(λ))`, with `N(λ)` and `M(λ)` proper and stable transfer 
function matrices. 
The resulting matrix pairs `(An,En)` and `(Am,Em)` are in (generalized) Schur form. 
The stability domain `Cs` of poles is defined by 
the keyword argument `smarg` for the stability margin, as follows: 
for a continuous-time system `sys`, `Cs` is the set of complex numbers 
with real parts at most `smarg < 0`, 
while for a discrete-time system `sys`, `Cs` is the set of complex numbers with 
moduli at most `smarg < 1` (i.e., the interior of a disc of radius `smarg` centered in the origin). 
If `smarg` is missing, then the employed default values are `smarg = -sqrt(eps)` 
for a continuous-time system and `smarg = 1-sqrt(eps)` for a discrete-time system. 

The keyword argument `sdeg` specifies the prescribed stability degree for the 
assigned eigenvalues of the factors. If both `sdeg` and `smarg` are missing, 
then the employed  default values are `sdeg = -0.05` for a continuous-time system and 
`sdeg = 0.95` for a discrete-time system, while if `smarg` is specified, 
then `sdeg = smarg` is used. 

The keyword argument `evals` is a real or complex vector, which contains a set 
of finite desired eigenvalues for the factors. 
For a system with real data, `evals` must be a self-conjugated complex set 
to ensure that the resulting factors are also real. 
   
If `mindeg = false`, both factors `sysn` and `sysm` have descriptor realizations
with the same order and with `An = Am`, `En = Em` and `Bn = Bm`. If `mindeg = true`, 
the realization of `sysm` is minimal. The number of (finite) poles of `sysm` is 
equal to the number of unstable finite poles of `sys`. 

If `mininf = false`, then `An-λEn` and `Am-λEm` may have simple infinite eigenvalues.
If `mininf = true`,  then `An-λEn` and `Am-λEm` have no simple infinite eigenvalues.
Note that the removing of simple infinite eigenvalues involves matrix inversions. 

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, 
the absolute tolerance for the nonzero elements of `E`, 
the absolute tolerance for the nonzero elements of `B`,  
and the relative tolerance for the nonzero elements of `A`, `E` and `B`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the machine epsilon of the element type of `A` 
and `n` is the order of the system `sys`.
The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`, `atol3 = atol`.

The preliminary separation of finite and infinite eigenvalues of `A-λE` is performed 
using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_  The Procedure GRCF from [2] is implemented, which represents
an extension of the recursive factorization approach of [1] to cope with  
infinite eigenvalues. All infinite poles are assigned to finite real values. 
If `evals` is missing or does not contain a sufficient 
 number of real values, then a part or all of infinite eigenvalues of `A-λE` are 
 assigned to the value specified by `sdeg`. The pairs `(An,En)` and `(Am,Em)`
 result in _generalized Schur form_ with both `An` and `Am` quasi-upper triangular 
 and `En` and `Em` either both upper triangular or both UniformScalings. 

_References:_

[1] A. Varga. Computation of coprime factorizations of rational matrices.
    Linear Algebra and Its Applications, vol. 271, pp.88-115, 1998.

[2] A. Varga. On recursive computation of coprime factorizations of rational matrices. 
    arXiv:1703.07307, https://arxiv.org/abs/1703.07307, 2020. (to appear in Linear Algebra and Its Applications)
"""
function grcf(sys::DescriptorStateSpace{T}; 
              evals::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, smarg::Union{Real,Missing} = missing, 
              atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, atol3::Real = atol, 
              rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(max(atol1,atol2,atol3)), 
              fast::Bool = true, mindeg::Bool = false, mininf::Bool = false) where T 

   T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
   disc = !iszero(sys.Ts)
   missingsmarg = ismissing(smarg)
   if missingsmarg
      offset = sqrt(eps(real(T1))); 
      smarg = disc ?  1-offset : -offset
   else
      disc && (smarg < 0 || smarg >= 1) && error("smarg must be non-negative and subunitary for a discrete system")
      !disc && smarg >= 0 && error("smarg must be negative for a continuous system")
   end

   if ismissing(evals)
      evals1 = missing
   else 
      evals1 = copy_oftype(evals,promote_type(T1,eltype(evals)))
   end

   # set default values of sdeg if evals = missing
   if ismissing(sdeg)
      if missingsmarg
         sdeg = disc ? real(T1)(0.95) : real(T1)(-0.05)
      else 
         sdeg = smarg
      end
   else
      sdeg = real(T1)(sdeg)
      sdeg <= smarg || error("sdeg must be must be at most $smarg")
   end
 
   At, Et, Bt, CN, DN = dssdata(T1,sys)
   n, m = size(Bt) 
   standsys = (Et == I)


   # quick exit for n = 0 or B = 0
   DM = Matrix{T1}(I,m,m)

   n == 0 && (return sys, dss(DM,Ts = sys.Ts)) 

   ZERO = zero(T1)
   ZEROR = zero(real(T1))
   ONE = one(T1)

   #  # check for zero rows in the leading positions
   #  ilob = n+1
   #  for i = 1:n
   #      !iszero(view(Bt,i,:)) && (ilob = i; break)
   #  end
 
   #  # return if B = 0
   #  ilob > n && (return sys, dss(DM,Ts = sys.Ts) ) 
 
   #  # check for zero rows in the trailing positions
   #  ihib = ilob
   #  for i = n:-1:ilob+1
   #      !iszero(view(Bt,i,:)) && (ihib = i; break)
   #  end
    
   #  # operate only on the nonzero rows of B
   #  ib = ilob:ihib
   #  nrmB = opnorm(view(Bt,ib,:),1)

   nrmB = opnorm(Bt,1)
   # return if B = 0
   iszero(nrmB) && (return gsvselect(sys,Int[]), dss(DM,Ts = sys.Ts) ) 
 
     
   complx = (T1 <: Complex)
    
   # sort desired eigenvalues
   if ismissing(evals1) 
      evalsr = missing
      evalsc = missing
   else
      if complx
         evalsr = copy(evals1)
         evalsc = missing
      else
         evalsr = evals1[imag.(evals1) .== 0]
         isempty(evalsr) && (evalsr = missing)
         tempc = evals1[imag.(evals1) .> 0]
         if isempty(tempc)
            evalsc = missing
         else
            tempc1 = conj(evals1[imag.(evals1) .< 0])
            isequal(tempc[sortperm(real(tempc))],tempc1[sortperm(real(tempc1))]) ||
                    error("evals must be a self-conjugated complex vector")
            evalsc = [transpose(tempc[:]); transpose(conj.(tempc[:]))][:]
         end
      end
      # check that all eigenvalues are inside of the stability region
      !ismissing(sdeg) && ((disc && any(abs.(evals1) .> sdeg) )  || (!disc && any(real.(evals1) .> sdeg)))  &&
            error("The elements of evals must lie in the stability region")
   end     
      
   nrmA = opnorm(At,1)
   tola = max(atol1, rtol*nrmA)
   tolb = max(atol3, rtol*nrmB)

   if standsys
      #
      # separate stable and unstable parts with respect to sdeg
      # compute orthogonal Z  such that
      #
      #      Z^T*A*Z = [ Ag   * ]
      #                [  0  Ab ]
      #
      # where Ag has eigenvalues within the stability degree region
      # and Ab has eigenvalues outside the stability degree region.
    
      _, Z, α = LAPACK.gees!('V', At)

      if disc
         select = Int.(abs.(α) .<= smarg)
      else
         select = Int.(real.(α) .<= smarg)
      end
      nsinf = 0; nb = length(select[select .== 0]); ng = n-nb; nb = n-ng; nbi = 0;
      nb == 0 && (Bt = Z'*Bt; return dss(At,I,Bt,CN*Z,DN,Ts=sys.Ts), mindeg ? dss(DM,Ts = sys.Ts) : dss(At,I,Bt,zeros(T1,m,n),DM,Ts=sys.Ts)  )      
   
       _, _, α = LAPACK.trsen!(select, At, Z) 
       Bt = Z'*Bt; CN = CN*Z 
   else
      # reduce the pair (A,E) to the specially ordered generalized real Schur
      # form (At,Et) = (Q'*A*Z,Q'*E*Z), with
      #
      #             [ Ai  *   *   *  ]          [ 0  *    *   *  ]
      #        At = [  0  Ag  *   *  ]  ,  Et = [ 0  Eg   *   *  ] ,
      #             [  0  0  Abf  *  ]          [ 0  0   Ebf  *  ]
      #             [  0  0   0  Abi ]          [ 0  0    0  Ebi ]
      # 
      # where
      #    (Ai,0)    contains the firt order inifinite eigenvalues
      #    (Ag,Eg)   contains the "good" finite eigenvalues
      #    (Abf,Ebf) contains the "bad" finite eigenvalues
      #    (Abi,Ebi) contains the "bad" (higher order) infinite eigenvalues

      At, Et, Q, Z, _, (nsinf, ng, nbf, nbi) = sfischursep(At, Et, smarg = smarg, disc = disc, 
                                                          fast = fast, finite_infinite = true, stable_unstable = true, 
                                                          atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = true, withZ = true);
      # finish if SYS is stable and Et is invertible or
      # SYS is stable and simple infinite eigenvalues have not to be removed 
      Bt = Q'*Bt; CN = CN*Z; 
      (nbf+nsinf == 0 || (nbf+nbi == 0 && !mininf)) && (return dss(At,Et,Bt,CN,DN,Ts=sys.Ts), mindeg ? dss(DM,Ts = sys.Ts) : dss(At,Et,Bt,zeros(T1,m,n),DM,Ts=sys.Ts) )
      nb = nbf+nbi
   end
   #return dss(At,Et,Bt,CN,DN,Ts=sys.Ts), dss(DM,Ts = sys.Ts)
   # initialization of the recursive factorization CN, DN and DM already initialized
   CM = zeros(T1,m,n); 
   fnrmtol = 10000*max(1,nrmA)/nrmB;  # set threshold for high feedback
   fwarn = 0;
   while nb > 0
      if nbi > 0 || nb == 1 || complx || At[n,n-1] == 0
         k = 1
      else
         k = 2
      end 
      kk = n-k+1:n;
      if standsys
         evb = ordeigvals(view(At,kk,kk));
      else
         evb = ordeigvals(view(At,kk,kk),view(Et,kk,kk))[1]
      end
      if norm(view(Bt,kk,:)) <= tolb
         nb -= k; n -= k; 
         nbi > 0 && (nbi -= 1)  
      else
         a2 = At[kk,kk]; b2 = Bt[kk,:]; 
         e2 = standsys ? Matrix{T1}(I,k,k) : Et[kk,kk]
         if !standsys && k == 1 && nbi > 0
            # move infinite eigenvalue to a stable (finite) position
            γ, evalsr = eigselect1(evalsr, sdeg, sdeg, disc; cflag = complx);
            γ === nothing && (γ = sdeg)
            F = qr(b2') 
            q2 = Matrix(F.Q)
            f2 = -(q2/F.R)*a2; W =  I-q2*q2';
            # update matrices
            i1 = 1:n-1;
            mul!(view(At,i1,kk),view(Bt,i1,:),f2,ONE,ONE); At[n,n] = γ; Et[n,n] = ONE;
            mul!(view(CN,:,kk),DN,f2,ONE,ONE)
            mul!(view(CM,:,kk),DM,f2,ONE,ONE)
            Bt[i1,:] = view(Bt,i1,:)*W; DN = DN*W; DM = DM*W;
            i2 = n-nb+1:n; i1 = 1:n-nb;
            if nb > 1
               Q3 = Matrix{T1}(I,nb,nb); Z3 = Matrix{T1}(I,nb,nb)
               MatrixPencils.tgexc!(true, true, nb, 1, view(At,i2,i2), view(Et,i2,i2), Q3, Z3) 
               At[i1,i2] = view(At,i1,i2)*Z3; Et[i1,i2] = view(Et,i1,i2)*Z3;
               Bt[i2,:] = Q3'*view(Bt,i2,:); 
               CN[:,i2] = view(CN,:,i2)*Z3; CM[:,i2] = view(CM,:,i2)*Z3;
            end
            nb -= 1; nbi -= 1;
         else
            if k == 1 
               if !(nb > 1 && ismissing(evalsr) && !ismissing(evalsc))
                   # assign a single eigenvalue 
                   γ, evalsr = eigselect1(evalsr, sdeg, complx ? evb[1] : real(evb[1]), disc; cflag = complx);
               else
                  γ = nothing
               end
               if γ === nothing
                  # no real pole available, adjoin a new 1x1 block if possible
                  if nb == 1
                     # incompatible poles with the eigenvalue structure
                     # assign the last real pole to SDEG (if possible)
                     # warning(['No real eigenvalue available for assignment: assigning instead SDEG = ', num2str(sdeg)])
                     f2 = -b2\(a2-e2*sdeg);
                  else
                     # adjoin a real block or interchange the last two blocks
                     k = 2; kk = n-k+1:n; 
                     if n == 2 || iszero(At[n-1,n-2])
                        # adjoin blocks and update evb 
                        if standsys
                           evb = ordeigvals(view(At,kk,kk));
                        else
                           evb = ordeigvals(view(At,kk,kk),view(Et,kk,kk))[1];
                        end
                     else
                        # interchange last two blocks
                        i1 = 1:n-3; i2 = n-2:n; i3 = n-1:n
                        if standsys
                           Z3 = Matrix{T1}(I,3,3)
                           ac = view(At,i3,i3)
                           if nb > 2 && At[n-1,n-2] != ZERO
                              # interchange last two blocks
                              LAPACK.trexc!('V', 3, 1, view(At,i2,i2), Z3) 
                              evb = ordeigvals(ac)
                           else
                              evb = disc ? maximum(abs.(ordeigvals(ac))) : maximum(real(ordeigvals(ac)))
                           end
                           At[i1,i2] = view(At,i1,i2)*Z3; Bt[i2,:] = Z3'*view(Bt,i2,:); 
                           CN[:,i2] = view(CN,:,i2)*Z3; CM[:,i2] = view(CM,:,i2)*Z3;
                        else
                           Q3 = Matrix{T1}(I,3,3); Z3 = Matrix{T1}(I,3,3)
                           ac = view(At,i3,i3); ec = view(Et,i3,i3); 
                           if nb > 2 && At[n-1,n-2] != ZERO
                              # interchange last two blocks
                              MatrixPencils.tgexc!(true, true, 3, 1, view(At,i2,i2), view(Et,i2,i2), Q3, Z3) 
                              evb = ordeigvals(ac,ec)[1]
                           else
                              evb = disc ? maximum(abs.(ordeigvals(ac,ec)[1])) : maximum(real(ordeigvals(ac,ec)[1]))
                           end
                           At[i1,i2] = view(At,i1,i2)*Z3; Et[i1,i2] = view(Et,i1,i2)*Z3; 
                           Bt[i2,:] = Q3'*view(Bt,i2,:); 
                           CN[:,i2] = view(CN,:,i2)*Z3; CM[:,i2] = view(CM,:,i2)*Z3;
                        end
                     end
                     a2 = At[kk,kk]; b2 = Bt[kk,:]; 
                     e2 = standsys ? Matrix{T1}(I,k,k) : Et[kk,kk]
                  end
               else
                  f2 = -b2\(a2-e2*γ);
               end
            end
            if k == 2
               # assign a pair of eigenvalues 
               γ, evalsr, evalsc = eigselect2(evalsr,evalsc,sdeg,evb[end],disc)
               f2, u, v = saloc2(a2,e2,b2,γ,tola,tolb)
               if f2 === nothing  # the case b2 = 0 can not occur
                  i1 = 1:n; 
                  At[kk,kk] = u'*view(At,kk,kk); At[i1,kk] = view(At,i1,kk)*v;
                  At[n,n-1] = ZERO
                  if !standsys 
                     Et[kk,kk] = u'*view(Et,kk,kk); Et[i1,kk] = view(Et,i1,kk)*v; 
                     Et[n,n-1] = ZERO
                  end
                  nb -= 1; n -= 1; 
                  # recover the failed selection 
                  imag(γ[1]) == 0 ? (ismissing(evalsr) ? evalsr = γ : evalsr = [γ; evalsr]) : (ismissing(evalsc) ? evalsc = γ : evalsc = [γ; evalsc])
                  Bt[kk,:] = u'*view(Bt,kk,:);
                  CN[:,kk] = view(CN,:,kk)*v; CM[:,kk] = view(CM,:,kk)*v;
               end
            end
            if f2 !== nothing
               norm(f2,Inf) > fnrmtol && (fwarn += 1)
               i1 = 1:n
               mul!(view(At,i1,kk),view(Bt,i1,:),f2,ONE,ONE)
               mul!(view(CN,:,kk),DN,f2,ONE,ONE)
               mul!(view(CM,:,kk),DM,f2,ONE,ONE)
               select = [zeros(Int,nb-k); ones(Int,k)]; 
               i2 = n-nb+1:n; i1 = 1:n-nb;
               if standsys
                  if k == 2 
                     # standardization step is necessary to use trsen 
                     k1 = kk[1]; k2 = kk[2]
                     RT1R, RT1I, RT2R, RT2I, CS, SN = lanv2(At[k1,k1], At[k1,k2], At[k2,k1], At[k2,k2]) 
                     Z2 = LinearAlgebra.Givens(1, 2, CS, -SN)
                     rmul!(view(At,1:n,kk),Z2)
                     lmul!(Z2',view(At,kk,kk)); 
                     At[k1,k1] = RT1R; At[k2,k2] = RT2R;
                     rmul!(view(CN,:,kk),Z2)
                     rmul!(view(CM,:,kk),Z2)
                     lmul!(Z2',view(Bt,kk,:))
                     tworeals = iszero(At[n,n-1])
                  else
                     tworeals = false
                  end
                  # reorder eigenvalues 
                  if nb > k
                     Z3 = Matrix{T1}(I,nb,nb)
                     LAPACK.trexc!('V', nb-k+1, 1, view(At,i2,i2), Z3) 
                     tworeals && LAPACK.trexc!('V', nb, 2, view(At,i2,i2), Z3)
                     At[i1,i2] = view(At,i1,i2)*Z3; Bt[i2,:] = Z3'*view(Bt,i2,:); 
                     CN[:,i2] = view(CN,:,i2)*Z3; CM[:,i2] = view(CM,:,i2)*Z3;
                  end
               else 
                  if k == 2
                     # standardization step is necessary to use tgsen
                     it = 1:n-2; 
                     _, _, _, _, Q2, Z2 = LAPACK.gges!('V','V',view(At,kk,kk),view(Et,kk,kk))
                     At[it,kk] = view(At,it,kk)*Z2
                     Et[it,kk] = view(Et,it,kk)*Z2
                     Bt[kk,:] = Q2'*view(Bt,kk,:)
                     CN[:,kk] = view(CN,:,kk)*Z2; CM[:,kk] = view(CM,:,kk)*Z2;
                     tworeals = iszero(At[n,n-1])
                  else
                     tworeals = false
                  end
                  if nb > k
                     Q3 = Matrix{T1}(I,nb,nb); Z3 = Matrix{T1}(I,nb,nb)
                     MatrixPencils.tgexc!(true, true, nb-k+1, 1, view(At,i2,i2), view(Et,i2,i2), Q3, Z3) 
                     tworeals && MatrixPencils.tgexc!(true, true, nb, ia+1, view(At,i2,i2), view(Et,i2,i2), Q3, Z3) 
                     At[i1,i2] = view(At,i1,i2)*Z3; Et[i1,i2] = view(Et,i1,i2)*Z3;
                     Bt[i2,:] = Q3'*view(Bt,i2,:); 
                     CN[:,i2] = view(CN,:,i2)*Z3; CM[:,i2] = view(CM,:,i2)*Z3;
                  end
               end
               nb -= k;
            end
         end
      end
   end

   if mininf && nsinf > 0
      # remove the simple infinite eigenvalues
      i1 = 1:nsinf; i2 = nsinf+1:n;
      ca = view(CN,:,i1)/view(At,i1,i1); ee = view(Et,i1,i2)/view(Et,i2,i2);
      DN = DN-ca*Bt[i1,:]+ca*ee*Bt[i2,:]; 
      CN = CN[:,i2]-ca*At[i1,i2]+ca*ee*At[i2,i2]; CM = CM[:,i2];
      At = At[i2,i2]; Et = Et[i2,i2]; Bt = Bt[i2,:];
      n = n-nsinf; 
   end
  
   if standsys
      i1 = 1:n
      sysn = dss(At[i1,i1],I,Bt[i1,:],CN[:,i1],DN,Ts=sys.Ts);
      mindeg && (i1 = ng+1:n)
      sysm = dss(At[i1,i1],I,Bt[i1,:],CM[:,i1],DM,Ts=sys.Ts);
   else
      i1 = 1:n
      sysn = dss(At[i1,i1],Et[i1,i1],Bt[i1,:],CN[:,i1],DN,Ts=sys.Ts);
      mindeg && (i1 = ng+1:n)
      sysm = dss(At[i1,i1],Et[i1,i1],Bt[i1,:],CM[:,i1],DM,Ts=sys.Ts);
   end
    
   fwarn > 0 && @warn("Possible loss of numerical reliability due to high feedback gain")
   return sysn, sysm

# end GRCF
end
"""
    glcfid(sys; mindeg = false, mininf = false, fast = true, offset = sqrt(ϵ), 
           atol = 0, atol1 = atol, atol2 = atol, atol3 = atol, rtol = n*ϵ) -> (sysni, sysmi)

Compute for the descriptor system `sys = (A-λE,B,C,D)`, the factors 
`sysni = (Ani-λEni,Bni,Cni,Dni)` and `sysmi = (Ami-λEmi,Bmi,Cmi,Dmi)` of its 
left coprime factorization with inner denominator. If `sys`, `sysni` and `sysmi`  
have the transfer function matrices `G(λ)`, `N(λ)` and `M(λ)`, respectively, then
`G(λ) = inv(M(λ))*N(λ)`, with `N(λ)` and `M(λ)` proper and stable transfer 
function matrices and the denominator factor `M(λ)` inner. 
The resulting matrix pairs `(Ani,Eni)` and `(Ami,Emi)` are in Schur forms. 
The system `sys` must not have poles on the boundary of the stability domain `Cs`.
In terms of eigenvalues, this requires for a continuous-time system, that 
`A-λE` must not have controllable eigenvalues on the imaginary axis 
(excepting simple infinite eigenvalues), while for a discrete-time system,  
`A-λE` must not have controllable eigenvalues on the unit circle centered 
in the origin. 

To assess the presence of poles on the boundary of `Cs`, a boundary offset  `β` 
can be specified via the keyword parameter `offset = β`. 
Accordingly, for a continuous-time system, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time system, then the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

If `mindeg = false`, both factors `sysni` and `sysmi` have descriptor realizations
with the same order and with `Ani = Ami`, `Eni = Emi` and `Cni = Cmi`. If `mindeg = true`, 
the realization of `sysmi` is minimal. The number of (finite) poles of `sysmi` is 
equal to the number of unstable finite poles of `sys`. 

If `mininf = false`, then `Ani-λEni` and `Ami-λEmi` may have simple infinite eigenvalues.
If `mininf = true`,  then `Ani-λEni` and `Ami-λEmi` have no simple infinite eigenvalues.
Note that the removing of simple infinite eigenvalues involves matrix inversions. 

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, 
the absolute tolerance for the nonzero elements of `E`, 
the absolute tolerance for the nonzero elements of `C`,  
and the relative tolerance for the nonzero elements of `A`, `E` and `C`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`, `atol3 = atol`.

The preliminary separation of finite and infinite eigenvalues of `A-λE`is performed 
using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_ An extension of the recursive factorization approach of [1] is used 
to the dual system (see [2] for details). The pairs `(Ani,Eni)` and `(Ami,Emi)`
result in _generalized Schur form_ with both `Ani` and `Ami` quasi-upper triangular 
and `Eni` and `Emi` either both upper triangular or both UniformScalings. 

_References:_

[1] A. Varga. Computation of coprime factorizations of rational matrices.
    Linear Algebra and Its Applications, vol. 271, pp.88-115, 1998.

[2] A. Varga. On recursive computation of coprime factorizations of rational matrices. 
    arXiv:1703.07307, https://arxiv.org/abs/1703.07307, 2020. (to appear in Linear Algebra and Its Applications)
"""
function glcfid(sys::DescriptorStateSpace; kwargs...)  
   sysn, sysm = grcfid(gdual(sys,rev = true); kwargs...)
   return gdual(sysn,rev = true), gdual(sysm,rev = true)
end
"""
    grcfid(sys; mindeg = false, mininf = false, fast = true, offset = sqrt(ϵ), 
           atol = 0, atol1 = atol, atol2 = atol, atol3 = atol, rtol = n*ϵ) -> (sysni, sysmi)

Compute for the descriptor system `sys = (A-λE,B,C,D)`, the factors 
`sysni = (Ani-λEni,Bni,Cni,Dni)` and `sysmi = (Ami-λEmi,Bmi,Cmi,Dmi)` of its 
right coprime factorization with inner denominator. If `sys`, `sysni` and `sysmi`  
have the transfer function matrices `G(λ)`, `N(λ)` and `M(λ)`, respectively, then
`G(λ) = N(λ)*inv(M(λ))`, with `N(λ)` and `M(λ)` proper and stable transfer 
function matrices and the denominator factor `M(λ)` inner. 
The resulting matrix pairs `(Ani,Eni)` and `(Ami,Emi)` are in (generalized) Schur form. 
The system `sys` must not have poles on the boundary of the stability domain `Cs`.
In terms of eigenvalues, this requires for a continuous-time system, that 
`A-λE` must not have controllable eigenvalues on the imaginary axis 
(excepting simple infinite eigenvalues), while for a discrete-time system,  
`A-λE` must not have controllable eigenvalues on the unit circle centered 
in the origin. 

To assess the presence of poles on the boundary of `Cs`, a boundary offset  `β` 
can be specified via the keyword parameter `offset = β`. 
Accordingly, for a continuous-time system, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time system, then the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

If `mindeg = false`, both factors `sysni` and `sysmi` have descriptor realizations
with the same order and with `Ani = Ami`, `Eni = Emi` and `Bni = Bmi`. If `mindeg = true`, 
the realization of `sysmi` is minimal. The number of (finite) poles of `sysmi` is 
equal to the number of unstable finite poles of `sys`. 

If `mininf = false`, then `Ani-λEni` and `Ami-λEmi` may have simple infinite eigenvalues.
If `mininf = true`,  then `Ani-λEni` and `Ami-λEmi` have no simple infinite eigenvalues.
Note that the removing of simple infinite eigenvalues involves matrix inversions. 

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, 
the absolute tolerance for the nonzero elements of `E`, 
the absolute tolerance for the nonzero elements of `B`,  
and the relative tolerance for the nonzero elements of `A`, `E` and `B`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`, `atol3 = atol`.

The preliminary separation of finite and infinite eigenvalues of `A-λE`is performed 
using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_ An extension of the recursive factorization approach of [1] 
is used (see [2] for details). The pairs `(Ani,Eni)` and `(Ami,Emi)`
result in _generalized Schur form_ with both `Ani` and `Ami` quasi-upper triangular 
and `Eni` and `Emi` either both upper triangular or both UniformScalings. 

_References:_

[1] A. Varga. Computation of coprime factorizations of rational matrices.
    Linear Algebra and Its Applications, vol. 271, pp.88-115, 1998.

[2] A. Varga. On recursive computation of coprime factorizations of rational matrices. 
    arXiv:1703.07307, https://arxiv.org/abs/1703.07307, 2020. (to appear in Linear Algebra and Its Applications)
"""
function grcfid(sys::DescriptorStateSpace{T}; offset::Real = sqrt(eps(float(real(T)))), 
             atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, atol3::Real = atol, 
             rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(max(atol1,atol2,atol3)), 
             fast::Bool = true, mindeg::Bool = false, mininf::Bool = false) where T 

   T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
   disc = !iszero(sys.Ts)
   
   smarg = disc ?  1-offset : -offset

   At, Et, Bt, CN, DN = dssdata(T1,sys)
   n, m = size(Bt) 
   standsys = (Et == I)

   # quick exit for n = 0 or B = 0
   DM = Matrix{T1}(I,m,m)
   n == 0 && (return sys, dss(DM,Ts = sys.Ts) ) 

   ZERO = zero(T1)
   ZEROR = zero(real(T1))
   ONE = one(T1)

   #  # check for zero rows in the leading positions
   #  ilob = n+1
   #  for i = 1:n
   #      !iszero(view(Bt,i,:)) && (ilob = i; break)
   #  end
 
   #  # return if B = 0
   #  ilob > n && (return sys, dss(DM,Ts = sys.Ts) ) 
 
   #  # check for zero rows in the trailing positions
   #  ihib = ilob
   #  for i = n:-1:ilob+1
   #      !iszero(view(Bt,i,:)) && (ihib = i; break)
   #  end
    
   #  # operate only on the nonzero rows of B
   #  ib = ilob:ihib
   #  nrmB = opnorm(view(Bt,ib,:),1)

   nrmB = opnorm(Bt,1)
   # return if B = 0
   iszero(nrmB) && (return gsvselect(sys,Int[]), dss(DM,Ts = sys.Ts) ) 
 
   complx = (T1 <: Complex)
    
      
   nrmA = opnorm(At,1)
   tola = max(atol1, rtol*nrmA)
   tolb = max(atol3, rtol*nrmB)

   if standsys
      #
      # separate stable and unstable parts with respect to sdeg
      # compute orthogonal Z  such that
      #
      #      Z^T*A*Z = [ Ag   * ]
      #                [  0  Ab ]
      #
      # where Ag has eigenvalues within the stability degree region
      # and Ab has eigenvalues outside the stability degree region.   
      _, Z, α = LAPACK.gees!('V', At)
      if disc
         select = Int.(abs.(α) .<= smarg)
      else
         select = Int.(real.(α) .<= smarg)
      end
      nsinf = 0; nb = length(select[select .== 0]); ng = n-nb; nb = n-ng; nbi = 0;
      nb == 0 && (Bt = Z'*Bt; return dss(At,I,Bt,CN*Z,DN,Ts=sys.Ts), mindeg ? dss(DM,Ts = sys.Ts) : dss(At,I,Bt,zeros(T1,m,n),DM,Ts=sys.Ts)  )      
   
       _, _, α = LAPACK.trsen!(select, At, Z) 
       Bt = Z'*Bt; CN = CN*Z 
    else
      # reduce the pair (A,E) to the specially ordered generalized real Schur
      # form (At,Et) = (Q'*A*Z,Q'*E*Z), with
      #
      #             [ Ai  *   *   *  ]          [ 0  *    *   *  ]
      #        At = [  0  Ag  *   *  ]  ,  Et = [ 0  Eg   *   *  ] ,
      #             [  0  0  Abf  *  ]          [ 0  0   Ebf  *  ]
      #             [  0  0   0  Abi ]          [ 0  0    0  Ebi ]
      # 
      # where
      #    (Ai,0)    contains the firt order inifinite eigenvalues
      #    (Ag,Eg)   contains the "good" finite eigenvalues
      #    (Abf,Ebf) contains the "bad" finite eigenvalues
      #    (Abi,Ebi) contains the "bad" (higher order) infinite eigenvalues

      At, Et, Q, Z, _, (nsinf, ng, nbf, nbi) = sfischursep(At, Et, smarg = smarg, disc = disc, 
                                                          fast = fast, finite_infinite = true, stable_unstable = true, 
                                                          atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = true, withZ = true);
      #!discr && nbi > 0 && error("The continuous-time system SYS is improper")
      # finish if SYS is stable and Et is invertible or
      # SYS is stable and simple infinite eigenvalues have not to be removed 
      Bt = Q'*Bt; CN = CN*Z; 
      (nbf+nsinf == 0 || (nbf+nbi == 0 && !mininf)) && (return dss(At,Et,Bt,CN,DN,Ts=sys.Ts), mindeg ? dss(DM,Ts = sys.Ts) : dss(At,Et,Bt,zeros(T1,m,n),DM,Ts=sys.Ts) )
      nb = nbf+nbi
   end

   # initialization of the recursive factorization CN, DN and DM already initialized
   CM = zeros(T1,m,n); 
   fnrmtol = 10000*max(1,nrmA)/nrmB;  # set threshold for high feedback
   fwarn = 0;
   while nb > 0
      if nbi > 0 || nb == 1 || complx || At[n,n-1] == 0
         k = 1
      else
         k = 2
      end 
      kk = n-k+1:n;
      if standsys
         evb = ordeigvals(view(At,kk,kk));
      else
         evb = ordeigvals(view(At,kk,kk),view(Et,kk,kk))[1]
      end
      if norm(view(Bt,kk,:)) <= tolb
         nb -= k; n -= k; 
         nbi > 0 && (nbi -= 1)  
      else
         a2 = At[kk,kk]; b2 = Bt[kk,:]; 
         e2 = standsys ? Matrix{T1}(I,k,k) : Et[kk,kk]
         if !standsys && k == 1 && nbi > 0
            !disc && error("The continuous-time system SYS is improper")
            # reflect infinite eigenvalue to the origin
            F = qr(b2') 
            q2 = Matrix(F.Q)
            f2 = -(q2/F.R)*a2; W =  I-q2*q2';
            # update matrices
            i1 = 1:n-1;
            mul!(view(At,i1,kk),view(Bt,i1,:),f2,ONE,ONE); At[n,n] = ZERO; 
            Bt[i1,:] = view(Bt,i1,:)*W; 
            if complx
               en = -a2[1,1]               
               (iszero(imag(en)) && real(en) > ZEROR) ? Et[n,n] = en : 
                  (tmp = conj(en)/abs(en); Et[n,n] = tmp*en; [@inbounds Bt[n,j] *= tmp for j = 1:m])
            else
               a2[1,1] > 0 ? (Et[n,n] = a2[1,1]; Bt[n,:] = -Bt[n,:]) : Et[n,n] = -a2[1,1]
            end
            mul!(view(CN,:,kk),DN,f2,ONE,ONE)
            mul!(view(CM,:,kk),DM,f2,ONE,ONE)
            DN = DN*W; DM = DM*W;
            i2 = n-nb+1:n; i1 = 1:n-nb;
            if nb > 1
               Q3 = Matrix{T1}(I,nb,nb); Z3 = Matrix{T1}(I,nb,nb)
               MatrixPencils.tgexc!(true, true, nb, 1, view(At,i2,i2), view(Et,i2,i2), Q3, Z3) 
               At[i1,i2] = view(At,i1,i2)*Z3; Et[i1,i2] = view(Et,i1,i2)*Z3;
               Bt[i2,:] = Q3'*view(Bt,i2,:); 
               CN[:,i2] = view(CN,:,i2)*Z3; CM[:,i2] = view(CM,:,i2)*Z3;
            end
            nb -= 1; nbi -= 1;
         else
            if disc
               abs(abs(evb[1])-1) < offset && error("Eigenvalue(s) on the unit circle present")
            else
               abs(real(evb[1])) < offset && error("Eigenvalue(s) on the imaginary axis present")
            end
            if disc
               # solve a2*y*a2'-e2*y*e2'-b2*b2' = 0 for y=S*S'
               S = plyapas2!(a2,e2,copy(b2),disc = true) 
               y = S*S'  
               f2 = -(S'\(S\(a2\b2)))' # f2 = -b2.'/(y*a2.');
               # update At <- At + Bt*[0 f2]
               mul!(view(At,kk,kk),view(Bt,kk,:),f2,ONE,ONE) 
               x = transpose((e2*S)\b2); 
               W = inv(UpperTriangular(qrupdate!(Matrix{T1}(I,m,m),x)))
            else
               # solve a2*y*e2'+e2*y*a2'-b2*b2' = 0 for y=S*S'
               S = plyapas2!(a2, e2, copy(b2), disc = false) 
               y = S*S'  
               f2 = -(S'\(S\(e2\b2)))'; # f2 = -b2.'/(y*e2.');
               mul!(view(At,kk,kk),view(Bt,kk,:),f2,ONE,ONE) 
            end
            norm(f2,Inf) > fnrmtol && (fwarn += 1)
            # update the rest of (sub)matrices
            i1 = 1:n-k
            mul!(view(At,i1,kk),view(Bt,i1,:),f2,ONE,ONE)
            mul!(view(CN,:,kk),DN,f2,ONE,ONE)
            mul!(view(CM,:,kk),DM,f2,ONE,ONE)
            # update DN, DM and Bt: exploit W upper triangular
            disc && (rmul!(DN,W); rmul!(DM,W); rmul!(view(Bt,1:n,:),W)) 
            i2 = n-nb+1:n; i1 = 1:n-nb;
            if standsys
               if k == 2 
                  # standardization step is necessary to use trsen 
                  k1 = kk[1]; k2 = kk[2]
                  RT1R, RT1I, RT2R, RT2I, CS, SN = lanv2(At[k1,k1], At[k1,k2], At[k2,k1], At[k2,k2]) 
                  Z2 = LinearAlgebra.Givens(1, 2, CS, -SN)
                  rmul!(view(At,1:n,kk),Z2)
                  lmul!(Z2',view(At,kk,kk)); 
                  At[k1,k1] = RT1R; At[k2,k2] = RT2R;
                  rmul!(view(CN,:,kk),Z2)
                  rmul!(view(CM,:,kk),Z2)
                  lmul!(Z2',view(Bt,kk,:))
               end
               if nb > k
                  Z3 = view(Z,i2,i2)
                  Z3 = Matrix{T1}(I,nb,nb)
                  LAPACK.trexc!('V', nb-k+1, 1, view(At,i2,i2), Z3) 
                  At[i1,i2] = view(At,i1,i2)*Z3; Bt[i2,:] = Z3'*view(Bt,i2,:); 
                  CN[:,i2] = view(CN,:,i2)*Z3; CM[:,i2] = view(CM,:,i2)*Z3;
               end
            else 
               if nb > k
                  Q3 = view(Q,i2,i2); Z3 = view(Z,i2,i2)
                  Q3 = Matrix{T1}(I,nb,nb); Z3 = Matrix{T1}(I,nb,nb)
                  MatrixPencils.tgexc!(true, true, nb-k+1, 1, view(At,i2,i2), view(Et,i2,i2), Q3, Z3) 
                  At[i1,i2] = view(At,i1,i2)*Z3; Et[i1,i2] = view(Et,i1,i2)*Z3;
                  Bt[i2,:] = Q3'*view(Bt,i2,:); 
                  CN[:,i2] = view(CN,:,i2)*Z3; CM[:,i2] = view(CM,:,i2)*Z3;
               end
            end
            nb -= k
         end
      end
   end

   if mininf && nsinf > 0
      # remove the simple infinite eigenvalues
      i1 = 1:nsinf; i2 = nsinf+1:n;
      ca = view(CN,:,i1)/view(At,i1,i1); ee = view(Et,i1,i2)/view(Et,i2,i2);
      DN = DN-ca*Bt[i1,:]+ca*ee*Bt[i2,:]; 
      CN = CN[:,i2]-ca*At[i1,i2]+ca*ee*At[i2,i2]; CM = CM[:,i2];
      At = At[i2,i2]; Et = Et[i2,i2]; Bt = Bt[i2,:];
      n = n-nsinf; 
   end
  
   if standsys
      i1 = 1:n
      sysn = dss(At[i1,i1],I,Bt[i1,:],CN[:,i1],DN,Ts=sys.Ts);
      mindeg && (i1 = ng+1:n)
      sysm = dss(At[i1,i1],I,Bt[i1,:],CM[:,i1],DM,Ts=sys.Ts);
   else
      i1 = 1:n
      #sysn = dss(view(At,i1,i1),view(Et,i1,i1),view(Bt,i1,:),view(CN,:,i1),DN,Ts=sys.Ts);
      sysn = dss(At[i1,i1],Et[i1,i1],Bt[i1,:],CN[:,i1],DN,Ts=sys.Ts);
      mindeg && (i1 = ng+1:n)
      #sysm = dss(view(At,i1,i1),view(Et,i1,i1),view(Bt,i1,:),view(CM,:,i1),DM,Ts=sys.Ts);
      sysm = dss(At[i1,i1],Et[i1,i1],Bt[i1,:],CM[:,i1],DM,Ts=sys.Ts);
   end
    
   fwarn > 0 && @warn("Possible loss of numerical reliability due to high feedback gain")
   return sysn, sysm

# end GRCFID
end
function plyapas2!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, B::AbstractMatrix{T1}; disc = false) where T1 <: BlasFloat
   n, m = size(B)
   TR = real(T1)
   ONE = one(TR)
   ZERO = zero(TR)
   TWO = 2*ONE
   small = MatrixEquations.safemin(TR)*n*n
   BIGNUM = ONE / small
   SMIN = eps(max(maximum(abs.(A)),maximum(abs.(E))))

   tau = similar(B,min(n,m))
   B, tau = LinearAlgebra.LAPACK.gerqf!(B,tau)
   if m < n
      U = UpperTriangular([zeros(T1,n,n-m) B])
   else
      U = UpperTriangular(B[1:n,m-n+1:m])
   end

   if disc
      if n == 1
         abs(E[1,1]) >= abs(A[1,1]) && error("A-λE must have only eigenvalues with moduli greater than one")
         TEMP = sqrt( real((A[1,1] - E[1,1])'*(A[1,1] + E[1,1])) )
         TEMP < SMIN && (TEMP = SMIN)
         DR = abs( U[1,1] )
         TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP &&
            error("Singular generalized discrete Lyapunov equation")
         iszero(DR) || (TEMP = sign(U[1,1])*TEMP)
         U[1,1] = U[1,1]/TEMP
      else
         MatrixEquations.pglyap2!(E, A, U, adj = false, disc = true)
      end
   else
      if n == 1
         δ = TWO*real(E[1,1]'*A[1,1])
         δ <= ZERO && error("A-λE has stable eigenvalues")
         TEMP = sqrt( δ )
         TEMP < SMIN && (TEMP = SMIN)
         DR = abs( U[1,1] )
         TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular generalized Lyapunov equation")
         iszero(DR) || (TEMP = sign(U[1,1])*TEMP)
         U[1,1] = U[1,1]/TEMP
      else
         MatrixEquations.pglyap2!(-A, E, U, adj = false, disc = false)
      end
   end
   return U
end
