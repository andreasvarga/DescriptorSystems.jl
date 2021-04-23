"""
    grasol(sysg, sysf[, γ]; L2sol = false, nehari = false, reltol = 0.0001, mindeg = false, poles, sdeg, 
           fast = true, offset = β, atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (sysx, info)

Determine for the descriptor systems `sysg = (Ag-λEg,Bg,Cg,Dg)` and `sysf = (Af-λEf,Bf,Cf,Df)` 
with the transfer function matrices `G(λ)` and `F(λ)`, 
respectively, the descriptor system `sysx` with the transfer function matrix `X(λ)` 
such that `X(λ)` is the approximate solution of the linear rational equation `G(λ)X(λ) = F(λ)`,
which achieves the minimum error norm ``{\\small \\text{mindist} := \\min \\|G(λ)X(λ) - F(λ)\\|}``. 
The resulting `X(λ)` has all poles stable or lying on the boundary of the stability domain `Cs`. 
If `L2sol = false` (default) then the `L∞`-norm optimal solution is computed, while if `L2sol = true` the
`L2`-norm optimal solution is computed. 
`sysg` and `sysf` must not have poles on the boundary of the stability domain `Cs`.

If  `γ > 0` is a desired sub-optimality degree, then the `γ`-suboptimal model-matching problem
 
```math
     \\text{mindist} := \\|G(λ)X(λ) - F(λ) \\| < γ
```
is solved and `mindist` is the achieved suboptimal distance.

The call with

    grasol(sysgf[, mf[, γ]]; L2sol = false, nehari = false, reltol = 0.0001, mindeg = false, poles, sdeg, 
           fast = true, offset = β, atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (sysx, info)

uses the compound descriptor system `sysgf = (A-λE,[Bg Bf],C,[Dg Df])`, 
where `Bf` and `Df` have `mf` columns, to define  
the descriptor systems `sysg = (A-λE,Bg,C,Dg)` and `sysf = (A-λE,Bf,C,Df)`
(i.e., `Ag-λEg = Af-λEf = A-λE` and `Cg = Cf = C`). 
`sysgf` must not have poles on the boundary of the stability domain `Cs`.

If `nehari = true`, the optimal or suboptimal Nehari approximation is used to 
compute a `L∞`-suboptimal solution of the underlying _least-distance problem_ (`LDP`).
If `nehari = false` (default), the `L∞`-optimal solution is computed using the `γ`-iteration 
in the underlying `LDP` [2]. 

If `mindeg = true`, a minimum order solution is determined (if possible), 
while if `mindeg = false` (default) a particular solution of non-minimal order is determined. 

The resulting named tuple `info` contains additional information:
`info.nrank` is the normal rank of `G(λ)`, 
`info.nr` is the number of freely assignable poles of the solution `X(λ)`,  
`info.mindist` is the achieved approximation error norm and 
`info.nonstandard` is `true` for a non-standard problem, with `G(λ)` 
having zeros on the boundary of the stability domain, and `false` for
a standard problem, when `G(λ)` has no zeros on the boundary 
of the stability domain.  

The keyword argument `reltol` specifies the relative tolerance for the desired accuracy of 
the `γ`-iteration employed to solve the underlying least-distance problem.  
The iterations are performed until the current estimations of maximum ``γ_u`` and minimum ``γ_l`` of  
the optimal distance satisfies 
``{\\small γ_u-γ_l \\leq \\text{reltol} * \\text{gap}}``, where `gap` is the initial estimation of the error gap.

To assess the presence of poles on the boundary of the stability domain `Cs`, a boundary offset  `β` 
can be specified via the keyword parameter `offset = β`. 
Accordingly, for a continuous-time setting, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time setting, the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The vector `poles` specified as a keyword argument, can be used to specify the desired poles
of `sysx` alternatively to or jointly with enforcing a desired stability degree `sdeg` of poles. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `Ag`, `Af`, `A`, `Bg`, `Bf`, `Cg`, `Cf`, `Dg`, `Df`,  
the absolute tolerance for the nonzero elements of `Eg`, `Ef`, and the relative tolerance 
for the nonzero elements of all above matrices. The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sysg`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol` and `atol2 = atol`. 

The rank decisions in the underlying pencil manipulation algorithms are 
based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_  An extension of the approach of [1] to descriptor systems is used.

_References:_

[1]  B. A. Francis. A Course in H-infinity Theory, 
       Vol. 88 of Lecture Notes in Control and Information Sciences, 
       Springer-Verlag, New York, 1987.

[2] C.-C. Chu, J. C. Doyle, and E. B. Lee.
    The general distance problem in H∞  optimal control theory,
    Int. J. Control, vol 44, pp. 565-596, 1986.

"""
function grasol(sysgf::DescriptorStateSpace{T}, mf::Int, γ::Union{Real,Missing} = missing;  
                L2sol::Bool = false, nehari::Bool = false, offset::Real = sqrt(eps(float(real(T)))), 
                poles::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing,  
                atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                rtol::Real = (sysgf.nx*eps(real(float(one(T)))))*iszero(max(atol1,atol2)), 
                reltol::Real = 0.0001, fast::Bool = true, mindeg::Bool = false) where T
   p, mgf = size(sysgf);
   (mf <= mgf && mf >= 0) || throw(DimensionMismatch("mf must be at most $mgf, got $mf"))
   m = mgf-mf
   return grasol(sysgf[:,1:m], sysgf[:,m+1:mgf], γ; L2sol = L2sol, nehari = nehari, offset = offset, 
                 reltol = reltol, poles = poles, sdeg = sdeg, fast = fast, 
                 mindeg = mindeg, atol1 = atol1, atol2 = atol2, rtol = rtol)
end
function grasol(sysg::DescriptorStateSpace{T1}, sysf::DescriptorStateSpace{T2}, 
                γ::Union{Real,Missing} = missing; L2sol::Bool = false, nehari::Bool = false, 
                offset::Real = sqrt(eps(float(real(T1)))), reltol::Real = 0.0001, 
                poles::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, 
                atol::Real = zero(real(T1)), atol1::Real = atol, atol2::Real = atol, 
                rtol::Real = (sysg.nx*eps(real(float(one(T1)))))*iszero(max(atol1,atol2)), 
                fast::Bool = true, mindeg::Bool = false) where {T1,T2} 
   T = promote_type(T1,T2)
   T <: BlasFloat || ( T = promote_type(Float64,T) )

   p, m = size(sysg)
   pf, mf = size(sysf)
   p == pf|| throw(DimensionMismatch("sysg and sysf must have the same number of outputs"))
   Ts = promote_Ts(sysg.Ts,sysf.Ts)
   disc = !iszero(Ts) 

   #compute the extended inner-quasi-outer factorization
   Gi, Go, info1 = giofac(sysg, atol1 = atol1, atol2 = atol2, atol3 = atol1, rtol = rtol, 
                          fast = fast, minphase = true, offset = offset)
   ro = info1.nrank 
   
   #detect nonstandard problem 
   nonstandard = (info1.nfuz + info1.niuz > 0)
   
   #define and solve the LDP
   # ensure stabilizability of F 
   F = gir(Gi'*sysf, atol1 = atol1, atol2 = atol2, rtol = rtol)
   if L2sol
      #set tolerance for feedthrough matrix
      atol1 > 0 ? told = atol1 : told = 1.e4*eps()
      disc || norm(F.D[ro+1:end,:],Inf) > told || (F.D[ro+1:end,:] = zeros(T1,p-ro,mf))
      #solve the H2-LDP min||[ F1-Xt; F2 ] ||_2
      Xt, Xtu = gsdec(F[1:ro,:], job = "stable")
      gopt = gl2norm(grcfid(gir([Xtu; F[ro+1:end,:] ], fast = fast, atol1 = atol1, atol2 = atol2))[1], 
                     atolinf = told) 
   else
      #solve the H_inf-LDP min ||[ F1-Xt; F2 ] ||_inf
      Yt, gopt = glinfldp(gdual(F), p-ro, γ; nehari = nehari, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, offset = offset, reltol = reltol);  
      Xt = gdual(Yt)
   end
   
   if ro == m
      nr = 0
      sysx = gir(Go\Xt, noseig = true, atol1 = atol1, atol2 = atol2)
   else
      sysx, info1, = grsol(Go, Xt, mindeg = mindeg, poles = poles, sdeg = sdeg, 
                           atol1 = atol1, atol2 = atol2, rtol = rtol) 
      if min(sysx.ny,sysx.nu) > 0
         ev = gpole(sysx, atol1 = atol1, atol2 = atol2);
         disc || (ev = ev[isfinite.(ev)])
         if !isempty(ev) && !nonstandard
            if (!disc && maximum(real.(ev)) > -offset) || (disc && maximum(abs.(ev)) > 1-offset)
               ismissing(poles) && ismissing(sdeg) && (sdeg = disc ? 0.95 : -0.05)
               sysx = grsol(Go, Xt, mindeg = false, poles = poles, sdeg = sdeg, 
                            atol1 = atol1, atol2 = atol2, rtol = rtol)[1] 
            end
         end
      end
      nr = info1.nr
   end
   
   info = (nrank = ro, nr = nr, mindist = gopt, nonstandard = nonstandard)
   return sysx, info
   
   #end GRASOL   
end

"""
    glasol(sysg, sysf[, γ]; L2sol = false, nehari = false, reltol = 0.0001, mindeg = false, poles, sdeg, 
           fast = true, offset = β, atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (sysx, info)

Determine for the descriptor systems `sysg = (Ag-λEg,Bg,Cg,Dg)` and 
`sysf = (Af-λEf,Bf,Cf,Df)` with the transfer function matrices `G(λ)` and `F(λ)`, 
respectively, the descriptor system `sysx` with the transfer function matrix `X(λ)` 
such that `X(λ)` is the approximate solution of the linear rational equation `X(λ)G(λ) = F(λ)`,
which achieves the minimum error norm ``{\\small \\text{mindist} := \\min \\|X(λ)G(λ) - F(λ)\\|}``. 
The resulting `X(λ)` has all poles stable or lying on the boundary of the stability domain `Cs`. 
If `L2sol = false` (default) then the `L∞`-norm optimal solution is computed, while if `L2sol = true` the
`L2`-norm optimal solution is computed. 
`sysg` and `sysf` must not have poles on the boundary of the stability domain `Cs`.

If  `γ > 0` is a desired sub-optimality degree, then the `γ`-suboptimal model-matching problem

```math
     \\text{mindist} := \\|X(λ)G(λ) - F(λ) \\| < γ
```

is solved and `mindist` is the achieved suboptimal distance.

The call with

    glasol(sysgf[, pf[, γ]]; L2sol = false, nehari = false, reltol = 0.0001, mindeg = false, poles, sdeg, 
           fast = true, offset = β, atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (sysx, info)

uses the compound descriptor system `sysgf = (A-λE,B,[Cg; Cf],[Dg; Df])`, 
where `Cf` and `Df` have `pf` rows, to define  
the descriptor systems `sysg = (A-λE,B,Cg,Dg)` and `sysf = (A-λE,B,Cf,Df)`
(i.e., `Ag-λEg = Af-λEf = A-λE` and `Bg = Bf = B`). 
`sysgf` must not have poles on the boundary of the stability domain `Cs`.

If `nehari = true`, the optimal or suboptimal Nehari approximation is used to solve the
 underlying _least-distance problem_ (`LDP`).
If `nehari = false` (default), the optimal solution is computed using the `γ`-iteration 
in the underlying `LDP` [2]. 

If `mindeg = true`, a minimum order solution is determined (if possible), 
while if `mindeg = false` (default) a particular solution of non-minimal order is determined. 

The resulting named tuple `info` contains additional information:
`info.nrank` is the normal rank of `G(λ)`, 
`info.nl` is the number of freely assignable poles of the solution `X(λ)`,  
`info.mindist` is the achieved approximation error norm and 
`info.nonstandard` is `true` for a non-standard problem, with `G(λ)` 
having zeros on the boundary of the stability domain, and `false` for
a standard problem, when `G(λ)` has no zeros on the boundary 
of the stability domain.  

The keyword argument `reltol` specifies the relative tolerance for the desired accuracy of 
the `γ`-iteration employed to solve the underlying least-distance problem.  
The iterations are performed until the current estimations of maximum ``γ_u`` and minimum ``γ_l`` of  
the optimal distance satisfies 
``{\\small γ_u-γ_l < \\text{reltol}* \\text{gap}}``, where `gap` is the initial estimation of the error gap.

To assess the presence of poles on the boundary of the stability domain `Cs`, a boundary offset  `β` 
can be specified via the keyword parameter `offset = β`. 
Accordingly, for a continuous-time setting, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time setting, the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The vector `poles` specified as a keyword argument, can be used to specify the desired poles
of `sysx` alternatively to or jointly with enforcing a desired stability degree `sdeg` of poles. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `Ag`, `Af`, `A`, `Bg`, `Bf`, `Cg`, `Cf`, `Dg`, `Df`,  
the absolute tolerance for the nonzero elements of `Eg`, `Ef`, and the relative tolerance 
for the nonzero elements of all above matrices. The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sysg`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol` and `atol2 = atol`. 

The rank decisions in the underlying pencil manipulation algorithms are 
based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_  An extension of the approach of [1] to descriptor systems is used.

_References:_

[1]  B. A. Francis. A Course in H-infinity Theory, 
       Vol. 88 of Lecture Notes in Control and Information Sciences, 
       Springer-Verlag, New York, 1987.

[2] C.-C. Chu, J. C. Doyle, and E. B. Lee.
    The general distance problem in H∞  optimal control theory,
    Int. J. Control, vol 44, pp. 565-596, 1986.

"""
function glasol(sysgf::DescriptorStateSpace{T}, pf::Int, γ::Union{Real,Missing} = missing;  
                L2sol::Bool = false, nehari::Bool = false, offset::Real = sqrt(eps(float(real(T)))), 
                poles::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing,  
                atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                rtol::Real = (sysgf.nx*eps(real(float(one(T)))))*iszero(max(atol1,atol2)), 
                reltol::Real = 0.0001, fast::Bool = true, mindeg::Bool = false) where T
   pgf, m = size(sysgf);
   (pf <= pgf && pf >= 0) || throw(DimensionMismatch("pf must be at most $pgf, got $pf"))
   p = pgf-pf
   return glasol(sysgf[1:p,:], sysgf[p+1:pgf,:], γ; L2sol = L2sol, nehari = nehari, offset = offset, 
                 reltol = reltol, poles = poles, sdeg = sdeg, fast = fast, 
                 mindeg = mindeg, atol1 = atol1, atol2 = atol2, rtol = rtol)
end
function glasol(sysg::DescriptorStateSpace{T1}, sysf::DescriptorStateSpace{T2}, 
                γ::Union{Real,Missing} = missing; L2sol::Bool = false, nehari::Bool = false, 
                offset::Real = sqrt(eps(float(real(T1)))), reltol::Real = 0.0001, 
                poles::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, 
                atol::Real = zero(real(T1)), atol1::Real = atol, atol2::Real = atol, 
                rtol::Real = (sysg.nx*eps(real(float(one(T1)))))*iszero(max(atol1,atol2)), 
                fast::Bool = true, mindeg::Bool = false) where {T1,T2} 
   T = promote_type(T1,T2)
   T <: BlasFloat || ( T = promote_type(Float64,T) )

   p, m = size(sysg)
   pf, mf = size(sysf)
   m == mf|| throw(DimensionMismatch("sysg and sysf must have the same number of inputs"))
   # isstable(sysf, atol1 = atol1, atol2 = atol2, rtol = rtol, offset = offset) ||
   #          error("sysf must be a stable system")      
   Ts = promote_Ts(sysg.Ts,sysf.Ts)
   disc = !iszero(Ts) 

   #compute the extended quasi-co-outer-co-inner factorization
   Gi, Go, info1 = goifac(sysg, atol1 = atol, atol2 = atol, atol3 = atol, rtol = rtol, 
                          fast = fast, minphase = true, offset = offset)
   ro = info1.nrank 
   
   #detect nonstandard problem 
   #nonstandard = order(gcrange(Go,struct('tol',tol,'zeros','s-unstable'))) > 0;
   nonstandard = (info1.nfuz + info1.niuz > 0)
   
   #define and solve the LDP
   # ensure detectability of F 
   F = gir(sysf*Gi', atol1 = atol1, atol2 = atol2, rtol = rtol)
   if L2sol
      #set tolerance for feedthrough matrix
      atol1 > 0 ? told = atol1 : told = 1.e4*eps()
      disc || norm(F.D[:,ro+1:end],Inf) > told || (F.D[:,ro+1:end] = zeros(T1,pf,m-ro))
      #solve the H2-LDP min||[ F1-Xt F2 ] ||_2
      Xt, Xtu = gsdec(F[:,1:ro],job = "stable")
      Y = glcfid(gir([Xtu F[:,ro+1:end] ], atol1 = atol1, atol2 = atol2))[1]
      gopt = gl2norm(glcfid(gir([Xtu F[:,ro+1:end] ], fast = fast, atol1 = atol1, atol2 = atol2))[1], 
                     atolinf = told) 
   else
      #solve the H_inf-LDP min ||[ F1-Xt F2 ] ||_inf
      Xt, gopt = glinfldp(F, m-ro, γ; nehari = nehari, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, offset = offset, reltol = reltol);  
   end
   
   if ro == p
      nl = 0
      sysx = gir(Xt/Go, noseig = true, atol1 = atol1, atol2 = atol2)
   else
      sysx, info1, = glsol(Go, Xt, mindeg = mindeg, poles = poles, sdeg = sdeg, 
                    atol1 = atol1, atol2 = atol2, rtol = rtol) 
      if min(sysx.ny,sysx.nu) > 0
         ev = gpole(sysx, atol1 = atol1, atol2 = atol2);
         disc || (ev = ev[isfinite.(ev)])
         if !isempty(ev) && !nonstandard
            if (!disc && maximum(real.(ev)) > -offset) || (disc && maximum(abs.(ev)) > 1-offset)
               ismissing(poles) && ismissing(sdeg) && (sdeg = disc ? 0.95 : -0.05)
               sysx = glsol(Go, Xt, mindeg = false, poles = poles, sdeg = sdeg, 
                            atol1 = atol1, atol2 = atol2, rtol = rtol)[1] 
            end
         end
      end
      nl = info1.nl
   end
   
   info = (nrank = ro, nl = nl, mindist = gopt, nonstandard = nonstandard)
   return sysx, info
   
   #end GLASOL   
end
"""
    glinfldp(sys1, sys2, [, γ]; nehari = false, reltol = 0.0001, fast = true, offset = β, 
             atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (sysx, mindist)

Determine for the descriptor systems `sys1 = (A1-λE1,B1,C1,D1)` and 
`sys2 = (A2-λE2,B2,C2,D2)` with the transfer function matrices ``G_1(λ)`` and ``G_2(λ)``, 
respectively, the descriptor system `sysx` with the transfer function matrix ``X(λ)`` 
such that ``X(λ)`` is the a stable solution of the 2-block `L∞` _least distance problem_ (`LDP`)

```math     
      \\text{mindist} := \\min \\|G_1(λ)-X(λ) \\mid   G_2(λ) \\|_\\infty 
```    

`mindist` is the achieved minimum distance corresponding to the optimal solution. 
If `sys2 = []`, an 1-block `LDP` is solved. 
`sys1` and `sys2` must not have poles on the boundary of the stability domain `Cs`.

If  ``{\\small γ > \\|G_2(λ)\\|_\\infty}`` is a desired sub-optimality degree, then the 
`γ`-suboptimal `LDP` 
 
```math     
     \\text{mindist} := \\|G_1(λ)-X(λ) \\mid G_2(λ) \\|_\\infty < γ
```

is solved and `mindist` is the achieved suboptimal distance.

The call with

    glinfldp(sys[, m2[, γ]]; nehari = false, fast = true, offset = β, 
             atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (sysx, mindist)

uses the compound descriptor system `sys = (A-λE,[B1 B2],C,[D1 D2])`, 
where `B2` has `m2` columns, to define  
the descriptor systems `sys1 = (A-λE,B1,C,D1)` and `sys2 = (A-λE,B2,C,D2)`
(i.e., `A1-λE1 = A2-λE2 = A-λE` and `C1 = C2 = C`). 
If `m2 = 0`, an 1-block `LDP` is solved. 
`sys` must not have poles on the boundary of the stability domain `Cs`.

If `nehari = true`, the optimal or suboptimal Nehari approximation is used to solve the `LDP`.
If `nehari = false` (default), the optimal solution is computed using the `γ`-iteration [1]. 

The keyword argument `reltol` specifies the relative tolerance for the desired accuracy of `γ`-iteration. 
The iterations are performed until the current estimations of maximum ``γ_u`` and minimum ``γ_l`` of  
the optimal distance ``γ_o``, ``γ_l \\leq γ_o \\leq γ_u``, satisfies 

```math
     γ_u-γ_l \\leq \\text{reltol} * \\text{gap} ,
```
where `gap` is the original gap (internally determined).
To assess the presence of poles on the boundary of the stability domain `Cs`, a boundary offset  `β` 
can be specified via the keyword parameter `offset = β`. 
Accordingly, for a continuous-time setting, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discete-time setting, the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol` and `atol2 = atol`. 

The rank decisions in the underlying pencil manipulation algorithms are 
based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_ The approach of [1] is used for the solution of the 2-block least distance problem.

_References:_
[1] C.-C. Chu, J. C. Doyle, and E. B. Lee
    The general distance problem in H∞  optimal control theory,
    Int. J. Control, vol 44, pp. 565-596, 1986.
"""
function glinfldp(sys1::DescriptorStateSpace{T1}, sys2::DescriptorStateSpace{T2}, 
                  γ::Union{Real,Missing} = missing; nehari::Bool = false, offset::Real = sqrt(eps(float(real(T1)))),  
                  atol::Real = zero(real(T1)), atol1::Real = atol, atol2::Real = atol, 
                  rtol::Real = (sys1.nx*eps(real(float(one(T1)))))*iszero(max(atol1,atol2)), 
                  reltol::Real = 0.0001, fast::Bool = true) where {T1,T2}
   size(sys1,1) == size(sys2,1) || throw(DimensionMismatch("sys1 and sys2 must have the same number of outputs"))
   return glinfldp(gir([sys1 sys2]; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol), size(sys2,2),γ; 
                   nehari = nehari, offset = offset, reltol = reltol, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)
end
function glinfldp(sys::DescriptorStateSpace{T}, m2::Int, γ::Union{Real,Missing} = missing;  nehari::Bool = false,
                  offset::Real = sqrt(eps(float(real(T)))), reltol::Real = 0.0001, 
                  atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                  rtol::Real = (sys.nx*eps(real(float(one(T)))))*iszero(max(atol1,atol2)), 
                  fast::Bool = true) where T 
 
   T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
   p, m = size(sys);
   (m2 <= m && m2 >= 0) || throw(DimensionMismatch("m2 must be at most $m, got $m2"))
   m1 = m-m2
   tol = sqrt(eps(real(float(one(T1)))))

   if nehari
      # compute a suboptimal solution as a Nehari approximation of G1
      sysx,  = gnehari(sys[:,1:m1], γ; offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)
      mindist = glinfnorm(sys-sysx*eye(m1,m), offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol)[1]
      return sysx, mindist
   end

   subopt = !ismissing(γ)
     
   # address constant case
   if order(sys) == 0 || m1 == 0
      # solve a 2-block Parrott problem
      return sys[:,1:m1], opnorm(sys.D[:,m1+1:end])
   end
   
   gl1 = m2 > 0 ? (glinfnorm(sys[:,m1+1:end], offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol)[1]) : 0 # prevent failure of opnorm 
   subopt && gl1 > γ && error("γ must be chosen greater than $gl1")
   
   hlu = ghanorm(gsdec(sys[:,1:m1], job = "unstable", atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)[1]')[1];

   if m2 == 0 || hlu <= atol || gl1 <= tol*hlu
      # solve one block LDP
      if subopt 
         # solve sub-optimal Nehari problem
         sysx,  = gnehari(sys[:,1:m1], γ; offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)
         mindist = glinfnorm(sys[:,1:m1]-sysx, offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol)[1]
      else
        # solve optimal Nehari problem
        sysx, mindist = gnehari(sys[:,1:m1]; offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)
      end
   else
      # solve 2-block LDP
      if subopt 
         # solve sub-optimal LDP
         gam = max(γ,gl1*1.01);
         V = glsfg(sys[:,m1+1:end], gam, stabilize = true, offset = offset, fast = fast, 
                   atol1 = atol1, atol2 = atol2, rtol = rtol)
         syse = V\sys[:,1:m1]
         hlu = ghanorm(gsdec(syse, job = "unstable", fast = fast, 
                             atol1 = atol1, atol2 = atol2, rtol = rtol)[1]')[1];
         # solve the sub-optimal Nehari problem
         if hlu < 1
            sysx =  gbalmr(V*gnehari(syse, hlu, offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)[1],
                           offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)[1];
         else
            subopt = false;
         end
         mindist = glinfnorm(sys-sysx*eye(m1,m), offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol)[1]
      end
      if !subopt
         # solve optimal LDP using gamma-iteration
         # initialize the gamma-iteration
         # compute upper bound and inital gap
         # hlu = ghanorm(gsdec(sys[:,1:m1], job = "unstable", atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)[1]')[1];
         gu = hypot(hlu,gl1); gl = max(gl1,hlu)
         gap = max(1,gu-gl);
         gam = (gl+gu)/2; 

         # stabilize G2 only once
         G2s = grcfid(sys[:,m1+1:end], mininf = true, fast = fast, offset = offset, 
                      atol1 = atol1, atol2 = atol2, atol3 = atol1, rtol = rtol)[1] 
         # perform the gamma-iteration
         iter = 0; iterl = 0; g0 = gam; hlu = 1;
         while gu-gl > reltol*gap && iter <= 50 
            iter += 1 
            V = glsfg(G2s, gam, stabilize = false, fast = fast, offset = offset, 
                      atol1 = atol1, atol2 = atol1, rtol = rtol)

            # compute inv(V)*G1
            syse = V\sys[:,1:m1]

            hlu = ghanorm(gsdec(syse, job = "unstable", fast = fast, 
                                atol1 = atol1, atol2 = atol2, rtol = rtol)[1]')[1];
            #fail && break
            if hlu < 1
               gu = gam;           # gamma > gamma_opt
            else
               gl = gam;           # gamma <= gamma_opt
               iterl += 1
            end   
            gam = (gl+gu)/2;
         end
        
         if iter > 0
            if hlu >= 1
              # recompute spectral factorization if gamma < gamma_opt
              V = glsfg(G2s, gu, stabilize = false) 
              # compute inv(V)*G1
              syse = V\sys[:,1:m1]
              gam = gu
            end
         else
            # compute spectral factorization for increased gamma
            gam = min(gu, gam*(1+reltol))
            V = glsfg(G2s, gam, stabilize = false, offset = offset, 
                     atol1 = atol1, atol2 = atol1, rtol = rtol)  
            # compute inv(V)*G1
            syse = V\sys[:,1:m1]
         end
         if iterl == 0 && gam - gl < reltol*gl1
            # handle the case gam = norm(G2)
            @warn "Optimal LDP solution has poles near to the boundary of stability domain"
            # set offset to machine precision 
            offset = eps(float(real(T1)))
         end

         # solve the Nehari problem and compute solution
         sysx =  gbalmr(V*gnehari(syse, offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)[1],
                        offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)[1];
         mindist = glinfnorm(sys-sysx*eye(m1,m), offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol)[1]
         #mindist = gam;
      end
   end
   return sysx, mindist
   # end of GLINFLDP
end
glinfldp(sys::DescriptorStateSpace{T}, γ::Union{Real,Missing} = missing;  kwargs...) where T = 
         glinfldp(sys, 0, γ; kwargs...)
"""
    gnehari(sys[, γ]; fast = true, offset = β, 
                    atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (sysx, σ1)

Compute for the descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`
the optimal or suboptimal stable Nehari approximation `sysx = (Ax-λEx,Bx,Cx,Dx)` 
with the transfer function matrix `X(λ)`. The optimal Nehari approximation  `X(λ)` satisfies
   
```math     
     \\| G(\\lambda) - X(\\lambda) \\|_\\infty = \\| G^{*}_u(\\lambda) \\|_H := \\sigma_1,
```

where ``{\\small G_u(\\lambda)}`` is the antistable part of `G(λ)`.
The resulting ``σ_1`` is the Hankel-norm of ``{\\small G^{*}_u(\\lambda)}`` 
(also the L∞-norm of the optimal approximation error). 
For a given ``γ > σ_1``, the suboptimal approximation satisfies

```math     
     \\| G(\\lambda) - X(\\lambda) \\|_\\infty \\leq \\gamma .
```

The system `sys` must not have poles on the boundary of the stability domain `Cs`. 

To assess the presence of poles on the boundary of the stability domain `Cs`, a boundary offset  `β` 
can be specified via the keyword parameter `offset = β`. 
Accordingly, for a continuous-time system, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time system, the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol` and `atol2 = atol`. 

The separation of the finite and infinite eigenvalues is performed using 
rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_ The Hankel-norm approximation methods of [1] and [2], 
with extensions for descriptor systems, are used for the approximation
of the unstable part.   

_References:_

[1] K. Glover. All optimal Hankel-norm approximations of linear
       multivariable systems and their L∞ error bounds,
       Int. J. Control, vol. 39, pp. 1115-1193, 1984.

[2] M. G. Safonov, R. Y. Chiang, and D. J. N. Limebeer. 
       Optimal Hankel model reduction for nonminimal systems. 
       IEEE Trans. Automat. Control, vol. 35, pp. 496–502, 1990.
"""
function gnehari(sys::DescriptorStateSpace{T}, γ::Union{Real,Missing} = missing;  
                 offset::Real = sqrt(eps(float(real(T)))), atol::Real = zero(float(real(T))), 
                 atol1::Real = atol, atol2::Real = atol, atol3::Real = atol, 
                 rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(max(atol1,atol2,atol3)), 
                 fast::Bool = true) where T 
   sys.nx == 0 && (return sys, 0)
   subopt_flg = !ismissing(γ)

   ONE = one(T)
   disc = (sys.Ts != 0)

   # set stability margin to separate stable and unstable parts
   smarg = disc ? 1-offset : -offset

   # Stable/antistable separation; possible eigenvalues on the boundary of 
   # the stability domain are included in the unstable part 
   sysx, sysu = gsdec(sys; job = "stable", smarg = smarg, fast = fast,  atol1 = atol1, atol2 = atol2, rtol = rtol) 

   s2eps = sqrt(eps(real(float(one(T)))))
   # only proper systems are handled in the continuous-time case
   if !disc && sysu.E != I && (norm(sysu.E) < atol2 || rcond(sysu.E) < s2eps)
      sysu = gss2ss(sysu, atol2 = s2eps)[1]
      if sysu.E != I && rcond(sysu.E) < s2eps
          error("gnehari: improper continuous-time system sys")
      else
          sysx.D[:,:] = sysx.D + sysu.D
          sysu.D[:,:] = zeros(T, sys.ny, sys.nu)
      end
   end
   
   # exit if the system has only stable part
   order(sysu) == 0 && (return sysx, zero(float(real(T))))
   
   a, e, b, c, d = dssdata(sysu)
   e == I ? (isqtriu(a) ? ev = ordeigvals(a) : ev = eigvals(a)) : ev = ordeigvals(a,e)[1] 
   if disc
      any(abs.(ev) .<= 1+offset) &&
          error("gnehari: the system has possibly poles on the unit circle")
   else
      any(real.(ev) .<= offset) &&
          error("gnehari: the system has possibly poles on the imaginary axis")
   end
   
   if disc
      # use the bilinear transformation to compute an equivalent unstable, 
      # possibly non-minimal, continuous-time system  
      sqrt2 = sqrt(2)
      esave = copy(e)
      e += a; a -= esave 
      c = c/e 
      mul!(d,c,b,-ONE,ONE) 
      lmul!(sqrt2,b); c = (sqrt2*c)*esave;
   end
   # compute a balanced minimal realization of antistable part and the
   # Hankel singular values
   sysr, hs = gbalmr(dss(-a,e,-b,c,d), balance = true, atolhsv = s2eps, atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)
   a, e, b, c, d = dssdata(sysr)

   na = size(a,1) 
   p, m = size(d)

   s1 = hs[1];  # expected L-inf norm of the optimal approximation error 

   # Determine the type of required approximation
   if subopt_flg 
      if abs(γ-s1) <= s1*s2eps
         subopt_flg = false; # enforce optimal approximation
      elseif γ < s1
        error("Suboptimal antistable Hankel approximation cannot be computed")
      end
   else
      γ = s1
   end
    
   if subopt_flg
      sk = γ; r = 0 
   else
      sk = s1; ep = s1*s2eps;
      if na > 0
         r = 1;
         for i = 2:na
             abs(sk-hs[i]) > ep && break
             r += 1;
         end
      else
         r = 0;
      end
   end
   #          ^
   #  Compute F(s)
   ns = na-r; sk2 = sk*sk; i1 = r+1:na; 
   # make system square by padding with zeros rows or column 
   Tw = eltype(a)
   pm = max(p,m)
   if m < p 
       b = [b zeros(Tw,na,p-m)] 
   elseif m > p
       c = [c; zeros(Tw,m-p,na)] 
   end
   if subopt_flg
      u = zeros(Tw,pm,pm);
      b1 = b; c1 = c;
   else
      i2 = 1:r;
      SV1 = svd(view(b,i2,:),full=true); # b[i2,:] = U1*Σ1*V1'
      SV2 = svd(view(c,:,i2)*SV1.U,full=true)
      #sig = [diag(SV2.Vt) .< 0; falses(pm-r)]
      sig = [real.(diag(SV2.Vt)) .< 0; falses(pm-r)]
      SV2.U[:,sig] = -SV2.U[:,sig]
      u = -SV2.U*SV1.Vt
      b1 = b[i1,:]; c1 = c[:,i1]
   end
   jm = 1:m; jp = 1:p;
   if ns > 0
      # optimal solution
      hsd = Diagonal(view(hs,i1))
      e = Matrix(hsd*hsd - sk2*I)
      a = -(sk2*a[i1,i1]' + hsd*a[i1,i1]*hsd - sk*c1'*u*b1')
      b = -(hsd*b1[:,jm] + sk*c1'*u[:,jm])
      c = c1[jp,:]*hsd + sk*u[jp,:]*b1'
   else
      a = zeros(Tw,0,0); e = I; b = zeros(Tw,0,m); c = zeros(Tw,p,0); 
   end
   if na > 0
      d -= sk*u[jp,jm]
   end
   
   if disc 
      # compute the equivalent discrete-time solution  
      esave = copy(e)
      e -= a; a += esave 
      c = c/e 
      mul!(d,c,b,ONE,ONE) 
      lmul!(sqrt2,b); c = sqrt2*c*esave;
   end
   
   return sysx + dss(a, e, b, c, d, Ts = sys.Ts), s1
   
   # end GNEHARI
end

