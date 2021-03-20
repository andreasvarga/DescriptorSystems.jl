"""
    sysinv = inv(sys; atol = 0, atol1 = atol, atol2 = atol, rtol, checkinv = true)

Compute for a descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`, a descriptor realization of its inverse system
`sysinv = (Ai-λEi,Bi,Ci,Di)`, such that the transfer function matrix `Ginv(λ)` of `sysinv` is the inverse of `G(λ)` (i.e., `G(λ)*Ginv(λ) = I`). 
The realization of `sysinv` is determined using inversion-free formulas and the invertibility condition is checked, unless `checkinv = false`.

The keyword arguments `atol1`, `atol2` and `rtol` specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `n` is the order of the square matrices `A` and `E`, and  `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function inv(sys::DescriptorStateSpace{T}; checkinv::Bool = true, 
             atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
             rtol::Real = (size(sys.A,1)+1)*eps(real(float(one(T))))*iszero(min(atol1,atol2))) where T
   p, m = sys.ny, sys.nu
   n = sys.nx
   m == p  || error("The system must have the same number of inputs and outputs")
   Ai = [sys.A sys.B; sys.C sys.D]
   Bi = [zeros(T,n,m); -Matrix{T}(I,m,m)]
   Ci = [zeros(T,m,n) Matrix{T}(I,m,m)]
   Di = zeros(T,m,m)
   Ei = (sys.E == I) ? blockdiag(Matrix{T}(I,n,n),zeros(T,m,m)) : 
                       blockdiag(sys.E,zeros(T,m,m)) 
   checkinv && (MatrixPencils.isregular(Ai, Ei, atol1 = atol1, atol2 = atol2, rtol = rtol ) || 
                error("The system is not invertible"))
   return DescriptorStateSpace{T}(Ai, Ei, Bi, Ci, Di, sys.Ts) 
end
"""
    sysldiv = ldiv(sys1, sys2; atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ, checkinv = true)
    sysldiv = sys1 \\ sys2

Compute for the descriptor systems `sys1 = (A1-λE1,B1,C1,D1)` with the transfer function matrix `G1(λ)` and 
`sys2 = (A2-λE2,B2,C2,D2)` with the transfer function matrix `G2(λ)`, a descriptor realization 
`sysldiv = (Ai-λEi,Bi,Ci,Di)` of `sysldiv = inv(sys1)*sys2`,
whose transfer-function matrix `Gli(λ)` represents the result of the left division `Gli(λ) = inv(G1(λ))*G2(λ)`. 
The realization of `sysldiv` is determined using inversion-free formulas and the invertibility condition for `sys1` is checked, 
unless `checkinv = false`.

The keyword arguments `atol1`, `atol2` and `rtol` specify, respectively, 
the absolute tolerance for the nonzero elements of `A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, 
the absolute tolerance for the nonzero elements of `E1` and `E2`, 
and the relative tolerance for the nonzero elements of `A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, `E1` and `E2`.  
The default relative tolerance is `n*ϵ`, where `n` is the maximum of orders of the square matrices `A1` and `A2`, and  `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function ldiv(sys1::DescriptorStateSpace{T1}, sys2::DescriptorStateSpace{T2}; 
              atol::Real = zero(real(T1)), atol1::Real = atol, atol2::Real = atol, 
              rtol::Real = (max(size(sys1.A,1),size(sys2.A,1))*eps(real(float(one(T1)))))*iszero(min(atol1,atol2))) where {T1,T2}
   p1, m1 = sys1.ny, sys1.nu
   m1 == p1  || error("The system SYS1 must be square")
   p2, m2 = sys2.ny, sys2.nu
   p1 == p2  || error("The systems SYS1 and SYS2 must have the same number of outputs")
   sys1.Ts == sys2.Ts ||  error("The systems SYS1 and SYS2 must have same sampling time")
   T = promote_type(T1,T2)
   n1 = sys1.nx
   n2 = sys2.nx
   A, E, B, C, D = dssdata(T,[sys1 sys2])
   Ai = [A B[:,1:m1]; C D[:,1:m1]]
   Ei = E == I ? [I zeros(T,n1+n2,m1); zeros(T,m1,n1+n2+m1)]  : blockdiag(E,zeros(T,m1,m1))
   MatrixPencils.isregular(Ai, Ei, atol1 = atol1, atol2 = atol2, rtol = rtol) || 
              error("The system SYS1 is not invertible")
   Bi = [B[:,m1+1:m1+m2]; D[:,m1+1:m1+m2]]
   Ci = [zeros(T,p1,n1+n2) -I] 

   return DescriptorStateSpace{T}(Ai, Ei, Bi, Ci, zeros(T,p1,m2), sys1.Ts) 
end
function (\)(sys1::AbstractDescriptorStateSpace, sys2::AbstractDescriptorStateSpace; kwargs...)
   ldiv(sys1,sys2; kwargs...)
end
"""
    sysrdiv = rdiv(sys1, sys2; atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ, checkinv = true)  
    sysrdiv = sys1 / sys2  

Compute for the descriptor systems `sys1 = (A1-λE1,B1,C1,D1)` with the transfer function matrix `G1(λ)` and 
`sys2 = (A2-λE2,B2,C2,D2)` with the transfer function matrix `G2(λ)`, a descriptor realization 
`sysrdiv = (Ai-λEi,Bi,Ci,Di)` of `sysrdiv = sys1*inv(sys2)`,
whose transfer-function matrix `Gri(λ)` represents the result of the right division `Gri(λ) = G1(λ)*inv(G2(λ))`. 
The realization of `sysrdiv` is determined using inversion-free formulas and the invertibility condition for `sys2` is checked, 
unless `checkinv = false`.

The keyword arguments `atol1`, `atol2` and `rtol` specify, respectively, 
the absolute tolerance for the nonzero elements of `A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, 
the absolute tolerance for the nonzero elements of `E1` and `E2`, 
and the relative tolerance for the nonzero elements of `A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, `E1` and `E2`.  
The default relative tolerance is `n*ϵ`, where `n` is the maximum of orders of the square matrices `A1` and `A2`, 
and  `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function rdiv(sys1::DescriptorStateSpace{T1}, sys2::DescriptorStateSpace{T2}; 
              atol::Real = zero(real(T1)), atol1::Real = atol, atol2::Real = atol, 
              rtol::Real = max(size(sys1.A,1),size(sys2.A,1))*eps(real(float(one(T1))))*iszero(min(atol1,atol2))) where {T1,T2}
   p2, m2 = sys2.ny, sys2.nu
   m2 == p2  || error("The system SYS2 must be square")
   p1, m1 = sys1.ny, sys1.nu
   m1 == m2  || error("The systems SYS1 and SYS2 must have the same number of inputs")
   sys1.Ts == sys2.Ts ||  error("The systems SYS1 and SYS2 must have same sampling time")
   T = promote_type(T1,T2)
   n1 = sys1.nx
   n2 = sys2.nx
   A, E, B, C, D = dssdata(T,[sys2; sys1])
   Ai = [A B; C[1:p2,:] D[1:p2,:]]
   Ei = E == I ? [I zeros(T,n1+n2,p2); zeros(T,p2,n1+n2+p2)] : blockdiag(E,zeros(T,p2,p2))
   MatrixPencils.isregular(Ai, Ei, atol1 = atol1, atol2 = atol2, rtol = rtol) || 
              error("The system SYS2 is not invertible")
   Ci = [C[p2+1:p1+p2,:] D[p2+1:p1+p2,:]]
   Bi = [zeros(T,n1+n2,m1); -I] 

   return DescriptorStateSpace{T}(Ai, Ei, Bi, Ci, zeros(T,p1,m1), sys1.Ts) 
end
function (/)(sys1::AbstractDescriptorStateSpace, sys2::AbstractDescriptorStateSpace; kwargs...)
    rdiv(sys1,sys2; kwargs...)
end
"""
    sysdual = gdual(sys, rev = false) 
    sysdual = transpose(sys, rev = false) 

Compute for a descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`, 
the descriptor system realization of its dual system 
`sysdual = (Ad-λEd,Bd,Cd,Dd)`, where `Ad = transpose(A)`, `Ed = transpose(E)`, `Bd = transpose(C)`, 
`Cd = transpose(B)` and `Dd = transpose(D)`, 
such that the transfer function matrix `Gdual(λ)` of `sysdual` is the transpose of `G(λ)` 
(i.e., `Gdual(λ) = transpose(G(λ))`). 

If `rev = true`, the tranposition is combined with the reverse permutation of the state variables, such that
`sysdual = (P*Ad*P-λP*Ed*P,P*Bd,Cd*P,Dd)`, where `P` is the permutation matrix with ones down the second diagonal. 
"""
function gdual(SYS::DescriptorStateSpace{T};kwargs...) where T
    transpose(SYS;kwargs...)
end
function transpose(SYS::DescriptorStateSpace{T}; rev = false) where T
   if rev
       return DescriptorStateSpace{T}(reverse(reverse(transpose(SYS.A),dims=1),dims=2), 
                                      SYS.E == I ? I : reverse(reverse(transpose(SYS.E),dims=1),dims=2),
                                      reverse(transpose(SYS.C),dims=1), reverse(transpose(SYS.B),dims=2), copy(transpose(SYS.D)), SYS.Ts)
   else
       return DescriptorStateSpace{T}(copy(transpose(SYS.A)), SYS.E == I ? I : copy(transpose(SYS.E)), copy(transpose(SYS.C)), copy(transpose(SYS.B)),
                                      copy(transpose(SYS.D)), SYS.Ts)
   end
end

"""
    sysconj = ctranspose(sys) 
    sysconj = sys' 

Compute the conjugate transpose (or adjoint) of a descriptor system (see [`adjoint`](@ref)). 
"""
function ctranspose(sys::DescriptorStateSpace{T}) where T
   if sys.Ts == 0
      # continuous-time case
      return DescriptorStateSpace{T}(copy(-sys.A'), sys.E == I ? I : copy(sys.E'), copy(-sys.C'), copy(sys.B'), copy(sys.D'), sys.Ts) 
   else
      # discrete-time case
      (nx, ny, nu) = (sys.nx, sys.ny, sys.nu)
      n = nx+ny
      return DescriptorStateSpace{T}(sys.E == I ? Matrix{T}(I,n,n) : blockdiag(sys.E',Matrix{T}(I,ny,ny)), 
                                     [sys.A' sys.C'; zeros(T,ny,n)], 
                                     [zeros(T,nx,ny) ; -I], 
                                     [sys.B' zeros(T,nu,ny)], copy(sys.D'), sys.Ts) 
   end
end
"""
    sysconj = adjoint(sys) 
    sysconj = sys' 

Compute for a descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`, 
the descriptor system realization of its adjoint (also called _conjugate transpose_) system 
`sysconj = (Ac-λEc,Bc,Cc,Dc)`, such that the transfer function matrix `Gconj(λ)` of `sysconj` 
is the appropriate conjugate transpose of `G(λ)`, as follows: 
for a continuous-time system with `λ = s`, `Gconj(s) := transpose(G(-s))`, while 
for a discrete-time system with `λ = z`, `Gconj(z) := transpose(G(1/z))`.
"""
function adjoint(sys::AbstractDescriptorStateSpace)
    ctranspose(sys)
end
"""
    gbilin(sys, g, compact = true, minimal = false, ss = true, atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (syst, ginv)

Compute for the descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)` and 
a first degree real rational transfer function `g = g(δ)`, 
the descriptor system realization `syst = (At-δEt,Bt,Ct,Dt)` of `G(g(δ))` corresponding to the bilinear transformation 
`λ = g(δ) = (aδ+b)/(cδ+d)`. For a continuous-time transfer function `g(δ)`, `δ = s`, the complex variable in 
the Laplace transform, while for a discrete-time transfer function,  
`δ = z`, the complex variable in the `Z`-transform. `syst` inherits the sampling-time of `sys1`. 
`sysi1` is the transfer function `ginv(λ) = (d*λ-b)/(-c*λ+a)` representing the inverse of the bilinear transformation `g(δ)` 
(i.e., `g(ginv(λ)) = 1`).

The keyword argument `compact` can be used to specify the option to compute a compact descriptor realization
without non-dynamic modes, if `compact = true` (the default option) or to disable the ellimination of non-dynamic modes if `compact = false`. 

The keyword argument `minimal` specifies the option to compute minimal descriptor realization, if  `minimal = true`, or
a nonminimal realization if `minimal = false` (the default option).

The keyword argument `ss` specifies the  option to compute a standard state-space (if possible)
realizations of `syst`, if `ss = true` (default), or a descriptor system realization if `ss = false`.  

The keyword arguments `atol1`, `atol2` and `rtol` specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `n` is the order of the square matrices `A` and `E`, and  `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function gbilin(sys::DescriptorStateSpace{T},g::RationalTransferFunction; compact = true, minimal = false, standard = true, 
                atol::Real = zero(float(real(T))), atol1::Real = atol, atol2::Real = atol, 
                rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2))) where T
    
  
    # check g is first order 
    num = g.num; degn = degree(num) 
    den = g.den; degd = degree(den)
      
    (degn > 1 || degd > 1 || max(degn,degd) == 0) && 
        error("The McMillan degree of g must be one")
    
    iszero(num) && error("g must be nonzero")
    
    Ts = sys.Ts;
    Ts1 = g.Ts;   
    Ts > 0 && Ts1 > 0 && Ts != Ts1 && error("sys and g must have the same sampling periods")
    
    # assume g(delta) = (a*delta+b)/(c*delta+d)
    degn > 0 ? (a = num[1]; b = num[0]) : (a = 0; b = num[0])
    degd > 0 ? (c = den[1]; d = den[0]) : (a = a/den[0]; b = b/den[0]; c = 0; d = 1)
    
    A, E, B, C, D = dssdata(sys);
    n, m = size(B); p = size(C,1);
    
    if degd > 0
       # rational case
       At = [-b*E+d*A d*B; zeros(m,n) -eye(m)];
       Et = [a*E-c*A -c*B; zeros(m,n+m)];
       Bt = [zeros(n,m); eye(m)];
       Ct = [C D]; Dt = zeros(p,m);
       syst = dss(At,Et,Bt,Ct,Dt, Ts = Ts1);
       if minimal 
          if standard
             syst = gss2ss(gir(syst,atol1 = atol1, atol2 = atol2, rtol = rtol),atol1 = atol1, atol2 = atol2, rtol = rtol)[1];
          else
             syst = gminreal(syst,atol1 = atol1, atol2 = atol2, rtol = rtol);
          end
       else
         syst = gss2ss(syst, atol1 = atol1, atol2 = atol2, rtol = rtol, Eshape = standard ? "ident" : "triu")[1];
       end
    else
       # polynomial case
       if E == I
          # preserve standard system form
          syst = dss((A-b*E)/a,B/a,C,D,Ts=Ts1);
       else
          syst = dss(A-b*E,a*E,B,C,D,Ts=Ts1);
          standard && (syst = gss2ss(syst, atol1 = atol1, atol2 = atol2, rtol = rtol)[1] )
       end
    end

    if Ts == 0 
       Tsi = 0
    elseif Ts != 0 && Ts1 == 0
       Tsi = Ts
    else # Ts != 0 && Ts1 != 0
       Ts1 < 0 ? Tsi = Ts : Tsi = Ts1;
    end
    #ginv = (d*s-b)/(-c*s+a)
     
    ginv = rtf(Polynomial([-b, d]), Polynomial([a, -c]), Ts=Tsi, var = Tsi == 0 ? :s : :z)

    return syst, ginv
    # end GBILIN
end
    
    