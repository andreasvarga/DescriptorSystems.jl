"""
    R = dss2rm(sys; fast = true, atol = 0, atol1 = atol, atol2 = atol, gaintol = atol, rtol = min(atol1,atol2) > 0 ? 0 : n*ϵ, val) 

Build for the descriptor system `sys = (A-λE,B,C,D)` the rational matrix `R(λ) = C*inv(λE-A)*B+D` representing the 
transfer function matrix of the system `sys`.  

The keyword arguments `atol1` and `atol2` specify the absolute tolerances for the elements of `A`, `B`, `C`, `D`, and,  
respectively, of `E`, and `rtol` specifies the relative tolerances for the nonzero elements of `A`, `B`, `C`, `D` and `E`.
The default relative tolerance is `n*ϵ`, where `n` is the maximal dimension of state, input and output vectors, 
and `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol`, `atol2 = atol` and `gaintol = atol`. 


The keyword argument `gaintol` specifies the threshold for the magnitude of the nonzero elements of the gain matrix 
`C*inv(γE-A)*B+D`, where `γ = val` if `val` is a number or `γ` is a randomly chosen complex value of unit magnitude, 
if `val = missing`. Generally, `val` should not be a zero of any of entries of `R`.

_Method:_ Each rational entry of `R(λ)` is constructed from its numerator and denominator polynomials corresponding to
its finite zeros, finite poles and gain using the method of [1]. 
   
_References:_

[1] A. Varga Computation of transfer function matrices of generalized state-space models. 
    Int. J. Control, 50:2543–2561, 1989.
"""
function dss2rm(sys::DescriptorStateSpace{T}; fast::Bool = true,  
                atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                gaintol::Real = atol, val::Union{Number,Missing} = missing,
                rtol::Real = (max(sys.nx,sys.nu,sys.ny)*eps(real(float(one(T)))))*iszero(min(atol1,atol2))) where T
   NUM, DEN =  ls2rm(dssdata(sys)...; fast, atol1, atol2, rtol, gaintol, val)  
   return rtf.(pm2poly(NUM)./pm2poly(DEN),Ts = sys.Ts, var = sys.Ts == 0 ? (:s) : (:z))
end
"""
    P = dss2pm(sys; fast = true, atol = 0, atol1 = atol, atol2 = atol, gaintol = 0, rtol = min(atol1,atol2) > 0 ? 0 : n*ϵ, val) 

Build for the descriptor system `sys = (A-λE,B,C,D)` the polynomial matrix `P(λ) = C*inv(λE-A)*B+D` representing the 
transfer function matrix of the system `sys`.  

The keyword arguments `atol1` and `atol2` specify the absolute tolerances for the elements of `A`, `B`, `C`, `D`, and,  
respectively, of `E`, and `rtol` specifies the relative tolerances for the nonzero elements of `A`, `B`, `C`, `D` and `E`.
The default relative tolerance is `n*ϵ`, where `n` is the maximal dimension of state, input and output vectors, 
and `ϵ` is the working machine epsilon.
The keyword argument `atol` can be used to simultaneously set `atol1 = atol`, `atol2 = atol` and `gaintol = atol`.  

The keyword argument `gaintol` specifies the threshold for the magnitude of the nonzero elements of the gain matrix 
`C*inv(γE-A)*B+D`, where `γ = val` if `val` is a number or `γ` is a randomly chosen complex value of unit magnitude, 
if `val = missing`. Generally, `val` should not be a zero of any of entries of `P`.

_Method:_ Each entry of `P(λ)` is constructed from the polynomial corresponding 
to its finite zeros and gain using the method of [1]. 
   
_References:_

[1] A. Varga Computation of transfer function matrices of generalized state-space models. 
    Int. J. Control, 50:2543–2561, 1989.
"""
function dss2pm(sys::DescriptorStateSpace{T}; fast::Bool = true,  
                atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                gaintol::Real = atol, val::Union{Number,Missing} = missing,
                rtol::Real = (max(sys.nx,sys.nu,sys.ny)*eps(real(float(one(T)))))*iszero(min(atol1,atol2))) where T
   sys.nx > 0 && sys.E == I && error("the given realization cannot be converted to a polynomial form")
   A, E, B, C, D = dssdata(sys)
   E == I && (E = Matrix{eltype(A)}(I, sys.nx, sys.nx))
   try
      P =  ls2pm(A, E, B, C, D; fast, atol1, atol2, rtol, gaintol, val)  
      return pm2poly(P, sys.Ts == 0 ? (:s) : (:z))
   catch err
      findfirst("linearization cannot",string(err)) === nothing ? error("$err") : 
         error("the given realization cannot be converted to a polynomial form")     
   end
end
"""
    c2d(sysc, Ts, meth = "zoh"; x0, u0, standard = true, fast = true, prewarp_freq = freq, 
               state_mapping = false, simple_infeigs = true, 
               atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (sysd, xd0, Mx, Mu)

Compute for the continuous-time descriptor system `sysc = (A-sE,B,C,D)` with the proper 
transfer function matrix `Gc(λ)` and for a sampling time `Ts`, the corresponding discretized
descriptor system `sysd = (Ad-zEd,Bd,Cd,Dd)` with the transfer function matrix `Gd(z)` 
according to the selected discretization method specified by `meth`. 
The keyword argument `standard` specifies the option to compute a standard state-space realization 
of `sysd` (i.e., with `Ed = I`), if `standard = true` (default), 
or a descriptor system realization if `standard = false`. 
The keyword argument `simple_infeigs = true` indicates that only simple infinite eigenvalues 
of the pair `(A,E)` are to be expected (default). The setting `simple_infeigs = false`
indicates that possible uncontrollable or unobservable 
higher order infinite generalized eigenvalues of the pair `(A,E)` are present and have to be removed. 
`xd0` is the mapped initial condition of the state of the discrete-time system `sysd` determined from the 
initial conditions of the state `x0` and input `u0` of the continuous-time system `sysc`. 
The keyword argument `state_mapping = true` specifies the option to compute the state mapping matrices `Mx` and `Mu` such that 
the values `xc(t)` and `xd(t)` of the system state vectors of the continuous-time system `sysc` and of the discrete-time system
`sysd`, respectively, and of the input vector `u(t)` are related as `xc(t) = Mx*xd(t)+Mu*u(t)`.   
If `state_mapping = false` (the default option), then `Mx = nothing` and `Mu = nothing`.

The following discretization methods can be performed by appropriately selecting `meth`:

    "zoh"     - zero-order hold on the inputs (default); 

    "foh"     - linear interpolation of inputs (also known as first-order hold);

    "impulse" - impulse-invariant discretization; 

    "Tustin"  - Tustin transformation (also known as trapezoidal integration): a nonzero prewarping frequency
                `freq` can be specified using the keyword parameter `prewarp_freq = freq` to ensure 
                `Gd(exp(im*freq*Ts)) = Gc(im*freq)`.

The underlying pencil manipulation algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true` (default), or the SVD-decomposition,
if `fast = false`. The rank decision based on the SVD-decomposition is generally more reliable, 
but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2` and `rtol` specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `n` is the order of the square matrices `A` and `E`, and  `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function c2d(sysc::DescriptorStateSpace{T}, Ts::Real, meth::String = "zoh"; simple_infeigs::Bool = true,
             x0::Vector = zeros(T,sysc.nx), u0::Vector = zeros(T,sysc.nu), state_mapping::Bool = false, 
             prewarp_freq::Real = 0, standard::Bool = true, fast::Bool = true, 
             atol::Real = zero(float(real(T))), atol1::Real = atol, atol2::Real = atol, 
             rtol::Real = sysc.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2))) where T

    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    ONE = one(T1)
    n = sysc.nx 
    m = sysc.nu
    length(x0) == n || error("initial state vector and system state vector dimensions must coincide")
    length(u0) == m || error("initial input vector and system input vector dimensions must coincide")

    state_mapping || ( Mx = nothing; Mu = nothing)
    # quick exit in constant case  
    n == 0 && (return dss(sysc.D; Ts), x0, state_mapping ? zeros(T1,0,0) : Mx, state_mapping ? zeros(T1,0,m) : Mu)

    meth = lowercase(meth)

    if meth == "zoh" || meth == "foh" || meth == "impulse"
       if sysc.E != I 
          # eliminate (if possible) all infinite eigenvalues in the continuous-time case with singular E
          syscr, xt0, Mx, Mu = dss2ss(sysc, x0; state_mapping, simple_infeigs, fast, atol1, atol2, rtol) 
          isnothing(Mx) && (state_mapping = false)
          state_mapping && norm(Mx*xt0+Mu*u0-x0,Inf) >= eps(norm(x0,Inf)*100) && (@warn "Inconsistent initial state")
          x0 = copy(xt0)
          A, _, B, C, D = dssdata(syscr)
          n, m = size(B) 
         # if rcond(sysc.E) >= n*eps(float(real(T1)))
         #     # E invertible
         #     A, E, B, C, D = dssdata(T1,sysc)
         #     F = lu!(E)
         #     # get rid of E matrix
         #     ldiv!(F,A); ldiv!(F,B)
         #     state_mapping && (Mx = I; Mu = zeros(T1,n,m))
         #  else
         #     # E singular
         #     syscr, xt0, Mx, Mu = dss2ss(sysc, x0; state_mapping, simple_infeigs, fast, atol1, atol2, rtol) 
         #     isnothing(Mx) && (state_mapping = false)
         #     state_mapping && norm(Mx*xt0+Mu*u0-x0,Inf) >= eps(norm(x0,Inf)*100) && (@warn "Inconsistent initial state")
         #     x0 = copy(xt0)
         #     A, _, B, C, D = dssdata(syscr)
         #     n, m = size(B) 
         #  end
       else
          A, _, B, C, D = dssdata(T1,sysc)
          state_mapping && (Mx = I; Mu = zeros(T1,n,m))
       end
       m = size(D,2)
       if meth == "zoh"
          G = exp([ rmul!(A,Ts) rmul!(B,Ts); zeros(T1,m,n+m)])
          return (dss(view(G,1:n,1:n), view(G,1:n,n+1:n+m), C, D; Ts), x0, Mx, Mu)
       elseif meth == "foh"
          G = exp([ rmul!(A,Ts) rmul!(B,Ts) zeros(T1,n,m); zeros(T1,m,n+m) 1/Ts*I; zeros(T1,m,n+2m)])
          Ad = view(G,1:n,1:n)
          G1 = view(G,1:n,n+1:n+m)
          G2 = view(G,1:n,n+m+1:n+2m)
          # discrete to continuous state map Mx = M1 Mu = Mu+Mx*G2
          state_mapping && mul!(Mu, Mx, G2, ONE, ONE) 
          return (dss(Ad, G1+(Ad-I)*G2, C, D+C*G2; Ts), x0-G2*u0, Mx, Mu)
       else # meth == "impulse"
         G = exp(rmul!(A,Ts))
         return (dss(G, G*B, C, C*B; Ts), x0, Mx, Mu)
       end
    elseif meth == "tustin"
       A, E, B, C, D = dssdata(T1,sysc)
       prewarp_freq == 0 ? t = Ts : t = 2*tan(prewarp_freq*Ts/2)/prewarp_freq 
       if standard
          Ed = E-t/2*A   
          xd0 = Ed*x0 
          state_mapping ? F = lu(Ed) : F = lu!(Ed)
          X = ldiv!(F,B*t)
          Ad = rdiv!(E+t/2*A,F)    # Ad = (E + A*T/2)/(E - A*T/2)
          Bd = E*X                 # Bd = E * (E - A*T/2) \ B*T
          mul!(D,C,X,0.5,1)        # Dd = D + C*(E - A*T/2)\B*(T/2) = D + C*X/2
          rdiv!(C,F)               # Cd = C/(E - A*T/2)
          xd0 -= (X*u0)/2          # xd0 = (E - A*T/2)*x0 - (E - A*T/2)\B*(T/2)*u0 = (E - A*T/2)*x0 - X*u0/2
          state_mapping && ( Mx = inv(F); Mu = copy(ldiv!(F,X/2)))
          return (dss(Ad, Bd, C, D; Ts), xd0, Mx, Mu)
       else
         Ed = E-t/2*A              # Ed = E - A*T/2 
         X = Ed\(B*t)              # X = (E - A*T/2)\B*T 
         mul!(D,C,X,0.5,1.)        # Dd = D + (T/2)*C*(E - A*T/2)\B = D + C*X/2
         Ad = E+t/2*A              # Ad = E + A*T/2;  Cd = C
         Bd = E*X                  # Bd = E * (E - A*T/2) \ B*T
         xd0 = x0 - (X*u0)/2       # xd0 = x0 - (E - A*T/2)\B*(T/2)*u0 = x0 - X*u0/2
         state_mapping && ( Mx = I; Mu = X/2)
         return (dss(Ad, Ed, Bd, C, D; Ts), xd0, Mx, Mu)
      end
    else
       error("no such method")
    end                        
    # end C2D
end
"""
    gbilin(sys, g; compact = true, minimal = false, standard = true, atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (syst, ginv)

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

The keyword argument `standard` specifies the  option to compute a standard state-space (if possible)
realizations of `syst`, if `standard = true` (default), or a descriptor system realization if `standard = false`.  

The keyword arguments `atol1`, `atol2` and `rtol` specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `n` is the order of the square matrices `A` and `E`, and  `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function gbilin(sys::DescriptorStateSpace{T},g::RationalTransferFunction; 
                compact::Bool = true, minimal::Bool = false, standard::Bool = true, 
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
"""
    timeresp(sys, u, t, x0 = 0; interpolation = "zoh", state_history = false, simple_infeigs = true, 
             fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (y, tout, x)

Compute the time response of a proper descriptor system `sys = (A-λE,B,C,D)` to the input signals 
described by `u` and `t`. The time vector `t` consists of regularly spaced time samples. The 
matrix `u` has as many columns as the inputs of `sys` and its `i`-th row specifies 
the input values at time `t[i]`. For discrete-time models, `u` should be sampled at the same rate as `sys`
if `sys.Ts > 0` and `t` must have all time steps equal to `sys.Ts` or can be set to an empty vector. 
The vector `x0` specifies the initial state vector at time `t[1]` and is set to zero when omitted. 

The matrix `y` contains the resulting time history of the outputs of `sys` and 
the vector `tout` contains the corresponding values of the time samples.
The `i`-th row of `y` contains the output values at time `tout[i]`.  
If the keyword parameter value `state_history = false` is used, then the matrix `x` contains 
the resulting time history of the state vector and its `i`-th row contains 
the state values at time `tout[i]`. By default, the state history is not computed and `x = nothing`.

For continuous-time models, the input values are interpolated between samples. By default, 
zero-order hold based interpolation is used. The linear interpolation method can be selected using 
the keyword parameter `interpolation = "foh"`.

By default, the uncontrollable infinite eigenvalues and simple infinite eigenvalues of the pair `(A,E)` 
are eliminated. 
The underlying pencil manipulation algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true` (default), or the SVD-decomposition,
if `fast = false`. The rank decision based on the SVD-decomposition is generally more reliable, 
but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2` and `rtol` specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `n` is the order of the square matrices `A` and `E`, and  `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function timeresp(sys::DescriptorStateSpace{T}, u::AbstractVecOrMat{<:Number}, t::AbstractVector{<:Real},  
                  x0::AbstractVector{<:Number} = zeros(T,sys.nx); interpolation::String = "zoh", 
                  state_history::Bool = false, fast::Bool = true, 
                  atol::Real = zero(float(real(T))), atol1::Real = atol, atol2::Real = atol, 
                  rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2))) where T

    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    T2 = promote_type(T1,eltype(x0),eltype(u)) 
    n = sys.nx 
    p, m = sys.ny, sys.nu
 
    N, m1 = size(u,1), size(u,2)
    m == m1 || error("u must have as many columns as system inputs")
    n == length(x0) || error("x0 must have the same length as the system state vector")

    disc = !iszero(sys.Ts)
    ns = length(t)
    ns > 0 && ns != N && error("u must have the same number of rows as the number of values in t")
    if ns > 1
       dt = t[2]-t[1]
       disc && sys.Ts > 0 && abs.(sys.Ts-dt) > 0.0000001*dt 
       (any(diff(t) .<= 0) || any(isinf.(t)) || any(abs.(diff(t).-dt).> 0.00001*dt)) && 
            error("time vector t must contain monotonically increasing and evenly spaced time samples")
    end
    if disc
       # set tout
       ns <= 1 && (dt = abs(sys.Ts) )
       tout = Vector{real(T1)}(0:dt:(N-1)*dt) 
    else
       tout = t
    end
    p == 1 ? y = Vector{T2}(undef, N) : y = Matrix{T2}(undef, N, p) 
    state_history ?  x = Matrix{T2}(undef, N, n) : x = nothing 
    if disc
       if sys.E == I 
          idmap = true
          xt = copy(x0)
          A, _, B, C, D = dssdata(T1,sys)
       else
          F = lu(sys.E;check=false)
          if issuccess(F) && rcond(UpperTriangular(F.U)) > n*eps(float(real(T1))) 
             A, _, B, C, D = dssdata(T1,sys)
             ldiv!(F,A); ldiv!(F,B)
             idmap = true
             xt = copy(x0)
          else
            sys, xt, Mx, Mu = dss2ss(sys, x0; state_mapping = state_history, simple_infeigs = false, fast, atol1, atol2, rtol)  
            state_history = !isnothing(Mx)
            idmap = state_history && Mx == I && iszero(Mu)
            A, _, B, C, D = dssdata(T1,sys)
         end
       end
    else
       sysd, xt, Mx, Mu = c2d(sys, dt, interpolation; x0, u0 = u[1,:], 
                              state_mapping = state_history, simple_infeigs = false, atol1, atol2, rtol, fast)
       A, _, B, C, D = dssdata(sysd) 
       idmap = state_history && Mx == I && iszero(Mu)
   end
   for i = 1:N 
      ut = view(u,i,:)
      y[i,:] = C*xt + D*ut
      state_history && (x[i,:] = idmap ? xt : Mx*xt + Mu*ut) 
      xt = A*xt + B*ut
   end
   return y, tout, x
end
