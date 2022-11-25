"""
    timeresp(sys, u, t, x0 = 0; interpolation = "zoh", state_history = false, 
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
"""
    stepresp(sys[, tfinal]; x0 = zeros(sys.nx), ustep = ones(sys.nu), timesteps = 100, 
             state_history = false, fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (y, tout, x)

Compute the step response of a proper descriptor system `sys = (A-λE,B,C,D)` to step input signals. 
The final time `tfinal`, if not specified, is set to 10 for a continuous-time system
or to `abs(sys.Ts)*timesteps` for a discrete-time system, where the keyword argument
`timesteps` specifies the number of desired simulation time steps (default: `timesteps = 100`). 
The keyword argument `ustep` is a vector with as many components 
as the inputs of `sys` and specifies the desired amplitudes of step inputs (default: all components are set to 1).   
The keyword argument `x0` is a vector which specifies the initial state vector at time `0` 
and is set to zero when omitted. 

If `ns` is the total number of simulation values, `n` the number of state components, 
`p` the number of system outputs and `m` the number of system inputs, then
the resulting `ns×p×m` array `y` contains the resulting time histories of the outputs of `sys`, such 
that `y[:,:,j]` is the time response for the `j`-th input set to `ustep[j]` and the rest of inputs set to zero.  
The vector `tout` contains the corresponding values of the time samples.
The `i`-th row `y[i,:,j]` contains the output values at time `tout[i]` of the `j`-th step response.  
If the keyword parameter value `state_history = true` is used, then the resulting `ns×n×m` array`x` contains 
the resulting time histories of the state vector and 
the `i`-th row `x[i,:,j]` contains the state values at time `tout[i]` of the `j`-th step response.  
By default, the state history is not computed and `x = nothing`.

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
function stepresp(sys::DescriptorStateSpace{T}, tfinal::Real = 0;  
                  x0::AbstractVector{<:Number} = zeros(T,sys.nx), ustep::AbstractVector{<:Number} = ones(T,sys.nu), 
                  state_history::Bool = false, timesteps::Int = 100, fast::Bool = true, 
                  atol::Real = zero(float(real(T))), atol1::Real = atol, atol2::Real = atol, 
                  rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2))) where T

    n, p, m = sys.nx, sys.ny, sys.nu
 
    m1 = length(ustep)
    m == m1 || error("ustep must have as many components as system inputs")
    n == length(x0) || error("x0 must have the same length as the system state vector")

    disc = !iszero(sys.Ts)
    ns = timesteps
    ns > 0 || error("Number of time steps must be positive")
    tfinal >= 0 || error("Final time must be positive")
    if disc
       dt = abs(sys.Ts) 
       tf = tfinal == 0 ? dt*timesteps : floor(tfinal)
       ns = Int(round(tf/dt))+1
    else
       tf = tfinal == 0 ? 10 : tfinal
       dt = tf/timesteps
       ns = timesteps+1
    end
    tout = Vector{Float64}(0:dt:(ns-1)*dt) 

    #y = similar(Array{Float64,3},ns,p,m)
    T1 = promote_type(T,Float64)
    y = similar(Array{T1,3},ns,p,m)
    state_history ?  x = similar(Array{T1,3},ns,n,m) : x = nothing 
    if disc
       if sys.E == I 
          idmap = true
          xt0 = copy(x0)
          A, _, B, C, D = dssdata(T1,sys)
       else
          F = lu(sys.E;check=false)
          if issuccess(F) && rcond(UpperTriangular(F.U)) > n*eps(float(real(T1))) 
             A, _, B, C, D = dssdata(T1,sys)
             ldiv!(F,A); ldiv!(F,B)
             idmap = true
             xt0 = copy(x0)
          else
             sys, xt0, Mx, Mu = dss2ss(sys, x0; state_mapping = state_history, simple_infeigs = false, fast, atol1, atol2, rtol)  
             state_history = !isnothing(Mx)
             idmap = state_history && Mx == I && iszero(Mu)
             A, _, B, C, D = dssdata(T1,sys)
          end
       end
    else
       sysd, xt0, Mx, Mu = c2d(sys, dt; x0, 
                               state_mapping = state_history, simple_infeigs = false, atol1, atol2, rtol, fast)
       A, _, B, C, D = dssdata(sysd) 
       idmap = state_history && Mx == I && iszero(Mu)
    end
   #  if disc
   #     A, E, B, C, D = dssdata(T1,sys)
   #     if E != I 
   #        F = lu!(E;check=false)
   #        (!issuccess(F) || rcond(UpperTriangular(F.U)) <= n*eps(float(real(T1)))) && error("systems with singular E not supported")
   #        ldiv!(F,A); ldiv!(F,B)
   #     end
   #  else
   #     A, E, B, C, D = dssdata(c2d(sys, dt; simple_infeigs = false, atol1, atol2, rtol, fast)[1]) 
   #  end
    for j = 1:m
        xt = copy(xt0)
        ut = ustep[j]
        for i = 1:ns 
            y[i,:,j] = C*xt + D[:,j]*ut
            state_history && (x[i,:,j] = idmap ? xt : Mx*xt + Mu[:,j]*ut) 
            xt = A*xt + B[:,j]*ut
        end
    end
    return y, tout, x
end