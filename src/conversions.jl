"""
    c2d(sysc, Ts, meth = "zoh"; x0, u0, standard = true, fast = true, prewarp_freq = freq, 
               atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) -> (sysd, xd0)

Compute for the continuous-time descriptor system `sysc = (A-sE,B,C,D)` with the proper 
transfer function matrix `Gc(λ)` and for a sampling time `Ts`, the corresponding discretized
descriptor system `sysd = (Ad-zEd,Bd,Cd,Dd)` with the transfer function matrix `Gd(z)` 
according to the selected discretization method specified by `meth`. 
The keyword argument `standard` specifies the option to compute a standard state-space realization 
of `sysd` (i.e., with `Ed = I`), if `standard = true` (default), 
or a descriptor system realization if `standard = false`. 
`xd0` is the mapped initial condition of the state of the discrete-time system `sysd` determined from the 
initial conditions of the state `x0` and input `u0` of the continuous-time system `sysc`. 

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
function c2d(sysc::DescriptorStateSpace{T}, Ts::Real, meth::String = "zoh"; 
             x0::Vector = zeros(T,sys.nx), u0::Vector = zeros(T,sys.nu), 
             prewarp_freq::Real = 0, standard::Bool = true, fast::Bool = true, 
             atol::Real = zero(float(real(T))), atol1::Real = atol, atol2::Real = atol, 
             rtol::Real = sysc.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2))) where T

    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    n = sysc.nx 

    # quick exit in constant case  
    n == 0 && (return dss(sysc.D; Ts))

    meth = lowercase(meth)

    if meth == "zoh" || meth == "foh" || meth == "impulse"
       if sysc.E != I 
          # eliminate (if possible) all infinite eigenvalues in the continuous-time case with singular E
          if rcond(sysc.E) >= n*eps(float(real(T1)))
             A, E, B, C, D = dssdata(T1,sysc)
          else
             ltran = !iszero(x0)  # determine the left projection for nonzero initial condition
             sysc, L, _ = gir_lrtran(sysc; fast, ltran, finite = false, noseig = true, atol1, atol2, rtol)
             A, E, B, C, D = dssdata(sysc)
             n = size(A,1) 
             # adjust initial condition
             ltran ? (x0 = L*x0) : x0 = zeros(T1,n)
          end

          # quick exit in constant case  
          n == 0 && (return dss(D; Ts), x0)

          # check properness in continuous-time case
          F = lu!(E;check=false)
          (!issuccess(F) || rcond(UpperTriangular(F.U)) <= n*eps(float(real(T1)))) && error("improper systems not supported")
          # get rid of E matrix
          ldiv!(F,A); ldiv!(F,B)
       else
           A, _, B, C, D = dssdata(T1,sysc)
       end
       p, m = size(D)
       if meth == "zoh"
          G = exp([ rmul!(A,Ts) rmul!(B,Ts); zeros(T1,m,n+m)])
          return (dss(view(G,1:n,1:n), view(G,1:n,n+1:n+m), C, D; Ts), x0)
       elseif meth == "foh"
          G = exp([ rmul!(A,Ts) rmul!(B,Ts) zeros(T1,n,m); zeros(T1,m,n+m) 1/Ts*I; zeros(T1,m,n+2m)])
          Ad = view(G,1:n,1:n)
          G1 = view(G,1:n,n+1:n+m)
          G2 = view(G,1:n,n+m+1:n+2m)
          return (dss(Ad, G1+(Ad-I)*G2, C, D+C*G2; Ts), x0-G2*u0)
       else # meth == "impulse"
         G = exp(rmul!(A,Ts))
         return (dss(G, G*B, C, C*B; Ts), x0)
       end
    elseif meth == "tustin"
       A, E, B, C, D = dssdata(T1,sysc)
       prewarp_freq == 0 ? t = Ts : t = 2*tan(prewarp_freq*Ts/2)/prewarp_freq 
       if standard
          Ed = E-t/2*A   
          xd0 = Ed*x0 
          F = lu!(Ed)
          X = ldiv!(F,B*t)
          Ad = rdiv!(E+t/2*A,F)    # Ad = (E + A*T/2)/(E - A*T/2)
          Bd = E*X                 # Bd = E * (E - A*T/2) \ B*T
          mul!(D,C,X,0.5,1)        # Dd = D + C*(E - A*T/2)\B*(T/2) = D + C*X/2
          rdiv!(C,F)               # Cd = C/(E - A*T/2)
          xd0 -= (X*u0)/2          # xd0 = (E - A*T/2)*x0 - (E - A*T/2)\B*(T/2)*u0 = (E - A*T/2)*x0 - X*u0/2
          return (dss(Ad, Bd, C, D; Ts), xd0)
       else
         Ed = E-t/2*A              # Ed = E - A*T/2 
         X = Ed\(B*t)              # X = (E - A*T/2)\B*T 
         mul!(D,C,X,0.5,1.)        # Dd = D + (T/2)*C*(E - A*T/2)\B = D + C*X/2
         Ad = E+t/2*A              # Ad = E + A*T/2;  Cd = C
         Bd = E*X                  # Bd = E * (E - A*T/2) \ B*T
         xd0 = x0 - (X*u0)/2       # xd0 = x0 - (E - A*T/2)\B*(T/2)*u0 = x0 - X*u0/2
         return (dss(Ad, Ed, Bd, C, D; Ts), xd0)
      end
    else
       error("no such method")
    end                        
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
