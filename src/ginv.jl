"""
    ginv(sys; type = "1-2", mindeg = false, fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol, 
              offset = sqrt(ϵ)) -> (sysinv, info)

Compute for a descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)` 
a generalized inverse system `sysinv = (Ai-λEi,Bi,Ci,Di)` with the transfer function matrix `Gi(λ)` 
such that two or more of the following _Moore-Penrose conditions_ are satisfied:

         (1) G(λ)*Gi(λ)*G(λ) = G(λ);        
         (2) Gi(λ)*G(λ)*Gi(λ) = Gi(λ);
         (3) G(λ)*Gi(λ) = (G(λ)*Gi(λ))';
         (4) Gi(λ)*G(λ) = (Gi(λ)*G(λ))'.

The desired type of the computed generalized inverse can be specified using the keyword parameter
`type` as follows: 

      "1-2"     - for a generalized inverse which satisfies conditions (1) and (2) (default);
      "1-2-3"   - for a generalized inverse which satisfies conditions (1), (2) and (3);
      "1-2-4"   - for a generalized inverse which satisfies conditions (1), (2) and (4);
      "1-2-3-4" - for the Moore-Penrose pseudoinverse, which satisfies all conditions (1)-(4).

The vector `poles` specified as a keyword argument, can be used to specify the desired eigenvalues
alternatively to or jointly with enforcing a desired stability degree `sdeg` of the poles of the 
computed generalized inverse. 

The keyword argument `mindeg` can be used to specify the option to determine a minimum order 
generalized inverse, if `mindeg = true`, or a particular generalized inverse which has 
possibly non-minimal order, if `mindeg = false` (default).

To assess the presence of zeros on the boundary of the stability domain `Cs`, a boundary offset  `β` 
can be specified via the keyword parameter `offset = β`. 
Accordingly, for a continuous-time setting, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time setting, the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The returned named tuple `info` has the components `info.nrank`, `info.nfp`, `info.fnorm` and `info.tcond`,
where:
`info.nrank` is the normal rank of `G(λ)`, 
`info.nfp` is the number of freely assignable poles of the inverse `Gi(λ)`,  
`info.fnorm` is the maximum of norms of employed feedback gains (also for pole assignment) and 
`info.tcond` is the maximum of condition numbers of employed non-orthogonal transformations (see below).  

The rank determinations in the performed reductions
 are based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or the more reliable SVD-decompositions, if `fast = false`.
The computation of a minimum order inverse is performed by solving suitable minimum dynamic cover
problems. These computations involve using
non-orthogonal transformations whose maximal condition number is returned in `info.tcond`, in conjunction with 
using feedback gains (also for pole assignment), whose maximal norms are returned in `info.fnorm`. 
High values of these quantities indicate a potential loss of numerical stability of computations.  

The keyword arguments `atol1`, `atol2` and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C` and `D`,
the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A`, `E`, `B`, `C` and `D`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the maximum dimension of state, input and output vectors of the system `sys`. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 

_Method:_  The methods proposed in [1] are employed in conjunction with 
full rank factorizations computed using the approach of [2].

_References:_

[1] A. Varga. Computing generalized inverse systems using matrix pencil methods.
    Int. J. of Applied Mathematics and Computer Science, vol. 11, pp. 1055-1068, 2001.

[2] Varga, A. A note on computing range space bases of rational matrices. 
       arXiv:1707.0048, [https://arxiv.org/abs/1707.00489](https://arxiv.org/abs/1707.00489), 2017.

"""
function ginv(sys::DescriptorStateSpace{T}; type::String = "1-2", fast::Bool = true, mindeg::Bool = false,
              poles::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, 
              atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol,  
              rtol::Real = (max(sys.nx,sys.nu,sys.ny)*eps(real(float(one(T)))))*iszero(min(atol1,atol2)), 
              offset::Real = sqrt(eps(float(real(T))))) where T  
   p, m = size(sys)
   Ts = sys.Ts;   
   
   T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
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
   
   nrank = gnrank(sys; atol1, atol2, rtol);
   nfp = 0;
   tcond = 1;
   fnorm = 0;
   if p == m && m == nrank
      # compute a standard inverse for an invertible system 
      sysinv = inv(sys);
      info = (nrank = nrank, nfp = nfp, tcond = 1, fnorm = 0)
      return sysinv, info
   end
   disc = !iszero(sys.Ts)
   if type == "1-2" 
      # compute an (1,2)-inverse
      if min(p,m) == nrank
         if m == nrank
            # compute a left inverse
            sysinv, info1 = glsol([sys; I], m; atol1, atol2, rtol, fast, poles, sdeg, mindeg)
            info1.nrank != nrank && (@warn "Inconsistent rank evaluations: check tolerances")
            mindeg || (nfp = info1.nl)
         else
            # compute a right inverse
            sysinv, info1 = grsol([sys I], p; atol1, atol2, rtol, fast, poles, sdeg, mindeg)
            info1.nrank != nrank && (@warn "Inconsistent rank evaluations: check tolerances")
            mindeg || (nfp = info1.nr)
         end
         tcond = info1.tcond;
         fnorm = info1.fnorm;
      else
         if mindeg
            # compute a full rank factorization G = U*V
            U, V, info1 = grange(sys; atol1, atol2, rtol, offset, fast)
            info1.nrank != nrank && (@warn "Inconsistent rank evaluations: check tolerances")
            # compute a left inverse of least order of U
            UL, info1, = glsol([U; I], nrank; atol1, atol2, rtol, fast, poles, sdeg, mindeg)
            tcond = info1.tcond;
            fnorm = info1.fnorm;
            # compute a right inverse of least order of V
            VR, info1, = grsol([V I], nrank; atol1, atol2, rtol, fast, poles, sdeg, mindeg)
            tcond = max(tcond, info1.tcond);
            fnorm = max(fnorm, info1.fnorm);
            sysinv = gminreal(VR*UL; atol1, atol2, rtol)
         else
            # compute a generalized 1-2 inverse using the Kronecker form 
            # of the system pencil for a system with no full rank TFM
            a, e, b, c, d = dssdata(T1,sys)
            n = size(a,1);
            A, E, Q, Z, νr, μr, νi, nf, νl, μl = sklf(a, e, b, c, d; atol1, atol2, rtol, fast) 
            B = Q[n+1:end,:]'; C =  -Z[n+1:end,:]; D = zeros(T1,m,p);
            nreg = nf+sum(νi); nr1 = sum(νr); nl = sum(μl); ninv = nr1+nreg+nl;
            mr = n+m-ninv; # pr = n+p-ninv;
            if stabilize
               # make spurious zeros stable 
               if nr1 > 0
                  i1 = 1:nr1; j1 = 1:mr; j2 = mr+1:mr+nr1; 
                  F, = saloc(A[i1,j2], E[i1,j2], A[i1,j1]; evals = poles, sdeg, disc, 
                             atol1, atol2, atol3 = atol1, rtol, sepinf = false, fast) 
                  mul!(view(A,i1,j2), view(A,i1,j1), F, ONE, ONE)
                  mul!(view(C,:,j2), view(C,:,j1), F, ONE, ONE)
                  fnorm = norm(F)
               end
               if nl > 0
                  i1 = nr1+nreg+1:nr1+nreg+nl; j1=n+m-nl+1:n+m; i2 = ninv+1:n+p;
                  K, = saloc(A[i1,j1]', E[i1,j1]', A[i2,j1]'; evals = poles, sdeg, disc, 
                             atol1, atol2, atol3 = atol1, rtol, sepinf = false, fast); K = copy(K') 
                  mul!(view(A,i1,j1), K, view(A,i2,j1), ONE, ONE)
                  mul!(view(B,i1,:), K, view(B,i2,:), ONE, ONE)
                  fnorm = max(fnorm,norm(K))
               end
            end
            nfp = nr1+nl;
            # select a square system
            ic = mr+1:n+m; ir = 1:ninv; 
            sysinv = gminreal(dss(view(A,ir,ic), view(E,ir,ic), view(B,ir,:), view(C,:,ic), D; Ts); 
                              atol1, atol2, rtol)
         end
      end
   elseif type == "1-2-3"
      # compute an (1,2,3)-inverse
      if p == nrank
         # for full row rank, any (1,2)-inverse is an (1,2,3)-inverse
         sysinv, info1 = grsol([sys I], p; atol1, atol2, rtol, fast, poles, sdeg, mindeg)
         mindeg || (nfp = info1.nr)
      else
         # compute the full-rank factorization G = U*G1 with U inner
         U, G1, info1 = grange(sys; inner = true, atol1, atol2, rtol, offset, fast)
         info1.nrank != nrank && (@warn "Inconsistent rank evaluations: check tolerances")
         G2, info1 = grsol([ G1 I],nrank; atol1, atol2, rtol, fast, poles, sdeg, mindeg)
         mindeg || (nfp = info1.nr)
         sysinv = gminreal(G2*U'; atol1, atol2, rtol)
      end
      tcond = info1.tcond;
      fnorm = info1.fnorm;
   elseif type == "1-2-4"
      # compute an (1,2,4)-inverse
      if m == nrank
         # for full column rank, any (1,2)-inverse is an (1,2,4)-inverse
         sysinv, info1 = glsol([sys; I], m; atol1, atol2, rtol, fast, poles, sdeg, mindeg)
         mindeg || (nfp = info1.nl)
      else
         # compute the full-rank factorization G = G1*V with V coinner
         V, G1, info1 = gcrange(sys; coinner = true, atol1, atol2, rtol, offset, fast)
         info1.nrank != nrank && (@warn "Inconsistent rank evaluations: check tolerances")
         G2, info1 = glsol([G1; I], nrank; atol1, atol2, rtol, fast, poles, sdeg, mindeg)
         mindeg || (nfp = info1.nl)
         sysinv = gminreal(V'*G2; atol1, atol2, rtol)
      end
      tcond = info1.tcond;
      fnorm = info1.fnorm;
   elseif type == "1-2-3-4"
      # compute an (1,2,3,4)-inverse
      if m == nrank
         # for full column rank, any (1,2,3)-inverse is an (1,2,3,4)-inverse
         # compute the full-rank factorization G = U*G2 with U inner and 
         # G2 square and invertible
         U, G2, = grange(sys; inner = true, atol1, atol2, rtol, offset, fast)
         sysinv = gminreal(G2\U'; atol1, atol2, rtol)  
      elseif p == nrank
         # for full row rank, any (1,2,4)-inverse is an (1,2,3,4)-inverse
         # compute the full-rank factorization G = G2*V with V coinner and 
         # G2 square and invertible
         V, G2, = gcrange(sys; coinner = true, atol1, atol2, rtol, offset, fast)
         sysinv = gminreal(V'/G2; atol1, atol2, rtol)   
      else
         # compute the full-rank factorization G = U*G1 with U inner
         U, G1, info1 = grange(sys; inner = true, atol1, atol2, rtol, offset, fast)
         info1.nrank != nrank && (@warn "Inconsistent rank evaluations: check tolerances")
         # compute the full-rank factorization G1 = G2*V with V coinner
         V, G2, = gcrange(G1; coinner = true, atol1, atol2, rtol, offset, fast)
         # compute the Moore-Penrose pseudo-inverse 
         sysinv = gminreal(V'*(G2\U'); atol1, atol2, rtol)      
      end
   else
      error("no such computable inverse type")
   end
   info = (nrank = nrank, nfp = nfp, tcond = tcond, fnorm = fnorm)
   return sysinv, info
   #  end GINV
end
