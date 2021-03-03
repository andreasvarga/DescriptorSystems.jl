"""
    glmcover1(sys1, sys2; fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol) -> (sysx, sysy, info)

Determine for the proper descriptor systems `sys1 = (A1-λE1,B1,C1,D1)` and 
`sys2 = (A2-λE2,B2,C2,D2)` with the transfer function matrices `X1(λ)` and `X2(λ)`, 
respectively, using a left minimum dynamic cover of Type 1 based 
order reduction, the descriptor systems `sysx` and `sysy` with the 
transfer function matrices `X(λ)` and `Y(λ)`, respectively, such that 

    X(λ) = X1(λ) + Y(λ)*X2(λ) , 

and `sysx` has order less than the order of `sys1`.  

The call with

    glmcover1(sys, p1; fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol) -> (sysx, sysy, info)

uses the compound descriptor system `sys = (A-λE,B,[C1; C2],[D1; D2])`, 
where `C1` and `D1` have `p1` rows, to define  
the proper descriptor systems `sys1 = (A-λE,B,C1,D1)` and `sys2 = (A-λE,B,C2,D2)`
(i.e., `A1-λE1 = A2-λE2 = A-λE` and `B1 = B2 = B`).   
    
The resulting descriptor systems `sysx` and `sysy` have observable realizations
of the form `sysx = (Ao-λEo,Bo1,Co,D1)` and `sysy = (Ao-λEo,Bo2,Co,0)`, 
where the pencil `[Ao-λEo; Co]` is in a (observability) staircase form,  
with `νl[i] x νl[i+1]` full row rank diagonal blocks, for `i = 1, ..., nl`, 
with `νl[nl+1] := p1`. 

The resulting named triple `ìnfo` contains `(stdim, tcond, fnorm) `, 
where `ìnfo.stdim = νl` is a vector which contains the column dimensions of the
blocks of the staircase form `[Ao-λEo; Co]`, `ìnfo.tcond` is the maximum of 
the Frobenius-norm condition numbers of the employed non-orthogonal 
transformation matrices, and `ìnfo.fnorm` is 
the Frobenius-norm of the (internally) employed output-injection gain to reduce the order. 
Large values of  `ìnfo.tcond` or `ìnfo.fnorm` indicate possible loss of 
numerical stability of computations. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, 
the absolute tolerance for the nonzero elements of `E1` and `E2`,  
and the relative tolerance for the nonzero elements of 
`A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, `E1` and `E2`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the maximal order of the systems `sys1` and `sys2`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`. 

The rank determinations in the performed reductions
 are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Note:_ `glmcover1` also works for arbitrary descriptor system `sys1`, 
if `sys2` is proper. For an improper system `sys1`, the order 
reduction is performed only for the proper part of `sys1`, while the 
polynomial part of `sys1` is included without modification in the  
resulting realization of `sysx`. In this case, `ìnfo.stdim = νl` contains
the information corresponding to the proper part of `sysx`. 

_Method:_ The dual of method  of [1] is used to compute Type 1 minimum dynamic covers 
for standard systems and the dual of method of [2] for proper descriptor systems.   
The resulting McMillan degree of `sysx` is the least achievable one
provided the realization of `sys2` is maximally controllable 
(i.e., the pair `(A2+F*C2-λE2,B2+F*D2)` is controllable for any `F`). 

_References:_

[1] A. Varga, Reliable algorithms for computing minimal dynamic covers,
    Proc. CDC'03, Maui, Hawaii, 2003.

[2] A. Varga. Reliable algorithms for computing minimal dynamic covers for 
    descriptor systems. Proc. MTNS Symposium, Leuven, Belgium, 2004. 
"""
function glmcover1(sys1::DescriptorStateSpace, sys2::DescriptorStateSpace; kwargs...)  
   size(sys1,2) == size(sys2,2) || throw(DimensionMismatch("sys1 and sys2 must have the same number of inputs"))
   sysx, sysy, info = grmcover1(gdual(sys1), gdual(sys2); kwargs...)
   info.stdim[:] = reverse(info.stdim)
   return gdual(sysx,rev = true), gdual(sysy,rev = true), info 
end
function glmcover1(sys::DescriptorStateSpace, p1::Int; kwargs...)  
   p = size(sys,1);
   (p1 <= p && p1 >= 0) || throw(DimensionMismatch("p1 must be at most $p, got $p1"))
   sysx, sysy, info = grmcover1(gdual(sys,rev = true), p1; kwargs...)
   info.stdim[:] = reverse(info.stdim)
   return gdual(sysx,rev = true), gdual(sysy,rev = true), info
end
"""
    glmcover2(sys1, sys2; fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol) -> (sysx, sysy, info)

Determine for the proper descriptor systems `sys1 = (A1-λE1,B1,C1,D1)` and 
`sys2 = (A2-λE2,B2,C2,D2)` with the transfer function matrices `X1(λ)` and `X2(λ)`, 
respectively, using a left minimum dynamic cover of Type 2 based 
order reduction, the descriptor systems `sysx` and `sysy` with the 
transfer function matrices `X(λ)` and `Y(λ)`, respectively, such that 

    X(λ) = X1(λ) + Y(λ)*X2(λ) , 

and `sysx` has order less than the order of `sys1`.  

The call with

    glmcover2(sys, p1; fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol) -> (sysx, sysy, info)

uses the compound descriptor system `sys = (A-λE,B, [C1; C2],[D1; D2])`, 
where `C1` and `D1` have `p1` rows and `E` is invertible, to define  
the proper descriptor systems `sys1 = (A-λE,B,C1,D1)` and `sys2 = (A-λE,B,C2,D2)`
(i.e., `A1-λE1 = A2-λE2 = A-λE` and `B1 = B2 = B`).   
   
The resulting descriptor systems `sysx` and `sysy` have observable realizations
of the form `sysx = (Ao-λEo,Bo1,Co,Do1)` and `sysy = (Ao-λEo,Bo2,Co,Do2)`, 
where the pencil `[Ao-λEo; Co]` is in a (observability) staircase form,  
with `νl[i] x νl[i+1]` full row rank diagonal blocks, for `i = 1, ..., nl`, 
with `νl[nl+1] := p1`. 

The resulting named triple `ìnfo` contains `(stdim, tcond, fnorm, gnorm) `, 
where `ìnfo.stdim = νl` is a vector which contains the column dimensions of the
blocks of the staircase form `[Ao-λEo; Co]`, `ìnfo.tcond` is the maximum of 
the Frobenius-norm condition numbers of the employed non-orthogonal 
transformation matrices, `ìnfo.fnorm` is 
the Frobenius-norm of the (internally) employed output-injection gain to reduce the order, and 
`ìnfo.gnorm` is the Frobenius-norm of the (internally) employed output-feedforward gain. 
Large values of  `ìnfo.tcond`,`ìnfo.fnorm` or `ìnfo.gnorm` indicate possible loss of 
numerical stability of computations. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, 
the absolute tolerance for the nonzero elements of `E1` and `E2`,  
and the relative tolerance for the nonzero elements of 
`A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, `E1` and `E2`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the maximal order of the systems `sys1` and `sys2`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`. 

The rank determinations in the performed reductions
 are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Note:_ `glmcover2` also works for arbitrary descriptor system `sys1`, 
if `sys2` is proper. For an improper system `sys1`, the order 
reduction is performed only for the proper part of `sys1`, while the 
polynomial part of `sys1` is included without modification in the  
resulting realization of `sysx`. In this case, `ìnfo.stdim = νl` contains
the information corresponding to the proper part of `sysx`. 

_Method:_ The dual of method  of [1] is used to compute Type 2 minimum dynamic covers 
for standard systems and the dual of method of [2] for proper descriptor systems.   
The resulting McMillan degree of `sysx` is the least achievable one
provided the realization of `sys2` is maximally controllable 
(i.e., the pair `(A2+F*C2-λE2,B2+F*D2)` is controllable for any `F`). 

References:

[1] A. Varga, Reliable algorithms for computing minimal dynamic covers,
    Proc. CDC'03, Maui, Hawaii, 2003.

[2] A. Varga. Reliable algorithms for computing minimal dynamic covers for 
    descriptor systems. Proc. MTNS Symposium, Leuven, Belgium, 2004. 
"""
function glmcover2(sys1::DescriptorStateSpace, sys2::DescriptorStateSpace; kwargs...)  
   size(sys1,2) == size(sys2,2) || throw(DimensionMismatch("sys1 and sys2 must have the same number of inputs"))
   sysx, sysy, info = grmcover2(gdual(sys1), gdual(sys2); kwargs...)
   info.stdim[:] = reverse(info.stdim)
   return gdual(sysx,rev = true), gdual(sysy,rev = true), info
end
function glmcover2(sys::DescriptorStateSpace, p1::Int; kwargs...)  
   p = size(sys,1);
   (p1 <= p && p1 >= 0) || throw(DimensionMismatch("p1 must be at most $p, got $p1"))
   sysx, sysy, info = grmcover2(gdual(sys,rev = true), p1; kwargs...)
   info.stdim[:] = reverse(info.stdim)
   return gdual(sysx,rev = true), gdual(sysy,rev = true), info
end
"""
    grmcover1(sys1, sys2; fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol) -> (sysx, sysy, info)

Determine for the proper descriptor systems `sys1 = (A1-λE1,B1,C1,D1)` and 
`sys2 = (A2-λE2,B2,C2,D2)` with the transfer function matrices `X1(λ)` and `X2(λ)`, 
respectively, using a right minimum dynamic cover of Type 1 based 
order reduction, the descriptor systems `sysx` and `sysy` with the 
transfer function matrices `X(λ)` and `Y(λ)`, respectively, such that 

    X(λ) = X1(λ) + X2(λ)*Y(λ) , 

and `sysx` has order less than the order of `sys1`.  

The call with

    grmcover1(sys, m1; fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol) -> (sysx, sysy, info)

uses the compound descriptor system `sys = (A-λE,[B1 B2],C,[D1 D2])`, 
where `B1` and `D1` have `m1` columns, to define  
the proper descriptor systems `sys1 = (A-λE,B1,C,D1)` and `sys2 = (A-λE,B2,C,D2)`
(i.e., `A1-λE1 = A2-λE2 = A-λE` and `C1 = C2 = C`).   
    
The resulting descriptor systems `sysx` and `sysy` have controllable realizations
of the form `sysx = (Ar-λEr,Br,Cr1,D1)` and `sysy = (Ar-λEr,Br,Cr2,0)`, 
where the pencil `[Br Ar-λEr]` is in a (controllability) staircase form,  
with `νr[i] x νr[i-1]` full row rank diagonal blocks, for `i = 1, ..., nr`, 
with `νr[0] := m1`. 

The resulting named triple `ìnfo` contains `(stdim, tcond, fnorm) `, 
where `ìnfo.stdim = νr` is a vector which contains the row dimensions of the
blocks of the staircase form `[Br Ar-λEr]`, `ìnfo.tcond` is the maximum of 
the Frobenius-norm condition numbers of the employed non-orthogonal 
transformation matrices, and `ìnfo.fnorm` is 
the Frobenius-norm of the (internally) employed state-feedback to reduce the order. 
Large values of  `ìnfo.tcond` or `ìnfo.fnorm` indicate possible loss of 
numerical stability of computations. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, 
the absolute tolerance for the nonzero elements of `E1` and `E2`,  
and the relative tolerance for the nonzero elements of 
`A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, `E1` and `E2`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the maximal order of the systems `sys1` and `sys2`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`. 

The rank determinations in the performed reductions
 are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Note:_ `grmcover1` also works for arbitrary descriptor system `sys1`, 
if `sys2` is proper. For an improper system `sys1`, the order 
reduction is performed only for the proper part of `sys1`, while the 
polynomial part of `sys1` is included without modification in the  
resulting realization of `sysx`. In this case, `ìnfo.stdim = νr` contains
the information corresponding to the proper part of `sysx`. 

_Method:_ The method  of [1] is used to compute Type 1 minimum dynamic covers 
for standard systems and the method of [2] for proper descriptor systems.   
The resulting order (McMillan degree) of `sysx` is the least achievable one
provided the realization of `sys2` is maximally observable 
(i.e., the pair `(A2+B2*F-λE2,C2+D2*F)` is observable for any `F`). 

References:

[1] A. Varga, Reliable algorithms for computing minimal dynamic covers,
    Proc. CDC'03, Maui, Hawaii, 2003.

[2] A. Varga. Reliable algorithms for computing minimal dynamic covers for 
    descriptor systems. Proc. MTNS Symposium, Leuven, Belgium, 2004. 
"""
function grmcover1(sys::DescriptorStateSpace{T}, m1::Int; atol::Real = zero(real(T)), 
                   atol1::Real = atol, atol2::Real = atol, atol3::Real = atol, 
                   rtol::Real = ((size(sys.A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2,atol3)), 
                   fast::Bool = true) where T 

   p, m = size(sys);
   (m1 <= m && m1 >= 0) || throw(DimensionMismatch("m1 must be at most $m, got $m1"))
   m2 = m-m1
   Ts = sys.Ts;   
  
   n = size(sys.A,1); 
   isys1 = 1:m1; isys2 = m1+1:m;
  
   # handle simple cases
   if n == 0
      sysx = sys[:,isys1];
      info = (stdim = Int[], tcond = 1, fnorm = 0)
      sysy = dss(zeros(T,m2,m1),Ts = Ts)
      return sysx, sysy, info
   end
  
   if  m1 == 0
      sysx = dss(zeros(T,p,m1),Ts = Ts); 
      info = (stdim = Int[], tcond = 1, fnorm = 0)
      sysy = dss(zeros(T,m2,m1),Ts = Ts)
      return sysx, sysy, info
   end
  
   T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
   At, Et, Bt, Ct, D = dssdata(T1,sys)
   sstype = (Et == I)
   sstype || istriu(Et) || MatrixPencils._qrE!(At, Et, nothing, Bt; withQ = false) 
   ONE = one(T1)
   ZERO = zero(T1)


   # handle improper case 
   if !sstype && LinearAlgebra.LAPACK.trcon!('1','U','N',triu(Et))  < n*eps(real(T1)) 
      isproper(sys[:,isys2], atol1 = atol1, atol2 = atol2, rtol = rtol) || 
             error("The system sys2 must be proper")
      At, Et, Bt, Ct, _, _, _, blkdims = fiblkdiag(At, Et, Bt, Ct; fast = fast, finite_infinite = true, 
                                                   atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = false, withZ = false) 
      n1 = blkdims[1]; i1 = 1:n1; i2 = n1+1:n
      sysi1 = dss(lsminreal(At[i2,i2], Et[i2,i2], Bt[i2,isys1], Ct[:,i2], zeros(T,p,m1), 
                  fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)[1:5]...,Ts = Ts)
      γ = iszero(Ts) ? ZERO : ONE
      DCgain = Ct[:,i2]*((lmul!(γ,view(Et,i2,i2))-view(At,i2,i2))\view(Bt,i2,isys2))
 
      sysfm = dss(At[i1,i1], Et[i1,i1], Bt[i1,:], Ct[:,i1], D + [zeros(T,p,m1) DCgain], Ts = Ts)
      sysfx, sysy, info = grmcover1(sysfm, m1, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)  
      return sysfx+sysi1, sysy, info
   end    

   
   # reduce to the special controllability form of [1] or [2]
   if sstype
       _, tau, ncont, nuc = sklf_right2!(At, Bt, m1, Ct; fast = fast, atol1 = atol1, atol2 = atol2, 
                                         rtol = rtol, withQ = false) 
   else
       _, _, tau, ncont, nuc = sklf_rightfin2!(At, Et, Bt, m1, Ct; fast = fast, atol1 = atol1, atol2 = atol2, 
                                               rtol = rtol, withQ = false, withZ = false)
   end
   if ncont == 0
       sysx = dss(D[:,isys1],Ts = Ts);
       sysx.Ts = Ts;
       info = (stdim = [], tcond = 1, fnorm = 0)
       sysy = dss(zeros(T,m2,m1),Ts = Ts)
       return sysx, sysy, info
   end
   
   if  m2 == 0
       if sstype
          sysx = dss(At,I,Bt,Ct,D,Ts = Ts);
       else
          sysx = dss(At,Et,Bt,Ct,D,Ts = Ts);
       end
       info = (stdim = tau[tau .> 0], tcond = 1, fnorm = 0)
       sysy = dss(zeros(T,m2,m1),Ts = Ts)
       return sysx, sysy, info
   end
    
   # perform permutation for Type I cover
   p = length(tau)
   cind = Int(p/2) 
   ind = zeros(Int,ncont);
   iodd = 1:2:p
   ieven = 2:2:p
   ind1 = tau[iodd]; n1 = sum(ind1)
   ind2 = tau[ieven]; n2 = sum(ind2)
   i2 = 1; i1 = 1; ioff1 = 0; ioff2 = 0; 
   for i = 1:cind
       i1i = ind1[i]
       i2i = ind2[i]
       ioff1 +=  i1i
       i2m = i2+i2i-1
       ind[(i2:i2m) .+ n1] =  Vector(i2:i2m) .+ ioff1 
       i2 = i2m+1
       i1m = i1+i1i-1
       ind[i1:i1m] =  Vector(i1:i1m) .+ ioff2 
       ioff2 +=  i2i
       i1 = i1m+1
   end
   At = At[ind,ind]; Bt = Bt[ind,:]; Ct = Ct[:,ind]; 
   sstype || (Et = Et[ind,ind])
   
   # anihilate lower left blocks of At and Et
   tcond = real(T1)(ncont); # the Frobenius condition number of identity matrix
   nlowc1 = n1;             nlowc2 = n1+n2-ind2[cind];
   nlowr1 = n1+n2;          nlowr2 = nlowr1;
   for i = cind:-1:2
      nlowc1 = nlowc1-ind1[i]; nlowc2 = nlowc2-ind2[i-1]; 
      nlowr1 = nlowr1-ind2[i]; nlowr2 = nlowr2-ind2[i];
      ic1 = nlowc1+1:n1; ic2 = nlowc2+1:nlowc2+ind2[i-1];
      ir2 =  nlowr2+1:nlowr2+ind2[i]; 
      x = -At[ir2,ic2]\At[ir2,ic1]; 
      tcond = max(tcond,ncont+norm(x));
      ir = 1:nlowr1;
      mul!(view(At,ir,ic1), view(At,ir,ic2), x, ONE, ONE) 
      At[ir2,ic1] .= ZERO
      mul!(view(Ct,:,ic1), view(Ct,:,ic2), x, ONE, ONE) 
      if sstype
         x = -x; 
         mul!(view(At,ic2,:), x, view(At,ic1,:), ONE, ONE) 
         mul!(view(Bt,ic2,:), x, view(Bt,ic1,:), ONE, ONE) 
      else
         mul!(view(Et,ir,ic1), view(Et,ir,ic2), x, ONE, ONE) 
         y = -Et[ic2,ic1]/Et[ic1,ic1]; 
         tcond = max(tcond,ncont+norm(y));
         mul!(view(Et,ic2,:), y, view(Et,ic1,:), ONE, ONE) 
         mul!(view(At,ic2,:), y, view(At,ic1,:), ONE, ONE) 
         mul!(view(Bt,ic2,:), y, view(Bt,ic1,:), ONE, ONE) 
      end
   end
      
   # form the reduced system
   ic = 1:n1; i3 = n1+1:n1+ind2[1]; 
   f2 = -Bt[i3,isys2]\At[i3,ic];
   mul!(view(At,ic,ic), view(Bt,ic,isys2), f2, ONE, ONE)
   mul!(view(Ct,:,ic), view(D,:,isys2), f2, ONE, ONE)
   if sstype
      sysx = dss(At[ic,ic], I, Bt[ic,isys1], Ct[:,ic], D[:,isys1], Ts = Ts)
      sysy = dss(At[ic,ic], I, Bt[ic,isys1], f2, zeros(T1,m2,m1), Ts = Ts)
   else
      sysx = dss(At[ic,ic], Et[ic,ic], Bt[ic,isys1], Ct[:,ic], D[:,isys1],Ts = Ts)
      sysy = dss(At[ic,ic], Et[ic,ic], Bt[ic,isys1], f2, zeros(T1,m2,m1),Ts = Ts)
   end
   info = (stdim = ind1, tcond = tcond, fnorm = norm(f2))

   return sysx, sysy, info
   # end GRMCOVER1
end
function grmcover1(sys1::DescriptorStateSpace, sys2::DescriptorStateSpace; kwargs...)  
    size(sys1,1) == size(sys2,1) || throw(DimensionMismatch("sys1 and sys2 must have the same number of outputs"))
    return grmcover1(gir([sys1 sys2]; kwargs...), size(sys1,2); kwargs...)
end
"""
    grmcover2(sys1, sys2; fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol) -> (sysx, sysy, info)

Determine for the proper descriptor systems `sys1 = (A1-λE1,B1,C1,D1)` and 
`sys2 = (A2-λE2,B2,C2,D2)` with the transfer function matrices `X1(λ)` and `X2(λ)`, 
respectively, using a right minimum dynamic cover of Type 2 based 
order reduction, the descriptor systems `sysx` and `sysy` with the 
transfer function matrices `X(λ)` and `Y(λ)`, respectively, such that 

    X(λ) = X1(λ) + X2(λ)*Y(λ) , 

and `sysx` has order less than the order of `sys1`.  

The call with

    grmcover2(sys, m1; fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol) -> (sysx, sysy, info)

uses the compound descriptor system `sys = (A-λE,[B1 B2],C,[D1 D2])`, 
where `B1` and `D1` haves `m1` columns, to define  
the proper descriptor systems `sys1 = (A-λE,B1,C,D1)` and `sys2 = (A-λE,B2,C,D2)`
(i.e., `A1-λE1 = A2-λE2 =: A-λE` and `C1 = C2 =: C`).   

The resulting descriptor systems `sysx` and `sysy` have controllable realizations
of the form `sysx = (Ar-λEr,Br,Cr1,Dr1)` and `sysy = (Ar-λEr,Br,Cr2,Dr2)`, 
where the pencil `[Br Ar-λEr]` is in a (controllability) staircase form,  
with `νr[i] x νr[i-1]` full row rank diagonal blocks, for `i = 1, ..., nr`, 
with `νr[0] := m1`. 

The resulting named triple `ìnfo` contains `(stdim, tcond, fnorm, gnorm) `, 
where `ìnfo.stdim = νr` is a vector which contains the row dimensions of the
blocks of the staircase form `[Br Ar-λEr]`, `ìnfo.tcond` is the maximum of 
the Frobenius-norm condition numbers of the employed non-orthogonal 
transformation matrices, `ìnfo.fnorm` is 
the Frobenius-norm of the (internally) employed state-feedback gain to reduce the order,
`ìnfo.gnorm` is 
the Frobenius-norm of the (internally) employed feedforward gain to reduce the order. 
Large values of  `ìnfo.tcond`, `ìnfo.fnorm`  or  `ìnfo.gnorm` indicate possible loss of 
numerical stability of computations. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, 
the absolute tolerance for the nonzero elements of `E1` and `E2`,  
and the relative tolerance for the nonzero elements of 
   `A1`, `B1`, `C1`, `D1`, `A2`, `B2`, `C2`, `D2`, `E1` and `E2`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the maximal order of the systems `sys1` and `sys2`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`. 

The rank determinations in the performed reductions
 are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Note:_ `grmcover2` also works for arbitrary descriptor system `sys1`, 
if `sys2` is proper. For an improper system `sys1`, the order 
reduction is performed only for the proper part of `sys1`, while the 
polynomial part of `sys1` is included without modification in the  
resulting realization of `sysx`. In this case, `ìnfo.stdim = νr` contains
the information corresponding to the proper part of `sysx`. 

_Method:_ The method  of [1] is used to compute Type 2 minimum dynamic covers 
for standard systems and the method of [2] for proper descriptor systems.   
The resulting McMillan degree of `sysx` is the least achievable one
provided the realization of `sys2` is maximally observable 
(i.e., the pair `(A2+B2*F-λE2,C2+D2*F)` is observable for any `F`). 

References:

[1] A. Varga, Reliable algorithms for computing minimal dynamic covers,
    Proc. CDC'03, Maui, Hawaii, 2003.

[2] A. Varga. Reliable algorithms for computing minimal dynamic covers for 
    descriptor systems. Proc. MTNS Symposium, Leuven, Belgium, 2004. 

"""
function grmcover2(sys::DescriptorStateSpace{T},m1::Int; atol::Real = zero(real(T)), 
   atol1::Real = atol, atol2::Real = atol, atol3::Real = atol, 
   rtol::Real = ((size(sys.A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2,atol3)), 
   fast::Bool = true) where T 
   
   p, m = size(sys);
   (m1 <= m && m1 >= 0) || throw(DimensionMismatch("m1 must be at most $m, got $m1"))
   m2 = m-m1
   Ts = sys.Ts;     
   
   # exchange the roles of B1 and B2
   n = size(sys.A,1); 
   isys1 = m1+1:m; isys2 = 1:m1; 
   
   # handle symple cases
   if n == 0
       sysx = sys[:,isys2];
       info = (stdim = Int[], tcond = 1, fnorm = 0, gnorm = 0)
       sysy = dss(zeros(T,m2,m1),Ts = Ts)
       return sysx, sysy, info
   end
   
   if  m1 == 0
      sysx = dss(zeros(T,p,m1),Ts = Ts); 
      info = (stdim = Int[], tcond = 1, fnorm = 0, gnorm = 0)
      sysy = dss(zeros(T,m2,m1),Ts = Ts)
      return sysx, sysy, info
   end

   T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
   At, Et, Bt, Ct, D = dssdata(T1,sys)
   sstype = (Et == I)
   ONE = one(T1)
   ZERO = zero(T1)

   sstype || istriu(Et) || MatrixPencils._qrE!(At, Et, nothing, Bt; withQ = false) 

   # handle improper case 
   if !sstype && LinearAlgebra.LAPACK.trcon!('1','U','N',triu(Et))  < n*eps(real(T1)) 
      isproper(sys[:,m1+1:m], atol1 = atol1, atol2 = atol2, rtol = rtol) || 
             error("The system sys2 must be proper")
      # sysf, sysi = gsdec(sys, job = "finite", fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
      # sysfm = sysf + [zeros(p,m1) dcgain(sysi[:,m1+1:m]) ];
      # sysfx, sysy, info = grmcover2(sysfm, m1, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)  
      # # sysf.D[:,:] = [sysf.D[:,1:m1] sysf.D[:,m1+1:m]+evalfr(sysi[:,m1+1:m],rand())]
      # # sysfx, sysy, info = grmcover2(sysf, m1, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)  
      # return sysfx+gminreal(sysi[:,1:m1],fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol), sysy, info
      At, Et, Bt, Ct, _, _, _, blkdims = fiblkdiag(At, Et, Bt, Ct; fast = fast, finite_infinite = true, 
                                                   atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = false, withZ = false) 
      n1 = blkdims[1]; i1 = 1:n1; i2 = n1+1:n
      sysi1 = dss(lsminreal(At[i2,i2], Et[i2,i2], Bt[i2,1:m1], Ct[:,i2], zeros(T,p,m1), 
                  fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)[1:5]...,Ts = Ts)
      γ = iszero(Ts) ? ZERO : ONE
      DCgain = Ct[:,i2]*((lmul!(γ,view(Et,i2,i2))-view(At,i2,i2))\view(Bt,i2,m1+1:m))
 
      sysfm = dss(At[i1,i1], Et[i1,i1], Bt[i1,:], Ct[:,i1], D + [zeros(T,p,m1) DCgain], Ts = Ts)
      sysfx, sysy, info = grmcover2(sysfm, m1, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)  
      return sysfx+sysi1, sysy, info
   end    


   ind21 = [Vector(isys1); Vector(isys2)]
   Bt = Bt[:,ind21]
 
   # reduce to the special controllability form of [1]
   if sstype
      _, tau, ncont, nuc = sklf_right2!(At, Bt, m2, Ct; fast = fast, atol1 = atol1, atol2 = atol2, 
                                        rtol = rtol, withQ = false) 
   else
      _, _, tau, ncont, nuc = sklf_rightfin2!(At, Et, Bt, m2, Ct; fast = fast, atol1 = atol1, atol2 = atol2, 
                                              rtol = rtol, withQ = false, withZ = false)
   end

   if ncont == 0
       sysx = dss(D[:,isys2],Ts = Ts);
       sysx.Ts = Ts;
       info = (stdim = [], tcond = 1, fnorm = 0, gnorm = 0)
       sysy = dss(zeros(T,m2,m1),Ts = Ts)
       return sysx, sysy, info
   end
   
   if  m2 == 0
      if sstype
         sysx = dss(At,I,Bt,Ct,D,Ts = Ts);
      else
         sysx = dss(At,Et,Bt,Ct,D,Ts = Ts);
      end
      info = (stdim = tau[tau .> 0], tcond = 1, fnorm = 0, gnorm = 0)
      sysy = dss(zeros(T,m2,m1),Ts = Ts)
      return sysx, sysy, info
   end
  
   # perform permutation for type II cover
   p = length(tau)
   cind = Int(p/2) 
   ind = zeros(Int,ncont);
   iodd = 1:2:p
   ieven = 2:2:p
   ind1 = tau[iodd]; n1 = sum(ind1)
   ind2 = tau[ieven]; n2 = sum(ind2)
   i2 = 1; i1 = 1; ioff1 = 0; ioff2 = 0; 
  for i = 1:cind
      i1i = ind1[i]
      i2i = ind2[i]
      ioff1 +=  i1i
      i2m = i2+i2i-1
      ind[i2:i2m] =  Vector(i2:i2m) .+ ioff1 
      i2 = i2m+1
      i1m = i1+i1i-1
      ind[(i1:i1m) .+ n2] =  Vector(i1:i1m) .+ ioff2 
      ioff2 +=  i2i
      i1 = i1m+1
   end
   At = At[ind,ind]; Bt = Bt[ind,:]; Ct = Ct[:,ind] 
   sstype || (Et = Et[ind,ind])
   ONE = one(T1)
   ZERO = zero(T1)
   
   # anihilate lower left blocks of At and Et
   tcond = real(T1)(ncont) # the Frobenius condition number of identity matrix
   nlowc1 = n2-ind2[cind];  nlowc2 = n1+n2-ind1[cind];
   nlowr1 = n1+n2;          nlowr2 = nlowr1-ind1[cind];
   for i = cind:-1:1
      if i == 1
         ice1 = 1:n2; ir1 = n2+1:n2+ind1[i];
      else
         nlowc1 = nlowc1-ind2[i-1]; nlowc2 = nlowc2-ind1[i-1]; 
         nlowr1 = nlowr1-ind1[i]; nlowr2 = nlowr2-ind1[i-1];
         ic1 = nlowc1+1:n2; ic2 = nlowc2+1:nlowc2+ind1[i-1];
         ir1 =  nlowr1+1:nlowr1+ind1[i]; 
         ice1 = nlowc1+ind2[i-1]+1:n2; 
      end
      if ~sstype
         y = -Et[ir1,ice1]/Et[ice1,ice1]; 
         tcond = max(tcond,ncont+norm(y));
         mul!(view(Et,ir1,:), y, view(Et,ice1,:), ONE, ONE)
         mul!(view(At,ir1,:), y, view(At,ice1,:), ONE, ONE)
         mul!(view(Bt,ir1,:), y, view(Bt,ice1,:), ONE, ONE)
      end
      i == 1 && break
      x = -At[ir1,ic2]\At[ir1,ic1]; 
      tcond = max(tcond,ncont+norm(x));
      ir = 1:nlowr1;
      mul!(view(At,ir,ic1), view(At,ir,ic2), x, ONE, ONE) 
      At[ir1,ic1] .= ZERO
      mul!(view(Ct,:,ic1), view(Ct,:,ic2), x, ONE, ONE) 
      if sstype
         x = -x; 
         mul!(view(At,ic2,:), x, view(At,ic1,:), ONE, ONE)
         mul!(view(Bt,ic2,:), x, view(Bt,ic1,:), ONE, ONE)
      else
         mul!(view(Et,ir,ic1), view(Et,ir,ic2), x, ONE, ONE) 
      end
   end
   
   
   # form the reduced system
   ic = 1:n2; i3 = n2+1:n2+ind1[1]; 
   f2 = -Bt[i3,1:m2]\At[i3,ic];
   g = -Bt[i3,1:m2]\Bt[i3,m2+1:m]; 
   mul!(view(Ct,:,ic), view(D,:,isys1), f2, ONE, ONE)
   mul!(view(D,:,isys2), view(D,:,isys1), g, ONE, ONE)
   if sstype
      sysx = dss(At[ic,ic], I, Bt[ic,m2+1:m], Ct[:,ic], D[:,isys2], Ts = Ts)
      sysy = dss(At[ic,ic], I, Bt[ic,m2+1:m], f2, g, Ts = Ts)
   else
      sysx = dss(At[ic,ic], Et[ic,ic], Bt[ic,m2+1:m], Ct[:,ic], D[:,isys2], Ts = Ts)
      sysy = dss(At[ic,ic], Et[ic,ic], Bt[ic,m2+1:m], f2, g, Ts = Ts)
   end
   info = (stdim = ind2[ind2 .> 0], tcond = tcond, fnorm = norm(f2), gnorm = norm(g))
      
   return sysx, sysy, info

   # end GRMCOVER2
end
   
   
function grmcover2(sys1::DescriptorStateSpace, sys2::DescriptorStateSpace; kwargs...)  
    size(sys1,1) == size(sys2,1) || throw(DimensionMismatch("sys1 and sys2 must have the same number of outputs"))
    return grmcover2(gir([sys1 sys2]; kwargs...), size(sys1,2);kwargs...)
end



