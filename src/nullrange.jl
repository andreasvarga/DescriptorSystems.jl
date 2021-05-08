"""
    gcrange(sys; zeros = "none", coinner = false, atol = 0, atol1 = atol, atol2 = atol, atol3 = atol, rtol, 
           fast = true, offset = sqrt(ϵ)) -> (sysr, sysx, info)

Compute for the descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`, 
the proper descriptor system `sysr = (Ar-λEr,Br,Cr,Dr)` with a full row rank 
transfer function matrix `R(λ)` such that `Coimage(G(λ)) = Coimage(R(λ))` and the  
descriptor system `sysx = (A-λE,B,Cx,Dx)` with the full column rank transfer function matrix `X(λ)`,
which satisfies
   
     G(λ) = X(λ)*R(λ) ,

representing a full rank factorization of `G(λ)`.  
The number of rows of `R(λ)` is the normal rank `r` of `G(λ)`. 
The rows of `R(λ)` form a rational basis of the coimage space of the rational matrix `G(λ)`. 
A selected set of zeros of `G(λ)` are included as zeros of `R(λ)`. 

The resulting named triple `ìnfo` contains `(nrank, nfuz, niuz) `, where `ìnfo.nrank = r`, 
the normal rank of `G(λ)`, `ìnfo.nfuz` is the number of finite zeros of `sys` on 
the boundary of the stability domain `Cs`, and `ìnfo.niuz` is the number of infinite zeros of `sys` in 
the continuous-time case and is set to `0` in the discrete-time case. 

The following options can be selected via the keyword parameter `zeros` for which zeros of `G(λ)` 
to be included in `R(λ)`:

     "none"       - include no zeros (default) 
     "all"        - include all zeros of `sys`
     "unstable"   - include all unstable zeros of `sys`
     "s-unstable" - include all strictly unstable zeros of `sys`, both finite and infinite
     "stable"     - include all stable zeros of `sys`
     "finite"     - include all finite zeros of `sys`
     "infinite"   - include all infinite zeros of `sys`

If `coinner = true`, the resulting basis `R(λ)` is _coinner_, i.e., `R(λ)*R(λ)' = I`, where `R(s)' = transpose(R(-s))` for a 
continuous-time system with `λ = s` and `R(z)' = transpose(R(1/z))` for a discrete-time system with `λ = z`. 
This option can be used only in conjunction with `zeros = "none"` or `zeros = "unstable"`. 

For a continuous-time system `sys`, the stability domain `Cs` is defined as the set of 
complex numbers with real parts at most `-β`, 
while for a discrete-time system `sys`, `Cs` is the set of complex numbers with 
moduli at most `1-β` (i.e., the interior of a disc of radius `1-β` centered in the origin). 
The boundary offset `β` to be used to assess the stability of zeros and their number 
on the boundary of `Cs` can be specified via the keyword parameter `offset = β`.
Accordingly, for a continuous-time system, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time system, the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The keyword arguments `atol1`, `atol2` and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C` and `D`,
the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A`, `E`, `B`, `C` and `D`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`, `atol3 = atol`. 

For the assessment of zeros, the dual system pencil `transpose([A-λE B; C D])` is reduced to a 
special Kronecker-like form (see [2]). In this reduction, the 
performed rank decisions are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_  The range computation method described in [1], is applied to the
dual descriptor system realization corresponding to the transpose of the 
rational matrix `G(λ)`. The underlying pencil reduction algorithm of [2], 
has been adapted to deal with several zero selection options. The computation of the involved 
Kronecker-like form is based on the algorithm of [3].

_References:_

[1] Varga, A.
    A note on computing the range of rational matrices. 
    arXiv:1707.0048, [https://arxiv.org/abs/1707.0048](https://arxiv.org/abs/1707.004), 2017.

[2] C. Oara.
    Constructive solutions to spectral and inner–outer factorizations 
    respect to the disk. Automatica,  41, pp. 1855–1866, 2005. 

[3] C. Oara and P. Van Dooren. 
    An improved algorithm for the computation of structural invariants of a system pencil and related geometric aspects. 
    Syst. Control Lett., 30:39–48, 1997.
"""
function gcrange(sys::DescriptorStateSpace{T}; zeros = "none", coinner::Bool = false, 
                 atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol,  
                 rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(max(atol1,atol2)), 
                 offset::Real = sqrt(eps(float(real(T)))), fast::Bool = true) where T  
   sysr, sysx, info = grange(gdual(sys); zeros = zeros, inner = coinner, offset = offset, 
                      atol = atol, atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)
   return gdual(sysr), gdual(sysx), info
end
"""
    grange(sys; zeros = "none", atol = 0, atol1 = atol, atol2 = atol, atol3 = atol, rtol, 
           fast = true, offset = sqrt(ϵ)) -> (sysr, sysx, info)

Compute for the descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`, 
the proper descriptor system `sysr = (Ar-λEr,Br,Cr,Dr)` with a full column rank 
transfer function matrix `R(λ)` such that `Range(G(λ)) = Range(R(λ))` and the  
descriptor system `sysx = (A-λE,B,Cx,Dx)` with the full row rank transfer function matrix `X(λ)`,
which satisfies
   
     G(λ) = R(λ)*X(λ) ,

representing a full rank factorization of `G(λ)`.  
The number of columns of `R(λ)` is the normal rank `r` of `G(λ)`. 
The columns of `R(λ)` form a rational basis of the range (or image) space of the rational matrix `G(λ)`. 
A selected set of zeros of `G(λ)` are included as zeros of `R(λ)`. 

The resulting named triple `ìnfo` contains `(nrank, nfuz, niuz) `, where `ìnfo.nrank = r`, 
the normal rank of `G(λ)`, `ìnfo.nfuz` is the number of finite zeros of `sys` on 
the boundary of the stability domain `Cs`, and `ìnfo.niuz` is the number of infinite zeros of `sys` in 
the continuous-time case and is set to `0` in the discrete-time case. 

Depending on the value of the keyword parameter `zeros`, the following options can be selected 
for the zeros of `G(λ)` to be included in `R(λ)`:

     "none"       - include no zeros (default) 
     "all"        - include all zeros of `sys`
     "unstable"   - include all unstable zeros of `sys`
     "s-unstable" - include all strictly unstable zeros of `sys`, both finite and infinite
     "stable"     - include all stable zeros of `sys`
     "finite"     - include all finite zeros of `sys`
     "infinite"   - include all infinite zeros of `sys`

If `inner = true`, the resulting basis `R(λ)` is _inner_, i.e., `R(λ)'*R(λ) = I`, where `R(s)' = transpose(R(-s))` for a 
continuous-time system with `λ = s` and `R(z)' = transpose(R(1/z))` for a discrete-time system with `λ = z`. 
This option can be used only in conjunction with `zeros = "none"` or `zeros = "unstable"`. 

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

The keyword arguments `atol1`, `atol2` and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C` and `D`,
the absolute tolerance for the nonzero elements of `E`,   
and the relative tolerance for the nonzero elements of `A`, `E`, `B`, `C` and `D`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`. 

For the assessment of zeros, the system pencil `[A-λE B; C D]` is reduced to a 
special Kronecker-like form (see [2]). In this reduction, the 
performed rank decisions are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

_Method:_  The range computation method is described in [1] and is based on 
the reduction algorithm of [2], which has been adapted to deal with 
several zero selection options. The computation of the involved 
Kronecker-like form is based on the algorithm of [3].

_References:_

[1] Varga, A.
    A note on computing the range of rational matrices. 
    arXiv:1707.0048, [https://arxiv.org/abs/1707.0048](https://arxiv.org/abs/1707.004), 2017.

[2] C. Oara.
    Constructive solutions to spectral and inner–outer factorizations 
    respect to the disk. Automatica,  41, pp. 1855–1866, 2005. 

[3] C. Oara and P. Van Dooren. 
    An improved algorithm for the computation of structural invariants of a system pencil and related geometric aspects. 
    Syst. Control Lett., 30:39–48, 1997.
"""
function grange(sys::DescriptorStateSpace{T}; zeros = "none", inner::Bool = false, 
                atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(max(atol1,atol2)), 
                offset::Real = sqrt(eps(float(real(T)))), fast::Bool = true) where T 

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
   # zeros = "none", or contains zeros according to the specified option in zeros.  
   
   
   At, Et, _, Z, dimsc, nmszer, nizer = gsklf(dssdata(sys)...; disc = disc, 
                jobopt = zeros,  
                offset = offset, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, 
                withQ = false, withZ = true)  
   
   n, m = size(sys.B); p = size(sys.D,1);

   # form the reduced system (Abl-λ*Ebl,Bbl,Cbl,Dbl)         
   nr = dimsc[1]; nric = dimsc[2]; mric = dimsc[3]; nsinf = dimsc[4]; 
   il = n-nsinf-nric+1:n-nsinf; jal = nr+1:nr+nric; jbl = nr+nric+1:nr+nric+mric; icl = n+1:n+p
   # A = At[il,jal]; E = Et[il,jal]; B = At[il,jbl]; C = At[icl,jal]; D = At[icl,jbl];
   A = view(At,il,jal); E = view(Et,il,jal); B = view(At,il,jbl); C = view(At,icl,jal); D = view(At,icl,jbl)

   if inner && (zeros == "none" || zeros == "unstable")
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
         H = cholesky(Hermitian(D'*D+B'*X*B)).U 
      else
         # X, _, F, = garec(A,E,B,D'*D,C'*C,C'*D); 
         X, _, F, = try 
            garec(A,E,B,D'*D,C'*C,C'*D) 
         catch err
            findfirst("dichotomic",string(err)) === nothing ? error("$err") :
               error("Solution of the CARE failed: Hamiltonian pencil has jw-axis eigenvalues") 
         end
         H = qr(D).R[1:mric,:]
      end
      # construct the inner basis
      sysr = dss(A-B*F, E, B/H, C-D*F, D/H, Ts = sys.Ts)
      # construct the full row rank factor
      CDt = [H*F H]*Z[:,nr+1:n+m-nsinf]' 
      sysx = dss(sys.A, sys.E, sys.B, CDt[:,1:n], CDt[:,n+1:n+m], Ts = sys.Ts);
   else
      sysr = dss(A, E, B, C, D, Ts = sys.Ts)
      sysx = dss(sys.A, sys.E, sys.B, Z[1:n,jbl]', Z[n+1:n+m,jbl]', Ts = sys.Ts);
   end
   info = (nrank = mric, nfuz = nmszer, niuz = nizer)

   return sysr, sysx, info

end
"""
    glnull(sys; polynomial = false, simple = false, coinner = false, fast = true, poles = missing, sdeg = missing,  
           atol = 0, atol1 = atol, atol2 = atol, rtol, offset = sqrt(ϵ) ) -> (syslnull, info)

Determine for the descriptor systems `sys = (A-λE,B,C,D)` with the `p x m` transfer function matrix `G(λ)`, 
the descriptor system `syslnull = (Al-λEl,Bl,Cl,Dl)` with the transfer function matrix `Nl(λ)` 
such that `Nl(λ)` is a minimal rational left nullspace basis of `G(λ)` and satisfies `Nl(λ)*G(λ) = 0`.     

For the call with

    glnull(sys, m2; polynomial = false, simple = false, coinner = false, fast = true, poles = missing, sdeg = missing,  
           atol = 0, atol1 = atol, atol2 = atol, rtol, offset = sqrt(ϵ) ) -> (syslnull, info)

`sys` contains the compound system `sys = [sys1 sys2]`, with `G(λ)`, the transfer function matrix of `sys1`, and 
`G2(λ)`, the transfer function matrix of `sys2`, and has the descriptor realization `sys = (A-λE,[B B2],C,[D D2])`, 
where `sys2` has `m2` inputs. The resulting `syslnull` contains the compound system 
`[syslnull1 syslnull1*sys2] = (Al-λEl,[Bl Bl2],Cr,[Dl Dl2])`, where
`syslnull1 = (Al-λEl,Bl,Cl,Dl)` has the transfer function matrix `Nl(λ)`, which is a rational left nullspace basis of `G(λ)`
satisfying `Nl(λ)*G(λ) = 0` and `syslnull1*sys2 = (Al-λEl,Bl2,Cl,Dl2)` has the transfer function matrix `Nl(λ)*G2(λ)`. 

The returned named tuple `info` has the components `info.nrank`, `info.stdim`, `info.degs`, `info.fnorm` and `info.tcond`.

If `polynomial = false`, the resulting `syslnull` has a proper transfer function matrix, 
while for `polynomial = true` the resulting `syslnull` has a polynomial transfer function matrix. 
The resulting basis `Nl(λ)` contains `p-r` basis vectors, where `r = rank G(λ)`. The rank `r` is returned in `info.nrank`.
If `simple = true`, the resulting basis is _simple_ and satisfies the condition that the sum of the 
number of poles of the `p-r` basis vectors is equal to the number of poles of `Nl(λ)` (i.e., its McMillan degree) . 

For a non-simple proper basis, the realization `(Al-λEl,Bl,Cl,Dl)` is observable and the pencil `[Al-λEl; Cl]`
is in an observable staircase form. The row dimensions of the full column rank diagonal blocks are returned in `info.stdim`
and the corresponding left Kronecker indices are returned in `info.degs`.  
For a simple basis, the regular pencil `Al-λEl` is block diagonal, with the `i`-th block of size `info.stdim[i]`. 
The increasing numbers of poles of the basis vectors are returned in `info.degs`. 
For the `i`-th basis vector `vi(λ)` (i.e., the `i`-th row of `Nl(λ)`) 
a minimal realization can be explicitly constructed as `(Ali-λEli,Bl,Cli,Dl[i,:])`, where `Ali`, `Eli` and `Cli` are
the `i`-th diagonal blocks of `Al`, `El`, and `Cl`, respectively, and `Dl[i,:]` is the `i`-th row of `Dl`. 
The corresponding realization of `vi(λ)*G2(λ)` can be constructed as `(Ali-λEli,Bl2,Cl2,Dl2[i,:])`, where
`Dl2[i,:]` is the `i`-th row of `Dl2`.

For a proper basis, the poles of `Nl(λ)` can be freely assigned, by assigning the eigenvalues of the pencil `Al-λEl`.
The vector `poles`, specified as a keyword argument, can be used to specify the desired eigenvalues,
alternatively to or jointly with enforcing a desired stability degree `sdeg` of the real parts of the eigenvalues, 
for a continuous-time system, or the moduli of eigenvalues, for a discrete-time system. 
If `coinner = true`, the resulting basis `Nl(λ)` is _coinner_, i.e., `Nl(λ)*Nl(λ)' = I`, where `Nl(s)' = transpose(Nl(-s))` for a 
continuous-time system with `λ = s` and `Nl(z)' = transpose(Nl(1/z))` for a discrete-time system with `λ = z`. 
If the proper basis is simple, each of the resulting individual basis vector is inner. 
If `sys2` has poles on the boundary of the appropriate stability domain `Cs`, which are not poles of `sys1` too, 
then there exists no inner `Nl(λ)` such that `Nl(λ)*G2(λ)` is stable. An offset can be specified via the keyword parameter `offset = β`
to be used to assess the existence of zeros on the stability domain boundary. Accordingly, for a continuous-time system, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time system, the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`,  
the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol` and `atol2 = atol`. 

The rank determinations in the performed reductions
 are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.
The computation of simple bases involves the solution of several Type 1 minimum dynamic cover
problems. This computation involves using
non-orthogonal transformations whose worst condition number is returned in `info.tcond`, in conjunction with 
using feedback gains, whose norms are returned in `info.fnorm`. High values of these quantities
indicate a potential loss of numerical stability of computations.  

_Note:_ The resulting realization of `syslnull` is minimal provided the realization of `sys` is minimal. 
However, `syslnull1` is a minimal basis only if the realization (A-lambda E,B,C,D) of `sys1` is 
minimal. In this case, `info.degs` are the degrees of the vectors of a minimal polynomial basis or, 
if `simple = true`, of the resulting minimal simple proper basis. 

_Method:_ The computation method for the computation of a right nullspace basis is applied
to the dual of descriptor system `sys`.
The computation of a minimal proper right nullspace basis is based
on [1]; see also [2]. For the computation of a minimal simple proper 
right nullspace basis the method of [3] is emloyed to compute a simple basis from a
minimal proper basis. For the computation of an inner proper right nullspace basis,
the inner factor of an inner-outer factorization of `Nl(λ)` is explicitly 
constructed using formulas given in [4]. 

_References:_

[1] T.G.J. Beelen.
    New algorithms for computing the Kronecker structure of a pencil 
    with applications to systems and control theory. 
    Ph. D. Thesis, Technical University Eindhoven, 1987.

[2] A. Varga.
    On computing least order fault detectors using rational nullspace bases. 
    IFAC SAFEPROCESS'03 Symposium, Washington DC, USA, 2003.

[3] A. Varga.
    On computing nullspace bases – a fault detection perspective. 
    Proc. IFAC 2008 World Congress, Seoul, Korea, pages 6295–6300, 2008.

[4] K. Zhou, J. C. Doyle, and K. Glover. 
    Robust and Optimal Control. Prentice Hall, 1996.
"""
function glnull(sys::DescriptorStateSpace{T}, m2::Int = 0; polynomial::Bool = false,
                poles::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, 
                offset::Real = sqrt(eps(float(real(T)))), 
                atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(max(atol1,atol2)), 
                fast::Bool = true, coinner::Bool = false, simple::Bool = false) where T 
   p, m = size(sys)
   (m2 <= m && m2 >= 0) || throw(DimensionMismatch("m2 must be at most $m, got $m2"))
   polynomial && coinner && (coinner = false; @warn "Coinner option not allowed for polynomial basis")  
   sysnull, info1 = grnull(gdual(sys), m2; polynomial = polynomial, poles = poles, sdeg = sdeg, offset = offset, 
                         atol = atol, atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast, inner = coinner, simple = simple)
   info = (nrank = info1.nrank, stdim = reverse(info1.stdim), degs = info1.degs, tcond = info1.tcond, fnorm = info1.fnorm)
   return gdual(sysnull[:,p-info.nrank:-1:1],rev = true), info
end
"""
    grnull(sys; polynomial = false, simple = false, inner = false, fast = true, poles = missing, sdeg = missing,  
           atol = 0, atol1 = atol, atol2 = atol, rtol, offset = sqrt(ϵ) ) -> (sysrnull, info)

Determine for the descriptor systems `sys = (A-λE,B,C,D)` with the `p x m` transfer function matrix `G(λ)`, 
the descriptor system `sysrnull = (Ar-λEr,Br,Cr,Dr)` with the transfer function matrix `Nr(λ)` 
such that `Nr(λ)` is a minimal rational right nullspace basis of `G(λ)` and satisfies `G(λ)*Nr(λ) = 0`.     

For the call with

    grnull(sys, p2; polynomial = false, simple = false, inner = false, fast = true, poles = missing, sdeg = missing,  
           atol = 0, atol1 = atol, atol2 = atol, rtol, offset = sqrt(ϵ) ) -> (sysrnull, info)

`sys` contains the compound system `sys = [sys1; sys2]`, with `G(λ)`, the transfer function matrix of `sys1`, and 
`G2(λ)`, the transfer function matrix of `sys2`, and has the descriptor realization `sys = (A-λE,B,[C;C2],[D;D2])`, 
where `sys2` has `p2` outputs. The resulting `sysrnull` contains the compound system 
`[sysrnull1; sys2*sysrnull1] = (Ar-λEr,Br,[Cr;Cr2],[Dr;Dr2])`, where
`sysrnull1 = (Ar-λEr,Br,Cr,Dr)` has the transfer function matrix `Nr(λ)`, which is a rational right nullspace basis of `G(λ)`
satisfying `G(λ)*Nr(λ) = 0` and `sys2*sysrnull1 = (Ar-λEr,Br,Cr2,Dr2)` has the transfer function matrix `G2(λ)*Nr(λ)`. 

The returned named tuple `info` has the components `info.nrank`, `info.stdim`, `info.degs`, `info.fnorm` and `info.tcond`.

If `polynomial = false`, the resulting `sysrnull` has a proper transfer function matrix, 
while for `polynomial = true` the resulting `sysrnull` has a polynomial transfer function matrix. 
The resulting basis `Nr(λ)` contains `m-r` basis vectors, where `r = rank G(λ)`. The rank `r` is returned in `info.nrank`.
If `simple = true`, the resulting basis is _simple_ and satisfies the condition that the sum of the 
number of poles of the `m-r` basis vectors is equal to the number of poles of `Nr(λ)` (i.e., its McMillan degree) . 

For a non-simple proper basis, the realization `(Ar-λEr,Br,Cr,Dr)` is controllable and the pencil `[Br Ar-λEr]`
is in a controllable staircase form. The column dimensions of the full row rank diagonal blocks are returned in `info.stdim`
and the corresponding right Kronecker indices are returned in `info.degs`.  
For a simple basis, the regular pencil `Ar-λEr` is block diagonal, with the `i`-th block of size `info.deg[i]` 
(the `i`-th right Kronecker index) for a proper basis and  `info.deg[i]+1` for a polynomial basis. 
The dimensions of the diagonal blocks are returned in this case in `info.stdim`, while the increasing numbers of poles of 
the basis vectors are returned in `info.degs`. For the `i`-th basis vector `vi(λ)` (i.e., the `i`-th column of `Nr(λ)`) 
a minimal realization can be explicitly constructed as `(Ari-λEri,Bri,Cr,Dr[:,i])`, where `Ari`, `Eri` and `Bri` are
the `i`-th diagonal blocks of `Ar`, `Er`, and `Br`, respectively, and `Dr[:,i]` is the `i`-th column of `Dr`. 
The corresponding realization of `G2(λ)*vi(λ)` can be constructed as `(Ari-λEri,Bri,Cr2,Dr2[:,i])`, where
`Dr2[:,i]` is the `i`-th column of `Dr2`.

For a proper basis, the poles of `Nr(λ)` can be freely assigned, by assigning the  eigenvalues of the pencil `Ar-λEr`.
The vector `poles`, specified as a keyword argument, can be used to specify the desired eigenvalues,
alternatively to or jointly with enforcing a desired stability degree `sdeg` of the real parts of the eigenvalues, 
for a continuous-time system, or the moduli of eigenvalues, for a discrete-time system. 
If `inner = true`, the resulting basis `Nr(λ)` is _inner_, i.e., `Nr(λ)'*Nr(λ) = I`, where `Nr(s)' = transpose(Nr(-s))` for a 
continuous-time system with `λ = s` and `Nr(z)' = transpose(Nr(1/z))` for a discrete-time system with `λ = z`. 
If the proper basis is simple, each of the resulting individual basis vector is inner. 
If `sys2` has poles on the boundary of the appropriate stability domain `Cs`, which are not poles of `sys1` too, 
then there exists no inner `Nr(λ)` such that `G2(λ)*Nr(λ)` is stable. An offset can be specified via the keyword parameter `offset = β`
to be used to assess the existence of zeros on the stability domain boundary. Accordingly, for a continuous-time system, 
the boundary of `Cs` contains the complex numbers with real parts within the interval `[-β,β]`, 
while for a discrete-time system, the boundary of `Cs` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`,  
the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol` and `atol2 = atol`. 

The rank determinations in the performed reductions
 are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.
The computation of simple bases involves the solution of several Type 1 minimum dynamic cover
problems. This computation involves using
non-orthogonal transformations whose worst condition number is returned in `info.tcond`, in conjunction with 
using feedback gains, whose norms are returned in `info.fnorm`. High values of these quantities
indicate a potential loss of numerical stability of computations.  

_Note:_ The resulting realization of `sysrnull` is minimal provided the realization of `sys` is minimal. 
However, `sysrnull1` is a minimal basis only if the realization (A-lambda E,B,C,D) of `sys1` is 
minimal. In this case, `info.degs` are the degrees of the vectors of a minimal polynomial basis or, 
if `simple = true`, of the resulting minimal simple proper basis. 


_Method:_ The computation of a minimal proper right nullspace basis is based
on [1]; see also [2]. For the computation of a minimal simple proper 
right nullspace basis the method of [3] is emloyed to compute a simple basis from a
minimal proper basis. For the computation of an inner proper right nullspace basis,
the inner factor of an inner-outer factorization of `Nr(λ)` is explicitly 
constructed using formulas given in [4]. 

_References:_

[1] T.G.J. Beelen.
    New algorithms for computing the Kronecker structure of a pencil 
    with applications to systems and control theory. 
    Ph. D. Thesis, Technical University Eindhoven, 1987.

[2] A. Varga.
    On computing least order fault detectors using rational nullspace bases. 
    IFAC SAFEPROCESS'03 Symposium, Washington DC, USA, 2003.

[3] A. Varga.
    On computing nullspace bases – a fault detection perspective. 
    Proc. IFAC 2008 World Congress, Seoul, Korea, pages 6295–6300, 2008.

[4] K. Zhou, J. C. Doyle, and K. Glover. 
    Robust and Optimal Control. Prentice Hall, 1996.
"""
function grnull(sys::DescriptorStateSpace{T}, p2::Int = 0; polynomial::Bool = false,
                poles::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, 
                offset::Real = sqrt(eps(float(real(T)))), 
                atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                rtol::Real = ((size(sys.A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2)), 
                fast::Bool = true, inner::Bool = false, simple::Bool = false) where T 

   p, m = size(sys)
   (p2 <= p && p2 >= 0) || throw(DimensionMismatch("p2 must be at most $p, got $p2"))
   Ts = sys.Ts;   
   polynomial && inner && (inner = false; @warn "Inner option not allowed for polynomial basis")
   
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
   
   # maximum allowed condition number 
   maxtcond = 1.e4;

   At, Et, Bt, Ct, Dt = dssdata(T1,sys);
       
   n = size(At,1)
   p1 = p-p2
   ip1 = 1:p1; ip2 = p1+1:p;

   M, N, Q, Z, νr, μr, = sklf_right(At, Et, Bt, Ct[ip1,:], Dt[ip1,:]; fast = true, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = true, withZ = true) 
   
   # separate the left Kronecker structure using the Kronecker-like form of 
   # the system pencil S(λ) = [ At-λEt Bt; Ct1 Dt1]: 
   #
   #        Q'*S(λ)*Z = [ Br Ar - λ Er      *     ]
   #                    [ 0     0       Al - λ El ] 
   # 
   # A = [a b ; c(ip1,:) d(ip1,:)]; E = [e zeros(n,m);zeros(p1,n+m)]; 
   # mode = 2; # perform separation of infinite part
   # qz = 2;   # accumulate only Z
   # [A,E,~,Z,nur,mur] = sl_klf(A,E,tolc,mode,qz); 
     
   # determine main dimensional parameters
   nr = sum(νr);    # order of left structure (also order of Ar)
   mr = sum(μr)-nr; # number of basis elements
   nrank = m - mr;   # normal rank of the transfer matrix SYS1
   
   tcond = 1;     # set condition number of used transformations
   fnorm = 0; 
   
   degs = zeros(Int,mr)
   if nrank == m 
      # full row rank: the null space is empty
      sysrnull = dss(zeros(T,m+p2,0), Ts = Ts)
      info = (nrank = nrank, stdim = zeros(Int,0), degs = degs, tcond = 1, fnorm = 0)
      return sysrnull, info
   end
   # in the case nrank = 0, a null space basis is the identity matrix 
   # and therefore sysrnull = [eye(p); sys(ip2,:)] and degs = zeros(1,m);
   # however, to handle properly this case, the right Kronecker indices 
   # must be determined in degs; the basis is generally not minimal
   
   # compute the right Kronecker indices, which are also the degrees of 
   # the vectors of a polynomial right nullspace basis
   nvi = μr-νr;    # there are nvi(i) vectors of degree i-1
   nb = length(νr); # number of blocks in [ Br Ar ]
   k = 1;
   for i = 1:nb
       for j = 1:nvi[i]
           degs[k] = i-1
           k += 1
       end
   end

   # normalize right part 
   klf_right_refineut!(νr, μr, M, N, Q, Z, withQ = false, withZ = true) 

   # form  CR1 = [0 I]*Z = [ Dr1 Cr1 * ] and 
   #       CR2 = [ C2 D2]*Z = [ Dr2 Cr2 * ]    
   CR1 =  Z[n+1:n+m,:] 
   CR2 = [ Ct[ip2,:] Dt[ip2,:]]*Z; 
   
   iar = 1:nr; jar = mr+1:mr+nr; jbr = 1:mr; 
   
   ar = view(M,iar,jar); cr = [ CR1[:,jar]; CR2[:,jar] ];  er = view(N,iar,jar); 
   br = view(M,iar,jbr); dr = [ CR1[:,jbr]; CR2[:,jbr] ]; 
   #sklf_rightnorm!(νr,ar,er,br,cr,dr) 
   sysrnull = dss(ar,er,br,cr,dr)
   if maximum(degs) == 0
      info = (nrank = nrank, stdim = simple ? zeros(Int,mr) : νr[νr .> 0], degs = degs, tcond = 1, fnorm = 0)
      return dss(ar,er,br,cr,dr,Ts = Ts), info
   end
    
   if inner && p2 > 0
      # check stabilizability of SYSF*NR1
      zer = gzero(dss(ar,er,br,cr[1:m,:],dr[1:m,:],Ts=Ts), atol1 = atol1,atol2 = atol2, rtol = rtol)
      if disc
         zf = abs.(zer[isfinite.(zer)]) 
         nostab = any((zf .>= 1-offset) .& (zf .<= 1+offset)) 
      else 
         zf = real.(zer[isfinite.(zer)]) 
         nostab = any(isinf.(zer)) || any((zf .>= -offset) .& (zf .<= offset)) 
      end
      if nostab 
         mes = "No inner nullspace Nr exists such that SYSF*Nr is stable - \n  standard nullspace computed instead"
         @warn mes
         inner = false;  # perform standard nullspace computation
      end
   end
       
   if !simple || mr == 1 
      if polynomial 
         # pr = size(dr,1)
         f, g, SF, = salocinf(ar, er, br; fast = fast, atol1 = atol1, atol2 = atol2, atol3 = atol1)
         # build (ar+br*f-λ(er+br*g), br, cr+dr*f+λdr*g, dr)
         # ar1 = [I zeros(T1,pr,nr); zeros(T1,nr,pr) SF.S]
         # er1 = [zeros(T1,pr,pr) dr*g*SF.Z; zeros(T1,nr,pr) SF.T]
         # br1 = [zeros(T1,pr,mr); SF.Q'*br]
         # cr1 = [-I (cr+dr*f)*SF.Z]
         # sysrnull = dss(ar1,er1,br1,cr1,dr,Ts = Ts) 
         sysrnull = dss(SF.S, SF.T, SF.Q'*br, missing, (cr+dr*f)*SF.Z, dr*g*SF.Z, dr, missing, Ts = Ts, 
                        compacted = true, atol1 = atol1, atol2 = atol2, rtol = rtol)
         fnorm = max(fnorm, norm(g))
         simple &&  (blkdims = [order(sysrnull)])
      else
         if inner 
            # compute the outer factor Nro of the inner-outer factorization 
            # NR1 = Nri*Nro and form explicitly [ NR1; SYS2*NR1]*inv(Nro)
            C = cr[1:m,:]; D = dr[1:m,:]; 
            if disc
               X, _,f, = try 
                  gared(ar,er,br,D'*D,C'*C,C'*D) 
               catch err
                  findfirst("dichotomic",string(err)) === nothing ? error("$err") : 
                     error("Solution of the DARE failed: Symplectic pencil has eigenvalues on the unit circle")            
               end
               H = cholesky(Hermitian(D'*D+br'*X*br)).U 
            else
               X, _, f, = try 
                  garec(ar,er,br,D'*D,C'*C,C'*D) 
               catch err
                  findfirst("dichotomic",string(err)) === nothing ? error("$err") :
                     error("Solution of the CARE failed: Hamiltonian pencil has jw-axis eigenvalues") 
               end
               H = qr(D).R[1:mr,:]; 
            end
            fnorm = norm(f)
            sysrnull = dss(ar-br*f,er,br/H,cr-dr*f,dr/H,Ts = Ts)
         else
            fnorm = 0
            if stabilize 
               # make poles stable
               if nr > 0
                  f, SF, = saloc(ar, er, br; evals = poles, sdeg = sdeg, disc = disc, fast = fast, atol1 = atol1, atol2 = atol2, atol3 = atol1)
                  sysrnull = dss(SF.S, SF.T, SF.Q'*br, (cr+dr*f)*SF.Z, dr, Ts = Ts)
                  # mul!(ar, br, f, ONE, ONE)
                  # mul!(cr, dr, f, ONE, ONE)
                  fnorm = norm(f)
               else
                  sysrnull = dss(ar,er,br,cr,dr,Ts = Ts)
               end
            else
               sysrnull = dss(ar,er,br,cr,dr,Ts = Ts)
            end
         end
         simple &&  (blkdims = [size(ar,1)])
      end
   else   
      # compute a simple basis  
      sysrnull = dss(ar,er,br,cr,dr,Ts = Ts)
      sysn = dss(zeros(T1,m+p2,0),Ts = Ts) 
      tcond = 1; 
      blkdims = zeros(Int,mr)
      for i = 1:mr
          imt = [[i]; Vector(1:i-1); Vector(i+1:mr)];  # row permutation indices
          sysni, _, info1 = grmcover1(sysrnull[:,imt],1,atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)
          ndegi = order(sysni)
          ndegi == degs[i] || 
               (@warn "Resulting McMillan degree of $i-th vector is $ndegi, expected $(degs[i])")
          tcond = max(tcond, info1.fnorm, info1.tcond)
          if polynomial
             if ndegi > 0
               a1, e1, b1, c1, d1 = dssdata(sysni)
               f, g, SF, = salocinf(a1, e1, b1; fast = fast, atol1 = atol1, atol2 = atol2, atol3 = atol1)
               sysni = dss(SF.S, SF.T, SF.Q'*b1, missing, (c1+d1*f)*SF.Z, d1*g*SF.Z, d1, missing, Ts = Ts, compacted = true, 
                           atol1 = atol1, atol2 = atol2, rtol = rtol)
               # f, g, SF, = salocinf(sysni.A, sysni.E, sysni.B; fast = fast, atol1 = atol1, atol2 = atol2, atol3 = atol1)
               # sysni = dss(SF.S, SF.T, SF.Q'*sysni.B, missing, (sysni.C+sysni.D*f)*SF.Z, sysni.D*g*SF.Z, sysni.D, missing, Ts = Ts)
               #  den = fromroots(eigvals(sysni.A,sysni.E))
               #  T <: Complex || (den = real(den))
               #  sysni = gir(sysni*dss(den,Ts=Ts),atol1=atol1, atol2=atol2, rtol = rtol, fast = fast, infinite = false)
               ndegi = order(sysni)
             end
          else   
             ar,er,br,cr,dr = dssdata(sysni)
             if inner
                C = cr[1:m,:]; D = dr[1:m,:]; 
                if disc
                   X, _,f, = try 
                     gared(ar,er,br,D'*D,C'*C,C'*D) 
                   catch err
                     findfirst("dichotomic",string(err)) === nothing ? error("$err") : 
                        error("Solution of the DARE failed: Symplectic pencil has eigenvalues on the unit circle")            
                   end
                   H = cholesky(Hermitian(D'*D+br'*X*br)).U 
                else
                   X, _, f, = try 
                     garec(ar,er,br,D'*D,C'*C,C'*D) 
                   catch err
                     findfirst("dichotomic",string(err)) === nothing ? error("$err") :
                        error("Solution of the CARE failed: Hamiltonian pencil has jw-axis eigenvalues") 
                   end
                   H = qr(D).R[1:1,1:1] 
                 end
                 fnorm = max(fnorm, norm(f))
                 sysni = dss(ar-br*f,er,br/H,cr-dr*f,dr/H,Ts = Ts) 
              else
                 if stabilize  
                    f, SF, = saloc(ar, er, br; evals = poles, sdeg = sdeg, disc = disc, fast = fast, atol1 = atol1, atol2 = atol2, atol3 = atol1)
                    sysni = dss(SF.S, SF.T, SF.Q'*br, (cr+dr*f)*SF.Z, dr, Ts = Ts)
                    #sysni.A[:,:] = ar + br*f; sysni.C[:,:] = sysni.C + sysni.D*f
                    fnorm = max(fnorm, norm(f))
                 end
              end             
          end
          blkdims[i] = ndegi
          sysn = [sysn sysni]
      end
      tcond > maxtcond && (@warn "Possible loss of numerical stability due to ill-conditioned transformations")
      sysrnull = sysn
   end
   
   info = (nrank = nrank, stdim = simple ? blkdims : νr[νr .> 0], degs = degs, tcond = tcond, fnorm = fnorm)
   return sysrnull, info

   # end GRNULL

end

