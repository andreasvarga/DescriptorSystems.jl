"""
    r = gnrank(sys, fastrank = true, atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ )

Compute the normal rank `r` of the transfer function matrix `G(λ)` of the descriptor system `sys = (A-λE,B,C,D)`. 

The normal rank of `G(λ)` is evaluated as `r = k - n`, where `k` is the normal rank of
the system matrix pencil 

              | A-λE | B | 
      S(λ) := |------|---|
              |  C   | D |  

and `n` is the order of the system `sys`. 

If `fastrank = true`, the normal rank of `S(λ)` is evaluated by counting how many singular values of `S(γ)` have magnitudes 
greater than `max(max(atol1,atol2), rtol*σ₁)`, where `σ₁` is the largest singular value of `S(γ)` and `γ` is a randomly generated value. 
If `fastrank = false`, the rank is evaluated as `nr + ni + nf + nl`, where `nr` and `nl` are the sums of right and left Kronecker indices, 
respectively, while `ni` and `nf` are the number of infinite and finite eigenvalues, respectively. The sums `nr+ni` and  
`nf+nl`, are determined from an appropriate Kronecker-like form of the pencil `S(λ)`, exhibiting the spliting of the right and left structures.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`,  
the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function gnrank(SYS::AbstractDescriptorStateSpace; fastrank = true, atol::Real = 0, atol1::Real = atol, atol2::Real = atol, rtol::Real = 0 ) 
    return sprank(dssdata(SYS)..., atol1 = atol1, atol2 = atol2, rtol = rtol, fastrank = fastrank) - SYS.nx 
end
"""
    val = gzero(sys; fast = false, atol = 0, atol1 = atol, atol2 = atol, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) 

Return for the descriptor system `sys = (A-λE,B,C,D)` the finite and infinite Smith zeros of the system matrix pencil `S(λ)` 

              | A-λE | B | 
       S(λ) = |------|---| .
              |  C   | D |  

The computation of the zeros is performed by reducing the pencil `S(λ)` to an appropriate Kronecker-like form  
using orthonal similarity transformations and involves rank decisions based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations.

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `A`, `B`, `C` and `D`, the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A`, `E`, `B`, `C` and `D`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `S(λ)`, and `ϵ` is the 
working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function gzero(SYS::AbstractDescriptorStateSpace;fast = false, atol::Real = 0, atol1::Real = atol, atol2::Real = atol, 
    rtol::Real = (SYS.nx+1)*eps(real(float(one(real(eltype(SYS.A)))))*iszero(min(atol1,atol2))) ) 
    # pzeros([SYS.A SYS.B; SYS.C SYS.D], [SYS.E zeros(SYS.nx,SYS.nu); zeros(SYS.ny,SYS.nx+SYS.nu)]; fast = fast, atol1 = atol1,
    # atol2 = atol2, rtol = rtol )[1]
    return spzeros(dssdata(SYS)...; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)[1]
end
"""
    val = gpole(sys; fast = false, atol = 0, atol1 = atol, atol2 = atol, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) 

Return for the descriptor system `sys = (A-λE,B,C,D)` the complex vector `val` containing 
the finite and infinite zeros of the system pole pencil `P(λ) = A-λE`. 
The values in `val` are the poles of the transfer function matrix of `sys`, if `A-λE` is _regular_ and the 
descriptor system realization `sys = (A-λE,B,C,D)` is _irreducible_. 
If the pencil `A-λE` is singular, `val` also contains `NaN` elements,
whose number is the rank deficiency of the pencil  `A-λE`.

For `E` nonsingular, `val` contains the generalized eigenvalues of the pair `(A,E)`. 
For `E` singular, `val` contains the zeros of `P(λ)`, which are computed 
by reducing the pencil `P(λ)` to an appropriate Kronecker-like form  
using orthonal similarity transformations and involves rank decisions based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations.

The regularity of `A-λE` is implicitly checked. If `check_reg = true`, an error message is issued if the pencil   
`A-λE` is singular. If `check_reg = false` and the pencil `A-λE` is singular, then `n-r` poles are set to `NaN`, where
`n` is the system order and `r` is the normal rank of `A-λE`. 

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `A`, the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A` and `E`, respectively. 
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function gpole(SYS::DescriptorStateSpace{T}; fast = false, atol::Real = 0, atol1::Real = atol, atol2::Real = atol, rtol::Real = 0, check_reg = false ) where T
    T <: BlasFloat ? T1 = T : T1 = promote_type(Float64,T)
    A = copy_oftype(SYS.A,T1)
    if SYS.E == I
       return isschur(A) ? ordeigvals(A) : MatrixPencils.eigvalsnosort(A)
    else
       E = copy_oftype(SYS.E,T1)
       epsm = eps(float(one(real(T1))))
       isschur(A,E) && rcond(UpperTriangular(E)) >= SYS.nx*epsm && (return ordeigvals(A,E)[1])
       istriu(E) && rcond(UpperTriangular(E)) >= SYS.nx*epsm && (return MatrixPencils.eigvalsnosort(A,E))
       rcond(E) >= SYS.nx*epsm && (return MatrixPencils.eigvalsnosort(A,E))
       # singular E
       poles, nip, krinfo = pzeros(A, E; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol )
       check_reg && (SYS.nx == krinfo.nrank || error("the system has a singular pole pencil"))
       return [poles;NaN*ones(SYS.nx-krinfo.nrank)]
    end
end
"""
    gzerostruct(sys, smarg, fast = false, atol = 0, atol1 = atol, atol2 = atol, 
                rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ, offset = sqrt(ϵ)) -> (val, info) 

Return for the descriptor system `sys = (A-λE,B,C,D)` the complex vector `val` containing 
the finite and infinite Smith zeros of the system matrix pencil `S(λ)` 

              | A-λE | B | 
       S(λ) = |------|---| 
              |  C   | D |  

and the named tuple `info` containing information on the Kronecker structure of the pencil `S(λ)`. 
The values in `val` are the _invariant zeros_ of the pencil `S(λ)` and the _transmission zeros_ of the 
transfer function matrix of `sys` if `A-λE` is _regular_ and the descriptor system realization 
`sys = (A-λE,B,C,D)` is _irreducible_. 

For stability analysis purposes, a stability margin `smarg` can be specified for the finite zeros,
in conjunction with a stability domain boundary offset `β` to numerically assess the  finite zeros 
which belong to the boundary of the stability domain as follows: 
in the continuous-time case, these are the finite zeros having real parts in the interval
`[smarg-β, smarg+β]`, while in the discrete-time case, these are the finite zeros having moduli in
`[smarg-β, smarg+β]`. The default value of the stability margin `smarg` is `0` for a continuous-time system and 
`1` for a discrete-time system. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The named tuple `info` contains the following information:

`info.nfz` is the number of finite eigenvalues of the pencil `S(λ)` (also the number of finite zeros of `sys`);

`info.niev` is the number of infinite eigenvalues of the pencil `S(λ)`;

`info.nisev` is the number of infinite _simple_ eigenvalues of the pencil `S(λ)`; 

`info.niz` is the number of infinite zeros of the system `sys`;

`info.nfsz` is the number of finite stable zeros, i.e., the finite zeros
having real parts or moduli less than `smarg-β` for a continuous- or discrete-time system, respectively;

`info.nfsbz` is the number of finite zeros on the boundary of the 
          stability domain, i.e., the finite zeros
          having real parts or moduli in the interval `[smarg-β, smarg+β]` for a continuous- or discrete-time system, respectively;

`info.nfuz` is the number of finite unstable zeros, i.e., the finite zeros
having real parts or moduli greater than `smarg+β` for a continuous- or discrete-time system, respectively;

`info.nrank` is the normal rank of the pencil `S(λ)`;

`info.miev` is an integer vector, which contains the multiplicities 
          of the infinite eigenvalues of the pencil `S(λ)`  
           (also the dimensions of the elementary infinite blocks in the
          Kronecker form of `S(λ)`);

`info.miz` is an integer vector, which contains the information on the  
           multiplicities of the infinite zeros of `S(λ)` as follows: 
           `S(λ)` has `info.mip[i]` infinite zeros of multiplicity `i`, and 
             is empty if `S(λ)` has no infinite zeros;

`info.rki` is an integer vector, which contains the right Kronecker 
           indices of the pencil `S(λ)` (empty for a regular pencil);

`info.lki` is an integer vector, which contains the left Kronecker 
           indices of the pencil `S(λ)` (empty for a regular pencil);

`info.regular` is set to `true`,  if the pencil `S(λ)` is regular and set to  
`false`, if the pencil `S(λ)` is singular;

`info.stable` is set to `true`, if the pencil `S(λ)` has only stable  
                finite zeros and all its infinite zeros are
                 simple (has only non-dynamic modes), and  is set to `false` otherwise.

_Note:_ The finite zeros and the finite eigenvalues of the pencil
`S(λ)` are the same, but the multiplicities of infinite eigenvalues 
   are in excess with one to the multiplicities of infinite zeros. 

The computation of the zeros is performed by reducing the pencil `S(λ)` to an appropriate Kronecker-like form  
using orthonal similarity transformations and involves rank decisions based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations.

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `A`, `B`, `C` and `D`, the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A`, `E`, `B`, `C` and `D`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `S(λ)`, and `ϵ` is the 
working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function gzeroinfo(SYS::DescriptorStateSpace{T}; smarg::Real = SYS.Ts == 0 ? 0 : 1, fast = false, 
                   atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                   rtol::Real = SYS.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2)), 
                   offset::Real = sqrt(eps(float(real(T)))) ) where T
    val, miz, krinfo = spzeros(dssdata(SYS)...; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)
    nfsz, nfsbz, nfuz = eigvals_info(val[isfinite.(val)], smarg, SYS.Ts != 0, offset)
    niev = sum(krinfo.id)
    nisev = niev == 0 ? 0 : krinfo.id[1]
    niz = sum(miz)
    return val, (nfz = krinfo.nf, niev = niev, nisev = nisev, niz = niz, nfsz = nfsz, nfsbz = nfsbz, 
                 nfuz = nfuz, nrank = krinfo.nrank, miev = krinfo.id, miz = miz, 
                 rki = krinfo.rki, lki = krinfo.lki, regular = (sum(krinfo.rki) == 0 && sum(krinfo.lki) == 0),  
                 stable = (niz == 0 && nfsz == nfz))
end
"""
    gpoleinfo(sys, smarg, fast = false, atol = 0, atol1 = atol, atol2 = atol, 
              rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ, offset = sqrt(ϵ)) -> (val, info) 

Return for the descriptor system `sys = (A-λE,B,C,D)` the complex vector `val` containing 
the finite and infinite zeros of the system pole pencil `P(λ) = A-λE` and the named tuple `info` containing information on 
the eigenvalue structure of the pole pencil `P(λ)`. The values in `val` are the _poles_ of the 
transfer function matrix of `sys`, if `A-λE` is _regular_ and the 
descriptor system realization `sys = (A-λE,B,C,D)` is _irreducible_. 
If the pencil `A-λE` is singular, `val` also contains `NaN` elements,
whose number is the rank deficiency of the pencil  `A-λE`.

For stability analysis purposes, a stability margin `smarg` can be specified for the finite eigenvalues,
in conjunction with a stability domain boundary offset `β` to numerically assess the  finite eigenvalues 
which belong to the boundary of the stability domain as follows: 
in the continuous-time case, these are the finite eigenvalues having real parts in the interval
`[smarg-β, smarg+β]`, while in the discrete-time case, these are the finite eigenvalues having moduli in
`[smarg-β, smarg+β]`. The default value of the stability margin `smarg` is `0` for a continuous-time system and 
`1` for a discrete-time system. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The named tuple `info` contains the following information:

`info.nfev` is the number of finite eigenvalues of the pencil `A-λE` (also the number of finite poles of `sys`);

`info.niev` is the number of infinite eigenvalues of the pencil `A-λE`;

`info.nisev` is the number of infinite _simple_ eigenvalues of the pencil `A-λE`; 

`info.nip` is the number of infinite poles of the system `sys`;

`info.nfsev` is the number of finite stable eigenvalues, i.e., the finite eigenvalues
having real parts or moduli less than `smarg-β` for a continuous- or discrete-time system, respectively;

`info.nfsbev` is the number of finite eigenvalues on the boundary of the 
          stability domain, i.e., the finite eigenvalues
          having real parts or moduli in the interval `[smarg-β, smarg+β]` for a continuous- or discrete-time system, respectively;

`info.nfuev` is the number of finite unstable eigenvalues, i.e., the finite eigenvalues
having real parts or moduli greater than `smarg+β` for a continuous- or discrete-time system, respectively;

`info.nhev` is the number of _hidden_ eigenvalues set to `NaN`
         (can be nonzero only if the pencil `A-λE` is singular);  

`info.nrank` is the normal rank of the pencil `A-λE`;

`info.miev` is an integer vector, which contains the multiplicities 
          of the infinite eigenvalues of the pencil `A-λE`  
           (also the dimensions of the elementary infinite blocks in the
          Kronecker form of `A-λE`);

`info.mip` is an integer vector, which contains the information on the  
           multiplicities of the infinite zeros of `A-λE` as follows: 
           `A-λE` has `info.mip[i]` infinite zeros of multiplicity `i`, and 
             is empty if `A-λE` has no infinite zeros;

`info.rki` is an integer vector, which contains the right Kronecker 
           indices of the pencil `A-λE` (empty for a regular pencil);

`info.lki` is an integer vector, which contains the left Kronecker 
           indices of the pencil `A-λE` (empty for a regular pencil);

`info.regular` is set to `true`,  if the pencil `A-λE` is regular and set to  
`false`, if the pencil `A-λE` is singular;

`info.proper` is set to `true`, if the pencil `A-λE` is regular and all its infinite 
                 eigenvalues are simple (has only non-dynamic modes), or 
                 is set to `false`, if the pencil `A-λE` is singular or has higher order infinite eigenvalues;

`info.stable` is set to `true`, if the pencil `A-λE` is regular, has only stable  
                finite eigenvalues and all its infinite eigenvalues are
                 simple (has only non-dynamic modes), and  is set to `false` otherwise.

For the reduction of the pencil `P(λ)` to an appropriate Kronecker-like form  
orthonal similarity transformations are performed, which involve rank decisions based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations.

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `A`, the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A` and `E`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of `P(λ)`, and `ϵ` is the 
working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function gpoleinfo(SYS::DescriptorStateSpace{T}; smarg::Real = SYS.Ts == 0 ? 0 : 1, fast = false, 
                   atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                   rtol::Real = SYS.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2)), 
                   offset::Real = sqrt(eps(float(real(T))))) where T
    disc = (SYS.Ts != 0)
    n = SYS.nx
    T <: BlasFloat ? T1 = T : T1 = promote_type(Float64,T)
    A = copy_oftype(SYS.A,T1)
    if SYS.E == I
       isschur(A) ? val = ordeigvals(A) : val = MatrixPencils.eigvalsnosort(A)
       nfsev, nfsbev, nfuev = eigvals_info(val, smarg, disc, offset)
       return val, (nfev = n, niev = 0, nisev = 0, nip = 0, nfsev = nfsev, nfsbev = nfsbev, 
                     nfuev = nfuev, nhev = 0, nrank = n, miev = Int[], mip = Int[], 
                     rki = Int[], lki = Int[], regular = true, proper = true, stable = (nfsev == n))
    else
       E = copy_oftype(SYS.E,T1)
       epsm = eps(float(one(real(T))))
       if isschur(A,E) && rcond(UpperTriangular(E)) >= n*epsm
          val = ordeigvals(A,E)[1]
       elseif istriu(E) && rcond(UpperTriangular(E)) >= n*epsm  
          val = MatrixPencils.eigvalsnosort(A,E)
       elseif rcond(E) >= n*epsm 
          val = MatrixPencils.eigvalsnosort(A,E)
       else
          # singular E
          val, mip, krinfo = pzeros(A, E; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol )
       end
       nfsev, nfsbev, nfuev = eigvals_info(val[isfinite.(val)], smarg, disc, offset)
       all(isfinite.(val)) && 
           (return val, (nfev = n, niev = 0, nisev = 0, nip = 0, nfsev = nfsev, nfsbev = nfsbev, 
                        nfuev = nfuev, nhev = 0, nrank = n, miev = Int[], mip = Int[], 
                        rki = Int[], lki = Int[], regular = true, proper = true, stable = (nfsev == n)))
       nhev = n - krinfo.nrank 
       nhev > 0 && (@warn "The system has a singular pole pencil")
       val = [val;NaN*ones(nhev)]
       niev = sum(krinfo.id)
       nip = sum(mip)
       nfev = n-niev
       return val, (nfev = nfev, niev = niev, nisev = count(krinfo.id .== 1), nip = nip, nfsev = nfsev, nfsbev = nfsbev, 
                    nfuev = nfuev, nhev = nhev, nrank = krinfo.nrank, miev = krinfo.id, mip = mip, 
                    rki = krinfo.rki, lki = krinfo.lki, regular = (nhev == 0), proper = (nip == 0), 
                    stable = (nip == 0 && nfsev == nfev))
    end
end
function eigvals_info(val::AbstractVector, smarg::Real, disc::Bool, offset::Real)
    if disc
       nf = count(abs.(val) .< smarg-offset)
       nu = count(abs.(val) .> smarg+offset)
    else
       nf = count(real.(val) .< smarg-offset)
       nu = count(real.(val) .> smarg+offset)
    end
    return nf, length(val)-nf-nu, nu
end
"""
    isregular(sys; atol = 0, atol1 = atol, atol2 = atol, rtol, offset = sqrt(ϵ))

Return `true` if the descriptor system `sys = (A-λE,B,C,D)` has a regular pole pencil `A-λE` and `false` otherwise.  

To test whether the pencil `A-λE` is regular (i.e., `det(A-λE) ̸≡ 0}`),  
the underlying computational procedure reduces the pencil `A-λE` to an appropriate Kronecker-like form, 
which provides information on the rank of `A-λE`. 

The keyword arguements `atol1`, `atol2` and `rtol` specify the absolute tolerance for the nonzero
elements of `A`, the absolute tolerance for the nonzero elements of `E`, and the relative tolerance 
for the nonzero elements of `A` and `E`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of  `A`, and `ϵ` is the 
working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function isregular(SYS::DescriptorStateSpace{T}; atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
    rtol::Real = SYS.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2))) where T
    SYS.E == I && (return true)
    epsm = eps(float(one(real(T))))
    istriu(SYS.E) && rcond(UpperTriangular(SYS.E)) > SYS.nx*epsm && (return true)
    rcond(SYS.E) > SYS.nx*epsm && (return true)
    return MatrixPencils.isregular(SYS.A, SYS.E, atol1 = atol1, atol2 = atol2, rtol = rtol )
end
"""
    isproper(sys; atol = 0, atol1 = atol, atol2 = atol, rtol, fast = true)

Return `true` if the transfer function matrix `G(λ)` of the descriptor system `sys = (A-λE,B,C,D)` is proper
and `false` otherwise.  

For a descriptor system realization `sys = (A-λE,B,C,D)` without uncontrollable and unobservable infinite eigenvalues,
it is checked that the pencil `A-λE` has no infinite eigenvalues or, if infinite eigenvalues exist,
all infinite eigenvalues are simple. If the original descriptor realization has uncontrollable or
unobservable infinite eigenvalues, these are elliminated using orthogonal pencil reduction algorithm. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`,  
the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function isproper(SYS::DescriptorStateSpace{T}; fast::Bool = true, atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
    rtol::Real = SYS.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2))) where T
    (SYS.E == I || SYS.nx == 0)   && (return true)
    epsm = eps(float(one(real(T))))
    istriu(SYS.E) && rcond(UpperTriangular(SYS.E)) > SYS.nx*epsm && 
                     (return true)
    rcond(SYS.E) > SYS.nx*epsm && (return true)
    # check regularity for singular E
    MatrixPencils.isregular(SYS.A, SYS.E, atol1 = atol1, atol2 = atol2, rtol = rtol ) || (return false)   
    # compute a realizations without uncontrollable and unobservable infinite eigenvalues
    A, E, = lsminreal2(SYS.A, SYS.E, SYS.B, SYS.C, SYS.D; 
             fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, infinite = true, 
             contr = true, obs = true, noseig = false) 
    krinfo = pkstruct(A, E; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol )
    return (isempty(krinfo.id) || maximum(krinfo.id) == 1) 
end
"""
    isstable(sys[, smarg]; atol = 0, atol1 = atol, atol2 = atol, rtol, fast = true, offset = sqrt(ϵ))

Return `true` if the descriptor system `sys = (A-λE,B,C,D)` has only stable poles and `false` otherwise.  

It is checked that the pencil `A-λE` has no infinite eigenvalues or, if infinite eigenvalues exist,
all infinite eigenvalues are simple, and additionally the real parts of all finite eigenvalues  are
less than `smarg-β` for a continuous-time system or 
have moduli less than `smarg-β` for a discrete-time system, where `smarg` is the stability margin and 
`β` is the stability domain boundary offset. 
The default value of the stability margin `smarg` is `0` for a continuous-time system and 
`1` for a discrete-time system.
The offset  `β` to be used to numerically assess the stability of eigenvalues 
can be specified via the keyword parameter `offset = β`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

For `E` singular, the computation of the poles is performed by reducing the pencil `P(λ)` to an appropriate Kronecker-like form  
using orthonal similarity transformations and involves rank decisions based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`,  
the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function isstable(SYS::DescriptorStateSpace{T}, smarg::Real = SYS.Ts == 0 ? 0 : 1; 
                  fast = false, atol::Real = 0, atol1::Real = atol, atol2::Real = atol, 
                  rtol::Real = SYS.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2)), 
                  offset::Real = sqrt(eps(float(real(T))))) where T
    disc = (SYS.Ts != 0)
    β = abs(offset); 
    if SYS.E == I
       isschur(SYS.A) ? poles = ordeigvals(SYS.A) : poles = eigvals(SYS.A)
    else
       poles = gpole(SYS; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)
       (any(isinf.(poles)) || any(isnan.(poles)))  && (return false)
    end
    return disc ? all(abs.(poles) .< smarg-β) : all(real.(poles) .< smarg-β)
end
"""
    ghanorm(sys, fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol) -> (hanorm, hs)

Compute for a proper and stable descriptor system `sys = (A-λE,B,C,D)` with the transfer function
matrix `G(λ)`, the Hankel norm ``||G(\\lambda)||_H`` and the Hankel singular values of the system.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`,  
the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol` and `atol2 = atol`. 

If `E` is singular, the uncontrollable infinite eigenvalues of the pair `(A,E)` and
the non-dynamic modes are elliminated using minimal realization techniques.
The rank determinations in the performed reductions
are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`. 
"""   
function ghanorm(sys::DescriptorStateSpace{T}; fast::Bool = true, 
                 atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                 rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2)))  where T 
    
    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    n = size(sys.A,1)  

    n == 0 && (return zero(real(T1)), zeros(real(T1),0))

    s2eps = sqrt(eps(real(T1)))       
    disc = !iszero(sys.Ts)
    
    
    if  sys.E == I
        # for a non-dynamic system, we set the Hankel norm to zero,
        # but the Hankel singular values are empty
        size(sys.A,1) == 0 && (return zero(real(T1)), zeros(real(T1),0))
        # reduce the system to Schur coordinate form
        SF = schur(sys.A)
        # check stability
        ((disc && maximum(abs.(SF.values)) >= 1-s2eps) || (!disc && maximum(real(SF.values)) >= -s2eps)) &&
              error("The system sys is unstable")
        S = plyaps(SF.T, SF.Z'*sys.B; disc = disc)
        R = plyaps(SF.T', (sys.C*SF.Z)'; disc = disc)
        hs = svdvals(R*S)
     else
        # eliminate non-dynamic modes if possible
        if rcond(sys.E) < n*eps(float(real(T1)))
           # sys = gss2ss(sys,s2eps,'triu');
           sys = gminreal(sys, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol);
           rcond(sys.E) < n*eps(float(real(T1))) && error("The system SYS is not proper")
        end
        # for a non-dynamic system, we set the Hankel norm to zero,
        # but the Hankel singular values are empty
        size(sys.A,1) == 0 && (return zero(real(T1)), zeros(real(T1),0))
        # reduce the system to generalized Schur coordinate form
        SF = schur(sys.A,sys.E)
        ((disc && maximum(abs.(SF.values)) >= 1-s2eps) || (!disc && maximum(real(SF.values)) >= -s2eps)) &&
              error("The system sys is unstable") 
        S = plyaps(SF.S, SF.T, SF.Q'*sys.B; disc = disc)
        R = plyaps(SF.S', SF.T', (sys.C*SF.Z)'; disc = disc)
        hs = svdvals(R*UpperTriangular(SF.T)*S)
    end

    return hs[1], hs
    # end GHANORM
end
"""
    gh2norm(sys, fast = true, offset = sqrt(ϵ), atol = 0, atol1 = atol, atol2 = atol, atol3 = atol, atolinf = atol, rtol) -> h2norm

Compute for a descriptor system `sys = (A-λE,B,C,D)` the `H2` norm of its transfer function  matrix `G(λ)`.
The `H2` norm is infinite, if `sys` has unstable poles, or, for a continuous-time, the system has nonzero gain at infinity.
To check the stability, the eigenvalues of the _pole pencil_ `A-λE` must have real parts less 
than `-β` for a continuous-time system or 
have moduli less than `1-β` for a discrete-time system, where `β` is the stability domain boundary offset.
The offset  `β` to be used can be specified via the keyword parameter `offset = β`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

For a continuous-time system `sys` with `E` singular, a reduced order realization is determined first, without 
uncontrollable and unobservable nonzero finite and infinite eigenvalues of the corresponding pole pencil. 
The rank determinations in the performed reductions
are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`,  
the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The keyword argument `atolinf` is the absolute tolerance for the gain of `G(λ)` at infinity. 
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol` and `atol2 = atol`.  
"""   
function gh2norm(sys::DescriptorStateSpace{T}; fast::Bool = true, offset::Real = sqrt(eps(float(real(T)))), 
                 atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, atolinf::Real = atol, 
                 rtol::Real = (size(sys.A,1)*eps(real(float(one(T)))))*iszero(min(atol1,atol2)))  where T 
    return gl2norm(sys; h2norm = true, fast = fast, offset = offset, atol1 = atol1, atol2 = atol2, atolinf = atolinf, rtol = rtol)
    
end
"""
    gl2norm(sys, h2norm = false, fast = true, offset = sqrt(ϵ), atol = 0, atol1 = atol, atol2 = atol, atol3 = atol, atolinf = atol, rtol) -> l2norm

Compute for a descriptor system `sys = (A-λE,B,C,D)` the `L2` norm of its transfer function  matrix `G(λ)`.
The `L2` norm is infinite if the _pole pencil_ `A-λE` has
zeros (i.e., poles) on the stability domain boundary, i.e., on the extended imaginary axis, in the continuous-time case, 
or on the unit circle, in the discrete-time case. 
The `L2` norm is also infinite for a continuous-time system having nonzero gain at infinity. 
To check the lack of poles on the stability domain boundary, the eigenvalues of the pencil `A-λE` 
must not have real parts in the interval `[-β,β]` for a continuous-time system or 
must not have moduli in the interval `[1-β,1+β]` for a discrete-time system, where `β` is the stability domain boundary offset.  
The offset  `β` to be used can be specified via the keyword parameter `offset = β`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

If `h2norm = true`, the `H2` norm is computed. 
The `H2` norm is infinite if the _pole pencil_ `A-λE` has unstable zeros (i.e., unstable poles). 
To check the stability, the eigenvalues of the pencil `A-λE` must have real parts less than `-β` for a continuous-time system or 
have moduli less than `1-β` for a discrete-time system. 

For a continuous-time system `sys` with `E` singular, a reduced order realization is determined first, without 
uncontrollable and unobservable nonzero finite and infinite eigenvalues of the corresponding pole pencil. 
The rank determinations in the performed reductions
are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, `B`, `C`, `D`,  
the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.  
The keyword argument `atol3` specifies the absolute tolerance for the nonzero elements of B`
and is only used if `h2norm = false` for controllability test of eigenvalues.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the working machine epsilon 
and `n` is the order of the system `sys`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""   
function gl2norm(sys::DescriptorStateSpace{T}; h2norm::Bool = false, fast::Bool = true, 
                 offset::Real = sqrt(eps(float(real(T)))), atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, atol3::Real = atol, 
                 atolinf::Real = atol, rtol::Real = (size(sys.A,1)*eps(real(float(one(T)))))*iszero(min(atol1,atol2)))  where T 
    
    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    disc = !iszero(sys.Ts)
       
    if  sys.E == I
        # quick return for a non-dynamic system or continuous-time system with nonzero D
        size(sys.A,1) == 0 && (return disc ? norm(sys.D) : (iszero(sys.D) ? zero(real(T1)) : Inf))
        disc || norm(sys.D) <= atolinf || (return Inf)
        if h2norm
            # compute the H2-norm
            # reduce the system to Schur coordinate form
            SF = schur(sys.A)
            # check stability
            ((disc && maximum(abs.(SF.values)) >= 1-offset) || (!disc && maximum(real(SF.values)) >= -offset)) && (return Inf)
            R = plyaps(SF.T', (sys.C*SF.Z)'; disc = disc)
            return disc ? norm([R*(SF.Z'*sys.B); sys.D]) : norm(R*(SF.Z'*sys.B))
        else
            # compute the L2-norm
            try
              sys = grcfid(sys, offset = offset, atol1 = atol1, atol3 = atol3, rtol = rtol)[1]
              R = plyaps(sys.A', sys.C'; disc = disc)
              return disc ? norm([R*sys.B; sys.D]) : norm(R*sys.B)
            catch
              return Inf
            end
        end
     else
        n = sys.nx
        # eliminate uncontrollable and unobservable infinite eigenvalues and non-dynamic modes if possible
        if rcond(sys.E) < n*eps(float(real(T1)))
           sys = gir(sys, finite = false, noseig = true, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)
           # check properness for a continuous-time system
           disc || rcond(sys.E) >= n*eps(float(real(T1))) || (return Inf)
        end
        size(sys.A,1) == 0 && (return disc ? norm(sys.D) : (iszero(sys.D) ? zero(real(T1)) : Inf))
        disc || norm(sys.D) <= atolinf || (return Inf)
        if h2norm
            # compute the H2-norm
            # reduce the system to generalized Schur coordinate form
            SF = schur(sys.A,sys.E)
            # check stability
            ((disc && maximum(abs.(SF.values)) >= 1-offset) || (!disc && maximum(real(SF.values)) >= -offset)) && (return Inf)
            R = plyaps(SF.S', SF.T', (sys.C*SF.Z)'; disc = disc)
            return disc ? norm([R*(SF.Q'*sys.B); sys.D]) : norm(R*(SF.Q'*sys.B))
        else
            # compute the L2-norm
            try
              sys = glcfid(sys, offset = offset, mininf = true, atol1 = atol1, atol2 = atol2, atol3 = atol3, rtol = rtol)[1]
              R = plyaps(sys.A', sys.E', sys.C'; disc = disc)
              return disc ? norm([R*sys.B; sys.D]) : norm(R*sys.B)
            catch
              return Inf
            end
        end
    end

    # end GL2NORM
end
"""
    ghinfnorm(sys, rtolinf = 0.001, fast = true, offset = sqrt(ϵ), atol = 0, atol1 = atol, atol2 = atol, rtol) -> (hinfnorm, fpeak)

Compute for a descriptor system `sys = (A-λE,B,C,D)` with the transfer function  matrix `G(λ)` 
the `H∞` norm `hinfnorm` (i.e.,  the peak gain of `G(λ)`) and 
the corresponding peak frequency `fpeak`, where the peak gain is achieved. 
The `H∞` norm is infinite if the _pole pencil_ `A-λE` has unstable zeros (i.e., unstable poles). 
To check the stability, the eigenvalues of the pencil `A-λE` must have real parts less than `-β` for a continuous-time system or 
have moduli less than `1-β` for a discrete-time system, where `β` is the stability domain boundary offset.
The offset  `β` to be used can be specified via the keyword parameter `offset = β`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The keyword argument `rtolinf` specifies the relative accuracy for the computed infinity norm. 
The  default value used for `rtolinf` is `0.001`.

For a continuous-time system `sys` with `E` singular, a reduced order realization is determined first, without 
uncontrollable and unobservable nonzero finite and infinite eigenvalues of the corresponding pole pencil. 
The rank determinations in the performed reductions
are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""   
function ghinfnorm(sys::DescriptorStateSpace{T}; rtolinf::Real = real(T)(0.001), fast::Bool = true, offset::Real = sqrt(eps(float(real(T)))), 
                   atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol,  
                   rtol::Real = (size(sys.A,1)*eps(real(float(one(T)))))*iszero(min(atol1,atol2)))  where T 
    return glinfnorm(sys; hinfnorm = true, rtolinf = rtolinf, fast = fast, offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol)

end
"""
    glinfnorm(sys, hinfnorm = false, rtolinf = 0.001, fast = true, offset = sqrt(ϵ), atol = 0, atol1 = atol, atol2 = atol, rtol) -> (linfnorm, fpeak)

Compute for a descriptor system `sys = (A-λE,B,C,D)` with the transfer function  matrix `G(λ)` 
the `L∞` norm `linfnorm` (i.e.,  the peak gain of `G(λ)`) and 
the corresponding peak frequency `fpeak`, where the peak gain is achieved. 
The `L∞` norm is infinite if the _pole pencil_ `A-λE` has
zeros (i.e., poles) on the stability domain boundary, i.e., on the extended imaginary axis, in the continuous-time case, 
or on the unit circle, in the discrete-time case.  
To check the lack of poles on the stability domain boundary, the eigenvalues of the pencil `A-λE` 
must not have real parts in the interval `[-β,β]` for a continuous-time system or 
must not have moduli in the interval `[1-β,1+β]` for a discrete-time system, where `β` is the stability domain boundary offset.  
The offset  `β` to be used can be specified via the keyword parameter `offset = β`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The keyword argument `rtolinf` specifies the relative accuracy for the computed infinity norm. 
The  default value used for `rtolinf` is `0.001`.

If `hinfnorm = true`, the `H∞` norm is computed. In this case, the stability of the zeros of `A-λE` is additionally checked and the
the `H∞` norm is infinite for an unstable system.
To check the stability, the eigenvalues of the pencil `A-λE` must have real parts less than `-β` for a continuous-time system or 
have moduli less than `1-β` for a discrete-time system.

For a continuous-time system `sys` with `E` singular, a reduced order realization is determined first, without 
uncontrollable and unobservable nonzero finite and infinite eigenvalues of the corresponding pole pencil. 
The rank determinations in the performed reductions
are based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""   
function glinfnorm(sys::DescriptorStateSpace{T}; hinfnorm::Bool = false, rtolinf::Real = real(T)(0.001), fast::Bool = true, 
                   offset::Real = sqrt(eps(float(real(T)))), atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol,  
                   rtol::Real = sys.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2)))  where T 
    
    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    β = abs(offset)
    Ts = abs(sys.Ts)
    disc = !iszero(Ts)

    # quick exit in constant case  
    sys.nx == 0 && (return opnorm(sys.D), real(T1)(0))

    # eliminate simple infinite eigenvalues in the continuous-time case with singular E
    if disc || sys.E == I || rcond(sys.E) >= sys.nx*eps(float(real(T1)))
       A, E, B, C, D = dssdata(T1,sys)
    else
       A, E, B, C, D = dssdata(gir(sys, fast = fast, finite = false, noseig = true, atol1 = atol1, atol2 = atol2, rtol = rtol))
    end

    n = size(A,1)

    # quick exit in constant case  
    n == 0 && (return opnorm(D), real(T1)(0))

    # check properness in continuous-time case
    disc || E == I || rcond(E) >= n*eps(float(real(T1))) || (return Inf, Inf)
    
    E == I ? ft = eigvals(A) : ft = eigvals(A,E); ft = ft[isfinite.(ft)]
    if disc
        hinfnorm && all(abs.(ft) .> 1-β) && (return Inf, NaN)
        for i = 1:length(ft)
            abs(ft[i]) >= 1-β && abs(ft[i]) <= 1+β && (return Inf, abs(log(ft[i])/Ts))
        end
    else
        hinfnorm && all(real.(ft) .> -β) && (return Inf, NaN)
        for i = 1:length(ft)
            real(ft[i]) >= -β && real(ft[i]) <= β && (return Inf, abs(imag(ft[i])))
        end
    end
    
    disc ? (return norminfd(A, E, B, C, D, ft, Ts, rtolinf)) : 
           (return norminfc(A, E, B, C, D, ft, rtolinf))
    # end GLINFNORM
end
function norminfc(a, e, b, c, d, ft0, tol)
    # Continuous-time L∞ norm computation
    # It is assumed that A-λE has no eigenvalues on the extended imaginary axis

    # Tolerance for jw-axis mode detection
    T = eltype(a)
    TR = real(T)
    epsm = eps(TR)
    toljw1 = 100 * epsm;       # for simple roots
    toljw2 = 10 * sqrt(epsm);  # for double root
    
    # Problem dimensions
    nx = size(a,1)
    ny, nu = size(d)
    desc = e != I
    
    if desc
       # Reduce (A,E) to (generalized) upper-Hessenberg form for
       # fast frequency response computation 
       at = copy(a)
       et = copy(e)
       bt = copy(b)
       ct = copy(c)
       # first reduce E to upper triangular form 
       _, tau = LinearAlgebra.LAPACK.geqrf!(et)
       T <: Complex ? tran = 'C' : tran = 'T'
       LinearAlgebra.LAPACK.ormqr!('L', tran, et, tau, at)
       LinearAlgebra.LAPACK.ormqr!('L', tran, et, tau, bt)
       # reduce A to Hessenberg form and keep E upper triangular
       _, _, Q, Z = MatrixPencils.gghrd!('I', 'I',  1, nx, at, et, similar(at),similar(et))
       if T <: Complex 
          bc = Q'*bt; cc = c*Z; ac = at; ec = et; dc = d;
       else
          bc = complex(Q'*bt); cc = complex(c*Z); ac = complex(at); ec = complex(et); dc = complex(d);
       end
    else
       # Reduce A to upper Hessenberg form for
       # fast frequency response computation 
       Ha = hessenberg(a)
       ac = Ha.H
       if T <: Complex 
          bc = Ha.Q'*b; cc = c*Ha.Q; dc = d; 
       else
          bc = complex(Ha.Q'*b); cc = complex(c*Ha.Q); dc = complex(d);
       end
    end
   
    # Build a new vector TESTFRQ of test frequencies containing the peaking
    # frequency for each mode (or an approximation thereof for non-resonant modes).
    # Add frequency w = 0 and set GMIN = || D || and FPEAK to infinity
    ar2 = abs.(real(ft0));  # magnitudes of real parts of test frequencies
    w0 = abs.(ft0);         # fundamental frequencies
    
    ikeep = (imag.(ft0) .>= 0) .& ( w0 .> 0)
    offset2 = max.(0.25,max.(1 .- 2 .*(ar2[ikeep]./w0[ikeep]).^2))
    testfrq = [[0]; w0[ikeep].*sqrt.(offset2)]
    gmin = opnorm(d)
    fpeak = Inf

    if VERSION < v"1.3.0-DEV.349"
       # Compute lower estimate GMIN as max. gain over the selected frequencies
       ac = complex(ac)
       for i = 1:length(testfrq)
           w = testfrq[i];
           desc ? gw = opnorm(dc+cc*(((im*w)*ec-ac)\bc)) : gw = opnorm(dc+cc*(((im*w)*I-ac)\bc)) 
           gw > gmin && (gmin = gw;  fpeak = w)
       end
    else  
       # Compute lower estimate GMIN as max. gain over the selected frequencies
       for i = 1:length(testfrq)
         w = testfrq[i];
         bct = copy(bc)
         desc ? ldiv!(UpperHessenberg(ac-(im*w)*ec),bct) : ldiv!(ac,bct,shift = -im*w)
         gw = opnorm(dc-cc*bct)
         gw > gmin && (gmin = gw;  fpeak = w)
       end
    end
    gmin == 0 && (return TR(0), TR(0))
    
    # modified gamma iterations (Bruinsma-Steinbuch algorithm) start:
    iter = 1;
    while iter < 30
       # Test if G = (1+TOL)*GMIN qualifies as upper bound
       g = (1+tol) * gmin;
       # Compute finite eigenvalues of Hamiltonian pencil 
       # deflate nu+ny simple infinite eigenvalues
       h1 = [g*I d; d' g*I; zeros(T,nx,ny) b ; c' zeros(T,nx,nu)]
       h2 = [c zeros(T,ny,nx) ; zeros(T,nu,nx) -b'; a zeros(T,nx,nx) ; zeros(T,nx,nx) -a']
       j2 = [zeros(T,ny+nu,2*nx); e zeros(T,nx,nx) ; zeros(T,nx,nx) e']
       _, tau = LinearAlgebra.LAPACK.geqrf!(h1)
       T <: Complex ? tran = 'C' : tran = 'T'
       LinearAlgebra.LAPACK.ormqr!('L',tran,h1,tau,h2)
       LinearAlgebra.LAPACK.ormqr!('L',tran,h1,tau,j2)
       i2 = ny+nu+1:ny+nu+2*nx
       heigs = eigvals!(view(h2,i2,:),view(j2,i2,:))
  
       mag = abs.(heigs);
       # Detect jw-axis modes.  Test is based on a round-off level of
       # eps*rho(H) resulting in worst-case
       # perturbations of order sqrt(eps*rho(H)) on the real part
       # of poles of multiplicity two (typical as g->norm(sys,inf))
       #jweig = heigs(abs(real(heigs)) < toljw2*(1 .+ mag)+toljw1*max(mag));
       jweig = heigs[abs.(real(heigs)) .< toljw2*(1 .+ mag) .+ toljw1*mag];
    
       # Compute frequencies where gain G is attained and
       # generate new test frequencies
       ws = imag(jweig);
       ws = unique(max.(epsm,ws[ws.> 0]))
       lws0 = length(ws);
       if lws0 == 0
          # No jw-axis eigenvalues for G = GMIN*(1+TOL): we're done
          return gmin, fpeak
       else
          lws0 == 1 && (ws = [ws;ws]) # correct pairing
          lws = length(ws);
       end
             
       # Form the vector of mid-points and compute
       # gain at new test frequencies
       gmin0 = gmin;   # save current lower bound
       ws = sqrt.(ws[1:lws-1].*ws[2:lws])
       if VERSION < v"1.3.0-DEV.349"
         # Compute lower estimate GMIN as max. gain over the selected frequencies
         for i = 1:lws-1
            w = ws[i]
            desc ? gw = opnorm(dc+cc*(((im*w)*ec-ac)\bc)) : gw = opnorm(dc+cc*(((im*w)*I-ac)\bc)) 
            gw > gmin && (gmin = gw;  fpeak = w)
         end
       else  
         # Compute lower estimate GMIN as max. gain over the selected frequencies
         for i = 1:lws-1
             w = ws[i]
             bct = copy(bc)
             desc ? ldiv!(UpperHessenberg(ac-(im*w)*ec),bct) : ldiv!(ac,bct,shift = -im*w)
             gw = opnorm(dc-cc*bct)
             gw > gmin && (gmin = gw;  fpeak = w)
         end
       end
      
       # If lower bound has not improved, exit (safeguard against undetected
       # jw-axis modes of Hamiltonian matrix)
       (lws0 < 2 || gmin < gmin0*(1+tol/10)) && (return gmin, fpeak)
       iter += 1
    end #while  
end  
function norminfd(a, e, b, c, d, ft0, Ts, tol)
    # Discrete-time L∞ norm computation
    # It is assumed that A-λE has no eigenvalues on the unit circle

    # Tolerance for detection of unit circle modes
    T = eltype(a)
    TR = real(T)
    epsm = eps(TR)
    toluc1 = 100 * epsm       # for simple roots
    toluc2 = 10 * sqrt(epsm)  # for double root
    
    # Problem dimensions
    nx = size(a,1);
    ny, nu = size(d)
    desc = e != I
       
    if desc
       # Reduce (A,E) to (generalized) upper-Hessenberg form for
       # fast frequency response computation 
       at = copy(a)
       et = copy(e)
       bt = copy(b)
       ct = copy(c)
       # first reduce E to upper triangular form 
       _, tau = LinearAlgebra.LAPACK.geqrf!(et)
       T <: Complex ? tran = 'C' : tran = 'T'
       LinearAlgebra.LAPACK.ormqr!('L', tran, et, tau, at)
       LinearAlgebra.LAPACK.ormqr!('L', tran, et, tau, bt)
       # reduce A to Hessenberg form and keep E upper triangular
       _, _, Q, Z = MatrixPencils.gghrd!('I', 'I',  1, nx, at, et, similar(at),similar(et))
       if T <: Complex 
          bc = Q'*bt; cc = c*Z; ac = at; ec = et; dc = d;
       else
          bc = complex(Q'*bt); cc = complex(c*Z); ac = complex(at); ec = complex(et); dc = complex(d);
       end
    else
       # Reduce A to upper Hessenberg form for
       # fast frequency response computation 
       Ha = hessenberg(a)
       ac = Ha.H
       if T <: Complex 
          bc = Ha.Q'*b; cc = c*Ha.Q; dc = d; 
       else
          bc = complex(Ha.Q'*b); cc = complex(c*Ha.Q); dc = complex(d);
       end
    end
    
    # Build a new vector TESTFRQ of test frequencies containing the peaking
    # frequency for each mode (or an approximation thereof for non-resonant modes).
    sr = log.(complex(ft0[(ft0 .!= 0) .& (abs.(ft0) .<= pi/Ts)]));   # equivalent jw-axis modes
    #sr = ft0[(ft0 .!= 0) .& (abs.(ft0) .<= pi/Ts)];   # equivalent jw-axis modes
    asr2 = abs.(real(sr))   # magnitude of real parts
    w0 = abs.(sr);           # fundamental frequencies

    ikeep = (imag.(sr) .>= 0) .& ( w0 .> 0)
    testfrq = w0[ikeep].*sqrt.(max.(0.25,1 .- 2 .*(asr2[ikeep]./w0[ikeep]).^2))
    
    # Back to unit circle, and add z = exp(0) and z = exp(pi)
    testz = [exp.(im*testfrq); [-1;1] ]
   
    gmin = 0
    fpeak = 0
    
    # Compute lower estimate GMIN as max. gain over test frequencies
    if VERSION < v"1.3.0-DEV.349"
       # Compute lower estimate GMIN as max. gain over the selected frequencies
       ac = complex(ac)
       for i = 1:length(testz)
           z = testz[i]
           desc ? gw = opnorm(dc+cc*((z*ec-ac)\bc)) : gw = opnorm(dc+cc*((z*I-ac)\bc)) 
           gw > gmin && (gmin = gw;  fpeak = abs(angle(z)))
       end
    else  
      # Compute lower estimate GMIN as max. gain over the selected frequencies
      for i = 1:length(testz)
          z = testz[i]
          bct = copy(bc)
          desc ? ldiv!(UpperHessenberg(ac-z*ec),bct) : ldiv!(ac,bct,shift = -z)
          gw = opnorm(dc-cc*bct)
          gw > gmin && (gmin = gw;  fpeak = abs(angle(z)))
      end
    end
    gmin == 0 && (return TR(0), TR(0))
    
    # Modified gamma iterations (Bruinsma-Steinbuch algorithm) starts:
    iter = 1;
    while iter < 30
       # Test if G = (1+TOL)*GMIN qualifies as upper bound
       g = (1+tol) * gmin;
       # Compute the finite eigenvalues of the symplectic pencil
       # deflate nu+ny simple infinite eigenvalues
       h1 = [a zeros(T,nx,nx+ny) b; zeros(T,nx,nx) e' zeros(nx,ny+nu)]
       h2 = [c zeros(T,ny,nx) g*I d; zeros(T,nu,nx) b' d' g*I]
       j1 = [e zeros(T,nx,nx+ny+nu); zeros(T,nx,nx) a' c' zeros(T,nx,nu) ]
       _, tau = LinearAlgebra.LAPACK.gerqf!(h2)
       T <: Complex ? tran = 'C' : tran = 'T'
       LinearAlgebra.LAPACK.ormrq!('R',tran,h2,tau,h1)
       LinearAlgebra.LAPACK.ormrq!('R',tran,h2,tau,j1)
       i1 = 1:(2*nx)
       heigs = eigvals!(view(h1,:,i1),view(j1,:,i1))
       heigs =  heigs[abs.(heigs) .< 1/toluc2]

       # Detect unit-circle eigenvalues
       mag = abs.(heigs)
       uceig = heigs[abs.(1 .- mag) .< toluc2 .+ toluc1*mag]
    
       # Compute frequencies where gain G is attained and
       # generate new test frequencies
       ang = angle.(uceig);
       ang = unique(max.(epsm,ang[ang .> 0]))
       lan0 = length(ang);
       if lan0 == 0
          # No unit-circle eigenvalues for G = GMIN*(1+TOL): we're done
          return gmin, fpeak/Ts
        else
          lan0 == 1 && (ang = [ang;ang])   # correct pairing
          lan = length(ang)
       end
    
       # Form the vector of mid-points and compute
       # gain at new test frequencies
       gmin0 = gmin;   # save current lower bound
       testz = exp.(im*((ang[1:lan-1]+ang[2:lan])/2))
       if VERSION < v"1.3.0-DEV.349"
         # Compute lower estimate GMIN as max. gain over the selected frequencies
         for i = 1:lan-1
             z = testz[i]
             desc ? gw = opnorm(dc+cc*((z*ec-ac)\bc)) : gw = opnorm(dc+cc*((z*I-ac)\bc)) 
             gw > gmin && (gmin = gw;  fpeak = abs(angle(z)))
         end
      else  
         # Compute lower estimate GMIN as max. gain over the selected frequencies
         for i = 1:lan-1
            z = testz[i]
            bct = copy(bc)
            desc ? ldiv!(UpperHessenberg(ac-z*ec),bct) : ldiv!(ac,bct,shift = -z)
            gw = opnorm(dc-cc*bct)
            gw > gmin && (gmin = gw;  fpeak = abs(angle(z)))
         end
       end
        
       # If lower bound has not improved, exit (safeguard against undetected
       # unit-circle eigenvalues).
       (lan0 < 2 || gmin < gmin0*(1+tol/10)) && (return gmin, fpeak/Ts)
       iter += 1
    end
end    