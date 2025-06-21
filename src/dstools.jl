"""

     H = freqresp(sys, ω)

Compute the frequency response `H` of the descriptor system `sys = (A-λE,B,C,D)` at the real frequencies `ω`. 
    
For a system with `p` outputs and `m` inputs, and for `N` frequency values in `ω`,
`H` is a `p × m × N` array, where `H[:,:,i]` contains the `i`-th value of the frequency response computed as
`C*((w[i]*E - A)^-1)*B + D`, where `w[i] = im*ω[i]` for a continuous-time system and `w[i] = exp(im*ω[i]*|Ts|)` 
for a discrete-time system with sampling time `Ts`. 

_Method:_ For an efficient evaluation of `C*((w[i]*E - A)^-1)*B + D` for many values of `w[i]`, a preliminary 
orthogonal coordinate transformation is performed such that for the input-output equivalent transformed 
system `sys = (At-λEt,Bt,Ct,Dt)`, the matrix `w[i]*Et - At` is upper Hessenberg. 
This allows an efficient computation of the frequency response using the Hessenberg-form based 
approach proposed in [1].

_Reference:_

[1] Laub, A.J., "Efficient Multivariable Frequency Response Computations",
    IEEE Transactions on Automatic Control, AC-26 (1981), pp. 407-408.
"""
function freqresp(sys::DescriptorStateSpace{T}, ω::Union{AbstractVector{<:Real},Real}) where T
    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    ONE = one(T1)
    typeof(ω) <: Real && (ω = [ω])
    N = length(ω)    
    a, e, b, c, d = dssdata(T1,sys)
    if isempty(a)
        # overcome bug in orghr! for null dimension #43680 
        H = similar(d, eltype(d), sys.ny, sys.nu, N)
        for i = 1:N
           copyto!(view(H,:,:,i),d)
       end
       return complex(H)
    end

    Ts = abs(sys.Ts)
    disc = !iszero(Ts)
    # create the complex frequency scaling
    sw = disc ? im*abs(sys.Ts) : im
    desc = !(e == I)
    # reduce to complex Hessenberg form
    ac, ec, bc, cc, dc = chess(a, e, b, c, d)
    H = similar(dc, eltype(dc), sys.ny, sys.nu, N)
    bct = similar(bc) 
    for i = 1:N
        if isinf(ω[i])
           # exceptional call to evalfr
           H[:,:,i] = evalfr(sys, fval=Inf)
        else
           Hi = view(H,:,:,i)
           copyto!(Hi,dc)
           copyto!(bct,bc)
           w = disc ? -exp(sw*ω[i]) : -sw*ω[i]
           desc ? ldiv!(UpperHessenberg(ac+w*ec),bct) : ldiv!(ac,bct,shift = w)
           mul!(Hi,cc,bct,-ONE,ONE)
        end
    end
    return H
end
function chess(a::MT, e::Union{MT,UniformScaling}, b::MT, c::MT, d::MT) where {T <: BlasFloat, MT<: AbstractMatrix{T}}
    # reduce the descriptor system (A-λE,B,C,D) to an equivalent complex descriptor system 
    # (Ac-λEc,Bc,Cc,Dc) such that Ac-λEc is in an upper Hessenberg form 
    # Note: This function is only used in the context of efficient frequency response computations. 
    isempty(a) && (return complex(a), e == I ? e : complex(e), complex(b), complex(c), complex(d))
    desc = !(e == I)
    if desc
        # Reduce (A,E) to (generalized) upper-Hessenberg form for
        # fast frequency response computation 
        at = copy(a)
        et = copy(e)
        bt = copy(b)
        # first reduce E to upper triangular form 
        _, tau = LinearAlgebra.LAPACK.geqrf!(et)
        T <: Complex ? tran = 'C' : tran = 'T'
        LinearAlgebra.LAPACK.ormqr!('L', tran, et, tau, at)
        LinearAlgebra.LAPACK.ormqr!('L', tran, et, tau, bt)
        # reduce A to Hessenberg form and keep E upper triangular
        _, _, Q, Z = MatrixPencils.gghrd!('I', 'I',  1, size(at,1), at, et, similar(at),similar(et))
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
        ec = e
    end
    return ac, ec, bc, cc, dc
end
"""

     H = freqresp(sys::SparseDescriptorStateSpace, ω)

Compute the frequency response `H` of the sparse descriptor system `sys = (A-λE,B,C,D)` at the real frequencies `ω`. 
    
For a system with `p` outputs and `m` inputs, and for `N` frequency values in `ω`,
`H` is a `p × m × N` array, where `H[:,:,i]` contains the `i`-th value of the frequency response computed as
`C*((w[i]*E - A)^-1)*B + D`, where `w[i] = im*ω[i]` for a continuous-time system and `w[i] = exp(im*ω[i]*|Ts|)` 
for a discrete-time system with sampling time `Ts`. 

The function `freqresp` can be also used if `sys` has the type `DescriptorStateSpaceExt`.

Parallel computation of the frequency response matrices can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  
"""

function freqresp(sys::Union{SparseDescriptorStateSpace{T},DescriptorStateSpaceExt{T}}, ω::Union{AbstractVector{<:Real},Real}) where T
    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    ONE = one(T1)
    typeof(ω) <: Real && (ω = [ω])
    N = length(ω)    
    a, e, b, c, d = dssdata(T1,sys)
    if isempty(a)
        # overcome bug in orghr! for null dimension #43680 
        dmat = Matrix(d)
        H = similar(dmat, eltype(dmat), sys.ny, sys.nu, N)
        for i = 1:N
           copyto!(view(H,:,:,i),dmat)
       end
       return complex(H)
    end

    Ts = abs(sys.Ts)
    disc = !iszero(Ts)
    # create the complex frequency scaling
    sw = disc ? im*abs(sys.Ts) : im
    dc = complex(Matrix(d))
    H = similar(dc, eltype(dc), sys.ny, sys.nu, N)
    Threads.@threads for i = 1:N
        H[:,:,i] = evalfr(sys, fval=ω[i])
    end
    return H
end

"""
    nx = order(sys)

Return the order `nx` of the descriptor system `sys` as the dimension of the state variable vector. 
For improper or non-minimal systems, `nx` is less than the McMillan degree of the system.   
"""
function order(sys::DSTYPE)
    return sys.nx
end
function Base.ndims(::DSTYPE)
# """
#     ndims(sys)

# Return the number of dimensions of the descriptor system array `sys` (1 for a single model).
# """
   return 1
end
"""
    size(sys) -> (p,m)
    size(sys,1) -> p
    size(sys,2) -> m

Return the number of outputs `p` and the number of inputs `m` of a descriptor system `sys`.
"""
function Base.size(sys::DSTYPE)
    size(sys.D)
end
function Base.size(sys::DSTYPE, d::Integer)
    d <= 2 ? size(sys)[d] : 1
end
function Base.length(sys::DSTYPE)
# """
#     length(sys)

# Return the number of elements of the descriptor system array `sys` (1 for a single model).
# """
   return 1
end
"""
     iszero(sys; atol = 0, atol1 = atol, atol2 = atol, rtol, fastrank = true)

Return `true` if the transfer function matrix of the descriptor system `sys` is zero. 
For a descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)` it is checked 
that the normal rank of `G(λ)` is zero, or equivalently (see [1]), that the normal rank of 
the system matrix pencil 

              | A-λE | B | 
      S(λ) := |------|---|
              |  C   | D |  

is equal to `n`, the order of the system `sys`. 

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

[1] A. Varga,  On checking null rank conditions of rational matrices, 2018. 
[arXiv:2006.06825](https://arxiv.org/pdf/1812.11396).
"""
function iszero(sys::DescriptorStateSpace{T}; atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                rtol::Real =  ((max(size(sys.A)...))+1)*eps(real(float(one(real(T)))))*iszero(max(atol1,atol2)), 
                fastrank::Bool = true) where T   
   return (sprank(dssdata(sys)..., atol1 = atol1, atol2 = atol2, rtol = rtol, fastrank = fastrank) == sys.nx)
end
function iszero(sys::SparseDescriptorStateSpace{T}; atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
    rtol::Real =  ((max(size(sys.A)...))+1)*eps(real(float(one(real(T)))))*iszero(max(atol1,atol2)), 
    fastrank::Bool = true) where T  
    if sys.E == I 
       nrmA = opnorm(sys.A,1)
       scale = max(nrmA,1)*rand()
       nrmM = max(nrmA, opnorm(sys.B,1), opnorm(sys.C,Inf), opnorm(sys.D,1))
       return rank([sys.A+scale*sys.E sys.B; sys.C sys.D], tol = max(atol1,atol2) + nrmM*rtol) == sys.nx
    else
       nrmN = opnorm(sys.E,1)
       nrmM = max(opnorm(sys.A,1), opnorm(sys.B,1), opnorm(sys.C,Inf), opnorm(sys.D,1))
       nrmN == zero(nrmN) && (return rank([sys.A sys.B; sys.C sys.D], tol = max(atol1,atol2) + nrmM*rtol) == sys.nx)
       scale = nrmM/nrmN*rand()
       return rank([sys.A+scale*sys.E sys.B; sys.C sys.D], tol = max(atol1,atol2) + max(nrmN,nrmM)*rtol) == sys.nx
    end
end
function iszero(sys::DescriptorStateSpaceExt{T}; atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
    rtol::Real =  ((max(size(sys.A)...))+1)*eps(real(float(one(real(T)))))*iszero(max(atol1,atol2)), 
    fastrank::Bool = true) where T  
    if issparse(sys.A) 
       if sys.E == I 
         nrmA = opnorm(sys.A,1)
         scale = max(nrmA,1)*rand()
         nrmM = max(nrmA, opnorm(sys.B,1), opnorm(sys.C,Inf), opnorm(sys.D,1))
         return rank([sys.A+scale*sys.E sys.B; sys.C sys.D], tol = max(atol1,atol2) + nrmM*rtol) == sys.nx
       else
         nrmN = opnorm(sys.E,1)
         nrmM = max(opnorm(sys.A,1), opnorm(sys.B,1), opnorm(sys.C,Inf), opnorm(sys.D,1))
         nrmN == zero(nrmN) && (return rank([sys.A sys.B; sys.C sys.D], tol = max(atol1,atol2) + nrmM*rtol) == sys.nx)
         scale = nrmM/nrmN*rand()
         return rank([sys.A+scale*sys.E sys.B; sys.C sys.D], tol = max(atol1,atol2) + max(nrmN,nrmM)*rtol) == sys.nx
      end
    end
    return (sprank(dssdatafull(sys)..., atol1 = atol1, atol2 = atol2, rtol = rtol, fastrank = fastrank) == sys.nx)
end

"""
    Gval = evalfr(sys, val; atol1 = 0, atol2 = 0, rtol, fast = true) 

Evaluate for a descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`,
`Gval`, the value of the rational matrix `G(λ) = C*inv(λE-A)*B+D` for `λ = val`. 
The computed `Gval` has infinite entries if `val` is a pole (finite or infinite) of `G(λ)`.
If `val` is finite and `val*E-A` is singular or if `val = Inf` and `E` is singular, 
then the entries of `Gval` are evaluated separately for minimal realizations of each input-output channel.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 

The computation of minimal realizations of individual input-output channels relies on pencil manipulation algorithms,
which employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.
"""
function evalfr(SYS::DescriptorStateSpace, val::Number; kwargs...) 
    return lseval(dssdata(SYS)..., val; kwargs...)
end
function evalfr(sys::Union{SparseDescriptorStateSpace{T},DescriptorStateSpaceExt{T}}, val::Number) where {T}
    T1 = promote_type(Float64, typeof(val), T)        
    isinf(val) && (return fill(NaN,size(sys.D)...))   
    return sys.C*((T1(val)*sys.E-sys.A)\Matrix(sys.B))+sys.D
end
"""
    Gval = evalfr(sys; fval = 0, atol = 0, atol1 = atol, atol2 = atol, rtol, fast = true) 

Evaluate for a descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`,
`Gval`, the value of the rational matrix `G(λ) = C*inv(λE-A)*B+D` for `λ = val`, where `val = im*fval` 
for a continuous-time system or `val = exp(im*fval*sys.Ts)` for a discrete-time system. 
The computed `Gval` has infinite entries if `val` is a pole (finite or infinite) of `G(λ)`.
If `val` is finite and `val*E-A` is singular or if `val = Inf` and `E` is singular, 
then the entries of `Gval` are evaluated separately for minimal realizations of each input-output channel.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`.
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 

The computation of minimal realizations of individual input-output channels relies on pencil manipulation algorithms,
which employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.
"""
function evalfr(SYS::DescriptorStateSpace{T}; fval::Real = 0, atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
    rtol::Real =  ((max(size(SYS.A)...))+1)*eps(real(float(one(real(T)))))*iszero(max(atol1,atol2)), fast::Bool = true) where T
    if SYS.Ts == 0
       val =  fval == 0 ? zero(T) : im*abs(fval) 
    else
       val = fval == 0 ? one(T) : exp(im*abs(fval*SYS.Ts)) 
    end
    return lseval(dssdata(SYS)..., val; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)
end
function evalfr(SYS::Union{SparseDescriptorStateSpace{T},DescriptorStateSpaceExt{T}}; fval::Real = 0) where {T}
    if SYS.Ts == 0
       val = fval == 0 ? zero(T) : im*abs(fval) 
    else
       val = fval == 0 ? one(T) : exp(im*abs(fval*SYS.Ts)) 
    end
    return evalfr(SYS,val) 
end
"""
    Gval = dcgain(sys; atol1, atol2, rtol, fast = true) 

Evaluate for a descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`,
`Gval`, the DC (or steady-state) gain. `Gval` is the value of the 
rational matrix `G(λ)` for `λ = val`, where for a continuous-time system `val = 0` and 
for a discrete-time system `val = 1`. The computed `Gval` has infinite entries if `val` is a pole of `G(λ)`.
In this case (i.e., `val*E-A` is singular), the entries of `Gval` are evaluated separately for minimal realizations 
of each input-output channel.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 

The computation of minimal realizations of individual input-output channels relies on pencil manipulation 
algorithms, which employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.
"""
function dcgain(SYS::DescriptorStateSpace; kwargs...)
    SYS.Ts == 0 ? (return lseval(dssdata(SYS)..., 0; kwargs...) ) : 
                  (return lseval(dssdata(SYS)..., 1; kwargs...) )
end
dcgain(SYS::Union{SparseDescriptorStateSpace,DescriptorStateSpaceExt}) = evalfr(SYS)
# function dcgain(SYS::DescriptorStateSpace; kwargs...) 
#     SYS.Ts == 0 ? (return lseval(dssdata(SYS)..., 0; kwargs...) ) : 
#                   (return lseval(dssdata(SYS)..., 1; kwargs...) )
# end
"""
     opnorm(sys[, p = Inf]; kwargs...) 
     opnorm(sys, 2; kwargs...) -> sysnorm
     opnorm(sys, Inf; kwargs...) -> (sysnorm, fpeak)
     opnorm(sys; kwargs...) -> (sysnorm, fpeak)

Compute for a descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix `G(λ)`, 
the `L2` or `L∞` system norm `sysnorm` induced by the vector `p`-norm, where valid values of `p` are `2` or `Inf`. 
For the `L∞` norm, the frequency `fpeak` is also returned, where `G(λ)` achieves its peak gain. 
See [`gh2norm`](@ref) and [`ghinfnorm`](@ref) for a description of the allowed keyword arguments.  
"""
function opnorm(SYS::DescriptorStateSpace{T}, p::Real=Inf; fast::Bool = true, offset::Real = sqrt(eps(float(real(T)))), 
    atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, atolinf::Real = atol, rtolinf::Real = real(T)(0.001), 
    rtol::Real = SYS.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2))) where T
    if p == 2
        return gl2norm(SYS, fast = fast, offset = offset, atol1 = atol1, atol2 = atol2, atolinf = atolinf, rtol = rtol)
    elseif p == Inf
        return glinfnorm(SYS, rtolinf = rtolinf, fast = fast, offset = offset, atol1 = atol1, atol2 = atol2, rtol = rtol)
    else
        throw(ArgumentError("invalid p-norm p=$p. Valid: p = 2 and p = Inf"))
    end
end
"""
    sys = rss(n, p, m; disc = false, Ts, T = Float64, stable = false, nuc = 0, nuo = 0, randt = true) 

Generate a randomized `n+nuc+nuo`-th order standard state-space system `sys = (A,B,C,D)` with `p` outputs and `m` inputs, with all matrices 
randomly generated of type `T`.
The resulting `sys` is a continuous-time system if `disc = false` and a discrete-time system if `disc = true`.
For a discrete-time system a sample time `Δ` can be specified using the keyword argument `Ts = Δ` (default: `Δ = -1`, i.e., not specified).

If `stable = true`, the resulting system is stable, with `A` having all eigenvalues with negative real parts for a continuous-time system, 
or with moduli less than one for a discrete-time system. 
If `nuc+nuo > 0`, the system `sys` is non-minimal, with `A` having `nuc` uncontrollable and `nuo` unobservable eigenvalues. 
If  `randt = true`, a randomly generated orthogonal or unitary similarity transformation is additionally applied.    
If `randt = false`, the system matrices `A`, `B`, and `C` result in block stuctured forms exhibitting the 
uncontrollable and unobservable eigenvalues of `A`:

    A = diag(A1, A2, A3),  B = [B1; 0; B3], C = [C1 C2 0]

with the diagonal blocks `A1`, `A2`, `A3` of orders `n`, `nuc`, and `nuo`, respectively. 
"""
function rss(n::Int, p::Int, m::Int; disc::Bool = false, Ts::Real = -1.0, T::Type = Float64,
             nuc::Int = 0, nuo::Int = 0, randt = true, stable::Bool = false) 
    !disc || Ts > 0 || Ts == -1 || throw(ArgumentError("sampling time must be either positive or equal to -1 "))             
    nf = n; Af = randn(T,nf,nf); Bf = randn(T,nf,m); Cf = randn(T,p,nf); 
    n > 0 && stable && (disc ? rmul!(Af,1/(rand()+opnorm(Af,1))) : Af -= I*(rand()+maximum(real(eigvals(Af)))) ) 
    Afuc = randn(T,nuc,nuc); Bfuc = zeros(T,nuc,m); Cfuc = randn(T,p,nuc);
    stable && nuc > 0 && (disc ? rmul!(Afuc,1/(rand()+opnorm(Afuc,1))) : Afuc -= I*(rand()+maximum(real(eigvals(Afuc)))) ) 
    Afuo = randn(T,nuo,nuo); Bfuo = randn(T,nuo,m); Cfuo = zeros(T,p,nuo); 
    stable && nuo > 0 && (disc ? rmul!(Afuo,1/(rand()+opnorm(Afuo,1))) : Afuo -= I*(rand()+maximum(real(eigvals(Afuo)))) ) 
    nx = nf+nuc+nuo
    Q = randt ? qr!(rand(nx,nx)).Q : I
    return DescriptorStateSpace{T}(Q'*blockdiag(Af,Afuc,Afuo)*Q, I, 
                                   Q'*[Bf;Bfuc;Bfuo], [ Cf Cfuc Cfuo ]*Q, rand(T,p,m), 
                                   disc ? Ts : zero(real(T))) 
end
"""
    sys = rdss(n, p, m; id = [ ], disc = false, Ts, T = Float64, stable = false, nfuc = 0, iduc = [ ], 
               nfuo = 0, iduo = [ ], randlt = true, randrt = true) 

Generate a randomized descriptor state-space system `sys = (A-λE,B,C,D)` with `p` outputs and `m` inputs, with all matrices 
randomly generated of type `T`. 
The resulting `sys` is a continuous-time system if `disc = false` and a discrete-time system if `disc = true`. 
For a discrete-time system a sample time `Δ` can be specified using the keyword argument `Ts = Δ` (default: `Δ = -1`, i.e., not specified).

If the vector `id` is nonempty, then `id[i]` specifies the order of the `i`-th infinite elementary divisor of the
resulting pencil `A-λE`, which thus has `n` finite eigenvalues and `ni = sum(id)` infinite eigenvalues which are
controllable and observable. 
If `nfuc+nfuo > 0`, the system `sys` is non-minimal, with `A` having `nfuc` uncontrollable and `nfuo` unobservable finite eigenvalues. 
If the vector `iduc` is a nonempty, then `iduc[i]` specifies the order of the `i`-th infinite elementary divisor 
with uncontrollable infinite eigenvalues of the resulting pencil `A-λE`, which thus has `niuc = sum(iduc)` uncontrollable infinite eigenvalues. 
If the vector `iduo` is a nonempty, then `iduo[i]` specifies the order of the `i`-th infinite elementary divisor with 
unobservable infinite eigenvalues of the resulting pencil `A-λE`, which thus has `niuo = sum(iduo)` unobservable infinite eigenvalues. 
If `niuc+niuo > 0`, the system `sys` is non-minimal, with `A` having `niuc` uncontrollable and `niuo` unobservable infinite eigenvalues. 

It follows, that the resulting pencil `A-λE` has  `n+nfuc+nfuo` finite eigenvalues and `ni+niuc+niuo` infinite eigenvalues. 
If `stable = true`, the proper part of the system `sys` is stable, with `A` having all finite eigenvalues with negative real parts 
for a continuous-time system, or with moduli less than one for a discrete-time system. 

If  `randlt = true`, a randomly generated orthogonal or unitary transformation is additionally applied to `A`, `E`, and `B` from the left.    
If  `randrt = true`, a randomly generated orthogonal or unitary transformation is additionally applied to `A`, `E`,  and `C` from the right.    
If `randlt = false` and `randrt = false`, the system matrices `A`, `E`, `B`, and `C` result in block stuctured forms exhibitting the 
uncontrollable and unobservable finite and infinite eigenvalues of `A-λE`:

    A-λE = diag(A1-λE1, A2-λE2, A3-λE3, A4-λE4, A5-λE5, A6-λE6),  
    B = [B1; B2; 0; 0; B5; B6 ], 
    C = [C1 C2 C3 C4 0 0]

with the diagonal blocks `A1`, `A2`, `A3`, `A4`, `A5`, `A6` of orders `n`, `ni`, `nfuc`, `niuc`, `nfuo` and `niuo`, respectively. 
"""
function rdss(n::Int, p::Int, m::Int; disc::Bool = false, stable::Bool = false, Ts::Real = -1.0, T::Type = Float64, id::Vector{Int} = Int[], 
                                      nfuc::Int = 0, nfuo::Int = 0, iduc::Vector{Int} = Int[], iduo::Vector{Int} = Int[], 
                                      randlt = true, randrt = true) 
    !disc || Ts > 0 || Ts == -1 || throw(ArgumentError("sample time must be either positive or equal to -1 "))             
    nf = n; Af = randn(T,nf,nf); Ef = randn(T,nf,nf); Bf = randn(T,nf,m); Cf = randn(T,p,nf); 
    n > 0 && stable && (disc ? Af = Ef*rmul!(Af,1/(rand()+opnorm(Af,1))) : Af = Ef*(Af-I*(rand()+maximum(real(eigvals(Af))))) ) 
    ni = sum(id); Ai = Matrix{T}(I,ni,ni); Ei = jordanblockdiag(zero(T),id); Bi = randn(T,ni,m); Ci = randn(T,p,ni);
    Afuc = randn(T,nfuc,nfuc); Efuc = randn(T,nfuc,nfuc); Bfuc = zeros(T,nfuc,m); Cfuc = randn(T,p,nfuc);
    stable && nfuc > 0 && (disc ? Afuc = Efuc*rmul!(Afuc,1/(rand()+opnorm(Afuc,1))) : Afuc = Efuc*(Afuc-I*(rand()+maximum(real(eigvals(Afuc))))) ) 
    niuc = sum(iduc); Aiuc = Matrix{T}(I,niuc,niuc); Eiuc = jordanblockdiag(zero(T),iduc); Biuc = zeros(T,niuc,m); Ciuc = randn(T,p,niuc);
    Afuo = randn(T,nfuo,nfuo); Efuo = randn(T,nfuo,nfuo); Bfuo = randn(T,nfuo,m); Cfuo = zeros(T,p,nfuo); 
    stable && nfuo > 0 && (disc ? Afuo = Efuo*rmul!(Afuo,1/(rand()+opnorm(Afuo,1))) : Afuo = Efuo*(Afuo-I*(rand()+maximum(real(eigvals(Afuo))))) ) 
    niuo = sum(iduo); Aiuo = Matrix{T}(I,niuo,niuo); Eiuo = jordanblockdiag(zero(T),iduo); Biuo = randn(T,niuo,m); Ciuo = zeros(T,p,niuo);
    nx = nf+ni+nfuc+nfuo+niuc+niuo
    Q = randlt ? qr!(rand(nx,nx)).Q : I
    Z = randrt ? qr!(rand(nx,nx)).Q : I
    return DescriptorStateSpace{T}(Q*blockdiag(Af,Ai,Afuc,Aiuc,Afuo,Aiuo)*Z, Q*blockdiag(Ef,Ei,Efuc,Eiuc,Efuo,Eiuo)*Z, 
                                   Q*[Bf;Bi;Bfuc;Biuc;Bfuo;Biuo], [ Cf Ci Cfuc Ciuc Cfuo Ciuo]*Z, rand(T,p,m), 
                                   disc ? Ts : zero(real(T))) 
end
"""
    sysr = dsxvarsel(sys,ind)

Construct for the descriptor system `sys = (A-λE,B,C,D)` of order `n` the descriptor system  
`sysr = (A[ind,ind]-λE[ind,ind],B[ind,:],C[:,ind],D)` of order `nr = length(ind)`, 
by selecting the state variables of `sys` with indices specified by `ind`. 
If `ind` is a permutation vector of length `n`, then `sysr` has the same transfer function matrix as `sys` 
and permuted state variables. 
"""
function dsxvarsel(SYS::DescriptorStateSpace{T},ind::Union{UnitRange{Int64},StepRange{Int64, Int64}, Array{Int64,1}}) where T
    isempty(ind) &&  (return DescriptorStateSpace{T}(zeros(T,0,0),I,zeros(T,0,SYS.nu),zeros(T,SYS.ny,0),SYS.D,SYS.Ts))
    (minimum(ind) < 1 || maximum(ind) > SYS.nx) && error("BoundsError: selected indices $ind out of range $(1:SYS.nx)")
    return DescriptorStateSpace{T}(SYS.A[ind,ind], SYS.E == I ? I : SYS.E[ind,ind], SYS.B[ind,:], SYS.C[:,ind], SYS.D, SYS.Ts)
end
function dsxvarsel(SYS::SparseDescriptorStateSpace{T},ind::Union{UnitRange{Int64},StepRange{Int64, Int64}, Array{Int64,1}}) where T
    isempty(ind) &&  (return DescriptorStateSpace{T}(zeros(T,0,0),I,zeros(T,0,SYS.nu),zeros(T,SYS.ny,0),SYS.D,SYS.Ts))
    (minimum(ind) < 1 || maximum(ind) > SYS.nx) && error("BoundsError: selected indices $ind out of range $(1:SYS.nx)")
    return SparseDescriptorStateSpace{T}(SYS.A[ind,ind], SYS.E == I ? I : SYS.E[ind,ind], SYS.B[ind,:], SYS.C[:,ind], SYS.D, SYS.Ts)
end

"""
     rtfbilin(type = "c2d"; Ts = T, Tis = Ti, a = val1, b = val2, c = val3, d = val4) -> (g, ginv)

Build the rational transfer functions of several commonly used bilinear transformations and their inverses. 
The resulting `g` describes the rational transfer function `g(δ)` in the bilinear transformation `λ = g(δ)` and 
`ginv` describes its inverse transformation `ginv(λ)` in the bilinear transformation `δ = ginv(λ)`.
In accordance with the values of `type` and the keyword argument values `Ts`, `Tis`, `a`, `b`, `c`, and `d`, 
the resulting `g` and `ginv` contain first order rational transfer functions of the form `g(δ) = (a*δ+b)/(c*δ+d)` 
and `ginv(λ) = (d*λ-b)/(-c*λ+a)`, respectively, which satisfy `g(ginv(λ)) = λ` and `ginv(g(δ)) = δ`. 

Depending on the value of `type`, the following types of transformations can be generated 
in conjunction with parameters specified in `Ts`, `Tis`, `a`, `b`, `c`, and `d`:

    "Cayley" - Cayley transformation: `s = g(z) = (z-1)/(z+1)` and `z = ginv(s) = (s+1)/(-s+1)`; 
               the sampling time `g.Ts` is set to the value `T ≠ 0` (default `T = -1`), while `ginv.Ts = 0`;
               g(z) and ginv(s) are also known as the continuous-to-discrete and discrete-to-continuous transformations, respectively; 

    "c2d"    - is alias to "Cayley"

    "Tustin" - Tustin transformation (also known as trapezoidal integration): `s = g(z) = (2*z-2)/(T*z+T)` and `z = ginv(s) = (T*s+2)/(-T*s+2)`; 
               the sampling time `g.Ts` is set to the value `T ≠ 0` (default `T = -1`), while `ginv.Ts = 0`;
               a nonzero prewarping frequency `freq` can be specified using the keyword parameter `prewarp_freq = freq`;

    "Euler"  - Euler integration (or forward Euler integration): `s = g(z) = s = (z-1)/T` and `z = ginv(s) = T*s+1`; 
               the sampling time `g.Ts` is set to the value `T ≠ 0` (default `T = -1`), while `ginv.Ts = 0`;

    "BEuler" - Backward Euler integration (or backward Euler integration): `s = g(z) = (z-1)/(T*z)` and `z = ginv(s) = 1/(-T*s+1)`;
               the sampling time `g.Ts` is set to the value `T ≠ 0` (default `T = -1`), while `ginv.Ts = 0`;

    "Moebius" - general (Moebius ) bilinear transformation: `λ = g(δ) = (a*δ+b)/(c*δ+d)` and `δ = ginv(λ) = (d*λ-b)/(-c*λ+a)`;  
                the sampling times `g.Ts` and `ginv.Ts` are  set to the values `T` and `Ti`, respectively;
                the default values of the parameters `a`, `b`, `c`, and `d` are `a = 1`, `b = 0`, `c = 0`, and `d = 1`.
                Some useful particular Moebius transformations correspond to the following choices of parameters: 
                - _translation_: `a = 1`, `b ≠ 0`, `c = 0`, `d = 1` (`λ = δ+b`)
                - _scaling_: `a ≠ 0`, `b = 0`, `c = 0`, `d = 1` (`λ = a*δ`)
                - _rotation_: `|a| = 1`, `b = 0`, `c = 0`, `d = 1` (`λ = a*δ`)
                - _inversion_: `a = 0`, `b = 1`, `c = 1`, `d = 0`  (`λ = 1/δ`)

    "lft"     - alias to "Moebius" (linear fractional transformation).   
"""
function rtfbilin(type::String = "c2d"; Ts::Union{Real,Missing} = missing, Tsi::Union{Real,Missing} = missing, 
                  prewarp_freq::Real = 0,  a::Number = 1, b::Number = 0, c::Number = 0, d::Number = 1 )   
    s = rtf('s'); 
    type = lowercase(type)
    if type == "c2d" || type == "cayley" 
        ismissing(Ts) && (Ts = -1)
        z = rtf('z',Ts = Ts)
        g = (z-1)/(z+1); ginv = (s+1)/(-s+1)
    elseif type == "d2c"
        ismissing(Tsi) && (Tsi = -1)
        z = rtf('z',Ts = Tsi);
        g = (s+1)/(-s+1); ginv = (z-1)/(z+1);
    elseif type ==  "tustin" 
        ismissing(Ts) && (Ts = 2) 
        Ts > 0 || error("sampling time must be positive") 
        z = rtf('z',Ts = Ts);
        prewarp_freq == 0 ? t = Ts : t = 2*tan(prewarp_freq*Ts/2)/prewarp_freq 
        g = (2*z-2)/(t*z+t); ginv = (t*s+2)/(-t*s+2);
    elseif type ==  "euler" 
        ismissing(Ts) && (Ts = 1) 
        Ts > 0 || error("sampling time must be positive") 
        z = rtf('z',Ts = Ts);
        g = (z-1)/Ts; ginv = Ts*s+1;
    elseif type ==  "beuler" 
        ismissing(Ts) && (Ts = 1) 
        Ts > 0 || error("sampling time must be positive") 
        z = rtf('z',Ts = Ts);
        g = (z-1)/(Ts*z); ginv = 1/(-Ts*s+1);
    elseif type ==  "lft" || type ==  "moebius"
        c == 0 && d == 0 &&  error("the denominator of g must be nonzero")
        a*d ≈ b*c &&  error("the McMillan degree of g must be one")
        ismissing(Ts) && (Ts = 0)
        if Ts == 0
            g = (a*s+b)/(c*s+d) 
        else
            z = rtf('z',Ts = Ts)
            g = (a*z+b)/(c*z+d) 
        end
        ismissing(Tsi) && (Tsi = 0)
        if Tsi == 0
            ginv = (d*s-b)/(-c*s+a)
       else
            z = rtf('z',Ts = Tsi)
            ginv = (d*z-b)/(-c*z+a)
        end
    else
        error("Unknown type")
    end
    return g, ginv
    # end RTFBILIN
end
"""
     res = dssubset(sys, subsys, rows, cols; minimal = true, fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol = nϵ)

Assign for a given descriptor system `sys = (A-λE,B,C,D)` its subsystem `sys[row,cols]` to `subsys` and 
return the modified system in `res`. `rows` and `cols` are indices, vectors of indices, index ranges, `:` or 
any combinations of them. 

If `minimal = true` (default), an irreducible realization of the resulting system `res` is computed, otherwise a
possibly non-minimal realization is returned if `minimal = false`. 

The underlying pencil manipulation based minimal realization algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 
The default relative tolerance is `nϵ`, where `ϵ` is the working _machine epsilon_ 
and `n` is the order of the system `sys`.  
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 

"""
function dssubset(sys::DescriptorStateSpace{T1}, subsys::DescriptorStateSpace{T2}, inds...; 
                        atol::Real = zero(real(T1)), atol1::Real = atol, atol2::Real = atol, 
                        rtol::Real =  sys.nx*eps(real(float(one(real(T1)))))*iszero(max(atol1,atol2)), 
                        fast::Bool = true, minimal::Bool = true  ) where {T1, T2}
    size(inds, 1) != 2 &&
        error("Must specify 2 indices to index descriptor state-space models")
    rows, cols = index2range(inds...) 
    p, m = size(sys)
    rows == Colon() && (rows = 1:p)
    cols == Colon() && (cols = 1:m)
    pred, mred = size(subsys)
    length(rows) == pred  || error("number of row indices must be equal to the number of outputs of subsys")
    length(cols) == mred  || error("number of column indices must be equal to the number of inputs of subsys")
    pred > 0 && maximum(rows) > p && error("output indices must not exceed $p")
    mred > 0 && maximum(cols) > m && error("input indices must not exceed $m")
    irows = setdiff(Vector(1:p),Vector(rows))
    jcols = setdiff(Vector(1:m),Vector(cols))
    if minimal 
        res = gir([sys[irows,[jcols;cols]]; sys[rows,jcols] subsys]; atol, atol1, atol2, rtol, fast )
    else
        res = [sys[irows,[jcols;cols]]; sys[rows,jcols] subsys]
    end
    ip = sortperm([irows;rows])
    jp = sortperm([jcols;cols])
    return DescriptorStateSpace{eltype(res)}(res.A, res.E, res.B[:, jp], res.C[ip, :], res.D[ip, jp], sys.Ts)
end
dssubset(sys::DST, subsys::AbstractNumOrArray, inds...; kwargs...) where {T, DST <: DescriptorStateSpace{T}} = dssubset(sys, dss(T.(subsys); Ts = sys.Ts), inds...; kwargs...)
"""
     res = dszeros(sys, rows, cols; minimal = true, fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol = nϵ)

Set for a given descriptor system `sys = (A-λE,B,C,D)` its subsystem `sys[row,cols]` to zero and 
return the modified system in `res`. `rows` and `cols` are indices, vectors of indices, index ranges, `:` or 
any combinations of them. 

If `minimal = true` (default), an irreducible realization of the resulting system `res` is computed, otherwise a
possibly non-minimal realization is returned if `minimal = false`. 

The underlying pencil manipulation based minimal realization algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 
The default relative tolerance is `nϵ`, where `ϵ` is the working _machine epsilon_ 
and `n` is the order of the system `sys`.  
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 

"""
function dszeros(sys::DescriptorStateSpace{T}, inds...; 
                        atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                        rtol::Real =  sys.nx*eps(real(float(one(real(T)))))*iszero(max(atol1,atol2)), 
                        fast::Bool = true, minimal::Bool = true  ) where T
    size(inds, 1) != 2 &&
        error("Must specify 2 indices to index descriptor state-space models")
    rows, cols = index2range(inds...) 
    p, m = size(sys)
    rows == Colon() && (rows = 1:p)
    cols == Colon() && (cols = 1:m)
    irows = setdiff(Vector(1:p),Vector(rows))
    jcols = setdiff(Vector(1:m),Vector(cols))
    subsys = dss(zeros(T,length(rows),length(cols)); Ts = sys.Ts)
    if minimal 
        res = gir([sys[irows,[jcols;cols]]; sys[rows,jcols] subsys]; atol, atol1, atol2, rtol, fast )
    else
        res = [sys[irows,[jcols;cols]]; sys[rows,jcols] subsys]
    end
    ip = sortperm([irows;rows])
    jp = sortperm([jcols;cols])
    return DescriptorStateSpace{eltype(res)}(res.A, res.E, res.B[:, jp], res.C[ip, :], res.D[ip, jp], sys.Ts)
end
"""
     res = dssubsel(sys, S; minimal = true, fast = true, atol = 0, atol1 = atol, atol2 = atol, rtol = nϵ)

Select for a given descriptor system `sys = (A-λE,B,C,D)` its subsystem `res` corresponding to the zero-one pattern specified by 
the binary structure matrix `S`. If `G(λ)` is the transfer function matrix of `sys`, then the transfer function matrix of `res`
is the element-wise product `S .* G(λ)`. 

If `minimal = true` (default), an irreducible realization of the resulting system `res` is computed, otherwise a
possibly non-minimal realization is returned if `minimal = false`. 

The underlying pencil manipulation based minimal realization algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 
The default relative tolerance is `nϵ`, where `ϵ` is the working _machine epsilon_ 
and `n` is the order of the system `sys`.  
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function dssubsel(sys::DescriptorStateSpace{T}, S::Union{BitMatrix,Matrix{Bool}}; 
                        atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
                        rtol::Real =  sys.nx*eps(real(float(one(real(T)))))*iszero(max(atol1,atol2)), 
                        fast::Bool = true, minimal::Bool = true ) where T
    p, m = size(sys)                    
    (p, m ) == size(S) || error("missmatch between the sizes of sys and S")
    syst = dss(zeros(T,p,m); Ts = sys.Ts)
    cols = collect(1:m)
    for i = 1:p
        colsi = cols[S[i,:]]
        syst = dssubset(syst,sys[i,colsi],i,colsi; atol1, atol2, rtol, fast, minimal = false)
    end
    return minimal ? gir(syst; atol, atol1, atol2, rtol, fast) : syst
end
"""
     sysdiag = dsdiag(sys, k)

Build for a given descriptor system `sys` and non-negative integer `k` a descriptor system `sysdiag`
obtained as `sysdiag = diag( sys, ..., sys )` (i.e., `k`-times repeated application of `append`).   
"""
function dsdiag(sys::DescriptorStateSpace{T}, k::Int) where T
    p, m = size(sys)   
    k >=0 || error("k must be a non-negative integer")
    n = sys.nx
    N = k*n
    M = k*m
    P = k*p
    a, e, b, c, d = dssdata(sys)
    desc = !(e == I)   
    A = zeros(T, N, N)
    E = desc ? zeros(T, N, N) : I
    B = zeros(T, N, M)
    C = zeros(T, P, N)
    D = zeros(T, P, M)
    ia = 1
    jb = 1
    ic = 1
    for i = 1:k  
        ia1 = ia:ia+n-1
        jb1 = jb:jb+m-1
        ic1 = ic:ic+p-1
        A[ia1,ia1] = a
        desc && (E[ia1,ia1] = e)
        B[ia1,jb1] = b
        C[ic1,ia1] = c  
        D[ic1,jb1] = d  
        ia += n
        jb += m
        ic += p
    end
    return DescriptorStateSpace{T}(A, E, B, C, D, sys.Ts)
end
"""
     syscl = feedback(sys, K, inp1, out1; negative = true)

Build for a given descriptor system `sys` with input vector `u` and output vector `y` and 
a static output feedback gain `K` the closed-loop descriptor system `syscl` is constructed
corresponding to the static output feedback `u[inp] = -K*y[out] + v`, where `inp` and `out` are 
are indices, vectors of indices, index ranges, `:` or any combinations of them. Only distinct indices 
can be specified. If `negative = false`, a positive feedback `u[inp] = K*y[out] + v` is used.
"""
function feedback(sys::DescriptorStateSpace{T1}, K::Matrix{T2}, inds...; negative::Bool = true) where {T1, T2}
    size(inds, 1) != 2 &&
        error("Must specify 2 indices to index descriptor state-space models")
    u1, y1 = index2range(inds...) 
    p, m = size(sys)
    u1 == Colon() && (u1 = 1:m) 
    y1 == Colon() && (y1 = 1:p)
    maximum(u1) > m && error("input indices must not exceed $m")
    maximum(y1) > m && error("output indices must not exceed $p")
    allunique(u1) || error("all input indices must be distinct")
    allunique(y1) || error("all output indices must be distinct")
    m1, p1 = size(K)
    length(u1) == m1  || error("number of row indices must be equal to the number of rows of K")
    length(y1) == p1  || error("number of column indices must be equal to the number of columns of K")
    y2 = setdiff(Vector(1:p),Vector(y1))
    u2 = setdiff(Vector(1:m),Vector(u1))

    A, E, B, C, D = dssdata(sys)
    B1 = view(B,:,u1)
    B2 = view(B,:,u2)
    C1 = view(C,y1,:)
    C2 = view(C,y2,:)
    D11 = view(D,y1,u1)
    D12 = view(D,y1,u2)
    D21 = view(D,y2,u1)
    D22 = view(D,y2,u2)
    nullD11 = iszero(D11)
    Li = nullD11 ? I : (negative ? inv(I+K*D11) : inv(I-K*D11))
    KT = nullD11 ? (negative ? -K : K) : (negative ? -Li*K : Li*K)
    ip = sortperm([y1;y2])
    jp = sortperm([u1;u2])
    
    return dss(A+B1*KT*C1, E, [B1*Li (B2+B1*KT*D12)][:,jp], [C1 + D11*KT*C1 ; C2 + D21*KT*C1][ip,:], 
                        [ D11*Li D12+D11*KT*D12; D21*Li D22+D21*KT*D12 ][ip,jp]; Ts = sys.Ts)

end
feedback(sys::DescriptorStateSpace{T1}, K::Matrix{T2}; negative::Bool = true) where {T1, T2} = feedback(sys,K,1:size(K,1),1:size(K,2); negative)
