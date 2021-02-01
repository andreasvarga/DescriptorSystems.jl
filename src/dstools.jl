"""
    order(sys)

Return the order (also the number of state variables) of the descriptor system `sys`. 
"""
function order(sys::AbstractDescriptorStateSpace)
    return sys.nx
end
function Base.ndims(::AbstractDescriptorStateSpace)
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
function Base.size(sys::AbstractDescriptorStateSpace)
    size(sys.D)
end
function Base.size(sys::AbstractDescriptorStateSpace, d::Integer)
    d <= 2 ? size(sys)[d] : 1
end
function Base.length(sys::AbstractDescriptorStateSpace)
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
"""
    Gval = evalfr(sys, val; atol1, atol2, rtol, fast = true) 

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
function evalfr(SYS::AbstractDescriptorStateSpace, val::Number; kwargs...) 
    return lseval(dssdata(SYS)..., val; kwargs...)
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
function dcgain(SYS::AbstractDescriptorStateSpace; kwargs...) 
    SYS.Ts == 0 ? (return lseval(dssdata(SYS)..., 0; kwargs...) ) : 
                  (return lseval(dssdata(SYS)..., 1; kwargs...) )
end
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
    sys = rss(n, p, m; disc = false, T = Float64, stable = false, nuc = 0, nuo = 0, randt = true) 

Generate a randomized `n+nuc+nuo`-th order standard state-space system `sys = (A,B,C,D)` with `p` outputs and `m` inputs, with all matrices 
randomly generated of type `T`.
The resulting `sys` is a continuous-time system if `disc = false` and a discrete-time system if `disc = true`.
If `stable = true`, the resulting system is stable, with `A` having all eigenvalues with negative real parts for a continuous-time system, 
or with moduli less than one for a discrete-time system. 
If `nuc+nuo > 0`, the system `sys` is non-minimal, with `A` having `nuc` uncontrollable and `nuo` unobservable eigenvalues. 
If  `randt = true`, a randomly generated orthogonal or unitary similarity transformation is additionally applied.    
If `randt = false`, the system matrices `A`, `B`, and `C` result in block stuctured forms exhibitting the 
uncontrollable and unobservable eigenvalues of `A`:

    A = diag(A1, A2, A3),  B = [B1; 0; B3], C = [C1 C2 0]

with the diagonal blocks `A1`, `A2`, `A3` of orders `n`, `nuc`, and `nuo`, respectively. 
"""
function rss(n::Int, p::Int, m::Int; disc::Bool = false, T::Type = Float64,
             nuc::Int = 0, nuo::Int = 0, randt = true, stable::Bool = false)    
    nf = n; Af = randn(T,nf,nf); Bf = randn(T,nf,m); Cf = randn(T,p,nf); 
    stable && (disc ? rmul!(Af,1/(rand()+opnorm(Af,1))) : Af -= I*(rand()+maximum(real(eigvals(Af)))) ) 
    Afuc = randn(T,nuc,nuc); Bfuc = zeros(T,nuc,m); Cfuc = randn(T,p,nuc);
    stable && nuc > 0 && (disc ? rmul!(Afuc,1/(rand()+opnorm(Afuc,1))) : Afuc -= I*(rand()+maximum(real(eigvals(Afuc)))) ) 
    Afuo = randn(T,nuo,nuo); Bfuo = randn(T,nuo,m); Cfuo = zeros(T,p,nuo); 
    stable && nuo > 0 && (disc ? rmul!(Afuo,1/(rand()+opnorm(Afuo,1))) : Afuo -= I*(rand()+maximum(real(eigvals(Afuo)))) ) 
    nx = nf+nuc+nuo
    Q = randt ? qr!(rand(nx,nx)).Q : I
    return DescriptorStateSpace{T}(Q'*blockdiag(Af,Afuc,Afuo)*Q, I, 
                                   Q'*[Bf;Bfuc;Bfuo], [ Cf Cfuc Cfuo ]*Q, rand(T,p,m), 
                                   disc ? -one(real(T)) : zero(real(T))) 
end
"""
    sys = rdss(n, p, m; id = [ ], disc = false, T = Float64, stable = false, nfuc = 0, iduc = [ ], 
               nfuo = 0, iduo = [ ], randlt = true, randrt = true) 

Generate a randomized descriptor state-space system `sys = (A-λE,B,C,D)` with `p` outputs and `m` inputs, with all matrices 
randomly generated of type `T`. 
The resulting `sys` is a continuous-time system if `disc = false` and a discrete-time system if `disc = true`.

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
function rdss(n::Int, p::Int, m::Int; disc::Bool = false, stable::Bool = false, T::Type = Float64, id::Vector{Int} = Int[], 
                                      nfuc::Int = 0, nfuo::Int = 0, iduc::Vector{Int} = Int[], iduo::Vector{Int} = Int[], 
                                      randlt = true, randrt = true) 
    nf = n; Af = randn(T,nf,nf); Ef = randn(T,nf,nf); Bf = randn(T,nf,m); Cf = randn(T,p,nf); 
    stable && (disc ? Af = Ef*rmul!(Af,1/(rand()+opnorm(Af,1))) : Af = Ef*(Af-I*(rand()+maximum(real(eigvals(Af))))) ) 
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
                                   disc ? -one(real(T)) : zero(real(T))) 
end
"""
    sysr = gsvselect(sys,ind)

Construct for the descriptor system `sys = (A-λE,B,C,D)` of order `n` the descriptor system  
`sysr = (A[ind,ind]-λE[ind,ind],B[ind,:],C[:,ind],D)` of order `nr = length(ind)`, 
by selecting the state variables of `sys` with indices specified by `ind`. 
If `ind` is a permutation vector of length `n`, then `sysr` has the same transfer function matrix as `sys` 
and permuted state variables. 
"""
function gsvselect(SYS::DescriptorStateSpace{T},ind::Union{UnitRange{Int64},Array{Int64,1}}) where T
    isempty(ind) &&  (return DescriptorStateSpace{T}(zeros(T,0,0),I,zeros(T,0,SYS.nu),zeros(T,SYS.ny,0),SYS.D,SYS.Ts))
    (minimum(ind) < 1 || maximum(ind) > SYS.nx) && error("BoundsError: selected indices $ind out of range $(1:SYS.nx)")
    return DescriptorStateSpace{T}(SYS.A[ind,ind], SYS.E == I ? I : SYS.E[ind,ind], SYS.B[ind,:], SYS.C[:,ind], SYS.D, SYS.Ts)
end
