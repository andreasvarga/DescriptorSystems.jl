const _RationalConcatGroup = Union{Vector{RationalTransferFunction}, Matrix{RationalTransferFunction}}
const _TypedRationalConcatGroup{T} = Union{Vector{RationalTransferFunction{T}}, Matrix{RationalTransferFunction{T}}}
# rational concatenations

# scalar elements
# conversions to rational transfer functions
promote_to_rtf_(n::Int, ::Type{T}, var::Symbol, A::Number, Ts::Union{Real,Nothing}) where {T} = rtf(Polynomial{T}(T(A),var), Ts = Ts)
#promote_to_rtf_(n::Int, ::Type{T}, var::Symbol, A::Polynomial, Ts::Union{Real,Nothing}) where {T} = rtf(Polynomial{T}(T.(A.coeffs), var), Polynomial{T}(one(T), var), Ts = Ts)
promote_to_rtf_(n::Int, ::Type{T}, var::Symbol, A::RationalTransferFunction, Ts::Union{Real,Nothing}) where {T} = 
     (T == eltype(A) && Ts == A.Ts && var == A.var) ? A : rtf(Polynomial{T}(T.(A.num.coeffs), var), Polynomial{T}(T.(A.den.coeffs), var), Ts = Ts)
promote_to_rtfs(Ts::Union{Real,Nothing}, var::Symbol, n, k, ::Type{T}, A) where {T} = (promote_to_rtf_(n[k], T, var, A, Ts),)
promote_to_rtfs(Ts::Union{Real,Nothing}, var::Symbol, n, k, ::Type{T}, A, B) where {T} =
    (promote_to_rtf_(n[k], T, var, A, Ts), promote_to_rtf_(n[k+1], T, var, B, Ts))
promote_to_rtfs(Ts::Union{Real,Nothing}, var::Symbol, n, k, ::Type{T}, A, B, C) where {T} =
    (promote_to_rtf_(n[k], T, var, A, Ts), promote_to_rtf_(n[k+1], T, var, B, Ts), promote_to_rtf_(n[k+2], T, var, C, Ts))
promote_to_rtfs(Ts::Union{Real,Nothing}, var::Symbol, n, k, ::Type{T}, A, B, Cs...) where {T} =
    (promote_to_rtf_(n[k], T, var, A, Ts), promote_to_rtf_(n[k+1], T, var, B, Ts), promote_to_rtfs(Ts, var, n, k+2, T, Cs...)...)


# function Base.hcat(A::Union{AbstractRationalTransferFunction,Number}...) 
#     T = Base.promote_eltype(A...)
#     var = promote_rtf_var(A...)
#     Ts = promote_rtf_SamplingTime(A...)
#     hvcat_fill(Matrix{RationalTransferFunction{T}}(undef, 1, length(A)), promote_to_rtfs(Ts, var, fill(1,length(A)), 1, T, A...))
# end
# function Base.vcat(A::Union{AbstractRationalTransferFunction,Number}...) 
#     T = Base.promote_eltype(A...)
#     var = promote_rtf_var(A...)
#     Ts = promote_rtf_SamplingTime(A...)
#     hvcat_fill(Matrix{RationalTransferFunction{T}}(undef, length(A), 1), promote_to_rtfs(Ts, var, fill(1,length(A)), 1, T, A...))
# end
  
function Base.hvcat(rows::Tuple{Vararg{Int}}, A::RationalTransferFunction...)
    nr = length(rows)
    nc = rows[1]
    sum(rows) == length(A) || throw(ArgumentError("mismatch between row sizes and number of arguments"))
    T = Base.promote_eltype(A...)
    var = promote_rtf_var(A...)
    Ts = promote_rtf_SamplingTime(A...)
    
    hvcat_fill(Matrix{RationalTransferFunction{T,var}}(undef, nr, nc), promote_to_rtfs(Ts, var, fill(1,length(A)), 1, T, A...))
end

function Base.hvcat(rows::Tuple{Vararg{Int}}, A::Union{RationalTransferFunction,Number}...)
    nr = length(rows)
    nc = rows[1]
    sum(rows) == length(A) || throw(ArgumentError("mismatch between row sizes and number of arguments"))
    T = Base.promote_eltype(A...)
    var = promote_rtf_var(A...)
    Ts = promote_rtf_SamplingTime(A...)
    
    hvcat_fill(Matrix{RationalTransferFunction{T,var}}(undef, nr, nc), promote_to_rtfs(Ts, var, fill(1,length(A)), 1, T, A...))
end
# function Base.hvcat(rows::Tuple{Vararg{Int}}, A::Union{RationalTransferFunction,Polynomial,Number}...)
#     nr = length(rows)
#     nc = rows[1]
#     sum(rows) == length(A) || throw(ArgumentError("mismatch between row sizes and number of arguments"))
#     T = Base.promote_eltype(A...)
#     var = promote_rtf_var(A...)
#     Ts = promote_rtf_SamplingTime(A...)
    
#     hvcat_fill(Matrix{RationalTransferFunction{T,var}}(undef, nr, nc), promote_to_rtfs(Ts, var, fill(1,length(A)), 1, T, A...))
# end
function hvcat_fill(a::Array, xs::Tuple) 
    k = 1
    nr, nc = size(a,1), size(a,2)
    for i=1:nr
        @inbounds for j=1:nc
            a[i,j] = xs[k]
            k += 1
        end
    end
    return a
end


# matrix elements
# conversions to rational matrices

promote_to_rtfmat_(n::Int, ::Type{T}, var::Symbol, A::Number, Ts::Union{Real,Nothing}) where {T} = [rtf(Polynomial{T}(T(A),var),Ts=Ts)]
# promote_to_rtfmat_(n::Int, ::Type{T}, var::Symbol, A::Polynomial, Ts::Union{Real,Nothing}) where {T} = 
#     (T == eltype(A) && var == A.num.var) ? [rtf(A,Ts=Ts)] : [rtf(Polynomial{T}(A.coeffs, var),Ts=Ts)]
promote_to_rtfmat_(n::Int, ::Type{T}, var::Symbol, A::RationalTransferFunction, Ts::Union{Real,Nothing}) where {T} = 
     (T == eltype(A) && Ts == A.Ts && var == A.var) ? [A] : [rtf(Polynomial{T}(T.(A.num.coeffs), var), Polynomial{T}(T.(A.den.coeffs), var), Ts = Ts)]
promote_to_rtfmat_(n::Int, ::Type{T}, var::Symbol, J::UniformScaling, Ts::Union{Real,Nothing}) where {T} = rtf.(Polynomial.(copyto!(Matrix{T}(undef, n,n), J),var),Ts=Ts)
promote_to_rtfmat_(n::Int, ::Type{T}, var::Symbol, A::AbstractVecOrMat{T1}, Ts::Union{Real,Nothing}) where {T, T1 <: Number} = rtf.(Polynomial.(to_matrix(T,A),var),Ts=Ts)
# promote_to_rtfmat_(n::Int, ::Type{T}, var::Symbol, A::VecOrMat{Polynomial{T1}}, Ts::Union{Real,Nothing}) where {T,T1} = 
#      (T == T1 && var == A[1].var) ? rtf.(A,Ts = Ts) : rtf.(Polynomial{T}.(coeffs.(A),var),Ts = Ts)
promote_to_rtfmat_(n::Int, ::Type{T}, var::Symbol, A::VecOrMat{<:RationalTransferFunction}, Ts::Union{Real,Nothing}) where T = 
     (length(A) > 0 && _eltype(A) == T && var == A[1].var && Ts == A[1].Ts) ? A : rtf.(Polynomial{T}.(coeffs.(numpoly.(A)),var),Polynomial{T}.(coeffs.(denpoly.(A)),var),Ts = Ts)
promote_to_rtfmats(Ts::Union{Real,Nothing}, var::Symbol, n, k, ::Type{T}, A) where {T} = (promote_to_rtfmat_(n[k], T, var, A, Ts),)
promote_to_rtfmats(Ts::Union{Real,Nothing}, var::Symbol, n, k, ::Type{T}, A, B) where {T} =
    (promote_to_rtfmat_(n[k], T, var, A, Ts), promote_to_rtfmat_(n[k+1], T, var, B, Ts))
promote_to_rtfmats(Ts::Union{Real,Nothing}, var::Symbol, n, k, ::Type{T}, A, B, C) where {T} =
    (promote_to_rtfmat_(n[k], T, var, A, Ts), promote_to_rtfmat_(n[k+1], T, var, B, Ts), promote_to_rtfmat_(n[k+2], T, var, C, Ts))
promote_to_rtfmats(Ts::Union{Real,Nothing}, var::Symbol, n, k, ::Type{T}, A, B, Cs...) where {T} =
    (promote_to_rtfmat_(n[k], T, var, A, Ts), promote_to_rtfmat_(n[k+1], T, var, B, Ts), promote_to_rtfmats(Ts, var, n, k+2, T, Cs...)...)

function Base.hcat(A::VecOrMat{<:RationalTransferFunction}...) 
    n = -1
    for a in A
        require_one_based_indexing(a); na = size(a, 1)
        n >= 0 && n != na &&
            throw(DimensionMismatch(string("number of rows of each array must match (got ", n, " and ", na, ")")))
        n = na
    end
    Tc = promote_rtf_eltype(A...)
    var = promote_rtf_var(A...)
    Ts = promote_rtf_SamplingTime(A...)
    return _hcat(RationalTransferFunction{Tc,var},promote_to_rtfmats(Ts, var, fill(n,length(A)), 1, Tc, A...)...)
end
function Base.hcat(A::Union{VecOrMat{<:RationalTransferFunction}, UniformScaling}...) 
    n = -1
    for a in A
        if !isa(a, UniformScaling)
            require_one_based_indexing(a); na = size(a, 1)
            n >= 0 && n != na &&
                throw(DimensionMismatch(string("number of rows of each array must match (got ", n, " and ", na, ")")))
            n = na
        end
    end
    n == -1 && throw(ArgumentError("hcat of only UniformScaling objects cannot determine the matrix size"))
    Tc = promote_rtf_eltype(A...)
    var = promote_rtf_var(A...)
    Ts = promote_rtf_SamplingTime(A...)
    Tp = promote_rtf_type(A...)
    return _hcat(RationalTransferFunction{Tc,var},promote_to_rtfmats(Ts, var, fill(n,length(A)), 1, Tc, A...)...)
end
function Base.hcat(A::Union{VecOrMat{<:RationalTransferFunction},RationalTransferFunction,Number,UniformScaling}...) 
    n = -1
    for a in A
        if !isa(a, UniformScaling)
            isa(a,Union{RationalTransferFunction,Polynomial,Number}) ? na = 1 : (require_one_based_indexing(a); na = size(a, 1))
            n >= 0 && n != na &&
                throw(DimensionMismatch(string("number of rows of each array must match (got ", n, " and ", na, ")")))
            n = na
        end
    end
    n == -1 && throw(ArgumentError("hcat of only UniformScaling objects cannot determine the matrix size"))
    Tc = promote_rtf_eltype(A...)
    var = promote_rtf_var(A...)
    Ts = promote_rtf_SamplingTime(A...)
    return _hcat(RationalTransferFunction{Tc,var},promote_to_rtfmats(Ts, var, fill(n,length(A)), 1, Tc, A...)...)
end
function _hcat(::Type{T},A::AbstractVecOrMat...) where T
    nargs = length(A)
    nrows = size(A[1], 1)
    ncols = 0
    for j = 1:nargs
        Aj = A[j]
        size(Aj, 1) != nrows &&
            throw(ArgumentError("number of rows of each array must match (got $(map(x->size(x,1), A)))"))
        nd = ndims(Aj)
        ncols += (nd==2 ? size(Aj,2) : 1)
    end
    B = similar(A[1], T, nrows, ncols)
    pos = 1
    for k=1:nargs
        Ak = A[k]
        n = length(Ak)
        copyto!(B, pos, Ak, 1, n)
        pos += n
    end
    return B
end
function Base.vcat(A::VecOrMat{<:RationalTransferFunction}...) 
    n = -1
    for a in A
        require_one_based_indexing(a); na = size(a,2)
        n >= 0 && n != na &&
            throw(DimensionMismatch(string("number of columns of each array must match (got ", n, " and ", na, ")")))
        n = na
    end
    Tc = promote_rtf_eltype(A...)
    var = promote_rtf_var(A...)
    Ts = promote_rtf_SamplingTime(A...)
    return _vcat(RationalTransferFunction{Tc,var},promote_to_rtfmats(Ts, var, fill(n,length(A)), 1, Tc, A...)...)
end
function Base.vcat(A::Union{VecOrMat{<:RationalTransferFunction},UniformScaling}...) 
    n = -1
    for a in A
        if !isa(a, UniformScaling)
            isa(a,Union{RationalTransferFunction,Polynomial,Number}) ? na = 1 : (require_one_based_indexing(a); na = size(a, 2))
            n >= 0 && n != na &&
                throw(DimensionMismatch(string("number of columns of each array must match (got ", n, " and ", na, ")")))
            n = na
        end
    end
    n == -1 && throw(ArgumentError("vcat of only UniformScaling objects cannot determine the matrix size"))
    Tc = promote_rtf_eltype(A...)
    var = promote_rtf_var(A...)
    Ts = promote_rtf_SamplingTime(A...)
    return _vcat(RationalTransferFunction{Tc,var},promote_to_rtfmats(Ts, var, fill(n,length(A)), 1, Tc, A...)...)
end

function Base.vcat(A::Union{VecOrMat{<:RationalTransferFunction},RationalTransferFunction,Number,UniformScaling}...) 
    n = -1
    for a in A
        if !isa(a, UniformScaling)
            isa(a,Union{RationalTransferFunction,Polynomial,Number}) ? na = 1 : (require_one_based_indexing(a); na = size(a, 2))
            n >= 0 && n != na &&
                throw(DimensionMismatch(string("number of columns of each array must match (got ", n, " and ", na, ")")))
            n = na
        end
    end
    n == -1 && throw(ArgumentError("vcat of only UniformScaling objects cannot determine the matrix size"))
    Tc = promote_rtf_eltype(A...)
    var = promote_rtf_var(A...)
    Ts = promote_rtf_SamplingTime(A...)
    return _vcat(RationalTransferFunction{Tc,var},promote_to_rtfmats(Ts, var, fill(n,length(A)), 1, Tc, A...)...)
end
function _vcat(::Type{T},A::AbstractVecOrMat...) where T
    nargs = length(A)
    nrows = sum(a->size(a, 1), A)::Int
    ncols = size(A[1], 2)
    for j = 2:nargs
        if size(A[j], 2) != ncols
            throw(ArgumentError("number of columns of each array must match (got $(map(x->size(x,2), A)))"))
        end
    end
    B = similar(A[1], T, nrows, ncols)
    pos = 1
    for k=1:nargs
        Ak = A[k]
        p1 = pos+size(Ak,1)::Int-1
        B[pos:p1, :] = Ak
        pos = p1+1
    end
    return B
end
function Base.hvcat(rows::Tuple{Vararg{Int}}, A::Union{AbstractVecOrMat{T1},AbstractVecOrMat{T2},AbstractVecOrMat{T3},RationalTransferFunction,UniformScaling}...) where 
    {T1 <: RationalTransferFunction, T2 <: Polynomial, T3 <: Number}
    nr = length(rows)
    sum(rows) == length(A) || throw(ArgumentError("mismatch between row sizes and number of arguments"))
    n = fill(-1, length(A))
    needcols = false # whether we also need to infer some sizes from the column count
    j = 0
    for i = 1:nr # infer UniformScaling sizes from row counts, if possible:
        ni = -1 # number of rows in this block-row, -1 indicates unknown
        for k = 1:rows[i]
            if !isa(A[j+k], UniformScaling)
                isa(A[j+k],Union{RationalTransferFunction,Polynomial,Number}) ? na = 1 : 
                      (require_one_based_indexing(A[j+k]); na = size(A[j+k], 1))
                ni >= 0 && ni != na &&
                    throw(DimensionMismatch("mismatch in number of rows"))
                ni = na
            end
        end
        if ni >= 0
            for k = 1:rows[i]
                n[j+k] = ni
            end
        else # row consisted only of UniformScaling objects
            needcols = true
        end
        j += rows[i]
    end
    if needcols # some sizes still unknown, try to infer from column count
        nc = -1
        j = 0
        for i = 1:nr
            nci = 0
            rows[i] > 0 && n[j+1] == -1 && (j += rows[i]; continue)
            for k = 1:rows[i]
                nci += isa(A[j+k], UniformScaling) ? n[j+k] : 
                      (isa(A[j+k],Union{RationalTransferFunction,Polynomial,Number}) ? 1 : size(A[j+k], 2))
            end
            nc >= 0 && nc != nci && throw(DimensionMismatch("mismatch in number of columns"))
            nc = nci
            j += rows[i]
        end
        nc == -1 && throw(ArgumentError("sizes of UniformScalings could not be inferred"))
        j = 0
        for i = 1:nr
            if rows[i] > 0 && n[j+1] == -1 # this row consists entirely of UniformScalings
                nci = nc ÷ rows[i]
                nci * rows[i] != nc && throw(DimensionMismatch("indivisible UniformScaling sizes"))
                for k = 1:rows[i]
                    n[j+k] = nci
                end
            end
            j += rows[i]
        end
    end
    Tc = promote_rtf_eltype(A...)
    var = promote_rtf_var(A...)
    Ts = promote_rtf_SamplingTime(A...)
    Tp = promote_rtf_type(A...)
    if Tp == RationalTransferFunction || Tp == Polynomial
        return _hvcat(RationalTransferFunction{Tc,var},rows,promote_to_rtfmats(Ts,var, n, 1, Tc, A...)...)
    else
       return _hvcat(Tc,rows,LinearAlgebra.promote_to_arrays(n, 1, Matrix, A...)...)
    end
end
function _hvcat(::Type{T}, rows::Tuple{Vararg{Int}}, as::AbstractVecOrMat...) where T 
    nbr = length(rows)  # number of block rows

    nc = 0
    for i=1:rows[1]
        nc += size(as[i],2)
    end

    nr = 0
    a = 1
    for i = 1:nbr
        nr += size(as[a],1)
        a += rows[i]
    end

    out = similar(as[1], T, nr, nc)

    a = 1
    r = 1
    for i = 1:nbr
        c = 1
        szi = size(as[a],1)
        for j = 1:rows[i]
            Aj = as[a+j-1]
            szj = size(Aj,2)
            if size(Aj,1) != szi
                throw(ArgumentError("mismatched height in block row $(i) (expected $szi, got $(size(Aj,1)))"))
            end
            if c-1+szj > nc
                throw(ArgumentError("block row $(i) has mismatched number of columns (expected $nc, got $(c-1+szj))"))
            end
            out[r:r-1+szi, c:c-1+szj] = Aj
            c += szj
        end
        if c != nc+1
            throw(ArgumentError("block row $(i) has mismatched number of columns (expected $nc, got $(c-1))"))
        end
        r += szi
        a += rows[i]
    end
    out
end

function promote_rtf_SamplingTime(A...)  
    # pick and check the common sampling time 
    Ts = nothing
    for a in A
        if eltype(a) <: RationalTransferFunction 
           length(a) > 0 && (Ts = promote_Ts(Ts,a[1].Ts))
        elseif typeof(a) <: RationalTransferFunction 
            Ts = promote_Ts(Ts,a.Ts)
        end
    end
    return Ts   # for transfer functions Ts = nothing is also allowed
end
function promote_rtf_eltype(A...)  
    T = Bool
    for a in A
        if eltype(a) <: RationalTransferFunction || eltype(a) <: Polynomial
           T = promote_type(T,_eltype(a))
        else
           T = promote_type(T,eltype(a)) 
        end
    end
    return T
end
function promote_rtf_type(A...)  
    T2 = Bool
    for a in A
        (eltype(a) <: RationalTransferFunction || typeof(a) <: RationalTransferFunction) && (return RationalTransferFunction)
        (eltype(a) <: Polynomial || typeof(a) <: Polynomial) && (T2 = Polynomial)
    end
    return T2
end
function promote_rtf_var(A...) 
    var = nothing
    for a in A
        if eltype(a) <: RationalTransferFunction && length(a) > 0
           t = a[1].var
           isnothing(var) ? (var = t) : 
                            (t != var && error("all transfer function matrix elements must have the same variable"))
        elseif eltype(a) <: Polynomial && length(a) > 0
            t = Polynomials.indeterminate(a[1])
            isnothing(var) ? (var = t) : 
               (t != var && error("all transfer function matrix elements must have the same variable"))
        elseif typeof(a) <: RationalTransferFunction 
            t = a.var
            isnothing(var) ? (var = t) : 
                   (t != var && error("all transfer function matrix elements must have the same variable"))
        elseif typeof(a) <: Polynomial 
            t = Polynomials.indeterminate(a)
            isnothing(var) ? (var = t) : 
                   (t != var && error("all transfer function matrix elements must have the same variable"))
        end
    end
    return isnothing(var) ? (:x) : var  # use default symbol from the Polynomials package
end
