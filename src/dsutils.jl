# The function to_matrix is a fork from ControlSystems.jl
# cover the case when a vector represents the row of a matrix 
to_matrix(T, A::AbstractVector, wide::Bool = false) = wide ? Matrix{T}(reshape(A, 1, length(A))) : Matrix{T}(reshape(A, length(A), 1))
to_matrix(T, A::AbstractMatrix, wide::Bool = false) = T.(A)  # Fallback
to_matrix(T, A::Number, wide::Bool = true) = fill(T(A), 1, 1)
# Handle Adjoint Matrices
to_matrix(T, A::Adjoint{R, MT}, wide::Bool = false) where {R<:Number, MT<:AbstractMatrix} = to_matrix(T, MT(A))

#
eye(n) = Matrix{Bool}(I, n, n)
eye(m,n) = Matrix{Bool}(I, m, n)
eye(::Type{T}, n) where {T} = Matrix{T}(I, n, n)
eye(::Type{T}, m, n) where {T} = Matrix{T}(I, m, n)

function rcond(A::DenseMatrix, tola::Real = 0)
    T = eltype(A)
    T1 = T <: BlasFloat ? T : T1 = promote_type(T,Float64)
    nrmA = opnorm(A,1)
    nrmA <= tola && (return zero(real(T1)))
    istriu(A) ? (return LinearAlgebra.LAPACK.trcon!('1','U','N',copy_oftype(A,T1))) : 
        (return LinearAlgebra.LAPACK.gecon!('1', LinearAlgebra.LAPACK.getrf!(copy_oftype(A,T1))[1],real(T1)(nrmA)) ) 
end
function rcond(A::UpperTriangular, tola::Real = 0) 
    T = eltype(A)
    T1 = T <: BlasFloat ? T : T1 = promote_type(T,Float64)
    nrmA = opnorm(A,1)
    nrmA <= tola && (return zero(real(T1)))
    return LinearAlgebra.LAPACK.trcon!('1','U','N', Matrix{T1}(A)) 
end
function jordanblockdiag(lambda::T, ed::Vector{Int}) where T
   n = sum(ed)
   J = zeros(T,n,n)
   i = 1
   for k = 1:length(ed)
       ni = ed[k]
       J[i:i+ni-1,i:i+ni-1] = jordanblock(ni,lambda)
       i += ni
   end
   return J
end
function jordanblock( n::Integer, lambda::T) where T
   # Generate a n-by-n elementary Jordan block with eigenvalue lambda
   J = lambda*Matrix{T}(I, n, n) + diagm(1 => ones(T, n-1, 1)[:,1])
   return J
end
blockdiag(mats::AbstractMatrix...) = blockdiag(promote(mats...)...)

function blockdiag(mats::AbstractMatrix{T}...) where T
    rows = Int[size(m, 1) for m in mats]
    cols = Int[size(m, 2) for m in mats]
    res = zeros(T, sum(rows), sum(cols))
    m = 1
    n = 1
    for ind=1:length(mats)
        mat = mats[ind]
        i = rows[ind]
        j = cols[ind]
        res[m:m + i - 1, n:n + j - 1] = mat
        m += i
        n += j
    end
    return res
end
function sblockdiag(blockdims::Vector{Int},mats::Union{AbstractMatrix{T},UniformScaling}...) where T
    nmat = sum(blockdims)
    res = zeros(T, nmat, nmat)
    i = 1
    for ind = 1:length(mats)
        mat = mats[ind]
        ni = blockdims[ind]
        i1 = i:i+ni-1
        res[i1,i1] = (mat == I) ? Matrix{T}(I,ni,ni) : mat
        i += ni
    end
    return res
end
