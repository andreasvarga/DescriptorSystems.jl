const AbstractNumOrArray = Union{AbstractVecOrMat,Number}
""" 
    DescriptorStateSpace{T}(A::Matrix{T}, E::Union{Matrix{T},UniformScaling}, 
                            B::Matrix{T}, C::Matrix{T}, D::Matrix{T},  
                            Ts::Real) where T <: Number

Construct a descriptor state-space model from a quintuple of matrices `(A,E,B,C,D)` and a sampling time `Ts`.

If `SYS::DescriptorStateSpace{T}` is a descriptor system model object 
defined by the 4-tuple `SYS = (A-λE,B,C,D)`, then:

`SYS.A` is the `nx × nx` state matrix `A` with elements of type `T`. 

`SYS.E` is the `nx × nx` descriptor matrix `E` with elements of type `T`.
 For a standard state-space system `SYS.E = I`, the `UniformScaling` of type `Bool`. 

`SYS.B` is the `nx × nu` system input matrix `B` with elements of type `T`. 

`SYS.C` is the `ny × nx` system output matrix `C` with elements of type `T`. 

`SYS.D` is the `ny × nu` system feedthrough matrix `D` with elements of type `T`. 

`SYS.Ts` is the real sampling time `Ts`, where `Ts = 0` for a continuous-time system,
    and `Ts > 0` or `Ts = -1` for a discrete-time system. 
    `Ts = -1` indicates a discrete-time system with an unspecified sampling time. 

`SYS.nx` is the system state vector dimension `nx`. 

`SYS.ny` is the system output vector dimension `ny`. 

`SYS.nu` is the system input vector dimension `nu`. 
"""
struct DescriptorStateSpace{T} <: AbstractLTISystem
    A::Matrix{T}
    E::Union{Matrix{T},UniformScaling}
    B::Matrix{T}
    C::Matrix{T}
    D::Matrix{T}
    Ts::Float64
    nx::Int
    nu::Int
    ny::Int
    function DescriptorStateSpace{T}(A::Matrix{T}, E::Union{Matrix{T},UniformScaling}, 
                                     B::Matrix{T}, C::Matrix{T}, D::Matrix{T},  Ts::Real) where T 
        nx, nu, ny = dss_validation(A, E, B, C, D, Ts)
        new{T}(A, E, B, C, D, Float64(Ts), nx, nu, ny)
    end
end
function dss_validation(A::Matrix{T}, E::Union{Matrix{T},UniformScaling}, 
                        B::Matrix{T}, C::Matrix{T}, D::Matrix{T},  Ts::Real) where T
    nx = LinearAlgebra.checksquare(A)
    (ny, nu) = size(D)
    
    # Validate dimensions
    if typeof(E) <: Matrix
       nx == LinearAlgebra.checksquare(E) || error("A and E must have the same size")
    end
    size(B, 1) == nx ||  error("B must have the same row size as A")
    size(C, 2) == nx ||  error("C must have the same column size as A")
    nu == size(B, 2) ||  error("D must have the same column size as B")
    ny == size(C, 1) ||  error("D must have the same row size as C")
 
    # Validate sampling time
    Ts >= 0 || Ts == -1 || error("Ts must be either a positive number, 0
                                (continuous system), or -1 (unspecified)")
    return nx, nu, ny
end

# Base.ndims(::AbstractDescriptorStateSpace) = 2 
# Base.size(sys::AbstractDescriptorStateSpace) = size(sys.D) 
# Base.length(sys::AbstractDescriptorStateSpace) = length(sys.D) 
# Base.size(sys::AbstractDescriptorStateSpace, d::Integer) = d <= 2 ? size(sys)[d] : 1
Base.eltype(sys::DescriptorStateSpace) = eltype(sys.A)

function Base.getindex(sys::DST, inds...) where DST <: DescriptorStateSpace
    size(inds, 1) != 2 &&
        error("Must specify 2 indices to index descriptor state-space model")
    rows, cols = index2range(inds...) 
    return DescriptorStateSpace{eltype(sys)}(copy(sys.A), copy(sys.E), sys.B[:, cols], sys.C[rows, :], sys.D[rows, cols], sys.Ts)
end
index2range(ind1, ind2) = (index2range(ind1), index2range(ind2))
index2range(ind::T) where {T<:Number} = ind:ind
index2range(ind::T) where {T<:AbstractArray} = ind
index2range(ind::Colon) = ind
function Base.lastindex(sys::DST, dim::Int) where DST <: DescriptorStateSpace
    lastindex(sys.D,dim)
end

# Basic Operations
function ==(sys1::DST1, sys2::DST2) where {DST1<:DescriptorStateSpace, DST2<:DescriptorStateSpace}
    # fieldnames(DST1) == fieldnames(DST2) || (return false)
    return all(getfield(sys1, f) == getfield(sys2, f) for f in fieldnames(DST1))
end

function isapprox(sys1::DST1, sys2::DST2; atol = zero(real(eltype(sys1))), 
                  rtol = rtol::Real =  ((max(size(sys1.A)...))+1)*eps(real(float(one(real(eltype(sys1))))))*iszero(atol)) where 
                  {DST1<:DescriptorStateSpace,DST2<:DescriptorStateSpace}
    #fieldnames(DST1) == fieldnames(DST2) || (return false)
    return all(isapprox(getfield(sys1, f), getfield(sys2, f); atol = atol, rtol = rtol) for f in fieldnames(DST1))
end

# sum sys1+sys2
function +(sys1::DescriptorStateSpace{T1}, sys2::DescriptorStateSpace{T2}) where {T1,T2}
    #Ensure systems have same dimensions and sampling times
    size(sys1) == size(sys2) || error("The systems have different shapes.")
    Ts = promote_Ts(sys1.Ts,sys2.Ts)

    T = promote_type(T1, T2)
    n1 = sys1.nx
    n2 = sys2.nx
    A = [sys1.A  zeros(T,n1,n2);
         zeros(T,n2,n1) sys2.A]
    B = [sys1.B ; sys2.B]
    C = [sys1.C sys2.C;]
    D = [sys1.D + sys2.D;]
    if sys1.E == I && sys2.E == I
        E = I
    else
        E = [sys1.E  zeros(T,n1,n2);
             zeros(T,n2,n1) sys2.E]
    end

    return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
end
# difference sys1-sys2
function -(sys1::DescriptorStateSpace{T1}, sys2::DescriptorStateSpace{T2}) where {T1,T2}
    #Ensure systems have same dimensions and sampling times
    size(sys1) == size(sys2) || error("The systems have different shapes.")
    Ts = promote_Ts(sys1.Ts,sys2.Ts)

    T = promote_type(T1, T2)
    n1 = sys1.nx
    n2 = sys2.nx
 
    A = [sys1.A  zeros(T,n1,n2);
         zeros(T,n2,n1) sys2.A]
    B = [sys1.B ; sys2.B]
    C = [sys1.C -sys2.C;]
    D = [sys1.D - sys2.D;]
    if sys1.E == I && sys2.E == I
        E = I
    else
        E = [sys1.E  zeros(T,n1,n2);
             zeros(T,n2,n1) sys2.E]
    end

    return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
end
# negation -sys
function -(sys::DescriptorStateSpace{T}) where T
    return DescriptorStateSpace{T}(sys.A, sys.E, sys.B, -sys.C, -sys.D, sys.Ts)
end
# sys+mat and mat+sys
function +(sys::DescriptorStateSpace{T1}, mat::VecOrMat{T2}) where {T1,T2}
    p, m = typeof(mat) <: Vector ? (length(mat),1) : size(mat)
    size(sys) == (p, m) || error("The input-output dimensions of system does not match the shape of matrix.")
    T = promote_type(T1, T2)
    return DescriptorStateSpace{T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
                                   copy_oftype(sys.D,T) + mat, sys.Ts)
end
+(mat::VecOrMat{T1}, sys::DescriptorStateSpace{T2}) where {T1,T2} = +(sys,mat)
# sys+I and I+sys
function +(sys::DescriptorStateSpace{T1}, mat::UniformScaling{T2}) where {T1,T2}
    size(sys,1) == size(sys,2) || error("The system must have the same number of inputs and outputs")
    T = promote_type(T1, T2)
    return DescriptorStateSpace{T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
                                   copy_oftype(sys.D,T) + mat, sys.Ts)
end

# sys-mat and mat-sys
function -(sys::DescriptorStateSpace{T1}, mat::VecOrMat{T2}) where {T1,T2}
    p, m = typeof(mat) <: Vector ? (length(mat),1) : size(mat)
    size(sys) == (p, m) || error("The input-output dimensions of system does not match the shape of matrix.")
    T = promote_type(T1, T2)
    return DescriptorStateSpace{T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
                                   copy_oftype(sys.D,T) - mat, sys.Ts)
end
-(mat::VecOrMat{T1}, sys::DescriptorStateSpace{T2}) where {T1,T2} = +(-sys,mat)

# sys-I and I-sys
function -(sys::DescriptorStateSpace{T1}, mat::UniformScaling{T2}) where {T1,T2}
    size(sys,1) == size(sys,2) || error("The system must have the same number of inputs and outputs")
    T = promote_type(T1, T2)
    return DescriptorStateSpace{T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
                                   copy_oftype(sys.D,T) - mat, sys.Ts)
end
+(mat::UniformScaling{T1}, sys::DescriptorStateSpace{T2}) where {T1,T2} = +(sys,mat)
-(mat::UniformScaling{T1}, sys::DescriptorStateSpace{T2}) where {T1,T2} = +(-sys,mat)
# sys ± n and n ± sys
function +(sys::DescriptorStateSpace{T1}, n::Number) where T1 
    T = promote_type(T1, eltype(n))
    return DescriptorStateSpace{T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
                                   copy_oftype(sys.D,T) .+ n, sys.Ts)
end
+(n::Number, sys::DescriptorStateSpace{T1}) where T1 = +(sys, n)
-(n::Number, sys::DescriptorStateSpace{T1}) where T1 = +(-sys, n)
-(sys::DescriptorStateSpace{T1},n::Number) where T1 = +(sys, -n)

# multiplication sys1*sys2
function *(sys1::DescriptorStateSpace{T1}, sys2::DescriptorStateSpace{T2}) where {T1,T2}
    sys1.nu == sys2.ny || error("sys1 must have same number of inputs as sys2 has outputs")
    Ts = promote_Ts(sys1.Ts, sys2.Ts)
    T = promote_type(T1, T2)
    n1 = sys1.nx
    n2 = sys2.nx

    A = [sys1.A    sys1.B*sys2.C;
         zeros(T,n2,n1)   sys2.A]
    B = [sys1.B*sys2.D ; sys2.B]
    C = [sys1.C   sys1.D*sys2.C;]
    D = [sys1.D*sys2.D;]
    if sys1.E == I && sys2.E == I
        E = I
    else
        E = [sys1.E  zeros(T,n1,n2);
             zeros(T,n2,n1) sys2.E]
    end

    return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
end
# sys*mat
function *(sys::DescriptorStateSpace{T1}, mat::VecOrMat{T2}) where {T1,T2}
    p, m = typeof(mat) <: Vector ? (length(mat),1) : size(mat)
    sys.nu == p || error("The input dimension of system does not match the number of rows of the matrix.")
    T = promote_type(T1, T2)
    return DescriptorStateSpace{T}(copy_oftype(sys.A,T), 
                                   sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T)*to_matrix(T,mat), copy_oftype(sys.C,T), 
                                   copy_oftype(sys.D,T)*to_matrix(T,mat), sys.Ts)
end
# mat*sys
function *(mat::VecOrMat{T1}, sys::DescriptorStateSpace{T2}) where {T1,T2}
    p, m = typeof(mat) <: Vector ? (length(mat),1) : size(mat)
    sys.ny == m || error("The output dimension of system does not match the number of columns of the matrix.")
    T = promote_type(T1, T2)
    return DescriptorStateSpace{T}(copy_oftype(sys.A,T), 
                                   sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T), to_matrix(T,mat)*copy_oftype(sys.C,T), 
                                   to_matrix(T,mat)*copy_oftype(sys.D,T), sys.Ts)
end
# sI*sys
function *(s::Union{UniformScaling,Number}, sys::DescriptorStateSpace{T}) where T
    T1 = promote_type(eltype(s),T)
    return DescriptorStateSpace{T1}(copy_oftype(sys.A,T1), sys.E == I ? I : copy_oftype(sys.E,T1), copy_oftype(sys.B,T1), 
                                   lmul!(s,copy_oftype(sys.C,T1)), lmul!(s,copy_oftype(sys.D,T1)), sys.Ts)

end
# sys*sI
function *(sys::DescriptorStateSpace{T},s::Union{UniformScaling,Number}) where T
    T1 = promote_type(eltype(s),T)
    return DescriptorStateSpace{T1}(copy_oftype(sys.A,T1), sys.E == I ? I : copy_oftype(sys.E,T1), rmul!(copy_oftype(sys.B,T1),s), 
                                   copy_oftype(sys.C,T1), rmul!(copy_oftype(sys.D,T1),s), sys.Ts)

end

# right division sys1/sys2 = sys1*inv(sys2)
/(n::Union{UniformScaling,Number}, sys::DescriptorStateSpace) = n*inv(sys)
/(sys::DescriptorStateSpace, n::Number) = sys*(1/n)
/(sys::DescriptorStateSpace, n::UniformScaling) = sys*(1/n.λ)

# left division sys1\sys2 = inv(sys1)*sys2
\(n::Number, sys::DescriptorStateSpace) = (1/n)*sys
\(n::UniformScaling, sys::DescriptorStateSpace) = (1/n.λ)*sys
\(sys::DescriptorStateSpace, n::Union{UniformScaling,Number}) = inv(sys)*n

# display sys
Base.print(io::IO, sys::DescriptorStateSpace) = show(io, sys)
Base.show(io::IO, sys::DescriptorStateSpace) = show(io, MIME("text/plain"), sys)

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, sys::DescriptorStateSpace)
    summary(io, sys); println(io)
    n = size(sys.A,1) 
    typeof(sys.D) <: Vector ? ((p,m) = (length(sys.D),1)) : ((p, m) = size(sys.D))
    if n > 0
       println(io, "\nState matrix A:")
       show(io, mime, sys.A)
       if sys.E != I
          println(io, "\n\nDescriptor matrix E:")
          show(io, mime, sys.E)
       end
       m > 0 ? println(io, "\n\nInput matrix B:") : println(io, "\n\nEmpty input matrix B.")
       show(io, mime, sys.B)
       p > 0 ? println(io, "\n\nOutput matrix C:") : println(io, "\n\nEmpty output matrix C.") 
       show(io, mime, sys.C)
       (m > 0 && p > 0) ? println(io, "\n\nFeedthrough matrix D:") : println(io, "\n\nEmpty feedthrough matrix D.") 
       show(io, mime, sys.D)
       iszero(sys.Ts) ? println(io, "\n\nContinuous-time state-space model.") :
                      println(io, "\n\nDiscrete-time state-space model") 
       sys.Ts < 0 && println(io, "Sample time: unspecified.")
       sys.Ts > 0 && println(io, "Sample time: $(sys.Ts) second(s).")
    elseif m > 0 && p > 0
       println(io, "\nFeedthrough matrix D:")
       show(io, mime, sys.D)
       println(io, "\nStatic gain.") 
    else
       println(io, "\nEmpty state-space model.")
    end
end
 

