#const AbstractNumOrArray = Union{AbstractVecOrMat{T},Number} where {T <: Number}
const AbstractNumOrArray = Union{AbstractVecOrMat,Number}
#const ETYPE{T} = Union{Matrix{T},UniformScaling}
#const ETYPE{T} = Union{Matrix{T},UniformScaling{Bool}}
const ETYPE{T} = Union{AbstractMatrix{T},UniformScaling{Bool}}
promote_Etype(T0::Type, ::Type{UniformScaling{Bool}}, ::Type{UniformScaling{Bool}}) = UniformScaling{Bool}
promote_Etype(T0::Type, T1::Type, ::Type{UniformScaling{Bool}}) = Matrix{promote_type(T0,eltype(T1))}
promote_Etype(T0::Type, ::Type{UniformScaling{Bool}}, T1::Type) = Matrix{promote_type(T0,eltype(T1))}
promote_Etype(T0::Type, T1::Type, T2::Type) = Matrix{promote_type(T0, eltype(T1), eltype(T2))}   
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

The dimensions `nx`, `ny` and `nu` can be obtained as `SYS.nx`, `SYS.ny` and `SYS.nu`, respectively. 
"""
struct DescriptorStateSpace{T, ET <: ETYPE{T}, MT<:AbstractMatrix{T}} <: AbstractDescriptorStateSpace
    A::MT
    E::ET
    B::MT
    C::MT
    D::MT
    Ts::Float64
    function DescriptorStateSpace{T}(A::MT, E::ETYPE{T},
                                     B::MT, C::MT, D::MT,  Ts::Real) where {T, MT<:AbstractMatrix{T}}
        dss_validation(A, E, B, C, D, Ts)
        new{T, typeof(E), MT}(A, E, B, C, D, Float64(Ts))
    end
end
struct DescriptorStateSpaceExt{T, AT<:AbstractMatrix, ET<:Union{AbstractMatrix,UniformScaling{Bool}}, BT<:AbstractMatrix, CT<:AbstractMatrix, DT<:AbstractMatrix} <: AbstractDescriptorStateSpace
    A::AT
    E::ET
    B::BT
    C::CT
    D::DT
    Ts::Float64
    function DescriptorStateSpaceExt{T}(A::AT, E::ET, B::BT, C::CT, D::DT, Ts::Real) where {T, AT<:AbstractMatrix{T}, ET<:Union{AbstractMatrix{T},UniformScaling{Bool}}, BT<:AbstractMatrix{T}, CT<:AbstractMatrix{T}, DT<:AbstractMatrix{T}}
       dss_validation(A, E, B, C, D, Ts)
       new{T,AT,ET,BT,CT,DT}(A, E, B, C, D, Float64(Ts))
    end
end
struct SparseDescriptorStateSpace{T, SM <: SparseMatrixCSC{T}} <: AbstractDescriptorStateSpace
    A::SM
    E::Union{SM,UniformScaling{Bool}}
    B::SM
    C::SM
    D::SM
    Ts::Float64
    function SparseDescriptorStateSpace{T}(A::SM, E::Union{SM,UniformScaling{Bool}}, 
                                     B::SM, C::SM, D::SM, Ts::Real) where {T, SM <: SparseMatrixCSC{T}}
        dss_validation(A, E, B, C, D, Ts)
        new{T,SM}(A, E, B, C, D, Float64(Ts))
    end
end

const DSTYPE{T} = Union{DescriptorStateSpace{T},DescriptorStateSpaceExt{T},SparseDescriptorStateSpace{T}}
const DSTYPEX{T} = Union{DescriptorStateSpace{T},DescriptorStateSpaceExt{T}}

#DS(sys) = getfield(DescriptorSystems,typeof(sys).name.name){eltype(sys)}
#DS(sys) = getfield(DescriptorSystems,typeof(sys).name.name)
DS(sys::DescriptorStateSpace) = DescriptorStateSpace
DS(sys::SparseDescriptorStateSpace) = SparseDescriptorStateSpace
DS(sys::DescriptorStateSpaceExt) = DescriptorStateSpaceExt

function dss_validation(A::AbstractMatrix{T}, E::ETYPE{T}, 
                        B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T},  Ts::Real) where T
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

function Base.getproperty(sys::DSTYPE, d::Symbol)  
    if d === :nx
        return size(getfield(sys, :A), 1)
    elseif d === :ny
        return size(getfield(sys, :C), 1)
    elseif d === :nu
        return size(getfield(sys, :B), 2)
    else
        getfield(sys, d)
    end
end

# Base.propertynames(sys::DescriptorStateSpace) =
#     (:nx, :ny, :nu, fieldnames(typeof(sys))...)
Base.propertynames(sys::DSTYPE) =
    (:nx, :ny, :nu, fieldnames(typeof(sys))...)

SparseArrays.issparse(sys::DSTYPE) =  isa(sys,DescriptorSystems.SparseDescriptorStateSpace) 


# Base.ndims(::AbstractDescriptorStateSpace) = 2 
# Base.size(sys::AbstractDescriptorStateSpace) = size(sys.D) 
# Base.length(sys::AbstractDescriptorStateSpace) = length(sys.D) 
# Base.size(sys::AbstractDescriptorStateSpace, d::Integer) = d <= 2 ? size(sys)[d] : 1
#Base.eltype(sys::DescriptorStateSpace) = eltype(sys.A)
Base.eltype(sys::DSTYPE) = eltype(sys.A)

function Base.getindex(sys::DST, inds...) where DST <: DSTYPE
    size(inds, 1) != 2 &&
        error("Must specify 2 indices to index descriptor state-space models")
    rows, cols = index2range(inds...) 
    #DS = getfield(DescriptorSystems,typeof(sys).name.name)
    return DS(sys){eltype(sys)}(copy(sys.A), copy(sys.E), sys.B[:, cols], sys.C[rows, :], sys.D[rows, cols], sys.Ts)
end
index2range(ind1, ind2) = (index2range(ind1), index2range(ind2))
index2range(ind::T) where {T<:Number} = ind:ind
index2range(ind::T) where {T<:AbstractArray} = ind
index2range(ind::Colon) = ind
function Base.lastindex(sys::DST, dim::Int) where DST <: DSTYPE
    lastindex(sys.D,dim)
end
# Basic Operations
function ==(sys1::DST1, sys2::DST2) where {DST1<:DSTYPE, DST2<:DSTYPE}
    return all(getfield(sys1, f) == getfield(sys2, f) for f in fieldnames(DST1))
end


function isapprox(sys1::DST1, sys2::DST2; atol = zero(real(eltype(sys1))), 
                  rtol::Real =  ((max(size(sys1.A)...))+1)*eps(real(float(one(real(eltype(sys1))))))*iszero(atol)) where 
                  {DST1<:DSTYPE, DST2<:DSTYPE}
    #fieldnames(DST1) == fieldnames(DST2) || (return false)
    return all(isapprox(getfield(sys1, f), getfield(sys2, f); atol = atol, rtol = rtol) for f in fieldnames(DST1))
end

Base.copy(sys::DSTYPE) = DS(sys){eltype(sys)}(copy(sys.A), copy(sys.E), copy(sys.B), copy(sys.C), copy(sys.D), sys.Ts)
convert(::Type{DescriptorStateSpace{T,Matrix{T}}}, sys::DescriptorStateSpace{T,UniformScaling{Bool}}) where T = DescriptorStateSpace{T}(copy(sys.A), eye(T,sys.nx), copy(sys.B), copy(sys.C), copy(sys.D), sys.Ts)

# sum sys1+sys2
function +(sys1::DSTYPE{T1}, sys2::DSTYPE{T2}) where {T1,T2}
    sys1.nu == 1 && sys1.ny == 1 && (sys2.ny > 1 || sys2.nu > 1) && (return ones(T1,sys2.ny,1)*sys1*ones(T1,1,sys2.nu) + sys2)
    sys2.nu == 1 && sys2.ny == 1 && (sys1.nu > 1 || sys1.ny > 1) && (return sys1 + ones(T1,sys1.ny,1)*sys2*ones(T1,1,sys1.nu))
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

    if typeof(A) <: Matrix && typeof(B)<: Matrix && typeof(C)<: Matrix && typeof(D)<: Matrix && E == I ? true : (typeof(E) <: Matrix)   
        return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
    else
        return DescriptorStateSpaceExt{T}(A, E, B, C, D, Ts)
    end
end
function +(sys1::SparseDescriptorStateSpace{T1}, sys2::SparseDescriptorStateSpace{T2}) where {T1,T2}
    sys1.nu == 1 && sys1.ny == 1 && (sys2.ny > 1 || sys2.nu > 1) && (return ones(T1,sys2.ny,1)*sys1*ones(T1,1,sys2.nu) + sys2)
    sys2.nu == 1 && sys2.ny == 1 && (sys1.nu > 1 || sys1.ny > 1) && (return sys1 + ones(T1,sys1.ny,1)*sys2*ones(T1,1,sys1.nu))
    #Ensure systems have same dimensions and sampling times
    size(sys1) == size(sys2) || error("The systems have different shapes.")
    Ts = promote_Ts(sys1.Ts,sys2.Ts)

    T = promote_type(T1, T2)
    n1 = sys1.nx
    n2 = sys2.nx
    A = cat(sys1.A,sys2.A; dims=(1,2))     
    B = [sys1.B ; sys2.B]
    C = [sys1.C sys2.C;]
    D = [sys1.D + sys2.D;]
    if sys1.E == I && sys2.E == I
        E = I
    else
        E = cat(sys1.E == I ? T.(I(n1)) : sys1.E, sys2.E == I ? T.(I(n2)) : sys2.E; dims=(1,2))     
    end
    return SparseDescriptorStateSpace{T}(A, E, B, C, D, Ts)
end
-(sys1::SparseDescriptorStateSpace, sys2::SparseDescriptorStateSpace) = +(sys1,-sys2)
+(sys1::SparseDescriptorStateSpace, sys2::DescriptorStateSpace) = +(sys1,DescriptorSystems.sparse(sys2))
+(sys1::DescriptorStateSpace, sys2::SparseDescriptorStateSpace) = +(DescriptorSystems.sparse(sys1),sys2)
# difference sys1-sys2
function -(sys1::DSTYPE{T1}, sys2::DSTYPE{T2}) where {T1,T2}
    sys1.nu == 1 && sys1.ny == 1 && (sys2.ny > 1 || sys2.nu > 1) && (return ones(T1,sys2.ny,1)*sys1*ones(T1,1,sys2.nu) - sys2)
    sys2.nu == 1 && sys2.ny == 1 && (sys1.nu > 1 || sys1.ny > 1) && (return sys1 - ones(T1,sys1.ny,1)*sys2*ones(T1,1,sys1.nu))
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
    if typeof(A) <: Matrix && typeof(B)<: Matrix && typeof(C)<: Matrix && typeof(D)<: Matrix && E == I ? true : (typeof(E) <: Matrix)   
       return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
    else
       return DescriptorStateSpaceExt{T}(A, E, B, C, D, Ts)
    end
end
# negation -sys
function -(sys::DescriptorStateSpace{T}) where {T}
    return DescriptorStateSpace{T}(sys.A, sys.E, sys.B, -sys.C, -sys.D, sys.Ts)
end
function -(sys::SparseDescriptorStateSpace{T}) where {T}
    return SparseDescriptorStateSpace{T}(sys.A, sys.E, sys.B, -sys.C, -sys.D, sys.Ts)
end

function -(sys::DescriptorStateSpaceExt{T}) where {T}
    return DescriptorStateSpaceExt{T}(sys.A, sys.E, sys.B, -sys.C, -sys.D, sys.Ts)
end
# sys+mat and mat+sys
function +(sys::DSTYPE{T1}, mat::AbstractVecOrMat{T2}) where {T1,T2}
    p, m = size(mat,1), size(mat,2)
    size(sys) == (p, m) || error("The input-output dimensions of system does not match the shape of matrix.")
    T = promote_type(T1, T2)
    return DS(sys){T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
                                   copy_oftype(sys.D,T) + (issparse(sys) ? sparse(mat) : mat), sys.Ts)
end

+(mat::AbstractVecOrMat{T1}, sys::DSTYPE{T2}) where {T1,T2} = +(sys,mat)
# sys+I and I+sys
function +(sys::DSTYPE{T1}, mat::UniformScaling{T2}) where {T1,T2}
    size(sys,1) == size(sys,2) || error("The system must have the same number of inputs and outputs")
    T = promote_type(T1, T2)
    return DS(sys){T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
                                   copy_oftype(sys.D,T) + mat, sys.Ts)
end

# sys-mat and mat-sys
function -(sys::DSTYPE{T1}, mat::AbstractVecOrMat{T2}) where {T1,T2}
    p, m = size(mat,1), size(mat,2)
    size(sys) == (p, m) || error("The input-output dimensions of system does not match the shape of matrix.")
    T = promote_type(T1, T2)
    return DS(sys){T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
                                    copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
                                    copy_oftype(sys.D,T) - (issparse(sys) ? sparse(mat) : mat), sys.Ts)
 end
-(mat::AbstractVecOrMat{T1}, sys::DSTYPE{T2}) where {T1,T2} = +(-sys,mat)

# sys-I and I-sys
function -(sys::DSTYPE{T1}, mat::UniformScaling{T2}) where {T1,T2}
    size(sys,1) == size(sys,2) || error("The system must have the same number of inputs and outputs")
    T = promote_type(T1, T2)
    return DS(sys){T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
                                   copy_oftype(sys.D,T) - mat, sys.Ts)
end
+(mat::UniformScaling{T1}, sys::DSTYPE{T2}) where {T1,T2} = +(sys,mat)
-(mat::UniformScaling{T1}, sys::DSTYPE{T2}) where {T1,T2} = +(-sys,mat)
# sys ± n and n ± sys
function +(sys::DSTYPE{T1}, n::Number) where T1 
    T = promote_type(T1, eltype(n))
    return DS(sys){T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
                                   copy_oftype(sys.D,T) .+ n, sys.Ts)
end

+(n::Number, sys::DSTYPE{T1}) where T1 = +(sys, n)
-(n::Number, sys::DSTYPE{T1}) where T1 = +(-sys, n)
-(sys::DSTYPE{T1},n::Number) where T1 = +(sys, -n)

# multiplication sys1*sys2
function *(sys1::DSTYPE{T1}, sys2::DSTYPE{T2}) where {T1,T2}
    sys1.nu == 1 && sys1.ny == 1 && sys2.ny > 1 && (return dsdiag(sys1,sys2.ny)*sys2)
    sys2.nu == 1 && sys2.ny == 1 && sys1.nu > 1 && (return sys1*dsdiag(sys2,sys1.nu))
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

    if typeof(A) <: Matrix && typeof(B)<: Matrix && typeof(C)<: Matrix && typeof(D)<: Matrix && E == I ? true : (typeof(E) <: Matrix)   
        return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
    else
        return DescriptorStateSpaceExt{T}(A, E, B, C, D, Ts)
    end
end
function *(sys1::SparseDescriptorStateSpace{T1}, sys2::SparseDescriptorStateSpace{T2}) where {T1,T2}
    sys1.nu == 1 && sys1.ny == 1 && sys2.ny > 1 && (return dsdiag(sys1,sys2.ny)*sys2)
    sys2.nu == 1 && sys2.ny == 1 && sys1.nu > 1 && (return sys1*dsdiag(sys2,sys1.nu))
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
        E = cat(sys1.E == I ? T.(I(n1)) : sys1.E, sys2.E == I ? T.(I(n2)) : sys2.E; dims=(1,2))     
    end
    return SparseDescriptorStateSpace{T}(A, E, B, C, D, Ts)
end

# sys*mat
function *(sys::DSTYPE{T1}, mat::AbstractVecOrMat{T2}) where {T1,T2}
    p, m = size(mat,1), size(mat,2)
    sys.nu == p || error("The input dimension of system does not match the number of rows of the matrix.")
    T = promote_type(T1, T2)
    mat1 = (issparse(sys) ? sparse(to_matrix(T,mat)) : to_matrix(T,mat))
    return DS(sys){T}(copy_oftype(sys.A,T), 
                                   sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T)*mat1, copy_oftype(sys.C,T), 
                                   copy_oftype(sys.D,T)*mat1, sys.Ts)
end
# mat*sys
function *(mat::AbstractVecOrMat{T1}, sys::DSTYPE{T2}) where {T1 <: Number,T2}
    p, m = typeof(mat) <: Vector ? (length(mat),1) : size(mat)
    sys.ny == m || error("The output dimension of system does not match the number of columns of the matrix.")
    T = promote_type(T1, T2)
    mat1 = (issparse(sys) ? sparse(to_matrix(T,mat)) : to_matrix(T,mat))
    return DescriptorStateSpace{T}(copy_oftype(sys.A,T), 
                                   sys.E == I ? I : copy_oftype(sys.E,T),
                                   copy_oftype(sys.B,T), mat1*copy_oftype(sys.C,T), 
                                   mat1*copy_oftype(sys.D,T), sys.Ts)
end
# sI*sys
function *(s::Union{UniformScaling,Number}, sys::DSTYPE{T}) where T
    T1 = promote_type(eltype(s),T)
    return DS(sys){T1}(copy_oftype(sys.A,T1), sys.E == I ? I : copy_oftype(sys.E,T1), copy_oftype(sys.B,T1), 
                                   lmul!(s,copy_oftype(sys.C,T1)), lmul!(s,copy_oftype(sys.D,T1)), sys.Ts)

end
# sys*sI
function *(sys::DSTYPE{T},s::Union{UniformScaling,Number}) where T
    T1 = promote_type(eltype(s),T)
    return DS(sys){T1}(copy_oftype(sys.A,T1), sys.E == I ? I : copy_oftype(sys.E,T1), rmul!(copy_oftype(sys.B,T1),s), 
                                   copy_oftype(sys.C,T1), rmul!(copy_oftype(sys.D,T1),s), sys.Ts)

end

# right division sys1/sys2 = sys1*inv(sys2)
/(n::Union{UniformScaling,Number}, sys::DescriptorStateSpace) = n*inv(sys)
/(n::Union{UniformScaling,Number}, sys::SparseDescriptorStateSpace) = n*inv(sys)
/(sys::DSTYPE, n::Number) = sys*(1/n)
/(sys::DSTYPE, n::UniformScaling) = sys*(1/n.λ)

# left division sys1\sys2 = inv(sys1)*sys2
\(n::Number, sys::DSTYPE) = (1/n)*sys
\(n::UniformScaling, sys::DSTYPE) = (1/n.λ)*sys
\(sys::DescriptorStateSpace, n::Union{UniformScaling,Number}) = inv(sys)*n
\(sys::SparseDescriptorStateSpace, n::Union{UniformScaling,Number}) = inv(sys)*n

# promotions
Base.promote_rule(::Type{DST}, ::Type{P}) where {DST <: DSTYPE, P <: Number } = DST
Base.promote_rule(::Type{DST}, ::Type{P}) where {DST <: DSTYPE, P <: UniformScaling } = DST
Base.promote_rule(::Type{DST}, ::Type{P}) where {DST <: DSTYPE, P <: Matrix{<:Number} } = DST
#Base.promote_rule(::Type{AbstractVecOrMat{<:Number}}, ::Type{DST}) where {DST <: DescriptorStateSpace} = DST
#Base.promote_rule(::Type{DST}, ::Type{Union{AbstractVecOrMat{<:Number}, Number, UniformScaling}}) where {DST <: DescriptorStateSpace} = DST
#Base.promote_rule(::Type{Union{AbstractVecOrMat{<:Number}, Number, UniformScaling}}, ::Type{DST}) where {DST <: DescriptorStateSpace} = DST
# Base.promote_rule(::Type{DST}, ::Type{UniformScaling}) where {DST <: DescriptorStateSpace, P <: Number} = DST
# Base.promote_rule(::Type{P}, ::Type{DST}) where {DST <: DescriptorStateSpace, P <: Number} = DST


# conversions
function Base.convert(::Type{DST}, p::Number) where {T, DST <: DescriptorStateSpace{T}}
    T1 = promote_type(eltype(p),T)
    dss(T1(p), Ts = sampling_time(DST))
end
sampling_time(::Type{DST}) where {DST <: DescriptorStateSpace} = 0.

SparseArrays.sparse(sys) = SparseDescriptorStateSpace{eltype(sys)}(sparse(sys.A), sys.E == I ? I : sparse(sys.E), sparse(sys.B), sparse(sys.C), sparse(sys.D),sys.Ts)

# display sys
Base.print(io::IO, sys::DSTYPE) = show(io, sys)
Base.show(io::IO, sys::DSTYPE) = show(io, MIME("text/plain"), sys)

#function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, sys::DescriptorStateSpace)
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, sys::DSTYPE)
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
       sys.Ts < 0 && println(io, "\n\nSample time: unspecified.")
       sys.Ts > 0 && println(io, "\n\nSample time: $(sys.Ts) second(s).")
       iszero(sys.Ts) ? println(io, "\n\nContinuous-time state-space model.") :
                        println(io, "Discrete-time state-space model.") 
    elseif m > 0 && p > 0
       println(io, "\nFeedthrough matrix D:")
       show(io, mime, sys.D)
       println(io, "\n\nStatic gain.") 
    else
       println(io, "\nEmpty state-space model.")
    end
end
 

