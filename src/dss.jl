# Basic constructors of descriptor systems
"""
    sys = dss(A, E, B, C, D; Ts = 0, check_reg = false, 
              atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ ) 

Create for `Ts = 0` a descriptor system model `sys = (A-λE,B,C,D)` for a continuous-time state space system of the form
    
    Edx(t)/dt = Ax(t) + Bu(t) ,
    y(t)      = Cx(t) + Du(t) ,

where `x(t)`, `u(t)` and `y(t)` are the system state vector, system input vector and system output vector, respectively, 
for the continuous time variable `t`. 

For a nonzero positive sampling time `Ts = ΔT`, the descriptor system model specifies a discrete-time state space system of the form  

    Ex(t+ΔT) = Ax(t) + Bu(t)
    y(t)     = Cx(t) + Du(t)

for the discrete values of the time variable `t = 0, ΔT, 2ΔT, ...`. 
Use `Ts = -1` if the sampling time is not specified. In this case, by convention  `ΔT = 1`. 

For a system with zero feedthrough matrix `D`, it is possible to set `D = 0` (the scalar zero).  

For a standard state space system, `E` is the identity matrix. In this case, it is possible to set `E = I` (the boolean uniform scaling).
Alternatively, use 

    sys = dss(A, B, C, D; Ts = 0) 

to create a standard system.

For a system corresponding to a static gain `D`, use

    sys = dss(D; Ts = 0)  

It is possible to specify a descriptor system via all or part of its matrices using the form 

    sys = dss(A = mat1, E = mat2, B = mat3, C = mat4, D = mat5; Ts = 0, check_reg = false, 
              atol = 0, atol1 = atol, atol2 = atol, rtol = n*ϵ) 

where `A`, `E`, `B`, `C`, and `D` are keyword parameters set to appropriate matrix values 
`mat1`, `mat2`, `mat3`, `mat4`, and `mat5`, respectively. If
some of the system matrices are omited, then zero matrices of appropriate sizes are employed instead.  

It is assumed that the pencil `A-λE` is regular (i.e., `det(A-λE) ̸≡ 0`), and therefore, in the interest of efficiency,
the regularity of `A-λE` is by default _not_ tested. If `check_reg = true`, the regularity of `A-λE` is 
additionally checked. In this case, the keyword arguments `atol1`, `atol2` and `rtol` specify, respectively, 
the absolute tolerance for the nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`, 
and the relative tolerance for the nonzero elements of `A` and `E`.  
The default relative tolerance is `n*ϵ`, where `n` is the order of the square matrices `A` and `E`, and  `ϵ` is the working machine epsilon. 
The keyword argument `atol` can be used to simultaneously set `atol1 = atol` and `atol2 = atol`. 
"""
function dss(A::AbstractNumOrArray, E::Union{AbstractNumOrArray,UniformScaling}, B::AbstractNumOrArray, C::AbstractNumOrArray, D::AbstractNumOrArray; 
             Ts::Real = 0, check_reg::Bool = false, 
             atol::Real = zero(real(eltype(A))), atol1::Real = atol, atol2::Real = atol, 
             rtol::Real = (typeof(A) <: Number ? 1 : min(size(A)...))*eps(real(float(one(eltype(A)))))*iszero(min(atol1,atol2))) 
    T = promote_type(eltype(A), E == I ? Bool : eltype(E), eltype(B), eltype(C), eltype(D))
    check_reg && E != I && !MatrixPencils.isregular(to_matrix(T,A), to_matrix(T,E), atol1 = atol1, atol2 = atol2, rtol = rtol ) && 
                            error("The pencil A-λE is not regular")
    p = typeof(C) <: Union{Vector,Number} ? (size(A,1) <= 1 ? size(C,1) : 1) : size(C,1)                        
    m = typeof(B) <: Union{Vector,Number} ? 1 : size(B,2)   
    if typeof(D) <: Number
       if iszero(D)
          ds = zeros(T,p,m)
       else
          ds = fill(T(D),p,m)
       end
    else
       ds = p <=1 ? to_matrix(T,D,m > p) : to_matrix(T,D)
    end
    MAT = Union{Matrix,Vector,Number,SubArray,Adjoint{T,Matrix{T}} where T} 
    if typeof(A) <: MAT && typeof(B) <: MAT && typeof(C) <: MAT && typeof(C) <: MAT && E == I ? true : (typeof(E) <: MAT)   
       return DescriptorStateSpace{T}(to_matrix(T,A), E == I ? I : to_matrixA(T,E), to_matrix(T,B), to_matrix(T,C,p <=1), ds, Ts)
                                   #typeof(D) <: Number && iszero(D) ? zeros(T,p,m) : p <=1 ? to_matrix(T,D,m > p) : to_matrix(T,D), Ts)
    else
        return DescriptorStateSpaceExt{T}(to_matrixA(T,A), E == I ? I : to_matrixA(T,E), to_matrix(T,B), to_matrix(T,C,p <=1), ds, Ts)
                                   #typeof(D) <: Number && iszero(D) ? zeros(T,p,m) : p <=1 ? to_matrix(T,D,m > p) : to_matrix(T,D), Ts)
    end
end
function dss(A::SparseMatrixCSC, E::Union{SparseMatrixCSC,UniformScaling}, B::Union{SparseMatrixCSC,SparseVector}, C::Union{SparseMatrixCSC,SparseVector}, D::Union{SparseMatrixCSC,SparseVector,Number}; 
             Ts::Real = 0, check_reg::Bool = false, 
             atol::Real = zero(real(eltype(A))), atol1::Real = atol, atol2::Real = atol, 
             rtol::Real = (typeof(A) <: Number ? 1 : min(size(A)...))*eps(real(float(one(eltype(A)))))*iszero(min(atol1,atol2))) 
    T = promote_type(eltype(A), E == I ? Bool : eltype(E), eltype(B), eltype(C), eltype(D))
    check_reg && E != I && isregular(T.(A), T.(E), tol = rtol ) && error("The pencil A-λE is not regular")
    p = typeof(C) <: SparseVector ? (size(A,1) <= 1 ? size(C,1) : 1) : size(C,1)                        
    m = typeof(B) <: SparseVector ? 1 : size(B,2)  
    if typeof(D) <: Number
       if iszero(D)
          ds = SparseArrays.spzeros(T,p,m)
       else
          ds = sparse(fill(T(D),p,m))
       end
    else
       ds = p <=1 ? to_matrixA(T,D,m > p) : to_matrixA(T,D)
    end
    return SparseDescriptorStateSpace{T}(to_matrixA(T,A), E == I ? I : to_matrixA(T,E), to_matrixA(T,B), to_matrixA(T,C,p <=1), ds, Ts)
end

function dss(A::AbstractNumOrArray, B::AbstractNumOrArray, C::AbstractNumOrArray, D::AbstractNumOrArray; Ts::Real = 0) 
    T = promote_type(eltype(A),eltype(B),eltype(C),eltype(D))
    p = typeof(C) <: Union{Vector,Number} ? (size(A,1) <= 1 ? size(C,1) : 1) : size(C,1)                        
    m = typeof(B) <: Union{Vector,Number} ? 1 : size(B,2)   
    if typeof(D) <: Number
       if iszero(D)
          ds = zeros(T,p,m)
       else
          ds = fill(T(D),p,m)
       end
    else
       ds = p <=1 ? to_matrix(T,D,m > p) : to_matrix(T,D)
    end
    MAT = Union{Matrix,Vector,Number,SubArray,Adjoint{T,Matrix{T}} where T} 
    if typeof(A) <: MAT && typeof(B) <: MAT && typeof(C) <: MAT && typeof(C) <: MAT   
        return DescriptorStateSpace{T}(to_matrix(T,A), I, to_matrix(T,B), to_matrix(T,C,p <=1), ds, Ts)
    else
        return DescriptorStateSpaceExt{T}(to_matrixA(T,A), I, to_matrix(T,B), to_matrix(T,C,p <=1), ds, Ts) 
    end
end
function dss(A::SparseMatrixCSC, B::Union{SparseMatrixCSC,SparseVector}, C::Union{SparseMatrixCSC,SparseVector}, D::Union{SparseMatrixCSC,SparseVector,Number}; Ts::Real = 0) 
    T = promote_type(eltype(A),eltype(B),eltype(C),eltype(D))
    p = typeof(C) <: SparseVector ? (size(A,1) <= 1 ? size(C,1) : 1) : size(C,1)                        
    m = typeof(B) <: SparseVector ? 1 : size(B,2)  
    if typeof(D) <: Number
       if iszero(D)
          ds = SparseArrays.spzeros(T,p,m)
       else
          ds = sparse(fill(T(D),p,m))
       end
    else
       ds = p <=1 ? to_matrixA(T,D,m > p) : to_matrixA(T,D)
    end
    return SparseDescriptorStateSpace{T}(to_matrixA(T,A), I, to_matrixA(T,B), to_matrixA(T,C,p <=1), ds, Ts)
                                      #typeof(D) <: Number && iszero(D) ? SparseArrays.spzeros(T,p,m) : p <=1 ? to_matrixA(T,D,m > p) : to_matrixA(T,D), Ts)
end
function dss(D::AbstractNumOrArray; Ts::Real = 0) 
    p, m = size(D,1),size(D,2)
    T = eltype(D)
    isempty(D) && (return DescriptorStateSpace{T}(zeros(T,0,0),I,zeros(T,0,m),zeros(T,p,0),zeros(T,p,m),Ts))
    return DescriptorStateSpace{T}(zeros(T,0,0),I,zeros(T,0,m),zeros(T,p,0),to_matrix(T,D),Ts)
end
function dss(D::SparseMatrixCSC; Ts::Real = 0) 
    p, m = size(D,1),size(D,2)
    T = eltype(D)
    return SparseDescriptorStateSpace{T}(SparseArrays.spzeros(T,0,0),I,SparseArrays.spzeros(T,0,m),SparseArrays.spzeros(T,p,0),to_matrixA(T,D),Ts)
end

function dss(;A::AbstractNumOrArray = zeros(Bool,0,0), E::Union{AbstractNumOrArray,UniformScaling} = I, B::Union{AbstractNumOrArray,Missing} = missing, 
              C::Union{AbstractNumOrArray,Missing} = missing, D::Union{AbstractNumOrArray,Missing} = missing, 
              Ts::Real = 0, check_reg::Bool = false, 
              atol::Real = zero(real(eltype(A))), atol1::Real = atol, atol2::Real = atol, 
              rtol::Real = (typeof(A) <: Number ? 1 : min(size(A)...))*eps(real(float(one(eltype(A)))))*iszero(min(atol1,atol2))) 
    T = Bool
    T = promote_type(T, eltype(A), E == I ? Bool : eltype(E), ismissing(B) ? T : eltype(B), 
                     ismissing(C) ? T : eltype(C), ismissing(D) ? T : eltype(D))
    A = to_matrix(T,A)
    n = size(A,1)
    E == I || (E = to_matrix(T,E))
    check_reg && E != I && !MatrixPencils.isregular(A, E, atol1 = atol1, atol2 = atol2, rtol = rtol ) && 
                            error("The pencil A-λE is not regular")

    ismissing(D) || (D1 = to_matrix(T,D))
    if ismissing(B) 
       ismissing(D) ? m = 0 : m = size(D,2) 
       B = zeros(T,n,m) 
    else
       B = to_matrix(T,B)
       m = size(B,2)
    end      
    if ismissing(C) 
        ismissing(D) ? p = 0 : p = size(D,1) 
        C = zeros(T,p,n) 
    else
        C = to_matrix(T,C)
        p = size(C,1)
    end      
    ismissing(D) && (D1 = zeros(T,p,m))
    return DescriptorStateSpace{T}(A, E == I ? I : E, B, C, D1, Ts)
end
"""
     sys = dss(A, E, B, F, C, G, D, H; compacted = false, 
               atol = 0, atol1 = atol, atol2 = atol, atol3 = atol, rtol = min(atol1,atol2,atol3)>0 ? 0 : n*ϵ) 

Construct an input-output equivalent descriptor system representation `sys = (Ad-λdE,Bd,Cd,Dd)` to a pencil based linearization 
`(A-λE,B-λF,C-λG,D-λH)` satisfying 

                -1                        -1
     Cd*(λEd-Ad)  *Bd + Dd = (C-λG)*(λE-A)  *(B-λF) + D-λH .

If `compacted = true`, a compacted descriptor system realization is determined by exploiting possible rank defficiencies of the
matrices `F`, `G`, and `H`. Any of the matrices `F`, `G`, and `H` can be set to `missing`. 

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `F`, the absolute tolerance for the nonzero elements of `G`, 
the absolute tolerance for the nonzero elements of `H`  and the relative tolerance 
for the nonzero elements of `F`, `G` and `H`. The default relative tolerance is `n*ϵ`, where `n` is the size of 
of `A`, and `ϵ` is the machine epsilon of the element type of `A`.
The keyword argument `atol` can be used to simultaneously set `atol1 = atol`, `atol2 = atol` and `atol3 = atol`. 
"""
function dss(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, B::AbstractMatrix, F::Union{AbstractMatrix,Missing},
             C::AbstractMatrix, G::Union{AbstractMatrix,Missing}, D::AbstractMatrix, H::Union{AbstractMatrix,Missing}; 
             Ts::Real = 0, compacted::Bool = false, 
             atol::Real = zero(real(eltype(A))), atol1::Real = atol, atol2::Real = atol, atol3::Real = atol, 
             rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2,atol3))) 

    T = promote_type(eltype(A), E == I ? Bool : eltype(E), eltype(B), eltype(C), eltype(D))
    T = promote_type(T, ismissing(F) ? T : eltype(F), ismissing(G) ? T : eltype(G), ismissing(H) ? T :  eltype(H))

    return DescriptorStateSpace{T}(lps2ls(A, E, B, F, C, G, D, H; compacted = compacted, atol1 = atol1, atol2 = atol2, atol3 = atol2, rtol = rtol)..., Ts)

end
# function dss(sys::PencilStateSpace; compacted::Bool = false, 
#              atol1::Real = zero(real(eltype(sys.A))), atol2::Real = zero(real(eltype(sys.A))), 
#              rtol::Real = (min(size(sys.A)...)*eps(real(float(one(eltype(sys.A))))))*iszero(min(atol1,atol2))) 
#     return DescriptorStateSpace{eltype(sys.A)}(lps2ls(sys.A,sys.E,sys.B,sys.F,sys.C,sys.G,sys.D,sys.H; compacted = compacted, atol1 = atol1, atol2 = atol2, rtol = rtol)..., sys.Ts)
# end
# function dss(D::VecOrMat, H::VecOrMat; Ts::Real = 0) 
#     typeof(D) <: Vector ? (p, m) = (length(D), 1) : (p,m) = size(D)
#     T = promote_type(eltype(D), eltype(H))
#     D1 = reshape(copy_oftype(D,T),p,m)
#     iszero(H) &&  (return DescriptorStateSpace{T}(zeros(T,0,0),I,zeros(T,0,m),zeros(T,p,0),D1,Ts))
#     if m <= p
#         A = Matrix{T}(I,2*m,2*m)
#         E = [zeros(T,m,m) I; zeros(T,m,2*m)]
#         B = [zeros(T,m,m); I]
#         C = [H zeros(T,p,m)]
#      else
#         A = Matrix{T}(I,2*p,2*p)
#         E = [zeros(T,p,p) I; zeros(T,p,2*p)]
#         B = [zeros(T,p,m); -H]
#         C = [-I zeros(T,p,p)]
#      end   
#      return DescriptorStateSpace{T}(A,E,B,C,D1,Ts)
# end
"""
    sys = dss(NUM, DEN; contr = false, obs = false, noseig = false, minimal = false, fast = true, atol = 0, rtol) 

Convert the rational matrix `R(λ) = NUM(λ) ./ DEN(λ)` to a descriptor system representation `sys = (A-λE,B,C,D)` such that 
the transfer function matrix of `sys` is `R(λ)`.

`NUM(λ)` is a polynomial matrix of the form `NUM(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`, for which  
the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `NUM`, 
where `NUM[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

`DEN(λ)` is a polynomial matrix of the form `DEN(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`, for which 
the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `DEN`, 
where `DEN[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

Alternatively, `NUM(λ)` and `DEN(λ)` can be specified as matrices of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 
 
If `n` is the order of `A-λE`, then the computed linearization satisfies:
 
(1) `A-λE` is regular and `R(λ) = C*inv(λE-A)*B+D`;

(2) `rank[B A-λE] = n` (controllability) if `minimal = true` or `contr = true`;

(3) `rank[A-λE; C] = n` (observability)  if `minimal = true` or `obs = true`;

(4) `A-λE` has no non-dynamic modes if `minimal = true` or `noseig = true`. 

If conditions (1)-(4) are satisfied, the realization is called `minimal` and the resulting order `n`
is the least achievable order. If conditions (1)-(3) are satisfied, the realization is called `irreducible` 
and the resulting order `n` is the least achievable order using orthogonal similarity transformations.
An irreducible realization preserves the pole-zero and singular structures of `R(λ)`. 

The descriptor system based realization is built using the methods described in [1] in conjunction with
pencil manipulation algorithms [2] and [3] to compute reduced order realization. These algorithms 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol` and `rtol`, specify the absolute and relative tolerances, respectively, for the 
nonzero coefficients of `NUM(λ)` and `DEN(λ)`.

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).

[2] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[3] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function dss(NUM::Array{T1,3},DEN::Array{T2,3};
    Ts::Real = 0, minimal::Bool = false, contr::Bool = false, obs::Bool = false, noseig::Bool = false, 
    atol::Real = zero(real(T1)), rtol::Real = (min(size(NUM)...)*eps(real(float(one(T1)))))*iszero(atol)) where {T1,T2}
    sysdata = rm2ls(NUM, DEN, contr = contr, obs = obs, noseig = noseig, minimal = minimal, atol = atol, rtol = rtol)
    return DescriptorStateSpace{eltype(sysdata[1])}(sysdata[1:5]...,Ts)
end
# dss(NUM::Union{VecOrMat{Polynomial{T1,X}},Polynomial{T1,X},Number,VecOrMat},
#     DEN::Union{VecOrMat{Polynomial{T2,X}},Polynomial{T2,X},Number,VecOrMat}; kwargs...) where {T1,T2,X} =
#     dss(poly2pm(NUM),poly2pm(DEN); kwargs...)
dss(NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,AbstractVecOrMat{<:Number}, Number},
    DEN::Union{AbstractVecOrMat{<:Polynomial},Polynomial,AbstractVecOrMat{<:Number}, Number}; kwargs...) =
    dss(poly2pm(NUM),poly2pm(DEN); kwargs...)
"""
    sys = dss(R; Ts=missing, contr = false, obs = false, noseig = false, minimal = false, fast = true, atol = 0, rtol) 

Convert the rational transfer function matrix `R(λ)` to a descriptor system representation `sys = (A-λE,B,C,D)` such that 
the transfer function matrix of `sys` is `R(λ)`. The resulting `sys` is a continuous-time system if `Ts = 0` or 
discrete-time system if `Ts = -1` or `Ts > 0`. 
If `Ts = missing`, the sampling time of `sys` is inherited from the sampling time `TRs` of the elements of `R`, unless `TRs = nothing`, 
in which case `Ts = 0` is used (by default).  

`R(λ)` is a matrix with rational transfer function entries (see [`RationalTransferFunction`](@ref) ) corresponding to a
multiple-input multiple-outputs system or a rational transfer function corresponding to a
single-input single-output system. The numerators and denominators of the elements of R are of type
 `Polynomial` as provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

If `n` is the order of `A-λE`, then the computed realization satisfies:

(1) `A-λE` is regular and `R(λ) = C*inv(λE-A)*B+D`;

(2) `rank[B A-λE] = n` (controllability) if `minimal = true` or `contr = true`;

(3) `rank[A-λE; C] = n` (observability)  if `minimal = true` or `contr = true`;

(4) `A-λE` has no non-dynamic modes if `minimal = true` or `noseig = true`. 

If conditions (1)-(4) are satisfied, the realization is called `minimal` and the resulting order `n`
is the least achievable order. If conditions (1)-(3) are satisfied, the realization is called `irreducible` 
and the resulting order `n` is the least achievable order using orthogonal similarity transformations.
An irreducible realization preserves the pole-zero and singular structures of `R(λ)`. 

The underlying pencil manipulation algorithms [1] and [2] to compute reduced order realizations 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol` and `rtol`, specify the absolute and relative tolerances for the 
nonzero coefficients of `R(λ)`.

[1] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[2] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function dss(R::VecOrMat{<:RationalTransferFunction}; Ts::Union{Real,Missing,Nothing} = missing, 
             minimal::Bool = false, contr::Bool = false, obs::Bool = false, noseig::Bool = false, 
             atol::Real = zero(float(real(_eltype(R)))), rtol::Real = 100*eps(one(atol))*iszero(atol)) 
    length(R) == 0 && (return dss(zeros(_eltype(R),size(R)...),Ts = Ts))
    ismissing(Ts) && (Ts = R[1].Ts)
    isnothing(Ts) && (Ts = 0)
    dss(poly2pm(numpoly.(R)), poly2pm(denpoly.(R)); Ts = Ts, contr = contr, obs = obs, 
            noseig = noseig, minimal = minimal, atol = atol, rtol = rtol)
end
function dss(R::VecOrMat{<:RationalFunction}; Ts::Real = 0, 
             minimal::Bool = false, contr::Bool = false, obs::Bool = false, noseig::Bool = false, 
             atol::Real = zero(float(real(eltype(eltype(eltype(R)))))), rtol::Real = 100*eps(one(atol))*iszero(atol)) 
    length(R) == 0 && (return dss(zeros(eltype(eltype(eltype(G))),size(R)...),Ts = Ts))
    dss(poly2pm(numerator.(R)), poly2pm(denominator.(R)); Ts = Ts, contr = contr, obs = obs, 
            noseig = noseig, minimal = minimal, atol = atol, rtol = rtol)
end
function dss(R::RationalTransferFunction; Ts::Union{Real,Missing,Nothing} = missing, 
             minimal::Bool = false, contr::Bool = false, obs::Bool = false, noseig::Bool = false, 
             atol::Real = zero(float(real(_eltype(R)))), rtol::Real = 100*eps(one(atol))*iszero(atol)) 
    ismissing(Ts) && (Ts = R.Ts)
    isnothing(Ts) && (Ts = 0)
    dss(poly2pm(numpoly(R)), poly2pm(denpoly(R)); Ts = Ts, contr = contr, obs = obs, 
            noseig = noseig, minimal = minimal, atol = atol, rtol = rtol)
end
function dss(R::RationalFunction; Ts::Real = 0, 
             minimal::Bool = false, contr::Bool = false, obs::Bool = false, noseig::Bool = false, 
             atol::Real = zero(float(real(eltype(eltype(R))))), rtol::Real = 100*eps(one(atol))*iszero(atol)) 
    dss(poly2pm(numerator(R)), poly2pm(denominator(R)); Ts = Ts, contr = contr, obs = obs, 
            noseig = noseig, minimal = minimal, atol = atol, rtol = rtol)
end
# function dss(R::Union{VecOrMat{<:RationalTransferFunction},RationalTransferFunction}; Ts::Union{Real,Missing} = missing, 
#              minimal::Bool = false, contr::Bool = false, obs::Bool = false, noseig::Bool = false, 
#              atol::Real = zero(float(real(_eltype(R)))), rtol::Real = 100*eps(one(atol))*iszero(atol)) 
#     ismissing(Ts) && (eltype(R) <: RationalTransferFunction ? Ts = R[1].Ts : Ts = R.Ts )
#     isnothing(Ts) && (Ts = 0)
#     dss(poly2pm(numpoly.(R)), poly2pm(denpoly.(R)); Ts = Ts, contr = contr, obs = obs, 
#             noseig = noseig, minimal = minimal, atol = atol, rtol = rtol)
# end
# function dss(R::VecOrMat{RationalTransferFunction{T,X}}; Ts::Union{Real,Missing} = missing, 
#              minimal::Bool = false, contr::Bool = false, obs::Bool = false, noseig::Bool = false, 
#              atol::Real = zero(float(real(T))), rtol::Real = 100*eps(one(atol))*iszero(atol)) where T where X
#     length(R) == 0 && (return dss(zeros(_eltype(R),size(R)...),Ts = Ts))
#     ismissing(Ts) && (Ts = R[1].Ts)
#     isnothing(Ts) && (Ts = 0)
#     dss(poly2pm(numpoly.(R)), poly2pm(denpoly.(R)); Ts = Ts, contr = contr, obs = obs, 
#             noseig = noseig, minimal = minimal, atol = atol, rtol = rtol)
# end
"""
    sys = dss(P; Ts = 0, contr = false, obs = false, noseig = false, minimal = false, fast = true, atol = 0, rtol) 

Convert the polynomial matrix `P(λ)` to a descriptor system representation `sys = (A-λE,B,C,D)` such that 
the transfer function matrix of `sys` is `P(λ)`. The resulting `sys` is a continuous-time system if `Ts = 0` or 
discrete-time system if `Ts = -1` or `Ts > 0`. 

`P(λ)` can be specified as a grade `k` polynomial matrix of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, 
for which the coefficient matrices `P_i`, `i = 1, ..., k+1`, are stored in the 3-dimensional matrix `P`, 
where `P[:,:,i]` contains the `i`-th coefficient matrix `P_i` (multiplying `λ**(i-1)`). 

`P(λ)` can also be specified as a matrix, vector or scalar of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

If `d` is the degree of `P(λ)` and `n` is the order of `A-λE`, then the computed realization satisfies:

(1) `A-λE` is regular and `P(λ) = C*inv(λE-A)*B+D`;

(2) `rank[B A-λE] = n` (controllability) if `minimal = true` or `contr = true`;

(3) `rank[A-λE; C] = n` (observability)  if `minimal = true` or `contr = true`;

(4) `A-λE` has no non-dynamic modes if `minimal = true` or `noseig = true`. 

If conditions (1)-(4) are satisfied, the realization is called `minimal` and the resulting order `n`
is the least achievable order. If conditions (1)-(3) are satisfied, the realization is called `irreducible` 
and the resulting order `n` is the least achievable order using orthogonal similarity transformations.
An irreducible realization preserves the pole-zero and singular structures of `P(λ)`. 

The underlying pencil manipulation algorithms [1] and [2] to compute reduced order realizations 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol` and `rtol`, specify the absolute and relative tolerances for the 
nonzero coefficients of `P(λ)`, respectively.

[1] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[2] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function dss(P::Array{T,3};
    Ts::Real = 0, minimal::Bool = false, contr::Bool = false, obs::Bool = false, noseig::Bool = false, 
    atol::Real = zero(real(T)), rtol::Real = (min(size(P)...)*eps(real(float(one(T)))))*iszero(atol)) where T
    sysdata = pm2ls(P, contr = contr, obs = obs, noseig = noseig, minimal = minimal, atol = atol, rtol = rtol)
    return DescriptorStateSpace{eltype(sysdata[1])}(sysdata..., Ts)
end
dss(P::Union{VecOrMat{<:Polynomial},Polynomial}; kwargs...)  = dss(poly2pm(P); kwargs...)
function dss(P::LaurentPolynomial; kwargs...)
    n = degree(P)
    m = length(P)-1-n
    T = eltype(P)
    t = Polynomial{T,Polynomials.indeterminate(P)}([zeros(T,m);ones(T,1)][:])
    dss(poly2pm(Polynomial(P*t)),poly2pm(t); kwargs...)
end
"""
    sys = dss(T, U, V, W; fast = true, contr = false, obs = false, minimal = false, atol = 0, rtol) 
           
Construct an input-output equivalent descriptor system representation `sys = (A-λE,B,C,D)` 
to a polynomial model specified by the polynomial matrices `T(λ)`, `U(λ)`, `V(λ)`, and `W(λ)`
such that 

      V(λ)*inv(T(λ))*U(λ)+W(λ) = C*inv(λE-A)*B+D. 
      
If `minimal = true`, the resulting realization `(A-λE,B,C,D)` has the least possible order `n` of `A-λE`. 

`T(λ)`, `U(λ)`, `V(λ)`, and `W(λ)` can be specified as polynomial matrices of the form `X(λ) = X_1 + λ X_2 + ... + λ**k X_(k+1)`, 
for `X = T`, `U`, `V`, and `W`, for which the coefficient matrices `X_i`, `i = 1, ..., k+1`, are stored in 
the 3-dimensional matrices `X`, where `X[:,:,i]` contains the `i`-th coefficient matrix `X_i` (multiplying `λ**(i-1)`). 

`T(λ)`, `U(λ)`, `V(λ)`, and `W(λ)` can also be specified as matrices, vectors or scalars of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   
In this case, no check is performed that `T(λ)`, `U(λ)`, `V(λ)`  and `W(λ)` have the same indeterminates.

The computed descriptor realization satisfies:

(1) `A-λE` is regular;

(2) `rank[B A-λE] = n` (controllability) if `minimal = true` or `contr = true`;

(3) `rank[A-λE; C] = n` (observability)  if `minimal = true` or `obs = true`;

(4) `A-λE` has no simple infinite eigenvalues if `minimal = true`.

The keyword arguments `atol` and `rtol`, specify, respectively, the absolute and relative tolerance for the 
nonzero coefficients of the matrices `T(λ)`, `U(λ)`, `V(λ)` and `W(λ)`. The default relative tolerance is `nt*ϵ`, 
where `nt` is the size of the square matrix `T(λ)` and `ϵ` is the machine epsilon of the element type of its coefficients. 

The descriptor realization is built using the methods described in [1].

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).
"""
function dss(T::Array{T1,3}, U::Array{T2,3}, V::Array{T3,3}, W::Array{T4,3}; 
             Ts::Real = 0, contr::Bool = false, obs::Bool = false, minimal::Bool = false, 
             fast::Bool = true, atol::Real = zero(real(T1)), 
             rtol::Real = size(T,1)*eps(real(float(one(T1))))*iszero(atol)) where {T1, T2, T3, T4}
    sysdata = spm2ls(T, U, V, W, contr = contr, obs = obs, minimal = minimal, atol = atol, rtol = rtol)
    return DescriptorStateSpace{eltype(sysdata[1])}(sysdata..., Ts)
end
# function dss(T::Union{VecOrMat{Polynomial{T1}},VecOrMat{Polynomial{T1,X}},Polynomial{T1},Polynomial{T1,X}}, 
#              U::Union{VecOrMat{Polynomial{T2}},VecOrMat{Polynomial{T2,X}},Polynomial{T2},Polynomial{T2,X}}, 
#              V::Union{VecOrMat{Polynomial{T3}},VecOrMat{Polynomial{T3,X}},Polynomial{T3},Polynomial{T3,X}}, 
#              W::Union{VecOrMat{Polynomial{T4}},VecOrMat{Polynomial{T4,X}},Polynomial{T4},Polynomial{T4,X}}; kwargs...) where {T1, T2, T3, T4, X}
#              println("T = $T")
#        dss(poly2pm(T),poly2pm(U),poly2pm(V),poly2pm(W); kwargs...)
# end
# function dss(T::Union{VecOrMat{TP},TP}, 
#              U::Union{VecOrMat{TP},TP,Number,VecOrMat{<:Number}}, 
#              V::Union{VecOrMat{TP},TP,Number,VecOrMat{<:Number}}, 
#              W::Union{VecOrMat{TP},TP,Number,VecOrMat{<:Number}}; 
#              Ts::Real = 0, contr::Bool = false, obs::Bool = false, minimal::Bool = false, 
#              fast::Bool = true, atol::Real = zero(float(real(eltype(eltype(T))))), 
#              rtol::Real = size(T,1)*eps(one(eltype(atol)))*iszero(atol)) where {TP <: Polynomial}
#     return dss(poly2pm(T),poly2pm(U),poly2pm(V),poly2pm(W); Ts = Ts, contr = contr, obs = obs, minimal = minimal, fast = fast, atol = atol, rtol = rtol)
# end
function dss(T::Union{VecOrMat{<:Polynomial},Polynomial}, 
             U::Union{VecOrMat{<:Polynomial},Polynomial,Number,VecOrMat{<:Number}}, 
             V::Union{VecOrMat{<:Polynomial},Polynomial,Number,VecOrMat{<:Number}}, 
             W::Union{VecOrMat{<:Polynomial},Polynomial,Number,VecOrMat{<:Number}}; 
             Ts::Real = 0, contr::Bool = false, obs::Bool = false, minimal::Bool = false, 
             fast::Bool = true, atol::Real = zero(float(real(eltype(eltype(T))))), 
             rtol::Real = size(T,1)*eps(one(eltype(atol)))*iszero(atol)) 
return dss(poly2pm(T),poly2pm(U),poly2pm(V),poly2pm(W); Ts = Ts, contr = contr, obs = obs, minimal = minimal, fast = fast, atol = atol, rtol = rtol)
end

"""
    A, E, B, C, D  = dssdata([T,] sys) 
    
Extract the matrices `A`, `E`, `B`, `C`, `D` of a descriptor system model `sys = (A-λE,B,C,D)`. 
If the type `T` is specified, the resulting matrices are converted to this type. 
"""
function dssdata(sys::DSTYPE)
    return copy(sys.A), copy(sys.E), copy(sys.B), copy(sys.C), copy(sys.D)
end
function dssdata(T::Type,sys::DSTYPE)
    #eltype(sys) == T ? (return sys.A, sys.E, sys.B, sys.C, sys.D) :
    return copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T), copy_oftype(sys.B,T), copy_oftype(sys.C,T), copy_oftype(sys.D,T)
end
"""
    A, E, B, C, D  = dssdatafull([T,] sys) 
    
Extract the full matrices `A`, `E`, `B`, `C`, `D` of a descriptor system model `sys = (A-λE,B,C,D)`. 
If the type `T` is specified, the resulting matrices are converted to this type. 
"""
function dssdatafull(sys::DSTYPE)
    return Matrix(copy(sys.A)), sys.E == I ? I : Matrix(copy(sys.E)), Matrix(copy(sys.B)), Matrix(copy(sys.C)), Matrix(copy(sys.D))
end
function dssdatafull(T::Type,sys::DSTYPE)
    #eltype(sys) == T ? (return sys.A, sys.E, sys.B, sys.C, sys.D) :
    return Matrix(copy_oftype(sys.A,T)), sys.E == I ? I : Matrix(copy_oftype(sys.E,T)), Matrix(copy_oftype(sys.B,T)), Matrix(copy_oftype(sys.C,T)), Matrix(copy_oftype(sys.D,T))
end
"""
    sysf = dss2full(sys) 
    
Convert the descriptor system representation `sys = (A-λE,B,C,D)` to an equivalent one `sysf`
with full matrices.   
"""
function dss2full(sys::DSTYPE)
    return dss(Matrix(sys.A), sys.E == I ? I : Matrix(sys.E), Matrix(sys.B), Matrix(sys.C), Matrix(sys.D),Ts = sys.Ts)
end
"""
    syss = dss2sparse(sys) 
    
Convert the descriptor system representation `sys = (A-λE,B,C,D)` to an equivalent one `syss`
with sparse matrices.   
"""
function dss2sparse(sys::DSTYPE)
    return dss(sparse(sys.A), sys.E == I ? I : sparse(sys.E), sparse(sys.B), sparse(sys.C), sparse(sys.D),Ts = sys.Ts)
end


