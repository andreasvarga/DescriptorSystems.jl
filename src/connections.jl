function promote_Ts(Ts1::Real, Ts2::Real)
    Ts1 == Ts2 && (return Ts1)  
    Ts1 == -1 && (Ts2 > 0 ? (return Ts2) : error("Sampling time mismatch"))
    Ts2 == -1 && (Ts1 > 0 ? (return Ts1) : error("Sampling time mismatch"))
    error("Sampling time mismatch")
end
"""
     sys = series(sys1, sys2) 
     sys = sys2*sys1

Connect the descriptor systems `sys1` and `sys2` in series such that `sys = sys2*sys1`.
This coupling corresponds to the multiplication of their transfer function matrices. 
Series coupling of systems with constant matrices and vectors having suitable dimensions 
or with UniformScalings is also supported. 
Series coupling with a constant is equivalent to elementwise multiplication of 
the transfer function matrix with the constant. 
"""
series(sys1::DescriptorStateSpace, sys2::DescriptorStateSpace) = sys2*sys1
series(sys1::DescriptorStateSpace, sys2::Union{AbstractNumOrArray,UniformScaling}) = sys2*sys1
series(sys1::Union{AbstractNumOrArray,UniformScaling}, sys2::DescriptorStateSpace) = sys2*sys1

"""
    sys = parallel(sys1, sys2) 
    sys = sys1 + sys2

Connect the descriptor systems `sys1` and `sys2` in parallel such that `sys = sys1 + sys2`. 
This coupling corresponds to the addition of their transfer function matrices. 
Parallel coupling of systems with constant matrices or vectors having the same row and column dimensions 
or with UniformScalings is also supported. 
Parallel coupling with a constant is equivalent to elementwise parallel coupling of 
the transfer function matrix with the constant. 
"""
parallel(sys1::DescriptorStateSpace, sys2::DescriptorStateSpace) = sys1 + sys2
parallel(sys1::DescriptorStateSpace, sys2::Union{AbstractNumOrArray,UniformScaling}) = sys1 + sys2
parallel(sys1::Union{AbstractNumOrArray,UniformScaling},sys2::DescriptorStateSpace) = sys1 + sys2
"""
    sys = append(systems...) 

Append the descriptor systems `systems` by concatenating the input and output vectors
of individual systems. This corresponds to the block diagonal concatenation of 
their transfer function matrices. 
Appending systems with constant matrices, vectors or scalars or with UniformScalings is also supported. 
"""
function append(systems::(DST where DST<:DescriptorStateSpace)...)
    T = promote_type(eltype.(systems)...)
    Ts = systems[1].Ts
    if !all(s.Ts == Ts for s in systems)
        error("All systems must have same sampling time")
    end
    A = blockdiag([s.A for s in systems]...)
    if all(s.E == I for s in systems) 
        E = I 
    elseif any(s.E == I for s in systems)
        blockdims = Int[size(s.A,1) for s in systems]
        E = sblockdiag(blockdims, [s.E for s in systems]...)
    else
        E = blockdiag([s.E for s in systems]...)
    end
    
    B = blockdiag([s.B for s in systems]...)
    C = blockdiag([s.C for s in systems]...)
    D = blockdiag([s.D for s in systems]...)
    return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
end


function append(A::Union{AbstractDescriptorStateSpace,AbstractNumOrArray,UniformScaling}...)
    for a in A
        isa(a, UniformScaling) && @warn "All UniformScaling objects in append are set to scalars"
        require_one_based_indexing(a)
    end
    Ts = promote_system_SamplingTime(A...)
    return append(promote_to_systems(Ts, fill(1,length(A)), 1, promote_type(eltype.(A)...), A...)...)
end


function hcat(SYS1 :: AbstractDescriptorStateSpace, SYS2 :: AbstractDescriptorStateSpace)
    ny = SYS1.ny
    ny == size(SYS2, 1) ||  error("The systems must have the same output dimension")
    T = promote_type(eltype(SYS1), eltype(SYS2))
    Ts = promote_Ts(SYS1.Ts, SYS2.Ts) 

    A = blockdiag(T.(SYS1.A), T.(SYS2.A))
  
    if SYS1.E == I && SYS2.E == I 
        E = I 
    elseif SYS1.E == I || SYS2.E == I 
        blockdims = [size(SYS1.A,1), size(SYS2.A,1)]
        E = sblockdiag(blockdims, SYS1.E == I ? SYS1.E : T.(SYS1.E), SYS2.E == I ? SYS2.E : T.(SYS2.E))
    else
        E = blockdiag(T.(SYS1.E), T.(SYS2.E))
    end   
    B = blockdiag(T.(SYS1.B), T.(SYS2.B))
    C = [ T.(SYS1.C) T.(SYS2.C)]
    D = [ T.(SYS1.D) T.(SYS2.D)]
    return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
end
hcat(SYS :: AbstractDescriptorStateSpace, MAT :: AbstractNumOrArray) = hcat(SYS,dss(MAT,Ts=SYS.Ts))
hcat(MAT :: AbstractNumOrArray, SYS :: AbstractDescriptorStateSpace) = hcat(dss(MAT,Ts=SYS.Ts),SYS)
hcat(SYS :: AbstractDescriptorStateSpace, MAT :: UniformScaling) = hcat(SYS,dss(Matrix{promote_type(eltype(SYS),eltype(MAT))}(MAT,SYS.ny,SYS.ny),Ts=SYS.Ts))
hcat(MAT :: UniformScaling, SYS :: AbstractDescriptorStateSpace) = hcat(dss(Matrix{promote_type(eltype(SYS),eltype(MAT))}(MAT,SYS.ny,SYS.ny),Ts=SYS.Ts),SYS)

function isadss(DST :: Union{DescriptorStateSpace,AbstractNumOrArray,UniformScaling}...)
    # pick the index i of first system
    for i = 1:length(DST)
        DST[i] isa DescriptorStateSpace && (return i)
    end
    # set index to 0 if no system is present
    return 0
end

"""
    sys = horzcat(sys1,sys2)
    sys = [sys1 sys2]
    sys = horzcat(systems...) 

Concatenate horizontally two systems `sys1` and `sys2` or several descriptor systems `systems...` 
by concatenating the input vectors of individual systems. This corresponds to the horizontal 
concatenation of their transfer function matrices. 
Concatenation of systems with constant matrices, vectors, or scalars having the same row dimensions 
or with UniformScalings is also supported.  
"""
horzcat(systems::Union{DST,AbstractNumOrArray,UniformScaling}...) where DST <: AbstractDescriptorStateSpace = hcat(systems...)
function Base.hcat(systems::DST...) where DST <: DescriptorStateSpace
    # Perform checks
    T = promote_type(eltype.(systems)...)
    Ts = systems[1].Ts
    !all(s.Ts == Ts for s in systems) && error("All systems must have the same sampling time")
    ny = systems[1].ny
    !all(s.ny == ny for s in systems) && error("All systems must have the same output dimension")
    A = blockdiag([s.A for s in systems]...)
    if all(s.E == I for s in systems) 
        E = I 
    elseif any(s.E == I for s in systems)
        blockdims = Int[size(s.A,1) for s in systems]
        E = sblockdiag(blockdims, [s.E for s in systems]...)
    else
        E = blockdiag([s.E for s in systems]...)
    end   
    B = blockdiag([s.B for s in systems]...)
    C = hcat([s.C for s in systems]...)
    D = hcat([s.D for s in systems]...)

    return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
end

"""
    sys = vertcat(sys1,sys2)
    sys = [sys1; sys2]
    sys = vert(systems...) 

Concatenate vertically two descriptor systems `sys1` and `sys2` or several descriptor systems `systems...` 
by concatenating the output vectors of individual systems. This corresponds to the vertical 
concatenation of their transfer function matrices. 
Concatenation of systems with constant matrices, vectors, or scalars having the same column dimensions 
or with UniformScalings is also supported.  
"""
vertcat(systems::Union{DST,AbstractNumOrArray,UniformScaling}...) where DST <: AbstractDescriptorStateSpace = vcat(systems...)

function Base.vcat(systems::DST...) where DST <: DescriptorStateSpace
    # Perform checks
    T = promote_type(eltype.(systems)...)
    Ts = systems[1].Ts
    !all(s.Ts == Ts for s in systems) && error("All systems must have the same sampling time")
    nu = systems[1].nu
    !all(s.nu == nu for s in systems) && error("All systems must have the same input dimension")
    A = blockdiag([s.A for s in systems]...)
    if all(s.E == I for s in systems) 
        E = I 
    elseif any(s.E == I for s in systems)
        blockdims = Int[size(s.A,1) for s in systems]
        E = sblockdiag(blockdims, [s.E for s in systems]...)
    else
        E = blockdiag([s.E for s in systems]...)
    end   
    B = vcat([s.B for s in systems]...)
    C = blockdiag([s.C for s in systems]...)
    D = vcat([s.D for s in systems]...)
    return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
end

function vcat(SYS1 :: DescriptorStateSpace, SYS2 :: DescriptorStateSpace)
    nu = SYS1.nu
    nu == size(SYS2, 2) ||  error("The systems must have the same input dimension")
    T = promote_type(eltype(SYS1), eltype(SYS2))
    Ts = promote_Ts(SYS1.Ts, SYS2.Ts) 

    A = blockdiag(T.(SYS1.A), T.(SYS2.A))
  
    if SYS1.E == I && SYS2.E == I 
        E = I 
    elseif SYS1.E == I || SYS2.E == I 
        blockdims = [size(SYS1.A,1), size(SYS2.A,1)]
        E = sblockdiag(blockdims, SYS1.E == I ? SYS1.E : T.(SYS1.E), SYS2.E == I ? SYS2.E : T.(SYS2.E))
    else
        E = blockdiag(T.(SYS1.E), T.(SYS2.E))
    end   
    C = blockdiag(T.(SYS1.C), T.(SYS2.C))
    B = [ T.(SYS1.B); T.(SYS2.B)]
    D = [ T.(SYS1.D); T.(SYS2.D)]
    return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
end

vcat(SYS :: AbstractDescriptorStateSpace, MAT :: AbstractNumOrArray) = vcat(SYS,dss(MAT,Ts=SYS.Ts))
vcat(MAT :: AbstractNumOrArray, SYS :: AbstractDescriptorStateSpace) = vcat(dss(MAT,Ts=SYS.Ts),SYS)
vcat(SYS :: AbstractDescriptorStateSpace, MAT :: UniformScaling) = vcat(SYS,dss(Matrix{promote_type(eltype(SYS),eltype(MAT))}(MAT,SYS.nu,SYS.nu),Ts=SYS.Ts))
vcat(MAT :: UniformScaling, SYS :: AbstractDescriptorStateSpace) = vcat(dss(Matrix{promote_type(eltype(SYS),eltype(MAT))}(MAT,SYS.nu,SYS.nu),Ts=SYS.Ts),SYS)

for (f,dim,name) in ((:hcat,1,"rows"), (:vcat,2,"cols"))
    @eval begin
        function $f(A::Union{AbstractDescriptorStateSpace,AbstractNumOrArray,UniformScaling}...)
            n = -1
            for a in A
                if !isa(a, UniformScaling)
                    require_one_based_indexing(a)
                    isa(a,Number) ? na = 1 : na = size(a,$dim)
                    n >= 0 && n != na &&
                        throw(DimensionMismatch(string("number of ", $name,
                            " of each array must match (got ", n, " and ", na, ")")))
                    n = na
                end
            end
            n == -1 && throw(ArgumentError($("$f of only UniformScaling objects cannot determine the matrix size")))
            if isadss(A...) == 0
                return $f(promote_to_systems(0, fill(n,length(A)), 1, promote_type(eltype.(A)...), A...)...).D
            else       
                Ts = promote_system_SamplingTime(A...)
                return $f(promote_to_systems(Ts, fill(n,length(A)), 1, promote_type(eltype.(A)...), A...)...)
            end
        end
    end
end

function Base.hvcat(rows :: Tuple{Vararg{Int}}, DST :: AbstractDescriptorStateSpace...)
    j2 = rows[1]
    sys = hcat(DST[1:j2]...)
    for i = 2:length(rows)
        j1 = j2+1
        j2 = j2+rows[i]
        sys = [sys; hcat(DST[j1:j2]...)]
    end
    return sys
end


function Base.hvcat(rows::Tuple{Vararg{Int}}, A::Union{AbstractDescriptorStateSpace,AbstractNumOrArray,UniformScaling}...)
    require_one_based_indexing(A...)
    nr = length(rows)
    sum(rows) == length(A) || throw(ArgumentError("mismatch between row sizes and number of arguments"))
    n = fill(-1, length(A))
    needcols = false # whether we also need to infer some sizes from the column count
    j = 0
    for i = 1:nr # infer UniformScaling sizes from row counts, if possible:
        ni = -1 # number of rows in this block-row, -1 indicates unknown
        for k = 1:rows[i]
            if !isa(A[j+k], UniformScaling)
                isa(A[j+k],Number) ? na = 1 : na = size(A[j+k], 1)
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
                nci += isa(A[j+k], UniformScaling) ? n[j+k] : (isa(A[j+k],Number) ? 1 : size(A[j+k], 2))
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
    if isadss(A...) == 0
        return hvcat(rows, promote_to_systems(0, n, 1, promote_type(eltype.(A)...), A...)...).D
    else
        Ts = promote_system_SamplingTime(A...)
        return hvcat(rows, promote_to_systems(Ts, n, 1, promote_type(eltype.(A)...), A...)...)
    end
end

# promotion to systems of constant matrices, vectors, scalars and UniformScalings
promote_to_system_(n::Int, ::Type{T}, J::UniformScaling, Ts::Real) where {T} = dss(copyto!(Matrix{T}(undef, n,n), J), Ts = Ts)
promote_to_system_(n::Int, ::Type{T}, A::AbstractNumOrArray, Ts::Real) where {T} = dss(to_matrix(T,A), Ts = Ts)
promote_to_system_(n::Int, ::Type{T}, A::DescriptorStateSpace, Ts::Real) where {T} = T == eltype(A) ? A : dss(dssdata(T,A)..., Ts = Ts)
promote_to_systems(Ts::Real, n, k, ::Type) = ()
promote_to_systems(Ts::Real, n, k, ::Type{T}, A) where {T} = (promote_to_system_(n[k], T, A, Ts),)
promote_to_systems(Ts::Real, n, k, ::Type{T}, A, B) where {T} =
    (promote_to_system_(n[k], T, A, Ts), promote_to_system_(n[k+1], T, B, Ts))
promote_to_systems(Ts::Real, n, k, ::Type{T}, A, B, C) where {T} =
    (promote_to_system_(n[k], T, A, Ts), promote_to_system_(n[k+1], T, B, Ts), promote_to_system_(n[k+2], T, C, Ts))
promote_to_systems(Ts::Real, n, k, ::Type{T}, A, B, Cs...) where {T} =
    (promote_to_system_(n[k], T, A, Ts), promote_to_system_(n[k+1], T, B, Ts), promote_to_systems(Ts, n, k+2, T, Cs...)...)
promote_to_system_type(A::Tuple{Vararg{Union{DescriptorStateSpace,AbstractNumOrArray,UniformScaling}}}) = DescriptorStateSpace
promote_to_systems(Ts::Union{Real,Nothing}, var::Symbol, n, k, ::Type) = ()

function promote_system_SamplingTime(A::Union{DescriptorStateSpace,AbstractNumOrArray,UniformScaling}...)
    # pick and check the common sampling time  
    Ts = nothing
    for a in A
        typeof(a) <: DescriptorStateSpace  && (Ts = promote_Ts(Ts,a.Ts))
    end
    return isnothing(Ts) ? (return 0) : (return Ts) # for systems use Ts = 0 as defualt
end
       
