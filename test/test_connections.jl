module Test_connections


using DescriptorSystems
using LinearAlgebra
using Test


println("Test_connections")
@testset "test_connections" begin

@testset "Standard state space" begin
# CONTINUOUS
C_111 = dss([1], [2], [3], [4])
C_211 = dss(eye(2), [1; 2], [1 0], [0])
C_212 = dss(eye(2), [1; 2], eye(2), [0; 0])
C_221 = dss(eye(2), [1 0; 0 2], [1 0], [0 0])
C_222 = dss(eye(2), [1 0; 0 2], eye(2), zeros(Int,2,2))
C_022 = dss(4*eye(2))

# DISCRETE
D_111 = dss([1], [2], [3], [4], Ts = 0.005)
D_211 = dss(eye(2), [1; 2], [1 0], [0], Ts = 0.005)
D_212 = dss(eye(2), [1;2], eye(2), [0; 0], Ts = 0.005)
D_221 = dss(eye(2), [1 0; 0 2], [1 0], [0 0], Ts = 0.005)
D_222 = dss(eye(2), [1 0; 0 2], eye(2), zeros(Int,2,2), Ts = 0.005)
D_022 = dss(4*eye(2), Ts = 0.005)

@test [C_111 C_221] == dss(eye(3), [2 0 0; 0 1 0; 0 0 2], [3 1 0], [4 0 0])
@test [C_111; C_212] == dss(eye(3), [2; 1; 2], [3 0 0; 0 1 0; 0 0 1], [4; 0; 0])
@test append(C_111, C_211) == dss(eye(3), [2 0; 0 1; 0 2], [3 0 0; 0 1 0], [4 0; 0 0])
@test [C_022 C_222] == dss(eye(2), [0 0 1 0; 0 0 0 2], [1 0; 0 1], [4 0 0 0; 0 4 0 0])
@test horzcat(C_022, C_222) == dss(eye(2), [0 0 1 0; 0 0 0 2], [1 0; 0 1], [4 0 0 0; 0 4 0 0])
@test [C_022; C_222] == dss(eye(2), [1 0; 0 2], [0 0; 0 0; 1 0; 0 1], [4 0; 0 4; 0 0; 0 0])
@test vertcat(C_022, C_222) == dss(eye(2), [1 0; 0 2], [0 0; 0 0; 1 0; 0 1], [4 0; 0 4; 0 0; 0 0])

@test [D_111 D_221] == dss(eye(3), [2 0 0; 0 1 0; 0 0 2], [3 1 0], [4 0 0], Ts = 0.005)
@test [D_111; D_212] == dss(eye(3), [2; 1; 2], [3 0 0; 0 1 0; 0 0 1], [4; 0; 0], Ts = 0.005)
@test append(D_111, D_211) == dss(eye(3), [2 0; 0 1; 0 2], [3 0 0; 0 1 0], [4 0; 0 0], Ts = 0.005)
@test [D_022 D_222] == dss(eye(2), [0 0 1 0; 0 0 0 2], [1 0; 0 1], [4 0 0 0; 0 4 0 0], Ts = 0.005)
@test horzcat(D_022, D_222) == dss(eye(2), [0 0 1 0; 0 0 0 2], [1 0; 0 1], [4 0 0 0; 0 4 0 0], Ts = 0.005)
@test [D_022; D_222] == dss(eye(2), [1 0; 0 2], [0 0; 0 0; 1 0; 0 1], [4 0; 0 4; 0 0; 0 0], Ts = 0.005)
@test vertcat(D_022, D_222) == dss(eye(2), [1 0; 0 2], [0 0; 0 0; 1 0; 0 1], [4 0; 0 4; 0 0; 0 0], Ts = 0.005)

@test series(C_111, C_212) == C_212*C_111
@test parallel(C_111, C_211) == C_111 + C_211

# Errors
@test_throws ErrorException [C_111 D_111]                 # Sampling time mismatch
@test_throws ErrorException [C_111; D_111]                # Sampling time mismatch
@test_throws ErrorException append(C_111, D_111)          # Sampling time mismatch
@test_throws ErrorException [C_111 C_212]                 # Dimension mismatch
@test_throws ErrorException [C_111; C_221]                # Dimension mismatch



# hcat and vcat for StateSpace and Matrix
A = [-1.1 -1.2; -1.3 -1.4]
B = [1 2; 3 4]
C = [5 6; 7 8]
D = [1 0; 0 1]
P = dss(A, B, C, D)
@test [P fill(2.5, 2, 1)] == dss(A, [B fill(0, 2, 1)], C, [D fill(2.5, 2, 1)])
@test [fill(2.5, 2, 1) P] == dss(A, [fill(0, 2, 1) B], C, [fill(2.5, 2, 1) D])
@test [P; fill(2.5, 1, 2)] == dss(A, B, [C; fill(0, 1, 2)], [D; fill(2.5, 1, 2)])
@test [fill(2.5, 1, 2); P] == dss(A, B, [fill(0, 1, 2); C], [fill(2.5, 1, 2); D])

# hcat and vcat for StateSpace and Number
P = dss(-1.0, 2.0, 3.0, 4.0)
@test [P 2.5] == dss(-1.0, [2.0 0.0], 3.0, [4.0 2.5])
@test [2.5 P] == dss(-1.0, [0.0 2.0], 3.0, [2.5 4.0])
@test [P; 2.5] == dss(-1.0, 2.0, [3.0; 0.0], [4.0; 2.5])
@test [2.5; P] == dss(-1.0, 2.0, [0.0; 3.0], [2.5; 4.0])

# 
@test [2.5I P 3.5] == dss(-1.0, [0.0 2.0 0.0], 3.0, [2.5 4.0 3.5])
@test [2.5; P; 3.5I] == dss(-1.0, 2.0, [0.0; 3.0; 0.0], [2.5; 4.0; 3.5])


# Concatenation of discrete system with constant
@test [D_111 1.] == dss([1.0], [2.0 0.0], [3.0], [4.0 1.0], Ts = 0.005)
@test [1. D_111] == dss([1.0], [0.0 2.0], [3.0], [1.0 4.0], Ts = 0.005)
@test [D_111 I] == dss([1.0], [2.0 0.0], [3.0], [4.0 1.0], Ts = 0.005)
@test [2I D_111] == dss([1.0], [0.0 2.0], [3.0], [2.0 4.0], Ts = 0.005)
# Type and sample time
@test [D_111 1.] isa DescriptorStateSpace{Float64}
@test [D_111 I].Ts == 0.005
# Continuous version
@test [C_111 1.] == dss([1.0], [2.0 0.0], [3.0], [4.0 1.0])
@test [1. C_111] == dss([1.0], [0.0 2.0], [3.0], [1.0 4.0])
@test [C_111 1.] isa DescriptorStateSpace{Float64}
@test [C_111 5.0I].Ts == 0.0

# Concatenation of discrete system with matrix
@test [D_222 fill(1.5, 2, 2)] == [D_222 dss(fill(1.5, 2, 2),Ts = 0.005)]
@test [C_222 fill(1.5, 2, 2)] == [C_222 dss(fill(1.5, 2, 2))]

# hvcat numbers 
@test [C_111 1.5; 2 3I] ==
    [C_111 dss(1.5); dss(2.0) dss(3.0)] 
@test [D_111 1.5; 2 3] ==
    [D_111 dss(1.5,Ts = 0.005); dss(2.0,Ts = 0.005) dss(3.0,Ts = 0.005)]
# hvcat matrices
@test [C_222 fill(1.5, 2, 2); fill(2, 2, 2) fill(3, 2, 2)] ==
    [C_222 dss(fill(1.5, 2, 2)); dss(fill(2, 2, 2)) dss(fill(3, 2, 2))]
@test [D_222 fill(1.5, 2, 2); fill(2, 2, 2) fill(3, 2, 2)] ==
    [D_222 dss(fill(1.5, 2, 2),Ts = 0.005); dss(fill(2, 2, 2),Ts = 0.005) dss(fill(3, 2, 2),Ts = 0.005)]

end # standard state space

@testset "Descriptor state space" begin
# CONTINUOUS
C_111 = dss([1], [0], [2], [3], [4])
C_211 = dss(eye(2), [1 0; 0 0], [1; 2], [1 0], [0])
C_212 = dss(eye(2), I, [1; 2], eye(2), [0; 0])
C_221 = dss(eye(2), zeros(2,2), [1 0; 0 2], [1 0], [0 0])
C_222 = dss(eye(2), 2*eye(2), [1 0; 0 2], eye(2), zeros(Int,2,2))
C_022 = dss(4*eye(2))
CS_211 = dss(eye(2), [1; 2], [1 0], [0])

# DISCRETE
D_111 = dss([1], [0], [2], [3], [4], Ts = 0.005)
D_211 = dss(eye(2), [1 0; 0 0], [1; 2], [1 0], [0], Ts = 0.005)
D_212 = dss(eye(2), I, [1;2], eye(2), [0; 0], Ts = 0.005)
D_221 = dss(eye(2), zeros(2,2), [1 0; 0 2], [1 0], [0 0], Ts = 0.005)
D_222 = dss(eye(2), 2*eye(2), [1 0; 0 2], eye(2), zeros(Int,2,2), Ts = 0.005)
D_022 = dss(4*eye(2), Ts = 0.005)
DS_211 = dss(eye(2), [1; 2], [1 0], [0], Ts = 0.005)

@test [C_111 C_221] == dss(eye(3), zeros(3,3), [2 0 0; 0 1 0; 0 0 2], [3 1 0], [4 0 0])
@test [C_111; C_212] == dss(eye(3), [ 0 0 0 ; 0 1 0; 0 0 1], [2; 1; 2], [3 0 0; 0 1 0; 0 0 1], [4; 0; 0])
@test append(C_111, C_211) == dss(eye(3), [ 0 0 0 ; 0 1 0; 0 0 0], [2 0; 0 1; 0 2], [3 0 0; 0 1 0], [4 0; 0 0])
@test append(C_111, CS_211) == dss(eye(3), [ 0 0 0 ; 0 1 0; 0 0 1], [2 0; 0 1; 0 2], [3 0 0; 0 1 0], [4 0; 0 0])
@test append(I, C_211) == dss(eye(2), [1 0; 0 0], [0 1; 0 2], [0 0; 1 0], [1 0; 0 0])
@test [C_022 C_222] == dss(eye(2), 2*eye(2), [0 0 1 0; 0 0 0 2], [1 0; 0 1], [4 0 0 0; 0 4 0 0])
@test [C_211 CS_211] == dss(eye(4), [1 0 0 0;0 0 0 0; 0 0 1 0; 0 0 0 1], [1 0;2 0;0 1;0 2], [1 0 1 0], [0 0])
@test [C_211 I] == dss(eye(2), [1 0 ;0 0], [1 0;2 0], [1 0], [0 1])
@test [I C_211] == dss(eye(2), [1 0 ;0 0], [0 1;0 2], [1 0], [1 0])

@test horzcat(C_022, C_222) == dss(eye(2), 2*eye(2), [0 0 1 0; 0 0 0 2], [1 0; 0 1], [4 0 0 0; 0 4 0 0])
@test [C_022; C_222] == dss(eye(2), 2*eye(2), [1 0; 0 2], [0 0; 0 0; 1 0; 0 1], [4 0; 0 4; 0 0; 0 0])
@test vertcat(C_022, C_222) == dss(eye(2), 2*eye(2), [1 0; 0 2], [0 0; 0 0; 1 0; 0 1], [4 0; 0 4; 0 0; 0 0])

@test [D_111 D_221] == dss(eye(3), zeros(3,3), [2 0 0; 0 1 0; 0 0 2], [3 1 0], [4 0 0], Ts = 0.005)
@test [D_111; D_212] == dss(eye(3), [ 0 0 0 ; 0 1 0; 0 0 1], [2; 1; 2], [3 0 0; 0 1 0; 0 0 1], [4; 0; 0], Ts = 0.005)
@test append(D_111, D_211) == dss(eye(3), [ 0 0 0 ; 0 1 0; 0 0 0], [2 0; 0 1; 0 2], [3 0 0; 0 1 0], [4 0; 0 0], Ts = 0.005)
@test append(D_111, DS_211) == dss(eye(3), [ 0 0 0 ; 0 1 0; 0 0 1], [2 0; 0 1; 0 2], [3 0 0; 0 1 0], [4 0; 0 0], Ts = 0.005)
@test [D_022 D_222] == dss(eye(2), 2*eye(2), [0 0 1 0; 0 0 0 2], [1 0; 0 1], [4 0 0 0; 0 4 0 0], Ts = 0.005)
@test horzcat(D_022, D_222) == dss(eye(2), 2*eye(2), [0 0 1 0; 0 0 0 2], [1 0; 0 1], [4 0 0 0; 0 4 0 0], Ts = 0.005)
@test [D_022; D_222] == dss(eye(2),2*eye(2),  [1 0; 0 2], [0 0; 0 0; 1 0; 0 1], [4 0; 0 4; 0 0; 0 0], Ts = 0.005)
@test vertcat(D_022, D_222) == dss(eye(2), 2*eye(2), [1 0; 0 2], [0 0; 0 0; 1 0; 0 1], [4 0; 0 4; 0 0; 0 0], Ts = 0.005)
@test [C_211; CS_211] == dss(eye(4), [1 0 0 0;0 0 0 0; 0 0 1 0; 0 0 0 1], [1;2;1;2], [1 0 0 0; 0 0 1 0], [0; 0])
@test [C_211; I] == dss(eye(2), [1 0 ;0 0], [1;2], [1 0; 0 0], [0; 1])
@test [I; C_211] == dss(eye(2), [1 0 ;0 0], [1;2], [0 0; 1 0], [1; 0])

@test series(C_111, C_212) == C_212*C_111
@test series(C_222, [1 1]) == dss([1 1])*C_222
@test series([1, 1],C_222) == C_222*dss([1, 1])
@test parallel(C_111, C_211) == C_111 + C_211
@test parallel([1,2], C_212) == dss([1,2]) + C_212
@test parallel(C_212,[1,2]) == C_212 + dss([1,2]) 


# Errors
@test_throws ErrorException [C_111 D_111]                 # Sampling time mismatch
@test_throws ErrorException [C_111; D_111]                # Sampling time mismatch
@test_throws ErrorException append(C_111, D_111)          # Sampling time mismatch
@test_throws ErrorException [C_111 C_212]                 # Dimension mismatch
@test_throws ErrorException [C_111; C_221]                # Dimension mismatch



# hcat and vcat for Descriptor StateSpace and Matrix
A = [-1.1 -1.2; -1.3 -1.4]
E = [-1. -2.; 0. -4]
B = [1 2; 3 4]
B = [1 2; 3 4]
C = [5 6; 7 8]
D = [1 0; 0 1]
P = dss(A, E, B, C, D)
@test [P fill(2.5, 2, 1)] == dss(A, E, [B fill(0, 2, 1)], C, [D fill(2.5, 2, 1)])
@test [fill(2.5, 2, 1) P] == dss(A, E, [fill(0, 2, 1) B], C, [fill(2.5, 2, 1) D])
@test [P; fill(2.5, 1, 2)] == dss(A, E, B, [C; fill(0, 1, 2)], [D; fill(2.5, 1, 2)])
@test [fill(2.5, 1, 2); P] == dss(A, E, B, [fill(0, 1, 2); C], [fill(2.5, 1, 2); D])

# hcat and vcat for Descriptor StateSpace and Number
P = dss(-1.0, 0.0, 2.0, 3.0, 4.0)
@test [P 2.5] == dss(-1.0, 0.0, [2.0 0.0], 3.0, [4.0 2.5])
@test [2.5 P] == dss(-1.0, 0.0, [0.0 2.0], 3.0, [2.5 4.0])
@test [P; 2.5] == dss(-1.0, 0.0, 2.0, [3.0; 0.0], [4.0; 2.5])
@test [2.5; P] == dss(-1.0, 0.0, 2.0, [0.0; 3.0], [2.5; 4.0])

# 
@test [2.5I P 3.5] == dss(-1.0, 0.0, [0.0 2.0 0.0], 3.0, [2.5 4.0 3.5])
@test [2.5; P; 3.5I] == dss(-1.0, 0.0, 2.0, [0.0; 3.0; 0.0], [2.5; 4.0; 3.5])


# Concatenation of discrete system with constant
@test [D_111 1.] == dss([1.0], [0], [2.0 0.0], [3.0], [4.0 1.0], Ts = 0.005)
@test [1. D_111] == dss([1.0], [0], [0.0 2.0], [3.0], [1.0 4.0], Ts = 0.005)
@test [D_111 I] == dss([1.0], [0], [2.0 0.0], [3.0], [4.0 1.0], Ts = 0.005)
@test [2I D_111] == dss([1.0], [0], [0.0 2.0], [3.0], [2.0 4.0], Ts = 0.005)
# Type and sample time
@test [D_111 1.] isa DescriptorStateSpace{Float64}
@test [D_111 I].Ts == 0.005
# Continuous version
@test [C_111 1.] == dss([1.0], [0], [2.0 0.0], [3.0], [4.0 1.0])
@test [1. C_111] == dss([1.0], [0], [0.0 2.0], [3.0], [1.0 4.0])
@test [C_111 1.] isa DescriptorStateSpace{Float64}
@test [C_111 5.0I].Ts == 0.0

# Concatenation of discrete system with matrix
@test [D_222 fill(1.5, 2, 2)] == [D_222 dss(fill(1.5, 2, 2),Ts = 0.005)]
@test [C_222 fill(1.5, 2, 2)] == [C_222 dss(fill(1.5, 2, 2))]

# hvcat numbers 
@test [C_111 1.5; 2 3I] ==
    [C_111 dss(1.5); dss(2.0) dss(3.0)] 
@test [D_111 1.5; 2 3] ==
    [D_111 dss(1.5,Ts = 0.005); dss(2.0,Ts = 0.005) dss(3.0,Ts = 0.005)]
# hvcat UniformScaling 
@test [C_111 I; 3I] == [C_111 dss(eye(Float64,1)); dss(3*eye(Float64,2,2))] 
@test [C_111 I; I 3I] == [C_111 dss(1); dss(1) dss(3*eye(1))] 

# hvcat matrices
@test [C_222 fill(1.5, 2, 2); fill(2, 2, 2) fill(3, 2, 2)] ==
    [C_222 dss(fill(1.5, 2, 2)); dss(fill(2, 2, 2)) dss(fill(3, 2, 2))]
@test [D_222 fill(1.5, 2, 2); fill(2, 2, 2) fill(3, 2, 2)] ==
    [D_222 dss(fill(1.5, 2, 2),Ts = 0.005); dss(fill(2, 2, 2),Ts = 0.005) dss(fill(3, 2, 2),Ts = 0.005)]

end # descriptor state space

end
end # module


