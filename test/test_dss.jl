module Test_dss

using DescriptorSystems
using LinearAlgebra
using MatrixPencils
using Test


@testset "test_dss" begin
# SCALARS
a_2 = [-5 -3; 2 -9]; 
CSS_111 = dss(-5, 2, 3, [0])
CSS_111_d = dss([3], 2, 1, 1)
CSS_211 = dss(a_2, [1; 2], [1 0], 0)
CSS_221 = dss(a_2, [1 0; 0 2], [1 0], 0)
CSS_222 = dss(a_2, [1 0; 0 2], eye(2), 0)

a_2 = [-5 -3; 2 -9]; e_2 = [1 1; 0 1];
CS_111 = dss(-5, 1, 2, 3, [0])
CS_111_d = dss([3], I, 2, 1, 1)
CS_211 = dss(a_2, e_2, [1; 2], [1 0], 0)
CS_221 = dss(a_2, e_2, [1 0; 0 2], [1 0], 0)
CS_222 = dss(a_2, e_2,[1 0; 0 2], eye(2), 0)

# CONTINUOUS
a_1 = [-5]
C_111 = dss(a_1, [2], [3], [0])
C_211 = dss(a_2, e_2,  [1; 2], [1 0], [0])
C_212 = dss(a_2, [1; 2], eye(2), [0; 0])
C_221 = dss(a_2, e_2, [1 0; 0 2], [1 0], [0 0])
C_222 = dss(a_2, e_2, [1 0; 0 2], eye(2), zeros(Int,2,2))
C_222_d = dss(a_2, [1 0; 0 2], eye(2), eye(2))
C_022 = dss(4.0*eye(2))

# DISCRETE
da_1 = [-0.5]
da_2 = [0.2 -0.8; -0.8 0.07]; de_2 = [1 1; 0 1];
D_111 = dss(da_1, [2], [3], [0], Ts = 0.005)
D_211 = dss(da_2, de_2,[1; 2], [1 0], [0], Ts = 0.005)
D_221 = dss(da_2, [1 0; 0 2], [1 0], [0 0], Ts = 0.005)
D_222 = dss(da_2, [1 0; 0 2], eye(2), zeros(2,2), Ts = 0.005)
D_222_d = dss(da_2, [1 0; 0 2], eye(2), eye(2), Ts = 0.005)
D_022 = dss(4.0*eye(2), Ts = 0.005)

# TESTS
# Contstruct with scalars
@test CS_111 == C_111
@test CS_111_d == dss([3],[2],[1],[1])
@test CS_211 == C_211
@test CS_221 == C_221
@test CS_222 == C_222

# Addition
@test C_111 + C_111 == dss([-5 0; 0 -5],[2; 2],[3 3],[0])
@test C_222 + C_222 == dss([-5 -3 0 0; 2 -9 0 0; 0 0 -5 -3;0 0 2 -9],
[ 1  1  0  0; 0  1  0  0; 0  0  1  1; 0  0  0  1],
[1 0; 0 2; 1 0; 0 2], [1 0 1 0; 0 1 0 1],[0 0; 0 0])
@test C_222 + 1 == dss(a_2,e_2,[1 0; 0 2],[1 0; 0 1],[1 1; 1 1]) 
@test D_111 + D_111 == dss([-0.5 0; 0 -0.5],[2; 2],[3 3],[0], Ts = 0.005)
@test C_222 + [1 2; 3 4] == dss([-5 -3; 2 -9],[ 1  1; 0 1],
[1 0; 0 2], [1 0; 0 1],[1 2; 3 4])
@test [1 2; 3 4] + C_222  == dss([-5 -3; 2 -9],[ 1  1; 0 1],
[1 0; 0 2], [1 0; 0 1],[1 2; 3 4])
@test C_222 + 2I == dss([-5 -3; 2 -9],[ 1  1; 0 1],
[1 0; 0 2], [1 0; 0 1],[2 0; 0 2])
@test I + C_222  == dss([-5 -3; 2 -9],[ 1  1; 0 1],
[1 0; 0 2], [1 0; 0 1],[1 0; 0 1])

# Subtraction
@test C_111 - C_211 == dss([-5 0 0; 0 -5 -3; 0 2 -9],[1  0  0; 0  1  1; 0  0  1],[2; 1; 2],[3 -1 -0],[0])
@test 1 - C_222 == dss(a_2,e_2,[1 0; 0 2],[-1 -0; -0 -1],[1 1; 1 1])  
@test D_111 - D_211 == dss([-0.5 0 0; 0 0.2 -0.8; 0 -0.8 0.07],  [1.0  0.0  0.0; 0.0  1.0  1.0; 0.0  0.0  1.0],
[2; 1; 2], [3 -1 -0],[0], Ts = 0.005)
@test C_222 - [1 2; 3 4] == dss([-5 -3; 2 -9],[ 1  1; 0 1],
[1 0; 0 2], [1 0; 0 1],[-1 -2; -3 -4])
@test [1 2; 3 4] - C_222  == dss([-5 -3; 2 -9],[ 1  1; 0 1],
[1 0; 0 2], [-1 0; 0 -1],[1 2; 3 4])
@test C_222 - 2I == dss([-5 -3; 2 -9],[ 1  1; 0 1],
[1 0; 0 2], [1 0; 0 1],[-2 0; 0 -2])
@test I - C_222  == dss([-5 -3; 2 -9],[ 1  1; 0 1],
[1 0; 0 2], [-1 0; 0 -1],[1 0; 0 1])
@test C_222 - 2 == dss([-5 -3; 2 -9],[ 1  1; 0 1],
[1 0; 0 2], [1 0; 0 1],[-2 -2; -2 -2])


# Multiplication
@test C_111 * C_221 == dss([-5 2 0; 0 -5 -3; 0 2 -9],[1  0  0; 0  1  1; 0  0  1],
[0 0; 1 0; 0 2],[3 0 0],[0 0])
@test C_212 * C_111 == dss([-5 -3 3; 2 -9 6; 0 0 -5],
[0; 0; 2],[1 0 0; 0 1 0],[0; 0])
@test 4*C_222 == dss(a_2,e_2,[1 0; 0 2],[4 0; 0 4],[0 0; 0 0])
@test C_222*4 == dss(a_2,e_2,[4 0; 0 8],[1 0; 0 1],[0 0; 0 0])
@test D_111 * D_221 == dss([-0.5 2 0; 0 0.2 -0.8; 0 -0.8 0.07],
[0 0; 1 0; 0 2],[3 0 0],[0 0],Ts = 0.005)
@test [4 0;0 1]*C_222 == dss(a_2,e_2,[1 0; 0 2],[4 0; 0 1],[0 0; 0 0])
@test C_222*[4 0;0 1] == dss(a_2,e_2,[4 0; 0 2],[1 0; 0 1],[0 0; 0 0])

# Right Division
@test iszero(1/C_222_d - inv(C_222_d))
@test iszero(C_221/C_222_d - C_221*inv(C_222_d))
@test iszero(1/D_222_d - inv(D_222_d))
@test C_222/(4I) == dss(a_2,e_2,[0.25 0; 0 0.5],[1 0; 0 1],[0 0; 0 0])
@test iszero(C_222/4 -C_222*(1/4))


# Left Division
@test iszero(C_222_d\1 - inv(C_222_d))
@test iszero(C_222_d\C_212 - inv(C_222_d)*C_212)
@test iszero(D_222_d\1 - inv(D_222_d))
@test 4I\C_222 == dss(a_2,e_2,[1 0; 0 2],[0.25 0; 0 0.25],[0 0; 0 0])
@test iszero(4\C_222 -(1/4)*C_222)


# Indexing, order, ndims
@test order(C_222) == 2
@test size(C_222) == (2, 2)
@test size(C_212) == (2, 1)
@test ndims(C_222) == 1
@test length(C_222) == 1
@test C_222[1,1] == dss([-5 -3; 2 -9],[1 1; 0 1],[1; 0],[1 0],[0])
@test C_222[1:1,1] == dss([-5 -3; 2 -9],[1 1; 0 1],[1; 0],[1 0],[0])
@test C_222[1,1:2] == C_221
@test size(C_222[1,[]]) == (1,0)
@test C_222[:,2:end]  ≈ dss([-5 -3; 2 -9],[1 1; 0 1],[0; 2],[1 0; 0 1],[0; 0])



A = [-1.0 -2.0; 0.0 -1.0]
E = [1.0 2.0; 0.0 1.0]
B = [0.0; -2.0]
C = [1.0 1.0]
D = 1.0
sys = dss(A, E, B, C, D)
sysd = dss(A, E, B, C, D, Ts = -1)

@test sys + 1.0 == dss(A, E, B, C, D + 1.0)
@test 2.0 + sys == dss(A, E, B, C, D + 2.0)

@test -sys == dss(A, E, B, -C, -D)

# transpose, dual, adjoints
@test sys == transpose(transpose(sys))
@test iszero(sys -gdual(gdual(sys,rev=true)))
@test iszero(sys - adjoint(adjoint(sys)))
@test iszero(sysd - adjoint(adjoint(sysd)))
@test iszero(sysd - gsvselect(sysd,[2,1]))

# Accessing Ts through .Ts
@test D_111.Ts == 0.005

# property names
@test propertynames(C_111) == (:A, :E, :B, :C, :D, :Ts, :nx, :nu, :ny)
@test propertynames(D_111) == (:A, :E, :B, :C, :D, :Ts, :nx, :nu, :ny)

# Errors
@test_throws ErrorException C_111 + C_222             # Dimension mismatch
@test_throws ErrorException C_111 - C_222             # Dimension mismatch
@test_throws ErrorException C_111 * C_222             # Dimension mismatch
@test_throws ErrorException D_111 + C_111             # Sampling time mismatch
@test_throws ErrorException D_111 - C_111             # Sampling time mismatch
@test_throws ErrorException D_111 * C_111             # Sampling time mismatch
D_diffTs = dss([1], [2], [3], [4], Ts=0.1)
@test_throws ErrorException D_111 + D_diffTs            # Sampling time mismatch
@test_throws ErrorException D_111 - D_diffTs            # Sampling time mismatch
@test_throws ErrorException D_111 * D_diffTs            # Sampling time mismatch
@test_throws ErrorException 1/C_221                     # Not invertible
@test_throws ErrorException 1/C_212                     # Not invertible
@test_throws DimensionMismatch dss([1 2], [1], [2], [3])      # Not square A
@test_throws ErrorException dss([1], [2 0], [1], [2])      # I/0 dim mismatch
@test_throws ErrorException dss([1], [2], [3 4], [1])      # I/0 dim mismatch
@test_throws ErrorException dss([1], [2], [3], [4], Ts = -0.1)  # Negative samping time
@test_throws ErrorException dss(eye(2), eye(2), eye(2), [0]) # Dimension mismatch

try
    s = sprint(show,rdss(1,1,1));
    s = sprint(show,rss(1,1,1));
    s = sprint(show,rdss(1,0,1));
    s = sprint(show,rdss(1,1,0));
    s = sprint(show,rdss(0,1,1));
    s = sprint(show,rdss(0,0,1));
    @test true   
catch
    @test false
end


end

end #module





