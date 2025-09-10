module Test_dss

using DescriptorSystems
using LinearAlgebra
using Polynomials
using SparseArrays
using Measurements
using Test


println("Test_dss")

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

C_222m = dssubset(C_222,dss(3.),1,1)
@test iszero(C_222m[1,1]-3, atol=1.e-7)
C_222m = dssubset(C_222,2 *C_221,1,1:2)
@test iszero(C_222m[1,1:2] - 2. *C_221, atol=1.e-7)
sys1 = rss(3,1,2)
syst = dssubset(dss(zeros(2,2)),sys1,1,1:2)
@test iszero(syst-[sys1; zeros(1,2)],atol1=1.e-7)
sys = rss(3,4,4)
syst = dszeros(sys,1:2,:)
@test iszero(syst[1:2,:],atol1=1.e-7) && iszero(syst[3:4,:]-sys[3:4,:],atol1=1.e-7) 
res1 = dssubsel(sys,sys.D .> 0.5, minimal = true)
res2 = dssubsel(sys,sys.D .< 0.5, minimal = true)
@test iszero(sys-res1-res2,atol=1.e-7)

res1 = dssubsel(sys,sys.D .> 0.5, minimal = false)
res2 = dssubsel(sys,sys.D .< 0.5, minimal = false)
@test iszero(sys-res1-res2,atol=1.e-7)
@test iszero(sys-copy(sys),atol=1.e-7)
@test iszero(dsdiag(sys,2)-append(sys,sys),atol=1.e-7)

# new constructors using polynomial coefficients (similar to MATLAB)
# example from https://github.com/JuliaControl/ControlSystems.jl/issues/992
G = rtf(1, [2, 3])
W1 = rtf(4, [5, 6])
W2 = rtf(7, [8, 9])
W3 = rtf(10, [11, 12])
P = [W1 -W1*G; 0 W2; 0 W3*G; 1 -G]
sys = dss(P; minimal=true)
@test order(sys) == 4


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
@test iszero(sysd - dsxvarsel(sysd,[2,1]))


A = sparse([-1.0 -2.0; 0.0 -1.0])
E = sparse([1.0 2.0; 0.0 1.0])
B = sparse([0.0; -2.0])
C = sparse([1.0 1.0])
D = sparse([1.0])
ssys = dss(A, E, B, C, D)
ssysd = dss(A, E, B, C, D, Ts = -1)

@test ssys + 1.0 == dss(A, E, B, C, D .+ 1.0)
@test 2.0 + ssys == dss(A, E, B, C, D .+ 2.0)
@test -ssys == dss(A, E, B, -C, -D)

# transpose, dual, adjoints
@test ssys == transpose(transpose(ssys))
@test iszero(ssys -gdual(gdual(ssys,rev=true)))
@test iszero(ssys - adjoint(adjoint(ssys)))
@test iszero(ssysd - adjoint(adjoint(ssysd)),atol=1.e-10)
@test iszero(ssysd - dsxvarsel(ssysd,[2,1]))


# Accessing Ts through .Ts
@test D_111.Ts == 0.005

# property names
@test propertynames(C_111) == (:nx, :ny, :nu, :A, :E, :B, :C, :D, :Ts)
@test propertynames(D_111) == (:nx, :ny, :nu, :A, :E, :B, :C, :D, :Ts)

# Building descriptor systems from rational functions and rational matrices
s = Polynomial([0, 1],'s'); 
g = (s+0.01)//(1+0.01*s);
sys1 = dss(g);
G  = [s^2 s//(s+1); 0 one(s)//s]     # define the 2-by-2 rational matrix G(s)
sys2 = dss(G);

# from Laurent polynomial
t = LaurentPolynomial([1],-1,:z)
p = LaurentPolynomial([24,10,-15,0,1],-2,:z)
q = LaurentPolynomial([1,0,1],-1,:z)
@test p ≈ 24t^2+10t-15+t^(-2)

dss2rm(dss(p,Ts=1))

# operations with scalar systems
@test iszero((C_111 + C_222) - (C_222 + C_111))       # Addition of scalar system
@test iszero((C_111 - C_222) + (C_222 - C_111))       # Substraction of scalar system
@test iszero(C_111 * C_222 -  C_222 * C_111)          # Multiplication with scalar system



# Errors
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

# some tests involving the new DescriptorStateSpace structure

t = rand(3,3)
sys = dss(UpperHessenberg(t), UpperTriangular(t),LowerTriangular(t),Diagonal(t),0)
@test istriu(sys.A,-1) && istriu(sys.E) && istril(sys.B) && isdiag(sys.C) && iszero(sys.D)
@test iszero(sys-sys)

ssys = dss(sparse(sys.A),sparse(sys.E),sparse(sys.B),sparse(sys.C),sparse(sys.D))
@test istriu(ssys.A,-1) && istriu(ssys.E) && istril(ssys.B) && isdiag(ssys.C) && iszero(ssys.D)
@test iszero(ssys-ssys)

ρ1 = measurement(0, 0.25); ρ2 = measurement(0, 0.25);
# build uncertain state matrix A(p)
A = [-.8 0 0;0 -.5*(1+ρ1) .6*(1+ρ2); 0 -0.6*(1+ρ2) -0.5*(1+ρ1)];
B = [1 1;1 0;0 1]; 
C = [0 1 1; 1 1 0]; D = zeros(2,2); 
# build an uncertain system 
try
    usys = dss(A,B,C,D)
    @test true
catch
    @test false
end




end

end #module





