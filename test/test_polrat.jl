module Test_polrat

using DescriptorSystems
using MatrixEquations
using MatrixPencils
using LinearAlgebra
using Polynomials
using Test

println("Test_polrat")
@testset "Polynomial and rational matrix realizations" begin

@testset "Polynomial matrix realizations" begin

# Various simple examples 

# P = λ 

λ = Polynomial([0,1],:λ)
P = λ

@test all(P .≈ pm2poly(ls2pm(dssdata(dss(P))...),:λ))    
sys1 = pm2lps(P)  # build pencil realization
@test all(P .≈ pm2poly(ls2pm(dssdata(dss(sys1...))...),:λ))


# P = [λ 1] 
λ = Polynomial([0,1],:λ)
P = [λ one(λ)]

@test all(P .≈ pm2poly(ls2pm(dssdata(dss(P))...),:λ))    
sys1 = pm2lps(P)  # build pencil realization
@test all(P .≈ pm2poly(ls2pm(dssdata(dss(sys1...))...),:λ))

# Example 3: P = [λ^2 λ; λ 1] DeTeran, Dopico, Mackey, ELA 2009
λ = Polynomial([0,1],:λ)
P = [λ^2 λ; λ 1]
@test all(P .≈ pm2poly(ls2pm(dssdata(dss(P))...),:λ))    
sys1 = pm2lps(P)  # build pencil realization
@test all(P .≈ pm2poly(ls2pm(dssdata(dss(sys1...))...),:λ))

P = zeros(2,2,3);
P[:,:,1] = [0 0; 0 1.];
P[:,:,2] = [0 1.; 1. 0];
P[:,:,3] = [1. 0; 0 0]; 
@test P ≈ ls2pm(dssdata(dss(P))...)
sys1 = pm2lps(P)  # build pencil realization
@test P ≈ ls2pm(dssdata(dss(sys1...))...)


# Strong reductions

# Example 4: Van Dooren, Dewilde, LAA, 1983 
P = zeros(Int,3,3,3)
P[:,:,1] = [1 2 -2; 0 -1 -2; 0 0 0]
P[:,:,2] = [1 3 0; 1 4 2; 0 -1 -2]
P[:,:,3] = [1 4 2; 0 0 0; 1 4 2]
@test P ≈ ls2pm(dssdata(dss(P))...)
sys1 = pm2lps(P)  # build pencil realization
@test P ≈ ls2pm(dssdata(dss(sys1...))...)

A2 = [
    1     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     1     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0
    0     0     0     0     0     0     0     0     1];
E2 = [
    0     0     0     0     0     0     0     0     0
    1     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];
B2 = [
    -1     0     0
     0     0     0
     0     0     0
     0    -1     0
     0     0     0
     0     0     0
     0     0    -1
     0     0     0
     0     0     0];
C2 = [
    0     1     1     0     3     4     0     0     2
    0     1     0     0     4     0     0     2     0
    0     0     1     0    -1     4     0    -2     2]; 

D2 = zeros(Int,3,3);

sys = dss(A2, E2, B2, C2, D2, Ts = -1)
@time P = ls2pm(dssdata(sys)...)
@test pmeval(P,1) ≈ dcgain(sys)

# Example 3 - (Varga, Kybernetika, 1990) 
A2 = [
1 0 0 0 -1 0 0 0
0 1 0 0 0 -1 0 0
0 0 1 0 0 0 0 0      
0 0 0 1 0 0 0 0
0 0 0 0 -1 0 0 0
0 0 0 0 0 -1 0 0
0 0 0 0 3 0 1 0
0 0 0 0 0 2 0 1
]
E2 = [
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0
0 0 1 0 0 0 0 0
0 0 0 1 0 0 0 0
]
B2 = [
      -1 1
      0 0
      0 0
      0 0
      1 -2
      -2 3
      0 0
      3 -3
]
C2 = [
      0 0 0 0 0 0 -1 0
      0 0 0 0 0 0 0 -1      
]
D2 = zeros(Int,2,2);  

# case of a non-minimal realization of a polynomial matrix with finite eigenvalues 
sys = dss([A2 B2; C2 D2], [E2 zeros(8,2); zeros(2,10)], [zeros(8,2);I], [zeros(2,8) -I], 0, Ts = -1)

@time P = ls2pm(dssdata(sys)...,atol1 = 1.e-7,atol2=1.e-7);
@test pmeval(P,1) ≈ dcgain(sys)

end # Polynomial matrix realizations


@testset "Rational matrix realizations" begin

s = Polynomial([0, 1],:s)
num = Polynomial([4:-1:1...],:s)
den = Polynomial([7:-1:4...,1],:s)
sys = dss(num,den)
@test rmeval(num,den,0) ≈ dcgain(sys)
@test dcgain(sys)  == dcgain([num/den]) == evalfr(sys)
@test dcgain(sys)[1]  == dcgain(num/den) == evalfr(sys)[1]


sys = dss(num,den,contr=true); 
@test rmeval(num,den,0) ≈ dcgain(sys)

num1, den1 = ls2rm(dssdata(sys)...);
@test num*pm2poly(den1,:s)[1,1] ≈ pm2poly(num1,:s)[1,1]*den

sys = dss(num,den,obs=true) ;
@test rmeval(num,den,0) ≈ dcgain(sys)

num1, den1 = ls2rm(dssdata(sys)...);
@test num*pm2poly(den1,:s)[1,1] ≈ pm2poly(num1,:s)[1,1]*den

sys = dss(num,den,minimal = true);
@test rmeval(num,den,0) ≈ dcgain(sys)

num1, den1 = ls2rm(dssdata(sys)...);
@test num*pm2poly(den1,:s)[1,1] ≈ pm2poly(num1,:s)[1,1]*den



num = rand(2,3,4);
den = rand(2,3,2) .+ 0.5;
sys = dss(num,den,minimal = true,atol = 1.e-7);
@test rmeval(num,den,0) ≈ dcgain(sys)
@test dcgain(sys) == evalfr(sys)



num1, den1 = ls2rm(dssdata(sys)..., atol1 = 1.e-7, atol2 = 1.e-7);
@test all(pm2poly(num) .* pm2poly(den1) .≈ pm2poly(num1) .* pm2poly(den))

num = rand(2,3,2)
den = rand(2,3,4)
sys = dss(num,den,minimal = true,atol = 1.e-7);
@test rmeval(num,den,0) ≈ dcgain(sys)

num1, den1 = ls2rm(dssdata(sys)..., atol1 = 1.e-7, atol2 = 1.e-7);
@test all(pm2poly(num) .* pm2poly(den1) .≈ pm2poly(num1) .* pm2poly(den))


# Example Varga, Sima 1997
#s = Polynomial([0, 1],:s)
s = rtf(:s)
num = [s 2; 1 s]
den = [s+1 (s+1)*(s+3); s+4 (s+2)*(s+4)]
G = num./den

@time sys = dss(G,contr=true)
@test evalfr.(G,1) ≈ evalfr(sys,1)

@time sys = dss(G,obs=true)  
@test evalfr(G,1) ≈ evalfr(sys,1)

@time sys = dss(G,minimal = true, atol = 1.e-7);
@test evalfr(G,1) ≈ evalfr(sys,1)

@time sys = dss(G,minimal = true, contr = true, atol = 1.e-7);
@test evalfr(G,1) ≈ evalfr(sys,1)

@time sys = dss(G,minimal = true, obs = true, atol = 1.e-7);
@test evalfr(G,1) ≈ evalfr(sys,1)

num1, den1 = ls2rm(dssdata(sys)..., atol1 = 1.e-7, atol2 = 1.e-7);
@test all(num .* pm2poly(den1,:s) .≈ pm2poly(num1,:s) .* den)

# Example 4.3 Antsaklis, Michel 2006
#s = Polynomial([0, 1],:s)
s = rtf(:s)
num = [s^2+1 s+1]
den = [s^2 s^3]
G = num./den

@time sys = dss(G,contr=true)
@test evalfr(G,1) ≈ evalfr(sys,1)

@time sys = dss(G,obs=true)
@test evalfr(G,1) ≈ evalfr(sys,1)

# Example 4.3 (modified) Antsaklis, Michel 2006
#s = Polynomial([0, 1],:s)
s = rtf(:s)
num = [s^3+1 s+1]
den = [s^2 s^3]
G = num./den

@time sys = dss(G,contr=true)
@test evalfr(G,1) ≈ evalfr(sys,1)


# Example 4.4 Antsaklis, Michel 2006
#s = Polynomial([0, 1],:s)
s = rtf(:s)
num = [2 1; 1 0];
den = [s+1 1; s 1];
G = num./den

@time sys = dss(G)
@test evalfr(G,1) ≈ evalfr(sys,1)


@time sys = dss(G,contr=true)
@test evalfr(G,1) ≈ evalfr(sys,1)

@time sys = dss(G,obs=true)
@test evalfr(G,1) ≈ evalfr(sys,1)


#s = Polynomial([0, 1],:s)
s = rtf(:s)
num = [2+s 1; 1 0];
den = [0.5 1; 0.5 1];
G = rtf.(num./den)

sys = dss(G)
@test evalfr(G,1) ≈ evalfr(sys,1)


sys = dss(G,contr=true)
@test evalfr(G,1) ≈ evalfr(sys,1)

sys = dss(G,obs=true)
@test evalfr(G,1) ≈ evalfr(sys,1)

nump = numpoly.(G)
sys = dss(nump,obs=true)  
@test pmeval(nump,1) ≈ evalfr(sys,1)  
@test rmeval(poly2pm(nump),1) ≈ evalfr(sys,1)

# Example 4.4 (transposed) Antsaklis, Michel 2006
#s = Polynomial([0, 1],:s)
s = rtf(:s)
num = [2 1; 1 0]
den = [s+1 s; 1 1]
G = num./den

sys = dss(G)
@test evalfr(G,1) ≈ evalfr(sys,1)

sys = dss(G,contr=true)
@test evalfr(G,1) ≈ evalfr(sys,1)

sys = dss(G,obs=true)
@test evalfr(G,1) ≈ evalfr(sys,1)




P = zeros(Int,3,3,3);
P[:,:,1] = [1 2 -2; 0 -1 -2; 0 0 0];
P[:,:,2] = [1 3 0; 1 4 2; 0 -1 -2];
P[:,:,3] = [1 4 2; 0 0 0; 1 4 2];

sys = dss(P,contr=true)
@test pmeval(P,1) ≈ evalfr(sys,1)
@test P ≈ ls2pm(dssdata(sys)...)
N, D = ls2rm(dssdata(sys)...);
@test all(pm2poly(P) .* pm2poly(D) .≈ pm2poly(N))

end # Rational matrix realizations

@testset "Polynomial system matrix realizations" begin

p1 = Polynomial(1)
sys = dss(p1,p1,p1,p1; minimal = true)  
@test iszero(sys.D)

sys = dss(p1,p1,p1,p1)  
@test iszero(dcgain(sys)) && iszero(sys.E)


#  simple test
D = Polynomial.(rand(3,3));
W = Polynomial.(zeros(3,3));   
sys = dss(D,D,D,W,atol = 1.e-7, minimal=true)  
@test all(Polynomial.(sys.D) .≈ -D)  

@time sys = dss(D,D,D,W,atol = 1.e-7)  
@test all(Polynomial.(dcgain(sys)) .≈ -D)  && iszero(sys.E)

# 
D = rand(3,3,2); V = zeros(3,3,1); V[:,:,1] = Matrix{eltype(V)}(I,3,3); W = zeros(3,3,1);
@time sys = dss(D,D,V,W,atol = 1.e-7, minimal=true)   
@test sys.D ≈ Matrix{eltype(V)}(I,3,3)

D = rand(3,3,4); W = zeros(3,3,1);
@time sys2 = dss(D,D,D,W,atol = 1.e-7, minimal=true)  
sys1 = dss(D) 
@test iszero(sys1-sys2,atol1 = 1.e-7,atol2 = 1.e-7) 



T = rand(3,3,4); U = rand(3,3,2); V = rand(3,3,4); W = rand(3,3,3);
@time sys = dss(T,U,V,W);
@test evalfr(sys,5im) ≈ pmeval(V,5im)*(pmeval(T,5im)\pmeval(U,5im))+pmeval(W,5im)

T = rand(Complex{Float64},3,3,4); U = rand(3,3,2); V = rand(3,3,4); W = rand(3,3,3);
@time sys = dss(T,U,V,W);
@test evalfr(sys,1) ≈ pmeval(V,1)*(pmeval(T,1)\pmeval(U,1))+pmeval(W,1)

T = rand(0,0,4); U = rand(0,3,2); V = rand(3,0,4); W = rand(3,3,3);
@time sys = dss(T,U,V,W);
@test evalfr(sys,5im) ≈ pmeval(V,5im)*(pmeval(T,5im)\pmeval(U,5im))+pmeval(W,5im)


# Example Rosenbrock
s = Polynomial([0, 1],:s);
T = [s+1 s^3+2s^2; s^2+3s+2 s^4+4s^3+4s^2+s+2];
U = [s^2+1;s^3+2s^2+s+3];
V = -[s^2+3s+1 s^4+4s^3+4s^2-1];
W = [s^3+2s^2+s+2];

# G = (2*s + 3)/(s^2 + 3*s + 2)
T1 = [s+1 0;0 s+2];
U1 = Polynomial.([1;1],:s);
V1 = Polynomial.([1 1],:s);
W1 = Polynomial.([0],:s);

@time sys = dss(T,U,V,W,atol = 1.e-7, minimal=true); 
@time sys1 = dss(T1,U1,V1,W1,atol = 1.e-7, minimal=true); 
@test iszero(sys1-sys,atol1=1.e-7,atol2=1.e-7)



#  simple transfer function realization
D = reshape([-2,-1,2,1],1,1,4);
N = reshape([-1,1,0,1],1,1,4);
V = reshape([1],1,1,1);
W = reshape([0.],1,1,1);
@time sys = dss(D,N,V,W,atol = 1.e-7, minimal=true); 
sys_poles = eigvals(sys.A,sys.E)
@test sort(sys_poles) ≈ [-2., -1., 1.]
sys_zeros = spzeros(dssdata(sys)...)[1] 
@test coeffs(fromroots(sys_zeros)) ≈ [-1,1,0,1]


D1 = Polynomial([-2,-1,2,1]);
N1 = Polynomial([-1,1,0,1]);
V1 = Polynomial(1);
W1 = Polynomial(0.);
sys = dss(D1,N1,V1,W1,atol = 1.e-7, minimal=true); 
sys_poles = eigvals(sys.A,sys.E)
@test sort(sys_poles) ≈ [-2., -1., 1.]
sys_zeros = spzeros(dssdata(sys)...)[1] 
@test coeffs(fromroots(sys_zeros)) ≈ [-1,1,0,1]

D1 = Polynomial([-2,-1,2,1]);
N1 = Polynomial([-1,1,0,1]);
V1 = 1;
W1 = 0;
@time sys = dss(D1,N1,V1,W1,atol = 1.e-7, minimal=true); 
sys_poles = eigvals(sys.A,sys.E)
@test sort(sys_poles) ≈ [-2., -1., 1.]
sys_zeros = spzeros(dssdata(sys)...)[1] 
@test coeffs(fromroots(sys_zeros)) ≈ [-1,1,0,1]


end # polynomial system matrix realizations


end # realization tools

@testset "Frequency response" begin

w = collect(LinRange(.1,100,1000));
sys = rss(50,4,3);
@time H = freqresp(sys,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(sys,fval=w[i]); end end
@test norm(H,Inf) < 1.e-7

sys = rss(50,4,3,disc=true);
@time H = freqresp(sys,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(sys,fval=w[i]); end end
@test norm(H,Inf) < 1.e-7

sys = rdss(50,4,3);
@time H = freqresp(sys,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(sys,fval=w[i]); end end
@test norm(H,Inf) < 1.e-7

sys = rdss(50,4,3,disc=true);
@time H = freqresp(sys,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(sys,fval=w[i]); end end
@test norm(H,Inf) < 1.e-7

end #freqresp

end #module