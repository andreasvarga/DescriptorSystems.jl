module Test_model_matching

using DescriptorSystems
using LinearAlgebra
using Polynomials
using Random
using Test


println("Test_model_matching")

Random.seed!(2135)

@testset "Model_matching " begin

@testset "grasol" begin

# continuous-time, both G and F unstable
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
G = rss(n1,p,m); F = rss(n2,p,mf);  
@time X, info = grasol(G, F; reltol=0.0001, atol = 1.e-7); info
@test abs(glinfnorm(G*X-F,offset=1.e-13)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = grasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 

# discrete-time, both G and F unstable 
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
G = rss(n1,p,m,disc=true); F = rss(n2,p,mf,disc=true);  
@time X, info = grasol(G, F; reltol=0.0001, atol = 1.e-7); info
@test abs(glinfnorm(G*X-F,offset=1.e-13)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = grasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 


# continuous-time, both G and F unstable and complex
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
A = ComplexF64[0.3 - 0.9im 1.28 - 1.08im 0.57 + 0.67im; 0.32 + 0.07im 0.33 - 0.52im -0.57 - 1.09im; -1.41 + 0.21im -0.93 + 1.46im -0.79 - 0.7im];
B = ComplexF64[0.86 + 0.33im -0.13 - 1.5im; 0.9 - 0.57im -0.79 + 0.2im; -0.08 + 1.93im -0.26 + 0.6im];
C = ComplexF64[0.93 + 0.37im 1.05 - 0.83im -0.32 - 0.39im; -0.01 + 0.16im -0.46 + 1.74im -0.7 - 0.7im; 1.45 - 0.7im 0.68 - 0.22im 1.04 - 1.25im];
D = ComplexF64[0.4 + 0.43im 0.61 + 0.95im; 0.58 + 0.25im 0.7 + 0.31im; 0.35 + 0.57im 0.7 + 0.32im];
G = dss(A,B,C,D);
A1 = ComplexF64[1.06 + 0.73im 0.38 + 0.09im; -0.09 + 0.12im 0.55 - 0.09im];
B1 = ComplexF64[-0.5 + 0.01im; -0.81 - 0.77im;;];
C1 = ComplexF64[-0.5 - 0.97im -0.34 - 0.72im; 0.27 + 0.62im -1.26 + 0.77im; 0.41 - 0.93im -0.45 - 0.48im];
D1 = ComplexF64[0.49 + 0.13im; 0.95 + 0.69im; 0.1 + 0.74im;;];
F = dss(A1,B1,C1,D1)
#G = rss(n1,p,m,T = Complex{Float64}); F = rss(n2,p,mf,T = Complex{Float64});  
@time X, info = grasol(G, F; reltol=0.0001, atol = 1.e-7); info
@test glinfnorm(G*X-F,offset=1.e-13)[1] ≈ info.mindist &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = grasol(G, F; nehari = true, atol = 1.e-7); info
@test glinfnorm(G*X-F,offset=1.e-13)[1] ≈ info.mindist &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

# discrete-time, both G and F unstable and complex
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
G = dss(A,B,C,D,Ts=1); F = dss(A1,B1,C1,D1,Ts=1)
#G = rdss(n1,p,m,T = Complex{Float64},disc=true); F = rss(n2,p,mf,T = Complex{Float64},disc=true);  
@time X, info = grasol(G, F; reltol=0.001, offset=1.e-13, atol = 1.e-13); info
@test glinfnorm(G*X-F,offset=1.e-13)[1] ≈ info.mindist &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = grasol(G, F; nehari = true, atol = 1.e-7,offset=1.e-15); info
@test glinfnorm(G*X-F,offset=1.e-13)[1] ≈ info.mindist &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 


# both G and F stable
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
G = rss(n1,p,m,stable=true); F = rss(n2,p,mf,stable=true);  
@time X, info = grasol(G, F; atol = 1.e-8); info
@test abs(glinfnorm(G*X-F,offset=1.e-13)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = grasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 

# exact stable solution exists, no free poles  
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
#G = rss(n1,p,m,stable=true); X0 = rss(n2,pf,p,stable=true);  F = X0*G;
G = rss(n1,p,m); X0 = rss(n2,m,mf,stable=true);  F = G*X0; 
@time X, info = grasol(G, F, offset = 1.e-13, atol = 1.e-7); info  
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 1.e-5*max(1,info.mindist)  &&  (!info.nonstandard && isstable(X)) 

# exact stable solution exists, free poles exist 
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
G = rss(n1,p,m,stable=true); X0 = rss(n2,m,mf,stable=true);  F = G*X0; 
@time X, info = grasol(G, F, atol = 1.e-7, poles = [-2, -3], sdeg = -1, mindeg = true); info
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 1.e-5*max(1,info.mindist) &&  (!info.nonstandard && isstable(X)) &&
      order(X) == order(X0)

## Glover & Packard (SCL,2017)
s = rtf('s');
epsi = -1; 
g = (s-1)*(s-epsi)/(s+1)/(s+2); f = 1/(s+1); 
sysg = gir(dss(g)); sysf = dss(f);
@time sysx, info = grasol(sysg, sysf, mindeg = true); info
@test glinfnorm(sysx*sysg-sysf)[1] ≈ info.mindist && info.nrank == 1 && info.nonstandard == false &&
      gpole(sysx) ≈ [-1]

s = rtf('s');
epsi = 1; 
g = (s-1)*(s-epsi)/(s+1)/(s+2); f = 1/(s+1); 
sysg = gir(dss(g)); sysf = dss(f);
@time sysx, info = grasol(sysg, sysf, atol=1.e-7); info
# check solution
γ0 = 0.25*(1+sqrt(1+8/(1+epsi))) 
α = 1 + 2(1 + epsi)*γ0
@test glinfnorm(sysx*sysg-sysf)[1] ≈ info.mindist && info.mindist ≈ γ0 && 
       gpole(sysx) ≈ [-α]

s = rtf('s');
epsi = 1; 
gf = [(s-1)*(s-epsi)/(s+1)/(s+2) 1/(s+1)]; 
sysgf = gir(dss(gf)); 
@time sysx, info = grasol(sysgf, 1, atol=1.e-7); info
# check solution
γ0 = 0.25*(1+sqrt(1+8/(1+epsi))) 
α = 1 + 2(1 + epsi)*γ0
@test glinfnorm(sysx*sysgf[:,1]-sysgf[:,2])[1] ≈ info.mindist && info.mindist ≈ γ0 && 
       gpole(sysx) ≈ [-α]

# improper L∞ solution for a nonstandard problem
s = rtf('s');
sysg = dss([1/(s+1); 1/(s+2)]); sysf = dss([ 1/(s+1); 0]); 
@time sysx, info = grasol(sysg, sysf, atol=1.e-7); info
@test abs(glinfnorm(sysg*sysx-sysf)[1] - info.mindist) < 0.0001 && info.nonstandard ≈ true && 
       gpole(sysx) ≈ [-1.5811388300841827, Inf]

# proper L2 solution for a nonstandard problem
@time sysx, info = grasol(sysg, sysf, atol=1.e-7, L2sol = true); info
@test gl2norm(sysg*sysx-sysf) ≈ info.mindist && info.nonstandard ≈ true && 
       gpole(sysx) ≈ [-1.5811388300841827]

G = rss(0,3,2); 
@time X, info = grasol(G, G, atol = 1.e-7); info
@test gl2norm(G*X-G,atolinf=1.e-7) < 1.e-7 && X.D ≈ eye(2)

G = rss(0,3,2); 
@time X, info = grasol(G, G, atol = 1.e-7, L2sol = true); info
@test gl2norm(G*X-G,atolinf=1.e-7) < 1.e-7 && X.D ≈ eye(2)

G = rss(3,3,2,stable=true); 
@time X, info = grasol(G, G, atol = 1.e-7); info
@test glinfnorm(G*X-G)[1] < 1.e-7 && X.D ≈ eye(2)

G = rss(3,3,2,stable=true); 
@time X, info = grasol(G, G, atol = 1.e-7, L2sol = true); info
@test gl2norm(G*X-G,atolinf=1.e-7) < 1.e-7 && X.D ≈ eye(2)

##  Francis (1987) Example 1, p. 112 
s = rtf('s'); # define the complex variable s
# enter W(s), G(s) and F(s)
W = (s+1)/(10*s+1); # weighting function
#G = W.*[ -(s-1)/(s^2+s+1); (s^2-2*s)/(s^2+s+1)];  # erroneous
G = [ -(s-1)/(s^2+s+1)*W; (s^2-2*s)/(s^2+s+1)*W];
F = [ W; 0 ];
# solve G(s)*X(s) = F(s) for the least order solution
sysg = dss(G); sysf = dss(F);
X, info = grasol(sysg,sysf,atol=1.e-7,mindeg = true); info
@test glinfnorm(sysg*X-sysf)[1] ≈ info.mindist 
# compute the suboptimal solution for gamma = 0.2729
Xsub, info = grasol(sysg,sysf,0.2729,atol=1.e-7,mindeg = true); info
@test glinfnorm(sysg*Xsub-sysf)[1] ≈ info.mindist 


# Wang and Davison Example (1973)
s = Polynomial([0, 1],'s'); 
gn = [ s+1 s+3 s^2+3*s; s+2 s^2+2*s 0 ]; 
gd = [s^2+3*s+2 s^2+3*s+2 s^2+3*s+2 ;
s^2+3*s+2 s^2+3*s+2 s^2+3*s+2]; 
m = 3; mf = 2;
sysg = dss(gn,gd,minimal = true, atol = 1.e-7); 
sysf = dss([1 0;0 1.]);

@time sysx, info = grasol(sysg, sysf); info
@test abs(glinfnorm(sysg*sysx-sysf)[1] ≈ info.mindist) < 1.e-7 && isstable(sysx)

@time sysx, info = grasol(sysg, sysf, mindeg = true); info
@test abs(glinfnorm(sysg*sysx-sysf)[1] ≈ info.mindist) < 1.e-7 && isstable(sysx)

evals=[-1, -2, -3];
@time sysx, info = grasol(sysg, sysf, poles = evals); info
@test abs(glinfnorm(sysg*sysx-sysf)[1] ≈ info.mindist) < 1.e-7 && sort(gpole(sysx)) ≈ sort(evals)

sdeg = -1;
@time sysx, info = grasol(sysg, sysf, sdeg = sdeg); info
@test abs(glinfnorm(sysg*sysx-sysf)[1] ≈ info.mindist) < 1.e-7 && 
      all(real(gpole(sysx)) .< sdeg*(1-eps()^(1/info.nr))) 

@time sysx, info = glasol(gdual(sysg), sysf); info
@test abs(glinfnorm(sysx*gdual(sysg)-sysf,atol=1.e-7)[1] - info.mindist) < 1.e-7 && isstable(sysx)

end #grasol


@testset "glasol" begin

# continuous-time, both G and F unstable
n1 = 3; n2 = 2; m = 3; p = 2; pf = 1; 
G = rss(n1,p,m); F = rss(n2,pf,m);  
@time X, info = glasol(G, F; reltol=0.001, atol = 1.e-7); info
@test abs(glinfnorm(X*G-F,offset=1.e-13)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = glasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(X*G-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 

# discrete-time, both G and F unstable #fails
n1 = 3; n2 = 2; m = 3; p = 2; pf = 1; 
G = rss(n1,p,m,disc=true); F = rss(n2,pf,m,disc=true);  
@time X, info = glasol(G, F; reltol=0.0001, atol = 1.e-9); info
@test abs(glinfnorm(X*G-F,offset=1.e-13)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = glasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(X*G-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 


# continuous-time, both G and F unstable and complex
n1 = 3; n2 = 2; m = 3; p = 2; pf = 1; 
G = rss(n1,p,m,T = Complex{Float64}); F = rss(n2,pf,m,T = Complex{Float64});  
@time X, info = glasol(G, F; reltol=0.0001, atol = 1.e-9); info
@test abs(glinfnorm(X*G-F,offset=1.e-13)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = glasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(X*G-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 

# # discrete-time, both G and F unstable and complex
# n1 = 3; n2 = 2; m = 3; p = 2; pf = 1; 
# G = rdss(n1,p,m,T = Complex{Float64},disc=true); F = rss(n2,pf,m,T = Complex{Float64},disc=true);  
# @time X, info = glasol(G, F; reltol=0.0001, atol = 1.e-9); info
# @test abs(glinfnorm(X*G-F,offset=1.e-10)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

# @time X, info = glasol(G, F; nehari = true, atol = 1.e-7); info
# @test abs(glinfnorm(X*G-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 


# both G and F stable  
n1 = 3; n2 = 2; m = 3; p = 2; pf = 1; 
G = rss(n1,p,m,stable=true); F = rss(n2,pf,m,stable=true);  
@time X, info = glasol(G, F; atol = 1.e-8); info
@test abs(glinfnorm(X*G-F,offset=1.e-13)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = glasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(X*G-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 

# exact solution exists, no free poles 
n1 = 3; n2 = 2; m = 3; p = 2; pf = 1; 
G = rss(n1,p,m); X0 = rss(n2,pf,p,stable=true);  F = X0*G; 
@time X, info = glasol(G, F, offset = 1.e-13, atol = 1.e-7); info  
@test abs(glinfnorm(X*G-F)[1] - info.mindist) < 1.e-5 &&  (!info.nonstandard && isstable(X)) 

# exact solution exists, free poles exist 
n1 = 3; n2 = 2; m = 2; p = 3; pf = 1; 
G = rss(n1,p,m,stable=true); X0 = rss(n2,pf,p,stable=true);  F = X0*G;
@time X, info = glasol(G, F, atol = 1.e-7, poles = [-2, -3], sdeg = -1); info
@test abs(glinfnorm(X*G-F)[1] - info.mindist) < 1.e-7 &&  (!info.nonstandard && isstable(X)) 


n1 = 3; n2 = 2; m = 2; p = 3; pf = 1; 
G = rss(n1,p,m,stable=true); X0 = rss(n2,pf,p,stable=true);  F = X0*G;
@time X, info = glasol(G, F, atol = 1.e-7, poles = [-2, -3], sdeg = -1, mindeg = true); info
@test abs(glinfnorm(X*G-F)[1] - info.mindist) < 1.e-7 &&  (!info.nonstandard && isstable(X)) &&
      order(X) == order(X0)


# n = 3;
# G = rss(n,2,3,stable=true); F = G+0.001*rss(1,2,3,stable=true);
# @time X, info = glasol(G, F, atol = 1.e-7); info
# abs(glinfnorm(X*G-F)[1] - info.mindist)
# @test abs(glinfnorm(X*G-F)[1] -info.mindist) < 0.1 &&  (!info.nonstandard && isstable(X))
# gpole(X)

# F = rss(3,2,3,stable=true); G = F+0.001*rss(3,2,3,stable=true);
# @time X, info = glasol(G, F, atol = 1.e-13); info
# @test abs(glinfnorm(X*G-F)[1] -info.mindist) < 0.1 &&  (!info.nonstandard && isstable(X))
# gpole(X,atol=1.e-10)

## Glover & Packard (SCL,2017)
s = rtf('s');
epsi = -1; 
g = (s-1)*(s-epsi)/(s+1)/(s+2); f = 1/(s+1); 
sysg = gir(dss(g)); sysf = dss(f);
@time sysx, info = glasol(sysg, sysf, mindeg = true, atol = 1.e-7); info
@test glinfnorm(sysx*sysg-sysf)[1] ≈ info.mindist && info.nrank == 1 && info.nonstandard == false &&
      gpole(sysx) ≈ [-1]

s = rtf('s');
epsi = 1; 
g = (s-1)*(s-epsi)/(s+1)/(s+2); f = 1/(s+1); 
sysg = gir(dss(g)); sysf = dss(f);
@time sysx, info = glasol(sysg, sysf, mindeg = true, atol = 1.e-7); info
# check solution
γ0 = 0.25*(1+sqrt(1+8/(1+epsi))) 
α = 1 + 2(1 + epsi)*γ0
@test glinfnorm(sysx*sysg-sysf)[1] ≈ info.mindist && info.mindist ≈ γ0 && 
       gpole(sysx) ≈ [-α]

s = rtf('s');
epsi = 1; 
gf = [(s-1)*(s-epsi)/(s+1)/(s+2); 1/(s+1)]; 
sysgf = gir(dss(gf)); 
@time sysx, info = glasol(sysgf, 1, mindeg = true, atol = 1.e-7); info
# check solution
γ0 = 0.25*(1+sqrt(1+8/(1+epsi))) 
α = 1 + 2(1 + epsi)*γ0
@test glinfnorm(sysx*sysgf[1,:]-sysgf[2,:])[1] ≈ info.mindist && info.mindist ≈ γ0 && 
       gpole(sysx) ≈ [-α]

# improper L∞ solution for a nonstandard problem
s = rtf('s');
sysg = dss([1/(s+1) 1/(s+2)]); sysf = dss([ 1/(s+1) 0]); 
@time sysx, info = glasol(sysg, sysf, atol=1.e-7); info
@test abs(glinfnorm(sysx*sysg-sysf)[1] - info.mindist) < 0.0001 && info.nonstandard ≈ true && 
       gpole(sysx) ≈ [-1.5811388300841827, Inf]

# proper L2 solution for a nonstandard problem
@time sysx, info = glasol(sysg, sysf, atol=1.e-7, L2sol = true); info
@test gl2norm(sysx*sysg-sysf) ≈ info.mindist && info.nonstandard ≈ true && 
       gpole(sysx) ≈ [-1.5811388300841827]

G = rss(0,2,3); 
@time X, info = glasol(G, G, atol = 1.e-7); info
@test gl2norm(X*G-G,atolinf=1.e-7) < 1.e-7 && X.D ≈ eye(2)

G = rss(0,2,3); 
@time X, info = glasol(G, G, atol = 1.e-7, L2sol = true); info
@test gl2norm(X*G-G,atolinf=1.e-7) < 1.e-7 && X.D ≈ eye(2)

G = rss(3,2,3,stable=true); 
@time X, info = glasol(G, G, atol = 1.e-7); info
@test glinfnorm(X*G-G)[1] < 1.e-7 && X.D ≈ eye(2)

G = rss(3,2,3,stable=true); 
@time X, info = glasol(G, G, atol = 1.e-7, L2sol = true); info
@test gl2norm(X*G-G,atolinf=1.e-7) < 1.e-7 && X.D ≈ eye(2)

# Wang and Davison Example (1973)
s = Polynomial([0, 1],'s'); 
gn = [ s+1 s+3 s^2+3*s; s+2 s^2+2*s 0 ]; 
gd = [s^2+3*s+2 s^2+3*s+2 s^2+3*s+2 ;
s^2+3*s+2 s^2+3*s+2 s^2+3*s+2]; 
m = 3; mf = 2;
sysg = gdual(dss(gn,gd,minimal = true, atol = 1.e-7)); 
sysf = dss([1 0;0 1.]);

@time sysx, info = glasol(sysg, sysf); info
@test abs(glinfnorm(sysx*sysg-sysf,atol=1.e-7)[1] - info.mindist) < 1.e-7 && isstable(sysx)

@time sysx, info = glasol(sysg, sysf, mindeg = true); info
@test abs(glinfnorm(sysx*sysg-sysf,atol=1.e-7)[1] - info.mindist) < 1.e-7 && isstable(sysx)

evals=[-1, -2, -3];
@time sysx, info = glasol(sysg, sysf, poles = evals); info
@test abs(glinfnorm(sysx*sysg-sysf)[1] ≈ info.mindist) < 1.e-7 && sort(gpole(sysx)) ≈ sort(evals)

sdeg = -1;
@time sysx, info = glasol(sysg, sysf, sdeg = sdeg); info
@test abs(glinfnorm(sysx*sysg-sysf)[1] ≈ info.mindist) < 1.e-7 && 
      all(real(gpole(sysx)) .< sdeg*(1-eps()^(1/info.nl))) 


end #glasol

@testset "glinfldp" begin

## Parrott's problem
sys1 = rss(0,3,4); sys2 = rss(0,3,2);
@time sysx, mindist = glinfldp(sys1,sys2)
@test glinfnorm([sys1-sysx sys2])[1] ≈ mindist

# Example from [1]
# [1] C.-C. Chu, J. C. Doyle, and E. B. Lee
#     "The general distance problem in H-infinity optimal control theory",
#     Int. J. Control, vol 44, pp. 565-596, 1986.


s = rtf('s');
a = 1; f1 = 1/(s-1); f2 = 1/(s-a); Q0 = -1;
sys1 = dss(f1); sys2 = dss(f2);
@time sysx, mindist = glinfldp([sys1 sys2], 1, atol = 1.e-7, reltol=1.e-9); 
@test glinfnorm([sys1-Q0 sys2])[1] ≈  glinfnorm([sys1-sysx sys2])[1] ≈ mindist ≈ 1/a

γ = 1.1;
@time sysx, mindist = glinfldp([sys1 sys2], 1, γ, reltol=1.e-9); 
@test glinfnorm([sys1-sysx sys2])[1] ≈ mindist < γ


s = rtf('s');
a = 0.5; f1 = 1/(s-1); f2 = 1/(s-a); Q0 = -((1+a)*s+2*a)/2/(s+a);
sys1 = dss(f1); sys2 = dss(f2); sys0 = dss(Q0);
@time sysx, mindist = glinfldp([sys1 sys2],1, reltol=1.e-9); 
@test glinfnorm([sys1-sys0 sys2])[1] ≈  glinfnorm([sys1-sysx sys2])[1] ≈ mindist ≈ 1/a

γ = 1.1/a;
@time sysx, mindist = glinfldp([sys1 sys2], 1, γ, reltol=1.e-9); 
@test glinfnorm([sys1-sysx sys2])[1] ≈ mindist < γ


s = rtf('s');
a = 2; f1 = 1/(s-1); f2 = 1/(s-a); 
g0 = 1/(2*(a-1))*(-1 + sqrt(a^2+ 4*(a-1)/(a+1))); Q0 = -(g0*s+g0+(a-1)/2)/(s+a);
sys1 = dss(f1); sys2 = dss(f2); sys0 = dss(Q0);
@time sysx, mindist = glinfldp([sys1 sys2],1, reltol=1.e-9); 
@test glinfnorm([sys1-sys0 sys2])[1] ≈  glinfnorm([sys1-sysx sys2])[1] ≈ mindist ≈ g0

γ = 1.1*g0;
@time sysx, mindist = glinfldp([sys1 sys2], 1, γ, reltol=1.e-9); 
@test glinfnorm([sys1-sysx sys2])[1] ≈ mindist < γ


s = rtf('s'); t = rtfbilin("tustin",Ts=0.5)[1];
a = 0.5; f1 = confmap(1/(s-1),t); f2 = confmap(1/(s-a),t); 
Q0 = confmap(-((1+a)*s+2*a)/2/(s+a),t);
sys1 = dss(f1); sys2 = dss(f2); sys0 = dss(Q0);
@time sysx, mindist = glinfldp(gir([sys1 sys2],atol=1.e-7),1, offset=1.e-13, atol=1.e-7, reltol=1.e-9); #fails
@test glinfnorm(gir([sys1-sys0 sys2],atol=1.e-7))[1] ≈  glinfnorm(gir([sys1-sysx sys2],atol=1.e-7))[1] ≈ mindist ≈ 1/a

γ = 1.1/a;
@time sysx, mindist = glinfldp(gir([sys1 sys2],atol=1.e-7), 1, γ, reltol=1.e-9); 
@test glinfnorm(gir([sys1-sysx sys2],atol=1.e-7))[1] ≈ mindist < γ



s = rtf('s'); t = rtfbilin("tustin",Ts=0.5)[1];
a = 2; f1 = confmap(1/(s-1),t); f2 = confmap(1/(s-a),t); 
g0 = 1/(2*(a-1))*(-1 + sqrt(a^2+ 4*(a-1)/(a+1))); Q0 = -(g0*s+g0+(a-1)/2)/(s+a);
Q0 = confmap(-(g0*s+g0+(a-1)/2)/(s+a),t);
sys1 = dss(f1); sys2 = dss(f2); sys0 = dss(Q0);
@time sysx, mindist = glinfldp(gir([sys1 sys2],atol=1.e-7),1, reltol=1.e-9); 
@test glinfnorm(gir([sys1-sys0 sys2],atol=1.e-7))[1] ≈  glinfnorm(gir([sys1-sysx sys2],atol=1.e-7))[1] ≈ mindist ≈ g0


z = rtf('z');
G = [z^2+z+1 4*z^2+3*z+2 2*z^2-2;
    z 4*z-1 2*z-2;
    z^2 4*z^2-z 2*z^2-2*z]; 
sys = gir(dss(G),atol=1.e-7); 
@time sysx, s2 = glinfldp(sys; atol = 1.e-7); 
@test glinfnorm(sysx-sys)[1] ≈ s2 && s2 ≈ 8.662176191833835

# poles of sys1 on the imaginary axis are tolerated
s = rtf('s');
sys = dss(1/s);
sysn, mindist = glinfldp(sys; atol = 1.e-7)
@test iszero(sys-sysn,atol=1.e-7) && mindist == 0
sysn, mindist = glinfldp(sys, atol = 1.e-7, nehari = true)
@test iszero(sys-sysn,atol=1.e-7) && mindist == 0

# poles of sys2 on the imaginary axis
@test_throws ErrorException sysn, s1 = glinfldp(sys,sys)
@test_throws ErrorException sysn, s1 = glinfldp(sys,sys; nehari = true)

# poles of sys1 on the unit circle are tolerated
z = rtf('z');
sys = dss(1/(z-1));
sysn, mindist = glinfldp(sys; atol = 1.e-7)
@test iszero(sys-sysn,atol=1.e-7) && mindist == 0
sysn, mindist = glinfldp(sys, atol = 1.e-7, nehari = true)
@test iszero(sys-sysn,atol=1.e-7) && mindist == 0

# poles of sys2 on the unit circle 
@test_throws ErrorException sysn, s1 = glinfldp(sys,sys)
@test_throws ErrorException sysn, s1 = glinfldp(sys,sys; nehari = true)


# improper continuous-time systems sys1 are tolerated
s = rtf('s');
sys = dss(s);
sysn, mindist = glinfldp(sys)
@test iszero(sys-sysn,atol=1.e-7) && mindist == 0
sysn, mindist = glinfldp(sys, atol = 1.e-7, nehari = true)
@test iszero(sys-sysn,atol=1.e-7) && mindist == 0

# improper continuous-time systems sys2 
@test_throws ErrorException sysn, s1 = glinfldp(sys,sys)
@test_throws ErrorException sysn, s1 = glinfldp(sys,sys; nehari = true)


end # glinfldp    
@testset "gnehari" begin
    

sys = rss(0,3,2) 
@time sysn, s1 = gnehari(sys,eps())
@test glinfnorm(sysn-sys)[1] == 0

sys = rss(0,10,2) 
@time sysn, s1 = gnehari(sys)
@test glinfnorm(sysn-sys)[1] == 0

# poles on the imaginary axis can be  tolerated by choosing negative offset
s = rtf('s');
sys = dss(1/s);
sysn, s1 = gnehari(sys; offset = -1.e-10)
@test iszero(sysn-sys, atol = 1.e-7) && s1 == 0
@test_throws ErrorException sysn, s1 = gnehari(sys, offset = 1.e-10)

# Improper continuous-time system sys can be tolerated by choosing negative offset
s = rtf('s');
sys = dss(s);
sysn, s1 = gnehari(sys; offset = -sqrt(eps(1.)))
@test iszero(sysn-sys, atol = 1.e-7) && s1 == 0
@test_throws ErrorException sysn, s1 = gnehari(sys)


# poles on the unit circle can be  tolerated by choosing negative offset
z = rtf('z');
sys = dss(1/(z-1));
sysn, s1 = gnehari(sys, offset = -sqrt(eps(1.)))
@test iszero(sys-sysn, atol=1.e-7) && s1 == 0
@test_throws ErrorException sysn, s1 = gnehari(sys, offset = sqrt(eps(1.)))

# stable continuous-time system
s = rtf('s');
sys = [(s-1)/(s+2) s/(s+2) 1/(s+2);
    0 (s-2)/(s+1)^2 (s-2)/(s+1)^2;
    (s-1)/(s+2) (s^2+2*s-2)/(s+1)/(s+2) (2*s-1)/(s+1)/(s+2)]; 
sys = gir(dss(sys),atol=1.e-7); 
@time sysn, s1 = gnehari(sys)
@test glinfnorm(sysn-sys)[1] < 1.e-7 && s1 == 0

z = rtf('z');
G = [z^2+z+1 4*z^2+3*z+2 2*z^2-2;
    z 4*z-1 2*z-2;
    z^2 4*z^2-z 2*z^2-2*z]; 
sys = gir(dss(G),atol=1.e-7); 
@time sysn, s1 = gnehari(sys)
@test glinfnorm(sysn-sys)[1] ≈ s1 && s1 ≈ 8.662176191833835


sys = rdss(50,10,14)'; 
@time sysn, s1 = gnehari(sys); s1
@test glinfnorm(gminreal(sysn-sys,atol=1.e-10))[1] ≈ s1

@time sysn, s1opt = gnehari(sys,s1*1.01); s1
@test glinfnorm(gminreal(sysn-sys,atol=1.e-10))[1] > s1opt

##  Francis (1987) - optimal Nehari, p. 60
s  = rtf('s');
sys = gir(dss([1/(s^2-1) 4; 1/(s^2-s+1) (s+1)/(s-1)]),atol=1.e-7); 
@time sysn, s1 = gnehari(sys);
@test glinfnorm(gminreal(sysn-sys,atol=1.e-7))[1] ≈ s1

##  Francis (1987) - sub-optimal Nehari, p. 128

s  = rtf('s');
sys = gir(dss([.5/(s-1) 0; 1/(s^2-s+1) 2/(s-1)]),atol=1.e-7)/1.28; 

@time sysn, s1 = gnehari(sys);
@test glinfnorm(gminreal(sysn-sys,atol=1.e-7))[1] ≈ s1

end # gnehari
end # model_matching
end # module