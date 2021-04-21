module Test_model_matching

using DescriptorSystems
using MatrixEquations
using MatrixPencils
using LinearAlgebra
using Polynomials
using Test

# using JLD
# F1 = load("test_glinfldp.jld","F1")
# Yt, gopt = glinfldp(F1, 1); gpole(Yt) # fails

# s1, s2 = gsdec(F1, job = "unstable",atol=1.e-7); gpole(s1) # fails

# A, E, B, C, _, _, _, blkdims, = gsblkdiag(F1.A, F1.E, F1.B, F1.C; 
#                                           finite_infinite = false, stable_unstable = false) 

# F2 = gss2ss(F1)
# s1, s2 = gsdec(F2, job = "unstable",atol=1.e-7); gpole(s1) # fails
# A1, B1, C1, _, _, _, blkdims, = ssblkdiag(F2.A, F2.B, F2.C; stable_unstable = false)


println("Test_model_matching")


@testset "Model_matching " begin

@testset "grasol" begin

# continuous-time, both G and F unstable
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
G = rss(n1,p,m); F = rss(n2,p,mf);  
@time X, info = grasol(G, F; reltol=0.0001, atol = 1.e-9); info
@test abs(glinfnorm(G*X-F,offset=1.e-13)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = grasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 

# discrete-time, both G and F unstable 
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
G = rss(n1,p,m,disc=true); F = rss(n2,p,mf,disc=true);  
@time X, info = grasol(G, F; reltol=0.0001, atol = 1.e-9); info
@test abs(glinfnorm(G*X-F,offset=1.e-13)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = grasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 


# continuous-time, both G and F unstable and complex
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
G = rss(n1,p,m,T = Complex{Float64}); F = rss(n2,p,mf,T = Complex{Float64});  
@time X, info = grasol(G, F; reltol=0.0001, atol = 1.e-9); info
@test abs(glinfnorm(G*X-F,offset=1.e-13)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = grasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 

# discrete-time, both G and F unstable and complex
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
G = rdss(n1,p,m,T = Complex{Float64},disc=true); F = rss(n2,p,mf,T = Complex{Float64},disc=true);  
@time X, info = grasol(G, F; reltol=0.0001, atol = 1.e-9); info
@test abs(glinfnorm(G*X-F,offset=1.e-13)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = grasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 


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
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 1.e-5  &&  (!info.nonstandard && isstable(X)) 

# exact solution exists, free poles exist 
n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
G = rss(n1,p,m,stable=true); X0 = rss(n2,m,mf,stable=true);  F = G*X0; 
@time X, info = grasol(G, F, atol = 1.e-7, poles = [-2, -3], sdeg = -1); info
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 1.e-7 &&  (!info.nonstandard && isstable(X)) 

n1 = 3; n2 = 2; m = 2; p = 3; mf = 1; 
G = rss(n1,p,m,stable=true); X0 = rss(n2,m,mf,stable=true);  F = G*X0; 
@time X, info = grasol(G, F, atol = 1.e-7, poles = [-2, -3], sdeg = -1, mindeg = true); info
@test abs(glinfnorm(G*X-F)[1] - info.mindist) < 1.e-7 &&  (!info.nonstandard && isstable(X)) &&
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


end #grasol


@testset "glasol" begin

# continuous-time, both G and F unstable
n1 = 3; n2 = 2; m = 3; p = 2; pf = 1; 
G = rss(n1,p,m); F = rss(n2,pf,m);  
@time X, info = glasol(G, F; reltol=0.0001, atol = 1.e-9); info
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

# discrete-time, both G and F unstable and complex
n1 = 3; n2 = 2; m = 3; p = 2; pf = 1; 
G = rdss(n1,p,m,T = Complex{Float64},disc=true); F = rss(n2,pf,m,T = Complex{Float64},disc=true);  
@time X, info = glasol(G, F; reltol=0.0001, atol = 1.e-9); info
@test abs(glinfnorm(X*G-F,offset=1.e-10)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X,offset=1.e-13)) 

@time X, info = glasol(G, F; nehari = true, atol = 1.e-7); info
@test abs(glinfnorm(X*G-F)[1] - info.mindist) < 0.01 &&  (!info.nonstandard && isstable(X)) 


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
@time sysx, info = glasol(sysg, sysf, mindeg = true); info
@test glinfnorm(sysx*sysg-sysf)[1] ≈ info.mindist && info.nrank == 1 && info.nonstandard == false &&
      gpole(sysx) ≈ [-1]

s = rtf('s');
epsi = 1; 
g = (s-1)*(s-epsi)/(s+1)/(s+2); f = 1/(s+1); 
sysg = gir(dss(g)); sysf = dss(f);
@time sysx, info = glasol(sysg, sysf, mindeg = true); info
# check solution
γ0 = 0.25*(1+sqrt(1+8/(1+epsi))) 
α = 1 + 2(1 + epsi)*γ0
@test glinfnorm(sysx*sysg-sysf)[1] ≈ info.mindist && info.mindist ≈ γ0 && 
       gpole(sysx) ≈ [-α]

s = rtf('s');
epsi = 1; 
gf = [(s-1)*(s-epsi)/(s+1)/(s+2); 1/(s+1)]; 
sysgf = gir(dss(gf)); 
@time sysx, info = glasol(sysgf, 1, mindeg = true); info
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
@time sysx, mindist = glinfldp([sys1 sys2],1, reltol=1.e-9); 
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
@time sysx, mindist = glinfldp([sys1 sys2],1, reltol=1.e-9); 
@test glinfnorm([sys1-sys0 sys2])[1] ≈  glinfnorm([sys1-sysx sys2])[1] ≈ mindist ≈ 1/a

γ = 1.1/a;
@time sysx, mindist = glinfldp([sys1 sys2], 1, γ, reltol=1.e-9); 
@test glinfnorm([sys1-sysx sys2])[1] ≈ mindist < γ



s = rtf('s'); t = rtfbilin("tustin",Ts=0.5)[1];
a = 2; f1 = confmap(1/(s-1),t); f2 = confmap(1/(s-a),t); 
g0 = 1/(2*(a-1))*(-1 + sqrt(a^2+ 4*(a-1)/(a+1))); Q0 = -(g0*s+g0+(a-1)/2)/(s+a);
Q0 = confmap(-(g0*s+g0+(a-1)/2)/(s+a),t);
sys1 = dss(f1); sys2 = dss(f2); sys0 = dss(Q0);
@time sysx, mindist = glinfldp([sys1 sys2],1, reltol=1.e-9); 
@test glinfnorm([sys1-sys0 sys2])[1] ≈  glinfnorm([sys1-sysx sys2])[1] ≈ mindist ≈ g0


z = rtf('z');
G = [z^2+z+1 4*z^2+3*z+2 2*z^2-2;
    z 4*z-1 2*z-2;
    z^2 4*z^2-z 2*z^2-2*z]; 
sys = gir(dss(G),atol=1.e-7); 
@time sysx, s2 = glinfldp(sys; reltol=1.e-7); 
@test glinfnorm(sysx-sys)[1] ≈ s2 && s2 == 8.662176191833835

# poles on the imaginary axis
s = rtf('s');
sys = dss(1/s);
@test_throws ErrorException sysn, s1 = glinfldp(sys)

# Improper continuous-time system sys
s = rtf('s');
sys = dss(s);
@test_throws ErrorException sysn, s1 = glinfldp(sys)


end # glinfldp    
@testset "gnehari" begin
    

sys = rss(0,3,2) 
@time sysn, s1 = gnehari(sys,eps())
@test glinfnorm(sysn-sys)[1] == 0

sys = rss(0,10,2) 
@time sysn, s1 = gnehari(sys)
@test glinfnorm(sysn-sys)[1] == 0

# poles on the imaginary axis
s = rtf('s');
sys = dss(1/s);
@test_throws ErrorException sysn, s1 = gnehari(sys)

# Improper continuous-time system sys
s = rtf('s');
sys = dss(s);
@test_throws ErrorException sysn, s1 = gnehari(sys)

# poles on the unit circle
z = rtf('z');
sys = dss(1/(z-1));
@test_throws ErrorException sysn, s1 = gnehari(sys) 

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
@test glinfnorm(sysn-sys)[1] ≈ s1 && s1 == 8.662176191833835


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