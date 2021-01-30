module Test_iofac

using DescriptorSystems
using MatrixEquations
using MatrixPencils
using LinearAlgebra
using Polynomials
using Test

# some test randomly fail

@testset "giofac and goifac" begin

@testset "giofac" begin

sys = rdss(0,0,0);
@time sysi, syso, info = giofac(sys, minphase = false)
@test gnrank(sys-sysi*syso) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & (isproper(sys) ? isproper(syso) : true) &&
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      isempty(gzero(gminreal(syso))) &&
      info.nrank == 0 && ismissing(info.nfuz) && info.niuz == 0

@time sysi, syso, info = giofac(sys)
@test gnrank(sys-sysi*syso) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & (isproper(sys) ? isproper(syso) : true) &&
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      isempty(gzero(gminreal(syso))) &&
      info.nrank == 0 && info.nfuz == 0 && info.niuz == 0

sys = rdss(0,0,0,disc=true);
@time sysi, syso, info = giofac(sys, minphase = false)
@test gnrank(sys-sysi*syso) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & (isproper(sys) ? isproper(syso) : true) &&
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      isempty(gzero(gminreal(syso))) &&
      info.nrank == 0 && ismissing(info.nfuz) && info.niuz == 0

@time sysi, syso, info = giofac(sys)
@test gnrank(sys-sysi*syso) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & (isproper(sys) ? isproper(syso) : true) &&
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      isempty(gzero(gminreal(syso))) &&
      info.nrank == 0 && info.nfuz == 0 && info.niuz == 0


a = -(1. +2im)*ones(1,1); e = (1. -im)*ones(1,1); b = (2. +im)*ones(1,1); 
c = (-1. +im)*ones(1,1); d = complex(0)*ones(1,1);
sys = dss(a,e,b,c,d);
@time sysi, syso, info = giofac(sys, minphase = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test gnrank(sys-sysi*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & isproper(syso) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), gzero(gminreal(syso))) == 1 &&
      info.nrank == 1 && ismissing(info.nfuz) && info.niuz == 1

@time sysi, syso, info = giofac(sys, minphase = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test gnrank(sys-sysi*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & isproper(syso) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), gzero(gminreal(syso))) == 1 &&
      info.nrank == 1 && info.nfuz == 0 && info.niuz == 1

a = -(1. +2im)*ones(1,1); e = (1. -im)*ones(1,1); b = (2. +im)*ones(1,1); 
c = (-1. +im)*ones(1,1); d = complex(0)*ones(1,1);
sys = dss(a,e,b,c,d,Ts = 1);
@time sysi, syso, info = giofac(sys, minphase = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test gnrank(sys-sysi*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & isproper(syso) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), gzero(gminreal(syso))) == 1 &&
      info.nrank == 1 && ismissing(info.nfuz) && info.niuz == 1

@time sysi, syso, info = giofac(sys, minphase = true, atol1 = 1.e-7, atol2 = 1.e-7)
zer = gzero(gminreal(syso));
@test gnrank(sys-sysi*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & isproper(syso) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> abs.(t) .>= 1, zer) == 0 &&
      info.nrank == 1 && info.nfuz == 0 && info.niuz == 0

sys = rss(0,1,0);
@time sysi, syso, info = giofac(sys, minphase = false)
r = size(syso,1);
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & (isproper(sys) ? isproper(syso) : true) &&
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      isempty(gzero(gminreal(syso))) &&
      info.nrank == r && ismissing(info.nfuz) && info.niuz == 0

sys = rss(0,0,2);
@time sysi, syso, info = giofac(sys, minphase = false)
r = size(syso,1);
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & (isproper(sys) ? isproper(syso) : true) &&
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      isempty(gzero(gminreal(syso))) &&
      info.nrank == r && ismissing(info.nfuz) && info.niuz == 0


sys = dss([1 0]);
@time sysi, syso, info = giofac(sys, minphase = false)
r = size(syso,1);
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & (isproper(sys) ? isproper(syso) : true) &&
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      isempty(gzero(gminreal(syso))) &&
      info.nrank == r && ismissing(info.nfuz) && info.niuz == 0


sys = dss([0; 1]);
@time sysi, syso, info = giofac(sys, minphase = false)
r = size(syso,1);
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & (isproper(sys) ? isproper(syso) : true) &&
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      isempty(gzero(gminreal(syso))) &&
      info.nrank == r && ismissing(info.nfuz) && info.niuz == 0


sys = rss(0,3,2);
@time sysi, syso, info = giofac(sys, minphase = false)
r = size(syso,1);
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & (isproper(sys) ? isproper(syso) : true) &&
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      isempty(gzero(gminreal(syso))) &&
      info.nrank == r && ismissing(info.nfuz) && info.niuz == 0


z = Polynomial([0, 1],'z');     # define the complex variable z                     

# Gd = [z^2 z/(z-2); 0 1/z]     # define the 2-by-2 improper Gd(z)
Nd = [z^2 z; 0 1]; Dd = [1 z-2; 1 z]; 

# build LTI minimal descriptor realizations of Gd(z) 
sysd = dss(Nd,Dd,minimal = true,Ts = 1);      
zeref = gzero(sysd,atol1=1.e-7)

sysi, syso, info = giofac(sysd, atol = 1.e-7)  
r = size(syso,1);
zer = gzero(syso,atol1=1.e-7)
@test gnrank(sysd-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sysd) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sysd) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (abs.(t) .>= 1 && isfinite(t)), zer) == 0 &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0


s = Polynomial([0, 1],'s'); 
gn = [(s-1) s 1;
    0 (s-2) (s-2);
    (s-1) (s^2+2*s-2) (2*s-1)]; 
gd = [(s+2) (s+2) (s+2);
    1 (s+1)^2 (s+1)^2;
    (s+2) (s+1)*(s+2) (s+1)*(s+2)]; 

sys = dss(gn,gd,minimal = true, atol = 1.e-7); 
sysi, syso, info = giofac(sys, atol = 1.e-7)  
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7);
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) = Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) & isproper(syso) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .>= 0 && isfinite(t)), zer) == 0 &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 1

sysi, syso, info = giofac(sys, minphase = false, atol = 1.e-7)  
zeref = gzero(gminreal(sys),atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7);
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && isproper(syso) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .>= 0 && isfinite(t)), zer) == count(t -> (real.(t) .>= 0 && isfinite(t)), zeref) &&
      info.nrank == r && ismissing(info.nfuz) && info.niuz == 1


s = Polynomial([0, 1],'s'); 
g = [s^2+s+1 4*s^2+3*s+2 2*s^2-2;
    s 4*s-1 2*s-2;
    s^2 4*s^2-s 2*s^2-2*s]; 

sys = dss(g,minimal = true, atol = 1.e-7); 
sysi, syso, info = giofac(sys, atol = 1.e-7)  
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .<= 0 && isfinite(t)), zer) >= count(t -> (isfinite(t)), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0

z = Polynomial([0, 1],'z')
g = [z^2+z+1 4*z^2+3*z+2 2*z^2-2;
    z 4*z-1 2*z-2;
    z^2 4*z^2-z 2*z^2-2*z];

sys = dss(g,minimal = true, atol = 1.e-7,Ts = 1)
sysi, syso, info = giofac(sys, atol = 1.e-7)  
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1)
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(z) - Gi(z)*Go(z) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(z))*Gi(z)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (abs.(t) .<= 1), zer) >= count(t -> (abs.(t) .>= 1), zeref) &&
      info.nrank == r && info.nfuz == 1 && info.niuz == 0

z = Polynomial([0, 1],'z')
# Eigenvalue(s) on the unit circle 
gn = 1
gd = z-1
sys = dss(gn,gd,Ts = 1)
sysi, syso, info = giofac(sys, atol = 1.e-7)  
@test gnrank(sys-sysi*syso,atol1=1.e-7) == 0   &&  
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  


z = Polynomial([0, 1],'z')
gn = [z^4-z^3/2-16*z^2-29/2*z+18 z^4+5*z^3-z^2-11*z+6 11/2*z^3+15*z^2+7/2*z-12
    -3*z^2+12 z^3-z^2-4*z+4 z^3+2*z^2-4*z-8;
    z^4-z^3/2-19*z^2-23/2*z+24 z^4+6*z^3-3*z^2-12*z+8 13/2*z^3+16*z^2-z/2-16]; 
gd = [z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2;
      z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2;
      z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2];
# error
sys = dss(gn,gd,minimal = true, atol = 1.e-7,Ts = 1); 
sysi, syso, info = giofac(sys, atol = 1.e-7)  
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1)
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(z) - Gi(z)*Go(z) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(z))*Gi(z)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (abs.(t) .<= 1), zer) >= count(t -> (abs.(t) .>= 1), zeref) &&
      info.nrank == r && info.nfuz == 1 && info.niuz == 0

fast = true; Ty = Complex{Float64}; Ty = Float64     
n = 5; p = 1; m = 4; 

for fast in (true, false)
# random examples
for Ty in (Float64, Complex{Float64})

# continuous, standard
sys = rss(n,p,m,T = Ty,disc=false);
@time sysi, syso, info = giofac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .<= 0 && isfinite(t)), zer) >= count(t -> (isfinite(t)), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0

# discrete, standard
sys = rss(n,p,m,T = Ty,disc=true);
@time sysi, syso, info = giofac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(z) - Gi(z)*Go(z) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(u))*Gi(z)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (abs.(t) .<= 1), zer) >= count(t -> (abs.(t) .>= 1), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0


# continuous, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false);
@time sysi, syso, info = giofac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .<= 0 && isfinite(t)), zer) >= count(t -> (isfinite(t)), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0

# discrete, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty, disc=true);
@time sysi, syso, info = giofac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(z) - Gi(z)*Go(z) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(u))*Gi(z)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (abs.(t) .<= 1), zer) >= count(t -> (abs.(t) .>= 1), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0

# continuous, descriptor, proper
sys = rdss(n,p,m,T = Ty,disc=false,id=ones(Int,3));
@time sysi, syso, info = giofac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .<= 0 && isfinite(t)), zer) >= count(t -> (isfinite(t)), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0


# continuous, descriptor, proper
sys = rdss(n,p,m,T = Ty,disc=false,id=ones(Int,3));
@time sysi, syso, info = giofac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .<= 0 && isfinite(t)), zer) >= count(t -> (isfinite(t)), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0

# discrete, descriptor, proper
sys = rdss(n,p,m,T = Ty, disc=true,id=ones(Int,3));
@time sysi, syso, info = giofac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(z) - Gi(z)*Go(z) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(u))*Gi(z)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (abs.(t) .<= 1), zer) >= count(t -> (abs.(t) .>= 1), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0

# continuous, descriptor, infinite poles
sys = rdss(n,p,m,T = Ty,disc=false,id=[3*ones(Int,1);2*ones(Int,1)]);
@time sysi, syso, info = giofac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .<= 0 && isfinite(t)), zer) >= count(t -> (isfinite(t)), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0


# discrete, descriptor, infinite poles
sys = rdss(n,p,m,T = Ty, disc=true,id=[3*ones(Int,1);2*ones(Int,1)]);
@time sysi, syso, info = giofac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(z) - Gi(z)*Go(z) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(u))*Gi(z)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (abs.(t) .<= 1), zer) >= count(t -> (abs.(t) .>= 1), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0

# continuous, descriptor, proper, uncontrollable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false,iduc=[3*ones(Int,1);2*ones(Int,1)]);
@time sysi, syso, info = giofac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .<= 0 && isfinite(t)), zer) >= count(t -> (isfinite(t)), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0
      
# fail
# discrete, descriptor, proper, uncontrollable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true,iduc=[3*ones(Int,1);2*ones(Int,1)]);
@time sysi, syso, info = giofac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(gminreal(sys),atol1=1.e-7,atol2=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7,atol2=1.e-7)
@test gnrank(sys-sysi[:,1:r]*syso,atol1=1.e-7) == 0   &&   #  G(z) - Gi(z)*Go(z) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(u))*Gi(z)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (abs.(t) .<= 1), zer) >= count(t -> (abs.(t) .>= 1), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0



end
end

end # giofac

@testset "goifac" begin

sys = rdss(0,0,0);
@time sysi, syso, info = goifac(sys, minphase = false)
@test gnrank(sys-syso*sysi) == 0   &&   #  G(s) - Go(s)*Gi(s) = 0
       gnrank(sysi'*sysi-I,atol=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
       isproper(sysi) & (isproper(sys) ? isproper(syso) : true) &&
       isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
       isempty(gzero(gminreal(syso))) &&
       info.nrank == 0 && ismissing(info.nfuz) && info.niuz == 0

fast = true; Ty = Complex{Float64}; Ty = Float64     
n = 5; p = 1; m = 4; 
for fast in (true, false)
# random examples
for Ty in (Float64, Complex{Float64})

# continuous, standard
sys = rss(n,p,m,T = Ty,disc=false);
@time sysi, syso, info = goifac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-syso*sysi[1:r,:],atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .<= 0 && isfinite(t)), zer) >= count(t -> (isfinite(t)), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0

# discrete, standard
sys = rss(n,p,m,T = Ty,disc=true);
@time sysi, syso, info = goifac(sys, fast = fast, atol = 1.e-7) ; 
zeref = gzero(sys,atol1=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol1=1.e-7)
@test gnrank(sys-syso*sysi[1:r,:],atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .<= 0 && isfinite(t)), zer) >= count(t -> (isfinite(t)), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0

# continuous, descriptor, infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false,id=[3*ones(Int,1);2*ones(Int,1)],randlt=true,randrt=false);
@time sysi, syso, info = goifac(sys, fast = fast, atol = 1.e-7) ; info
zeref = gzero(sys,atol=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso),atol=1.e-7)
@test gnrank(sys-syso*sysi[1:r,:],atol1=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      gnrank(sysi'*sysi-I,atol1=1.e-7) == 0  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(t -> isinf(t), zer) == info.niuz &&
      count(t -> (real.(t) .<= 0 && isfinite(t)), zer) >= count(t -> (isfinite(t)), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0

# discrete, descriptor, infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true,id=[ones(Int,3);2*ones(Int,2)],randlt=true,randrt=true);
@time sysi, syso, info = goifac(sys, fast = fast, atol = 1.e-7) ; info
zeref = gzero(sys,atol=1.e-7)
r = size(syso,1);
zer = gzero(gminreal(syso,atol=1.e-7),atol=1.e-7)
@test gnrank(sys-syso*sysi[1:r,:],atol=1.e-7) == 0   &&   #  G(s) - Gi(s)*Go(s) = 0
      iszero(sysi'*sysi-I,atol=1.e-5)  && # conj(Gi(s))*Gi(s)-I = 0
      isproper(sysi) && (isproper(sys) ? isproper(syso) : true) && # checking properness of factors
      isstable(sysi) && (isstable(sys) ? isstable(syso) : true) &&
      count(isinf.(zer)) == info.niuz &&
      count(t -> (real.(t) .<= 0 && isfinite(t)), zer) >= count(t -> (isfinite(t)), zeref) &&
      info.nrank == r && info.nfuz == 0 && info.niuz == 0


end
end

end # goifac


end #test

end #module





