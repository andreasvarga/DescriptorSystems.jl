module Test_cfid

using DescriptorSystems
using MatrixEquations
using MatrixPencils
using LinearAlgebra
using Polynomials
using Test


@testset "grcfid and glcfid" begin

@testset "grcfid" begin

sys = rdss(0,0,0);
@time sysni, sysmi = grcfid(sys);
@test gnrank(sys*sysmi-sysni) == 0   &&   #  G(s)*Mi(s)-Ni(s) = 0
      gnrank(sysmi'*sysmi-I) == 0  && # conj(Mi(s))*Mi(s)-I = 0
      isproper(sysmi) & isproper(sysni) && # checking properness of factors
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

a = -(1. +2im)*ones(1,1); e = (1. -im)*ones(1,1); b = (2. +im)*ones(1,1); 
c = (-1. +im)*ones(1,1); d = complex(0)*ones(1,1);
sys = dss(a,e,b,c,d);
@time sysni, sysmi = grcfid(sys,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0 &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

a = -(1. +2im)*ones(1,1); e = (1. -im)*ones(1,1); b = (2. +im)*ones(1,1); 
c = (-1. +im)*ones(1,1); d = complex(0)*ones(1,1);
sys = dss(a,e,b,c,d,Ts=1);
@time sysni, sysmi = grcfid(sys,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0 &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

s = Polynomial([0, 1],'s'); z = Polynomial([0, 1],'z');     # define the complex variables s and z                     

# Gd = [z^2 z/(z-2); 0 1/z]     # define the 2-by-2 improper Gd(z)
Nd = [z^2 z; 0 1]; Dd = [1 z-2; 1 z]; 

# build LTI (nonminimal) descriptor realizations of Gc(s) and Gd(z) 
sysd = dss(Nd,Dd,Ts = 1);      # build discrete-time descriptor system realization

sysni, sysmi = grcfid(sysd);
@test gnrank(sysd*sysmi-sysni,atol=1.e-7) == 0  &&    #  Gd(z)*M(z)-N(z) = 0
      iszero(sysmi'-inv(sysmi),atol=1.e-7)  && # conj(M(z))*M(z)-I = 0
      isproper(sysmi) & isproper(sysni)  &&  
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

s = Polynomial([0, 1],'s'); 
gn = [(s-1) s 1;
    0 (s-2) (s-2);
    (s-1) (s^2+2*s-2) (2*s-1)]; 
gd = [(s+2) (s+2) (s+2);
    1 (s+1)^2 (s+1)^2;
    (s+2) (s+1)*(s+2) (s+1)*(s+2)]; 

sys = dss(gn,gd,minimal = true, atol = 1.e-7); 
sysni, sysmi = grcfid(sys);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0  &&    #  Gd(z)*M(z)-N(z) = 0
      iszero(sysmi'-inv(sysmi),atol=1.e-7)  && # conj(M(z))*M(z)-I = 0
      isproper(sysmi) & isproper(sysni)  &&  
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7))) &&
      order(sysmi) == 0

sys = sys'
sysni, sysmi = grcfid(sys);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0  &&    #  Gd(z)*M(z)-N(z) = 0
      iszero(sysmi'-inv(sysmi),atol=1.e-7)  && # conj(M(z))*M(z)-I = 0
      isproper(sysmi) & isproper(sysni)  &&  
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7))) &&
      order(sysmi) == 4

s = Polynomial([0, 1],'s'); 
gn = [(s-1) s 1;
    0 (s-2) (s-2);
    (s-1) (s^2+2*s-2) (2*s-1)]; 
gd = [(s-2) (s+2) (s+2);
    1 (s+1)^2 (s+1)^2;
    (s+2) (s-1)*(s+2) (s+1)*(s-2)];

sys = dss(gn,gd,minimal = true, atol = 1.e-7); 
sysni, sysmi = grcfid(sys,mindeg = true);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0  &&    #  Gd(z)*M(z)-N(z) = 0
      iszero(sysmi'-inv(sysmi),atol=1.e-7)  && # conj(M(z))*M(z)-I = 0
      isproper(sysmi) & isproper(sysni)  &&  
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7))) &&
      order(sysmi) == 3

z = Polynomial([0, 1],'z')
g = [z^2+z+1 4*z^2+3*z+2 2*z^2-2;
    z 4*z-1 2*z-2;
    z^2 4*z^2-z 2*z^2-2*z];

sys = dss(g,minimal = true, atol = 1.e-7,Ts = 1); 
sysni, sysmi = grcfid(sys,mindeg = true,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0  &&    #  Gd(z)*M(z)-N(z) = 0
      iszero(sysmi'-inv(sysmi),atol=1.e-7)  && # conj(M(z))*M(z)-I = 0
      isproper(sysmi) & isproper(sysni)  &&  
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7))) 

z = Polynomial([0, 1],'z')
# Eigenvalue(s) on the unit circle 
gn = 1
gd = z-1
sys = dss(gn,gd,Ts = 1)
try
   sysni, sysmi = grcfid(sys)  
   @test false
catch
   @test true
end

fast = true; Ty = Complex{Float64}; #Ty = Float64     
n = 5; p = 3; m = 2; 

for fast in (true, false)
# random examples
for Ty in (Float64, Complex{Float64})

# continuous, standard
sys = rss(n,p,m,T = Ty,disc=false);
@time sysni, sysmi = grcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      iszero(sysmi'-inv(sysmi),atol=1.e-7)   && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7),atol=1.e-7))

# discrete, standard
sys = rss(n,p,m,T = Ty,disc=true);
@time sysni, sysmi = grcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      iszero(sysmi'-inv(sysmi),atol=1.e-7)   && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7),atol=1.e-7))

# continuous, standard, uncontrollable
sys = rss(n,p,m,T = Ty,disc=false, nuc = 5);
@time sysni, sysmi = grcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      iszero(sysmi'-inv(sysmi),atol=1.e-7)   && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7),atol=1.e-7))

# discrete, standard, uncontrollable
sys = rss(n,p,m,T = Ty,disc=true, nuc = 5);
@time sysni, sysmi = grcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      iszero(sysmi'-inv(sysmi),atol=1.e-7)   && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7),atol=1.e-7))

#failure
# continuous, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false);
@time sysni, sysmi = grcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

# discrete, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty, disc=true);
@time sysni, sysmi = grcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

# continuous, descriptor, proper
sys = rdss(n,p,m,T = Ty,disc=false,id=ones(Int,3));
@time sysni, sysmi = grcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

# discrete, descriptor, proper
sys = rdss(n,p,m,T = Ty, disc=true);
@time sysni, sysmi = grcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

# continuous, descriptor, proper
sys = rdss(n,p,m,T = Ty,disc=false,id=ones(Int,3));
@time sysni, sysmi = grcfid(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

# discrete, descriptor, proper
sys = rdss(n,p,m,T = Ty, disc=true,id=ones(Int,3));
@time sysni, sysmi = grcfid(sys, fast = fast,mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

# continuous, descriptor, infinite poles -> error
sys = rdss(n,p,m,T = Ty,disc=false,id=[3*ones(Int,1);2*ones(Int,1)]);
try
   @time sysni, sysmi = grcfid(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
   @test false
catch
   @test true
end

# discrete, descriptor, infinite poles
sys = rdss(n,p,m,T = Ty, disc=true,id=[3*ones(Int,1);2*ones(Int,1)]);
@time sysni, sysmi = grcfid(sys, fast = fast,mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

# continuous, descriptor, proper, uncontrollable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false,iduc=[3*ones(Int,1);2*ones(Int,1)]);
@time sysni, sysmi = grcfid(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

# discrete, descriptor, proper, uncontrollable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true,iduc=[3*ones(Int,1);2*ones(Int,1)]);
@time sysni, sysmi = grcfid(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

# continuous, descriptor, proper, uncontrollable finite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false, nfuc = 5);
@time sysni, sysmi = grcfid(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

# discrete, descriptor, proper, uncontrollable finite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true, nfuc = 5);
@time sysni, sysmi = grcfid(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysmi-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni;sysmi],atol=1.e-7)))

end
end

end # grcfid

@testset "glcfid" begin

sys = rdss(0,0,0);
@time sysni, sysmi = glcfid(sys);
@test gnrank(sysmi*sys-sysni) == 0   &&   #  Mi(s)*G(s)-Ni(s) = 0
      gnrank(sysmi*sysmi'-I) == 0  && # Mi(s)*conj(Mi(s))-I = 0
      isproper(sysmi) & isproper(sysni) && # checking properness of factors
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni sysmi])))

fast = true; Ty = Complex{Float64}; #Ty = Float64     
n = 5; p = 3; m = 2; 
for fast in (true, false)
# random examples
for Ty in (Float64, Complex{Float64})

# continuous, standard
sys = rss(n,p,m,T = Ty,disc=false);
@time sysni, sysmi = glcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sysmi*sys-sysni,atol=1.e-7) == 0   &&   
      iszero(sysmi'-inv(sysmi),atol=1.e-7)   && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni sysmi],atol=1.e-7),atol=1.e-7))

# discrete, standard
sys = rss(n,p,m,T = Ty,disc=true);
@time sysni, sysmi = glcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sysmi*sys-sysni,atol=1.e-7) == 0   &&   
      iszero(sysmi'-inv(sysmi),atol=1.e-7)   && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni sysmi],atol=1.e-7),atol=1.e-7))

# continuous, standard, unobservable
sys = rss(n,p,m,T = Ty,disc=false, nuo = 5);
@time sysni, sysmi = glcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sysmi*sys-sysni,atol=1.e-7) == 0   &&   
      iszero(sysmi'-inv(sysmi),atol=1.e-7)   && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni sysmi],atol=1.e-7),atol=1.e-7))

# discrete, standard, unobservable
sys = rss(n,p,m,T = Ty,disc=true, nuo = 5);
@time sysni, sysmi = glcfid(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sysmi*sys-sysni,atol=1.e-7) == 0   &&   
      iszero(sysmi'-inv(sysmi),atol=1.e-7)   && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni sysmi],atol=1.e-7),atol=1.e-7))

# continuous, descriptor, proper, unobservable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false,iduo=[3*ones(Int,1);2*ones(Int,1)],randlt=true,randrt=false);
sys = rdss(n,p,m,T = Ty,disc=false,randlt=true,randrt=false);
@time sysni, sysmi = glcfid(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@time sysni, sysmi = glcfid(sys, fast = false, mindeg = false, mininf = false, atol=1.e-7,atol3=1.e-7);
@test gnrank(sysmi*sys-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni sysmi])))

# discrete, descriptor, proper, unobservable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true,iduo=[3*ones(Int,1);2*ones(Int,1)]);
@time sysni, sysmi = glcfid(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sysmi*sys-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni sysmi])))

# continuous, descriptor, proper, unobservable finite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false, nfuo = 5);
@time sysni, sysmi = glcfid(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sysmi*sys-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni sysmi])))

# discrete, descriptor, proper, unobservable finite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true, nfuo = 5);
@time sysni, sysmi = glcfid(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sysmi*sys-sysni,atol=1.e-7) == 0   &&   
      gnrank(sysmi'*sysmi-I,atol=1.e-7) == 0  && 
      isproper(sysmi) && isproper(sysni) && 
      isstable(sysmi,atol=1.e-7) && 
      isstable(sysni,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysni sysmi])))

end
end

end # glcfid


end #test

end #module





