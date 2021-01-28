module Test_pscf

using DescriptorSystems
using MatrixEquations
using MatrixPencils
using LinearAlgebra
using Polynomials
using Test


@testset "grcf and glcf" begin

@testset "grcf" begin

sys = rdss(0,0,0);
@time sysn, sysm = grcf(sys);
@test gnrank(sys*sysm-sysn) == 0   &&   #  G(s)*M(s)-N(s) = 0
      isproper(sysm) & isproper(sysn) && # checking properness of factors
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

a = -(1. +2im)*ones(1,1); e = (1. -im)*ones(1,1); b = (2. +im)*ones(1,1); 
c = (-1. +im)*ones(1,1); d = complex(0)*ones(1,1);
sys = dss(a,e,b,c,d);
@time sysn, sysm = grcf(sys,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0 &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

a = -(1. +2im)*ones(1,1); e = (1. -im)*ones(1,1); b = (2. +im)*ones(1,1); 
c = (-1. +im)*ones(1,1); d = complex(0)*ones(1,1);
sys = dss(a,e,b,c,d,Ts=1);
@time sysn, sysm = grcf(sys,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0 &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

s = Polynomial([0, 1],'s'); z = Polynomial([0, 1],'z');     # define the complex variables s and z                     

# Gc = [s^2 s/(s+1); 0 1/s]     # define the 2-by-2 improper Gc(s)
Nc = [s^2 s; 0 1]; Dc = [1 (s+1); 1 s]; 
sysc = dss(Nc,Dc);             # build continuous-time descriptor system realization
sysn, sysm = grcf(sysc, evals = [-2,-3,-4], sdeg = -0.99, mindeg = true, mininf = true);
@test gnrank(sysc*sysm-sysn,atol=1.e-7) == 0 &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))



# Gd = [z^2 z/(z-2); 0 1/z]     # define the 2-by-2 improper Gd(z)
Nd = [z^2 z; 0 1]; Dd = [1 z-2; 1 z]; 
# build LTI (nonminimal) descriptor realizations of Gc(s) and Gd(z) 
sysd = dss(Nd,Dd,Ts = 1);      # build discrete-time descriptor system realization

sysn, sysm = grcf(sysd);
@test gnrank(sysd*sysm-sysn,atol=1.e-7) == 0  &&    #  Gd(z)*M(z)-N(z) = 0
      isproper(sysm) & isproper(sysn)  &&  
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

s = Polynomial([0, 1],'s'); 
gn = [(s-1) s 1;
    0 (s-2) (s-2);
    (s-1) (s^2+2*s-2) (2*s-1)]; 
gd = [(s+2) (s+2) (s+2);
    1 (s+1)^2 (s+1)^2;
    (s+2) (s+1)*(s+2) (s+1)*(s+2)]; 
sys = dss(gn,gd,minimal = true, atol = 1.e-7); 

sysn, sysm = grcf(sys);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0  &&    
      isproper(sysm) & isproper(sysn)  &&  
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7))) &&
      order(sysm) == 0


sysn, sysm = grcf(sys',sdeg=-0.5);
@test gnrank(sys'*sysm-sysn,atol=1.e-7) == 0  &&    
      isproper(sysm) & isproper(sysn)  &&  
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7))) &&
      order(sysm) == 4

s = Polynomial([0, 1],'s'); 
gn = [(s-1) s 1;
    0 (s-2) (s-2);
    (s-1) (s^2+2*s-2) (2*s-1)]; 
gd = [(s-2) (s+2) (s+2);
    1 (s+1)^2 (s+1)^2;
    (s+2) (s-1)*(s+2) (s+1)*(s-2)];

sys = dss(gn,gd,minimal = true, atol = 1.e-7); 
sysn, sysm = grcf(sys,mindeg = true, atol = 1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0  &&    
      isproper(sysm) & isproper(sysn)  &&  
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7))) &&
      order(sysm) == 3


z = Polynomial([0, 1],'z')
g = [z^2+z+1 4*z^2+3*z+2 2*z^2-2;
    z 4*z-1 2*z-2;
    z^2 4*z^2-z 2*z^2-2*z];
sys = dss(g,minimal = true, atol = 1.e-7,Ts = 1); 

sysn, sysm = grcf(sys,mindeg = true,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0  &&    #  Gd(z)*M(z)-N(z) = 0
      isproper(sysm) & isproper(sysn)  &&  
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7))) 

z = Polynomial([0, 1],'z')
# Eigenvalue(s) on the unit circle 
gn = 1
gd = z-1
sys = dss(gn,gd,Ts = 1)
sysn, sysm = grcf(sys)  
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0  &&    #  Gd(z)*M(z)-N(z) = 0
      isproper(sysm) & isproper(sysn)  &&  
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7))) 

fast = true; Ty = Complex{Float64}; Ty = Float64     
n = 5; p = 3; m = 2; 

for fast in (true, false)
# random examples
for Ty in (Float64, Complex{Float64})

# continuous, standard
sys = rss(n,p,m,T = Ty,disc=false);
@time sysn, sysm = grcf(sys, smarg = -1, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,-0.99, atol=1.e-7) && 
      isstable(sysn,-0.99,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7),atol=1.e-7))

#fail
sys = rss(n,p,m,T = Ty,disc=false);
@time sysn, sysm = grcf(sys, smarg = -1, evals = -Vector(1:n),fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,-0.99, atol=1.e-7) && 
      isstable(sysn,-0.99,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7),atol=1.e-7))


# discrete, standard
sys = rss(n,p,m,T = Ty,disc=true);
@time sysn, sysm = grcf(sys, smarg = 0.8, sdeg = 0.5, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,0.81,atol=1.e-7) && 
      isstable(sysn,0.81,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7),atol=1.e-7))
     
# discrete, standard
sys = rss(n,p,m,T = Ty,disc=true);
@time sysn, sysm = grcf(sys, evals = -Vector(1:n)/10, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7),atol=1.e-7))
     
# continuous, standard, uncontrollable
sys = rss(n,p,m,T = Ty,disc=false, nuc = 5);
@time sysn, sysm = grcf(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7),atol=1.e-7))

# discrete, standard, uncontrollable
sys = rss(n,p,m,T = Ty,disc=true, nuc = 5);
@time sysn, sysm = grcf(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7),atol=1.e-7))


# continuous, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false);
@time sysn, sysm = grcf(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

# discrete, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty, disc=true);
@time sysn, sysm = grcf(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

# continuous, descriptor, proper
sys = rdss(n,p,m,T = Ty,disc=false,id=ones(Int,3));
@time sysn, sysm = grcf(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

# discrete, descriptor, proper
sys = rdss(n,p,m,T = Ty, disc=true);
@time sysn, sysm = grcf(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

# continuous, descriptor, proper
sys = rdss(n,p,m,T = Ty,disc=false,id=ones(Int,3));
@time sysn, sysm = grcf(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

# discrete, descriptor, proper
sys = rdss(n,p,m,T = Ty, disc=true,id=ones(Int,3));
@time sysn, sysm = grcf(sys, fast = fast,mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

# continuous, descriptor, infinite poles 
sys = rdss(n,p,m,T = Ty,disc=false,id=[3*ones(Int,1);2*ones(Int,1)]);
@time sysn, sysm = grcf(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
#@time sysn, sysm = grcf(sys, fast = fast, mindeg = false, mininf = false, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))


# discrete, descriptor, infinite poles
sys = rdss(n,p,m,T = Ty, disc=true,id=[3*ones(Int,1);2*ones(Int,1)]);
@time sysn, sysm = grcf(sys, fast = fast,mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

# continuous, descriptor, proper, uncontrollable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false,iduc=[3*ones(Int,1);2*ones(Int,1)]);
@time sysn, sysm = grcf(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

# discrete, descriptor, proper, uncontrollable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true,iduc=[3*ones(Int,1);2*ones(Int,1)]);
@time sysn, sysm = grcf(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

# continuous, descriptor, proper, uncontrollable finite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false, nfuc = 5);
@time sysn, sysm = grcf(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

# discrete, descriptor, proper, uncontrollable finite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true, nfuc = 5);
@time sysn, sysm = grcf(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sys*sysm-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn;sysm],atol=1.e-7)))

end
end

end # grcfid

@testset "glcf" begin

sys = rdss(0,0,0);
@time sysn, sysm = glcf(sys);
@test gnrank(sysm*sys-sysn) == 0   &&   #  M(s)*G(s)-N(s) = 0
      isproper(sysm) & isproper(sysn) && # checking properness of factors
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn sysm])))

fast = true; Ty = Complex{Float64}; #Ty = Float64     
n = 5; p = 3; m = 2; 
for fast in (true, false)
# random examples
for Ty in (Float64, Complex{Float64})

# continuous, standard
sys = rss(n,p,m,T = Ty,disc=false);
@time sysn, sysm = glcf(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sysm*sys-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn sysm],atol=1.e-7),atol=1.e-7))

# discrete, standard
sys = rss(n,p,m,T = Ty,disc=true);
@time sysn, sysm = glcf(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sysm*sys-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn sysm],atol=1.e-7),atol=1.e-7))

# continuous, standard, unobservable
sys = rss(n,p,m,T = Ty,disc=false, nuo = 5);
@time sysn, sysm = glcf(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sysm*sys-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn sysm],atol=1.e-7),atol=1.e-7))

# discrete, standard, unobservable
sys = rss(n,p,m,T = Ty,disc=true, nuo = 5);
@time sysn, sysm = glcf(sys, fast = fast,atol=1.e-7,atol3=1.e-7);
@test gnrank(sysm*sys-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn sysm],atol=1.e-7),atol=1.e-7))

# continuous, descriptor, proper, unobservable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false,iduo=[3*ones(Int,1);2*ones(Int,1)]);
@time sysn, sysm = glcf(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sysm*sys-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn sysm])))

# discrete, descriptor, proper, unobservable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true,iduo=[3*ones(Int,1);2*ones(Int,1)]);
@time sysn, sysm = glcf(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sysm*sys-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn sysm])))

# continuous, descriptor, proper, unobservable finite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false, nfuo = 5);
@time sysn, sysm = glcf(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sysm*sys-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn sysm])))

# discrete, descriptor, proper, unobservable finite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true, nfuo = 5);
@time sysn, sysm = glcf(sys, fast = fast, mindeg = true, mininf = true, atol=1.e-7,atol3=1.e-7);
@test gnrank(sysm*sys-sysn,atol=1.e-7) == 0   &&   
      isproper(sysm) && isproper(sysn) && 
      isstable(sysm,atol=1.e-7) && 
      isstable(sysn,atol=1.e-7)  &&
      isempty(gzero(gminreal([sysn sysm])))

end
end

end # glcf


end #test

end #module





