module Test_conversions

using DescriptorSystems
using MatrixEquations
using MatrixPencils
using LinearAlgebra
using Polynomials
using Test


@testset "gbilin" begin

s = rtf('s'); # define the complex variable s
Gc = [s^2 s/(s+1); 0 1/s] # define the 2-by-2 improper Gc(s)
sysc = dss(Gc);            # build continuous-time descriptor system realization
g = (s+0.01)/(1+0.01*s); 

syst, = gbilin(sysc,g,atol = 1.e-7,minimal=true); 

@test opnorm(evalfr(sysc-syst,1)) < 1.e-7
# compute the ν-gap
# nugap = gnugap(sysc,syst,atol=1.e-4)


z = rtf('z');
Gd = [z^2 z/(z-2); 0 1/z]     # define the 2-by-2 improper Gd(z)
syst, = gbilin(dss(Gd),rtfbilin("cayley")[2],atol=1.e-7,minimal = true); 
sysi, = gbilin(syst,rtfbilin("c2d")[1],atol = 1.e-7,minimal = true); 

@test ghinfnorm(gminreal(dss(Gd)-sysi,atol=1.e-7))[1] < 1.e-7

for Ty in (Float64,Complex{Float64})

type = "c2d"; 
sys = rss(T = Ty, 3,2,3); 
g, ginv = rtfbilin(type)
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "c2d"; 
sys = rdss(T = Ty, 3,2,3); 
g, ginv = rtfbilin(type)
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "cayley"; 
sys = rss(T = Ty, 3,2,3,disc = true); 
g, ginv = rtfbilin(type)
syst, g1 = gbilin(sys,ginv,atol=1.e-7); 
sysi, gi1 = gbilin(syst,g,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && g1 == g && ginv == gi1

type = "cayley"; 
sys = rdss(T = Ty, 3,2,3,disc = true); 
g, ginv = rtfbilin(type)
syst, g1 = gbilin(sys,ginv,atol=1.e-7); 
sysi, gi1 = gbilin(syst,g,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && g1 == g && ginv == gi1

type = "tustin"; 
sys = rss(T = Ty, 3,2,3); 
g, ginv = rtfbilin(type,Ts=0.1)
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1


type = "lft"; 
sys = rss(T = Ty, 3,2,3); 
val = rand(Ty,4)
g, ginv = rtfbilin(type,a=val[1],b=val[2],c=val[3],d=val[4])
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rss(T = Ty, 3,2,3,disc=true); 
val = rand(Ty,4)
g, ginv = rtfbilin(type,Ts=1, Tsi=1, a=val[1],b=val[2],c=val[3],d=val[4])
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rss(T = Ty, 3,2,3,disc=true); 
val = rand(Ty,4)
g, ginv = rtfbilin(type,Ts=0, Tsi=-1, a=val[1],b=val[2],c=val[3],d=val[4])
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rss(T = Ty, 3,2,3,disc=false); 
val = rand(Ty,4)
g, ginv = rtfbilin(type,Ts=-1, Tsi=0, a=val[1],b=val[2],c=val[3],d=val[4])
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rss(T = Ty, 3,2,3,disc=false); 
val = rand(Ty,2)
g, ginv = rtfbilin(type,Ts=-1, Tsi=0, a=val[1],b=val[2])
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi ≈ ginv && g ≈ g1

type = "lft"; 
sys = rdss(T = Ty, 3,2,3); 
val = rand(Ty,4)
g, ginv = rtfbilin(type,a=val[1],b=val[2],c=val[3],d=val[4])
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rdss(T = Ty, 3,2,3); 
val = rand(Ty,2)
g, ginv = rtfbilin(type,a=val[1],b=val[2])
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi ≈ ginv && g ≈ g1

type = "lft"; 
sys = rdss(T = Ty, 3,2,3,disc=true); 
val = rand(Ty,4)
g, ginv = rtfbilin(type,Ts=1, Tsi=1, a=val[1],b=val[2],c=val[3],d=val[4])
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rdss(T = Ty, 3,2,3,disc=true); 
val = rand(4)
g, ginv = rtfbilin(type,Ts=0, Tsi=-1, a=val[1],b=val[2],c=val[3],d=val[4])
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rdss(T = Ty, 3,2,3,disc=false); 
val = rand(Ty,4)
g, ginv = rtfbilin(type,Ts=-1, Tsi=0, a=val[1],b=val[2],c=val[3],d=val[4])
syst, gi = gbilin(sys,g,atol=1.e-7); 
sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

end #Ty
end # gbilin


end #module





