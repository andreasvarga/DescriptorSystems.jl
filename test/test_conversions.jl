module Test_conversions

using DescriptorSystems
using LinearAlgebra
using Polynomials
using Test


println("Test_conversions")

@testset "gbilin" begin

s = rtf('s'); # define the complex variable s
Gc = [s^2 s/(s+1); 0 1/s] # define the 2-by-2 improper Gc(s)
@time sysc = dss(Gc);            # build continuous-time descriptor system realization
g = (s+0.01)/(1+0.01*s); 

@time syst, = gbilin(sysc,g,atol = 1.e-7,minimal=true); 

@test opnorm(evalfr(sysc-syst,1)) < 1.e-7
# compute the ν-gap
@time nugap = gnugap(sysc,syst,atol=1.e-4)[1];
@test nugap < 0.02


z = rtf('z');
Gd = [z^2 z/(z-2); 0 1/z]     # define the 2-by-2 improper Gd(z)
@time syst, = gbilin(dss(Gd),rtfbilin("cayley")[2],atol=1.e-7,minimal = true); 
@time sysi, = gbilin(syst,rtfbilin("c2d")[1],atol = 1.e-7,minimal = true); 

@test ghinfnorm(gminreal(dss(Gd)-sysi,atol=1.e-7))[1] < 1.e-7

for Ty in (Float64,Complex{Float64})

type = "c2d"; 
sys = rss(T = Ty, 3,2,3); 
@time g, ginv = rtfbilin(type)
@time syst, gi = gbilin(sys,g,atol=1.e-7); 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "c2d"; 
sys = rdss(T = Ty, 3,2,3); 
@time g, ginv = rtfbilin(type)
@time syst, gi = gbilin(sys,g,atol=1.e-7); 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "cayley"; 
sys = rss(T = Ty, 3,2,3,disc = true); 
@time g, ginv = rtfbilin(type)
@time syst, g1 = gbilin(sys,ginv,atol=1.e-7); 
@time sysi, gi1 = gbilin(syst,g,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && g1 == g && ginv == gi1

type = "cayley"; 
sys = rdss(T = Ty, 3,2,3,disc = true); 
@time g, ginv = rtfbilin(type)
@time syst, g1 = gbilin(sys,ginv,atol=1.e-7); 
@time sysi, gi1 = gbilin(syst,g,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && g1 == g && ginv == gi1

type = "tustin"; 
sys = rss(T = Ty, 3,2,3); 
@time g, ginv = rtfbilin(type,Ts=0.1)
@time syst, gi = gbilin(sys,g,atol=1.e-7); 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1


type = "lft"; 
sys = rss(T = Ty, 3,2,3); 
val = rand(Ty,4)
@time g, ginv = rtfbilin(type,a=val[1],b=val[2],c=val[3],d=val[4])
@time syst, gi = gbilin(sys,g,atol=1.e-7); 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rss(T = Ty, 3,2,3,disc=true); 
val = rand(Ty,4)
@time g, ginv = rtfbilin(type,Ts=1, Tsi=1, a=val[1],b=val[2],c=val[3],d=val[4])
@time syst, gi = gbilin(sys,g,atol=1.e-7); 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rss(T = Ty, 3,2,3,disc=true); 
val = rand(Ty,4)
@time g, ginv = rtfbilin(type,Ts=0, Tsi=-1, a=val[1],b=val[2],c=val[3],d=val[4])
@time syst, gi = gbilin(sys,g,atol=1.e-7); 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rss(T = Ty, 3,2,3,disc=false); 
val = rand(Ty,4)
@time g, ginv = rtfbilin(type,Ts=-1, Tsi=0, a=val[1],b=val[2],c=val[3],d=val[4])
@time syst, gi = gbilin(sys,g,atol=1.e-7); 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rss(T = Ty, 3,2,3,disc=false); 
val = rand(Ty,2)
@time g, ginv = rtfbilin(type,Ts=-1, Tsi=0, a=val[1],b=val[2])
@time syst, gi = gbilin(sys,g,atol=1.e-7); 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
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
@time g, ginv = rtfbilin(type,a=val[1],b=val[2])
@time syst, gi = gbilin(sys,g,atol=1.e-7); 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi ≈ ginv && g ≈ g1

type = "lft"; 
sys = rdss(T = Ty, 3,2,3,disc=true); 
val = rand(Ty,4)
@time g, ginv = rtfbilin(type,Ts=1, Tsi=1, a=val[1],b=val[2],c=val[3],d=val[4])
@time syst, gi = gbilin(sys,g,atol=1.e-7) 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rdss(T = Ty, 3,2,3,disc=true); 
val = rand(4)
@time g, ginv = rtfbilin(type,Ts=0, Tsi=-1, a=val[1],b=val[2],c=val[3],d=val[4])
@time syst, gi = gbilin(sys,g,atol=1.e-7); 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

type = "lft"; 
sys = rdss(T = Ty, 3,2,3,disc=false); 
val = rand(Ty,4)
@time g, ginv = rtfbilin(type,Ts=-1, Tsi=0, a=val[1],b=val[2],c=val[3],d=val[4])
@time syst, gi = gbilin(sys,g,atol=1.e-7); 
@time sysi, g1 = gbilin(syst,ginv,atol=1.e-7); 
@test iszero(sys-sysi,atol=1.e-7) && gi == ginv && g == g1

end #Ty
end # gbilin

@testset "c2d" begin

a = [-4 -2;1 0]; b = [2;0]; c = [0.5 1]; d = [0]; x0 = [1,2]; u0 = [1];
sysc = dss(a,b,c,d);
@time sysd, xd0 = c2d(sysc,1; x0 = [1,2], u0 = [1]); 
EAt = exp(a); 
@test dcgain(sysd) ≈ dcgain(sysc) && sort(gpole(sysd)) ≈ sort(exp.(gpole(sysc))) && 
      sysd.A ≈ EAt && sysd.B ≈ a\(EAt-I)*b
@time sysd_foh, xd0 = c2d(sysc,1,"foh"; x0, u0); 
@test dcgain(sysd_foh) ≈ dcgain(sysc) 
@time sysd_imp, xd0 = c2d(sysc,1,"impulse"; x0, u0); 
@test sort(gpole(sysd_imp)) ≈ sort(exp.(gpole(sysc)))
@time sysd1, xd1 = c2d(sysc,1,"tustin"; x0, u0, standard = true)
@time sysd2, xd2 = c2d(sysc,1,"tustin"; x0, u0, standard = false)
syst = gbilin(sysc,rtfbilin("Tustin"; Ts = 1)[1])[1] 
@test gnrank(sysd1 - sysd2, atol=1.e-7) == 0 && iszero(sysd1-syst, atol=1.e-7) &&
      xd1-xd2 ≈ (sysd2.E-I)*x0

@time sysd1, xd1 = c2d(sysc,1,"tustin"; prewarp_freq=1, x0, u0, standard = true)
@time sysd2, xd2 = c2d(sysc,1,"tustin"; prewarp_freq=1, x0, u0, standard = false)
syst = gbilin(sysc,rtfbilin("Tustin"; Ts = 1, prewarp_freq=1)[1])[1] 
@test gnrank(sysd1 - sysd2, atol=1.e-7) == 0 && iszero(sysd1-syst, atol=1.e-7) && xd1-xd2 ≈ (sysd2.E-I)*x0 &&
      evalfr(sysc,fval=1) ≈ evalfr(sysd1,fval=1) && evalfr(sysc,fval=1) ≈ evalfr(sysd2,fval=1) 

@test_throws ErrorException c2d(sysc,1,"Euler")

a = [-4 -2;1 0]; b = [2;0]; c = [0.5 1]; d = [0]; e = [1 2; 3 0]; x0 = [1,2]; u0 = [1];  
sysc = dss(a,e,b,c,d);
@time (sysd,xd0) = c2d(sysc,1; x0, u0); 
a1, e1, b1, c1, d1 = dssdata(sysc);
EAt = exp(e1\a1); 
@test dcgain(sysd) ≈ dcgain(sysc) && sort(gpole(sysd)) ≈ sort(exp.(gpole(sysc))) && 
      sysd.A ≈ EAt && sysd.B ≈ (e1\a1)\(EAt-I)*(e1\b1)

a = [-4 -2;1 0]; b = [2;0]; c = [0.5 1]; d = [0]; e = [1 2; 0 0]; x0 = [1,2]; u0 = [1];  
sysc = dss(a,e,b,c,d);
@time (sysd,xd0) = c2d(sysc,1; x0, u0); 
a1, e1, b1, c1, d1 = dssdata(gir(sysc,noseig=true));
EAt = exp(e1\a1); 
@test dcgain(sysd) ≈ dcgain(sysc) && sort(gpole(sysd)) ≈ sort(exp.(gpole(sysc))) && 
      sysd.A ≈ EAt && sysd.B ≈ (e1\a1)\(EAt-I)*(e1\b1)
@time sysd_foh, xd0 = c2d(sysc,1,"foh"; x0, u0); 
@test dcgain(sysd_foh) ≈ dcgain(sysc) 
@time sysd_imp, xd0 = c2d(sysc,1,"impulse"; x0, u0); 
@test sort(gpole(sysd_imp)) ≈ sort(exp.(gpole(sysc)))

@time sysd1, xd0 = c2d(sysc,1,"tustin"; x0, u0, standard = true)
@time sysd2, xd0 = c2d(sysc,1,"tustin"; x0, u0, standard = false)
syst = gbilin(sysc,rtfbilin("Tustin"; Ts = 1)[1])[1] 
@test gnrank(sysd1 - sysd2, atol=1.e-7) == 0 && iszero(sysd1-syst, atol=1.e-7)

@time sysd1, xd0 = c2d(sysc,1,"tustin"; prewarp_freq=1, x0, u0, standard = true)
@time sysd2, xd0 = c2d(sysc,1,"tustin"; prewarp_freq=1, x0, u0, standard = false)
syst = gbilin(sysc,rtfbilin("Tustin"; Ts = 1, prewarp_freq=1)[1])[1] 
@test gnrank(sysd1 - sysd2, atol=1.e-7) == 0 && iszero(sysd1-syst, atol=1.e-7) &&
      evalfr(sysc,fval=1) ≈ evalfr(sysd1,fval=1) && evalfr(sysc,fval=1) ≈ evalfr(sysd2,fval=1) 

end # c2d

@testset "dss2pm & dss2rm" begin

# Example 3: P = [λ^2 λ; λ 1] DeTeran, Dopico, Mackey, ELA 2009
λ = Polynomial([0,1],:s)
P = [λ^2 λ; λ 1]
@test all(P .≈ dss2pm(dss(P)))    
@test all(P .≈ dss2rm(dss(P)))  

z = rtf('z')
@test_throws ErrorException dss2pm(dss(z+1/z))

s = Polynomial([0, 1],:s);
R = rtf.([s^2+3*s+3 1; -1 2*s^2+7*s+4] ./ [(s+1)^2 s+2; (s+1)^3 (s+1)*(s+2)]);
@test all(R .≈ dss2rm(dss(R))) 

end # dss2pm & dss2rm

end #module





