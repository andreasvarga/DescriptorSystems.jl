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

@testset "c2d - descriptor systems" begin

a = [-4 -2;1 0]; b = [2;0]; c = [0.5 1]; d = [0]; x0 = [1,2]; u0 = [1];
sysc = dss(a,b,c,d);
@time sysd, xd0, Mx, Mu = c2d(sysc,1; x0 = [1,2], u0 = [1], state_mapping = true); 
EAt = exp(a); 
@test dcgain(sysd) ≈ dcgain(sysc) && sort(gpole(sysd)) ≈ sort(exp.(gpole(sysc))) && 
      sysd.A ≈ EAt && sysd.B ≈ a\(EAt-I)*b && x0 ≈ Mx*xd0+Mu*u0
@time sysd_foh, xd0, Mx, Mu = c2d(sysc,1,"foh"; x0, u0, state_mapping = true); 
@test dcgain(sysd_foh) ≈ dcgain(sysc) && x0 ≈ Mx*xd0+Mu*u0
@time sysd_imp, xd0, Mx, Mu = c2d(sysc,1,"impulse"; x0, u0, state_mapping = true); 
@test sort(gpole(sysd_imp)) ≈ sort(exp.(gpole(sysc))) && x0 ≈ Mx*xd0+Mu*u0
@time sysd1, xd1, Mx1, Mu1 = c2d(sysc,1,"tustin"; x0, u0, standard = true, state_mapping = true)
@time sysd2, xd2, Mx2, Mu2 = c2d(sysc,1,"tustin"; x0, u0, standard = false, state_mapping = true)
syst = gbilin(sysc,rtfbilin("Tustin"; Ts = 1)[1])[1] 
@test iszero(sysd1 - sysd2, atol=1.e-7) && iszero(sysd1-syst, atol=1.e-7) &&
      xd1-xd2 ≈ (sysd2.E-I)*x0 && x0 ≈ [Mx1 Mu1]*[xd1;u0] && x0 ≈ [Mx2 Mu2]*[xd2;u0]

@time sysd1, xd1, Mx1, Mu1 = c2d(sysc,1,"tustin"; prewarp_freq=1, x0, u0, standard = true, state_mapping = true)
@time sysd2, xd2, Mx2, Mu2 = c2d(sysc,1,"tustin"; prewarp_freq=1, x0, u0, standard = false, state_mapping = true)
syst = gbilin(sysc,rtfbilin("Tustin"; Ts = 1, prewarp_freq=1)[1])[1] 
@test gnrank(sysd1 - sysd2, atol=1.e-7) == 0 && iszero(sysd1-syst, atol=1.e-7) && xd1-xd2 ≈ (sysd2.E-I)*x0 &&
      evalfr(sysc,fval=1) ≈ evalfr(sysd1,fval=1) && evalfr(sysc,fval=1) ≈ evalfr(sysd2,fval=1)  && 
      x0 ≈ [Mx1 Mu1]*[xd1;u0] && x0 ≈ [Mx2 Mu2]*[xd2;u0]


@test_throws ErrorException c2d(sysc,1,"Euler")

a = [-4 -2;1 0]; b = [2;0]; c = [0.5 1]; d = [0]; e = [1 2; 3 0]; x0 = [1,2]; u0 = [1];  
sysc = dss(a,e,b,c,d);
@time sysd, xd0, Mx, Mu = c2d(sysc,1; x0, u0, state_mapping = true); 
a1, e1, b1, c1, d1 = dssdata(sysc);
EAt = exp(e1\a1); 
@test dcgain(sysd) ≈ dcgain(sysc) && sort(gpole(sysd)) ≈ sort(exp.(gpole(sysc))) && 
      sysd.A ≈ EAt && sysd.B ≈ (e1\a1)\(EAt-I)*(e1\b1) && x0 ≈ Mx*xd0+Mu*u0

a = [-4 -2;1 0]; b = [2;0]; c = [0.5 1]; d = [0]; e = [1 2; 0 0]; x0 = [1,2]; u0 = [1];  
sysc = dss(a,e,b,c,d);
@time sysd, xd0, Mx, Mu = c2d(sysc,1; x0, u0, state_mapping = true, simple_infeigs = false); 
@test norm(Mx*xd0+Mu*u0 - x0) > 0.001

x0 = Mx*xd0+Mu*u0;
@time sysd, xd1, Mx, Mu = c2d(sysc,1; x0, u0, state_mapping = true, simple_infeigs = false); 
@test norm(Mx*xd1+Mu*u0 - x0) < 0.001

@time sysd, xd0, Mx, Mu = c2d(sysc,1; x0, u0, state_mapping = true, simple_infeigs = true); 
@test norm(Mx*xd0+Mu*u0 - x0) > 0.001

x0 = Mx*xd0+Mu*u0;
@time sysd, xd1, Mx, Mu = c2d(sysc,1; x0, u0, state_mapping = true, simple_infeigs = true); 
@test norm(Mx*xd1+Mu*u0 - x0) < 0.001

a = [-4 -2;1 0]; b = [2;1]; c = [0.5 1]; d = [0]; e = [1 2; 0 0]; x0 = [1,2]; u0 = [1];  
sysc = dss(a,e,b,c,d);
@time sysd, xd0, Mx, Mu = c2d(sysc,1; x0, u0, state_mapping = true); 
a1, e1, b1, c1, d1 = dssdata(gir(sysc,contr=false,obs=false,noseig=true));
EAt = exp(e1\a1); 
@test dcgain(sysd) ≈ dcgain(sysc) && sort(gpole(sysd)) ≈ sort(exp.(gpole(sysc))) && 
      sysd.A ≈ EAt && sysd.B ≈ (e1\a1)\(EAt-I)*(e1\b1) && x0 ≈ Mx*xd0+Mu*u0
@time sysd_foh, xd0, Mx, Mu = c2d(sysc,1,"foh"; x0, u0, state_mapping = true); 
@test dcgain(sysd_foh) ≈ dcgain(sysc) && x0 ≈ Mx*xd0+Mu*u0
@time sysd_imp, xd0, Mx, Mu = c2d(sysc,1,"impulse"; x0, u0, state_mapping = true); 
@test sort(gpole(sysd_imp)) ≈ sort(exp.(gpole(sysc))) && x0 ≈ Mx*xd0+Mu*u0

@time sysd1, xd1, Mx1, Mu1 = c2d(sysc,1,"tustin"; x0, u0, standard = true, state_mapping = true)
@time sysd2, xd2, Mx2, Mu2 = c2d(sysc,1,"tustin"; x0, u0, standard = false, state_mapping = true)
syst = gbilin(sysc,rtfbilin("Tustin"; Ts = 1)[1])[1] 
@test iszero(sysd1 - sysd2, atol=1.e-7) && iszero(sysd1-syst, atol=1.e-7) &&
      xd1-xd2 ≈ (sysd2.E-I)*x0 && x0 ≈ [Mx1 Mu1]*[xd1;u0] && x0 ≈ [Mx2 Mu2]*[xd2;u0]

@time sysd1, xd1, Mx1, Mu1 = c2d(sysc,1,"tustin"; prewarp_freq=1, x0, u0, standard = true, state_mapping = true)
@time sysd2, xd2,Mx2, Mu2 = c2d(sysc,1,"tustin"; prewarp_freq=1, x0, u0, standard = false, state_mapping = true)
syst = gbilin(sysc,rtfbilin("Tustin"; Ts = 1, prewarp_freq=1)[1])[1] 
@test iszero(sysd1 - sysd2, atol=1.e-7) && iszero(sysd1-syst, atol=1.e-7) &&
      evalfr(sysc,fval=1) ≈ evalfr(sysd1,fval=1) && evalfr(sysc,fval=1) ≈ evalfr(sysd2,fval=1) && 
      x0 ≈ [Mx1 Mu1]*[xd1;u0] && x0 ≈ [Mx2 Mu2]*[xd2;u0]

a = [-4 -2;1 0]; b = [2;0]; c = [0.5 1]; d = [0]; e = [0 0; 0 0]; x0 = [1,2]; u0 = [1];  
sysc = dss(a,e,b,c,d);
@time sysd, xd0, Mx, Mu = c2d(sysc,1; x0, u0, state_mapping = true, simple_infeigs = true); 
@test norm(Mx*xd0+Mu*u0 - x0) > 0.001
x0 = Mx*xd0+Mu*u0;
@time sysd, xd1, Mx, Mu = c2d(sysc,1; x0, u0, state_mapping = true, simple_infeigs = true); 
@test norm(Mx*xd1+Mu*u0 - x0) < 0.001


end # c2d

@testset "c2d - rational transfer functions" begin

s = rtf('s'); z = rtf('z',Ts=1);
rc = 1/s;

rd1 = 1/(z-1);
@time rd = c2d(rc, 1, "zoh"); 
@test rd1 ≈ rd

rd2 = 0.5*(z+1)/(z-1);
@time rd = c2d(rc, 1, "foh"); 
@test rd2 ≈ rd

rd3 = z/(z-1);
@time rd = c2d(rc, 1, "impulse"); 
@test rd3 ≈ rd

rd4 = 0.5*(z+1)/(z-1);
@time rd = c2d(rc, 1, "tustin"); 
@test rd4 ≈ rd

rd5 = 1.0001/(z-1);
@time rd = c2d(rc, 1, "matched"); 
@test norm((rd5-rd).num.coeffs,Inf) < 0.001


rc = 1/(s+1); 

z1 = exp(-1);
rd1 = (1-z1)/(z-z1);
@time rd = c2d(rc, 1, "zoh"); 
@test rd1 ≈ rd

z2 = 1-2*z1;
rd2 = (z1*z+z2)/(z-z1); 
@time rd = c2d(rc, 1, "foh"); 
@test rd2 ≈ rd

rd3 = z/(z-z1);
@time rd = c2d(rc, 1, "impulse"); 
@test rd3 ≈ rd

rd4 = 1/3*(z+1)/(z-1/3);
@time rd = c2d(rc, 1, "tustin"); 
@test rd4 ≈ rd

rd5 = (1-z1)/(z-z1);
@time rd = c2d(rc, 1, "matched"); 
@test rd5 ≈ rd

rc = s+1; 

rd4 = (3*z-1)/(z+1);
@time rd = c2d(rc, 1, "tustin"); 
@test rd4 ≈ rd

z1 = exp(-1);
k1 = 1/(1-z1); 
rd5 = k1*(z-z1);
@time rd = c2d(rc, 1, "matched"); 
@test rd5 ≈ rd

s = rtf('s'); 
Gc = [s^2 s/(s+1); 0 1/s]
@time Gd = c2d(Gc, 1, "matched");

end 

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

@testset "timeresp" begin

a = [-4 -2;1 0]; b = [2 1;0 1]; c = [0.5 1]; d = [0 1]; x0 = [1,2]; u0 = [1, 1]; Ts = 1;
sysc = dss(a,b,c,d);
@time sysd, xd0, = c2d(sysc, Ts; x0, u0); 
@time y, tout, x = timeresp(sysd, ones(11,2), Int[], xd0; state_history = true);

@time y1, tout1, x1 = timeresp(sysc, ones(11,2), tout, x0; state_history = true, interpolation = "zoh")
@test norm(y-y1,Inf) < 1.e-7 && norm(x-x1,Inf) < 1.e-7

@time y2, tout2, x2 = timeresp(sysc, ones(11,2), tout, x0; state_history = true, interpolation = "foh")
@test norm(y-y2,Inf) < 1.e-7 && norm(x-x2,Inf) < 1.e-7

ed = rand(2,2);
sysdd = dss(ed*sysd.A,ed,ed*sysd.B,sysd.C,sysd.D,Ts=sysd.Ts)
@time y3, tout3, x3 = timeresp(sysdd, ones(11,2), Int[], xd0; state_history = true)
@test norm(y-y3,Inf) < 1.e-7 && norm(x-x3,Inf) < 1.e-7

u = rand(11,2);
@time sysd, xd0, Mx, Mu = c2d(sysc, Ts, "foh"; x0, u0 = u[1,:], state_mapping = true); 

@time y, tout, x = timeresp(sysd, u, Int[], xd0; state_history = true)

@time y2, tout2, x2 = timeresp(sysc, u, tout, x0; state_history = true, interpolation = "foh")
@test norm(y-y2,Inf) < 1.e-7 && norm([Mx Mu]*[x';u']-x2',Inf) < 1.e-7

a = [-4 -2;1 0]; b = [2 1;0 1]; c = [0.5 1]; d = [0 1]; x0 = [1,2]; u0 = [1, 1]; Ts = 1;
e = [1 3; 1 1];
u = ones(11,2);
sysc = dss(a,b,c,d); t = 0:Ts:10;
@time y, tout, x = timeresp(sysc, u, t, x0; state_history = true)

sysc1 = dss(e*a,e,e*b,c,d);
@time y1, tout, x1 = timeresp(sysc1, u, t, x0; state_history = true, interpolation = "zoh")
@test norm(y-y1,Inf) < 1.e-7 && norm(x-x1,Inf) < 1.e-7

@time sysd, xd0, Mx, Mu = c2d(sysc, Ts; state_mapping = true, x0, u0); 
@time y2, tout, x2 = timeresp(sysd, u, t, xd0; state_history = true) # OK
@test norm(y-y2,Inf) < 1.e-7 && norm(x-x2,Inf) < 1.e-7

sysdd = dss(e*sysd.A,e,e*sysd.B,sysd.C,sysd.D,Ts = 1);
@time yd, tout, xd = timeresp(sysdd, u, t, xd0; state_history = true) # OK
@test norm(y-yd,Inf) < 1.e-7 && norm(x-xd,Inf) < 1.e-7

@time sysd1, xd1, Mx, Mu = c2d(sysc1, Ts; state_mapping = true, x0, u0); 
@time y3, tout3, x3 = timeresp(sysd1, u, t, xd1; state_history = true) # OK
@test norm(y-y3,Inf) < 1.e-7 && norm([Mx Mu]*[x3';u']-x',Inf) < 1.e-7
    
@time y4, tout, x4 = timeresp(sysc, u, t, x0; state_history = true, interpolation = "foh")
@test norm(y-y4,Inf) < 1.e-7 && norm(x-x4,Inf) < 1.e-7

@time y5, tout, x5 = timeresp(sysc1, u, t, x0; state_history = true, interpolation = "foh")
@test norm(y-y5,Inf) < 1.e-7 && norm(x-x5,Inf) < 1.e-7

@time sysd2, xd2, Mx, Mu = c2d(sysc, Ts, "foh"; state_mapping = true, x0, u0); 
@time y6, tout, x6 = timeresp(sysd2, u, t, xd2; state_history = true) # OK
@test norm(y-y6,Inf) < 1.e-7 && norm([Mx Mu]*[x6';u']-x',Inf) < 1.e-7

sysdd2 = dss(e*sysd2.A,e,e*sysd2.B,sysd2.C,sysd2.D,Ts = 1);
@time ydd2, tout, xdd2 = timeresp(sysdd2, u, t, xd2; state_history = true) # OK
@test norm(y-ydd2,Inf) < 1.e-7 && norm([Mx Mu]*[xdd2';u']-x',Inf) < 1.e-7

@time sysd3, xd3, Mx, Mu = c2d(sysc1, Ts, "foh"; state_mapping = true, x0, u0); 
@time y7, tout, x7 = timeresp(sysd3, u, t, xd3; state_history = true) # OK
@test norm(y-y7,Inf) < 1.e-7 && norm([Mx Mu]*[x7';u']-x',Inf) < 1.e-7
     
u = rand(11,2);
@time sysd, xd0, Mx, Mu = c2d(sysc, Ts, "foh"; x0, u0 = u[1,:], state_mapping = true); 

@time y, tout, x = timeresp(sysd, u, Int[], xd0; state_history = true)

@time y2, tout2, x2 = timeresp(sysc, u, tout, x0; state_history = true, interpolation = "foh")
@test norm(y-y2,Inf) < 1.e-7 && norm([Mx Mu]*[x';u']-x2',Inf) < 1.e-7

a = [-4 -2;1 0]; b = [2 1;0 1]; c = [0.5 1]; d = [0 1]; x0 = [1,2]; u0 = [1, 1]; Ts = 1;
sysc = dss(a,b,c,d); t = 0:Ts:10;
u = ones(11,2);
@time y, tout, x = timeresp(sysc, u, t, x0; state_history = true);
e = rand(Complex{Float64},2,2);
syscc = dss(e*a,e,e*b,c,d); t = 0:Ts:10;
@time sysdc, xd0, Mx, Mu = c2d(syscc, Ts; x0, u0, state_mapping = true); 
@time yc, tout, xc = timeresp(sysdc, u, t, xd0; state_history = true)
@test norm(y-yc,Inf) < 1.e-7 && norm([Mx Mu]*[xc';u']-x',Inf) < 1.e-7

@time yc1, tout, xc1 = timeresp(syscc, u, t, x0; state_history = true)
@test norm(y-yc1,Inf) < 1.e-7 && norm(x-xc1,Inf) < 1.e-7

@time yc2, tout1, xc2 = timeresp(syscc, u, t, x0; state_history = true, interpolation = "foh")
@test norm(y-yc2,Inf) < 1.e-7 && norm(x-xc2,Inf) < 1.e-7

      
u = exp.(im*rand(11,2));
@time y, tout, x = timeresp(sysc, u, t, x0; state_history = true, interpolation = "foh");

@time sysdc, xd0, Mx, Mu = c2d(syscc, Ts, "foh"; x0, u0 = u[1,:], state_mapping = true); 

@time yc1, tout, xc1 = timeresp(sysdc, u, t, xd0; state_history = true)
@test norm(y-yc1,Inf) < 1.e-7 && norm([Mx Mu]*[xc1';u']-x',Inf) < 1.e-7

@time yc2, tout2, xc2 = timeresp(syscc, u, tout, x0; state_history = true, interpolation = "foh")
@test norm(y-yc2,Inf) < 1.e-7 && norm(x-xc2,Inf) < 1.e-7

a = [-4 -2;1 0]; b = [2 1;0 1]; c = [0.5 1]; d = [0 1]; x0 = [1,2]; u0 = [1, 1]; Ts = 1;
e = [1 2; 0 0]; 
sysc = dss(a,e,b,c,d);
@time sysd, xd0, Mx, Mu = c2d(sysc, Ts; x0, u0, state_mapping = true, atol=1.e-7); 

@time y, tout, x = timeresp(sysd, ones(11,2), Int[],xd0; state_history = true, atol=1.e-7)

@time y1, tout1, x1 = timeresp(sysc, ones(11,2), tout, x0; state_history = true, interpolation = "zoh", atol=1.e-7)
@test norm(y-y1,Inf) < 1.e-7 

@time y2, tout2, x2 = timeresp(sysc, ones(11,2), tout, x0; state_history = true, interpolation = "foh", atol=1.e-7)
@test norm(y-y2,Inf) < 1.e-7 

sysd = rdss(3,2,2,disc=true,stable=true,id=ones(Int,2))
sysdr = gss2ss(sysd)[1]
u = rand(11,2);
x0 = rand(5); 

@time y, tout, x = timeresp(sysd, u, Int[], x0; state_history = true, atol=1.e-7)
sys, xt, Mx, Mu = dss2ss(sysd, x0; state_mapping = true)  
@time y1, tout1, x1 = timeresp(sys, u, Int[], xt; state_history = true, atol=1.e-7)
@test norm(y-y1,Inf) < 1.e-7 && [Mx Mu]*[x1';u'] ≈ x'

@time y2, tout2, x2 = timeresp(sys, u, Int[]; state_history = false, atol=1.e-7)
@time y3, tout3, x3 = timeresp(sysdr, u, Int[]; state_history = false, atol=1.e-7)
@test norm(y2-y3,Inf) < 1.e-7 

end # timeresp    

end #module





