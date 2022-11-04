module Test_timeresp

using DescriptorSystems
using LinearAlgebra
using Polynomials
using Test

println("Test_timeresp")

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
@test norm(y-y1,Inf) < 1.e-6 

@time y2, tout2, x2 = timeresp(sysc, ones(11,2), tout, x0; state_history = true, interpolation = "foh", atol=1.e-7)
@test norm(y-y2,Inf) < 1.e-6 




sysd = rdss(3,2,2,disc=true,stable=true,id=ones(Int,2))
sysdr = gss2ss(sysd)[1]
u = rand(11,2);
x0 = rand(5); 

@time y, tout, x = timeresp(sysd, u, Int[], x0; state_history = true, atol=1.e-7)
sys, xt, Mx, Mu = dss2ss(sysd, x0; state_mapping = true)  
@time y1, tout1, x1 = timeresp(sys, u, Int[], xt; state_history = true, atol=1.e-7)
@test norm(y-y1,Inf) < 1.e-6 && [Mx Mu]*[x1';u'] ≈ x'

@time y2, tout2, x2 = timeresp(sys, u, Int[]; state_history = false, atol=1.e-7)
@time y3, tout3, x3 = timeresp(sysdr, u, Int[]; state_history = false, atol=1.e-7)
@test norm(y2-y3,Inf) < 1.e-6 

end # timeresp    

@testset "stepresp" begin

a = [-4 -2;1 0]; b = [2 1;0 1]; c = [0.5 1]; d = [0 1]; x0 = [1,2]; u0 = [1, 1]; Ts = 1;
sysc = dss(a,b,c,d);
@time sysd, xd0, = c2d(sysc, Ts; x0, u0); 
@time y, tout, x = stepresp(sysd, 10; x0 = xd0, ustep = u0, state_history = true);

@time y1, tout1, x1 = stepresp(sysc, 10; x0 = xd0, timesteps = 10, ustep = u0, state_history = true)
@test norm(y-y1,Inf) < 1.e-7 && norm(x-x1,Inf) < 1.e-7

p, m = size(sysc); n = length(x0)
ns = length(tout)
y2 = similar(Array{Float64,3},ns,p,m)
x2 = similar(Array{Float64,3},ns,n,m)
@time begin
   for j = 1:m
       y2[:,:,j], tt, x2[:,:,j] = timeresp(sysd[:,j], ones(ns), tout, x0; state_history=true)    
   end  
end
@test norm(y-y2,Inf) < 1.e-7 && norm(x-x2,Inf) < 1.e-7

y3 = similar(Array{Float64,3},ns,p,m)
x3 = similar(Array{Float64,3},ns,n,m)
@time begin
   for j = 1:m
       y3[:,:,j], tt, x3[:,:,j] = timeresp(sysc[:,j], ones(ns), tout, x0; state_history=true)    
   end  
end
@test norm(y-y3,Inf) < 1.e-7 && norm(x-x3,Inf) < 1.e-7

ed = rand(2,2);
sysdd = dss(ed*sysd.A,ed,ed*sysd.B,sysd.C,sysd.D,Ts=sysd.Ts)
@time y4, tout4, x4 = stepresp(sysdd, 10; x0 = xd0, timesteps = 10, ustep = u0, state_history = true)
@test norm(y-y4,Inf) < 1.e-7 && norm(x-x4,Inf) < 1.e-7

a = [-4 -2;1 0]; b = [2 1;0 1]; c = [0.5 1]; d = [0 1]; x0 = [1,2]; u0 = [1, 1]; Ts = 1;
e = [1 3; 1 1];
sysc = dss(a,b,c,d); 
@time y, tout, x = stepresp(sysc, 10; x0, timesteps = 10, ustep = u0, state_history = true);

sysc1 = dss(e*a,e,e*b,c,d);
@time y1, tout1, x1 = stepresp(sysc1, 10; x0, timesteps = 10, ustep = u0, state_history = true)
@test norm(y-y1,Inf) < 1.e-7 && norm(x-x1,Inf) < 1.e-7

@time sysd, xd0, Mx, Mu= c2d(sysc, Ts; x0, u0); 
@time y2, tout, x2 = stepresp(sysd, 10; x0 = xd0, ustep = u0, state_history = true);
@test norm(y-y2,Inf) < 1.e-7 && norm(x-x2,Inf) < 1.e-7

sysdd = dss(e*sysd.A,e,e*sysd.B,sysd.C,sysd.D,Ts = 1);
@time y3, tout3, x3 = stepresp(sysdd, 10; x0 = xd0, timesteps = 10, ustep = u0, state_history = true)
@test norm(y-y3,Inf) < 1.e-7 && norm(x-x3,Inf) < 1.e-7

@time sysd1, xd1, Mx, Mu = c2d(sysc1, Ts; state_mapping = true, x0, u0); 
@time y4, tout, x4 = stepresp(sysd1, 10; x0 = xd1, ustep = u0, state_history = true);
@test norm(y-y4,Inf) < 1.e-7 && maximum([norm(Mx*x4[:,:,i]'-x[:,:,i]') for i in 1:2]) < 1.e-7

a = [-4 -2;1 0]; b = [2 1;0 1]; c = [0.5 1]; d = [0 1]; x0 = [1,2]; u0 = [1, 1]; Ts = 1;
e = [1 2; 0 0]; 
sysc = dss(a,e,b,c,d);
@time sysd, xd0, Mx, Mu = c2d(sysc, Ts; x0, u0, state_mapping = true, atol=1.e-7); 

@time y, tout, x = stepresp(sysd, 10; x0 = xd0, ustep = u0, state_history = true);

@time y1, tout1, x1 = stepresp(sysc, 10; x0, timesteps = 10, ustep = u0, state_history = true)
@test norm(y-y1,Inf) < 1.e-7 

end # stepresp    

end # module