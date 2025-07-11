module Test_analysis

using DescriptorSystems
using LinearAlgebra
using Polynomials
using Test
using SparseArrays
using JLD2

println("Test_analysis")
@testset "Frequency response" begin

w = collect(LinRange(.1,100,1000));
sys = rss(50,4,3);
@time H = freqresp(sys,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(sys,fval=w[i]); end end
@test norm(H,Inf) < 1.e-5

sys = rss(50,4,3,disc=true);
@time H = freqresp(sys,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(sys,fval=w[i]); end end
@test norm(H,Inf) < 1.e-6

sys = rdss(50,4,3);
@time H = freqresp(sys,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(sys,fval=w[i]); end end
@test norm(H,Inf) < 1.e-6

sys = rdss(50,4,3,disc=true);
@time H = freqresp(sys,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(sys,fval=w[i]); end end
@test norm(H,Inf) < 1.e-6

# sparse models
syss = dss(sprand(50,50,0.7),sprand(50,3,.4),sprand(4,50,0.5),sprand(4,3,0.1));
@time H = freqresp(syss,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(syss,fval=w[i]); end end
@test norm(H,Inf) < 1.e-5

syss = dss(sprand(50,50,0.7),sprand(50,3,.4),sprand(4,50,0.5),sprand(4,3,0.1),Ts=0.1);
@time H = freqresp(syss,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(syss,fval=w[i]); end end
@test norm(H,Inf) < 1.e-5

syss = dss(sprand(50,50,0.7),sprand(50,50,0.7), sprand(50,3,.4),sprand(4,50,0.5),sprand(4,3,0.1));
@time H = freqresp(syss,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(syss,fval=w[i]); end end
@test norm(H,Inf) < 1.e-5

syss = dss(sprand(50,50,0.7),sprand(50,50,0.7),sprand(50,3,.4),sprand(4,50,0.5),sprand(4,3,0.1),Ts=0.1);
@time H = freqresp(syss,w);
@time begin for i = 1:1000 H[:,:,i] -= evalfr(syss,fval=w[i]); end end
@test norm(H,Inf) < 1.e-5

end #freqresp


@testset "test_analysis" begin
@testset "zeros, rank, poles" begin

# test example for SLICOT subroutine AB08ND
A = [
   1.0   0.0   0.0   0.0   0.0   0.0
   0.0   1.0   0.0   0.0   0.0   0.0
   0.0   0.0   3.0   0.0   0.0   0.0
   0.0   0.0   0.0  -4.0   0.0   0.0
   0.0   0.0   0.0   0.0  -1.0   0.0
   0.0   0.0   0.0   0.0   0.0   3.0];
E = I;
B = [
   0.0  -1.0
  -1.0   0.0
   1.0  -1.0
   0.0   0.0
   0.0   1.0
  -1.0  -1.0];
C = [
   1.0   0.0   0.0   1.0   0.0   0.0
   0.0   1.0   0.0   1.0   0.0   1.0
   0.0   0.0   1.0   0.0   0.0   1.0];
D = [
   0.0   0.0
   0.0   0.0
   0.0   0.0]; 
sys = dss(A,E,B,C,D);

# zeros 
@time val  = gzero(sys)
@test sort(real(val)) ≈ [-1, 2, Inf,Inf] 

@time val  = gzero(sys[:,1])
@test sort(real(val)) ≈ [-1, 2, Inf] 

@time val, info  = gzeroinfo(sys)
@test sort(real(val)) ≈ [-1, 2, Inf,Inf]  && 
      info.nfz == 2 && info.niev == 4 && info.nisev == 2 && info.niz == 2 && 
      info.nfsz == 1 && info.nfsbz == 0 && info.nfuz == 1 && 
      info.nrank == 8 && info.miev == [2, 2] && info.miz == [1, 1] && 
      info.rki == Int64[] && info.lki == [2] && !info.regular && !info.stable

@time val, info  = gzeroinfo(dss(A,E,B,C,D,Ts=-1))
@test sort(real(val)) ≈ [-1, 2, Inf,Inf]  && 
      info.nfz == 2 && info.niev == 4 && info.nisev == 2 && info.niz == 2 && 
      info.nfsz == 0 && info.nfsbz == 1 && info.nfuz == 1 && 
      info.nrank == 8 && info.miev == [2, 2] && info.miz == [1, 1] && 
      info.rki == Int64[] && info.lki == [2] && !info.regular && !info.stable


# poles
@time val  = gpole(sys)
@test sort(real(val)) ≈ sort([1.0, 1.0, 3.0, -4.0, -1.0, 3.0])

@time val  = gzero(dss(A = sys.A, E = sys.E))
@test sort(real(val)) ≈ sort([1.0, 1.0, 3.0, -4.0, -1.0, 3.0])

@test !isstable(sys)
@test isstable(dss(A = sys.A-5I))

@time val, info  = gpoleinfo(sys)
@test sort(real(val)) ≈ sort([1.0, 1.0, 3.0, -4.0, -1.0, 3.0]) && 
      info.nfev == 6 && info.nfsev == 2 && info.nfuev == 4

# rank
@test gnrank(sys, fastrank = true) == 2 && gnrank(sys, fastrank = false) == 2

@test gnrank(sys[:,1], fastrank = true) == 1 && gnrank(sys[:,1], fastrank = false) == 1

# output decoupling zeros
@time val  = gzero(dss(A = sys.A,E = sys.E,C = sys.C))
@test val ≈ [-1] 

# input decoupling zeros
@time val  = gzero(dss(A = sys.A,E = sys.E,B = sys.B))
@test val ≈ [-4] 


# test example for SLICOT subroutine AB08ND
A = [
    1     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     1     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0
    0     0     0     0     0     0     0     0     1];
E = [
    0     0     0     0     0     0     0     0     0
    1     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];
B = [
    -1     0     0
     0     0     0
     0     0     0
     0    -1     0
     0     0     0
     0     0     0
     0     0    -1
     0     0     0
     0     0     0];
C = [
    0     1     1     0     3     4     0     0     2
    0     1     0     0     4     0     0     2     0
    0     0     1     0    -1     4     0    -2     2];
D = [
    1     2    -2
    0    -1    -2
    0     0     0]; 

sys = dss(A,E,B,C,D);

# zeros
@time val  = gzero(sys)
@test val ≈ [1, Inf,Inf]

@time val, info  = gzeroinfo(sys)
@test val ≈ [1, Inf,Inf]  && 
      info.nfz == 1 && info.niev == 7 && info.nisev == 1 && info.niz == 2 && 
      info.nfsz == 0 && info.nfsbz == 0 && info.nfuz == 1 && 
      info.nrank == 11 && info.miev == [1, 1, 1, 1, 3] && info.miz == [2] && 
      info.rki == [2] && info.lki == [1] && !info.regular && !info.stable

@time val, info  = gzeroinfo(dss(A,E,B,C,D,Ts=-1))
@test val ≈ [1, Inf,Inf]  && 
      info.nfz == 1 && info.niev == 7 && info.nisev == 1 && info.niz == 2 && 
      info.nfsz == 0 && info.nfsbz == 1 && info.nfuz == 0 && 
      info.nrank == 11 && info.miev == [1, 1, 1, 1, 3] && info.miz == [2] && 
      info.rki == [2] && info.lki == [1] && !info.regular && !info.stable


# poles
@time val  = gpole(sys,check_reg = true)
@test val ≈ [Inf, Inf, Inf, Inf, Inf, Inf]

@time val  = gzero(dss(A = sys.A, E = sys.E))
@test val ≈ [Inf, Inf, Inf, Inf, Inf, Inf] 

@test !isstable(sys)
@test isstable(dss(A = sys.E, E = sys.A, Ts=-1))
 
@time val, info  = gpoleinfo(sys)
@test val ≈ [Inf, Inf, Inf, Inf, Inf, Inf] && 
      info.nfev == 0 && info.nfsev == 0 && info.nfuev == 0 &&
      info.niev == 9 && info.nip == 6 && info.miev == [3, 3, 3] && info.mip == [2, 2, 2] &&
      info.regular && !info.proper && !info.stable 


@test gnrank(sys, atol = 1.e-7, fastrank = true) == 2 && gnrank(sys, atol = 1.e-7, fastrank = false) ==2 

@test gnrank(sys[:,1], fastrank = true) == 1 && gnrank(sys[:,1], fastrank = false) == 1

# output decoupling zeros
@time val  = gzero(dss(A = sys.A,E = sys.E,C = sys.C))
@test val ≈ [Inf, Inf, Inf, Inf] 

# input decoupling zeros
@time val  = gzero(dss(A = sys.A,E = sys.E,B = sys.B))
@test val ≈ Float64[] 

# stability tests
@test isstable(rdss(5,3,3, stable = true))
@test isstable(rdss(5,3,3, id = ones(Int,3), stable = true), atol = 1.e-7)

# infinite structure test
n = 5; m = 2; p = 3; 
sys = rdss(n,p,m,id=[ones(Int,3);2*ones(Int,2)]);
@time val, info  = gpoleinfo(sys,atol=1.e-7)
@test count(isinf.(val)) == 2 && 
      info.nfev == n && info.niev == 7 && info.nisev == 3 && info.nfsev + info.nfuev == n &&
      info.nip == 2 && info.miev == [1, 1, 1, 2, 2] && info.mip == [1, 1] &&
      info.regular && !info.proper && isregular(sys)


A = rand(n,n); E = rand(n,n); 
@time val, info  = gpoleinfo(dss(A=A,E=E))
@test count(isinf.(val)) == 0 && 
      info.nfev == n && info.niev == 0 && info.nisev == 0 && info.nfsev + info.nfuev == n &&
      info.nip == 0 && info.miev == [] && info.mip == [] &&
      info.regular && info.proper  

As, Es, = schur(A,E)
@time val, info  = gpoleinfo(dss(A=As,E=Es))
@test count(isinf.(val)) == 0 && 
      info.nfev == n && info.niev == 0 && info.nisev == 0 && info.nfsev + info.nfuev == n &&
      info.nip == 0 && info.miev == [] && info.mip == [] &&
      info.regular && info.proper  

@time val, info  = gpoleinfo(dss(A=A,E=triu(E)))
@test count(isinf.(val)) == 0 && 
      info.nfev == n && info.niev == 0 && info.nisev == 0 && info.nfsev + info.nfuev == n &&
      info.nip == 0 && info.miev == [] && info.mip == [] &&
      info.regular && info.proper  





end


@testset "gl2norm & gh2norm" begin

sys = rdss(0,0,0);
@time l2norm = gl2norm(sys)
@test l2norm == 0 

sys = rdss(0,2,2);
@time l2norm = gl2norm(sys)
@test l2norm == Inf 

sys = rdss(0,2,2,disc=true);
@time l2norm = gl2norm(sys)
@test l2norm == norm(sys.D)

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
sys = dss(a,b,c,d);
@time l2norm = gh2norm(sys)
@test l2norm ≈ 1.419727086450068e+01 

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
e = rand(2,2);
sys = dss(e*a,e,e*b,c,d);
@time l2norm = gh2norm(sys)
@test l2norm ≈ 1.419727086450068e+01 


a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
sys = dss(a,b,c,d);
@time l2norm = gl2norm(sys)
@test l2norm ≈ 1.419727086450068e+01 

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
sys = dss(a,b,c,d);
@time l2norm = opnorm(sys,2)
@test l2norm ≈ 1.419727086450068e+01 


a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
e = rand(2,2)
sys = dss(e*a,e,e*b,c,d);
@time l2norm = gl2norm(sys)
@test l2norm ≈ 1.419727086450068e+01 

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
sys = dss(a,b,c,d);
@time l2norm = gl2norm(sys')
@test l2norm ≈ 1.419727086450068e+01 

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
e = rand(2,2)
sys = dss(e*a,e,e*b,c,d);
@time l2norm = gl2norm(sys')
@test l2norm ≈ 1.419727086450068e+01 


# discrete standard & descriptor
a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
sys = dss(a,b,c,d,Ts = 1);
@time l2norm = gh2norm(sys)
@test l2norm ≈ 3.438689619923066e+01

a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
e = rand(2,2)
sys = dss(e*a,e,e*b,c,d,Ts = 1);
@time l2norm = gh2norm(sys)
@test l2norm ≈ 3.438689619923066e+01

a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
sys = dss(a,b,c,d,Ts = 1);
@time l2norm = gl2norm(sys)
@test l2norm ≈ 3.438689619923066e+01

a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
e = rand(2,2)
sys = dss(e*a,e,e*b,c,d,Ts = 1);
@time l2norm = gl2norm(sys)
@test l2norm ≈ 3.438689619923066e+01

a = [-1 2;0 0]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
sys = dss(a,b,c,d);
@time l2norm = gl2norm(sys)
@test l2norm ≈ Inf

a = [-1 2;0 0]; e = [1 0; 0 2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
sys = dss(a,e,b,c,d);
@time l2norm = gl2norm(sys)
@test l2norm ≈ Inf



a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
sys = dss(a,b,c,d,Ts = 1);
@time l2norm = gl2norm(sys')
@test l2norm ≈ 3.438689619923066e+01

a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
e = rand(2,2)
sys = dss(e*a,e,e*b,c,d,Ts = 1);
@time l2norm = gl2norm(sys')
@test l2norm ≈ 3.438689619923066e+01

z = Polynomial([0, 1],'z')
g = [z^2+z+1 4*z^2+3*z+2 2*z^2-2;
    z 4*z-1 2*z-2;
    z^2 4*z^2-z 2*z^2-2*z];

sys = dss(g,minimal = true, atol = 1.e-7,Ts = 1)
@time l2norm = gl2norm(sys, atol = 1.e-7)
@test l2norm ≈ gl2norm(sys', atol = 1.e-7)

n = 5; m = 3; p = 2;

Ty = Float64; fast = true; 
for Ty in (Float64, Complex{Float64})

for fast in (true, false)

sys = rss(n,p,m,T = Ty, stable = true); 
sys.D[:,:] = zeros(Ty,p,m);
@test gh2norm(sys) ≈ gl2norm(sys')

sys = rss(n,p,m,T = Ty, stable = true); 
@test gh2norm(sys) ≈ gl2norm(sys')

sys = rss(n,p,m,T = Ty, stable = true, disc = true); 
@test gh2norm(sys) ≈ gl2norm(sys')


sys = rdss(n,p,m,T = Ty, stable = true); 
sys.D[:,:] = zeros(Ty,p,m);
@test gh2norm(sys) ≈ gl2norm(sys')

sys = rdss(n,p,m,T = Ty, stable = true); 
@test gh2norm(sys) ≈ gl2norm(sys')

sys = rdss(n,p,m,T = Ty, stable = true, disc = true); 
@test gh2norm(sys) ≈ gl2norm(sys')

sys = rdss(n,p,m, T = Ty, stable = true, id=ones(Int,3)); 
@test gh2norm(sys) ≈ gl2norm(sys')

end # fast
end # Ty

end # gl2norm

@testset "ghanorm" begin

sys = rdss(0,0,0);
@time hanorm, hsv = ghanorm(sys)
@test hanorm == 0 && hsv == Float64[]

n = 5; m = 3; p = 2;

Ty = Float64; fast = true; 
for Ty in (Float64, Complex{Float64})

for fast in (true, false)

sys = rss(n,p,m,stable = true); 

try
  @time hanorm, hsv = ghanorm(sys,fast = fast)
  @test true
catch
  @test false
end

try
    @time hanorm, hsv = ghanorm(sys',fast = fast)
    @test false
catch
    @test true
end

sys = rss(n,p,m,T = Ty, stable = true); 

@time hanorm, hsv = ghanorm(sys-sys,fast = fast)
@test hanorm < 1.e-7

@time hanorm, hsv = ghanorm(sys+sys,fast = fast)
@test norm(hsv[n+1:end]) < 1.e-7

@time hanorm, hsv = ghanorm([sys sys],fast = fast)
@test norm(hsv[n+1:end]) < 1.e-7

sys = rss(n,p,m,T = Ty, disc = true, stable = true); 
try
    @time hanorm, hsv = ghanorm(sys,fast = fast)
    @test true
catch
    @test false
end

try
    @time hanorm, hsv = ghanorm(sys',fast = fast)
    @test false
catch
    @test true
end
@time hanorm, hsv = ghanorm(sys-sys,fast = fast)
@test hanorm < 1.e-7

@time hanorm, hsv = ghanorm(sys+sys,fast = fast)
@test norm(hsv[n+1:end]) < 1.e-7

@time hanorm, hsv = ghanorm([sys sys],fast = fast)
@test norm(hsv[n+1:end]) < 1.e-7
  

sys = rdss(n,p,m,T = Ty, stable = true,id=ones(Int,3)); 
try
  @time hanorm, hsv = ghanorm(sys,atol=1.e-7)
  @test true
catch
  @test false
end

try
    @time hanorm, hsv = ghanorm(sys',fast = fast)
    @test false
catch
    @test true
end

sys = rdss(n,p,m,T = Ty, stable = true); 
@time hanorm, hsv = ghanorm(sys-sys,fast = fast)
@test hanorm < 1.e-7

@time hanorm, hsv = ghanorm(sys+sys,fast = fast)
@test norm(hsv[n+1:end]) < 1.e-7

@time hanorm, hsv = ghanorm([sys sys],fast = fast)
@test norm(hsv[n+1:end]) < 1.e-7

sys = rdss(n,p,m,T = Ty, disc = true,stable = true,id=ones(Int,3)); 
try
    @time hanorm, hsv = ghanorm(sys,fast = fast)
    @test true
catch
    @test false
end

try
    @time hanorm, hsv = ghanorm(sys',fast = fast)
    @test false
catch
    @test true
end

sys = rdss(n,p,m,T = Ty, disc = true,stable = true); 
@time hanorm, hsv = ghanorm(sys-sys,fast = fast)
@test hanorm < 1.e-7

@time hanorm, hsv = ghanorm(sys+sys,fast = fast)
@test norm(hsv[n+1:end]) < 1.e-7

@time hanorm, hsv = ghanorm([sys sys],fast = fast)
@test norm(hsv[n+1:end]) < 1.e-7

end # fast
end # Ty

end # ghanorm

@testset "glinfnorm & ghinfnorm" begin

sys = rdss(0,0,0);
@time linfnorm, fpeak = glinfnorm(sys)
@test linfnorm == 0 && fpeak == 0

sys = rdss(0,2,2);
@time linfnorm, fpeak = glinfnorm(sys)
@test linfnorm == opnorm(sys.D) && fpeak == 0

sys = rdss(0,2,2,disc=true);
@time linfnorm, fpeak = glinfnorm(sys)
@test linfnorm == opnorm(sys.D) && fpeak == 0

a = [0 1;-1 0]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
sys = dss(a,b,c,d);
@time hinfnorm, fpeak = ghinfnorm(sys)
@test isinf(hinfnorm) && isnan(fpeak)

a = [0 1;-1 0]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
sys = dss(a,b,c,d);
@time linfnorm, fpeak = glinfnorm(sys)
@test isinf(linfnorm) && fpeak ≈ 1

a = [0 1;-1 0]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3); 
sys = dss(a,b,c,d, Ts = 2);
@time linfnorm, fpeak = glinfnorm(sys)
@test isinf(linfnorm) && fpeak ≈ pi/4


a = [-1 2;0 -2]; e = [0 1;0 0]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
sys = dss(a,e,b,c,d);
@time linfnorm, fpeak = glinfnorm(sys)
@test isinf(linfnorm) && isinf(fpeak)

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
sys = dss(a,b,c,d);
@time hinfnorm, fpeak = ghinfnorm(sys,rtolinf = 0.0000001)
@test opnorm(evalfr(sys,im*fpeak)) ≈ hinfnorm &&
      round(hinfnorm, digits=6) ≈ round(1.163398400218353e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.695926597795647,digits=3)

sys = dss(a+im*a,b+im*b,c+im*c,d+im*d);
@time hinfnorm, fpeak = glinfnorm(sys,rtolinf = 0.0000001)
@test opnorm(evalfr(sys,im*fpeak)) ≈ hinfnorm &&
      abs(hinfnorm - 56.237554184492573) < 1.e-6 &&  # compare with Matlab results
      abs(fpeak+3.952242830041314) < 0.001

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
e = rand(2,2);
sys = dss(e*a,e,e*b,c,d);
@time hinfnorm, fpeak = ghinfnorm(sys,rtolinf = 0.0000001)
@test opnorm(evalfr(sys,im*fpeak)) ≈ hinfnorm &&
      round(hinfnorm, digits=6) ≈ round(1.163398400218353e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.695926597795647,digits=3)

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
e = rand(Complex{Float64},2,2);
sys = dss(e*(a+im*a),e,e*(b+im*b),c+im*c,d+im*d);
@time hinfnorm, fpeak = glinfnorm(sys,rtolinf = 0.0000001)
@test opnorm(evalfr(sys,im*fpeak)) ≈ hinfnorm &&
      abs(hinfnorm - 56.237554184492573) < 1.e-6 &&  # compare with Matlab results
      abs(fpeak+3.952242830041314) < 0.001

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
sys = dss(a,b,c,d);
@time linfnorm, fpeak = glinfnorm(sys,rtolinf = 0.0000001)
@test opnorm(evalfr(sys,im*fpeak)) ≈ linfnorm &&
      round(linfnorm, digits=6) ≈ round(1.163398400218353e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.695926597795647,digits=3) 

@time linfnorm, fpeak = opnorm(sys,rtolinf = 0.0000001)
@test opnorm(evalfr(sys,im*fpeak)) ≈ linfnorm &&
      round(linfnorm, digits=6) ≈ round(1.163398400218353e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.695926597795647,digits=3) 

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
e = rand(2,2);
sys = dss(e*a,e,e*b,c,d);
@time linfnorm, fpeak = glinfnorm(sys,rtolinf = 0.0000001)
@test opnorm(evalfr(sys,im*fpeak)) ≈ linfnorm &&
      round(linfnorm, digits=6) ≈ round(1.163398400218353e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.695926597795647,digits=3)

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
sys = dss(a,b,c,d);
@time linfnorm, fpeak = glinfnorm(sys',rtolinf = 0.0000001)
@test opnorm(evalfr(sys,im*fpeak)) ≈ linfnorm &&
      round(linfnorm, digits=6) ≈ round(1.163398400218353e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.695926597795647,digits=3)

a = [-1 2;-3 -2]; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = zeros(3,3);
e = rand(2,2)
sys = dss(e*a,e,e*b,c,d);
@time linfnorm, fpeak = glinfnorm(sys',rtolinf = 0.0000001)
@test opnorm(evalfr(sys,im*fpeak)) ≈ linfnorm &&
      round(linfnorm, digits=6) ≈ round(1.163398400218353e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.695926597795647,digits=3)


# discrete standard & descriptor
a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
sys = dss(a,b,c,d,Ts = 1);
@time hinfnorm, fpeak = ghinfnorm(sys,rtolinf = 0.0000001) 
@test opnorm(evalfr(sys,exp(im*fpeak))) ≈ hinfnorm &&
      round(hinfnorm, digits=6) ≈ round(4.133509781281479e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.753904640692288,digits=3)

a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
e = rand(2,2)
sys = dss(e*a,e,e*b,c,d,Ts = 1);
@time hinfnorm, fpeak = ghinfnorm(sys,rtolinf = 0.0000001)
@test opnorm(evalfr(sys,exp(im*fpeak))) ≈ hinfnorm &&
      round(hinfnorm, digits=6) ≈ round(4.133509781281479e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.753904640692288,digits=3)

a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
sys = dss(a,b,c,d,Ts = 1);
@time linfnorm, fpeak = glinfnorm(sys,rtolinf = 0.0000001)
@test opnorm(evalfr(sys,exp(im*fpeak))) ≈ linfnorm &&
      round(linfnorm, digits=6) ≈ round(4.133509781281479e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.753904640692288,digits=3)

a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
e = rand(2,2)
sys = dss(e*a,e,e*b,c,d,Ts = 1);
@time linfnorm, fpeak = glinfnorm(sys,rtolinf = 0.0000001)
@test opnorm(evalfr(sys,exp(im*fpeak))) ≈ linfnorm &&
      round(linfnorm, digits=6) ≈ round(4.133509781281479e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.753904640692288,digits=3)


a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
sys = dss(a,b,c,d,Ts = 1);
@time linfnorm, fpeak = glinfnorm(sys',rtolinf = 0.0000001)
@test opnorm(evalfr(sys,exp(im*fpeak))) ≈ linfnorm &&
      round(linfnorm, digits=6) ≈ round(4.133509781281479e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.753904640692288,digits=3)

a = [-1 2;-3 -2]/10; b = [2 3 4; 1 2 3]; c = [1 4; 2 2; 1 3]; d = ones(3,3);
e = rand(2,2)
sys = dss(e*a,e,e*b,c,d,Ts = 1);
@time linfnorm, fpeak = glinfnorm(sys',rtolinf = 0.0000001)
@test opnorm(evalfr(sys,exp(im*fpeak))) ≈ linfnorm &&
      round(linfnorm, digits=6) ≈ round(4.133509781281479e+01,digits=6) && 
      round(fpeak,digits = 3) ≈ round(2.753904640692288,digits=3)

z = Polynomial([0, 1],'z')
g = [z^2+z+1 4*z^2+3*z+2 2*z^2-2;
    z 4*z-1 2*z-2;
    z^2 4*z^2-z 2*z^2-2*z];

sys = dss(g,minimal = true, atol = 1.e-7,Ts = 1)
@time linfnorm, fpeak = glinfnorm(sys, atol = 1.e-7)
@test glinfnorm(sys, atol = 1.e-7)[1] ≈ glinfnorm(sys', atol = 1.e-7)[1] &&
      opnorm(evalfr(sys,exp(im*fpeak))) ≈ linfnorm &&
      round(linfnorm, digits=6) ≈ round(1.048808848170152e+01,digits=6) && 
      fpeak == 0 


n = 50; m = 30; p = 20;

# continuous-time standard, real
Ty = Float64; fast = true; 
sys = rss(n,p,m,T = Ty,disc=false); 
@time linfnorm, fpeak = glinfnorm(sys, fast = fast, rtolinf = 0.0000000001)
syst = (linfnorm)^2*I-sys'*sys; ev=gpole(inv(syst),atol=1.e-7); 
@test sort(abs.(real(ev)))[1] < 1.e-5

# discrete-time standard, real
Ty = Float64; fast = true; 
sys = rss(n,p,m,T = Ty,disc=true); 
@time linfnorm, fpeak = glinfnorm(sys,fast = fast, rtolinf = 0.0000000001);
syst = (linfnorm)^2*I-sys'*sys; ev=gpole(inv(syst),atol=1.e-7); 
@test sort(abs.(abs.(ev).-1))[1] < 1.e-5

# continuous-time descriptor, real
Ty = Float64; fast = true; 
sys = rdss(n,p,m,T = Ty,disc=false); 
@time linfnorm, fpeak = glinfnorm(sys,fast = fast, rtolinf = 0.0000000001);
syst = (linfnorm)^2*I-sys'*sys; ev=gpole(inv(syst),atol=1.e-7); 
@test sort(abs.(real(ev)))[1] < 1.e-5

# discrete-time descriptor, real
Ty = Float64; fast = true; 
sys = rdss(n,p,m,T = Ty,disc=true); 
@time linfnorm, fpeak = glinfnorm(sys,fast = fast, rtolinf = 0.0000000001);
syst = (linfnorm)^2*I-sys'*sys; ev=gpole(inv(syst),atol=1.e-7); 
@test sort(abs.(abs.(ev).-1))[1] < 1.e-5

# continuous-time standard, complex
Ty = Complex{Float64}; fast = true; 
sys = rss(n,p,m,T = Ty,disc=false); 
@time linfnorm, fpeak = glinfnorm(sys,fast = fast, rtolinf = 0.0000000001);
syst = (linfnorm)^2*I-sys'*sys; ev=gpole(inv(syst),atol=1.e-7); 
@test sort(abs.(real(ev)))[1] < 1.e-5 && opnorm(evalfr(sys,im*fpeak)) ≈ linfnorm

# discrete-time standard, complex
Ty = Complex{Float64}; fast = true; 
sys = rss(n,p,m,T = Ty,disc=true); 
@time linfnorm, fpeak = glinfnorm(sys,fast = fast, rtolinf = 0.0000000001);
syst = (linfnorm)^2*I-sys'*sys; ev=gpole(inv(syst),atol=1.e-7); 
@test sort(abs.(abs.(ev).-1))[1] < 1.e-5 && opnorm(evalfr(sys,exp(im*fpeak))) ≈ linfnorm

# continuous-time descriptor, complex
Ty = Complex{Float64}; fast = true; 
sys = rdss(n,p,m,T = Ty,disc=false); 
@time linfnorm, fpeak = glinfnorm(sys,fast = fast, rtolinf = 0.0000000001);
syst = (linfnorm)^2*I-sys'*sys; ev=gpole(inv(syst),atol=1.e-7); 
@test sort(abs.(real(ev)))[1] < 1.e-5 && opnorm(evalfr(sys,im*fpeak)) ≈ linfnorm

# discrete-time descriptor, complex
Ty = Complex{Float64}; fast = true; 
sys = rdss(n,p,m,T = Ty,disc=true); 
@time linfnorm, fpeak = glinfnorm(sys,fast = fast, rtolinf = 0.0000000001);
syst = (linfnorm)^2*I-sys'*sys; ev=gpole(inv(syst),atol=1.e-7); 
@test sort(abs.(abs.(ev).-1))[1] < 1.e-5 && opnorm(evalfr(sys,exp(im*fpeak))) ≈ linfnorm

n = 5; m = 3; p = 2;

Ty = Float64; fast = true; 
Ty = Complex{Float64}; fast = true; 
rtolinf = 0.000001
for Ty in (Float64, Complex{Float64})

for fast in (true, false)

sys = rss(n,p,m,T = Ty, stable = true); 
sys.D[:,:] = zeros(Ty,p,m);
@time hinf, hfpeak = ghinfnorm(sys,fast = fast,rtolinf = rtolinf)
@time linf, lfpeak = glinfnorm(sys',fast = fast,rtolinf = rtolinf)
@test abs(hinf-linf)/hinf < 2*rtolinf && (hfpeak ≈ lfpeak || abs(hfpeak-lfpeak) <= 0.01*abs(lfpeak))

sys = rss(n,p,m,T = Ty, stable = true); 
@time hinf, hfpeak = ghinfnorm(sys,fast = fast,rtolinf = rtolinf)
@time linf, lfpeak = glinfnorm(sys',fast = fast,rtolinf = rtolinf)
@test abs(hinf-linf)/hinf < 2*rtolinf && (hfpeak ≈ lfpeak || abs(hfpeak-lfpeak) <= 0.01*abs(lfpeak))

sys = rss(n,p,m,stable = true, disc = true); 
@time hinf, hfpeak = ghinfnorm(sys,fast = fast,rtolinf = rtolinf)
@time linf, lfpeak = glinfnorm(sys',fast = fast,rtolinf = rtolinf)
@test abs(hinf-linf)/hinf < 2*rtolinf && (hfpeak ≈ lfpeak || abs(hfpeak-lfpeak) <= 0.01*abs(lfpeak))

sys = rdss(n,p,m,T = Ty, stable = true); 
sys.D[:,:] = zeros(Ty,p,m);
@time hinf, hfpeak = ghinfnorm(sys,fast = fast,rtolinf = rtolinf)
@time linf, lfpeak = glinfnorm(sys',fast = fast,rtolinf = rtolinf)
@test abs(hinf-linf)/hinf < 2*rtolinf && (hfpeak ≈ lfpeak || abs(hfpeak-lfpeak) <= 0.01*abs(lfpeak))

sys = rdss(n,p,m,T = Ty, stable = true); 
@time hinf, hfpeak = ghinfnorm(sys,fast = fast,rtolinf = rtolinf)
@time linf, lfpeak = glinfnorm(sys',fast = fast,rtolinf = rtolinf)
@test abs(hinf-linf)/hinf < 2*rtolinf && (hfpeak ≈ lfpeak || abs(hfpeak-lfpeak) <= 0.01*abs(lfpeak))

sys = rdss(n,p,m,T = Ty, stable = true, disc = true); 
@time hinf, hfpeak = ghinfnorm(sys,fast = fast,rtolinf = rtolinf)
@time linf, lfpeak = glinfnorm(sys',fast = fast,rtolinf = rtolinf)
@test abs(hinf-linf)/hinf < 2*rtolinf && (hfpeak ≈ lfpeak || abs(hfpeak-lfpeak) <= 0.01*abs(lfpeak))

sys = rdss(n,p,m,T = Ty, stable = true, id=ones(Int,3)); 
@time hinf, hfpeak = ghinfnorm(sys,fast = fast,rtolinf = rtolinf)
@time linf, lfpeak = glinfnorm(sys',fast = fast,rtolinf = rtolinf)
@test abs(hinf-linf)/hinf < 2*rtolinf && (hfpeak ≈ lfpeak || abs(hfpeak-lfpeak) <= 0.01*abs(lfpeak))

end # fast
end # Ty
end # glinfnorm


@testset "gnugap" begin


## Examples from the Vinnicombe's book, p. 115
s = rtf('s'); N = 3;
sys = (dss(100/(2*s+1)), dss(100/(2*s-1)), dss(100/(s+1)^2))
distgap = zeros(N); 
for j = 1:N
    distgap[j], = gnugap(sys[1],sys[j],atol=1.e-7);
end
@test norm(distgap-[0,0.02,0.8988]) <0.001

## Examples from the Vinnicombe's book, p. 125
s = rtf('s'); 
sys1 = dss(1/(s+0.2));                               # P1
sys2 = dss((s+0.1)^2/(s-0.2)/(s^2+0.1*s+0.01));      # P2
nugap, fpeak = gnugap(sys2,sys1,atol=1.e-7)
@test abs(nugap-0.403) < 0.0001 && abs(fpeak-0.32) < 0.01


## Examples from the Vinnicombe's book, p. 160 with wno ~= 0
s = rtf('s'); N = 3;
sys = (dss(0.3*(s-1)/(s^2+1)), dss(0.3*(s-1)/(s^2+2)), dss(0.3*(s-1)*(s-2)/(s^2+2)/(s-sqrt(2))))
distgap = zeros(N); 
for j = 1:N
    distgap[j], = gnugap(sys[1],sys[j],atol=1.e-7);
end

@test norm(distgap-[0,1,1]) < 1.e-6

## Examples from the Vinnicombe's book, p. 201
s = rtf('s');
sys1 = dss(1/s^2);    # P1
sys2 = dss(1/s);      # P2
nugap, fpeak = gnugap(sys2,sys1,atol=1.e-7)
@test nugap ≈ 1/sqrt(2) && fpeak ≈ 1


## Examples from the Vinnicombe's book, p. 202  nugap = 1/sqt(2) for epsi>0
s = rtf('s'); epsi = 0.01;
sys1 = dss([1/s; 1/(s+epsi)]);    # P1
sys2 = dss([1/s; 1/s]);           # P2
nugap, = gnugap(sys2,sys1,atol=1.e-7)
@test nugap ≈ 1/sqrt(2) 

## Examples from the Vinnicombe's book, p. 202  modified 
#  realization of sys2 is not stabilizable
s = rtf('s'); epsi = 0.01;
sys1 = dss([1/s^2; 1/(s+epsi)]);    # P1
sys2 = dss([1/s^2; 1/s]);           # P2
nugap, = gnugap(sys2,sys1,atol=1.e-7)
@test nugap < epsi

## Examples from the Vinnicombe's book, p. 202  modified 
#  realization of sys2 is not stabilizable
s = rtf('s'); epsi = 0.01;
sys1 = dss([s^2+epsi; (s+epsi)]);    # P1
sys2 = dss([s^2; s]);           # P2
nugap, = gnugap(sys2,sys1,atol=1.e-7)
@test nugap < sqrt(2)*epsi

## 
z = rtf('z');
sys1 = dss(1/z); sys2 = dss(0.5/z);
nugap, = gnugap(sys1,sys2,atol=1.e-6)
@test nugap ≈ 0.316227766016838

## 
z = rtf('z');
sys1 = dss(1/z); sys2 = dss(0.5/z^2);
nugap, = gnugap(sys1,sys2,atol=1.e-6)
@test nugap ≈ 0.948683298050514

## ν-gap = 1
s = rtf('s');
sys1 = dss((s+1)/(s+2)); sys2 = dss(-1);
nugap, = gnugap(sys1,sys2,atol=1.e-7)
@test nugap == 1


## gapmetric produces the wrong result nugap = 0
sys1 = dss(1); sys2 = dss(-1);
nugap, = gnugap(sys1,sys2,atol=1.e-5)
@test nugap == 1

## ν-gap = 1 
sys1 = rss(3,2,2); sys1.D[:,:] = eye(2); sys2 = -inv(sys1)';
nugap, fpeak = gnugap(sys1,sys2,atol=1.e-6)
@test nugap == 1


n = 5; m = 3; p = 2;

Ty = Float64; fast = true; 
Ty = Complex{Float64}; fast = true; 
rtolinf = 0.000001
for Ty in (Float64, Complex{Float64})

for fast in (true, false)

## nugap = 0
sys = rss(n,p,m, T = Ty); 
@time nugap, fpeak = gnugap(sys,sys,fast = fast, atol = 1.e-7)
@test nugap < 1.e-10

sys = rss(n,p,m,T = Ty, disc=true); 
@time nugap, fpeak = gnugap(sys,sys, fast = fast, atol = 1.e-7)
@test nugap < 1.e-10

sys = rdss(n,p,m,T = Ty); 
@time nugap, fpeak = gnugap(sys,sys,fast = fast, atol = 1.e-7)
@test nugap < 1.e-10

sys = rdss(n,p,m,T = Ty, disc=true); 
@time nugap, fpeak = gnugap(sys,sys, fast = fast, atol = 1.e-7)
@test nugap < 1.e-10

sys = rss(n,p,m, T = Ty); 
@time nugap, fpeak = gnugap(sys,sys,fast = fast, atol = 1.e-7, freq = collect(range(0.1, 0.3, length=5)))
@test maximum(nugap) < 1.e-10

sys = rss(n,p,m,T = Ty, disc=true); 
@time nugap, fpeak = gnugap(sys, sys, fast = fast, atol = 1.e-7, freq = collect(range(0.1, 0.3, length=5)))
@test maximum(nugap) < 1.e-10

sys = rdss(n,p,m,T = Ty); 
@time nugap, fpeak = gnugap(sys, sys,fast = fast, atol = 1.e-7, freq = collect(range(0.1, 0.3, length=5)))
@test maximum(nugap) < 1.e-10

sys = rdss(n,p,m,T = Ty, disc=true); 
@time nugap, fpeak = gnugap(sys, sys, fast = fast, atol = 1.e-7, freq = collect(range(0.1, 0.3, length=5)))
@test maximum(nugap) < 1.e-10

sys1 = rss(n,p,m,T = Ty); sys1d = sys1*0.001/glinfnorm(sys1)[1]; sys2 = sys1+sys1d;
@time nugap, fpeak = gnugap(sys2,sys1,fast = fast, atol = 1.e-7)
@time nugap1, fpeak1 = gnugap(sys2,sys1,fast = fast, atol = 1.e-7,freq=fpeak)
@test nugap ≈ nugap1[1] &&  nugap < 0.01

sys1 = rss(n,p,m,T = Ty); sys1d = rss(3,2,3,stable=true,T = Ty); sys2 = sys1+sys1d*0.001/glinfnorm(sys1d)[1];
@time nugap, fpeak = gnugap(sys2,sys1,fast = fast, atol = 1.e-7)
@time nugap1, fpeak1 = gnugap(sys2,sys1,fast = fast, atol = 1.e-7,freq=fpeak)
@test nugap ≈ nugap1[1] &&  nugap < 0.01

@time nugap, fpeek = gnugap(sys1,sys2,fast = fast, atol=1.e-7)
sysd1 = gbilin(sys1,rtfbilin("c2d")[1])[1]; sysd2 = gbilin(sys2,rtfbilin("c2d")[1])[1]; 
@time nugap1, fpeak1 = gnugap(sysd2,sysd1,fast = fast, atol=1.e-7)
@test abs(nugap-nugap1) < 0.001


sysc1 = gbilin(sysd1,rtfbilin("d2c")[1])[1]; sysc2 = gbilin(sysd2,rtfbilin("d2c")[1])[1]; 
nugap2, fpeak2 = gnugap(sysc2,sysc1,fast = fast, atol=1.e-7)
@test abs(nugap-nugap2) < 0.0001

end # fast
end # Ty
end # gnugap

@testset "Sparse models" begin

# sparse model norms
cd(joinpath(pkgdir(DescriptorSystems), "test"))
#G = load("mirror315.jld2","G");
# alternatively use
A, E, B, C, D = load("mirror315.jld2","A","E","B","C","D");
G = dss(A-0.0001I,E,B,C,D);
Ge = dss(Symmetric(A-0.0001I),Diagonal(E),B,C,D);
Gs = dss(sparse(A-0.0001I),sparse(E),sparse(B),sparse(C),sparse(D));
@test gh2norm(G) ≈ gh2norm(Ge) ≈ gh2norm(Gs)
@test ghanorm(G)[1] ≈ ghanorm(Ge)[1] ≈ ghanorm(Gs)[1]
@test abs(ghinfnorm(G)[1]-ghinfnorm(Gs,fmin=0)[1]) < 1.e-7

@test gnrank(G-G) == 0 &&  gnrank(Ge-Ge) == 0 &&  gnrank(Ge-Ge) == 0 && gnrank(Gs) == 2
@test isregular(Gs)

G = dss(A,E,B,C,D,Ts=0.1);
Ge = dss(Symmetric(A),Diagonal(E),B,C,D,Ts=0.1);
Gs = dss(sparse(A),sparse(E),sparse(B),sparse(C),sparse(D),Ts=0.1);
@test gh2norm(G) ≈ gh2norm(Ge) ≈ gh2norm(Gs)
@test ghanorm(G)[1] ≈ ghanorm(Ge)[1] ≈ ghanorm(Gs)[1]
@test abs(ghinfnorm(G)[1]-ghinfnorm(Gs)[1]) < 1.e-6

end # sparse models

end # test_analysis

end # module