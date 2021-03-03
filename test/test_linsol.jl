module Test_linsol

using DescriptorSystems
using MatrixEquations
using MatrixPencils
using LinearAlgebra
using Polynomials
using Test


@testset "linsol " begin

@testset "grsol" begin

m = 1; mf = 0
sysgf = dss(ones(1,1));
@time sysx, info, sysgen = grsol(sysgf, mf, solgen = true)
isysg = 1:m; isysf = m+1:m+mf;
mnull = size(sysgen,2)-mf; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysgf[:,isysg]*sysx-sysgf[:,isysf]) == 0 &&
      gnrank(sysgf[:,isysg]*sysgen[:,i0]-sysgf[:,isysf], atol = 1.e-7) == 0 &&
      gnrank(sysgf[:,isysg]*sysgen[:,inull], atol = 1.e-7) == 0 &&
      gnrank(sysgf[:,isysg]*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysgf[:,isysf]) == 0 &&
      info.nrank == 1 && info.rdeg == Int[] 

m = 1; mf = 0
sysg = dss(ones(m,m)); sysf = dss(ones(m,mf));
@time sysx, info, sysgen = grsol(sysg, sysf, solgen = true)
isysg = 1:m; isysf = m+1:m+mf;
mnull = size(sysgen,2)-mf; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysg*sysx-sysf) == 0 &&
      gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
      gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf) == 0 &&
      info.nrank == 1 && info.rdeg == Int[] 


m = 5; mf = 2
sysgf = dss(rand(m,m+mf));
@time sysx, info, sysgen = grsol(sysgf, mf, solgen = true)
isysg = 1:m; isysf = m+1:m+mf;
mnull = size(sysgen,2)-mf; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysgf[:,isysg]*sysx-sysgf[:,isysf], atol=1.e-7) == 0 &&
      gnrank(sysgf[:,isysg]*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysgf[:,isysf], atol = 1.e-7) == 0 &&
      info.nrank == 5 && info.rdeg == [0, 0] && info.nr == 0 

m = 5; mf = 2
sysg = dss(rand(m,m)); sysf = dss(rand(m,mf));
@time sysx, info, sysgen = grsol(sysg, sysf, solgen = true)
isysg = 1:m; isysf = m+1:m+mf;
mnull = size(sysgen,2)-mf; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysg*sysx-sysf, atol=1.e-7) == 0 &&
      gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 5 && info.rdeg == [0, 0] && info.nr == 0 

p = 5; m = 8; mf = 2
sysg = dss(rand(p,m)); sysf = dss(rand(p,mf));
@time sysx, info, sysgen = grsol(sysg, sysf, solgen = true)
isysg = 1:m; isysf = m+1:m+mf;
mnull = size(sysgen,2)-mf; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysg*sysx-sysf, atol=1.e-7) == 0 &&
      gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 5 && info.rdeg == [0, 0]  && info.nr == 0 


# Wang and Davison Example (1973)
s = Polynomial([0, 1],'s'); 
gn = [ s+1 s+3 s^2+3*s; s+2 s^2+2*s 0 ]; 
gd = [s^2+3*s+2 s^2+3*s+2 s^2+3*s+2 ;
s^2+3*s+2 s^2+3*s+2 s^2+3*s+2]; 
m = 3; mf = 2;
sysg = dss(gn,gd,minimal = true, atol = 1.e-7); 
sysf = dss([1 0;0 1.]);

@time sysx, info, sysgen = grsol(sysg, sysf, solgen = true)
isysg = 1:m; isysf = m+1:m+mf;
mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysg*sysx-sysf, atol=1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
      gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 2 && info.rdeg == [0, 0] && info.nr == 3

evals=[-1, -2, -3];
@time sysx, info, sysgen = grsol(sysg, sysf, poles = evals, solgen = true)
isysg = 1:m; isysf = m+1:m+mf;
mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysg*sysx-sysf, atol=1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
      gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 2 && info.rdeg == [0, 0] && info.nr == 3 && 
      sort(gpole(sysx)) ≈ sort(evals)

sdeg = -1;
@time sysx, info, sysgen = grsol(sysg, sysf, sdeg = sdeg, solgen = true)
isysg = 1:m; isysf = m+1:m+mf;
mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysg*sysx-sysf, atol=1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
      gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 2 && info.rdeg == [0, 0] && info.nr == 3 && 
      all(real(gpole(sysx)) .< sdeg*(1-eps()^(1/info.nr))) 


@time sysx, info, sysgen = grsol(sysg, sysf; mindeg = true, solgen = true)
isysg = 1:m; isysf = m+1:m+mf;
mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysg*sysx-sysf, atol=1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
      gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 2 && info.rdeg == [0, 0] && info.nr == 3

# Gao & Antsaklis (1989)
s = Polynomial([0, 1],'s'); 
Gn = [s-1 s-1];
Gd = [s*(s+1) s*(s+2)];
Fn = [s-1 s-1];
Fd = [(s+1)*(s+3) (s+1)*(s+4)];
m = 2; mf = 2;
sysg = dss(Gn,Gd,minimal = true, atol = 1.e-7); 
sysf = dss(Fn,Fd,minimal = true, atol = 1.e-7); 

@time sysx, info, sysgen = grsol(sysg, sysf; mindeg = true, solgen = true)
mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysg*sysx-sysf, atol=1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
      gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 1 && info.rdeg == [0, 0] && info.nr == 1

s = Polynomial([0, 1],'s'); 
g = [s^2+s+1 4*s^2+3*s+2 2*s^2-2;
    s 4*s-1 2*s-2;
    s^2 4*s^2-s 2*s^2-2*s]; 
f = g[:,1:2];
m = 3; mf = 2;
sysg = dss(g,minimal = true, atol = 1.e-7); 
sysf = dss(f,minimal = true, atol = 1.e-7); 
@time sysx, info, sysgen = grsol(sysg, sysf; atol = 1.e-7, mindeg = true, solgen = true)
mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysg*sysx-sysf, atol=1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
      gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 2 && info.rdeg == [0, 0] && info.nr == 0

z = Polynomial([0, 1],'s'); 
gn = [z^4-z^3/2-16*z^2-29/2*z+18 z^4+5*z^3-z^2-11*z+6 11/2*z^3+15*z^2+7/2*z-12
    -3*z^2+12 z^3-z^2-4*z+4 z^3+2*z^2-4*z-8;
    z^4-z^3/2-19*z^2-23/2*z+24 z^4+6*z^3-3*z^2-12*z+8 13/2*z^3+16*z^2-z/2-16]; 
gd = [z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2;
z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2;
z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2 z^4+5/2*z^3+2*z^2+z/2];
fn = gn[:,1:2];
fd = gd[:,1:2];
m = 3; mf = 2;
sysg = dss(gn,gd,minimal = true, atol = 1.e-7); 
sysf = dss(fn,fd,minimal = true, atol = 1.e-7); 

@time sysx, info, sysgen = grsol(sysg, sysf; atol = 1.e-7, mindeg = true, solgen = true)
mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
@test gnrank(sysg*sysx-sysf, atol=1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
      gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 2 && info.rdeg == [0, 0] && info.nr == 0

fast = true; Ty = Complex{Float64}; Ty = Float64     
mgf = 6; n = 5; p = 3; mmax = 3; mfmax = 2; mgfmax = mmax+mfmax;
m = 3; mf = 2; 
mindeg = false;

# random examples
for Ty in (Float64, Complex{Float64})

#sysgf = rss(n,p,mgf,T = Ty,disc=false);

for fast in (true, false)

for mindeg in (true, false)

      # continuous, standard 
      sysg = rss(n,p,m,T = Ty, disc=false);
      sysx0 = rss(3,m,mf,T = Ty, disc=false);
      sysf = sysg*sysx0;
      @time sysx, info, sysgen = grsol(sysg, sysf; atol = 1.e-7, mindeg = mindeg, solgen = true, fast = fast)
      mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
      @test gnrank(sysg*sysx-sysf, atol = 1.e-7) == 0 &&
            gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
            gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
            gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf, atol = 1.e-7) == 0 

      # discrete, descriptor
      sysg = rdss(n,p,m,T = Ty, disc=true);
      sysx0 = rss(3,m,mf,T = Ty, disc=true);
      sysf = sysg*sysx0;
      @time sysx, info, sysgen = grsol(sysg, sysf; atol = 1.e-7, mindeg = mindeg, solgen = true, fast = fast)
      mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
      @test gnrank(sysg*sysx-sysf, atol = 1.e-7) == 0 &&
            gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
            gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
            gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf,disc=true))-sysf, atol = 1.e-7) == 0 

      
      # discrete, polynomial
      sysg = rdss(n,p,m,T = Ty, disc=true);
      sysx0 = rdss(0,m,mf,T = Ty, disc=true,id=[3*ones(Int,1);2*ones(Int,1)]);
      sysf = sysg*sysx0;
      @time sysx, info, sysgen = grsol(sysg, sysf; atol = 1.e-7, mindeg = mindeg, solgen = true, fast = fast)
      mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
      @test gnrank(sysg*sysx-sysf, atol = 1.e-7) == 0 &&
            gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
            gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
            gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf, disc=true))-sysf, atol = 1.e-7) == 0 

      # continuous, improper
      sysg = rdss(n,p,m,T = Ty, disc=false, id=[2*ones(Int,1);2*ones(Int,1)]);
      sysx0 = rdss(3,m,mf,T = Ty, disc=false,id=[3*ones(Int,1);2*ones(Int,1)]);
      sysf = sysg*sysx0;
      @time sysx, info, sysgen = grsol(sysg, sysf; atol = 1.e-7, mindeg = mindeg, solgen = true, fast = fast)
      #@time sysx, info, sysgen = grsol(gir([sysg sysf],atol=1.e-7), mf; atol = 1.e-6, mindeg = mindeg, solgen = true, fast = fast)
      mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
      @test gnrank(sysg*sysx-sysf, atol = 1.e-7) == 0 &&
            gnrank(sysg*sysgen[:,i0]-sysf, atol = 1.e-7) == 0 &&
            gnrank(sysg*sysgen[:,inull], atol = 1.e-7) == 0 &&
            gnrank(sysg*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysf, atol = 1.e-7) == 0 

for m = p:mgfmax
      mf = mgfmax-m
      # continuous, standard 
      sysgf = rss(n,p,mgfmax,T = Ty, disc=false);
      @time sysx, info, sysgen = grsol(sysgf, mf; atol = 1.e-7, mindeg = mindeg, solgen = true, fast = fast)
      isysg = 1:m; isysf = m+1:m+mf;
      mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
      @test gnrank(sysgf[:,isysg]*sysx-sysgf[:,isysf], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*sysgen[:,i0]-sysgf[:,isysf], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*sysgen[:,inull], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysgf[:,isysf], atol = 1.e-7) == 0 
      
      # discrete, descriptor
      sysgf = rdss(n,p,mgfmax,T = Ty, disc=true);
      @time sysx, info, sysgen = grsol(sysgf, mf; atol = 1.e-7, mindeg = mindeg, solgen = true, fast = fast)
      isysg = 1:m; isysf = m+1:m+mf;
      mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
      @test gnrank(sysgf[:,isysg]*sysx-sysgf[:,isysf], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*sysgen[:,i0]-sysgf[:,isysf], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*sysgen[:,inull], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf, disc=true))-sysgf[:,isysf], atol = 1.e-7) == 0 

      # discrete, polynomial
      sysgf = rdss(0,p,mgfmax,T = Ty, disc=false,id=[3*ones(Int,1);2*ones(Int,1)]);
      @time sysx, info, sysgen = grsol(sysgf, mf; atol = 1.e-7, mindeg = mindeg, solgen = true, fast = fast)
      isysg = 1:m; isysf = m+1:m+mf;
      mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
      @test gnrank(sysgf[:,isysg]*sysx-sysgf[:,isysf], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*sysgen[:,i0]-sysgf[:,isysf], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*sysgen[:,inull], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf))-sysgf[:,isysf], atol = 1.e-7) == 0 

      # continuous, improper
      sysgf = rdss(n,p,mgfmax,T = Ty, disc=true,id=[3*ones(Int,1);2*ones(Int,1)]);
      @time sysx, info, sysgen = grsol(sysgf, mf; atol = 1.e-7, mindeg = mindeg, solgen = true, fast = fast)
      isysg = 1:m; isysf = m+1:m+mf;
      mnull = m - info.nrank; i0 = 1:mf; inull = mf+1:mf+mnull
      @test gnrank(sysgf[:,isysg]*sysx-sysgf[:,isysf], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*sysgen[:,i0]-sysgf[:,isysf], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*sysgen[:,inull], atol = 1.e-7) == 0 &&
            gnrank(sysgf[:,isysg]*(sysgen[:,i0]+sysgen[:,inull]*rdss(3,mnull,mf, disc=true))-sysgf[:,isysf], atol = 1.e-7) == 0 
end

end # mindeg
end # fast
end # Ty

end # grsol

@testset "glsol" begin

p = 1; pf = 0
sysgf = dss(ones(1,1));
@time sysx, info, sysgen = glsol(sysgf, pf, solgen = true)
isysg = 1:p; isysf = p+1:p+pf;
pnull = p-info.nrank; i0 = 1:pf; inull = pf+1:pf+pnull
@test gnrank(sysx*sysgf[isysg,:]-sysgf[isysf,:]) == 0 &&
      gnrank(sysgen[i0,:]*sysgf[isysg,:]-sysgf[isysf,:], atol = 1.e-7) == 0 &&
      gnrank(sysgen[inull,:]*sysgf[isysg,:], atol = 1.e-7) == 0 &&
      gnrank((sysgen[i0,:]+rdss(3,pf,pnull)*sysgen[inull,:])*sysgf[isysg,:]-sysgf[isysf,:]) == 0 &&
      info.nrank == 1 && info.rdeg == Int[] 

p = 1; pf = 0
sysg = dss(ones(p,p)); sysf = dss(ones(pf,p));
@time sysx, info, sysgen = glsol(sysg, sysf, solgen = true)
pnull = p-info.nrank; i0 = 1:pf; inull = pf+1:pf+pnull
@test gnrank(sysx*sysg-sysf) == 0 &&
      gnrank(sysgen[i0,:]*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[inull,:]*sysg, atol = 1.e-7) == 0 &&
      gnrank((sysgen[i0,:]+rdss(3,pf,pnull)*sysgen[inull,:])*sysg-sysf) == 0 &&
      info.nrank == 1 && info.rdeg == Int[] 


p = 5; pf = 2
sysgf = dss(rand(p+pf,p));
@time sysx, info, sysgen = glsol(sysgf, pf, solgen = true)
isysg = 1:p; isysf = p+1:p+pf;
pnull = p-info.nrank; i0 = 1:pf; inull = pf+1:pf+pnull
@test gnrank(sysx*sysgf[isysg,:]-sysgf[isysf,:], atol = 1.e-7) == 0 &&
      gnrank(sysgen[i0,:]*sysgf[isysg,:]-sysgf[isysf,:], atol = 1.e-7) == 0 &&
      gnrank(sysgen[inull,:]*sysgf[isysg,:], atol = 1.e-7) == 0 &&
      gnrank((sysgen[i0,:]+rdss(3,pf,pnull)*sysgen[inull,:])*sysgf[isysg,:]-sysgf[isysf,:], atol = 1.e-7) == 0 &&
      info.nrank == 5 && info.rdeg == [0, 0] && info.nl == 0 

p = 5; pf = 2
sysg = dss(rand(p,p)); sysf = dss(rand(pf,p));
@time sysx, info, sysgen = glsol(sysg, sysf, solgen = true)
pnull = p-info.nrank; i0 = 1:pf; inull = pf+1:pf+pnull
@test gnrank(sysx*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[i0,:]*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[inull,:]*sysg, atol = 1.e-7) == 0 &&
      gnrank((sysgen[i0,:]+rdss(3,pf,pnull)*sysgen[inull,:])*sysg-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 5 && info.rdeg == [0, 0] && info.nl == 0 

m = 5; p = 8; pf = 2
sysg = dss(rand(p,m)); sysf = dss(rand(pf,m));
@time sysx, info, sysgen = glsol(sysg, sysf, solgen = true)
pnull = p-info.nrank; i0 = 1:pf; inull = pf+1:pf+pnull
@test gnrank(sysx*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[i0,:]*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[inull,:]*sysg, atol = 1.e-7) == 0 &&
      gnrank((sysgen[i0,:]+rdss(3,pf,pnull)*sysgen[inull,:])*sysg-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 5 && info.rdeg == [0, 0]  && info.nl == 0 


# Wang and Davison Example (1973)
s = Polynomial([0, 1],'s'); 
gn = [ s+1 s+2; s+3 s^2+2*s; s^2+3*s 0 ]; 
gd = [s^2+3*s+2 s^2+3*s+2; s^2+3*s+2 s^2+3*s+2; s^2+3*s+2 s^2+3*s+2]; 
p = 3; pf = 2;
sysg = dss(gn,gd,minimal = true, atol = 1.e-7); 
sysf = dss([1 0;0 1.]);

@time sysx, info, sysgen = glsol(sysg, sysf, solgen = true)
pnull = info.pl; i0 = 1:pf; inull = pf+1:pf+pnull
@test gnrank(sysx*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[i0,:]*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[inull,:]*sysg, atol = 1.e-7) == 0 &&
      gnrank((sysgen[i0,:]+rdss(3,pf,pnull)*sysgen[inull,:])*sysg-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 2 && info.rdeg == [0, 0] && info.nl == 3

evals=[-1, -2, -3];
@time sysx, info, sysgen = glsol(sysg, sysf, poles = evals, solgen = true)
pnull = info.pl; i0 = 1:pf; inull = pf+1:pf+pnull
@test gnrank(sysx*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[i0,:]*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[inull,:]*sysg, atol = 1.e-7) == 0 &&
      gnrank((sysgen[i0,:]+rdss(3,pf,pnull)*sysgen[inull,:])*sysg-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 2 && info.rdeg == [0, 0] && info.nl == 3 && info.pl == 1 &&
      sort(gpole(sysx)) ≈ sort(evals)

sdeg = -1;
@time sysx, info, sysgen = glsol(sysg, sysf, sdeg = sdeg, solgen = true)
pnull = info.pl; i0 = 1:pf; inull = pf+1:pf+pnull
@test gnrank(sysx*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[i0,:]*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[inull,:]*sysg, atol = 1.e-7) == 0 &&
      gnrank((sysgen[i0,:]+rdss(3,pf,pnull)*sysgen[inull,:])*sysg-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 2 && info.rdeg == [0, 0] && info.nl == 3 && info.pl == 1 &&
      all(real(gpole(sysx)) .< sdeg*(1-eps()^(1/info.nl))) 


@time sysx, info, sysgen = glsol(sysg, sysf; mindeg = true, solgen = true)
pnull = info.pl; i0 = 1:pf; inull = pf+1:pf+pnull
@test gnrank(sysx*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[i0,:]*sysg-sysf, atol = 1.e-7) == 0 &&
      gnrank(sysgen[inull,:]*sysg, atol = 1.e-7) == 0 &&
      gnrank((sysgen[i0,:]+rdss(3,pf,pnull)*sysgen[inull,:])*sysg-sysf, atol = 1.e-7) == 0 &&
      info.nrank == 2 && info.rdeg == [0, 0] && info.nl == 3 && info.pl == 1 

end # glsol

end # linsol

end # module


