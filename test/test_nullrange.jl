module Test_nullrange

using DescriptorSystems
using MatrixEquations
using MatrixPencils
using LinearAlgebra
using Polynomials
using Test


@testset "nullrange" begin

@testset "grnull" begin

p1 = 0; p2 = 0; m = 0; 
sys = dss(ones(p1+p2,m));
@time sysnull, info = grnull(sys,p2)
isys1 = 1:p1; isys2 = p1+1:p1+p2;
indf = m+1:m+p2; indnull = 1:m;
@test gnrank(sys[isys1,:]*sysnull[indnull,:]) == 0 &&
      gnrank(sys[isys2,:]*sysnull[indnull,:]-sysnull[indf,:], atol = 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [] && info.stdim == []

p1 = 0; p2 = 0; m = 1; 
sys = dss(ones(p1+p2,m));
@time sysnull, info = grnull(sys,p2)
isys1 = 1:p1; isys2 = p1+1:p1+p2;
indf = m+1:m+p2; indnull = 1:m;
@test gnrank(sys[isys1,:]*sysnull[indnull,:]) == 0 &&
      gnrank(sys[isys2,:]*sysnull[indnull,:]-sysnull[indf,:], atol = 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [0] && info.stdim == []

p1 = 1; p2 = 1; m = 1; 
sys = dss(ones(p1+p2,m));
@time sysnull, info = grnull(sys,p2)
isys1 = 1:p1; isys2 = p1+1:p1+p2;
indf = m+1:m+p2; indnull = 1:m
@test gnrank(sys[isys1,:]*sysnull[indnull,:]) == 0 &&
      gnrank(sys[isys2,:]*sysnull[indnull,:]-sysnull[indf,:], atol = 1.e-7) == 0 &&
      info.nrank == 1 && info.degs == Int[]  && info.stdim == []

p1 = 1; p2 = 1; m = 2; 
sys = dss(ones(p1+p2,m));
@time sysnull, info = grnull(sys,p2)
isys1 = 1:p1; isys2 = p1+1:p1+p2;
indf = m+1:m+p2; indnull = 1:m;
@test gnrank(sys[isys1,:]*sysnull[indnull,:]) == 0 &&
      gnrank(sys[isys2,:]*sysnull[indnull,:]-sysnull[indf,:], atol = 1.e-7) == 0 &&
      info.nrank == 1 && info.degs == [0] && info.stdim == []

p1 = 0; p2 = 1; m = 2; 
sys = dss(ones(p1+p2,m));
@time sysnull, info = grnull(sys,p2)
isys1 = 1:p1; isys2 = p1+1:p1+p2;
indf = m+1:m+p2; indnull = 1:m;
@test gnrank(sys[isys1,:]*sysnull[indnull,:]) == 0 &&
      gnrank(sys[isys2,:]*sysnull[indnull,:]-sysnull[indf,:], atol = 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [0, 0] && info.stdim == []

p1 = 1; p2 = 1; m = 2; 
sys = dss([1 0; 0 1]);
@time sysnull, info = grnull(sys,p2)
isys1 = 1:p1; isys2 = p1+1:p1+p2;
indf = m+1:m+p2; indnull = 1:m;
@test gnrank(sys[isys1,:]*sysnull[indnull,:]) == 0 &&
      gnrank(sys[isys2,:]*sysnull[indnull,:]-sysnull[indf,:], atol = 1.e-7) == 0 &&
      info.nrank == 1 && info.degs == [0] && info.stdim == []

# uncontrollable realization
s = Polynomial([0, 1],'s') 
m = 2; p1 = 0; p2 = 1; 
G = dss(zeros(0,m)); F = dss([1 1], [(s+1) (s+2)]);
sys = [G; F]

@time nr, info = grnull(sys,p2; atol = 1.e-7); info

@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [1, 1] && info.stdim == [2]

# uncontrollable realization
s = Polynomial([0, 1],'s') 
m = 2; p1 = 0; p2 = 1; 
G = dss(zeros(0,m)); F = dss([1 1], [(s+1) (s+2)]);
sys = [G; F]

@time nr, info = grnull(sys,p2; atol = 1.e-7, simple = true); info

@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [1, 1] && info.stdim == [1, 1]

# uncontrollable realization
s = Polynomial([0, 1],'s') 
m = 2; p1 = 0; p2 = 1; 
G = dss(zeros(0,m)); F = dss([1 1], [(s-1) (s+2)]);
sys = [G; F]

@time nr, info = grnull(sys,p2; atol = 1.e-7, inner = true); info

@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      gnrank(nr[1:m,:]'*nr[1:m,:]-I,atol= 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [1, 1] && info.stdim == [2]

# Example 3, Zuniga & Henrion AMC (2009) 

s = Polynomial([0, 1],'s'); 
m = 5; p1 = 2; p2 = 0; 
G = copy([ 1 2; -2 s; 2 2*s; -2*s^2 s^3; -2+s^3 s+s^3]');
sys = dss(G[:,1:m]);

# minimal rational basis
@time nr, info = grnull(sys,p2,atol=1.e-7); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [3, 2, 1]


# minimal polynomial basis
@time nr, info = grnull(sys,p2;atol=1.e-7,polynomial = true); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      all(isinf.(gpole(nr,atol=1.e-7))) &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [3, 2, 1]


# minimal inner rational basis
@time nr, info = grnull(sys,p2;atol=1.e-7, inner = true); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      gnrank(nr[1:m,:]'*nr[1:m,:]-I,atol= 1.e-7) == 0 &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [3, 2, 1]

z = Polynomial([0, 1],'z'); 
m = 5; p1 = 2; p2 = 0; 
G = copy([ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3]');
sys = dss(G,Ts = 1);

# minimal inner rational basis (discrete-time)
@time nr, info = grnull(sys,atol=1.e-7, inner = true); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      gnrank(nr[1:m,:]'*nr[1:m,:]-I,atol= 1.e-7) == 0 &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [3, 2, 1]

z = Polynomial([0, 1],'z'); 
m = 5; p1 = 2; p2 = 0; 
G = copy([ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3]');
sys = dss(G,Ts = 1);

# minimal rational simple basis (discrete-time)
@time nr, info = grnull(sys,atol=1.e-7, inner = false, simple = true); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [1, 2, 3]

z = Polynomial([0, 1],'z'); 
m = 5; p1 = 2; p2 = 0; 
G = copy([ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3]');
sys = dss(G,Ts = 1);

# minimal rational simple basis with innner vectors (discrete-time)
@time nr, info = grnull(sys,atol=1.e-7, inner = true, simple = true); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      gnrank(nr[:,1]'*nr[:,1]-I,atol= 1.e-7) == 0 &&
      gnrank(nr[:,2]'*nr[:,2]-I,atol= 1.e-7) == 0 &&
      gnrank(nr[:,3]'*nr[:,3]-I,atol= 1.e-7) == 0 &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [1, 2, 3]

z = Polynomial([0, 1],'z'); 
m = 5; p1 = 2; p2 = 0; 
G = copy([ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3]');
sys = dss(G,Ts = 1);

# minimal polynomial simple basis (discrete-time)
@time nr, info = grnull(sys,atol=1.e-7, polynomial = true, simple = false); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      all(isinf.(gpole(nr,atol=1.e-7))) &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [3, 2, 1]

# minimal polynomial simple basis (discrete-time)
@time nr, info = grnull(sys,atol=1.e-7, polynomial = true, simple = true); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      all(isinf.(gpole(nr,atol=1.e-7))) &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [2, 3, 4]

z = Polynomial([0, 1],'z'); 
m = 5; p1 = 2; p2 = 0; 
G = copy([ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3]');
sys = dss(G,Ts = 1);

# minimal simple basis with poles assigned to 0 (discrete-time)
@time nr, info = grnull(sys,atol=1.e-7, poles = [0,0,0], simple = true); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      all(abs.(gpole(nr,atol=1.e-7)).< 0.001) &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [1, 2, 3]

z = Polynomial([0, 1],'z'); 
m = 5; p1 = 2; p2 = 0; 
G = copy([ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3]');
sys = dss(G,Ts = 1);

# minimal rational basis with poles assigned to 0 (discrete-time)
@time nr, info = grnull(sys,atol=1.e-7, sdeg = 0, simple = false); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      all(abs.(gpole(nr,atol=1.e-7)).< 0.01) &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [3, 2, 1]

# Kailath 1980, pag. 459, rank 2 matrix with both left and right nullspaces
s = Polynomial([0, 1],'s'); 
m = 4; p1 = 3; p2 = 0; 
gn = [1 0 1 s;
    0 (s+1)^2 (s+1)^2 0;
    -1 (s+1)^2 s^2+2*s -s^2];  
gd = [s 1 s 1;
    1 1 1 1;
    1 1 1 1];
sys = dss(gn,gd,minimal = true, atol = 1.e-7);

# minimal rational basis with poles assigned to -1
@time nr, info = grnull(sys,atol=1.e-7,poles = [-1,-1]); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      all(real(gpole(nr).+1) .< 2*sqrt(eps())) && all(abs.(imag(gpole(nr))) .< 0.000001) &&
      info.nrank == 2 && info.degs == [0, 2] && info.stdim == [1, 1]

# minimal simple basis with poles assigned to -1
@time nr, info = grnull(sys,atol=1.e-7,poles = [-1,-1], simple = true); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      all(real(gpole(nr)) .≈ -1) && all(abs.(imag(gpole(nr))) .< 0.000001) &&
      info.nrank == 2 && info.degs == [0, 2] && info.stdim == [0, 2]

# minimal simple polynomial basis 
@time nr, info = grnull(sys,atol=1.e-7,polynomial = true, simple = true); info
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      all(isinf.(gpole(nr,atol=1.e-7))) &&
      info.nrank == 2 && info.degs == [0, 2] && info.stdim == [0, 3]

# random examples
m = 5; p1 = 2; p2 = 2; p = p1+p2; n = 5;  

Ty = Float64; fast = true; simple = false; polynomial = false;
for Ty in (Float64, Complex{Float64})

#sysgf = rss(n,p,mgf,T = Ty,disc=false);

for fast in (true, false)

for polynomial in (false, true)

#non-simple basis
# continuous, standard 
sys = rss(n,p,m,T = Ty, disc=false);
@time nr, info = grnull(sys,p2, atol=1.e-7,polynomial = polynomial, simple = false); info 
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      info.nrank == p1 && info.degs == [1, 2, 2] && info.stdim == [3, 2]

# discrete, descriptor
sys = rdss(n,p,m,T = Ty, disc=true);
@time nr, info = grnull(sys,p2, atol=1.e-7,polynomial = polynomial, simple = false); info 
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      info.nrank == p1 && info.degs == [1, 2, 2] && info.stdim == [3, 2]

# discrete, descriptor
sys = rdss(n,p,m,T = Ty, disc=true);
@time nr, info = grnull(sys,p2, atol=1.e-7,polynomial = polynomial, inner = true, simple = false); info 
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      (polynomial || gnrank(nr[1:m,:]'*nr[1:m,:]-I,atol= 1.e-7) == 0) &&
      info.nrank == p1 && info.degs == [1, 2, 2] && info.stdim == [3, 2]
   
# discrete, polynomial
sys = rdss(5,p,m,T = Ty, disc=true, id=[3*ones(Int,1);2*ones(Int,1)]);
@time nr, info = grnull(sys,p2, atol=1.e-7,polynomial = polynomial, simple = false); info 
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      info.nrank == p1 && info.degs == [2, 3, 3] && info.stdim == [3, 3, 2] 

# simple basis
# continuous, standard 
sys = rss(n,p,m,T = Ty, disc=false);
@time nr, info = grnull(sys,p2, atol=1.e-7,polynomial = polynomial, simple = true); info 
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      info.nrank == p1 && info.degs == [1, 2, 2] && info.stdim == (polynomial ? [2, 3, 3] : [1, 2, 2])

# discrete, descriptor
sys = rdss(n,p,m,T = Ty, disc=true);
@time nr, info = grnull(sys,p2, atol=1.e-7,polynomial = polynomial, simple = true); info 
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      info.nrank == p1 && info.degs == [1, 2, 2] && info.stdim == (polynomial ? [2, 3, 3] : [1, 2, 2])

# discrete, descriptor
sys = rdss(n,p,m,T = Ty, disc=true);
@time nr, info = grnull(sys,p2, atol=1.e-7,polynomial = polynomial, inner = true, simple = true); info 
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
      (polynomial || gnrank(nr[1:m,1]'*nr[1:m,1]-I,atol= 1.e-7) == 0) &&
      (polynomial || gnrank(nr[1:m,2]'*nr[1:m,2]-I,atol= 1.e-7) == 0) &&
      info.nrank == p1 && info.degs == [1, 2, 2] && info.stdim == (polynomial ? [2, 3, 3] : [1, 2, 2])
      
# discrete, polynomial
sys = rdss(5,p,m,T = Ty, disc=true, id=[3*ones(Int,1);2*ones(Int,1)]);
@time nr, info = grnull(sys,p2, atol=1.e-7,polynomial = polynomial, simple = true); info 
@test gnrank(sys[1:p1,:]*nr[1:m,:],atol=1.e-7) == 0 &&    
      gnrank(sys[p1+1:p1+p2,:]*nr[1:m,:]-nr[m+1:m+p2,:],atol= 1.e-7) == 0 &&
       info.nrank == p1 && info.degs == [2, 3, 3] && info.stdim == (polynomial ? [3, 4, 4] : [2, 3, 3] )


end # polynomial

end # fast

end # Ty

end # grnull

@testset "glnull" begin

m1 = 0; m2 = 0; p = 0; 
sys = dss(ones(p,m1+m2));
@time sysnull, info = glnull(sys,m2)
isys1 = 1:m1; isys2 = m1+1:m1+m2;
indf = p+1:p+m2; indnull = 1:p;
@test gnrank(sysnull[:,indnull]*sys[:,isys1]) == 0 &&
      gnrank(sysnull[:,indnull]*sys[:,isys2]-sysnull[:,indf], atol = 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [] && info.stdim == []

m1 = 0; m2 = 0; p = 1; 
sys = dss(ones(p,m1+m2));
@time sysnull, info = glnull(sys,m2)
isys1 = 1:m1; isys2 = m1+1:m1+m2;
indf = p+1:p+m2; indnull = 1:p;
@test gnrank(sysnull[:,indnull]*sys[:,isys1]) == 0 &&
      gnrank(sysnull[:,indnull]*sys[:,isys2]-sysnull[:,indf], atol = 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [0] && info.stdim == []

m1 = 1; m2 = 1; p = 1; 
sys = dss(ones(p,m1+m2));
@time sysnull, info = glnull(sys,m2)
isys1 = 1:m1; isys2 = m1+1:m1+m2;
indf = p+1:p+m2; indnull = 1:p;
@test gnrank(sysnull[:,indnull]*sys[:,isys1]) == 0 &&
      gnrank(sysnull[:,indnull]*sys[:,isys2]-sysnull[:,indf], atol = 1.e-7) == 0 &&
      info.nrank == 1 && info.degs == Int[]  && info.stdim == []

m1 = 1; m2 = 1; p = 2; 
sys = dss(ones(p,m1+m2));
@time sysnull, info = glnull(sys,m2)
isys1 = 1:m1; isys2 = m1+1:m1+m2;
indf = p+1:p+m2; indnull = 1:p;
@test gnrank(sysnull[:,indnull]*sys[:,isys1]) == 0 &&
      gnrank(sysnull[:,indnull]*sys[:,isys2]-sysnull[:,indf], atol = 1.e-7) == 0 &&
      info.nrank == 1 && info.degs == [0] && info.stdim == []

m1 = 0; m2 = 1; p = 2; 
sys = dss(ones(p,m1+m2));
@time sysnull, info = glnull(sys,m2)
isys1 = 1:m1; isys2 = m1+1:m1+m2;
indf = p+1:p+m2; indnull = 1:p;
@test gnrank(sysnull[:,indnull]*sys[:,isys1]) == 0 &&
      gnrank(sysnull[:,indnull]*sys[:,isys2]-sysnull[:,indf], atol = 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [0, 0] && info.stdim == []

m1 = 1; m2 = 1; p = 2; 
sys = dss([1 0; 0 1]);
@time sysnull, info = glnull(sys,m2)
isys1 = 1:m1; isys2 = m1+1:m1+m2;
indf = p+1:p+m2; indnull = 1:p;
@test gnrank(sysnull[:,indnull]*sys[:,isys1]) == 0 &&
      gnrank(sysnull[:,indnull]*sys[:,isys2]-sysnull[:,indf], atol = 1.e-7) == 0 &&
      info.nrank == 1 && info.degs == [0] && info.stdim == []

# unobservable realization
s = Polynomial([0, 1],'s') 
p = 2; m1 = 0; m2 = 1; 
G = dss(zeros(p,0)); F = dss([1; 1], [(s+1); (s+2)]);
sys = [G F]

@time nl, info = glnull(sys,m2; atol = 1.e-7); info

@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [1, 1] && info.stdim == [2]

# unobservable realization
s = Polynomial([0, 1],'s') 
p = 2; m1 = 0; m2 = 1; 
G = dss(zeros(p,0)); F = dss([1; 1], [(s+1); (s+2)]);
sys = [G F]

@time nl, info = glnull(sys,m2; atol = 1.e-7, simple = true); info

@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [1, 1] && info.stdim == [1, 1]

# unobservable realization
s = Polynomial([0, 1],'s') 
p = 2; m1 = 0; m2 = 1; 
G = dss(zeros(p,0)); F = dss([1; 1], [(s-1); (s+2)]);
sys = [G F]

@time nl, info = glnull(sys,m2; atol = 1.e-7, coinner = true); info

@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      gnrank(nl[:,1:p]*nl[:,1:p]'-I,atol= 1.e-7) == 0 &&
      info.nrank == 0 && info.degs == [1, 1] && info.stdim == [2]

# Example 3, Zuniga & Henrion AMC (2009) 

s = Polynomial([0, 1],'s'); 
p = 5; m1 = 2; m2 = 0; 
G = [ 1 2; -2 s; 2 2*s; -2*s^2 s^3; -2+s^3 s+s^3];
sys = dss(G[1:p,:]);

# minimal rational basis
@time nl, info = glnull(sys,m2,atol=1.e-7); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [1, 2, 3]


# minimal polynomial basis
@time nl, info = glnull(sys,m2,atol=1.e-7,polynomial = true); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      all(isinf.(gpole(nl,atol=1.e-7))) &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [1, 2, 3]


# minimal inner rational basis
@time nl, info = glnull(sys,m2,atol=1.e-7,coinner = true); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      gnrank(nl[:,1:p]*nl[:,1:p]'-I,atol= 1.e-7) == 0 &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [1, 2, 3]

z = Polynomial([0, 1],'z'); 
p = 5; m1 = 2; m2 = 0; 
G = [ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3];
sys = dss(G,Ts = 1);

# minimal inner rational basis (discrete-time)
@time nl, info = glnull(sys,m2,atol=1.e-7,coinner = true); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      gnrank(nl[:,1:p]*nl[:,1:p]'-I,atol= 1.e-7) == 0 &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [1, 2, 3]

z = Polynomial([0, 1],'z'); 
p = 5; m1 = 2; m2 = 0; 
G = [ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3];
sys = dss(G,Ts = 1);

# minimal rational simple basis (discrete-time)
@time nl, info = glnull(sys,m2,atol=1.e-7,simple = true); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [3, 2, 1]

z = Polynomial([0, 1],'z'); 
p = 5; m1 = 2; m2 = 0; 
G = [ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3];
sys = dss(G,Ts = 1);

# minimal rational simple basis with innner vectors (discrete-time)
@time nl, info = glnull(sys,atol=1.e-7, coinner = true, simple = true); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      gnrank(nl[1,:]*nl[1,:]'-I,atol= 1.e-7) == 0 &&
      gnrank(nl[1,:]*nl[1,:]'-I,atol= 1.e-7) == 0 &&
      gnrank(nl[1,:]*nl[1,:]'-I,atol= 1.e-7) == 0 &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [3, 2, 1]

z = Polynomial([0, 1],'z'); 
p = 5; m1 = 2; m2 = 0; 
G = [ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3];
sys = dss(G,Ts = 1);

# minimal polynomial simple basis (discrete-time)
@time nl, info = glnull(sys,atol=1.e-7, polynomial = true, simple = false); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      all(isinf.(gpole(nl,atol=1.e-7))) &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [1, 2, 3]

# minimal polynomial simple basis (discrete-time)
@time nl, info = glnull(sys,atol=1.e-7, polynomial = true, simple = true); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      all(isinf.(gpole(nl,atol=1.e-7))) &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [4, 3, 2]

z = Polynomial([0, 1],'z'); 
p = 5; m1 = 2; m2 = 0; 
G = [ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3];
sys = dss(G,Ts = 1);

# minimal simple basis with poles assigned to 0 (discrete-time)
@time nl, info = glnull(sys,atol=1.e-7, poles = [0,0,0], simple = true); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      all(abs.(gpole(nl,atol=1.e-7)).< 0.001) &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [3, 2, 1]

z = Polynomial([0, 1],'z'); 
p = 5; m1 = 2; m2 = 0; 
G = [ 1 2; -2 z; 2 2*z; -2*z^2 z^3; -2+z^3 z+z^3];
sys = dss(G,Ts = 1);

# minimal rational basis with poles assigned to 0 (discrete-time)
@time nl, info = glnull(sys,atol=1.e-7, sdeg = 0, simple = false); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      all(abs.(gpole(nl,atol=1.e-7)).< 0.01) &&
      info.nrank == 2 && info.degs == [1, 2, 3] && info.stdim == [1, 2, 3]

# Kailath 1980, pag. 459, rank 2 matrix with both left and right nullspaces
s = Polynomial([0, 1],'s'); 
p = 3; m1 = 4; m2 = 0; 
gn = [1 0 1 s;
    0 (s+1)^2 (s+1)^2 0;
    -1 (s+1)^2 s^2+2*s -s^2];  
gd = [s 1 s 1;
    1 1 1 1;
    1 1 1 1];
sys = dss(gn,gd,minimal = true, atol = 1.e-7);

# minimal rational basis with poles assigned to -1
@time nl, info = glnull(sys,atol=1.e-7,poles = [-1]); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      all(real(gpole(nl)) .≈ -1) && all(abs.(imag(gpole(nl))) .< 0.000001) &&
      info.nrank == 2 && info.degs == [1] && info.stdim == [1]

# minimal simple basis with poles assigned to -1
@time nl, info = glnull(sys,atol=1.e-7,poles = [-1], simple = true); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      all(real(gpole(nl)) .≈ -1) && all(abs.(imag(gpole(nl))) .< 0.000001) &&
      info.nrank == 2 && info.degs == [1] && info.stdim == [1]

# minimal simple polynomial basis 
@time nl, info = glnull(sys,atol=1.e-7,polynomial = true, simple = true); info
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      all(isinf.(gpole(nl,atol=1.e-7))) &&
      info.nrank == 2 && info.degs == [1] && info.stdim == [2]

# random examples
p = 5; m1 = 2; m2 = 2; m = m1+m2; n = 5;  

Ty = Float64; fast = true; simple = false; polynomial = false;
for Ty in (Float64, Complex{Float64})

#sysgf = rss(n,p,mgf,T = Ty,disc=false);

for fast in (true, false)

for polynomial in (false, true)

#non-simple basis
# continuous, standard 
sys = rss(n,p,m,T = Ty, disc=false);
@time nl, info = glnull(sys,m2, atol=1.e-7,polynomial = polynomial, simple = false); info 
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      info.nrank == m1 && info.degs == [1, 2, 2] && info.stdim == [2, 3]

# discrete, descriptor
sys = rdss(n,p,m,T = Ty, disc=true);
@time nl, info = glnull(sys,m2, atol=1.e-7,polynomial = polynomial, simple = false); info 
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      info.nrank == m1 && info.degs == [1, 2, 2] && info.stdim == [2, 3]

# discrete, descriptor
sys = rdss(n,p,m,T = Ty, disc=true);
@time nl, info = glnull(sys,m2, atol=1.e-7,polynomial = polynomial, coinner = true, simple = false); info 
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      (polynomial || gnrank(nl[:,1:p]*nl[:,1:p]'-I,atol= 1.e-7) == 0) &&
      info.nrank == m1 && info.degs == [1, 2, 2] && info.stdim == [2, 3]
   
# discrete, polynomial
sys = rdss(n,p,m,T = Ty, disc=true, id=[3*ones(Int,1);2*ones(Int,1)]);
@time nl, info = glnull(sys,m2, atol=1.e-7,polynomial = polynomial, simple = false); info 
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      info.nrank == m1 && info.degs == [2, 3, 3] && info.stdim == [2, 3, 3]

# simple basis
# continuous, standard 
sys = rss(n,p,m,T = Ty, disc=false);
@time nl, info = glnull(sys,m2, atol=1.e-7,polynomial = polynomial, simple = true); info 
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      info.nrank == m1 && info.degs == [1, 2, 2] && info.stdim == (polynomial ? [3, 3, 2] : [2, 2, 1])

# discrete, descriptor
sys = rdss(n,p,m,T = Ty, disc=true);
@time nl, info = glnull(sys,m2, atol=1.e-7,polynomial = polynomial, simple = true); info 
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      info.nrank == m1 && info.degs == [1, 2, 2] && info.stdim == (polynomial ? [3, 3, 2] : [2, 2, 1])

# discrete, descriptor
sys = rdss(n,p,m,T = Ty, disc=true);
@time nl, info = glnull(sys,m2, atol=1.e-7,polynomial = polynomial, coinner = true, simple = true); info 
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      (polynomial || gnrank(nl[1,1:p]*nl[1,1:p]'-I,atol= 1.e-7) == 0) &&
      (polynomial || gnrank(nl[2,1:p]*nl[2,1:p]'-I,atol= 1.e-7) == 0) &&
      info.nrank == m1 && info.degs == [1, 2, 2] && info.stdim == (polynomial ? [3, 3, 2] : [2, 2, 1])
      
# discrete, polynomial
sys = rdss(5,p,m,T = Ty, disc=true, id=[3*ones(Int,1);2*ones(Int,1)]);
@time nl, info = glnull(sys,m2, atol=1.e-7,polynomial = polynomial, simple = true); info 
@test gnrank(nl[:,1:p]*sys[:,1:m1],atol=1.e-7) == 0 &&    
      gnrank(nl[:,1:p]*sys[:,m1+1:m1+m2]-nl[:,p+1:p+m2],atol= 1.e-7) == 0 &&
      info.nrank == m1 && info.degs == [2, 3, 3] && info.stdim == (polynomial ? [4, 4, 3] : [3, 3, 2] )


end # polynomial

end # fast

end # Ty

end # glnull

end # nullrange
end # module
