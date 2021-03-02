module Test_covers

using DescriptorSystems
using MatrixEquations
using MatrixPencils
using LinearAlgebra
using Polynomials
using Test
# a = [1 0 0 0 0 0 0 0; 0 1 0 0 0 0 0 0; 0 0 1 0 0 0 0 0; 0 0 0 1 0 0 0 0; 0 0 0 0 1 0 0 0; 0 0 0 0 0 1 0 0; 0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 1];
# e = [0 0 1 0 0 0 0 0; 0 0 0 1 0 0 0 0; 0 0 0 0 1 0 0 0; 0 0 0 0 0 1 0 0; 0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 1; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0];
# b = [0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 1 2 0 1; 0 0 0 -2 0; 0 0 0 0 0; 0 0 0 0 1; 0 0 0 1 1];
# c = [-1 0 0 0 0 0 0 0; 0 -1 0 0 0 0 0 0];
# d = [1 -2 2 0 -2; 2 0 0 0 0];
# sys = dss(a,e,b,c,d);

# ar = [0.1013041538268196 0.09080764507483553 -0.9853410397633098 1.0625181290357943e-16 -1.6653345369377348e-16 -0.8260167694180769; -0.0005673678948759502 -0.0005085807489476311 0.005518538483832262 -4.7704895589362195e-18 2.7755575615628914e-17 0.12639777820751477; -0.010467495525681368 -0.009382918494570711 0.10181273457576707 2.5370330836160804e-17 -2.0816681711721685e-17 0.14781313467002488; -0.00933668474982429 0.9958241178953786 0.0908138344954867 3.0125556170112225e-19 6.25384034058763e-19 2.363088129880722e-18; 0.9947564548000715 0.0 0.10227216451016272 2.0022949695945216e-16 -2.147003351019628e-16 5.76606014745462e-17; 0.0 0.0 0.0 0.9887803758214612 -1.4220806476393557 -0.5837300238472751];
# er = [-1.0000000000000004 -5.18293280715798e-18 -4.69677931598888e-19 -5.910125634223323e-19 -7.470211367054873e-17 -1.1931792334319894e-16; 0.0 1.0 -1.1809147958772554e-34 2.699076362284027e-20 3.411546923017809e-18 5.449092056987511e-18; 0.0 0.0 1.0 -1.4401486225397715e-19 -1.820302185802655e-17 -2.907476990877804e-17; 0.0 0.0 0.0 0.9999843525393174 -0.0039556097592260845 -0.0016327664389422837; 0.0 0.0 0.0 0.0 0.7071178457852374 -0.29186044483441237; 0.0 0.0 0.0 0.0 0.0 0.5837300238472755];
# br = [-0.31833814540765926 -0.17450546765309427 2.1017388200065024; -0.05905544958766715 -0.9447785971039424 0.0; 0.8910324856738809 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0];
# cr = [-4.593393415828979e-18 -2.0977421392776113e-19 1.1192941757927094e-18 4.0657581468206416e-20 1.214306433183765e-17 0.025379566254229304; 4.59339341582899e-17 2.097742139277617e-18 -1.1192941757927125e-17 -2.168404344971009e-19 -6.938893903907228e-17 -0.25379566254229374; -4.593393415828988e-17 -2.0977421392776155e-18 1.1192941757927118e-17 -1.0842021724855044e-19 4.163336342344337e-17 0.2537956625422936; 1.0564804856406674e-16 4.8248069203385166e-18 -2.574376604323237e-17 -0.005594164506161534 -0.707084652162298 -0.29186501192363756; -1.0564804856406674e-16 -4.8248069203385166e-18 2.574376604323237e-17 -2.168404344971009e-19 1.1102230246251565e-16 0.5837300238472755];
# dr = [0.4278145627018142 0.03423935233046432 0.10702588754730953; 0.2841891267513193 -0.24695475791445637 -0.6507436354541494; 0.06185414762518484 0.20763505425690798 -0.7199160645308871; -5.204170427930421e-18 -6.938893903907228e-18 1.1102230246251565e-16; -5.204170427930421e-18 -2.0816681711721685e-17 1.1102230246251565e-16];
# sysr = dss(ar,er,br,cr,dr);
# #sysr = dss(er\ar,er\br,cr,dr);
# gnrank(sys*sysr,atol=1.e-7)

# # A, E, B, C, D = sklf_rightnorm!([3,2,1], copy(ar), copy(er), copy(br), copy(cr), copy(dr)) 
# # sysrn = dss(A, E, B, C, D)
# sklf_rightnorm!([3,2,1], ar,er,br,cr,dr) 
# sysrn = dss(ar,er,br,cr,dr)
# gnrank(sys*sysrn,atol=1.e-7)

# nr1, sysy, info = grmcover1(sysrn[:,[1,2,3]], 1, atol = 1.e-7); 
# order(nr1)
# nr2, sysy, info = grmcover1(sysrn[:,[2,1,3]], 1, atol = 1.e-7); 
# order(nr2)
# nr3, sysy, info = grmcover2(sysrn[:,[3,1,2]], 1, atol = 1.e-7); 
# order(nr3)
# gnrank([nr1 nr2 nr3],atol=1.e-7)

# h = [0,1,1]
# h = qr(rand(3,3)).Q[1,:]
# sysr1 = dss(ar,er,[br*h br],cr, [dr*h dr])
# #sysr1 = dss(ar,er,br*h,cr, dr*h)
# sysx1, sysy1, info1 = grmcover1(sysr1, 1, atol = 1.e-7); info
# order(sysx1)



@testset "covers " begin

@testset "grmcover1" begin

m1 = 0
sys = rdss(0,0,0);
@time sysx, sysy, info = grmcover1(sys, m1)
isys1 = 1:m1; isys2 = m1+1:size(sys,2);

@test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy) == 0   &&   
      info.stdim == [] 

sys1 = rdss(0,0,0); sys2 = rdss(0,0,0); 
@time sysx, sysy, info = grmcover1(sys1, sys2)

@test gnrank(sysx-sys1-sys2*sysy) == 0   &&   
      info.stdim == [] 


m1 = 0
sys = rdss(1,1,1);
@time sysx, sysy, info = grmcover1(sys, m1)
isys1 = 1:m1; isys2 = m1+1:size(sys,2);

@test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy) == 0   &&   
      info.stdim == [] 

sys1 = rdss(1,1,0); sys2 = rdss(1,1,1); 
@time sysx, sysy, info = grmcover1(sys1, sys2)

@test gnrank(sysx-sys1-sys2*sysy) == 0   &&   
      info.stdim == [] 


A = [0 0 0 0 -1;
1 0 0 0 -1;
0 0 -1 -1 0;
0 0 -1 1 0;
1 1 1 0 0.];
B = [1 0 0 0;
0 0 1 0;
0 1 0 0;
0 1 0 1;
0 -1 0 0.];
C = Matrix{Float64}(I,5,5);
E = Matrix{Float64}(I,5,5);
D = zeros(5,4);
sys = dss(A,E,B,C,D);

for m1 = 1:4 
    @time sysx, sysy, info = grmcover1(sys, m1)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0   &&   
          info.stdim == [m1, 1] 
end


for m1 = 1:4 
    sys1 = dss(A,E,B[:,1:m1],C,D[:,1:m1]);
    sys2 = dss(A,E,B[:,m1+1:4],C,D[:,m1+1:4]);
    @time sysx, sysy, info = grmcover1(sys1, sys2)
    @test gnrank(sysx-sys1-sys2*sysy, atol1 = 1.e-7) == 0    &&   
          info.stdim == [m1, 1] 
end

sys = dss(rand(3,4))
m1 = 2
@time sysx, sysy, info = grmcover1(sys, m1)
isys1 = 1:m1; isys2 = m1+1:size(sys,2);

@test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0 &&   
info.stdim == [] 

fast = true; Ty = Complex{Float64}; Ty = Float64     
m = 6; n = 10; p = 3; 
m = 6; n = 5; p = 3; 

for fast in (true, false)
# random examples
for Ty in (Float64, Complex{Float64})

# continuous, standard
sys = rss(n,p,m,T = Ty,disc=false);
for m1 = 0:m
    @time sysx, sysy, info = grmcover1(sys, m1)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end
      
# discrete, standard
sys = rss(n,p,m,T = Ty,disc=true);
for m1 = 0:m
    @time sysx, sysy, info = grmcover1(sys, m1)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end


# continuous, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false);
for m1 = 0:m
    @time sysx, sysy, info = grmcover1(sys, m1)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7,atol2=1.e-7) == 0  
end


# discrete, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty, disc=true);
for m1 = 0:m
    @time sysx, sysy, info = grmcover1(sys, m1)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end

# continuous, descriptor, proper
sys = rdss(n,p,m,T = Ty,disc=false,id=ones(Int,3)); 
for m1 = 0:m
    @time sysx, sysy, info = grmcover1(sys, m1, atol = 1.e-7,atol2=1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7,atol2=1.e-7) == 0  
end

# discrete, descriptor, proper
sys = rdss(n,p,m,T = Ty, disc=true,id=ones(Int,3));
for m1 = 0:m
    @time sysx, sysy, info = grmcover1(sys, m1, atol = 1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end

# continuous, descriptor, infinite poles
for m1 = 0:m
    sys = [ rdss(n,p,m1,T = Ty,disc=false,id=[3*ones(Int,1);2*ones(Int,1)])  rdss(n,p,m-m1,T = Ty,disc=false,id = ones(Int,3)) ] ;
    @time sysx, sysy, info = grmcover1(sys, m1, atol = 1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7,atol2 = 1.e-7) == 0  
end


# discrete, descriptor, uncontrollable poles
sys = rdss(n,p,m,T = Ty, disc=true,nfuc = 4);
for m1 = 0:m
    @time sysx, sysy, info = grmcover1(sys, m1, atol = 1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end

# continuous, descriptor, proper, uncontrollable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false,iduc=[3*ones(Int,1);2*ones(Int,1)]);
for m1 = 0:m
    @time sysx, sysy, info = grmcover1(sys, m1, atol = 1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end

# discrete, descriptor, proper, uncontrollable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true,iduc=[3*ones(Int,1);2*ones(Int,1)]);
for m1 = 0:m
    @time sysx, sysy, info = grmcover1(sys, m1, atol = 1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end



end
end

end # grmcover1

@testset "glmcover1" begin

p1 = 0
sys = rdss(0,0,0);
@time sysx, sysy, info = glmcover1(sys, p1)
osys1 = 1:p1; osys2 = p1+1:size(sys,1);

@test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:]) == 0   &&   
      info.stdim == [] 

sys1 = rdss(0,0,0); sys2 = rdss(0,0,0); 
@time sysx, sysy, info = glmcover1(sys1, sys2)

@test gnrank(sysx-sys1-sysy*sys2) == 0   &&   
      info.stdim == [] 


p1 = 0
sys = rdss(1,1,1);
@time sysx, sysy, info = glmcover1(sys, p1)
osys1 = 1:p1; osys2 = p1+1:size(sys,1);

@test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:]) == 0    &&   
      info.stdim == [] 

sys1 = rdss(1,0,1); sys2 = rdss(1,1,1); 
@time sysx, sysy, info = glmcover1(sys1, sys2)
@test gnrank(sysx-sys1-sysy*sys2) == 0   &&   
      info.stdim == [] 


A = [0 0 0 0 -1;
1 0 0 0 -1;
0 0 -1 -1 0;
0 0 -1 1 0;
1 1 1 0 0.];
B = [1 0 0 0;
0 0 1 0;
0 1 0 0;
0 1 0 1;
0 -1 0 0.];
C = Matrix{Float64}(I,5,5);
E = Matrix{Float64}(I,5,5);
D = zeros(5,4);
sys = gdual(dss(A,E,B,C,D));

for p1 = 1:4 
    @time sysx, sysy, info = glmcover1(sys, p1)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0   &&   
          info.stdim == [1, p1] 
end


for p1 = 1:4 
    sys1 = sys[1:p1,:];
    sys2 = sys[p1+1:end,:];
    @time sysx, sysy, info = glmcover1(sys1, sys2)
    @test gnrank(sysx-sys1-sysy*sys2, atol1 = 1.e-7) == 0    &&   
          info.stdim == [1, p1] 
end

sys = dss(rand(4,3))
p1 = 2
@time sysx, sysy, info = glmcover1(sys, p1)
osys1 = 1:p1; osys2 = p1+1:size(sys,1);
@test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0   &&   
     info.stdim == [] 


fast = true; Ty = Complex{Float64}; Ty = Float64     
p = 6; n = 10; m = 3; 

for fast in (true, false)
# random examples
for Ty in (Float64, Complex{Float64})

# continuous, standard
sys = rss(n,p,m,T = Ty,disc=false);
for p1 = 0:p
    @time sysx, sysy, info = glmcover1(sys, p1)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end
      
# discrete, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty, disc=true);
for p1 = 0:p
    @time sysx, sysy, info = glmcover1(sys, p1, atol1=1.e-7,atol2=1.e-7)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end

# continuous, descriptor, proper
sys = rdss(n,p,m,T = Ty,disc=false,id=ones(Int,3));
for m1 = 0:m
    @time sysx, sysy, info = grmcover1(sys, m1, atol1 = 1.e-7,atol2=1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7,atol2=1.e-7) == 0  
end

# discrete, descriptor, proper
sys = rdss(n,p,m,T = Ty, disc=true,id=ones(Int,3));
for p1 = 0:p
    @time sysx, sysy, info = glmcover1(sys, p1,atol1=1.e-7,atol2=1.e-7)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end

# continuous, descriptor, infinite poles
for p1 = 0:p
    sys = [ rdss(n,p1,m,T = Ty,disc=false,id=[3*ones(Int,1);2*ones(Int,1)]);  rdss(n,p-p1,m,T = Ty,disc=false,id = ones(Int,3)) ] ;
    @time sysx, sysy, info = glmcover1(sys, p1,atol1=1.e-7,atol2=1.e-7)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end


# discrete, descriptor, unobservable poles
sys = rdss(n,p,m,T = Ty, disc=true,nfuo = 4);
for p1 = 0:p
    @time sysx, sysy, info = glmcover1(sys, p1,atol1=1.e-7,atol2=1.e-7)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end

# continuous, descriptor, proper, unobservable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false,iduo=[3*ones(Int,1);2*ones(Int,1)]);
for p1 = 0:p
    @time sysx, sysy, info = glmcover1(sys, p1,atol1=1.e-7,atol2=1.e-7)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end

end
end

end # glmcover1



@testset "grmcover2" begin

m1 = 0
sys = rdss(0,0,0);
@time sysx, sysy, info = grmcover2(sys, m1)
isys1 = 1:m1; isys2 = m1+1:size(sys,2);

@test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy) == 0   &&   
      info.stdim == [] 

sys1 = rdss(0,0,0); sys2 = rdss(0,0,0); 
@time sysx, sysy, info = grmcover2(sys1, sys2)

@test gnrank(sysx-sys1-sys2*sysy) == 0   &&   
      info.stdim == [] 


m1 = 0
sys = rdss(1,1,1);
@time sysx, sysy, info = grmcover2(sys, m1)
isys1 = 1:m1; isys2 = m1+1:size(sys,2);

@test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy) == 0   &&   
      info.stdim == [] 

sys1 = rdss(1,1,0); sys2 = rdss(1,1,1); 
@time sysx, sysy, info = grmcover2(sys1, sys2)

@test gnrank(sysx-sys1-sys2*sysy) == 0   &&   
      info.stdim == [] 


A = [0 0 0 0 -1;
1 0 0 0 -1;
0 0 -1 -1 0;
0 0 -1 1 0;
1 1 1 0 0.];
B = [1 0 0 0;
0 0 1 0;
0 1 0 0;
0 1 0 1;
0 -1 0 0.];
C = Matrix{Float64}(I,5,5);
E = Matrix{Float64}(I,5,5);
D = zeros(5,4);
sys = dss(A,E,B,C,D);

for m1 = 1:4 
    @time sysx, sysy, info = grmcover2(sys, m1)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7,atol2=1.e-7) == 0   &&   
          (info.stdim == [m1] || info.stdim == [m1, 1]) 
end


for m1 = 1:4 
    sys1 = dss(A,E,B[:,1:m1],C,D[:,1:m1]);
    sys2 = dss(A,E,B[:,m1+1:4],C,D[:,m1+1:4]);
    @time sysx, sysy, info = grmcover2(sys1, sys2)
    @test gnrank(sysx-sys1-sys2*sysy, atol1 = 1.e-7) == 0    &&   
        (info.stdim == [m1] || info.stdim == [m1, 1]) 
end

sys = dss(rand(3,4))
m1 = 2
@time sysx, sysy, info = grmcover2(sys, m1)
isys1 = 1:m1; isys2 = m1+1:size(sys,2);

@test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0 &&   
info.stdim == [] 


fast = true; Ty = Complex{Float64}; Ty = Float64     
m = 6; n = 10; p = 3; 

for fast in (true, false)
# random examples
for Ty in (Float64, Complex{Float64})

# continuous, standard
sys = rss(n,p,m,T = Ty,disc=false);
for m1 = 0:m
    @time sysx, sysy, info = grmcover2(sys, m1)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end
      
# discrete, standard
sys = rss(n,p,m,T = Ty,disc=true);
for m1 = 0:m
    @time sysx, sysy, info = grmcover2(sys, m1)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end


# continuous, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false);
for m1 = 0:m
    @time sysx, sysy, info = grmcover2(sys, m1)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7,atol2=1.e-7) == 0  
end


# discrete, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty, disc=true);
for m1 = 0:m
    @time sysx, sysy, info = grmcover2(sys, m1)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end

#error
# continuous, descriptor, proper
sys = rdss(n,p,m,T = Ty,disc=false,id=ones(Int,3));
for m1 = 0:m
    @time sysx, sysy, info = grmcover2(sys, m1, atol = 1.e-7,atol2=1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7,atol2=1.e-7) == 0  
end

# discrete, descriptor, proper
sys = rdss(n,p,m,T = Ty, disc=true,id=ones(Int,3));
for m1 = 0:m
    @time sysx, sysy, info = grmcover2(sys, m1, atol = 1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end

# continuous, descriptor, infinite poles
for m1 = 0:m
    sys = [ rdss(n,p,m1,T = Ty,disc=false,id=[3*ones(Int,1);2*ones(Int,1)])  rdss(n,p,m-m1,T = Ty,disc=false,id = ones(Int,3)) ] ;
    @time sysx, sysy, info = grmcover2(sys, m1, atol = 1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end


# discrete, descriptor, uncontrollable poles
sys = rdss(n,p,m,T = Ty, disc=true,nfuc = 4);
for m1 = 0:m
    @time sysx, sysy, info = grmcover2(sys, m1, atol = 1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end

# continuous, descriptor, proper, uncontrollable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false,iduc=[3*ones(Int,1);2*ones(Int,1)]);
for m1 = 0:m
    @time sysx, sysy, info = grmcover2(sys, m1, atol = 1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end

# discrete, descriptor, proper, uncontrollable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=true,iduc=[3*ones(Int,1);2*ones(Int,1)]);
for m1 = 0:m
    @time sysx, sysy, info = grmcover2(sys, m1, atol = 1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7) == 0  
end



end
end

end # grmcover2

@testset "glmcover2" begin

p1 = 0
sys = rdss(0,0,0);
@time sysx, sysy, info = glmcover2(sys, p1)
osys1 = 1:p1; osys2 = p1+1:size(sys,1);

@test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:]) == 0   &&   
      info.stdim == [] 

sys1 = rdss(0,0,0); sys2 = rdss(0,0,0); 
@time sysx, sysy, info = glmcover2(sys1, sys2)

@test gnrank(sysx-sys1-sysy*sys2) == 0   &&   
      info.stdim == [] 


p1 = 0
sys = rdss(1,1,1);
@time sysx, sysy, info = glmcover2(sys, p1)
osys1 = 1:p1; osys2 = p1+1:size(sys,1);

@test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:]) == 0    &&   
      info.stdim == [] 

sys1 = rdss(1,0,1); sys2 = rdss(1,1,1); 
@time sysx, sysy, info = glmcover2(sys1, sys2)
@test gnrank(sysx-sys1-sysy*sys2) == 0   &&   
      info.stdim == [] 


A = [0 0 0 0 -1;
1 0 0 0 -1;
0 0 -1 -1 0;
0 0 -1 1 0;
1 1 1 0 0.];
B = [1 0 0 0;
0 0 1 0;
0 1 0 0;
0 1 0 1;
0 -1 0 0.];
C = Matrix{Float64}(I,5,5);
E = Matrix{Float64}(I,5,5);
D = zeros(5,4);
sys = gdual(dss(A,E,B,C,D));

for p1 = 1:4 
    @time sysx, sysy, info = glmcover2(sys, p1, atol1=1.e-7,atol2=1.e-7)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0   &&   
    (info.stdim == [p1] || info.stdim == [1, p1])
end


for p1 = 1:4 
    sys1 = sys[1:p1,:];
    sys2 = sys[p1+1:end,:];
    @time sysx, sysy, info = glmcover2(sys1, sys2, atol1=1.e-7,atol2=1.e-7)
    @test gnrank(sysx-sys1-sysy*sys2, atol1 = 1.e-7) == 0    &&   
    (info.stdim == [p1] || info.stdim == [1, p1])
end

sys = dss(rand(4,3))
p1 = 2
@time sysx, sysy, info = glmcover2(sys, p1, atol1=1.e-7,atol2=1.e-7)
osys1 = 1:p1; osys2 = p1+1:size(sys,1);
@test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0   &&   
     info.stdim == [] 


fast = true; Ty = Complex{Float64}; Ty = Float64     
p = 6; n = 10; m = 3; 

for fast in (true, false)
# random examples
for Ty in (Float64, Complex{Float64})

# continuous, standard
sys = rss(n,p,m,T = Ty,disc=false);
for p1 = 0:p
    @time sysx, sysy, info = glmcover2(sys, p1)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end
      
# discrete, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty, disc=true);
for p1 = 0:p
    @time sysx, sysy, info = glmcover2(sys, p1, atol1=1.e-7,atol2=1.e-7)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end

# continuous, descriptor, proper
sys = rdss(n,p,m,T = Ty,disc=false,id=ones(Int,3));
for m1 = 0:m
    @time sysx, sysy, info = grmcover1(sys, m1, atol1 = 1.e-7,atol2=1.e-7)
    isys1 = 1:m1; isys2 = m1+1:size(sys,2);
    @test gnrank(sysx-sys[:,isys1]-sys[:,isys2]*sysy,atol1=1.e-7,atol2=1.e-7) == 0  
end

# discrete, descriptor, proper
sys = rdss(n,p,m,T = Ty, disc=true,id=ones(Int,3));
for p1 = 0:p
    @time sysx, sysy, info = glmcover2(sys, p1,atol1=1.e-7,atol2=1.e-7)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end

# continuous, descriptor, infinite poles
for p1 = 0:p
    sys = [ rdss(n,p1,m,T = Ty,disc=false,id=[3*ones(Int,1);2*ones(Int,1)]);  rdss(n,p-p1,m,T = Ty,disc=false,id = ones(Int,3)) ] ;
    @time sysx, sysy, info = glmcover2(sys, p1,atol1=1.e-7,atol2=1.e-7)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end


# discrete, descriptor, unobservable poles
sys = rdss(n,p,m,T = Ty, disc=true,nfuo = 4);
for p1 = 0:p
    @time sysx, sysy, info = glmcover2(sys, p1,atol1=1.e-7,atol2=1.e-7)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end

# continuous, descriptor, proper, unobservable infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false,iduo=[3*ones(Int,1);2*ones(Int,1)]);
for p1 = 0:p
    @time sysx, sysy, info = glmcover2(sys, p1,atol1=1.e-7,atol2=1.e-7)
    osys1 = 1:p1; osys2 = p1+1:size(sys,1);
    @test gnrank(sysx-sys[osys1,:]-sysy*sys[osys2,:],atol1=1.e-7,atol2=1.e-7) == 0  
end

end
end

end # glmcover2


end #test covers

end #module





