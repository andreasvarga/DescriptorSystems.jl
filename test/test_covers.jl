module Test_covers

using DescriptorSystems
using MatrixEquations
using MatrixPencils
using LinearAlgebra
using Polynomials
using Random
using Test

println("Test_covers")
@testset "covers " begin
Random.seed!(2123)

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





