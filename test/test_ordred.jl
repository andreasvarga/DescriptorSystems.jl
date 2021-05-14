module Test_ordred

using DescriptorSystems
using LinearAlgebra
using Polynomials
using Random
using Test


println("Test_ordred")
Random.seed!(2123)

@testset "Order Reduction Tools" begin

@testset "gss2ss" begin


sys = rdss(0,0,0);
sys1, r = gss2ss(sys);
@test iszero(sys-sys1) && r == 0

a = [8.872223171059933e-01     1.082089117437485e-01    -1.536042862271242e+00
                         0    -5.047254200920694e-01    -5.841082955835114e-02
                         0                         0     1.966838801162540e-01];
b = [ 1.688311161078364e-02     1.129239736565561e+00
     -9.928988024046033e-01     5.713290486116640e-02
      5.766407371577509e-02     8.543665825437958e-01];
c = [ -1.092400401898314e-01     1.061567864551283e+00    -1.535723490367775e+00
       3.312197206498771e-01     4.039678086153220e-02     3.924365487077013e+00];

d = [0     0
     0     0];
e = [4.992358310049688e-01     6.088864643615568e-02    -8.643241045903737e-01
                         0                         0                         0
                         0                         0                         0];

sys = dss(a,e,b,c,d); 
sys1, r = gss2ss(sys)
@test iszero(sys-sys1,atol=1.e-7) && r == 1

Ty = Float64
for Ty in (Float64,Complex{Float64})
sys = rdss(T = Ty,7,2,3);
sys1, r = gss2ss(sys);
@test iszero(sys-sys1,atol=1.e-7) && r == 7 && sys1.E == I

sys1, r = gss2ss(sys,Eshape="triu");
@test iszero(sys-sys1,atol=1.e-7) && r == 7 && istriu(sys1.E)

sys2, r2 = gss2ss(sys1,Eshape="triu");
@test iszero(sys-sys2,atol=1.e-7) && r2 == 7 && istriu(sys2.E)

sys3, r3 = gss2ss(sys1,Eshape="ident");
@test iszero(sys-sys3,atol=1.e-7) && r3 == 7 && sys3.E == I


sys1, r = gss2ss(sys,Eshape="diag");
@test iszero(sys-sys1,atol=1.e-7) && r == 7 && isdiag(sys1.E)

sys = rdss(T = Ty,3,2,3,id=ones(Int,2));
sys1, r = gss2ss(sys);
@test iszero(sys-sys1,atol=1.e-7) && r == 3 && sys1.E == I

sys1, r = gss2ss(sys,Eshape="triu");
@test iszero(sys-sys1,atol=1.e-7) && r == 3 && istriu(sys1.E)

sys1, r = gss2ss(sys,Eshape="diag");
@test iszero(sys-sys1,atol=1.e-7) && r == 3 && isdiag(sys1.E)

sys = rdss(T = Ty,3,2,3,id=[ones(Int,2); 2*ones(Int,1)]);
sys1, r = gss2ss(sys);
@test iszero(sys-sys1,atol=1.e-7) && r == 4 && sys1.E[1:r,1:r] == I

sys1, r = gss2ss(sys,Eshape="triu");
@test iszero(sys-sys1,atol=1.e-7) && r == 4 && istriu(sys1.E)

sys1, r = gss2ss(sys,Eshape="diag");
@test iszero(sys-sys1,atol=1.e-7) && r == 4 && isdiag(sys1.E)

sys = rdss(T = Ty,disc=true,3,2,3,id=[ones(Int,2); 2*ones(Int,1)])';
sys1, r = gss2ss(sys,Eshape="diag");
@test iszero(sys1-sys,atol=1.e-7) && r == 7 && isdiag(sys1.E)

sys = rdss(T = Ty,3,2,2,id=[ones(Int,2); 2*ones(Int,1)],disc=true);
sys1  = gminreal(gss2ss(sys/sys)[1],atol=1.e-7)
@test iszero(sys1-I,atol=1.e-7)

end #Ty



end #gss2ss

@testset "gminreal and gir" begin

for fast in (true, false)

A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(0,0); B2 = zeros(0,0); D2 = zeros(0,0);
sys = dss(A2,E2,B2,C2,D2);

sys1 = gminreal(sys, fast = fast)
@test iszero(sys-sys1,atol=1.e-7) 

sys1 = gir(sys, fast = fast)
@test iszero(sys-sys1,atol=1.e-7) 

A2 = rand(3,3); E2 = zeros(3,3); C2 = zeros(0,3); B2 = zeros(3,0); D2 = zeros(0,0);
sys = dss(A2,E2,B2,C2,D2);

sys1 = gminreal(sys, fast = fast)
@test iszero(sys-sys1,atol=1.e-7) 

sys1 = gminreal(sys, fast = fast, contr = false)
@test iszero(sys-sys1,atol=1.e-7) 

sys1 = gir(sys, fast = fast)
@test iszero(sys-sys1,atol=1.e-7) 

sys1 = gir(sys, fast = fast, contr = false)
@test iszero(sys-sys1,atol=1.e-7) 

# B and D vectors
A2 = rand(3,3); E2 = zeros(3,3); C2 = zeros(1,3); B2 = zeros(3,1); D2 = zeros(1);
sys = dss(A2,E2,B2,C2,D2);

sys1 = gminreal(sys, fast = fast)
@test iszero(sys-sys1,atol=1.e-7) 

sys1 = gminreal(sys, fast = fast, contr = false)
@test iszero(sys-sys1,atol=1.e-7) 

sys1 = gir(sys, fast = fast)
@test iszero(sys-sys1,atol=1.e-7) 

sys1 = gir(sys, fast = fast, contr = false)
@test iszero(sys-sys1,atol=1.e-7) 

# Example 1: DeTeran, Dopico, Mackey, ELA 2009

A2 = [1.0  0.0  0.0  0.0  0.0  0.0
0.0  1.0  0.0  0.0  0.0  0.0
0.0  0.0  1.0  0.0  0.0  0.0
0.0  0.0  0.0  1.0  0.0  0.0
0.0  0.0  0.0  0.0  1.0  0.0
0.0  0.0  0.0  0.0  0.0  1.0];

E2 = [0.0  0.0  1.0  0.0  0.0  0.0
0.0  0.0  0.0  1.0  0.0  0.0
0.0  0.0  0.0  0.0  1.0  0.0
0.0  0.0  0.0  0.0  0.0  1.0
0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0];

B2 = [0.0  0.0
0.0  0.0
0.0  1.0
1.0  0.0
1.0  0.0
0.0  0.0];

C2 = [-1.0   0.0  0.0  0.0  0.0  0.0
0.0  -1.0  0.0  0.0  0.0  0.0];

D2 = [0.0  0.0
0.0  1.0];

sys = dss(A2,E2,B2,C2,D2);

# compute minimal realization 
sys1 = gminreal(sys, fast = fast)
@test iszero(sys-sys1,atol=1.e-7) && order(sys)-order(sys1) == 3 #&& nuo == 0 && nse == 1
# an order reduction without enforcing controllability and observability may not be possible
sys1 = gminreal(sys,contr=false,obs=false, fast = fast)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 0 
# compute an irreducible realization which still contains a non-dynamic mode
sys1 = gminreal(sys,noseig=false, fast = fast)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 2 

sys = dss(E2,A2,B2,C2,D2); 
# compute minimal realization for a standard system (i.e., irreducible realization)
sys1 = gminreal(sys, fast = fast)
@test iszero(sys-sys1,atol=1.e-7) && order(sys)-order(sys1) == 2 

sys = dss(A2,E2,B2,C2,D2);
# compute irreducible realization which still contains a non-dynamic mode using only orthogonal transformations
sys1 = gir(sys, fast = fast)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 2
sys1, L, R = gir_lrtran(sys, fast = fast, ltran = true, rtran = true)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 2 &&
      L*sys.A*R ≈ sys1.A && L*sys.E*R ≈ sys1.E && L*sys.B ≈ sys1.B && sys.C*R ≈ sys1.C

# minimal realization requires elimination of non-dynamic modes
sys1 = gir(sys, noseig=true, fast = fast)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 3
sys1, L, R = gir_lrtran(sys, noseig=true, fast = fast, ltran = true, rtran = true)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 3 &&
      L*sys.A*R ≈ sys1.A && L*sys.E*R ≈ sys1.E && L*sys.B ≈ sys1.B && sys.C*R ≈ sys1.C

# order reduction may results even when applying the infinite controllability/observability algorithm
sys1 = gir(sys, finite = false, fast = fast)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 2
# order reduction may results even when applying the finite controllability/observability algorithm
sys1 = gir(sys,infinite = false, fast = fast)
@test iszero(sys-sys1,atol=1.e-7)   && order(sys)-order(sys1) == 1
# an order reduction without enforcing controllability and observability may not be possible
sys1 = gir(sys,contr=false,obs=false, noseig=true, fast = fast)
@test iszero(sys-sys1,atol=1.e-7)   && order(sys)-order(sys1) == 0


sys = dss(E2,A2,B2,C2,D2); 
# compute minimal realization for a standard system (i.e., irreducible realization)
sys1 = gir(sys, fast = fast)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 2
# an order reduction without enforcing controllability is not be possible
sys1 = gir(sys,contr=false,fast = fast)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 0



# Example Van Dooren & Dewilde, LAA 1983.
# P = zeros(3,3,3)
# P[:,:,1] = [1 2 -2; 0 -1 -2; 0 0 0]
# P[:,:,2] = [1 3 0; 1 4 2; 0 -1 -2]
# P[:,:,3] = [1 4 2; 0 0 0; 1 4 2]

# observable realization with A2 = I
A2 = [ 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0];

E2 = [ 0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0];

B2 = [ 0.0   0.0   0.0
0.0   0.0   0.0
0.0   0.0   0.0
1.0   3.0   0.0
1.0   4.0   2.0
0.0  -1.0  -2.0
1.0   4.0   2.0
0.0   0.0   0.0
1.0   4.0   2.0];

C2 = [ -1.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  -1.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0   0.0  -1.0  0.0  0.0  0.0  0.0  0.0  0.0];

D2 = [ 1.0   2.0  -2.0
0.0  -1.0  -2.0
0.0   0.0   0.0];

sys = dss(A2,E2,B2,C2,D2);
# build a strong (least order) minimal realization 
@time sys1  = gminreal(sys, fast = fast);
@test iszero(sys-sys1,atol=1.e-7) && order(sys)-order(sys1) == 6

# the system is observable
@time sys1 = gminreal(sys,obs=false, fast = fast);
@test iszero(sys-sys1,atol=1.e-7) && order(sys)-order(sys1) == 6

# irreducible realization is not minimal
@time sys1 = gir(sys, fast = fast);
@test iszero(sys-sys1,atol=1.e-7) && order(sys)-order(sys1) == 5

# minimal realization requires removing of non-dynamic modes
@time sys1 = gir(sys, atol = 1.e-7, obs=false, finite = false, noseig = true, fast = fast);
@test iszero(sys-sys1,atol=1.e-7) && order(sys)-order(sys1) == 6
sys1, L, R = gir_lrtran(sys, atol = 1.e-7, obs=false, finite = false, noseig=true, fast = fast, ltran = true, rtran = true)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 6 &&
      L*sys.A*R ≈ sys1.A && L*sys.E*R ≈ sys1.E && L*sys.B ≈ sys1.B && sys.C*R ≈ sys1.C


# Example 1 - (Varga, Kybernetika, 1990) 
A2 = [
1 0 0 0 -1 0 0 0
0 1 0 0 0 -1 0 0
0 0 1 0 0 0 0 0      
0 0 0 1 0 0 0 0
0 0 0 0 -1 0 0 0
0 0 0 0 0 -1 0 0
0 0 0 0 3 0 1 0
0 0 0 0 0 2 0 1
]
E2 = [
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0
0 0 1 0 0 0 0 0
0 0 0 1 0 0 0 0
]
B2 = [
      -1 1
      0 0
      0 0
      0 0
      1 -2
      -2 3
      0 0
      3 -3
]
C2 = [
      0 0 0 0 0 0 -1 0
      0 0 0 0 0 0 0 -1      
]
D2 = zeros(Int,2,2);  

sys = dss(A2,E2,B2,C2,D2);
# build a strong (least order) minimal realization 
@time sys1  = gminreal(sys, fast = fast);
@test iszero(sys-sys1,atol=1.e-7) && order(sys)-order(sys1) == 7

# irreducible realization is not minimal
@time sys1 = gir(sys, atol = 1.e-7, fast = fast);
@test iszero(sys-sys1,atol=1.e-7) && order(sys)-order(sys1) == 6
sys1, L, R = gir_lrtran(sys, atol = 1.e-7, noseig=true, fast = fast, ltran = true, rtran = true)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 7 &&
      L*sys.A*R ≈ sys1.A && L*sys.E*R ≈ sys1.E && L*sys.B ≈ sys1.B && sys.C*R ≈ sys1.C


# SISO standard system, B and D vectors
A2 = [
      0 0 0 -24 0 0 0 0 0 0 0
      1 0 0 -50 0 0 0 0 0 0 0
      0 1 0 -35 0 0 0 0 0 0 0
      0 0 1 -10 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 -30 0 0 0
      0 0 0 0 1 0 0 -61 0 0 0
      0 0 0 0 0 1 0 -41 0 0 0
      0 0 0 0 0 0 1 -11 0 0 0
      0 0 0 0 0 0 0 0 0 0 -15
      0 0 0 0 0 0 0 0 1 0 -23
      0 0 0 0 0 0 0 0 0 1 -9
]
E2 = I;
B2 = [18; 42; 30; 6; 10;17;8;1;0;-10;-2;]
C2 = [0 0 0 0 0 0 0 1 0 0 0]
D2 = [0]


sys = dss(A2,B2,C2,D2);
# build a strong (least order) minimal realization 
@time sys1  = gminreal(sys, fast = fast, atol=1.e-7);
@test iszero(sys-sys1,atol=1.e-7) && order(sys)-order(sys1) == 10

# irreducible realization is also minimal
@time sys1 = gir(sys, atol = 1.e-7, fast = fast);
@test iszero(sys-sys1,atol=1.e-7) && order(sys)-order(sys1) == 10
sys1, L, R = gir_lrtran(sys, atol = 1.e-7, fast = fast, ltran = true, rtran = true)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == 10 &&
      L*sys.A*R ≈ sys1.A && L*sys.E*R ≈ sys1.E && L*sys.B ≈ sys1.B && sys.C*R ≈ sys1.C


for Ty in (Float64, Complex{Float64})

#fast = true; Ty = Complex{Float64}    
n = 10; m = 5; p = 6;
nuc = 3; nuo = 4;
sys = rss(n, p, m; T = Ty, nuc = 3, nuo = 4);

@time sys1  = gminreal(sys, atol = 1.e-7, fast = fast);
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuc+nuo

@time sys1  = gir(sys, atol = 1.e-7, fast = fast);
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuc+nuo
sys1, L, R = gir_lrtran(sys, atol = 1.e-7, fast = fast, ltran = true, rtran = true)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == nuc+nuo &&
      L*sys.A*R ≈ sys1.A && L*sys.E*R ≈ sys1.E && L*sys.B ≈ sys1.B && sys.C*R ≈ sys1.C

@time sys1  = gminreal(sys, atol = 1.e-7, contr = false, fast = fast);
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuo

@time sys1  = gir(sys, atol = 1.e-7, contr = false, fast = fast);
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuo

@time sys1  = gminreal(sys, atol = 1.e-7, obs = false, fast = fast);
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuc

@time sys1  = gir(sys, atol = 1.e-7, obs = false, fast = fast);
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuc

n = 10; m = 5; p = 6;
nfuc = 3; nfuo = 4; Ty = Float64; fast=true;
iduc = ones(Int,3); ; iduo = [ones(Int,3); 2*ones(Int,2)];
nuc = nfuc+sum(iduc); nuo = nfuo+sum(iduo);
sys = rdss(n, p, m; T = Ty, nfuc, nfuo, iduc, iduo);

@time sys1  = gminreal(sys, atol = 1.e-7, fast = fast);
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuc+nuo 

@time sys1  = gir(sys, atol = 1.e-7, fast = fast);
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuc+nuo 
sys1, L, R = gir_lrtran(sys,  atol = 1.e-7, fast = fast, ltran = true, rtran = true)
@test iszero(sys-sys1,atol=1.e-7)  && order(sys)-order(sys1) == nuc+nuo  &&
      L*sys.A*R ≈ sys1.A && L*sys.E*R ≈ sys1.E && L*sys.B ≈ sys1.B && sys.C*R ≈ sys1.C 

@time sys1  = gminreal(sys, atol = 1.e-7, contr = false, noseig = false, fast = fast); 
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuo

@time sys1  = gir(sys, atol = 1.e-7, contr = false, fast = fast);
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuo

@time sys1  = gminreal(sys, atol = 1.e-7, obs = false, noseig = false, fast = fast); 
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuc

@time sys1  = gir(sys, atol = 1.e-7, obs = false, fast = fast);
@test iszero(sys-sys1, atol = 1.e-7) && order(sys)-order(sys1) == nuc



end

end
end


@testset "gbalmr" begin

sys = rss(0,0,0);
@time sysr, hsv = gbalmr(sys)
@test iszero(sys-sysr,atol=1.e-7) && hsv == Float64[]

sys = rdss(0,0,0);
@time sysr, hsv = gbalmr(sys)
@test iszero(sys-sysr,atol=1.e-7) && hsv == Float64[]


n = 5; m = 3; p = 2;

Ty = Float64; fast = true; 
for Ty in (Float64, Complex{Float64})

for fast in (true, false)

# standard continuous-time
sys = rss(n,p,m,stable = true, T = Ty); 

@time sysr, hsv = gbalmr(sys,fast = fast, balance = true)
@test iszero(sys-sysr,atol=1.e-7)

@time sysr, hsv = gbalmr(sys,fast = fast, balance = true, matchdc = true, ord = 3)
@test dcgain(sys) ≈ dcgain(sysr)

@time sysr, hsv = gbalmr(sys,fast = fast, matchdc = true, ord = 3)
@test dcgain(sys) ≈ dcgain(sysr)

try
    @time sysr, hsv = gbalmr(sys',fast = fast)
    @test false
catch
    @test true
end

@time sysr, hsv = gbalmr(sys-sys,fast = fast, atolhsv = 1.e-7)
@test hsv[1] < 1.e-7 && order(sysr) == 0

@time sysr, hsv = gbalmr(sys+sys,fast = fast, atolhsv = 1.e-7, balance = true)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n && iszero(2*sys-sysr, atol=1.e-7)

@time sysr, hsv = gbalmr(sys+sys,fast = fast, atolhsv = 1.e-7)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n && iszero(2*sys-sysr, atol=1.e-7)

@time sysr, hsv = gbalmr([sys sys],fast = fast, atolhsv = 1.e-7, balance = true)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n && iszero([sys sys] - sysr, atol=1.e-7)

@time sysr, hsv = gbalmr([sys sys],fast = fast, atolhsv = 1.e-7)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n && iszero([sys sys] - sysr, atol=1.e-7)

# standard discrete-time
sys = rss(n,p,m,T = Ty, disc = true, stable = true); 
@time sysr, hsv = gbalmr(sys,fast = fast, balance = true)
@test iszero(sys-sysr,atol=1.e-7)

@time sysr, hsv = gbalmr(sys,fast = fast, balance = true, matchdc = true, ord = 3)
@test dcgain(sys) ≈ dcgain(sysr)

@time sysr, hsv = gbalmr(sys,fast = fast, matchdc = true, ord = 3)
@test dcgain(sys) ≈ dcgain(sysr)


try
    @time sysr, hsv = gbalmr(sys',fast = fast)
    @test false
catch
    @test true
end

@time sysr, hsv = gbalmr(sys-sys,fast = fast, atolhsv = 1.e-7)
@test hsv[1]< 1.e-7 && order(sysr) == 0

@time sysr, hsv = gbalmr(sys+sys,fast = fast, atolhsv = 1.e-7)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n && iszero(2*sys-sysr,atol=1.e-7)

@time sysr, hsv = gbalmr([sys sys],fast = fast)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n 
  
# descriptor continuous-time non-singular E
sys = rdss(n,p,m,stable = true, T = Ty); 

@time sysr, hsv = gbalmr(sys,fast = fast, balance = true)
@test iszero(sys-sysr, atol = 1.e-7)

@time sysr, hsv = gbalmr(sys,fast = fast, balance = true, matchdc = true, ord = 3)
@test dcgain(sys) ≈ dcgain(sysr)

@time sysr, hsv = gbalmr(sys,fast = fast, matchdc = true, ord = 3)
@test dcgain(sys) ≈ dcgain(sysr)

try
    @time sysr, hsv = gbalmr(sys',fast = fast)
    @test false
catch
    @test true
end

@time sysr, hsv = gbalmr(sys-sys,fast = fast, atolhsv = 1.e-7)
@test hsv[1] < 1.e-7 && order(sysr) == 0

@time sysr, hsv = gbalmr(sys+sys,fast = fast, atolhsv = 1.e-7)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n && iszero(2*sys-sysr,atol=1.e-7)

@time sysr, hsv = gbalmr([sys sys],fast = fast, atolhsv = 1.e-7)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n 

# descriptor discrete-time non-singular E
sys = rdss(n,p,m,T = Ty, disc = true, stable = true); 
@time sysr, hsv = gbalmr(sys,fast = fast, balance = true)
@test iszero(sys-sysr,atol=1.e-7)

@time sysr, hsv = gbalmr(sys,fast = fast, balance = true, matchdc = true, ord = 3)
@test dcgain(sys) ≈ dcgain(sysr)

@time sysr, hsv = gbalmr(sys,fast = fast, matchdc = true, ord = 3)
@test dcgain(sys) ≈ dcgain(sysr)

try
    @time sysr, hsv = gbalmr(sys',fast = fast)
    @test false
catch
    @test true
end

@time sysr, hsv = gbalmr(sys-sys,fast = fast, atolhsv = 1.e-7)
@test hsv[1]< 1.e-7 && order(sysr) == 0

@time sysr, hsv = gbalmr(sys+sys,fast = fast, atolhsv = 1.e-7)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n && iszero(2*sys-sysr,atol=1.e-7)

@time sysr, hsv = gbalmr([sys sys],fast = fast)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n 

# proper descriptor continuous-time singular E
sys = rdss(n,p,m,T = Ty, stable = true,id=ones(Int,3)); 

@time sysr, hsv = gbalmr(sys,atol=1.e-7,balance = true)
@test iszero(sys-sysr,atol=1.e-7)

@time sysr, hsv = gbalmr(sys,fast = fast, balance = true, matchdc = true, ord = 3)
@test dcgain(sys) ≈ dcgain(sysr)

try
    @time sysr, hsv = gbalmr(sys',fast = fast)
    @test false
catch
    @test true
end

@time sysr, hsv = gbalmr(sys+sys, fast = fast, atolhsv = 1.e-7, atol = 1.e-7)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n && iszero(2*sys-sysr,atol=1.e-7)


@time sysr, hsv = gbalmr([sys sys], ord = n, fast = fast, atol = 1.e-7)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n 


# proper descriptor discrete-time singular E
sys = rdss(n,p,m,T = Ty, stable = true, disc = true, id=ones(Int,3)); 

@time sysr, hsv = gbalmr(sys,atol=1.e-7,balance = true)
@test iszero(sys-sysr,atol=1.e-7)

@time sysr, hsv = gbalmr(sys,fast = fast, balance = true, matchdc = true, ord = 3)
@test dcgain(sys) ≈ dcgain(sysr)

try
    @time sysr, hsv = gbalmr(sys',fast = fast)
    @test false
catch
    @test true
end


@time sysr, hsv = gbalmr(sys+sys,fast = fast, atolhsv = 1.e-7,atol = 1.e-7)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n && iszero(2*sys-sysr,atol=1.e-7)

@time sysr, hsv = gbalmr([sys sys],fast = fast,atol = 1.e-7)
@test norm(hsv[n+1:end]) < 1.e-7 && order(sysr) == n 

end # fast
end # Ty
end # gbalmr

end
end # module