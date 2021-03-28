module Test_rtf

using LinearAlgebra
using MatrixPencils
using Polynomials
using Test
using DescriptorSystems

@testset "test_rtf" begin

# two polynomial entries
a = 1; b = 2; c = 3; d = 4;
@time sys = rtf(Polynomial([b, a],:s), Polynomial([d, c],:s), Ts = 0)
s = Polynomial([0,1],:s)
@time sys1 = rtf(a*s+b, c*s+d)
@time sys2 = rtf(Polynomial([b, a]), Polynomial([d, c]), Ts = 0, var = :s)

@test sys == sys1 && sys == sys2
@test sys ≈ sys1  && sys ≈ sys2

# some basic tests
@test propertynames(sys) == (:zeros, :poles, :gain, :var, :num, :den, :Ts)
@test poles(sys) ≈ [-d/c] && gain(sys) ≈ a/c && length(sys) == 1
@test poles(sys.num) ≈ Int[] && gain(sys.den) ≈ c
@test !isconstant(sys) && !isconstant(sys.num) && isconstant(a)
@test variable(sys) == variable(sys.num)
@test all(zpk(sys) .≈ ([-2.0], [-1.3333333333333333], 0.3333333333333333))
@test all(zpk(sys.num) .≈ ([-2.0], Int64[], 1))
@test sys' == rtf(-a*s+b, -c*s+d)
@test numpoly(5) == Polynomial(5) && denpoly(5) == Polynomial(1)
@test order(sys) == 1
@test convert(RationalTransferFunction{Float64},5) == rtf(5.)
@test DescriptorSystems.promote_var(a*s+b,Polynomial(1)) == :s && 
      DescriptorSystems.promote_var(Polynomial(1),a*s+b) == :s && 
      DescriptorSystems.promote_var(Polynomial(1),Polynomial(1)) == :s 
@test zero(RationalTransferFunction) == rtf(0) && 
      zero(RationalTransferFunction{Float64}) == rtf(0.) &&
      zero(sys) == rtf(0.) && zero(sys) == 0 && zero(sys) == Polynomial(0) &&
      one(RationalTransferFunction) == rtf(1) && rtf(1) == 1 && rtf(1) == Polynomial(1) &&
      one(RationalTransferFunction{Float64}) == rtf(1.) && 
      one(sys) == rtf(1.) 


a = 1; b = 2; c = 3; d = 4;
sysd = rtf(Polynomial([b, a],:z), Polynomial([d, c],:z), Ts = 1)
z = Polynomial([0,1],:z)
sysd1 = rtf(a*z+b, c*z+d, Ts = 1)
sysd2 = rtf(Polynomial([b, a]), Polynomial([d, c]), Ts = 1, var = :z)
@test sysd == sysd1 && sysd == sysd2
@test sysd ≈ sysd1  && sysd == sysd2
@test sysd' == rtf(a+b*z, c+d*z, Ts = 1)
@test rtf(a*z+b, 1, Ts = 1)' == rtf(a+b*z, z, Ts = 1)  
@test rtf(1, a*z+b, Ts = 1)' == rtf(z, a+b*z, Ts = 1)
@test promote_type(typeof(sysd),Float64) == RationalTransferFunction{Float64,:z}

a = 1; b = 2; c = 3; d = 4;
sys = rtf(Polynomial([b, a],:s), d, Ts = 0)
s = Polynomial([0,1],:s)
sys1 = rtf(a*s+b, d)
sys2 = rtf(Polynomial([b, a]), Polynomial([d]), Ts = 0, var = :s)

@test sys == sys1 && sys == sys2
@test sys ≈ sys1  && sys ≈ sys2

a = 1; b = 2; c = 3; d = 4;
sysd = rtf(Polynomial([b],:z), Polynomial([d, c],:z), Ts = 1)
z = Polynomial([0,1],:z)
sysd1 = rtf(b, c*z+d, Ts = 1)
sysd2 = rtf(Polynomial([b]), Polynomial([d, c]), Ts = 1, var = :z)
@test sysd == sysd1 && sysd == sysd2
@test sysd ≈ sysd1  && sysd == sysd2

a = 1; b = 2; c = 3; d = 4;
sys = rtf(Polynomial(b,:s), d, Ts = 0)
s = Polynomial([0,1],:s)
sys1 = rtf(b, d, var=:s)
sys2 = rtf(Polynomial([b]), Polynomial([d]), Ts = 0, var = :s)

@test sys == sys1 && sys == sys2
@test sys ≈ sys1  && sys ≈ sys2

# one polynomial
a = 1; b = 2; c = 3; d = 4;
sys = rtf(Polynomial([b, a],:s), Ts = 0)
s = Polynomial([0,1],:s)
sys1 = rtf(a*s+b,Ts=0)
sys2 = rtf(Polynomial([b, a]), Ts = 0, var = :s)

@test sys == sys1 && sys == sys2
@test sys ≈ sys1  && sys ≈ sys2

a = 1; b = 2; c = 3; d = 4;
sys = rtf(Polynomial([b],:s), Ts = 0)
s = Polynomial([0,1],:s)
sys1 = rtf(b,Ts=0, var=:s)
sys2 = rtf(Polynomial([b]), Ts = 0, var = :s)

@test sys == sys1 && sys == sys2
@test sys ≈ sys1  && sys ≈ sys2

# one rational transfer function
a = 1; b = 2; c = 3; d = 4;
sys = rtf(Polynomial([b, a],:s), Polynomial([d, c],:s), Ts = 0)
z = Polynomial([0,1],:z)
sys1 = rtf((a*z+b)/(c*z+d),Ts=1)

@test rtf(sys,Ts=1,var=:z) == sys1 

# Special constructors

@time t = rtf('s')
@test t == rtf("s") &&  t == rtf(:s) && t.var == :s && t.Ts == 0
@time t = rtf('z')
@test t == rtf("z") &&  t == rtf(:z) && t.var == :z && t.Ts == -1
@time t = rtf('z',Ts=2)
@test t == rtf("z",Ts=2) &&  t == rtf(:z,Ts=2) && t.var == :z && t.Ts == 2
@time t = rtf('λ')
@test t == rtf("λ") &&  t == rtf(:λ) && t.var == :λ 

a = 1; b = 2; c = 3; d = 4;
sys = rtf(Polynomial([b, a],:s), Polynomial([d, c],:s), Ts = 0)
s = rtf('s') 
@test sys == (a*s+b)/(c*s+d)

a = 1; b = 2; c = 3; d = 4;
sysd = rtf(Polynomial([b, a],:z), Polynomial([d, c],:z), Ts = 1)
z = rtf('z',Ts = 1) 
@test sysd == (a*z+b)/(c*z+d)

# zpk 
a = 1; b = 2; c = 3; d = 4;
@time sys = rtf(Polynomial([b, a],:s), Polynomial([d, c],:s), Ts = 0)
@time sys1 = rtf([-b/a],[-d/c],a/c,Ts = 0, var = :s)
@test sys ≈ sys1 && sys1.zeros ≈ [-b/a] && sys1.poles ≈ [-d/c] && 
      sys1.gain ≈ a/c && sys1.Ts == 0 && sys.var == :s


zer = eigvals(rand(5,5)) #fail
pol = eigvals(rand(7,7))
k = .5
@time sys = rtf(k*fromroots(zer),fromroots(pol),var=:s)
@time sys1 = rtf(zer,pol,k,Ts = 0)
@test sys ≈ sys1

# operations
@time r1 = rtf(Polynomial(1), Polynomial([-1, 1])) # r1 =  1/(x-1)
@time r2 = rtf(Polynomial(-1), Polynomial([1, 1])) # r2 = -1/(x+1)
@time r3 = rtf(Polynomial(1), Polynomial([-1, 1]), var = :s) # r3 =  1/(s-1)
@time r4 = rtf(Polynomial(1), Polynomial([-1, 1]), Ts=1, var = :z) # r3 =  1/(z-1)

@test_throws MethodError r3+r4
@test_throws MethodError r3*r4
@test_throws ErrorException rtf(Polynomial(1), Polynomial(0))

@test r1+r2 ≈ rtf(Polynomial(2), Polynomial([-1, 0, 1]))
@test r1-r2 ≈ rtf(Polynomial([0, 2]), Polynomial([-1, 0, 1]))
@test r1*r2 ≈ rtf(Polynomial(-1), Polynomial([-1, 0, 1]))
@test r1/r2 ≈ inv(r2/r1) ≈ -rtf(Polynomial([1, 1]), Polynomial([-1, 1]))

# with polynomials
@time p1 = Polynomial([1, 1], :x)     # p1 = (x+1)
@time p2 = Polynomial([1, 1], :s)     # p2 = (s+1)
r1 = rtf(p1) # r1 = (x+1)/1

@test p1 == r1 == p1
@test p2 ≠ r1 ≠ p2

@test p1 ≈ r1 ≈ p1
@test !(p2 ≈ r1)
@test !(r1 ≈ p2)

@time r1 = rtf(Polynomial([0, 1, 1]), Polynomial([-1, 1])) 

@test p1+r1 ≈ r1+p1 ≈ rtf(Polynomial([-1, 1, 2]), Polynomial([-1, 1]))
@test r1-p1 ≈ -(p1-r1) ≈ rtf(Polynomial([1, 1]),Polynomial([-1, 1]))
@test r1*p1 ≈ p1*r1 ≈ rtf(Polynomial([0, 1, 2, 1]), Polynomial([-1, 1]))
@test r1/p1 ≈ inv(p1/r1) ≈ rtf(Polynomial([0, 1]), Polynomial([-1, 1]))

# with numbers
n   = 3.
@time r1  = rtf(Polynomial(n*[1, 1]), Polynomial([1, 1]))

@test n == r1 == n
@test n+1 ≠ r1 ≠ n+2

@test n ≈ r1 ≈ n
@test n + 0*1im ≈ r1 ≈ n + 0*1im

@time r1  = rtf(Polynomial([-1, 1]), Polynomial([1, 1])) # r1 = (x-1)/(x+1)

@test r1+n ≈ n+r1 ≈ rtf(Polynomial([2, 4]), Polynomial([1, 1]))
@test r1-n ≈ -(n-r1) ≈ rtf(Polynomial([-4, -2]), Polynomial([1, 1]))
@test r1*n ≈ n*r1 ≈ rtf(Polynomial([-3, 3]), Polynomial([1, 1]))
@test r1/n ≈ inv(n/r1) ≈ rtf(Polynomial([-1, 1]), Polynomial([3, 3])) ≈ rtf(Polynomial([-1, 1]/3), Polynomial([1, 1]))

# bilinear transformation
a = 1; b = 2; c = 3; d = 4;
@time sys = rtf(Polynomial([b, a],:s), Polynomial([d, c],:s), Ts = 0)
@time sysi = rtf(Polynomial([-b, d],:s), Polynomial([a, -c],:s))
@test (a*sysi+b)/(c*sysi+d) == s
@test confmap(sys,sysi) == s

s = rtf('s'); z = rtf('z'); λ = rtf(:λ);
@time g, gi = rtfbilin("cayley");
@test confmap(g,gi) == s
@test confmap(gi,g) == z

@time g, gi = rtfbilin("tustin",Ts = 0.01);
@test confmap(g,gi) == s
@test confmap(gi,g) == rtf('z',Ts = 0.01)

@time g, gi = rtfbilin("tustin",Ts = 0.01);
@test confmap(g,gi) == s
@test confmap(gi,g) == rtf('z',Ts = 0.01)

g, gi = rtfbilin("euler",Ts = 0.01);
@test confmap(g,gi) == s
@test confmap(gi,g) == rtf('z',Ts = 0.01)

g, gi = rtfbilin("beuler",Ts = 0.01);
@test confmap(g,gi) == s
@test confmap(gi,g) == rtf('z',Ts = 0.01)

@time g, gi = rtfbilin("Moebius",Ts = 0.01,Tsi = 0.02, a = 1, b = 2, c = 3, d = 4);
@test confmap(g,gi) == rtf('z',Ts = 0.02)
@test confmap(gi,g) == rtf('z',Ts = 0.01)

@time g, gi = rtfbilin("Moebius", a = 1, b = 2, c = 3, d = 4);
@test confmap(g,gi) == s
@test confmap(gi,g) == s

@test_throws ErrorException rtfbilin("ddd")


# normalization
a = 1; b = 2; c = 3; d = 4;
sys = rtf(Polynomial([b, a],:s), Polynomial([d, c],:s), Ts = 0)
@time sysn = normalize(sys)
@test sysn ≈ sys && sysn.den.coeffs[end] == 1 

# simplify
@time r = rtf(Polynomial(rand(3)),Polynomial(rand(4)))
@time t = simplify(r/r)
@test t ≈ 1


# manipulation of rational matrices

# s and z rational functions
s = rtf('s'); z = rtf('z');     # define the complex variables s and z  
@time Gc = [s^2 s/(s+1); 0 1/s] 
@time Gd = [z^2 z/(z-2); 0 1/z] 

# s and z polynomials
s = Polynomial([0, 1],'s'); z = Polynomial([0, 1],'z');  
@time Gc1 = [s^2 s/(s+1); 0 1/s] 
@time Gd1 = [z^2 z/(z-2); 0 1/z] 

@test all( Gc .== rtf.(Gc1,Ts = Gc[1].Ts,var=:s))
@test all( Gd .== rtf.(Gd1,Ts = Gd[1].Ts,var=:z))

@time g, ginv = rtfbilin("c2d")
@test all(confmap.(confmap.(Gc,[g]),[ginv]) .== Gc)
@test all(confmap.(confmap.(Gd,[ginv]),[g]) .== Gd)

@test all(rmconfmap(rmconfmap(Gc,g),ginv) .== Gc)
@test all(rmconfmap(rmconfmap(Gd,ginv),g) .== Gd)

# concatenations
s = rtf('s'); z = rtf('z');     # define the complex variables s and z  
@time Gc = [s^2 s/(s+1); 0 1/s] 
@time Gd = [z^2 z/(z-2); 0 1/z] 
@test all([s^2 s/(s+1)] .== Gc[1:1,:])
@test all([s/(s+1); 1/s] .== Gc[:,2:2])
@time Gc1 = [s^2 s/(s+1) I]
@time Gc2 = [s^2 s/(s+1) 1]
@test all(Gc1 .== Gc2)
@test all([s^2; s/(s+1); I] .==[s^2; s/(s+1); 1])
@test all([[s^2; s/(s+1)]; I] .==[s^2; s/(s+1); 1])

@test all([z^2 z/(z-2)] .== Gd[1:1,:])
@test all([z^2 z/(z-2) I; I] .== [Gd[1:1,:] I;I])
@test all([z/(z-2); 1/z] .== Gd[:,2:2])
@test all([[z/(z-2); 1/z;I] I] .== [[Gd[:,2:2];I] I])

@test_throws ErrorException [Gc Gd]
@test_throws ErrorException [Gc; Gd]
@time [Gc Gc]; [Gd;Gd]

@time Rc=[Gc[:,2:2];I]
@time Rd=[Gd[:,2:2];I]
@test_throws ErrorException [Rc Rd]
@test_throws ErrorException [Rc; Rd]

@test_throws DimensionMismatch [Rc Gc]
@test_throws DimensionMismatch [Rc; Gc]

@time Rc=[Gc[:,2:2] I]
@time Rd=[Gd[2:2,:]; I]

try
   @time [Rc Gc I]
   @test true
catch
   @test false
end

try
   @time [Rd; Gd; I;]
   @test true
catch
   @test false
end

end #test

end #module