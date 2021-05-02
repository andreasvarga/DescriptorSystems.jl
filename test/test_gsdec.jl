module Test_gsdec

using DescriptorSystems
using LinearAlgebra
using Polynomials
using Test


println("Test_gsdec")
@testset "gsdec " begin

sys = rdss(0,0,0);

for job in ("finite", "infinite", "stable", "unstable")
    @time sys1, sys2 = gsdec(sys,job = job)
    @test iszero(sys-sys1-sys2)    
end

fast = true; Ty = Complex{Float64}; Ty = Float64     
m = 2; n = 5; p = 3; 

for fast in (true, false)
# random examples
for Ty in (Float64, Complex{Float64})

# continuous, standard
sys = rss(n,p,m,T = Ty,disc=false);
@time sys1, sys2 = gsdec(sys,fast = fast, job = "finite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    
@time sys1, sys2 = gsdec(sys,fast = fast, job = "infinite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    
@time sys1, sys2 = gsdec(sys,fast = fast, job = "stable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    &&
      all(real(gpole(sys1)) .< 0) && all(real(gpole(sys2)) .> 0)
@time sys1, sys2 = gsdec(sys,fast = fast, job = "unstable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)   &&
      all(real(gpole(sys2)) .< 0) && all(real(gpole(sys1)) .> 0)

# discrete, standard
sys = rss(n,p,m,T = Ty,disc=true);
@time sys1, sys2 = gsdec(sys,fast = fast, job = "finite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    
@time sys1, sys2 = gsdec(sys,fast = fast, job = "infinite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    
@time sys1, sys2 = gsdec(sys,fast = fast, job = "stable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    &&
      all(abs.(gpole(sys1)) .< 1) && all(abs.(gpole(sys2)) .> 1)
@time sys1, sys2 = gsdec(sys,fast = fast, job = "unstable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)   &&
      all(abs.(gpole(sys2)) .< 1) && all(abs.(gpole(sys1)) .> 0)

# continuous, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty,disc=false);
@time sys1, sys2 = gsdec(sys,fast = fast, job = "finite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    
@time sys1, sys2 = gsdec(sys,fast = fast, job = "infinite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    
@time sys1, sys2 = gsdec(sys,fast = fast, job = "stable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    &&
      all(real(gpole(sys1)) .< 0) && all(real(gpole(sys2)) .> 0)
@time sys1, sys2 = gsdec(sys,fast = fast, job = "unstable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)   &&
      all(real(gpole(sys2)) .< 0) && all(real(gpole(sys1)) .> 0)

# discrete, descriptor, no infinite eigenvalues
sys = rdss(n,p,m,T = Ty, disc=true);
@time sys1, sys2 = gsdec(sys,fast = fast, job = "finite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    
@time sys1, sys2 = gsdec(sys,fast = fast, job = "infinite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    
@time sys1, sys2 = gsdec(sys,fast = fast, job = "stable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    &&
      all(abs.(gpole(sys1)) .< 1) && all(abs.(gpole(sys2)) .> 1)
@time sys1, sys2 = gsdec(sys,fast = fast, job = "unstable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)   &&
      all(abs.(gpole(sys2)) .< 1) && all(abs.(gpole(sys1)) .> 0)

# continuous, descriptor, infinite poles
sys = rdss(n,p,m,T = Ty, disc=false,id=[3*ones(Int,1);2*ones(Int,1)]);
@time sys1, sys2 = gsdec(sys,fast = fast, job = "finite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)   &&
      all(isinf.(gpole(sys2,atol=1.e-7))) && all(isfinite.(gpole(sys1,atol=1.e-7)))
@time sys1, sys2 = gsdec(sys,fast = fast, job = "infinite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)   &&
      all(isinf.(gpole(sys1,atol=1.e-7))) && all(isfinite.(gpole(sys2,atol=1.e-7)))
@time sys1, sys2 = gsdec(sys,fast = fast, job = "stable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    &&
      all(real(gpole(sys1)) .< 0) && all(real(gpole(sys2,atol=1.e-7)) .> 0)
@time sys1, sys2 = gsdec(sys,fast = fast, job = "unstable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)   &&
      all(real(gpole(sys2)) .< 0) && all(real(gpole(sys1,atol=1.e-7)) .> 0)
  
# discrete, descriptor, infinite poles
sys = rdss(n,p,m,T = Ty, disc=true,id=[3*ones(Int,1);2*ones(Int,1)]);
@time sys1, sys2 = gsdec(sys,fast = fast, job = "finite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)   &&
      all(isinf.(gpole(sys2,atol=1.e-7))) && all(isfinite.(gpole(sys1,atol=1.e-7))) 
@time sys1, sys2 = gsdec(sys,fast = fast, job = "infinite", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    &&
      all(isinf.(gpole(sys1,atol=1.e-7))) && all(isfinite.(gpole(sys2,atol=1.e-7)))
@time sys1, sys2 = gsdec(sys,fast = fast, job = "stable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)    &&
      all(abs.(gpole(sys1)) .< 1) && all(abs.(gpole(sys2,atol=1.e-7)) .> 1)
@time sys1, sys2 = gsdec(sys,fast = fast, job = "unstable", atol = 1.e-7);
@test iszero(sys-sys1-sys2, atol=1.e-7)   &&
      all(abs.(gpole(sys2)) .< 1) && all(abs.(gpole(sys1,atol=1.e-7)) .> 1)

end
end

end # gsdec

end # module 
