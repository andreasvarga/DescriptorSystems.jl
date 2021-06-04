module Test_ginv

using DescriptorSystems
using LinearAlgebra
using Polynomials
using Random
using Test

println("Test_ginv")
Random.seed!(2123)
@testset "ginv" begin

fast = true; Ty = Complex{Float64}; Ty = Float64  
disc = true;   

# random examples
for Ty in (Float64, Complex{Float64})
      # for fast in (true, false)
      for disc in (false, true)

sys = rdss(0,0,0);
@time sysinv, info = ginv(sys);
@test iszero(sysinv) && info.nrank == 0 && info.nfp == 0 

# invertible 
sys = rss(4,3,3; T = Ty, disc);
@time sysinv, info = ginv(sys; fast);
@test iszero(sys*sysinv-I,atol=1.e-7) &&
      info.nrank == 3 && info.nfp == 0 

sys = rdss(4,3,3; T = Ty, disc);
@time sysinv, info = ginv(sys);
@test iszero(sys*sysinv-I,atol=1.e-7)

# full row rank 1-2 inverse
sys = rss(2,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys,atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 2 && order(sysinv) == 2

sys = rss(2,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, mindeg = true, atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 && order(sysinv) == 1

sys = rdss(2,2,3; T = Ty, disc);
disc ? poles = [0.5, 0.8] : poles = [-1,-2]
@time sysinv, info = ginv(sys; poles, atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 2 && sort(real(gpole(sysinv))) ≈ sort(real(poles)) && 
      sort(abs.(gpole(sysinv))) ≈ sort(abs.(poles)) 

# full row rank 1-2-3-4 inverse
sys = rdss(2,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-3-4", atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sys*sysinv-(sys*sysinv)',atol=1.e-7) && iszero(sysinv*sys-(sysinv*sys)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 


sys = rss(0,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-3-4", atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sys*sysinv-(sys*sysinv)',atol=1.e-7) && iszero(sysinv*sys-(sysinv*sys)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 && order(sysinv) == 0


# full column rank
sys = rss(2,3,2; T = Ty, disc);
@time sysinv, info = ginv(sys,atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 2 && order(sysinv) == 2

sys = rss(0,3,2; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-3-4", atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sys*sysinv-(sys*sysinv)',atol=1.e-7) && iszero(sysinv*sys-(sysinv*sys)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 && order(sysinv) == 0


# full column rank: minimum order
sys = rdss(2,3,2; T = Ty, disc);
@time sysinv, info = ginv(sys, mindeg = true, atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 && order(sysinv) == 1


sys = rdss(2,3,2; T = Ty, disc);
disc ? poles = [0.5, 0.8] : poles = [-1,-2]
@time sysinv, info = ginv(sys; poles, atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 2 && sort(real(gpole(sysinv))) ≈ sort(real(poles)) && 
      sort(abs.(gpole(sysinv))) ≈ sort(abs.(poles)) 

# full column rank 1-2-3-4 inverse
sys = rdss(2,3,2; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-3-4", atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sys*sysinv-(sys*sysinv)',atol=1.e-7) && iszero(sysinv*sys-(sysinv*sys)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 

# rank less than minimum dimension
sys = rss(2,3,2; T = Ty, disc)*rss(2,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys,atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 4 && order(sysinv) == 4

sys = rss(0,3,2; T = Ty, disc)*rss(0,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2", atol=1.e-7)
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sys*sysinv-(sys*sysinv)',atol=1.e-7) && iszero(sysinv*sys-(sysinv*sys)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 && order(sysinv) == 0 

sys = rss(0,3,2; T = Ty, disc)*rss(0,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-3-4", atol=1.e-7)
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sys*sysinv-(sys*sysinv)',atol=1.e-7) && iszero(sysinv*sys-(sysinv*sys)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 && order(sysinv) == 0 


sys = rss(2,3,2; T = Ty, disc)*rss(2,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, mindeg = true, atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 && order(sysinv) == 2 

sys = rdss(2,3,2; T = Ty, disc)*rss(2,2,3; T = Ty, disc);
disc ? poles = [0.5, 0.8] : poles = [-1,-2]
@time sysinv, info = ginv(sys; poles, atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 4 &&
      minimum(abs.(gpole(sysinv).-poles[1])) < 1.e-5 && minimum(abs.(gpole(sysinv).-poles[2])) < 1.e-5 

sys = rss(2,3,2; T = Ty, disc)*rdss(2,2,3; T = Ty, disc);
disc ? sdeg = 0.8 : sdeg = -1
@time sysinv, info = ginv(sys; sdeg, atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 4 &&
      maximum(real(gpole(sysinv)).-sdeg) < 1.e-4 

# 1-2-3-4 inverse
sys = rss(2,3,2; T = Ty, disc)*rdss(2,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-3-4", atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sys*sysinv-(sys*sysinv)',atol=1.e-7) && iszero(sysinv*sys-(sysinv*sys)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0  

# 1-2-3 inverse
sys = rss(2,3,2; T = Ty, disc)*rdss(2,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-3", atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sys*sysinv-(sys*sysinv)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 2 

# 1-2-4 inverse
sys = rss(2,3,2; T = Ty, disc)*rdss(2,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-4", atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
       iszero(sysinv*sys-(sysinv*sys)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 2 

sys = rss(0,3,2; T = Ty, disc)*rss(0,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-3-4", atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sys*sysinv-(sys*sysinv)',atol=1.e-7) && iszero(sysinv*sys-(sysinv*sys)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 && order(sysinv) == 0

sys = rss(0,3,2; T = Ty, disc)*rss(0,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-3", atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sys*sysinv-(sys*sysinv)',atol=1.e-7) && 
      info.nrank == 2 && info.nfp == 0 && order(sysinv) == 0

sys = rss(0,3,2; T = Ty, disc)*rss(0,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-4", atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sysinv*sys-(sysinv*sys)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 && order(sysinv) == 0

sys = rss(0,3,2; T = Ty, disc)*rss(0,2,3; T = Ty, disc);
@time sysinv, info = ginv(sys, type = "1-2-4", atol=1.e-7);
@test iszero(sys*sysinv*sys-sys,atol=1.e-7) && iszero(sysinv*sys*sysinv-sysinv,atol=1.e-7) &&
      iszero(sysinv*sys-(sysinv*sys)',atol=1.e-7) &&
      info.nrank == 2 && info.nfp == 0 && order(sysinv) == 0


@test_throws  ErrorException ginv(sys, atol=1.e-7, type = "1")

end # disc
# end # fast
end # Ty
end # ginv

end # module