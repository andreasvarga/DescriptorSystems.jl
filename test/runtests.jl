module Runtests

using Test, DescriptorSystems

@testset "Test DescriptorSystems" begin
include("test_dss.jl")
include("test_rtf.jl")
include("test_polrat.jl")
include("test_connections.jl")
include("test_conversions.jl")
include("test_ordred.jl")
include("test_analysis.jl")
# test factorizations
include("test_pscf.jl")
include("test_cfid.jl")
include("test_iofac.jl")
# test advanced operations
include("test_gsdec.jl")
include("test_nullrange.jl")
include("test_covers.jl")
end

end
