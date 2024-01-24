module Runtests

using Test, DescriptorSystems, Aqua

@testset "code quality" begin
    #Aqua.test_all(LinearMaps,piracy = (broken=true,))
    Aqua.test_stale_deps(DescriptorSystems)
end


@testset "Test DescriptorSystems" begin
# test constructors
# include("test_dss.jl")
# include("test_rtf.jl")
# include("test_polrat.jl")
# include("test_connections.jl")
# # test basic functions
# include("test_conversions.jl")
# include("test_timeresp.jl")
# include("test_ordred.jl")
# include("test_analysis.jl")
# # test factorizations
# include("test_pscf.jl")
# include("test_cfid.jl")
# include("test_iofac.jl")
# # test advanced operations
# include("test_gsdec.jl")
# include("test_nullrange.jl")
# include("test_covers.jl")
# include("test_linsol.jl")
# include("test_ginv.jl")
# include("test_model_matching.jl")
end

end
