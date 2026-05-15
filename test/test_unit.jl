include("dependencies_for_runtests.jl")

using Mici.Mici: EuclideanHMC

@testset "Adapter tests" begin

    
    integrator = LeapfrogAdapterIntegrator(h, 0.2, 10)

end