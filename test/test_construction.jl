include("dependencies_for_runtests.jl")

using Mici.Mici: HMC

@testset "Sampling" begin

    system = (type=:euclidean, metric_type=:unit)

    HMC(system, integrator=:leapfrog, integration_transition=:metropolis, momentum_transition=:independent)

end