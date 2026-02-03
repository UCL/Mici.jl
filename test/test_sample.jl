include("dependencies_for_runtests.jl")

using Mici.Mici: HMC, instantiate

@testset "Sampling" begin

    spec = HMC()

    hmc = instantiate(spec, normal_model.logdensity)

end