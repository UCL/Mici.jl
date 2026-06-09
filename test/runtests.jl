using Mici
using Test

@testset "Mici.jl" begin
    include("constrained_integrator_tests.jl")
    include("composition_integrator_tests.jl")
    include("gni_splitting_integrator_tests.jl")
    include("gni_composition_integrator_tests.jl")
    include("abstractmcmc_sampling_tests.jl")
end
