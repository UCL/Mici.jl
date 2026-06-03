using Mici
using Test

@testset "Mici.jl" begin
    include("gni_splitting_integrator_tests.jl")
    include("abstractmcmc_sampling_tests.jl")
end
