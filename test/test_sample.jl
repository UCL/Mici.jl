include("dependencies_for_runtests.jl")

using Mici.Mici:
    EuclideanHMC,
    EuclideanNUTS,
    MetropolisHMCState,
    state_type

@testset "state_type" begin
    @test state_type(EuclideanHMC(1.0)) === MetropolisHMCState
    @test state_type(EuclideanNUTS(7)) === MetropolisHMCState
end
