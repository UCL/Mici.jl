using Test
using Arianna
import Arianna.RandomWalk
# using Arianna.RandomWalk
using Distributions
using LogDensityProblems
using Random
using AbstractMCMC
using Plots
using LinearAlgebra

@testset "Multidimensional Random Walk" begin
    rng = Random.default_rng()

    # 2D Gaussian
    dist = MvNormal([0.0, 0.0], I(2))
    model = RandomWalk.DistributionModel(dist)

    @test LogDensityProblems.dimension(model) == 2
    @test LogDensityProblems.logdensity(model, [0.0, 0.0]) isa Float64



    # Sampler Tests
    s = RandomWalk.RWSampler(0.1)

    @test s isa RandomWalk.RWSampler
    @test s.stepsize == 0.1

    # Step Tests
    # Initial state
    state1, _ = AbstractMCMC.step(rng, model, s)

    @test state1 isa Vector{Float64}
    @test length(state1) == 2

    # State provided
    state2, _ = AbstractMCMC.step(rng, model, s, state1)

    @test state2 isa Vector{Float64}
    @test length(state2) == 2

    # Sampling Tests
    samples = AbstractMCMC.sample(rng, model, s, 100)

    @test samples isa Matrix{Float64}
    @test size(samples) == (100, 2)

    # Plot each dimension separately
    # x = samples[:,1]
    # y = samples[:,2]
    # t = 1:size(samples,1)

    # p = plot3d(x, y, t,
    # xlabel = "x₁",
    # ylabel = "x₂",
    # zlabel = "Step",
    # title = "3D Random Walk Trajectory",
    # linealpha = 5,
    # legend = false)

    # savefig("trace_mv.png")
end



