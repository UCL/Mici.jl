include("dependencies_for_runtests.jl")

using Mici.Mici: EuclideanHMC

@testset "Abstract MCMC e2e" begin

    ℓ = LogDensityModel(𝒩())

    rng = TaskLocalRNG()
    Random.seed!(rng, 1234)

    sampler = EuclideanHMC(1.0)

    samples = AbstractMCMC.sample(rng, model, sampler, 100000; initial_ϵ=0.05)

    @test norm(mean(samples, dims=1)[1] - ℓ.logdensity.μ) < 0.3
    @test maximum(abs, cov(samples) - ℓ.logdensity.Σ) < 0.3
end