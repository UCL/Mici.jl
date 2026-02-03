include("dependencies_for_runtests.jl")

using Mici.Mici: HMC

@testset "Abstract MCMC e2e" begin

    nsamples = 300

    rng = Random.MersenneTwister(42)

    sampler = HMC()

    @time samples = sample(rng, normal_model, sampler, nsamples, progress=false)

    @test norm(mean(samples, dims=1)[1] - normal_model.logdensity.μ) < 0.3

    @test maximum(abs, cov(samples) - normal_model.logdensity.Σ) < 0.3
end