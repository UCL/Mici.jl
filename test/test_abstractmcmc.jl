include("dependencies_for_runtests.jl")

using Mici.Mici: EuclideanHMC

@testset "Abstract MCMC e2e" begin

    ℓ = LogDensityModel(𝒩())

    rng = TaskLocalRNG()
    Random.seed!(rng, 1234)

    sampler = EuclideanHMC(1.0)
    @time samples = sample(rng, ℓ, sampler, 100000; initial_ϵ=0.05)

    q = samples.traces.q

    @test norm(mean(q, dims=2) - ℓ.logdensity.μ) < 0.1
    @test maximum(abs, cov(q') - ℓ.logdensity.Σ) < 0.1
end