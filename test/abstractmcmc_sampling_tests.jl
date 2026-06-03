include("dependencies_for_runtests.jl")

using Mici.Mici: EuclideanHMC, EuclideanHMC1

@testset "Abstract MCMC e2e" begin

    ℓ = 𝒩()
    model = LogDensityModel(ℓ)
    rng = Random.MersenneTwister(1234)

    sampler = EuclideanHMC(1.5, 3.0)

    initial_q = randn(rng, 2)

    n_samples = 100
    samples = sample(rng, model, sampler, n_samples; initial_q, initial_ϵ=0.25, progress=false)
    q = samples.traces.q
    @test size(q) == (2, n_samples)
    @test all(isfinite, q)
end

@testset "GNI Abstract MCMC e2e" begin

    ℓ = 𝒩()
    model = LogDensityModel(ℓ)
    rng = Random.MersenneTwister(1234)

    sampler = EuclideanHMC1(1.5)

    initial_q = randn(rng, 2)

    n_samples = 100
    samples = sample(rng, model, sampler, n_samples; initial_q, initial_ϵ=0.25, progress=false)
    q = samples.traces.q
    @test size(q) == (2, n_samples)
    @test all(isfinite, q)
end
