include("dependencies_for_runtests.jl")

using Mici.Mici: EuclideanHMC

@testset "Abstract MCMC e2e" begin

    ℓ = 𝒩()
    model = LogDensityModel(ℓ)
    rng = StableRNG(1234)
    C1 = 2.
    C2 = 5.

    sampler = EuclideanHMC(1.5, 3.0)

    initial_q = randn(rng, 2)

    for n_samples in (1000, 10000, 100000)
        samples = sample(rng, model, sampler, n_samples; initial_q, initial_ϵ=0.25, progress=false)
        q = samples.traces.q
        @test norm(mean(q, dims=2) - ℓ.μ) < C1 / sqrt(n_samples)
        @test norm(cov(q, dims=2) - ℓ.Σ) < C2 / sqrt(n_samples)
    end
end