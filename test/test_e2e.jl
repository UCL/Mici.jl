include("dependencies_for_runtests.jl")

using Mici.Mici: EuclideanHMC

@testset "Abstract MCMC e2e" begin

    ℓ = 𝒩()
    model = LogDensityModel(ℓ)
    rng = TaskLocalRNG()
    Random.seed!(rng, 1234)
    C1 = 30
    C2 = 45

    sampler = EuclideanHMC(0.3)

    for n_samples in (1000, 10000, 100000)
        samples = sample(rng, model, sampler, n_samples; initial_ϵ=0.05, progress=false)
        q = samples.traces.q
        @test norm(mean(q, dims=2) - ℓ.μ) < C1 / sqrt(n_samples)
        @test norm(q * q' / size(q, 2) - (ℓ.Σ + ℓ.μ * ℓ.μ')) < C2 / sqrt(n_samples)
    end
end