include("dependencies_for_runtests.jl")

using Mici.Mici: EuclideanHMC

@testset "Abstract MCMC e2e" begin

    ℓ = 𝒩()
    model = LogDensityModel(ℓ)
    rng = TaskLocalRNG()
    Random.seed!(rng, 1234)

    sampler = EuclideanHMC(0.3)
    samples = sample(rng, model, sampler, 100000; initial_ϵ=0.05)
    q = samples.traces.q

    @test norm(mean(q, dims=2) - ℓ.μ) < 0.1
    @test maximum(abs, cov(q') - ℓ.Σ) < 0.1

end