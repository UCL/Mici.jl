include("dependencies_for_runtests.jl")

using Mici.Mici: MiciSampler, LeapfrogIntegrator, EuclideanSystem

@testset "Abstract MCMC e2e" begin

    μ = [0.0 ; 0.0]
    Σ = [1.0 0.2; 0.2 0.35]
    metric = [1.0 0.03; 0.03 0.6]
    q₀ = [4.0; 4.0]
    nsamples = 300

    neg_log_dens, grad_neg_log_dens, metric = setup_gaussian(μ, Σ, metric)
    model = EuclideanSystem(neg_log_dens, grad_neg_log_dens, metric)
    sampler = MiciSampler(LeapfrogIntegrator(model, 0.2, 10), q₀)

    samples = sample(Random.default_rng(), model, sampler, nsamples, chain_type=Any, progress=false)

    @test norm(mean(samples, dims=1)[1] - μ) < 0.3

    @test maximum(abs, cov(samples) - Σ) < 0.3
end