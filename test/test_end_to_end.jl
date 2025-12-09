include("dependencies_for_runtests.jl")

using Mici.Mici: LeapfrogIntegrator, EuclideanSystem, sample_chain

@testset "Euclidean HMC sampler" begin

    μ = [0.0 ; 0.0]
    Σ = [1.0 0.2; 0.2 0.35]
    metric = [1.0 0.03; 0.03 0.6]
    q₀ = [4.0; 4.0]
    nsamples = 300

    neg_log_dens, grad_neg_log_dens, metric = setup_gaussian(μ, Σ, metric)

    model = EuclideanSystem(neg_log_dens, grad_neg_log_dens, metric)
    integrator = LeapfrogIntegrator(model, 0.2, 10)
    
    samples, chain_state = sample_chain(model, integrator, q₀, nsamples, Random.default_rng())

    @test size(samples) == (nsamples, 2)

    @test all(isfinite, samples)

    @test any(chain_state.accepts[] > 0) && any(chain_state.accepts[] < nsamples)

    @test norm(mean(samples, dims=1)' - μ) < 0.3

    @test maximum(abs, cov(samples) - Σ) < 0.3
end
