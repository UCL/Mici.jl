include("dependencies_for_runtests.jl")

using Mici.Mici: MetropolisHMCSampler, MetropolisTransition, GIIntegrator, GIIntegrator2, LeapfrogIntegrator, EuclideanSystem, MarkovChainState, gi_problem

@testset "GeneralIntegrators" begin
    μ = [1.0 ; 1.0]
    Σ = [1.0 0.2; 0.2 0.35]
    metric = [1.0 0.03; 0.03 0.6]
    nsamples = 500

    rng = Random.MersenneTwister(42)

    neg_log_dens, grad_neg_log_dens, metric = setup_gaussian(μ, Σ, metric)
    model = EuclideanSystem(neg_log_dens, grad_neg_log_dens, metric)
    q₀ = [0.0, 1.0]
    p₀ = [1.0, 0.0]

    state = MarkovChainState(q₀, p₀)
    GI1_sampler = MetropolisHMCSampler(GIIntegrator(SymplecticEulerB(),0.2, 10), MetropolisTransition())
    GI1_samples = sample(rng, model, GI1_sampler, nsamples, chain_type=Any, progress=false)

    @test norm(mean(GI1_samples, dims=1)[1] - μ) < 0.1
    @test maximum(abs, cov(GI1_samples) - Σ) < 0.1

    state = MarkovChainState(q₀, p₀)
    GI2_sampler = MetropolisHMCSampler(GIIntegrator2(SymplecticEulerB(),0.2, 10, model, state), MetropolisTransition())
    GI2_samples = sample(rng, model, GI2_sampler, nsamples, chain_type=Any, progress=false)

    @test norm(mean(GI2_samples, dims=1)[1] - μ) < 0.1
    @test maximum(abs, cov(GI2_samples) - Σ) < 0.1

    state = MarkovChainState(q₀, p₀)
    Leapfrog_sampler = MetropolisHMCSampler(LeapfrogIntegrator(0.2, 10), MetropolisTransition())
    Leapfrog_samples = sample(rng, model, Leapfrog_sampler, nsamples, chain_type=Any, progress=false)
    @test norm(mean(Leapfrog_samples, dims=1)[1] - μ) < 0.1
    @test maximum(abs, cov(Leapfrog_samples) - Σ) < 0.1


end