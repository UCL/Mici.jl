using Test
using PDMats: AbstractPDMat, PDMat, logdet, invquad
using Random
using LinearAlgebra
using Distributions
using Mici
using Plots

struct GaussianDensity{M<:AbstractVector, L<:AbstractPDMat}
    μ::M
    Σ::L
end

logdensity(g::GaussianDensity, x::AbstractVector) = -0.5(size(g.Σ, 1)*log(2π) + logdet(g.Σ) + invquad(g.Σ, x .- g.μ))

gradlogdensity(g::GaussianDensity, x::AbstractVector) = - g.Σ \ (x - g.μ)

@testset "Euclidean HMC sampler" begin

    μ = [0.0 ; 0.0]
    Σ = PDMat([1.0 0.2; 0.2 0.35])
    m = GaussianDensity(μ, Σ)
    neg_log_dens = q -> -logdensity(m, q)
    grad_neg_log_dens = q -> -gradlogdensity(m, q)

    metric = PDMat([1.0 0.03; 0.03 0.6])

    rng = MersenneTwister(42)

    h = EuclideanSystem(neg_log_dens, grad_neg_log_dens, metric)
    integrator = LeapfrogAdapterIntegrator(h, 0.2, 10)

    # Run sampler
    x0 = [4.0; 4.0]
    nsamples = 200
    samples, accepts = sample_chain(h, integrator, x0, nsamples, rng)

    #-------------------------------------------------------------------------------
    # Sanity checks
    #-------------------------------------------------------------------------------

    @test size(samples) == (nsamples, 2)

    @test all(isfinite, samples)

    @test any(accepts) && any(.!accepts)

    @test norm(mean(samples, dims=1)' - μ) < 0.3

    @test maximum(abs, cov(samples) - Σ) < 0.3
end
