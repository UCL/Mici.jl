using Test
using Mici
using Random
using PDMats: AbstractPDMat, PDMat, logdet, invquad
using Distributions

# struct GaussianDensity{M<:AbstractVector, L<:AbstractPDMat}
#     μ::M
#     Σ::L
# end

logdensity(g::GaussianDensity, x::AbstractVector) =
    -0.5(size(g.Σ, 1) * log(2π) + logdet(g.Σ) + invquad(g.Σ, x .- g.μ))

gradlogdensity(g::GaussianDensity, x::AbstractVector) = -g.Σ \ (x - g.μ)

@testset "Building SolutionStep Buffer for integration work" begin
    μ = [0.0; 0.0]
    Σ = PDMat([1.0 0.2; 0.2 0.35])
    m = GaussianDensity(μ, Σ)
    neg_log_dens = q -> -logdensity(m, q)
    grad_neg_log_dens = q -> -gradlogdensity(m, q)
    metric = PDMat([1.0 0.03; 0.03 0.6])

    h = EuclideanSystem(neg_log_dens, grad_neg_log_dens, metric)
    initial_state = [4.0; 4.0; 0.0; 0.0]

    adapter = Mici.Gni.LeapfrogAdapter(h, initial_state, (0.0, 1.0), 0.2)

    @test isa(adapter, Mici.Gni.SeparableODE)
    @test !isnothing(adapter.core.problem)
    @test !isnothing(adapter.core.solution)
    @test !isnothing(adapter.core.integrator)
end

@testset "Instantiating Integration Adapter" begin
    μ = [0.0; 0.0]
    Σ = PDMat([1.0 0.2; 0.2 0.35])
    m = GaussianDensity(μ, Σ)
    neg_log_dens = q -> -logdensity(m, q)
    grad_neg_log_dens = q -> -gradlogdensity(m, q)
    metric = PDMat([1.0 0.03; 0.03 0.6])
    rng = MersenneTwister(42)

    h = EuclideanSystem(neg_log_dens, grad_neg_log_dens, metric)
    integrator = LeapfrogAdapterIntegrator(h, 0.2, 10)

    x0 = [4.0; 4.0]
    nsamples = 50
    samples, accepts = sample_chain(h, integrator, x0, nsamples, rng)

    @test size(samples) == (nsamples, 2)
    @test all(isfinite, samples)
    @test any(accepts) || any(.!accepts)
end
