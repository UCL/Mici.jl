using PDMats: AbstractPDMat, PDMat, logdet, invquad
using Test
using Random
using LinearAlgebra
using Distributions
using Plots
using StatsBase: sample
using GeometricIntegrators

struct GaussianDensity{M<:AbstractVector, L<:AbstractPDMat}
    μ::M
    Σ::L
end

logdensity(g::GaussianDensity, x::AbstractVector) = -0.5(size(g.Σ, 1)*log(2π) + logdet(g.Σ) + invquad(g.Σ, x .- g.μ))

gradlogdensity(g::GaussianDensity, x::AbstractVector) = - g.Σ \ (x - g.μ)

function setup_gaussian(μ, Σ, metric)

    m = GaussianDensity(μ, PDMat(Σ))
    neg_log_dens = q -> -logdensity(m, q)
    grad_neg_log_dens = q -> -gradlogdensity(m, q)

    return neg_log_dens, grad_neg_log_dens, PDMat(metric)

end