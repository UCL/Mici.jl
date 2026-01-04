using PDMats: AbstractPDMat, PDMat, logdet, invquad
using Test
using Random
using LinearAlgebra
using Distributions
using Plots
using AbstractMCMC: sample, LogDensityModel
using LogDensityProblems

"""
Define a multivariate Normal distribution to sample from
"""
struct NormalModel{V<:AbstractVector{<:Real}, L<:AbstractPDMat{<:Real}}
    μ::V
    Σ::L
end

LogDensityProblems.dimension(p::NormalModel) = length(p.μ)

LogDensityProblems.capabilities(::Type{<:NormalModel}) = LogDensityProblems.LogDensityOrder{1}()

LogDensityProblems.logdensity(p::NormalModel, x::AbstractVector{<:Real}) = -0.5*invquad(p.Σ, x .- p.μ)

# logdensity_and_gradient(ℓ, x) :: (Real, AbstractVector)
function LogDensityProblems.logdensity_and_gradient(p::NormalModel, x::AbstractVector{<:Real})

    δ = x .- p.μ
    ℓπ = -0.5*invquad(p.Σ, δ)
    ∇ℓπ = - p.Σ \ δ

    return ℓπ, ∇ℓπ 
end

"""
Define example data
"""
normal_model = LogDensityModel(NormalModel([1.0 ; 1.0], PDMat([1.0 0.2; 0.2 0.35])))
