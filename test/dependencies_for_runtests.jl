using AbstractMCMC: sample, LogDensityModel
using LinearAlgebra
using LogDensityProblems
using PDMats
using StableRNGs
using Statistics
using Test

@kwdef struct 𝒩{T, M}
    μ::Vector{T} = [0.0 ; 0.0]
    Σ::M = ScalMat(2, 1.)
end

LogDensityProblems.dimension(p::𝒩) = length(p.μ)

LogDensityProblems.capabilities(::Type{<:𝒩}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity(p::𝒩{T, M}, θ) where {T, M <:AbstractPDMat}
    δ = θ .- p.μ
    -0.5*invquad(p.Σ, δ)
end

function LogDensityProblems.logdensity_and_gradient(p::𝒩{T, M}, θ) where {T, M<:AbstractPDMat}
    δ = θ .- p.μ
    ℓπ = -0.5*invquad(p.Σ, δ)
    ∇ℓπ = - p.Σ \ δ
    return ℓπ, ∇ℓπ 
end
