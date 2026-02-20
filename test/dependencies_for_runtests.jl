using PDMats: AbstractPDMat, PDMat, invquad
using Test
using Random
using LinearAlgebra
using Distributions
using Plots
using AbstractMCMC: sample, LogDensityModel
using LogDensityProblems

@kwdef struct 𝒩{T, M}
    μ::Vector{T} = [0.0 ; 0.0]
    Σ::M = PDMat([1.0 0.5; 0.5 1.0])
end

LogDensityProblems.dimension(p::𝒩) = length(p.μ)

LogDensityProblems.capabilities(::Type{<:𝒩}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity(p::𝒩{T, M}, x::Vector) where {T, M <:AbstractPDMat}
    δ = x .- p.μ
    -0.5*invquad(p.Σ, δ)
end

function LogDensityProblems.logdensity_and_gradient(p::𝒩{T, M}, x::Vector) where {T, M<:AbstractPDMat}
    δ = x .- p.μ
    ℓπ = -0.5*invquad(p.Σ, δ)
    ∇ℓπ = - p.Σ \ δ
    return ℓπ, ∇ℓπ 
end
