using Mici.Mici: EuclideanHMC

using AbstractMCMC: sample, LogDensityModel
using CairoMakie
using Distributions
using Enzyme
using LinearAlgebra
using LogDensityProblems
using LogDensityProblemsAD
using PDMats: AbstractPDMat, PDMat, invquad
using StableRNGs


@kwdef struct 𝒩{T, M}
    μ::Vector{T} = [0.0 ; 0.0]
    Σ::M = PDMat([1.0 0.5; 0.5 1.0])
end

@kwdef struct LoopProblem{T} 
    σ::T = 0.2
    y::T = 1.
end

LogDensityProblems.dimension(p::𝒩) = length(p.μ)
LogDensityProblems.dimension(::LoopProblem) = 2

LogDensityProblems.capabilities(::Type{<:𝒩}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity(p::𝒩{T, M}, θ) where {T, M<:AbstractPDMat}
    δ = θ .- p.μ
    -0.5*invquad(p.Σ, δ)
end

function LogDensityProblems.logdensity(ℓ::LoopProblem, θ)
    (; σ, y) = ℓ
    f = θ[2]^2 + 2 * θ[1]^2 * (θ[1]^2 - 0.5)
    -sum(θ.^2) / 2 - ((y - f) / σ)^2 / 2
end

function LogDensityProblems.logdensity_and_gradient(p::𝒩{T, M}, θ) where {T, M<:AbstractPDMat}
    δ = θ .- p.μ
    ℓπ = -0.5*invquad(p.Σ, δ)
    ∇ℓπ = - p.Σ \ δ
    return ℓπ, ∇ℓπ 
end

rng = StableRNG(1234)

# sample from models
ℓ_normal = 𝒩()
model = LogDensityModel(ℓ_normal)
sampler = EuclideanHMC(0.3)
normal_samples = sample(rng, model, sampler, 100000; initial_ϵ=0.05)

# NOTE: sampling seems to fail if random seed not reset
rng = StableRNG(1234)
ℓ_loop = LoopProblem()
ℓ_with_grad = ADgradient(:Enzyme, ℓ_loop)
model = LogDensityModel(ℓ_with_grad)
sampler = EuclideanHMC(0.05, 0.1)
loop_samples = sample(rng, model, sampler, 100000; initial_ϵ=0.01)

# create figures
θ₁_grid = -2:0.02:2
θ₂_grid = -2:0.02:2
normal_grid = [LogDensityProblems.logdensity(ℓ_normal, [θ₁, θ₂]) for θ₁ in θ₁_grid, θ₂ in θ₂_grid]
loop_grid = [LogDensityProblems.logdensity(ℓ_loop, [θ₁, θ₂]) for θ₁ in θ₁_grid, θ₂ in θ₂_grid]

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel="θ₁", ylabel="θ₂", title="Normal distribution")
ax2 = Axis(fig[1, 2], xlabel="θ₁", ylabel="θ₂", title="Loop problem")

contour!(ax1, θ₁_grid, θ₂_grid, exp.(normal_grid))
scatter!(ax1, normal_samples.traces.q[1, :], normal_samples.traces.q[2, :], markersize = 10, alpha = 0.01)

contour!(ax2, θ₁_grid, θ₂_grid, exp.(loop_grid))
scatter!(ax2, loop_samples.traces.q[1, :], loop_samples.traces.q[2, :], markersize = 10, alpha = 0.01)

display(fig)
