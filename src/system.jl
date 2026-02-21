# Functions and types for defining Hamiltonian systems and their dynamics

"""All systems must implement a field `ℓ`, a function for evaluating the log density and gradient of the target distribution."""
ℓ(system::AbstractSystem) = system.ℓ

h(z::PhasePoint, system::AbstractSystem) = h₁(z, system) + h₂(z, system)
h₁(z::PhasePoint, system::AbstractSystem) = -logdens(z, system)
h₂(z::PhasePoint, system::AbstractSystem) =
    error("h₂(z, system) not implemented for $(typeof(system))")

∂h∂q(z::PhasePoint, system::AbstractSystem) = ∂h₁∂q(z, system) .+ ∂h₂∂q(z, system)
∂h₁∂q(z::PhasePoint, system::AbstractSystem) = grad(z, system)
∂h₂∂q(z::PhasePoint, system::AbstractSystem) =
    error("∂h₂∂q(z, system) not implemented for $(typeof(system))")

∂h∂p(z::PhasePoint, system::AbstractSystem) = ∂h₂∂p(z, system)
∂h₂∂p(z::PhasePoint, system::AbstractSystem) =
    error("∂h₂∂p(z, system) not implemented for $(typeof(system))")

""" 
    AbstractTractableFlowSystem <: AbstractSystem

Abstract supertype for systems where the Hamiltonian dynamics can be solved in closed form, allowing for exact flow transitions in MCMC samplers.
"""
abstract type AbstractTractableFlowSystem <: AbstractSystem end

"""Flow transition for h₁"""
function Φ₁!(z::PhasePoint, system::AbstractTractableFlowSystem, ϵ::Real)
    z.p .-= ϵ .* ∂h₁∂q(z, system)
    return nothing
end

"""Flow transition for h₂"""
function Φ₂!(z::PhasePoint, system::AbstractTractableFlowSystem, ϵ::Real)
    z.q .+= ϵ .* ∂h₂∂p(z, system)
    refresh!(z)
    return nothing
end

"""
    EuclideanSystem{M, L} <: AbstractTractableFlowSystem

Struct for a Euclidean Hamiltonian system, where the kinetic energy is defined by a metric `M`
"""
struct EuclideanSystem{M, L} <: AbstractTractableFlowSystem
    metric::M
    ℓ::L
end

dimension(system::EuclideanSystem) = LogDensityProblems.dimension(ℓ(system))

metric(system::EuclideanSystem) = system.metric

function h₂(z::PhasePoint, system::EuclideanSystem{M}) where M <: AbstractPDMat
     0.5*invquad(metric(system), z.p)
end

function ∂h₂∂p(z::PhasePoint, system::EuclideanSystem)
    metric(system) \ z.p
end

function rand!(rng::AbstractRNG, p::Vector, ::PhasePoint, system::EuclideanSystem)
    randn!(rng, p)
    unwhiten!(system.metric, p)
end

abstract type AbstractRiemannianSystem <: AbstractSystem end

