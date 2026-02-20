using AbstractMCMC

"""
    AbstractSystem

Base abstract type for Hamiltonian systems with energy
    H(q, p) = H₁(q) + H₂(q, p)
where
    q   --  current position
    p   --  current momentum
    H₁  --  energy term depending on position only
    H₂  --  energy term depending on momentum and position (optionally)

In a standard Euclidean System, H₁ and H₂ correspond to potential energy and
kinetic energy respectively. However, solving the Hamiltonian dynamics in more
complex systems may benefit from a more flexible distinction between (position) and
(momentum, position) energy components.
"""
h(z::PhasePoint, system::AbstractSystem) = h₁(z, system) + h₂(z, system)
h₁(z::PhasePoint, system::AbstractSystem) = -logdens(z, system)
h₂(z::PhasePoint, system::AbstractSystem) =
    error("H₂(z, system) not implemented for $(typeof(system))")

∂h∂q(z::PhasePoint, system::AbstractSystem) = ∂h₁∂q(z, system) .+ ∂h₂∂q(z, system)
∂h₁∂q(z::PhasePoint, system::AbstractSystem) = grad(z, system)
∂h₂∂q(z::PhasePoint, system::AbstractSystem) =
    error("∂H₂∂q(z, system) not implemented for $(typeof(system))")

∂h∂p(z::PhasePoint, system::AbstractSystem) = ∂h₂∂p(z, system)
∂h₂∂p(z::PhasePoint, system::AbstractSystem) =
    error("∂H₂∂p(z, system) not implemented for $(typeof(system))")

"""
    EuclideanSystem

Composite type for an (Unconstrained) Euclidean System, with kinetic energy of the form
    H₂(q, p) = ½ pᵀ M⁻¹ p
where M is a constant positive definite matrix.
"""
abstract type AbstractTractableFlowSystem <: AbstractSystem end

function Φ₁!(z::PhasePoint, system::AbstractTractableFlowSystem, ϵ::Real)
    z.p .-= ϵ .* ∂h₁∂q(z, system)
    return nothing
end

function Φ₂!(z::PhasePoint, system::AbstractTractableFlowSystem, ϵ::Real)
    z.q .+= ϵ .* ∂h₂∂p(z, system)
    refresh!(z)
    return nothing
end
struct EuclideanSystem{M, L} <: AbstractTractableFlowSystem
    metric::M
    ℓ::L
end

metric(system::EuclideanSystem) = system.metric

function h₂(z::PhasePoint, system::EuclideanSystem{M}) where M <: AbstractPDMat
     0.5*invquad(metric(system), z.p)
end

function ∂h₂∂p(z::PhasePoint, system::EuclideanSystem)
    metric(system) \ z.p
end

∂H₂∂q(z::PhasePoint, system::EuclideanSystem) = zeros(size(metric(system), 1))

abstract type AbstractRiemannianSystem <: AbstractSystem end

