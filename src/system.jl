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
abstract type AbstractSystem end

H(h::AbstractSystem, point::AbstractPhasePoint) = H₁(h, point) + H₂(h, point)
H₁(h::AbstractSystem, point::AbstractPhasePoint) = -ℓπ(point)
H₂(h::AbstractSystem, point::AbstractPhasePoint) =
    error("H₂(h, point) not implemented for $(typeof(h))")

∂H∂q(h::AbstractSystem, point::AbstractPhasePoint) = ∂H₁∂q(h, point) .+ ∂H₂∂q(h, point)
∂H₁∂q(h::AbstractSystem, point::AbstractPhasePoint) = -∇ℓπ(point)
∂H₂∂q(h::AbstractSystem, point::AbstractPhasePoint) =
    error("∂H₂∂q(h, point) not implemented for $(typeof(h))")

∂H∂p(h::AbstractSystem, point::AbstractPhasePoint) = ∂H₂∂p(h, point)
∂H₂∂p(h::AbstractSystem, point::AbstractPhasePoint) =
    error("∂H₂∂p(h, point) not implemented for $(typeof(h))")

sample_p(h::AbstractSystem, rng::AbstractRNG) =
    error("sample_p(h, point) not implemented for $(typeof(h))")



"""
    AbstractEuclideanSystem

Base abstract type for Euclidean Hamiltonian systems. All Euclidean systems require a 
constant positive definite matrix corresponding to the metric on the *unconstrained* 
position space. Positive definiteness is enforced by requiring metric to be of type 
`AbstractPDMat`.
"""
abstract type AbstractEuclideanSystem <: AbstractSystem end

metric(h::AbstractEuclideanSystem) = h.metric

"""
    EuclideanSystem

Composite type for an (Unconstrained) Euclidean System, with kinetric energy of the form
    H₂(q, p) = ½ pᵀ M⁻¹ p
where M is a constant positive definite matrix.
"""
struct EuclideanSystem{M<:AbstractPDMat} <: AbstractEuclideanSystem
    metric::M
end

H₂(h::EuclideanSystem, point::AbstractPhasePoint) = 0.5*invquad(metric(h), point.p)
∂H₂∂p(h::EuclideanSystem, point::AbstractPhasePoint) = metric(h) \ point.p
function sample_p(h::EuclideanSystem, rng::AbstractRNG)
    sqrt(metric(h)) * randn(rng, size(metric(h), 1))
end

# todo: implement cache
∂H₂∂q(h::EuclideanSystem, point::AbstractPhasePoint) = zeros(size(metric(h), 1))

"""
    AbstractRiemannianSystem

Base abstract type for Riemannian systems. 
"""
abstract type AbstractRiemannianSystem <: AbstractSystem end

