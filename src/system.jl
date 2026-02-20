"""
    AbstractSystem

Base abstract type for Hamiltonian systems with energy
    H(q, p) = H₁(q) + H₂(q, p)
where
    q   --  current position
    p   --  current momentum
    H₁  --  energy term depending on state only
    H₂  --  energy term depending on momentum and state (optionally)

In a standard Euclidean System, H₁ and H₂ correspond to potential energy and
kinetic energy respectively. However, solving the Hamiltonian dynamics in more
complex systems may benefit from a more flexible distinction between (state) and
(momentum, state) energy components.

All abstract systems must have fields:
    neg_log_dens        --  function to calculate negative log density of the
                            target function at the current position
    grad_neg_log_dens   --  function to calculate gradient of the negative log density of 
                            the target function at the current position
"""
abstract type AbstractSystem end

H(h::AbstractSystem, state::ChainState) = H₁(h, state) + H₂(h, state)
H₁(h::AbstractSystem, state::ChainState) = h.neg_log_dens(q(state))
H₂(h::AbstractSystem, state::ChainState) =
    error("H₂(h, state) not implemented for $(typeof(h))")

∂H∂q(h::AbstractSystem, state::ChainState) = ∂H₁∂q(h, state) .+ ∂H₂∂q(h, state)
∂H₁∂q(h::AbstractSystem, state::ChainState) = h.grad_neg_log_dens(q(state))
∂H₂∂q(h::AbstractSystem, state::ChainState) =
    error("∂H₂∂q(h, state) not implemented for $(typeof(h))")

∂H∂p(h::AbstractSystem, state::ChainState) = ∂H₂∂p(h, state)
∂H₂∂p(h::AbstractSystem, state::ChainState) =
    error("∂H₂∂p(h, state) not implemented for $(typeof(h))")

h1_flow(h::AbstractSystem, state::ChainState, Δt::Real) = p(state) .-= Δt .* ∂H₁∂q(h, state)
# TODO Too much to put at this level?
h2_flow(h::AbstractSystem, state::ChainState, Δt::Real) = q(state) .+= Δt .* ∂H₂∂p(h, state)
sample_p(h::AbstractSystem, rng::AbstractRNG) =
    error("sample_p(h, state) not implemented for $(typeof(h))")



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
struct EuclideanSystem{F1, F2, M<:AbstractPDMat} <: AbstractEuclideanSystem
    neg_log_dens::F1
    grad_neg_log_dens::F2
    metric::M
end

H₂(h::EuclideanSystem, state::ChainState) = 0.5*invquad(metric(h), p(state))
∂H₂∂p(h::EuclideanSystem, state::ChainState) = metric(h) \ p(state)
sample_p(h::EuclideanSystem, rng::AbstractRNG) =
    sqrt(metric(h)) * randn(rng, size(metric(h), 1))

# todo: implement cache
∂H₂∂q(h::EuclideanSystem, state::ChainState) = zeros(size(metric(h), 1))

"""
    AbstractRiemannianSystem

Base abstract type for Riemannian systems. 
"""
abstract type AbstractRiemannianSystem <: AbstractSystem end
