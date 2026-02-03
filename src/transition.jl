# Transitions between states

"""  
    AbstractTransition

Abstract supertype for transitions between states.
"""
abstract type AbstractTransition end

"""
    IndependentMomentumTransition <: AbstractTransition

Resample momentum independently from the target distribution.
"""
struct IndependentMomentumTransition <: AbstractTransition end

"""
    MetropolisTransition <: AbstractTransition

Perform a Metropolis accept/reject step.
"""
struct MetropolisTransition <: AbstractTransition end

struct CorrelatedMomentumTransition <: AbstractTransition
    resample_coefficient::Float64
    function CorrelatedMomentumTransition(resample_coefficient)
        @assert 0.0 ≤ resample_coefficient ≤ 1.0
        new(resample_coefficient)
    end
end

"""
    transition!(transition::AbstractTransition, system::AbstractSystem, state::ChainState, rng::AbstractRNG)

Refresh the momentum in the given `state` independently from the target distribution.
"""
function transition!(state::ChainState, rng::AbstractRNG, ::IndependentMomentumTransition, system::EuclideanSystem)
    state.pᶜ .= sample_p(system, rng)
    state.pᵖ .= state.pᶜ
end

"""
    transition!(::MetropolisTransition,
                integrator::AbstractIntegrator,
                system::AbstractSystem,
                state::ChainState,
                rng::AbstractRNG,
                ℓπ)

Update the state with a Metropolis accept/reject step.
"""
function transition!(
    state::ChainState,
    rng::AbstractRNG,
    ::MetropolisTransition,
    integrator::AbstractIntegrator,
    system::AbstractSystem,
    ℓπ)

    integrate!(integrator, system, state.xᵖ, ℓπ)
    ΔH = H(system, state.xᶜ) - H(system, state.xᵖ)
    accepted = log(rand(rng)) < ΔH

    update_state!(state, accepted, ℓπ)
end
