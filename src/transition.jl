"""
    AbstractTransition

Abstract supertype for transitions in MCMC samplers. A transition represents a change in state 
of the Markov chain in either position or momentum.
"""
abstract type AbstractTransition end

"""
    AbstractIntegrationTransition <: AbstractTransition

Abstract supertype for integration transitions in MCMC samplers. Integration transitions
update the state according to the Hamiltonian dynamics of the system, typically using a numerical integrator.
"""
abstract type AbstractIntegrationTransition <: AbstractTransition end

"""
    AbstractMetropolisIntegrationTransition{T} <: AbstractIntegrationTransition

Abstract supertype for Metropolis-adjusted integration transitions in MCMC samplers, parameterized by:
    T -- type of the integration time (e.g., `Float64`)
"""
abstract type AbstractMetropolisIntegrationTransition{T} <: AbstractIntegrationTransition end

"""
    AbstractMomentumTransition <: AbstractTransition

Abstract supertype for momentum transitions in MCMC samplers. 
Momentum transitions update the momentum component of the state, typically by resampling from a distribution or applying a transformation.
"""
abstract type AbstractMomentumTransition <: AbstractTransition end

"""
    IndependentMomentumTransition <: AbstractMomentumTransition

Struct representing an independent momentum transition, where the momentum is resampled independently from a distribution 
defined by the system (e.g., a Gaussian distribution with covariance given by the metric of the system).
"""
struct IndependentMomentumTransition <: AbstractTransition end

""" 
    StaticMetropolisIntegrationTransition{T} <: AbstractMetropolisIntegrationTransition{T}

Struct representing a static Metropolis-adjusted integration transition, where the integration time is fixed and specified by the parameter `T`.
"""
struct StaticMetropolisIntegrationTransition{T} <:
       AbstractMetropolisIntegrationTransition{T}
    integration_time::T
end

"""
    metropolis_integration_transition!(state::MetropolisHMCState, rng::AbstractRNG, integration_time::Real)

Perform a Metropolis-adjusted integration transition
"""
function metropolis_integration_transition!(
    state::MetropolisHMCState, rng::AbstractRNG, integration_time::Real
)
    n_step = Int(integration_time ÷ state.integrator.ϵ)
    copy!(state.proposed_phase_point, state.phase_point)
    for s in 1:n_step
        step!(state.proposed_phase_point, state.integrator, state.system)
    end
    state.proposed_phase_point.p .*= -1
    Δh = h(state.phase_point, state.system) - h(state.proposed_phase_point, state.system)
    accept_probability = isnan(Δh) ? 0.0 : exp(min(0.0, Δh))
    accepted = rand(rng) < accept_probability
    if accepted
        copy!(state.phase_point, state.proposed_phase_point)
    end
    return (; accept_probability, accepted, n_step)
end

function transition!(
    state::MetropolisHMCState,
    rng::AbstractRNG,
    transition::StaticMetropolisIntegrationTransition,
)
    metropolis_integration_transition!(state, rng, transition.integration_time)
end

function transition!(
    state::AbstractState, rng::AbstractRNG, ::IndependentMomentumTransition
)
    rand!(rng, state.phase_point.p, state.phase_point, state.system)
    return nothing
end