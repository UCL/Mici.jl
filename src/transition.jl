abstract type AbstractTransition end
abstract type AbstractIntegrationTransition <: AbstractTransition end
abstract type AbstractMetropolisIntegrationTransition{T} <: AbstractIntegrationTransition end
abstract type AbstractMomentumTransition <: AbstractTransition end

struct IndependentMomentumTransition <: AbstractTransition end

struct StaticMetropolisIntegrationTransition{T} <:
       AbstractMetropolisIntegrationTransition{T}
    integration_time::T
end

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