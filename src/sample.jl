using AbstractMCMC

abstract type AbstractMCMCSampler <: AbstractMCMC.AbstractSampler end

abstract type AbstractHMCSampler <: AbstractMCMCSampler end

struct MetropolisHMCSampler{I<:AbstractIntegrator, TI<:AbstractTransition, TM<:AbstractTransition} <: AbstractHMCSampler
    integrator::I
    integration_transition::TI
    momentum_transition::TM
end

function MetropolisHMCSampler(
    integrator,
    integration_transition
)
    MetropolisHMCSampler(integrator, integration_transition, IndependentMomentumTransition())
end

function sample_init_state(h::H, rng::AbstractRNG) where {H<:AbstractSystem}
    M = MarkovChainState(zeros(Float64, size(h.metric, 1)), zeros(Float64, size(h.metric, 1)))
    return ChainState(M)
end


# function sample(
#     sampler::MetropolisHMCSampler,
#     state::AbstractChainState,
#     rng::AbstractRNG,
# )
#     q₁ = q(state)
#     p₁ = sample_p(h, rng)
#     current_state = MarkovChainState(q₁, p₁)
#     proposed_state = MarkovChainState(copy(q₁), copy(p₁))

#     integrate!(sampler.integrator, proposed_state)

#     accept_prob = exp(H(h, current_state) - H(h, proposed_state))
#     accepted = rand(rng) < accept_prob
#     new_state = accepted ? proposed_state : current_state

#     update_state!(chain_state, q(new_state), p(new_state), accepted)

#     return q(new_state), chain_state
# end
