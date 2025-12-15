using AbstractMCMC

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::EuclideanSystem,
    sampler::MetropolisHMCSampler;
    kwargs...,
)
    state = sample_init_state(model, rng)
    AbstractMCMC.step(rng, model, sampler, state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::EuclideanSystem,
    sampler::MetropolisHMCSampler,
    state::AbstractChainState;
    kwargs...,
)

    p = transition(sampler.momentum_transition, model, rng)
    update_state!(state, p)

    new_state, accepted = transition(
        sampler.integration_transition,
        sampler.integrator,
        model,
        state,
        rng,
    )

    update_state!(state, new_state, accepted)

    return q(state), state
end