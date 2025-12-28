using AbstractMCMC

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::EuclideanSystem,
    sampler::MetropolisHMCSampler;
)
    state = sample_init_state(model, rng)
    AbstractMCMC.step(rng, model, sampler, state)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::EuclideanSystem,
    sampler::MetropolisHMCSampler,
    state::AbstractChainState;
)

    transition!(sampler.momentum_transition, model, state, rng)

    transition!(
        sampler.integration_transition,
        sampler.integrator,
        model,
        state,
        rng,
    )

    return state.current_state.q, state
end