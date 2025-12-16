using AbstractMCMC

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::EuclideanSystem,
    sampler::MetropolisHMCSampler;
    kwargs...,
)
    state = sample_init_state(model, rng)
    sampler.state = state
    AbstractMCMC.step(rng, model, sampler, sampler.state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::EuclideanSystem,
    sampler::MetropolisHMCSampler,
    state::AbstractChainState;
    kwargs...,
)

    transition!(sampler.momentum_transition, model, sampler.state, rng)

    transition!(
        sampler.integration_transition,
        sampler.integrator,
        model,
        sampler.state,
        rng,
    )

    return sampler.state.current_state.q, state
end