using AbstractMCMC

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractSystem,
    sampler::MiciSampler;
    kwargs...,
)
    AbstractMCMC.step(rng, model, sampler, sampler.state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractSystem,
    sampler::MiciSampler,
    state::AbstractChainState;
    kwargs...,
)
    return sample(model, sampler.integrator, state, rng)
end