using AbstractMCMC

struct MiciModel{H<:AbstractSystem} <: AbstractMCMC.model
    H::H
end

struct MiciSampler{I<:AbstractIntegrator, S<:AbstractChainState} <: AbstractMCMC.AbstractSampler
    integrator::I
    state::S
end

# sample, state = AbstractMCMC.step(rng, model, sampler; kwargs...)
# sample = exposed to user
# state = used for tracking information

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::MiciModel,
    sampler::MiciSampler;
    kwargs...,
)
    AbstractMCMC.step(rng, model, sampler, sampler.state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::MiciModel,
    sampler::MiciSampler,
    state::AbstractChainState;
    kwargs...,
)
    return sample(model.H, sampler.integrator, state, rng)
end