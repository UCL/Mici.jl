using AbstractMCMC

struct MiciSampler{I<:AbstractIntegrator, S<:AbstractChainState} <: AbstractMCMC.AbstractSampler
    integrator::I
    state::S
end

function MiciSampler(I::AbstractIntegrator, q::AbstractVector)
    return MiciSampler(I, ChainState(q))
end

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