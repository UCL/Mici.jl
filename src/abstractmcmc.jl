using AbstractMCMC
using LogDensityProblems

# Mici sampler interface for AbstractMCMC.jl

"""
    AbstractMCMC.step(rng, model, sampler::AbstractMiciSampleSpec, state)

Allow initialization of sampler from specification before stepping.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AbstractMiciSampler;
)
    ℓπ = model.logdensity
    state = sample_initial_state(sampler, ℓπ)

    metric = resolve_metric(sampler.metric, LogDensityProblems.dimension(ℓπ))
    system = resolve_system(sampler.system, sampler.metric)
    integrator = resolve_integrator(spec.integrator, spec.ϵ, spec.T)

    sampler = instantiate(sampler, model.logdensity)
    copy(state.qᶜ), state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.LogDensityModel, sampler::AbstractMiciSampler)

Perform a single MCMC step using the provided `sampler` on the `model`, starting from an initial state sampled from the model.
Returns a tuple containing the new sample and the updated chain state.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AbstractMiciSampler;
)
    ℓπ = model.logdensity
    state = sample_initial_state(sampler, ℓπ)
    copy(state.qᶜ), state
end


"""
    AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.LogDensityModel, sampler::AbstractMiciSampler, state::ChainState)

Perform a single MCMC step using the provided `sampler` on the `model`, starting from the given `state`.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AbstractMiciSampler,
    state::ChainState;
)
    ℓπ = model.logdensity
    ℳ = resolve(sampler.momentum_transition)
    𝒯 = resolve(sampler.integration_transition)
    ℋ = state.ℋ
    ℐ = state.ℐ

    transition!(state, rng, ℳ, ℋ)

    transition!(state, rng, 𝒯, ℐ, ℋ, ℓπ)

    return copy(state.qᶜ), state
end
