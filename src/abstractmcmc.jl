using AbstractMCMC
using LogDensityProblems

# Mici sampler interface for AbstractMCMC.jl

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
    initialize_sampler!(sampler, LogDensityProblems.dimension(ℓπ))
    state = sample_initial_state(sampler, ℓπ)
    AbstractMCMC.step(rng, model, sampler, state)
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

    # Todo: generalize to generic systems implied by the sampler and model
    system = EuclideanSystem(sampler.metric)

    # Perform a momentum refreshment
    transition!(sampler.momentum_transition, system, state, rng)

    # Perform integration and Metropolis update
    transition!(
        sampler.integration_transition,
        sampler.integrator,
        system,
        state,
        rng,
        ℓπ,
    )

    return copy(state.qᶜ), state
end
