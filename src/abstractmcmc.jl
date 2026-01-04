using AbstractMCMC
using LogDensityProblems

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

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AbstractMiciSampler,
    state::ChainState;
)
    ℓπ = model.logdensity

    system = EuclideanSystem(sampler.metric)

    transition!(sampler.momentum_transition, system, state, rng)

    transition!(
        sampler.integration_transition,
        sampler.integrator,
        system,
        state,
        rng,
        ℓπ,
    )

    return copy(state.q), state
end
