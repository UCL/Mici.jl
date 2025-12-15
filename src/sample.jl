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

