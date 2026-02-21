using AbstractMCMC
using LogDensityProblems

"""
    AbstractMiciSampler{S, I} <: AbstractMCMC.AbstractSampler

Abstract supertype for Mici samplers, parameterized by the system type `S` and integrator type `I`.
"""
abstract type AbstractMiciSampler{S, I} <: AbstractMCMC.AbstractSampler end

"""
    HMC{S,I,TI,TM} <: AbstractMiciSampler{S,I}

Struct representing a Hamiltonian Monte Carlo sampler, parameterized by:
    S  -- type of the system (e.g., `EuclideanSystem`)
    I  -- type of the integrator (e.g., `LeapfrogIntegrator`)
    TI -- type of the integration transition (e.g., `StaticMetropolisIntegrationTransition`)
    TM -- type of the momentum transition (e.g., `IndependentMomentumTransition`)
"""
struct HMC{S,I,TI,TM} <: AbstractMiciSampler{S,I}
    integration_transition::TI
    momentum_transition::TM
end

function HMC{S,I}(integration_time::Real) where {S,I}
    HMC{S,I}(StaticMetropolisIntegrationTransition(integration_time))
end

function HMC{S,I}(integration_transition::TI, momentum_transition::TM=IndependentMomentumTransition()) where {S,I,TI,TM}
    HMC{S,I,TI,TM}(integration_transition, momentum_transition)
end

function HMC{S,I}(integration_time_lower::Real, integration_time_upper::Real) where {S,I}
    HMC{S,I}(
        RandomMetropolisIntegrationTransition(
            integration_time_lower, integration_time_upper
        ),
    )
end

const EuclideanHMC{I,TI,TM} = HMC{EuclideanSystem,I,TI,TM}

function EuclideanHMC(integration_time::Real)
    EuclideanHMC{LeapfrogIntegrator}(StaticMetropolisIntegrationTransition(integration_time))
end

function EuclideanHMC(integration_time_lower::Real, integration_time_upper::Real)
    EuclideanHMC{LeapfrogIntegrator}(integration_time_lower, integration_time_upper)
end

function state_type(
    ::HMC{S,I,TI,TM}
) where {S,I,TI<:AbstractMetropolisIntegrationTransition,TM}
    MetropolisHMCState
end
