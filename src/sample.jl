"""
    AbstractMiciSampler{S, I} <: AbstractMCMC.AbstractSampler

Abstract supertype for Mici samplers, parameterized by the system type `S` and integrator type `I`.
"""
abstract type AbstractMiciSampler{S, I, A} <: AbstractMCMC.AbstractSampler end

"""
    HMC{S,I,A,TI,TM} <: AbstractMiciSampler{S,I,A}

Struct representing a Hamiltonian Monte Carlo sampler, parameterized by:
  - `S`  - type of the system (e.g., `EuclideanSystem`),
  - `I`  - type of the integrator (e.g., `LeapfrogIntegrator`),
  - `A`  - type of the adaptation strategy (e.g., `DualAveraging`),
  - `TI` - type of the integration transition (e.g., `StaticMetropolisIntegrationTransition`),
  - `TM` - type of the momentum transition (e.g., `IndependentMomentumTransition`).
"""
struct HMC{S,I,A,TI,TM} <: AbstractMiciSampler{S,I,A}
    integration_transition::TI
    momentum_transition::TM
end

function HMC{S,I,A}(integration_time::Real) where {S,I,A}
    HMC{S,I,A}(StaticMetropolisIntegrationTransition(integration_time))
end

function HMC{S,I,A}(integration_transition::TI, momentum_transition::TM=IndependentMomentumTransition()) where {S,I,A,TI,TM}
    HMC{S,I,A,TI,TM}(integration_transition, momentum_transition)
end

function HMC{S,I,A}(integration_time_lower::Real, integration_time_upper::Real) where {S,I,A}
    HMC{S,I,A}(
        RandomMetropolisIntegrationTransition(
            integration_time_lower, integration_time_upper
        ),
    )
end

const EuclideanHMC{I,A,TI,TM} = HMC{EuclideanSystem,I,A,TI,TM}

function EuclideanHMC(integration_time::Real)
    EuclideanHMC{LeapfrogIntegrator, DualAveragingAdapter}(StaticMetropolisIntegrationTransition(integration_time))
end

function EuclideanHMC(integration_time_lower::Real, integration_time_upper::Real)
    EuclideanHMC{LeapfrogIntegrator, DualAveragingAdapter}(integration_time_lower, integration_time_upper)
end

function state_type(
    ::HMC{S,I,A,TI,TM}
) where {S,I,TI<:AbstractMetropolisIntegrationTransition,TM}
    MetropolisHMCState
end
