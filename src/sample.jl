using AbstractMCMC
using LogDensityProblems

abstract type AbstractMiciSampler{S, I} <: AbstractMCMC.AbstractSampler end

struct HMC{S,I,TI,TM} <: AbstractMiciSampler{S,I}
    integration_transition::TI
    momentum_transition::TM
end

function HMC{S,I}(integration_transition::TI, momentum_transition::TM=IndependentMomentumTransition()) where {S,I,TI,TM}
    HMC{S,I,TI,TM}(integration_transition, momentum_transition)
end

function HMC{S,I}(integration_time::Real) where {S,I}
    HMC{S,I}(StaticMetropolisIntegrationTransition(integration_time))
end

const EuclideanHMC{I,TI,TM} = HMC{EuclideanSystem,I,TI,TM}

function EuclideanHMC(integration_time::Real)
    EuclideanHMC{LeapfrogIntegrator}(StaticMetropolisIntegrationTransition(integration_time))
end

function state_type(
    ::HMC{S,I,TI,TM}
) where {S,I,TI<:AbstractMetropolisIntegrationTransition,TM}
    MetropolisHMCState
end
