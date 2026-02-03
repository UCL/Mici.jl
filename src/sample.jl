using AbstractMCMC
using LogDensityProblems

# Markov Chain Monte Carlo samplers. 
# Composite types and constructors to perform sampling of an unnormalized density. 

"""
    AbstractMiciSampler

Abstract supertype for Mici samplers. Subtypes from AbstractMCMC.AbstractSampler.
"""
abstract type AbstractMiciConcreteSampler <: AbstractMCMC.AbstractSampler end

abstract type AbstractMiciSampler <: AbstractMCMC.AbstractSampler end

"""
    HMC <: AbstractMiciSampler

Hamiltonian Monte Carlo sampler with specified integrator, transitions, and metric.
"""
# Base.@kwdef struct HMC <: AbstractMiciSampler
#     ℋ::Symbol = :euclidean              # system type
#     metric::Symbol = :unit              # metric type
#     ℐ::Symbol = :leapfrog               # integrator type
#     𝒯::Symbol = :metropolis             # integration transition
#     ℳ::Symbol = :independent            # momentum transition
# end

struct HMC{S, I, T, M} <: AbstractMiciSampler
    system::S
    integrator::I
    integration_transition::T
    momentum_transition::M
end


"""
    sample_initial_state(sampler, logdensity)

Create a chain state initialized at zero position and momentum.

# Todo: Generalize to non-zero initializations.
"""
function sample_initial_state(sampler::HMC, ℓπ)

    metric = resolve_metric(sampler.metric, LogDensityProblems.dimension(ℓπ))
     system = resolve_system(sampler.system, sampler.metric)
     integrator = resolve_integrator(sampler.integrator, sampler.ϵ, sampler.T)
    chainstate_from_ld(zeros(Float64, LogDensityProblems.dimension(ℓπ)), zeros(Float64, LogDensityProblems.dimension(ℓπ)), ℓπ)
end
