using AbstractMCMC
using LinearAlgebra: I

# Markov Chain Monte Carlo samplers. 
# Composite types and constructors to perform sampling of an unnormalized density. 

"""
    AbstractMiciSampler

Abstract supertype for Mici samplers. Subtypes from AbstractMCMC.AbstractSampler.
"""
abstract type AbstractMiciSampler <: AbstractMCMC.AbstractSampler end

"""
    HMCSampler <: AbstractMiciSampler

Hamiltonian Monte Carlo sampler with specified integrator, transitions, and metric.
Metric can be a symbolic spec (e.g., `:diag`) or a concrete `AbstractPDMat`.
"""
mutable struct HMCSampler{IN<:AbstractIntegrator, TI<:AbstractTransition, TM<:AbstractTransition} <: AbstractMiciSampler
    integrator::IN
    integration_transition::TI
    momentum_transition::TM
    metric::Union{Symbol, AbstractPDMat}
end

"""
    build_metric(metric_spec, d) -> AbstractPDMat

Construct the metric matrix for a `d`-dimensional sampler from a symbolic spec.
"""
function build_metric(metric_spec::Symbol, d::Integer)::AbstractPDMat
    if metric_spec === :diag
        return PDMat(Matrix{Float64}(I, d, d))
    else
        error("Unknown metric spec: $metric_spec")
    end
end

"""
    initialize_sampler!(sampler, d)

Ensure the sampler has a concrete metric matrix for a `d`-dimensional target.
"""
function initialize_sampler!(s::HMCSampler, d::Integer)
    s.metric = s.metric isa Symbol ? build_metric(s.metric, d) : s.metric
end

"""
    dimension(sampler) -> Integer

Return the dimensionality implied by the sampler's metric matrix.
"""
dimension(sampler::HMCSampler) = size(sampler.metric, 1)

####### 
####### Constructors
#######

"""
    MetropolisHMCSampler(integrator, momentum_transition=:IndependentMomentumTransition(), metric=:diag)

Construct a HMCSampler with a Metropolis accept/reject step.
"""
function MetropolisHMCSampler(
    integrator;
    momentum_transition = IndependentMomentumTransition(),
    metric::Union{Symbol,AbstractPDMat} = :diag
)
    HMCSampler(integrator, MetropolisTransition(), momentum_transition, metric)
end

####### 
####### Initialization
#######

"""
    sample_initial_state(sampler, logdensity)

Create a chain state initialized at zero position and momentum.

# Todo: Generalize to non-zero initializations.
"""
function sample_initial_state(sampler::HMCSampler, ℓπ)
    chainstate_from_ld(zeros(Float64, dimension(sampler)), zeros(Float64, dimension(sampler)), ℓπ)
end


