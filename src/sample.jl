using AbstractMCMC
using LinearAlgebra: I

abstract type AbstractMiciSampler <: AbstractMCMC.AbstractSampler end

mutable struct MetropolisHMCSampler{IN<:AbstractIntegrator, TI<:AbstractTransition, TM<:AbstractTransition} <: AbstractMiciSampler
    integrator::IN
    integration_transition::TI
    momentum_transition::TM
    metric::Union{Symbol, AbstractPDMat}
end

function MetropolisHMCSampler(
    integrator;
    integration_transition = MetropolisTransition(),
    momentum_transition = IndependentMomentumTransition(),
    metric::Union{Symbol,AbstractPDMat} = :diag
)
    MetropolisHMCSampler(integrator, integration_transition,
                         momentum_transition, metric)
end


function build_metric(metric_spec::Symbol, d::Integer)::AbstractPDMat
    if metric_spec === :diag
        return PDMat(Matrix{Float64}(I, d, d))
    else
        error("Unknown metric spec: $metric_spec")
    end
end

function initialize_sampler!(s::MetropolisHMCSampler, d::Integer)
    s.metric = s.metric isa Symbol ? build_metric(s.metric, d) : s.metric
end

function sample_initial_state(sampler::MetropolisHMCSampler, ℓπ)
    chainstate_from_ld(zeros(Float64, dimension(sampler)), zeros(Float64, dimension(sampler)), ℓπ)
end

dimension(sampler::MetropolisHMCSampler) = size(sampler.metric, 1)
