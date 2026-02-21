# Methods for AbstractMCMC integration with Mici sampling algorithms

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AbstractMiciSampler{S,I};
    initial_q=nothing,
    initial_ϵ=nothing,
    initial_metric=nothing,
    kwargs...,
) where {S<:AbstractSystem,I<:AbstractIntegrator}
    ℓπ = model.logdensity
    dimension = LogDensityProblems.dimension(ℓπ)
    metric = isnothing(initial_metric) ? ScalMat(dimension, 1.0) : initial_metric
    system = S(metric, ℓπ)
    phase_point = sample_initial_phase_point(rng, system, initial_q)
    integrator = I(initial_ϵ)
    state = state_type(sampler)(phase_point, system, integrator)
    return AbstractMCMC.step(rng, model, sampler, state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::HMC,
    state::AbstractState;
    trace_function=default_trace_function,
    kwargs...,
)
    transition!(state, rng, sampler.momentum_transition)
    transition_stats = transition!(state, rng, sampler.integration_transition)
    return (; traces=trace_function(state), statistics=transition_stats), state
end

function default_trace_function(state::AbstractState)
    (; q=state.phase_point.q)
end

function samples(sample::NamedTuple, N::Integer)
    NamedTuple(
        k => Array{eltype(v),ndims(v) + 1}(undef, (size(v)..., N)) for
        (k, v) in pairs(sample)
    )
end

function AbstractMCMC.samples(
    sample::NamedTuple,
    model::AbstractMCMC.LogDensityModel,
    sampler::HMC,
    N::Integer;
    kwargs...,
)
    return NamedTuple(key => samples(value, N) for (key, value) in pairs(sample))
end

function write_sample!(samples::NamedTuple, sample::NamedTuple, iteration::Integer)
    for (k, v) in pairs(sample)
        selectdim(samples[k], ndims(samples[k]), iteration) .= v
    end
end

function AbstractMCMC.save!!(
    samples::NamedTuple,
    sample::NamedTuple,
    iteration::Integer,
    model::AbstractMCMC.LogDensityModel,
    sampler::HMC,
    N::Integer;
    kwargs...,
)
    for (key, value) in pairs(sample)
        write_sample!(samples[key], value, iteration)
    end
    return samples
end

