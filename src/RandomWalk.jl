module RandomWalk

using AbstractMCMC
using Random
using Distributions
using LogDensityProblems

# Model
struct DistributionModel <: AbstractMCMC.AbstractModel
    dist::Distribution
end

LogDensityProblems.logdensity(model::DistributionModel, x) = 
    logpdf(model.dist, x)

# LogDensityProblems.dimension(model::DistributionModel) = length(mean(model.dist))
LogDensityProblems.dimension(model::DistributionModel) =
    isa(mean(model.dist), Number) ? 1 : length(mean(model.dist))


# Sampler
struct RWSampler{T} <: AbstractMCMC.AbstractSampler 
    stepsize::T
end

# Step
# Initial state
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DistributionModel,
    sampler::RWSampler;
    kwargs...
)
    d = LogDensityProblems.dimension(model)

    init_state = zeros(d)

    return AbstractMCMC.step(rng, model, sampler, init_state)
end

# Ignore weights
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DistributionModel,
    sampler::RWSampler,
    weight::Float64;
    kwargs...
    )

    return AbstractMCMC.step(rng, model, sampler; kwargs...)
end

# Random Walk Metropolis 
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DistributionModel,
    sampler::RWSampler,
    state::S
    ) where {S}

    proposed_state = state .+ sampler.stepsize .* randn(rng, length(state))
    
    logp_current = LogDensityProblems.logdensity(model, state)
    logp_proposed = LogDensityProblems.logdensity(model, proposed_state)
    
    log_accept_ratio = logp_proposed - logp_current
    
    if log(rand(rng)) < log_accept_ratio
        new_state = proposed_state
    else
        new_state = state
    end
    
    return new_state, new_state
end

# Sampling Interface
function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DistributionModel,
    sampler::RWSampler,
    n_samples::Integer,
    kwargs...
    )

    d = LogDensityProblems.dimension(model)
    samples = Matrix{Float64}(undef, n_samples, d)

    # First iteration
    state, _ = AbstractMCMC.step(rng, model, sampler)
    samples[1, :] = state

    # Remaining iterations
    for i in 2:n_samples
        state, _ = AbstractMCMC.step(rng, model, sampler, state)
        samples[i, :] = state
    end

    return samples
end
end