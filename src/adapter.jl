abstract type AbstractAdapter end
abstract type AbstractAdaptationState end

mutable struct DualAveragingState{T} <: AbstractAdaptationState 
    running_average::T
end

struct DualAveragingAdapter{T} <: AbstractAdapter 
    initial_stepsize::T
    adaptation_state::DualAveragingState{T}
end

function DualAveragingAdapter(initial_stepsize::T, running_average::T) where {T}
    return DualAveragingAdapter(initial_stepsize, DualAveragingState(running_average))
end 


update_adapter_state(::AbstractAdapte, transition_stats)



# Helper functions:
"""
Computes the arithmetic mean of step sizes from their logarithms

# Arguments
- `log_step_sizes`: Logarithms of per-chain estimated step sizes.

# Returns
Arithmetic mean of the estimated step sizes.
"""
function arithmetic_mean_log_step_size_reducer(log_step_sizes::AbstractVector{<:Real})
    return sum(exp, log_step_sizes) / length(log_step_sizes)
end


"""
Computes the geometric mean of step sizes from their logarithms

# Arguments
- `log_step_sizes`: Logarithms of per-chain estimated step sizes.

# Returns
Geometric mean of the estimated step sizes.
"""
function geometric_mean_log_step_size_reducer(log_step_sizes::AbstractVector{<:Real})
    return exp(sum(log_step_sizes) / length(log_step_sizes))
end


"""
Computes the minimum of step sizes from their logarithms

# Arguments
- `log_step_sizes`: Logarithms of per-chain estimated step sizes.

# Returns
Minimum of the estimated step sizes.
"""
function min_log_step_size_reducer(log_step_sizes::AbstractVector{<:Real})
    return exp(minimum(log_step_sizes))
end


"""
Function to extract default statistic used for step-size adaptation.

# Arguments
- `stats`: Dictionary of transition statistics.

# Returns
The acceptance statistic.
"""
function default_adapt_stat_func(stats)
    return stats["accept_stat"]
end