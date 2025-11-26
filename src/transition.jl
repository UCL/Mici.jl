abstract type AbstractTransition end

struct IndependentMomentumTransition{S<:AbstractSystem} <: AbstractTransition
    h::S
end

struct CorrelatedMomentumTransition{S<:AbstractSystem} <: AbstractTransition
    h::S
    state::ChainState
    resample_coefficient::Float64
    function CorrelatedMomentumTransition(h, state, resample_coefficient)
        @assert 0.0 ≤ resample_coefficient ≤ 1.0
        new(h, state, resample_coefficient)
    end
end

sample(transition::IndependentMomentumTransition) = sample_p(transition.h)
