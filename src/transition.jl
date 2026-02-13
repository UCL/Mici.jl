abstract type AbstractTransition end

struct IndependentMomentumTransition <: AbstractTransition end

struct MetropolisTransition <: AbstractTransition end

struct CorrelatedMomentumTransition <: AbstractTransition
    resample_coefficient::Float64
    function CorrelatedMomentumTransition(resample_coefficient)
        @assert 0.0 ≤ resample_coefficient ≤ 1.0
        new(resample_coefficient)
    end
end


function transition(::IndependentMomentumTransition, h::EuclideanSystem, rng::AbstractRNG)
    sample_p(h, rng)
end

"""
This function represents a proposal generation along a full HMC trajectory
"""
function transition(::MetropolisTransition, integrator::I, h::S, state::C, rng::R) where {
    I<:AbstractIntegrator,
    S<:AbstractSystem,
    C<:AbstractChainState,
    R<:AbstractRNG
}
    current = MarkovChainState(q(state), p(state))
    proposed = MarkovChainState(copy(q(state)), copy(p(state)))

    integrate!(integrator, h, proposed)
    ΔH = H(h, current) - H(h, proposed)
    accepted = log(rand(rng)) < ΔH

    new_state = accepted ? proposed : current

    return new_state, accepted
end