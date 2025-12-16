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


function transition!(::IndependentMomentumTransition, h::EuclideanSystem, state::C, rng::AbstractRNG) where {C<:AbstractChainState}
    sample_p!(h, state, rng)
end

function transition!(::MetropolisTransition, integrator::I, h::S, state::C, rng::R) where {
    I<:AbstractIntegrator,
    S<:AbstractSystem,
    C<:AbstractChainState,
    R<:AbstractRNG
}
    current = state.current_state
    proposed = MarkovChainState(copy(q(state)), copy(p(state)))

    integrate!(integrator, h, proposed)
    ΔH = H(h, current) - H(h, proposed)
    accepted = log(rand(rng)) < ΔH

    new_state = accepted ? proposed : current

    update_state!(state, new_state, accepted)
end