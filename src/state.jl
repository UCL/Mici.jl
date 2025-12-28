abstract type AbstractChainState end

q(state::AbstractChainState) =
    error("q(state) not implemented for $(typeof(state))")

p(state::AbstractChainState) =
    error("p(state) not implemented for $(typeof(state))")


"""
    MarkovChainState

Instantaneous state of position and momentum
"""
mutable struct MarkovChainState{V<:AbstractVector} <: AbstractChainState
    q::V
    p::V
end

q(state::MarkovChainState) = state.q
p(state::MarkovChainState) = state.p


"""
    NonMarkovChainState

Base abstract type representing non Markovian chain states. That is,
chain states containing history of trajectory and related information.
"""

abstract type NonMarkovChainState <: AbstractChainState end

q(state::NonMarkovChainState) = q(state.current_state)
p(state::NonMarkovChainState) = p(state.current_state)

mutable struct ChainState{M<:MarkovChainState} <: NonMarkovChainState
    current_state::M
    proposed_state::M
    accepts::Base.RefValue{Int}
end

function ChainState(current_state::M) where {M<:MarkovChainState}
    ChainState(current_state, current_state, Ref(0))
end

function update_state!(state::ChainState, accepted::Bool)
    if accepted
        copyto!(state.current_state.q, state.proposed_state.q)
        copyto!(state.current_state.p, state.proposed_state.p)
    else
        copyto!(state.proposed_state.q, state.current_state.q)
        copyto!(state.proposed_state.p, state.current_state.p)
    end
    state.accepts[] += accepted
end


