abstract type AbstractChainState end

q(state::AbstractChainState) =
    error("q(state) not implemented for $(typeof(state))")

p(state::AbstractChainState) =
    error("p(state) not implemented for $(typeof(state))")


"""
    MarkovChainState

Instantaneous state of position and momentum
"""
struct MarkovChainState{V<:AbstractVector} <: AbstractChainState
    q::V
    p::V
end

q(state::MarkovChainState) = state.q
p(state::MarkovChainState) = state.p

function update_state!(state::MarkovChainState{V}, p::V) where {V<:AbstractVector}
    state.p .= p
end

function update_state!(state::MarkovChainState{V}, q::V, p::V) where {V<:AbstractVector}
    state.q .= q
    state.p .= p
end

"""
    NonMarkovChainState

Base abstract type representing non Markovian chain states. That is,
chain states containing history of trajectory and related information.
"""

abstract type NonMarkovChainState <: AbstractChainState end

q(state::NonMarkovChainState) = q(state.current_state)
p(state::NonMarkovChainState) = p(state.current_state)

struct ChainState{M<:MarkovChainState} <: NonMarkovChainState
    current_state::M
    accepts::Base.RefValue{Int}
end

function ChainState(current_state::M) where {M<:MarkovChainState}
    ChainState(current_state, Ref(0))
end

function ChainState(q::V) where {V<:AbstractVector}
    ChainState(MarkovChainState(q, zeros(eltype(q), size(q))), Ref(0))
end

function update_state!(state::ChainState, p::V) where {V<:AbstractVector}
    update_state!(state.current_state, p)
end

function update_state!(state::ChainState, new_state::MarkovChainState, accepted::Bool)
    update_state!(state.current_state, q(new_state), p(new_state))
    state.accepts[] += accepted
end


