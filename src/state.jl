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
    accepts::Integer
end

function ChainState(current_state::M) where {M<:MarkovChainState}
    ChainState(current_state, 0)
end

function ChainState(q::V) where {V<:AbstractVector}
    ChainState(MarkovChainState(q, zeros(eltype(q), size(q))), 0)
end

function update_state!(state::ChainState, new_state::MarkovChainState, accepted::Bool)
    state.current_state.q = new_state.q
    state.current_state.p = new_state.p
    state.accepts += accepted
end


