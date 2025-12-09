abstract type AbstractChainState end

q(state::AbstractChainState) =
    error("q(state) not implemented for $(typeof(state))")

p(state::AbstractChainState) =
    error("p(state) not implemented for $(typeof(state))")

update_state!(state::AbstractChainState) = 
    error("update_state!(state) not implemented for $(typeof(state))")

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

function update_state!(state::MarkovChainState, q::AbstractVector, p::AbstractVector; kwargs...)
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

function ChainState(M::MarkovChainState) 
    return ChainState(M, Ref(0))
end

function ChainState(q::AbstractVector)
    return ChainState(MarkovChainState(q, zeros(eltype(q), size(q))), Ref(0))
end

function update_state!(state::ChainState, q::AbstractVector, p::AbstractVector, accepted::Bool; kwargs...)
    update_state!(state.current_state, q, p; kwargs...)
    state.accepts[] += accepted
end

