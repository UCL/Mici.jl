abstract type AbstractChainState end

struct ChainState{V<:AbstractVector} <: AbstractChainState
    q::V
    p::V
end

q(state::ChainState) = state.q
p(state::ChainState) = state.p
