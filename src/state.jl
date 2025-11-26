abstract type AbstractChainState end

struct ChainState{V<:AbstractVector} <: AbstractChainState
    q::V
    p::V
end

q(state::AbstractChainState) = state.q
p(state::AbstractChainState) = state.p
