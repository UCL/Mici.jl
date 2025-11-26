abstract type AbstractChainState end

struct ChainState{M<:AbstractVector} <: AbstractChainState
    q::M
    p::M
end

q(state::ChainState) = state.q
p(state::ChainState) = state.p
