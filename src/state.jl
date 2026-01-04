using LogDensityProblems: logdensity_and_gradient

abstract type AbstractPhasePoint end

struct ValueAndGrad{V,G}
    value::V
    gradient::G
end

mutable struct PhasePoint{V<:AbstractVector, T} <: AbstractPhasePoint
    q::V
    p::V
    ℓπ::T
end

function phasepoint_from_ld(q::V, p::V, ℓπ) where {V<:AbstractVector}
    val, grad = logdensity_and_gradient(ℓπ, q)
    lg = ValueAndGrad(val, grad)
    PhasePoint{V, typeof(lg)}(q, p, lg)
end

function refresh_phasepoint!(point::PhasePoint, ℓπ)
    val, grad = logdensity_and_gradient(ℓπ, point.q)
    point.ℓπ = ValueAndGrad(val, grad)
    return point
end

ℓπ(point::PhasePoint) = point.ℓπ.value
∇ℓπ(point::PhasePoint) = point.ℓπ.gradient

abstract type AbstractChainState end

struct ChainState{V<:AbstractVector, T<:ValueAndGrad} <: AbstractChainState
    current::PhasePoint{V,T}
    proposal::PhasePoint{V,T}
    accepts::Base.RefValue{Int}
end

function chainstate_from_ld(q::V, p::V, ℓπ) where {V<:AbstractVector}
    current = phasepoint_from_ld(q, p, ℓπ)
    proposal = phasepoint_from_ld(copy(q), copy(p), ℓπ)
    ChainState{V, typeof(current.ℓπ)}(current, proposal, Ref(0))
end

function update_state!(state::ChainState, accepted::Bool, ℓπ)
    if accepted
        copyto!(state.q, state.q_prop)
        copyto!(state.p, state.p_prop)
        refresh_phasepoint!(state.current, ℓπ)
    else
        copyto!(state.q_prop, state.q)
        copyto!(state.p_prop, state.p)
        refresh_phasepoint!(state.proposal, ℓπ)
    end
    state.accepts[] += accepted
end

function Base.getproperty(s::ChainState, sym::Symbol)
    if sym === :q
        return s.current.q
    elseif sym === :p
        return s.current.p
    elseif sym === :q_prop
        return s.proposal.q
    elseif sym === :p_prop
        return s.proposal.p
    else
        return getfield(s, sym)
    end
end
