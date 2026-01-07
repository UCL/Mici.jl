using LogDensityProblems: logdensity_and_gradient

# Representation of phase points and chain states during sampling

#####
##### PhasePoint Representation
#####

abstract type AbstractPhasePoint end

struct ValueAndGrad{V,G}
    value::V
    gradient::G
end

"""
    PhasePoint{V<:AbstractVector, T} <: AbstractPhasePoint

A phase point in the Hamiltonian system, consisting of position `q`, momentum `p`, and log-density information `ℓπ`.
"""
mutable struct PhasePoint{V<:AbstractVector, T} <: AbstractPhasePoint
    q::V
    p::V
    ℓπ::T
end

"""
    ℓπ(point::PhasePoint)

Get the log-density value at the given `point`.
"""
ℓπ(point::PhasePoint) = point.ℓπ.value
"""
    ∇ℓπ(point::PhasePoint)

Get the gradient of the log-density at the given `point`.
"""
∇ℓπ(point::PhasePoint) = point.ℓπ.gradient


"""
    phasepoint_from_ld(q::V, p::V, ℓπ) where {V<:AbstractVector}

Create a `PhasePoint` from by evaluating log-density `ℓπ` at the given position `q`.
"""
function phasepoint_from_ld(q::V, p::V, ℓπ) where {V<:AbstractVector}
    val, grad = logdensity_and_gradient(ℓπ, q)
    lg = ValueAndGrad(val, grad)
    PhasePoint{V, typeof(lg)}(q, p, lg)
end

"""
    refresh_phasepoint!(point::PhasePoint, ℓπ)

Refresh the log-density and gradient information at the given `point`.
"""
function refresh_phasepoint!(point::PhasePoint, ℓπ)
    val, grad = logdensity_and_gradient(ℓπ, point.q)
    point.ℓπ = ValueAndGrad(val, grad)
    return point
end


#####
##### Chain State Representation
#####

abstract type AbstractChainState end


"""
    ChainState{V<:AbstractVector, T<:ValueAndGrad} <: AbstractChainState

A chain state during MCMC sampling, consisting of:
    - current phase point xᶜ
    - proposed phase point xᵖ
    - number of accepted proposals
"""
struct ChainState{V<:AbstractVector, T<:ValueAndGrad} <: AbstractChainState
    xᶜ::PhasePoint{V,T}             # current phasepoint
    xᵖ::PhasePoint{V,T}             # proposed phasepoint
    accepts::Base.RefValue{Int}     # number of accepted proposals
end

"""
    chainstate_from_ld(q::V, p::V, ℓπ) where {V<:AbstractVector}

Create a `ChainState` from position `q` and momentum `p` by evaluating the log-density `ℓπ`.
"""
function chainstate_from_ld(q::V, p::V, ℓπ) where {V<:AbstractVector}
    xᶜ = phasepoint_from_ld(q, p, ℓπ)
    xᵖ = phasepoint_from_ld(copy(q), copy(p), ℓπ)
    ChainState{V, typeof(xᶜ.ℓπ)}(xᶜ, xᵖ, Ref(0))
end

"""
    update_state!(state::ChainState, accepted::Bool, ℓπ)

Update the `state` based on whether the proposed move was `accepted`.
"""
function update_state!(state::ChainState, accepted::Bool, ℓπ)
    if accepted
        copyto!(state.qᶜ, state.qᵖ)
        copyto!(state.pᶜ, state.pᵖ)
        refresh_phasepoint!(state.xᶜ, ℓπ)
    else
        copyto!(state.qᵖ, state.qᶜ)
        copyto!(state.pᵖ, state.pᶜ)
        refresh_phasepoint!(state.xᵖ, ℓπ)
    end
    state.accepts[] += accepted
end

function Base.getproperty(s::ChainState, sym::Symbol)
    if sym === :qᶜ
        return s.xᶜ.q
    elseif sym === :pᶜ
        return s.xᶜ.p
    elseif sym === :qᵖ
        return s.xᵖ.q
    elseif sym === :pᵖ
        return s.xᵖ.p
    else
        return getfield(s, sym)
    end
end
