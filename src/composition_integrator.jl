abstract type AbstractCompositionMethod end

struct StrangA <: AbstractCompositionMethod end
struct StrangB <: AbstractCompositionMethod end
struct McLachlan2 <: AbstractCompositionMethod end
struct McLachlan4 <: AbstractCompositionMethod end
struct TripleJump <: AbstractCompositionMethod end
struct SuzukiFractal <: AbstractCompositionMethod end

order(::Union{StrangA, Type{StrangA}}) = 2
order(::Union{StrangB, Type{StrangB}}) = 2
order(::Union{McLachlan2, Type{McLachlan2}}) = 2
order(::Union{McLachlan4, Type{McLachlan4}}) = 4
order(::Union{TripleJump, Type{TripleJump}}) = 4
order(::Union{SuzukiFractal, Type{SuzukiFractal}}) = 4

function _coefficients_general(r::Int, a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    @assert length(a) == length(b)
    s = length(a)
    f = Vector{Int}(undef, 2r * s)
    c = Vector{T}(undef, 2r * s)

    for i in 1:s
        for j in 1:r
            f[(2i - 2) * r + j] = j
            c[(2i - 2) * r + j] = a[i]
        end
        for j in 1:r
            f[(2i - 1) * r + j] = r - j + 1
            c[(2i - 1) * r + j] = b[i]
        end
    end

    return Tuple(f), Tuple(c)
end

function _coefficients_symmetric_stages(r::Int, a::AbstractVector{T}) where {T}
    stages = vcat(a, a[end - 1:-1:1]) ./ 2
    s = length(stages)
    f = Vector{Int}(undef, 2r * s)
    c = Vector{T}(undef, 2r * s)

    for i in 1:s
        for j in 1:r
            f[(2i - 2) * r + j] = j
            c[(2i - 2) * r + j] = stages[i]
            f[(2i - 1) * r + j] = r - j + 1
            c[(2i - 1) * r + j] = stages[i]
        end
    end

    return Tuple(f), Tuple(c)
end

function coefficients(::StrangA, ::Type{T}=Float64) where {T}
    return (1, 2, 1), (T(1) / 2, one(T), T(1) / 2)
end

function coefficients(::StrangB, ::Type{T}=Float64) where {T}
    return (2, 1, 2), (T(1) / 2, one(T), T(1) / 2)
end

function coefficients(::McLachlan2, ::Type{T}=Float64; α=0.1932) where {T}
    a = T[α, T(1) / 2 - α]
    b = T[T(1) / 2 - α, α]
    return _coefficients_general(2, a, b)
end

function coefficients(::McLachlan4, ::Type{T}=Float64) where {T}
    s19 = sqrt(T(19))
    a = T[
        (T(146) + T(5) * s19) / T(540),
        (-T(2) + T(10) * s19) / T(135),
        one(T) / T(5),
        (-T(23) - T(20) * s19) / T(270),
        (T(14) - s19) / T(108),
    ]
    return _coefficients_general(2, a, reverse(a))
end

function coefficients(::TripleJump, ::Type{T}=Float64) where {T}
    fac = T(2)^(one(T) / T(3))
    den = inv(T(2) - fac)
    a = T[den, -fac * den]
    return _coefficients_symmetric_stages(2, a)
end

function coefficients(::SuzukiFractal, ::Type{T}=Float64) where {T}
    fac = T(4)^(one(T) / T(3))
    den = inv(T(4) - fac)
    a = T[den, den, -fac * den]
    return _coefficients_symmetric_stages(2, a)
end

struct CompositionIntegrator{M, T, F, C} <: AbstractIntegrator
    ϵ::T
    f::F
    c::C
end

function CompositionIntegrator{M}(; step_size=0.1, kwargs...) where {M<:AbstractCompositionMethod}
    ϵ = isnothing(step_size) ? 0.1 : step_size
    f, c = coefficients(M(), typeof(float(ϵ)))
    return CompositionIntegrator{M, typeof(ϵ), typeof(f), typeof(c)}(ϵ, f, c)
end

function CompositionIntegrator(; method::M=StrangA(), step_size=0.1, kwargs...) where {M<:AbstractCompositionMethod}
    return CompositionIntegrator{M}(; step_size=step_size, kwargs...)
end

step_size(integrator::CompositionIntegrator) = integrator.ϵ
method(::CompositionIntegrator{M}) where {M} = M()

@inline function _apply_split_flow!(z::PhasePoint, system::AbstractTractableFlowSystem, flow::Int, ϵ::Real)
    if flow == 1
        Φ₁!(z, system, ϵ)
    elseif flow == 2
        Φ₂!(z, system, ϵ)
    else
        throw(ArgumentError("Unsupported flow index $flow"))
    end
    return nothing
end

function step!(
    z::PhasePoint,
    integrator::CompositionIntegrator,
    system::AbstractTractableFlowSystem,
)
    @inbounds for i in eachindex(integrator.f)
        _apply_split_flow!(z, system, integrator.f[i], integrator.c[i] * integrator.ϵ)
    end
    return nothing
end
