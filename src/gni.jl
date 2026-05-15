using GeometricIntegrators
using ..Mici: AbstractSystem, ChainState, h1_flow, h2_flow, ∂H₁∂q, ∂H₂∂p

#=
This is a module offering a thin wrapper to GeometricIntegrators.jl
https://github.com/JuliaGNI/GeometricIntegrators.jl

The purpose of this adaptor is to translate between the MCMC semantics
that are defined as part of the transitions and sampling steps.

The interface should do the following
=#



# TODO add some traits or capabilities to decorate this type
struct SeparableSystem <: AbstractSystem end

struct V1Field{S}
    system::S
    d::Int
end

function (f::V1Field)(v,t,q,params)
    # integer divison to get a int
    d = f.d
    f.z.q = @view q[begin:d]
    f.z.p = @view q[d+1:end]
    v[begin:d] .= 0
    v[d+1:end] .= ∂h₂∂p(f.z, f.system)
    return nothing
end

struct V2Field{S}
    system::S
    d::Int
end

function (f::V2Field)(v, t, x, params)
    d = f.d
    q = @view x[begin:d]

    v[begin:d] .= ∂h₁∂q(f.z, f.system)
    v[d+1:end] .= 0
    return nothing
end

function field_generator(system, initial_state)
    d = length(initial_state) ÷ 2
    return (V1Field(system, d), V2Field(system, d))
end

struct Q1Flow{S}
    system::S
    d::Int
    z::PhasePoint{Float64}
end

function Q1Flow(system::S, d::Integer) where {S<:AbstractTractableFlowSystem}
    z = PhasePoint(zeros(d), zeros(d), NaN, zeros(d), false)
    return Q1Flow{S}(system, Int(d), z)
end

function (f::Q1Flow)(x1, t1, x0, t0, params)
    d = f.d
    x1 .= x0
    f.z.q = @view(x1[begin:d])
    f.z.p = @view(x1[d+1:end])
    Φ₂!(f.z, f.system, t1 - t0)
    x1[begin:d] .= f.z.q
    return nothing
end

struct Q2Flow{S}
    system::S
    d::Int
    z::PhasePoint{Float64}
end

function Q2Flow(system::S, d::Integer) where {S<:AbstractTractableFlowSystem}
    z = PhasePoint(zeros(d), zeros(d), NaN, zeros(d), false)
    return Q2Flow{S}(system, Int(d), z)
end

function (f::Q2Flow)(x1, t1, x0, t0, params)
    d = f.d
    x1 .= x0
    f.z.q = @view(x1[begin:d])
    f.z.p = @view(x1[d+1:end])
    Φ₁!(f.z, f.system, t1 - t0)
    x1[d+1:end] .= f.z.p
    return nothing
end

function flow_generator(system, initial_state)
    d = length(initial_state) ÷ 2
    return (Q1Flow(system, d), Q2Flow(system, d))
end

function construct_split_ode_problem(system::AbstractSystem, initial_state::AbstractArray, timespan::Tuple, step_size::Real)
    @assert step_size > 0 "step_size must be greater than 0"

    vector_fields = field_generator(system, initial_state)
    subflows = flow_generator(system, initial_state)
    problem = SODEProblem(vector_fields, subflows, timespan, step_size, initial_state)
    return problem
end
