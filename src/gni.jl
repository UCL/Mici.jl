module Gni
using GeometricIntegrators
using ..Mici: AbstractSystem, ChainState, h1_flow, h2_flow, ∂H₁∂q, ∂H₂∂p

#=
This is a module offering a thin wrapper to GeometricIntegrators.jl
https://github.com/JuliaGNI/GeometricIntegrators.jl

The purpose of this adaptor is to translate between the MCMC semantics
that are defined as part of the transitions and sampling steps.

The interface should do the following
=#

abstract type AbstractIntegratorAdapter end

struct IntegratorAdapterCore{P,M,S,I}
    problem::P
    method::M
    solution::S
    integrator::I
end


# TODO add some traits or capabilities to decorate this type
struct SeparableSystem <: AbstractSystem end

struct V1Field{S}
    system::S
    d::Int
end

function (f::V1Field)(v,t,q,params)
    # integer divison to get a int
    d = f.d
    p = @view q[d+1:end]
    v[begin:d] .= 0
    v[d+1:end] .= ∂H₂∂p(f.system, p)
    return nothing
end

struct V2Field{S}
    system::S
    d::Int
end

function (f::V2Field)(v, t, x, params)
    d = f.d
    q = @view x[begin:d]

    v[begin:d] .= ∂H₁∂q(f.system, q)
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
end

function (f::Q1Flow)(q1, t1, q0, t0, params)
    d = f.d
    q1 .= q0
    state = ChainState(@view(q1[begin:d]), @view(q1[d+1:end]))
    h1_flow(f.system, state, t1 - t0)
    return nothing
end

struct Q2Flow{S}
    system::S
    d::Int
end

function (f::Q2Flow)(q1, t1, q0, t0, params)
    d = f.d
    q1 .= q0
    state = ChainState(@view(q1[begin:d]), @view(q1[d+1:end]))
    h2_flow(f.system, state, t1 - t0)
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


struct SeparableODE{C<:IntegratorAdapterCore} <: AbstractIntegratorAdapter
    core::C

    function SeparableODE(system::AbstractSystem, initial_state::AbstractArray, timespan::Tuple, step_size::Real, method::GeometricMethod)
        problem = construct_split_ode_problem(system, initial_state, timespan, step_size)
        integrator = GeometricIntegrator(problem, method)
        solution = SolutionStep(problem)
        core = IntegratorAdapterCore(problem, method, solution, integrator)
        return new{typeof(core)}(core)
    end

    SeparableODE(system, initial_state, timespan, step_size; method::GeometricMethod=StrangA()) =  SeparableODE(system, initial_state, timespan, step_size, method)
end


LeapfrogAdapter(args...; kwargs...) = SeparableODE(args...; method=StrangA(), kwargs...)

end
