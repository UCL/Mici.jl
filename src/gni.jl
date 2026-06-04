using GeometricIntegrators

"""
    GNISplittingIntegrator

Adapter which runs Mici's separable Hamiltonian flows through
GeometricIntegrators.jl's SODE splitting machinery.

The adapter owns the combined `(q, p)` buffer and GNI objects. A `PhasePoint`
is copied into the buffer before each step and copied back afterwards, keeping
Mici's accepted/proposed state and cache invalidation semantics unchanged.
"""
struct GNISplittingIntegrator{T,B,P,M,S,I} <: AbstractIntegrator
    ϵ::T
    buffer::B
    problem::P
    method::M
    solstep::S
    integrator::I
end

"""
    GNICompositionIntegrator

Adapter which runs Mici's separable Hamiltonian flows through
GeometricIntegrators.jl's composition machinery.

The outer composition integrator uses `NoCache`, while each exact substep still
uses the package's exact-solution machinery. This keeps the same phase-point
semantics as the splitting adapter but changes the orchestration path used by
GeometricIntegrators.
"""
struct GNICompositionIntegrator{T,B,P,M,S,I} <: AbstractIntegrator
    ϵ::T
    buffer::B
    problem::P
    method::M
    solstep::S
    integrator::I
end

struct GNIPotentialVectorField{S,P}
    system::S
    scratch::P
end

struct GNIKineticVectorField{S,P}
    system::S
    scratch::P
end

struct GNIPotentialFlow{S,P}
    system::S
    scratch::P
end

struct GNIKineticFlow{S,P}
    system::S
    scratch::P
end

function _phasepoint_to_buffer!(buffer::AbstractVector, z::PhasePoint)
    d = dimension(z)
    @views begin
        buffer[1:d] .= z.q
        buffer[(d + 1):(2d)] .= z.p
    end
    return buffer
end

function _buffer_to_phasepoint!(z::PhasePoint, buffer::AbstractVector)
    d = dimension(z)
    @views begin
        z.q .= buffer[1:d]
        z.p .= buffer[(d + 1):(2d)]
    end
    refresh!(z)
    return z
end

function _buffer_to_scratch!(scratch::PhasePoint, buffer::AbstractVector)
    _buffer_to_phasepoint!(scratch, buffer)
    return scratch
end

function _scratch_to_buffer!(buffer::AbstractVector, scratch::PhasePoint)
    _phasepoint_to_buffer!(buffer, scratch)
    return buffer
end

function (field::GNIPotentialVectorField)(v, t, x, params)
    scratch = _buffer_to_scratch!(field.scratch, x)
    d = dimension(scratch)
    @views begin
        v[1:d] .= 0
        v[(d + 1):(2d)] .= .-∂h₁∂q(scratch, field.system)
    end
    return nothing
end

function (field::GNIKineticVectorField)(v, t, x, params)
    scratch = _buffer_to_scratch!(field.scratch, x)
    d = dimension(scratch)
    @views begin
        v[1:d] .= ∂h₂∂p(scratch, field.system)
        v[(d + 1):(2d)] .= 0
    end
    return nothing
end

function (flow::GNIPotentialFlow)(x₁, t₁, x₀, t₀, params)
    scratch = _buffer_to_scratch!(flow.scratch, x₀)
    Φ₁!(scratch, flow.system, t₁ - t₀)
    _scratch_to_buffer!(x₁, scratch)
    return nothing
end

function (flow::GNIKineticFlow)(x₁, t₁, x₀, t₀, params)
    scratch = _buffer_to_scratch!(flow.scratch, x₀)
    Φ₂!(scratch, flow.system, t₁ - t₀)
    _scratch_to_buffer!(x₁, scratch)
    return nothing
end

function _gni_scratch(z::PhasePoint{T}) where {T}
    PhasePoint(undef, dimension(z), T)
end

function _gni_problem(
    system::AbstractTractableFlowSystem,
    phase_point::PhasePoint,
    ϵ,
)
    buffer = Vector{eltype(phase_point.q)}(undef, 2 * dimension(phase_point))
    _phasepoint_to_buffer!(buffer, phase_point)

    vector_fields = (
        GNIPotentialVectorField(system, _gni_scratch(phase_point)),
        GNIKineticVectorField(system, _gni_scratch(phase_point)),
    )
    flows = (
        GNIPotentialFlow(system, _gni_scratch(phase_point)),
        GNIKineticFlow(system, _gni_scratch(phase_point)),
    )
    problem = SODEProblem(vector_fields, flows, (zero(ϵ), ϵ), ϵ, buffer)
    return buffer, problem
end

function GNISplittingIntegrator(;
    system::AbstractTractableFlowSystem,
    phase_point::PhasePoint,
    step_size=0.1,
    method=StrangA(),
    kwargs...,
)
    ϵ = isnothing(step_size) ? 0.1 : step_size
    buffer, problem = _gni_problem(system, phase_point, ϵ)
    integrator = GeometricIntegrator(problem, method)
    solstep = SolutionStep(problem)
    return GNISplittingIntegrator(ϵ, buffer, problem, method, solstep, integrator)
end

function GNICompositionIntegrator(;
    system::AbstractTractableFlowSystem,
    phase_point::PhasePoint,
    step_size=0.1,
    method=StrangA(),
    kwargs...,
)
    ϵ = isnothing(step_size) ? 0.1 : step_size
    buffer, problem = _gni_problem(system, phase_point, ϵ)
    integrator = GeometricIntegrator(problem, Composition(method))
    solstep = SolutionStep(problem)
    return GNICompositionIntegrator(ϵ, buffer, problem, method, solstep, integrator)
end

step_size(integrator::GNISplittingIntegrator) = integrator.ϵ
step_size(integrator::GNICompositionIntegrator) = integrator.ϵ

function step!(
    z::PhasePoint,
    integrator::GNISplittingIntegrator,
    system::AbstractTractableFlowSystem,
)
    _phasepoint_to_buffer!(integrator.buffer, z)
    copy!(integrator.solstep, (t=zero(integrator.ϵ), q=integrator.buffer))
    GeometricIntegrators.integrate!(integrator.solstep, integrator.integrator)
    integrator.buffer .= integrator.solstep.q
    _buffer_to_phasepoint!(z, integrator.buffer)
    return nothing
end

function step!(
    z::PhasePoint,
    integrator::GNICompositionIntegrator,
    system::AbstractTractableFlowSystem,
)
    _phasepoint_to_buffer!(integrator.buffer, z)
    copy!(integrator.solstep, (t=zero(integrator.ϵ), q=integrator.buffer))
    GeometricIntegrators.integrate!(integrator.solstep, integrator.integrator)
    integrator.buffer .= integrator.solstep.q
    _buffer_to_phasepoint!(z, integrator.buffer)
    return nothing
end

const LeapfrogAdapter = GNISplittingIntegrator
const CompositionAdapter = GNICompositionIntegrator
