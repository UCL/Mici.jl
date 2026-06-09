include("dependencies_for_runtests.jl")

using Mici.Mici:
    CompositionAdapter,
    EuclideanSystem,
    LeapfrogIntegrator,
    PhasePoint,
    grad,
    h,
    step!,
    step_size

@testset "GNI composition adapter unit tests" begin
    ℓ = 𝒩()
    system = EuclideanSystem(ScalMat(2, 1.0), ℓ)
    q = [0.3, -0.7]
    p = [1.2, 0.4]
    z_native = PhasePoint(copy(q), copy(p), NaN, similar(q), false)
    z_composition = PhasePoint(copy(q), copy(p), NaN, similar(q), false)

    native_integrator = LeapfrogIntegrator(step_size=0.1)
    composition_integrator = CompositionAdapter(
        system=system, phase_point=z_composition, step_size=0.1
    )

    @test step_size(native_integrator) == 0.1
    @test step_size(composition_integrator) == 0.1

    buffer = composition_integrator.buffer
    solstep = composition_integrator.solstep

    step!(z_native, native_integrator, system)
    step!(z_composition, composition_integrator, system)

    @test z_composition.q ≈ z_native.q
    @test z_composition.p ≈ z_native.p
    @test !z_composition.valid
    @test composition_integrator.buffer === buffer
    @test composition_integrator.solstep === solstep

    energy = h(z_composition, system)
    gradient = grad(z_composition, system)
    @test isfinite(energy)
    @test all(isfinite, gradient)
    @test z_composition.valid
end
