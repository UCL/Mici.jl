include("dependencies_for_runtests.jl")

using Mici.Mici:
    EuclideanSystem,
    GNISplittingIntegrator,
    LeapfrogIntegrator,
    PhasePoint,
    grad,
    h,
    step!,
    step_size

@testset "GNI adapter unit tests" begin
    ℓ = 𝒩()
    system = EuclideanSystem(ScalMat(2, 1.0), ℓ)
    q = [0.3, -0.7]
    p = [1.2, 0.4]
    z_native = PhasePoint(copy(q), copy(p), NaN, similar(q), false)
    z_gni = PhasePoint(copy(q), copy(p), NaN, similar(q), false)

    native_integrator = LeapfrogIntegrator(step_size=0.1)
    gni_integrator = GNISplittingIntegrator(
        system=system, phase_point=z_gni, step_size=0.1
    )

    @test step_size(native_integrator) == 0.1
    @test step_size(gni_integrator) == 0.1

    buffer = gni_integrator.buffer
    solstep = gni_integrator.solstep

    step!(z_native, native_integrator, system)
    step!(z_gni, gni_integrator, system)

    @test z_gni.q ≈ z_native.q
    @test z_gni.p ≈ z_native.p
    @test !z_gni.valid
    @test gni_integrator.buffer === buffer
    @test gni_integrator.solstep === solstep

    energy = h(z_gni, system)
    gradient = grad(z_gni, system)
    @test isfinite(energy)
    @test all(isfinite, gradient)
    @test z_gni.valid
end
