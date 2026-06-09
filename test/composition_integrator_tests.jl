include("dependencies_for_runtests.jl")

using Mici.Mici:
    CompositionIntegrator,
    EuclideanSystem,
    LeapfrogIntegrator,
    McLachlan2,
    McLachlan4,
    PhasePoint,
    StrangA,
    StrangB,
    SuzukiFractal,
    TripleJump,
    grad,
    h,
    step!,
    step_size,
    Φ₂!

function _reverse_trajectory!(z, integrator, system, n_step)
    for _ in 1:n_step
        step!(z, integrator, system)
    end
    return z
end

negative_integrator(integrator::LeapfrogIntegrator) =
    LeapfrogIntegrator(step_size = -step_size(integrator))
negative_integrator(integrator::CompositionIntegrator{M}) where {M} =
    CompositionIntegrator{M}(step_size = -step_size(integrator))

@testset "Direct composition integrators" begin
    ℓ = 𝒩()
    system = EuclideanSystem(ScalMat(2, 1.0), ℓ)
    q = [0.3, -0.7]
    p = [1.2, 0.4]
    z0 = PhasePoint(copy(q), copy(p), NaN, similar(q), false)

    integrators = [
        ("Leapfrog", LeapfrogIntegrator(step_size=0.1)),
        ("StrangA", CompositionIntegrator{StrangA}(step_size=0.1)),
        ("StrangB", CompositionIntegrator{StrangB}(step_size=0.1)),
        ("McLachlan2", CompositionIntegrator{McLachlan2}(step_size=0.1)),
        ("McLachlan4", CompositionIntegrator{McLachlan4}(step_size=0.1)),
        ("TripleJump", CompositionIntegrator{TripleJump}(step_size=0.1)),
        ("SuzukiFractal", CompositionIntegrator{SuzukiFractal}(step_size=0.1)),
    ]

    @test step_size(integrators[1][2]) == 0.1
    @test step_size(integrators[2][2]) == 0.1

    reference = copy(z0)
    step!(reference, integrators[1][2], system)

    strang_a = copy(z0)
    step!(strang_a, integrators[2][2], system)
    @test strang_a.q ≈ reference.q
    @test strang_a.p ≈ reference.p

    refresh_probe = copy(z0)
    Φ₂!(refresh_probe, system, 0.1)
    @test !refresh_probe.valid
    @test isfinite(h(refresh_probe, system))
    @test isfinite(sum(grad(refresh_probe, system)))
    @test refresh_probe.valid

    for (_, integrator) in integrators
        z = copy(z0)
        step!(z, integrator, system)

        energy = h(z, system)
        gradient = grad(z, system)
        @test isfinite(energy)
        @test all(isfinite, gradient)
        @test z.valid

        forward = copy(z0)
        _reverse_trajectory!(forward, integrator, system, 20)
        backward = copy(forward)
        reverse_integrator = negative_integrator(integrator)
        _reverse_trajectory!(backward, reverse_integrator, system, 20)
        @test backward.q ≈ z0.q
        @test backward.p ≈ z0.p
    end

    Hamiltonian = h(copy(z0), system)
    for (_, integrator) in integrators
        z = copy(z0)
        h_vals = Float64[Hamiltonian]
        for _ in 1:100
            step!(z, integrator, system)
            push!(h_vals, h(z, system))
        end
        diff_h = mean(h_vals[1:50]) - mean(h_vals[51:end])
        @test abs(diff_h) < 5e-3
    end
end
