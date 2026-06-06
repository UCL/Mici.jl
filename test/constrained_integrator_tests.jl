include("dependencies_for_runtests.jl")

using Mici.Mici:
    ConstrainedCompositionIntegrator,
    ConstrainedEuclideanMetricSystem,
    ConstrainedLeapfrogIntegrator,
    PhasePoint,
    grad,
    h,
    jacob_constr,
    project_onto_cotangent_space!,
    step!,
    step_size,
    constr,
    StrangA

function _negative_integrator(integrator::ConstrainedLeapfrogIntegrator)
    return ConstrainedLeapfrogIntegrator(step_size = -step_size(integrator))
end

function _negative_integrator(integrator::ConstrainedCompositionIntegrator{M}) where {M}
    return ConstrainedCompositionIntegrator{M}(step_size = -step_size(integrator))
end

function _reverse_trajectory!(z, integrator, system, n_step)
    for _ in 1:n_step
        step!(z, integrator, system)
    end
    return z
end

@testset "Constrained integrators" begin
    linear_system = ConstrainedEuclideanMetricSystem(
        ScalMat(3, 1.0),
        𝒩(μ = [0.0, 0.0, 0.0], Σ = ScalMat(3, 1.0)),
        constr = q -> q[1:1],
        jacob_constr = q -> reshape([1.0, 0.0, 0.0], 1, 3),
    )

    q0 = [0.0, 0.4, -0.2]
    p0 = [0.7, 0.2, -0.1]
    z0 = PhasePoint(copy(q0), copy(p0), NaN, similar(q0), false)
    project_onto_cotangent_space!(z0.p, z0, linear_system)
    @test isapprox(
        jacob_constr(z0, linear_system) * (linear_system.metric \ z0.p),
        zeros(1);
        atol = 1e-12,
    )

    native_leapfrog = ConstrainedLeapfrogIntegrator(step_size = 0.1)
    native_composition = ConstrainedCompositionIntegrator{StrangA}(step_size = 0.1)
    @test step_size(native_leapfrog) == 0.1
    @test step_size(native_composition) == 0.1

    z_leapfrog = copy(z0)
    z_composition = copy(z0)

    step!(z_leapfrog, native_leapfrog, linear_system)
    step!(z_composition, native_composition, linear_system)

    @test z_leapfrog.q ≈ z_composition.q
    @test z_leapfrog.p ≈ z_composition.p
    @test !z_leapfrog.valid
    @test !z_composition.valid
    @test isapprox(constr(z_leapfrog, linear_system), zeros(1); atol = 1e-12)
    @test isapprox(
        jacob_constr(z_leapfrog, linear_system) * (linear_system.metric \ z_leapfrog.p),
        zeros(1);
        atol = 1e-12,
    )

    @test isfinite(h(z_leapfrog, linear_system))
    @test all(isfinite, grad(z_leapfrog, linear_system))
    @test z_leapfrog.valid

    for integrator in (native_leapfrog, native_composition)
        z = copy(z0)
        forward = copy(z)
        _reverse_trajectory!(forward, integrator, linear_system, 10)
        reverse_integrator = _negative_integrator(integrator)
        backward = copy(forward)
        _reverse_trajectory!(backward, reverse_integrator, linear_system, 10)
        @test backward.q ≈ z.q
        @test backward.p ≈ z.p
    end

    nonlinear_system = ConstrainedEuclideanMetricSystem(
        ScalMat(3, 1.0),
        𝒩(μ = [0.0, 0.0, 0.0], Σ = ScalMat(3, 1.0)),
        constr = q -> [q[1]^2 + q[2]^2 - 1.0],
        jacob_constr = q -> reshape([2 * q[1], 2 * q[2], 0.0], 1, 3),
        projection_solver_kwargs = (; constraint_tol = 1e-12, position_tol = 1e-12),
    )

    nonlinear_q = [1.0, 0.0, 0.2]
    nonlinear_p = [0.0, 0.3, -0.1]
    nonlinear_z0 = PhasePoint(copy(nonlinear_q), copy(nonlinear_p), NaN, similar(nonlinear_q), false)
    project_onto_cotangent_space!(nonlinear_z0.p, nonlinear_z0, nonlinear_system)

    nonlinear_integrator = ConstrainedLeapfrogIntegrator(step_size = 0.05)
    nonlinear_z = copy(nonlinear_z0)
    _reverse_trajectory!(nonlinear_z, nonlinear_integrator, nonlinear_system, 8)
    reverse_nonlinear = _negative_integrator(nonlinear_integrator)
    _reverse_trajectory!(nonlinear_z, reverse_nonlinear, nonlinear_system, 8)

    @test nonlinear_z.q ≈ nonlinear_z0.q
    @test nonlinear_z.p ≈ nonlinear_z0.p
    @test isapprox(constr(nonlinear_z, nonlinear_system), zeros(1); atol = 1e-10)
    @test isapprox(
        jacob_constr(nonlinear_z, nonlinear_system) * (nonlinear_system.metric \ nonlinear_z.p),
        zeros(1);
        atol = 1e-10,
    )
end
