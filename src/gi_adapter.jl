function gi_problem(
    h::EuclideanSystem,
    state::AbstractChainState,
    T::Int,
    ε::Float64
)

    q₀ = copy(q(state))
    p₀ = copy(p(state))

    function v!(dq, t, q, p, params)
        state_tmp = MarkovChainState(q, p)
        dq .= ∂H₂∂p(h, state_tmp)
    end

    function f!(dp, t, q, p, params)
        state_tmp = MarkovChainState(q, p)
        dp .= -∂H₁∂q(h, state_tmp)
    end

    function hamiltonian(t, q, p, params)
        state_tmp = MarkovChainState(q, p)
        return H(h, state_tmp)
    end

    return HODEProblem(v!,f!,hamiltonian, T, ε, q₀, p₀)
end

function initialise_step(problem::GeometricProblem, method::GeometricMethod)
    integrator = GeometricIntegrator(problem, method)

    solstep = SolutionStep(
        problem;
        internal = internal_variables(integrator)
    )

    return solstep, integrator
end

function current_qp(solstep::SolutionStep)
    sol = GeometricIntegratorsBase.solution(solstep)
    return Vector(sol.q[0]), Vector(sol.p[0])   #Should be 0?
end

function set_initial_condition!(solstep, state)
    sol = GeometricIntegratorsBase.solution(solstep)

    sol.q[0] .= q(state)
    sol.p[0] .= p(state)
end
