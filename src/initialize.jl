@inline function resolve(x, table::NamedTuple, args...)
    if x isa Symbol
        try
            f = getproperty(table, x)
            return f(args...)
        catch
            throw(ArgumentError("Unknown option: $x"))
        end
    else
        return x
    end
end

const SYSTEMS = (
    euclidean = (metric) -> EuclideanSystem(metric),
)

const METRICS = (
    unit = (d) -> ScalMat(d, 1.0),
    diag = (d, v) -> ScalMat(d, v),
)

const INTEGRATORS = (
    leapfrog = (ϵ, T) -> LeapfrogIntegrator(ϵ, T),
)

const INTEGRATION_TRANSITIONS = (
    metropolis = () -> MetropolisTransition(),
)

const MOMENTUM_TRANSITIONS = (
    independent = () -> IndependentMomentumTransition(),
)

function resolve_system(s)

    s isa NamedTuple && begin
        metric = resolve_metric(get(s, :metric_type, :unit), get(s, :dimension, nothing), get(s, :metric_values, nothing))
    end

    
    s === :euclidean && return EuclideanSystem(metric)
    s isa AbstractSystem && return s
    error("unknown system spec: $s")
end


resolve_system(x, args...) = resolve(x, SYSTEMS, args...)
resolve_metric(x, args...) = resolve(x, METRICS, args...)
resolve_integrator(x, args...) = resolve(x, INTEGRATORS, args...)
resolve_integration_transition(x, args...) = resolve(x, INTEGRATION_TRANSITIONS, args...)
resolve_momentum_transition(x, args...) = resolve(x, MOMENTUM_TRANSITIONS, args...)