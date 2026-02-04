using LogDensityProblems

# Rough pseudo-code for constrained problems
struct ConstrainedModel{LD, C}
    ldp::LD
    constrait::C
end

LogDensityProblems.dimension(m::ConstrainedModel) = 
    LogDensityProblems.dimension(m.ldp)

LogDensityProblems.logdensity(m::ConstrainedModel, x) = 
    LogDensityProblems.logdensity(m.ldp, x)

constraint(m::ConstrainedModel, x) =
    m.constrait(x)

constraint_dimension(m::ConstrainedModel, x) = 
    length(constraint(m, x))

# Autodiff