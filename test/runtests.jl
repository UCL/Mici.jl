using Mici
using Test

@testset "Mici.jl" begin
    include("test_unit.jl")
    include("test_e2e.jl")
    # include("gni_integration_tests.jl")
end
