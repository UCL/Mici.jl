using Mici
using Test

@testset "Mici.jl" begin
    include("test_e2e.jl")
    include("test_end_to_end.jl")
    include("gni_integration_tests.jl")
end
