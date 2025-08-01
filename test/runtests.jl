using MannTurbulence
using Test
using Aqua
using JET

include("test_helpers.jl")

@testset "MannTurbulence.jl" begin
    # @testset "Code quality (Aqua.jl)" begin
    #     Aqua.test_all(MannTurbulence)
    # end
    # @testset "Code linting (JET.jl)" begin
    #     JET.test_package(MannTurbulence; target_defined_modules = true)
    # end

    # Include tensor tests
    @testset "tensors.jl" include("test_tensors.jl")

    # Include spectra tests
    @testset "spectra.jl" include("test_spectra.jl")
end
