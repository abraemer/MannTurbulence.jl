using MannTurbulence
using Test
using Aqua
using JET

@testset "MannTurbulence.jl" begin
    # @testset "Code quality (Aqua.jl)" begin
    #     Aqua.test_all(MannTurbulence)
    # end
    # @testset "Code linting (JET.jl)" begin
    #     JET.test_package(MannTurbulence; target_defined_modules = true)
    # end
    # Include tensor tests
    include("test_tensors.jl")
    
    # Include spectra tests
    include("test_spectra.jl")
end
