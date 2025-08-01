"""
Tests for spectral generation functionality.

Tests the MannParameters struct, spectral tensor calculations, Mann spectra computation,
and turbulence generation against reference data from the Rust implementation.
"""

using Statistics
using LinearAlgebra
using MannTurbulence

const TEST_TOLERANCE = 1e-5

@testset "Spectral Generation Tests" begin
    
    @testset "MannParameters Tests" begin
        @testset "Valid Parameters" begin
            # Test basic construction
            params = MannParameters(33.6, 3.9, 1.0)
            @test params.L ≈ 33.6
            @test params.Γ ≈ 3.9
            @test params.ae ≈ 1.0
            @test params.kappa ≈ 0.4  # default
            @test params.q ≈ 5.0/3.0  # default
            
            # Test with custom parameters
            params2 = MannParameters(50.0, 2.5, 0.5, 0.35, 1.6, 1.2, 0.9)
            @test params2.L ≈ 50.0
            @test params2.Γ ≈ 2.5
            @test params2.ae ≈ 0.5
            @test params2.kappa ≈ 0.35
            @test params2.q ≈ 1.6
            @test params2.C ≈ 1.2
            @test params2.C_coherence ≈ 0.9
        end
        
        @testset "Parameter Validation" begin
            # Test negative values
            @test_throws ArgumentError MannParameters(-1.0, 3.9, 1.0)  # negative L
            @test_throws ArgumentError MannParameters(33.6, -1.0, 1.0)  # negative Γ
            @test_throws ArgumentError MannParameters(33.6, 3.9, -1.0)  # negative ae
            @test_throws ArgumentError MannParameters(33.6, 3.9, 1.0, -0.1)  # negative kappa
            
            # Test zero values
            @test_throws ArgumentError MannParameters(0.0, 3.9, 1.0)  # zero L
            @test_throws ArgumentError MannParameters(33.6, 0.0, 1.0)  # zero Γ
            @test_throws ArgumentError MannParameters(33.6, 3.9, 0.0)  # zero ae
        end
        
        @testset "Type Consistency" begin
            # Test Float32
            params_f32 = MannParameters{Float32}(33.6f0, 3.9f0, 1.0f0)
            @test params_f32.L isa Float32
            @test params_f32.Γ isa Float32
            @test params_f32.ae isa Float32
            
            # Test Float64 (default)
            params_f64 = MannParameters(33.6, 3.9, 1.0)
            @test params_f64.L isa Float64
            @test params_f64.Γ isa Float64
            @test params_f64.ae isa Float64
        end
    end
    
    @testset "Trapezoidal Integration Tests" begin
        @testset "Simple Functions" begin
            # Test integration of constant function
            x = [0.0, 1.0, 2.0]
            y = [0.0, 1.0]
            f_const = ones(3, 2)
            result = trapezoidal_integral_2d(f_const, x, y)
            @test result ≈ 2.0  # 2×1 rectangle
            
            # Test integration of linear function f(x,y) = x
            f_linear = [0.0 0.0; 1.0 1.0; 2.0 2.0]
            result = trapezoidal_integral_2d(f_linear, x, y)
            @test result ≈ 2.0  # ∫₀² ∫₀¹ x dy dx = ∫₀² x dx = 2
        end
        
        @testset "Non-uniform Grids" begin
            # Test with non-uniform spacing
            x = [0.0, 0.5, 2.0]  # non-uniform
            y = [0.0, 0.3, 1.0]  # non-uniform
            f = ones(3, 3)
            result = trapezoidal_integral_2d(f, x, y)
            expected = 2.0 * 1.0  # total area
            @test result ≈ expected
        end
        
        @testset "Error Handling" begin
            x = [0.0, 1.0]
            y = [0.0, 1.0]
            f_wrong = ones(3, 2)  # wrong dimensions
            @test_throws ArgumentError trapezoidal_integral_2d(f_wrong, x, y)
        end
    end
    
    @testset "Spectral Tensor Tests" begin
        @testset "Basic Functionality" begin
            params = MannParameters(33.6, 3.9, 1.0)
            K = [1.0, 0.5, 0.2]
            
            tensor_matrix = spectral_tensor(params, K)
            
            # Check dimensions
            @test size(tensor_matrix) == (3, 3)
            
            # Check that result is finite
            @test all(isfinite.(tensor_matrix))
            
            # Check symmetry (spectral tensor should be symmetric)
            @test tensor_matrix ≈ tensor_matrix' atol=1e-10
        end
        
        @testset "Zero Wave Vector" begin
            params = MannParameters(33.6, 3.9, 1.0)
            K_zero = [0.0, 0.0, 0.0]
            
            tensor_matrix = spectral_tensor(params, K_zero)
            @test all(tensor_matrix .== 0.0)
        end
        
        @testset "Input Validation" begin
            params = MannParameters(33.6, 3.9, 1.0)
            
            # Wrong dimension
            @test_throws ArgumentError spectral_tensor(params, [1.0, 2.0])  # only 2 components
            @test_throws ArgumentError spectral_tensor(params, [1.0, 2.0, 3.0, 4.0])  # 4 components
        end
    end
    
    @testset "Mann Spectra Tests" begin
        @testset "Against Rust Reference Data" begin
            # Load reference data from Rust implementation
            if test_data_exists("mann_spectra.json")
                test_data = load_test_data("mann_spectra.json")
                
                # Extract parameters
                γ = test_data["parameters"]["gamma"]
                L = test_data["parameters"]["l"]
                ae = test_data["parameters"]["ae"]
                
                # Create parameters
                params = MannParameters(L, γ, ae)
                
                # Extract inputs and convert to Float64
                kx = Float64.(test_data["inputs"]["kx"])
                
                # Compute spectra
                suu, svv, sww, suw = mann_spectra(kx, params)
                
                # Compare with expected results
                tolerance = test_data["metadata"]["tolerance"]
                
                suu_expected = test_data["outputs"]["suu_expected"]
                svv_expected = test_data["outputs"]["svv_expected"] 
                sww_expected = test_data["outputs"]["sww_expected"]
                suw_expected = test_data["outputs"]["suw_expected"]
                
                @test suu ≈ suu_expected atol=tolerance rtol=tolerance
                @test svv ≈ svv_expected atol=tolerance rtol=tolerance
                @test sww ≈ sww_expected atol=tolerance rtol=tolerance
                @test suw ≈ suw_expected atol=tolerance rtol=tolerance
                
                println("Mann spectra test passed with tolerance $tolerance")
            else
                @warn "Reference data file not found: mann_spectra.json"
                @test_skip "Mann spectra validation against Rust data"
            end
        end
        
        @testset "Basic Properties" begin
            params = MannParameters(33.6, 3.9, 1.0)
            kx = [0.1, 1.0, 10.0]
            
            suu, svv, sww, suw = mann_spectra(kx, params)
            
            # Check dimensions
            @test length(suu) == length(kx)
            @test length(svv) == length(kx)
            @test length(sww) == length(kx)
            @test length(suw) == length(kx)
            
            # Check that results are finite
            @test all(isfinite.(suu))
            @test all(isfinite.(svv))
            @test all(isfinite.(sww))
            @test all(isfinite.(suw))
            
            # Check that diagonal components are positive (energy spectra)
            @test all(suu .> 0)
            @test all(svv .> 0)
            @test all(sww .> 0)
        end
        
        @testset "Integration Parameters" begin
            params = MannParameters(33.6, 3.9, 1.0)
            kx = [1.0]
            
            # Test with different integration parameters
            suu1, svv1, sww1, suw1 = mann_spectra(kx, params, nr=50, ntheta=15)
            suu2, svv2, sww2, suw2 = mann_spectra(kx, params, nr=100, ntheta=20)
            
            # Results should be similar but not identical due to different resolution
            @test suu1[1] ≈ suu2[1] rtol=0.1  # 10% tolerance for integration resolution
            @test svv1[1] ≈ svv2[1] rtol=0.1
            @test sww1[1] ≈ sww2[1] rtol=0.1
            @test suw1[1] ≈ suw2[1] rtol=0.1
        end
    end
    
    @testset "Turbulence Generation Tests" begin
        @testset "Basic Generation" begin
            params = MannParameters(33.6, 3.9, 1.0)
            Lx, Ly, Lz = 10.0, 10.0, 10.0
            Nx, Ny, Nz = 16, 16, 16
            
            U, V, W = generate_turbulence(params, Lx, Ly, Lz, Nx, Ny, Nz, seed=42)
            
            # Check dimensions
            @test size(U) == (Nx, Ny, Nz)
            @test size(V) == (Nx, Ny, Nz)
            @test size(W) == (Nx, Ny, Nz)
            
            # Check that results are finite
            @test all(isfinite.(U))
            @test all(isfinite.(V))
            @test all(isfinite.(W))
            
            # Check statistical properties
            stats = validate_turbulence_statistics(U, V, W)
            @test stats["mean_valid"]  # means should be close to zero
            @test stats["finite_check"]  # all values should be finite
            @test stats["total_energy"] > 0  # should have positive energy
        end
        
        @testset "Deterministic Generation" begin
            params = MannParameters(33.6, 3.9, 1.0)
            Lx, Ly, Lz = 10.0, 10.0, 10.0
            Nx, Ny, Nz = 8, 8, 8
            
            # Generate with same seed twice
            U1, V1, W1 = generate_turbulence(params, Lx, Ly, Lz, Nx, Ny, Nz, seed=123)
            U2, V2, W2 = generate_turbulence(params, Lx, Ly, Lz, Nx, Ny, Nz, seed=123)
            
            # Results should be identical
            @test U1 ≈ U2
            @test V1 ≈ V2
            @test W1 ≈ W2
            
            # Generate with different seed
            U3, V3, W3 = generate_turbulence(params, Lx, Ly, Lz, Nx, Ny, Nz, seed=456)
            
            # Results should be different
            @test !(U1 ≈ U3)
            @test !(V1 ≈ V3)
            @test !(W1 ≈ W3)
        end
        
        @testset "Parallel vs Serial" begin
            params = MannParameters(33.6, 3.9, 1.0)
            Lx, Ly, Lz = 10.0, 10.0, 10.0
            Nx, Ny, Nz = 8, 8, 8
            
            # Generate with parallel and serial
            U_par, V_par, W_par = generate_turbulence(params, Lx, Ly, Lz, Nx, Ny, Nz, 
                                                    seed=42, parallel=true)
            U_ser, V_ser, W_ser = generate_turbulence(params, Lx, Ly, Lz, Nx, Ny, Nz, 
                                                    seed=42, parallel=false)
            
            # Results should be identical (same algorithm, same seed)
            @test U_par ≈ U_ser
            @test V_par ≈ V_ser
            @test W_par ≈ W_ser
        end
        
        @testset "Sinc Correction" begin
            # Use smaller L to ensure some frequencies satisfy k_norm2 < 3.0/L condition
            params = MannParameters(1.0, 3.9, 1.0)  # L = 1.0 instead of 33.6
            Lx, Ly, Lz = 10.0, 10.0, 10.0
            Nx, Ny, Nz = 8, 8, 8
            
            # Generate with and without sinc correction
            U1, V1, W1 = generate_turbulence(params, Lx, Ly, Lz, Nx, Ny, Nz,
                                            seed=42, use_sinc_correction=false)
            U2, V2, W2 = generate_turbulence(params, Lx, Ly, Lz, Nx, Ny, Nz,
                                            seed=42, use_sinc_correction=true)
            
            # Results should be different (sinc correction affects low frequencies)
            @test !(U1 ≈ U2)
            @test !(V1 ≈ V2)
            @test !(W1 ≈ W2)
            
            # But both should have valid statistics
            stats1 = validate_turbulence_statistics(U1, V1, W1)
            stats2 = validate_turbulence_statistics(U2, V2, W2)
            @test stats1["mean_valid"]
            @test stats2["mean_valid"]
            @test stats1["finite_check"]
            @test stats2["finite_check"]
        end
        
        @testset "Against Rust Reference Data" begin
            # Load reference data for deterministic turbulence generation
            if test_data_exists("deterministic_turbulence_generation.json")
                test_data = load_test_data("deterministic_turbulence_generation.json")
                
                # Extract parameters
                ae = test_data["parameters"]["ae"]
                L = test_data["parameters"]["L"]
                γ = test_data["parameters"]["gamma"]
                Lx = test_data["parameters"]["Lx"]
                Ly = test_data["parameters"]["Ly"]
                Lz = test_data["parameters"]["Lz"]
                Nx = test_data["parameters"]["Nx"]
                Ny = test_data["parameters"]["Ny"]
                Nz = test_data["parameters"]["Nz"]
                seed = test_data["parameters"]["seed"]
                
                # Create parameters
                params = MannParameters(L, γ, ae)
                
                # Generate turbulence
                U, V, W = generate_turbulence(params, Lx, Ly, Lz, Nx, Ny, Nz, seed=seed)
                
                # Check dimensions match expected
                expected_shape = test_data["outputs"]["U_sample_shape"]
                @test size(U) == tuple(expected_shape...)
                @test size(V) == tuple(expected_shape...)
                @test size(W) == tuple(expected_shape...)
                
                # Check statistical properties if available
                if haskey(test_data["outputs"], "U_mean")
                    @test mean(U) ≈ test_data["outputs"]["U_mean"] atol=1e-10
                    @test mean(V) ≈ test_data["outputs"]["V_mean"] atol=1e-10
                    @test mean(W) ≈ test_data["outputs"]["W_mean"] atol=1e-10
                end
                
                if haskey(test_data["outputs"], "U_variance")
                    @test var(U) ≈ test_data["outputs"]["U_variance"] rtol=0.01
                    @test var(V) ≈ test_data["outputs"]["V_variance"] rtol=0.01
                    @test var(W) ≈ test_data["outputs"]["W_variance"] rtol=0.01
                end
                
                println("Deterministic turbulence generation test passed")
            else
                @warn "Reference data file not found: deterministic_turbulence_generation.json"
                @test_skip "Turbulence generation validation against Rust data"
            end
        end
    end
    
    @testset "Utility Functions Tests" begin
        @testset "Frequency Components Precomputation" begin
            Lx, Ly, Lz = 10.0, 8.0, 6.0
            Nx, Ny, Nz = 16, 12, 8
            
            kx, ky, kz = precompute_frequency_components(Lx, Ly, Lz, Nx, Ny, Nz)
            
            # Check dimensions
            @test length(kx) == Nx
            @test length(ky) == Ny
            @test length(kz) == Nz ÷ 2 + 1  # rfft frequencies
            
            # Check that frequencies are finite
            @test all(isfinite.(kx))
            @test all(isfinite.(ky))
            @test all(isfinite.(kz))
            
            # Check frequency ranges
            @test kx[1] == 0.0  # DC component
            @test kz[1] == 0.0  # DC component for rfft
            @test all(kz .>= 0)  # rfft frequencies are non-negative
        end
        
        @testset "Statistics Validation" begin
            # Create test data with known properties
            U = randn(10, 10, 10)
            V = randn(10, 10, 10) .+ 0.1  # small mean
            W = randn(10, 10, 10)
            
            stats = validate_turbulence_statistics(U, V, W, tolerance=0.2)
            
            @test haskey(stats, "mean_U")
            @test haskey(stats, "var_U")
            @test haskey(stats, "total_energy")
            @test haskey(stats, "finite_check")
            @test stats["finite_check"] == true
            @test stats["total_energy"] > 0
            
            # Test with infinite values
            U_inf = copy(U)
            U_inf[1] = Inf
            stats_inf = validate_turbulence_statistics(U_inf, V, W)
            @test stats_inf["finite_check"] == false
        end
    end
end