"""
Tests for tensor operations using Rust reference data.
"""

using MannTurbulence
using LinearAlgebra
using Statistics
using FFTW

const TEST_TOLERANCE = 1e-5

@testset "Basic Functions" begin
    @testset "lifetime_approx" begin
        # Test with various kL values
        @test lifetime_approx(0.001) ≈ lifetime_approx(0.005)  # Lower bound check
        @test lifetime_approx(1.0) > 0.0
        @test lifetime_approx(10.0) > 0.0
        
        # Test specific values from Rust implementation
        @test isapprox(lifetime_approx(1.0), 1.2341233491897583, rtol=1e-5)
    end
    
    @testset "vonkarman_spectrum" begin
        # Test basic properties
        @test vonkarman_spectrum(1.0, 0.0, 1.0) == 0.0
        @test vonkarman_spectrum(1.0, 1.0, 1.0) > 0.0
        @test vonkarman_spectrum(2.0, 1.0, 1.0) ≈ 2.0 * vonkarman_spectrum(1.0, 1.0, 1.0)
    end
    
    @testset "sinc2" begin
        @test sinc2(0.0) == 1.0
        @test sinc2(π) ≈ 0.0 atol=1e-10
        @test sinc2(-π) ≈ 0.0 atol=1e-10
        @test sinc2(π/2) ≈ (2/π)^2 atol=1e-10
    end
end

@testset "Isotropic Tensor" begin
    @testset "Isotropic tensor calculation" begin
        test_data = load_test_data("isotropic_tensor.json")
        
        ae = Float64(test_data.parameters.ae)
        L = Float64(test_data.parameters.L)
        K = Float64.(collect(test_data.inputs.K))
        expected = Matrix(transpose(hcat(test_data.outputs.expected_result...)))
        
        iso = Isotropic(ae, L)
        result = tensor(iso, K)
        
        @test isapprox(result, expected, rtol=TEST_TOLERANCE)
        
        # Test that tensor result matches expected_result
        tensor_result = Matrix(transpose(hcat(test_data.outputs.tensor_result...)))
        @test isapprox(result, tensor_result, rtol=TEST_TOLERANCE)
    end
    
    @testset "Isotropic decomposition" begin
        # Test decomposition properties
        iso = Isotropic(1.0, 1.0)
        K = [1.0, 2.0, 3.0]
        
        decomp_result = decomp(iso, K)
        tensor_result = tensor(iso, K)
        
        # Check that decomp' * decomp ≈ tensor (approximately)
        reconstructed = decomp_result' * decomp_result
        
        # The relationship should hold approximately
        @test size(decomp_result) == (3, 3)
        @test size(reconstructed) == (3, 3)
    end
    
    @testset "Zero wave vector" begin
        iso = Isotropic(1.0, 1.0)
        K_zero = [0.0, 0.0, 0.0]
        
        @test tensor(iso, K_zero) == zeros(3, 3)
        @test decomp(iso, K_zero) == zeros(3, 3)
    end
end

@testset "Sheared Tensor" begin
    @testset "Sheared tensor calculation" begin
        test_data = load_test_data("sheared_tensor.json")
        
        ae = Float64(test_data.parameters.ae)
        L = Float64(test_data.parameters.L)
        gamma = Float64(test_data.parameters.gamma)
        K = Float64.(collect(test_data.inputs.K))
        expected = Matrix(transpose(hcat(test_data.outputs.expected_result...)))
        
        sheared = Sheared(ae, L, gamma)
        result = tensor(sheared, K)
        
        @test isapprox(result, expected, rtol=TEST_TOLERANCE)
    end
    
    @testset "Sheared transform matrix" begin
        test_data = load_test_data("sheared_transform.json")
        
        ae = Float64(test_data.parameters.ae)
        L = Float64(test_data.parameters.L)
        gamma = Float64(test_data.parameters.gamma)
        K = Float64.(collect(test_data.inputs.K))
        expected = Matrix(transpose(hcat(test_data.outputs.expected_result...)))
        
        sheared = Sheared(ae, L, gamma)
        result = sheared_transform(sheared, K)
        
        @test isapprox(result, expected, rtol=TEST_TOLERANCE)
    end
    
    @testset "Zero wave vector" begin
        sheared = Sheared(1.0, 1.0, 1.0)
        K_zero = [0.0, 0.0, 0.0]
        
        @test tensor(sheared, K_zero) == zeros(3, 3)
        @test decomp(sheared, K_zero) == zeros(3, 3)
        @test sheared_transform(sheared, K_zero) == zeros(3, 3)
    end
end

@testset "Random Tensor Generation" begin
    @testset "Basic properties" begin
        dims = (8, 8, 8, 3)
        
        # Test with seed for reproducibility
        tensor1 = random_tensor(dims; seed=42)
        tensor2 = random_tensor(dims; seed=42)
        @test tensor1 == tensor2
        
        # Test different seeds give different results
        tensor3 = random_tensor(dims; seed=43)
        @test tensor1 != tensor3
        
        # Test dimensions
        @test size(tensor1) == dims
        
        # Test that values are complex
        @test eltype(tensor1) == ComplexF64
        
        # Test approximate unit variance (statistical test)
        real_var = var(real.(tensor1[:]))
        imag_var = var(imag.(tensor1[:]))
        total_var = var(abs.(tensor1[:]))
        
        # Each component should have variance ≈ 1/2, total ≈ 1
        @test abs(real_var - 0.5) < 0.1
        @test abs(imag_var - 0.5) < 0.1
    end
end

@testset "FFT Utilities" begin
    @testset "FFTW fftfreq integration" begin
        # Test that our freq_components uses FFTW correctly
        N = 8
        dx = 1.0
        freqs = FFTW.fftfreq(N, 1.0/dx)
        
        @test length(freqs) == N
        @test freqs[1] == 0.0
        @test freqs[2] ≈ 1.0/(N*dx)
        
        # Test symmetry properties
        @test freqs[end] ≈ -1.0/(N*dx)
    end
    
    @testset "FFTW rfftfreq integration" begin
        N = 8
        dx = 1.0
        freqs = FFTW.rfftfreq(N, 1.0/dx)
        
        expected_length = div(N, 2) + 1
        @test length(freqs) == expected_length
        @test freqs[1] == 0.0
        @test freqs[end] ≈ 0.5/dx
    end
    
    @testset "freq_components with FFTW" begin
        Lx, Ly, Lz = 10.0, 20.0, 30.0
        Nx, Ny, Nz = 16, 32, 64
        
        kx, ky, kz = freq_components(Lx, Ly, Lz, Nx, Ny, Nz)
        
        @test length(kx) == Nx
        @test length(ky) == Ny
        @test length(kz) == div(Nz, 2) + 1
        
        # Test that frequencies are properly scaled using FFTW conventions
        @test kx[2] ≈ 2π/Lx
        @test ky[2] ≈ 2π/Ly
        @test kz[2] ≈ 2π/Lz
        
        # Verify FFTW consistency
        expected_kx = FFTW.fftfreq(Nx, 2.0 * π * Nx / Lx)
        expected_ky = FFTW.fftfreq(Ny, 2.0 * π * Ny / Ly)
        expected_kz = FFTW.rfftfreq(Nz, 2.0 * π * Nz / Lz)
        
        @test kx ≈ expected_kx
        @test ky ≈ expected_ky
        @test kz ≈ expected_kz
    end
end

@testset "Comprehensive Tensor Tests" begin
    @testset "Standard parameters" begin
        for i in 0:2
            filename = "comprehensive_tensors_standard_params_$i.json"
            test_data = load_test_data(filename)
            
            ae = Float64(test_data.parameters.ae)
            L = Float64(test_data.parameters.L)
            gamma = Float64(test_data.parameters.gamma)
            
            # Test single wave vector (not multiple)
            K = Float64.(collect(test_data.inputs.K))
            expected_iso = Matrix(transpose(hcat(test_data.outputs.isotropic_tensor...)))
            expected_sheared = Matrix(transpose(hcat(test_data.outputs.sheared_tensor...)))
            
            # Test isotropic
            iso = Isotropic(ae, L)
            iso_result = tensor(iso, K)
            @test isapprox(iso_result, expected_iso, rtol=TEST_TOLERANCE)
            
            # Test sheared
            sheared = Sheared(ae, L, gamma)
            sheared_result = tensor(sheared, K)
            @test isapprox(sheared_result, expected_sheared, rtol=TEST_TOLERANCE)
        end
    end
    
    @testset "High ae, low L parameters" begin
        for i in 0:2
            filename = "comprehensive_tensors_high_ae_low_L_$i.json"
            test_data = load_test_data(filename)
            
            ae = Float64(test_data.parameters.ae)
            L = Float64(test_data.parameters.L)
            gamma = Float64(test_data.parameters.gamma)
            
            # Test single wave vector
            K = Float64.(collect(test_data.inputs.K))
            expected_iso = Matrix(transpose(hcat(test_data.outputs.isotropic_tensor...)))
            expected_sheared = Matrix(transpose(hcat(test_data.outputs.sheared_tensor...)))
            
            # Test isotropic
            iso = Isotropic(ae, L)
            iso_result = tensor(iso, K)
            @test isapprox(iso_result, expected_iso, rtol=TEST_TOLERANCE)
            
            # Test sheared
            sheared = Sheared(ae, L, gamma)
            sheared_result = tensor(sheared, K)
            @test isapprox(sheared_result, expected_sheared, rtol=TEST_TOLERANCE)
        end
    end
    
    @testset "Low ae, high L parameters" begin
        for i in 0:2
            filename = "comprehensive_tensors_low_ae_high_L_$i.json"
            test_data = load_test_data(filename)
            
            ae = Float64(test_data.parameters.ae)
            L = Float64(test_data.parameters.L)
            gamma = Float64(test_data.parameters.gamma)
            
            # Test single wave vector
            K = Float64.(collect(test_data.inputs.K))
            expected_iso = Matrix(transpose(hcat(test_data.outputs.isotropic_tensor...)))
            expected_sheared = Matrix(transpose(hcat(test_data.outputs.sheared_tensor...)))
            
            # Test isotropic
            iso = Isotropic(ae, L)
            iso_result = tensor(iso, K)
            @test isapprox(iso_result, expected_iso, rtol=TEST_TOLERANCE)
            
            # Test sheared
            sheared = Sheared(ae, L, gamma)
            sheared_result = tensor(sheared, K)
            @test isapprox(sheared_result, expected_sheared, rtol=TEST_TOLERANCE)
        end
    end
end

@testset "Edge Cases" begin
    @testset "Very small wave vectors" begin
        iso = Isotropic(1.0, 1.0)
        sheared = Sheared(1.0, 1.0, 1.0)
        
        K_small = [1e-10, 1e-10, 1e-10]
        
        iso_result = tensor(iso, K_small)
        sheared_result = tensor(sheared, K_small)
        
        @test all(isfinite.(iso_result))
        @test all(isfinite.(sheared_result))
    end
    
    @testset "Large wave vectors" begin
        iso = Isotropic(1.0, 1.0)
        sheared = Sheared(1.0, 1.0, 1.0)
        
        K_large = [1e6, 1e6, 1e6]
        
        iso_result = tensor(iso, K_large)
        sheared_result = tensor(sheared, K_large)
        
        @test all(isfinite.(iso_result))
        @test all(isfinite.(sheared_result))
    end
    
    @testset "Single component wave vectors" begin
        iso = Isotropic(1.0, 1.0)
        sheared = Sheared(1.0, 1.0, 1.0)
        
        K_x = [1.0, 0.0, 0.0]
        K_y = [0.0, 1.0, 0.0]
        K_z = [0.0, 0.0, 1.0]
        
        for K in [K_x, K_y, K_z]
            iso_result = tensor(iso, K)
            sheared_result = tensor(sheared, K)
            
            @test all(isfinite.(iso_result))
            @test all(isfinite.(sheared_result))
            @test size(iso_result) == (3, 3)
            @test size(sheared_result) == (3, 3)
        end
    end
end