"""
Spectral generation for Mann turbulence model.

This module implements spectral tensor calculations, Mann spectra computation,
and turbulence generation using FFT-based methods, translated from the Rust implementation.
"""

using LinearAlgebra
using FFTW
using Random
using Statistics

"""
    MannParameters{T<:AbstractFloat}

Parameters for the Mann turbulence model with validation.

# Fields
- `L::T`: Length scale L (must be > 0)
- `Γ::T`: Lifetime parameter Γ (must be > 0) 
- `ae::T`: Energy parameter α·ε^(2/3) (must be > 0)
- `kappa::T`: Von Kármán constant κ (default: 0.4)
- `q::T`: Spectral exponent q (default: 5/3)
- `C::T`: Kolmogorov constant C (default: 1.5)
- `C_coherence::T`: Coherence constant (default: 1.0)

# Example
```julia
params = MannParameters(L=33.6, Γ=3.9, ae=1.0)
```
"""
mutable struct MannParameters{T<:AbstractFloat}
    L::T
    Γ::T
    ae::T
    kappa::T
    q::T
    C::T
    C_coherence::T
    
    function MannParameters{T}(L, Γ, ae, kappa=T(0.4), q=T(5.0/3.0), C=T(1.5), C_coherence=T(1.0)) where T<:AbstractFloat
        # Validation checks
        L > 0 || throw(ArgumentError("Length scale L must be positive, got $L"))
        Γ > 0 || throw(ArgumentError("Lifetime parameter Γ must be positive, got $Γ"))
        ae > 0 || throw(ArgumentError("Energy parameter ae must be positive, got $ae"))
        kappa > 0 || throw(ArgumentError("Von Kármán constant κ must be positive, got $kappa"))
        q > 0 || throw(ArgumentError("Spectral exponent q must be positive, got $q"))
        C > 0 || throw(ArgumentError("Kolmogorov constant C must be positive, got $C"))
        C_coherence > 0 || throw(ArgumentError("Coherence constant must be positive, got $C_coherence"))
        
        # Range checks based on typical values
        0.1 <= L <= 1000 || @warn "Length scale L=$L is outside typical range [0.1, 1000]"
        0.1 <= Γ <= 10 || @warn "Lifetime parameter Γ=$Γ is outside typical range [0.1, 10]"
        0.01 <= ae <= 100 || @warn "Energy parameter ae=$ae is outside typical range [0.01, 100]"
        
        new{T}(T(L), T(Γ), T(ae), T(kappa), T(q), T(C), T(C_coherence))
    end
end

"""
    MannParameters(L, Γ, ae; kwargs...) -> MannParameters{Float64}

Create Mann parameters with Float64 precision.
"""
MannParameters(L, Γ, ae, kappa=0.4, q=5.0/3.0, C=1.5, C_coherence=1.0) =
    MannParameters{Float64}(L, Γ, ae, kappa, q, C, C_coherence)

"""
    trapezoidal_integral_2d(f::Matrix{T}, x::Vector{T}, y::Vector{T}) -> T where T<:AbstractFloat

Compute 2D trapezoidal integral using two successive 1D trapezoidal integrations.
First integrates along the x-axis, then along the y-axis.

# Arguments
- `f`: 2D matrix representing function values at grid points
- `x`: 1D array of x-coordinates (non-uniform spacing allowed)
- `y`: 1D array of y-coordinates (non-uniform spacing allowed)

# Returns
- Approximate integral value

# Mathematical formulation
For a function f(x,y) sampled on a grid, computes:
∫∫ f(x,y) dx dy ≈ ∑ᵢ ∑ⱼ f(xᵢ,yⱼ) Δxᵢ Δyⱼ

using the trapezoidal rule for both dimensions.
"""
function trapezoidal_integral_2d(f::Matrix{T}, x::Vector{T}, y::Vector{T}) where T<:AbstractFloat
    nx, ny = size(f)
    length(x) == nx || throw(ArgumentError("x length $(length(x)) must match f rows $nx"))
    length(y) == ny || throw(ArgumentError("y length $(length(y)) must match f cols $ny"))
    
    # Step 1: Integrate along the x-axis for each fixed y
    integral_x = Vector{T}(undef, ny)
    @inbounds for j in 1:ny
        sum_x = zero(T)
        for i in 1:(nx-1)
            dx = x[i+1] - x[i]
            sum_x += 0.5 * (f[i, j] + f[i+1, j]) * dx
        end
        integral_x[j] = sum_x
    end
    
    # Step 2: Integrate the intermediate result along the y-axis
    integral = zero(T)
    @inbounds for j in 1:(ny-1)
        dy = y[j+1] - y[j]
        integral += 0.5 * (integral_x[j] + integral_x[j+1]) * dy
    end
    
    return integral
end

"""
    spectral_tensor(params::MannParameters{T}, K::AbstractVector{T}) -> Matrix{T} where T<:AbstractFloat

Generate the Mann spectral tensor for a given wave vector K.

Uses the sheared tensor implementation from tensors.jl.

# Arguments
- `params`: Mann turbulence parameters
- `K`: Wave vector [kx, ky, kz]

# Returns
- 3×3 spectral tensor matrix Φᵢⱼ(k)
"""
function spectral_tensor(params::MannParameters{T}, K::AbstractVector{T}) where T<:AbstractFloat
    length(K) == 3 || throw(ArgumentError("Wave vector K must have 3 components"))
    
    # Create sheared tensor generator with parameters
    sheared_gen = Sheared(params.ae, params.L, params.Γ)
    
    # Generate tensor using existing implementation
    return tensor(sheared_gen, K)
end

"""
    mann_spectra(kx::AbstractVector{T}, params::MannParameters{T}; 
                 nr::Int=150, ntheta::Int=30) -> Tuple{Vector{T}, Vector{T}, Vector{T}, Vector{T}} where T<:AbstractFloat

Compute Mann velocity spectra using 2D integration over polar coordinates.

# Arguments
- `kx`: Array of streamwise wave numbers
- `params`: Mann turbulence parameters
- `nr`: Number of radial integration points (default: 150)
- `ntheta`: Number of angular integration points (default: 30)

# Returns
- Tuple of (Suu, Svv, Sww, Suw) spectra arrays

# Mathematical formulation
Computes the one-dimensional spectra by integrating the 3D spectral tensor:
Sᵢⱼ(kₓ) = ∫₀^∞ ∫₀^{2π} Φᵢⱼ(kₓ, r cos θ, r sin θ) r dr dθ

where r = √(ky² + kz²) and θ = atan2(kz, ky).
"""
function mann_spectra(kx::AbstractVector{T}, params::MannParameters{T}; 
                      nr::Int=150, ntheta::Int=30) where T<:AbstractFloat
    
    # Create sheared tensor generator
    tensor_gen = Sheared(params.ae, params.L, params.Γ)
    
    # Integration grid in polar coordinates
    # Logarithmic spacing for radial component to capture wide range
    rs = T[10^x for x in range(-4.0, 7.0, length=nr)]
    thetas = range(T(0), T(2π), length=ntheta)
    
    # Pre-allocate output arrays
    uu_vals = Vector{T}(undef, length(kx))
    vv_vals = Vector{T}(undef, length(kx))
    ww_vals = Vector{T}(undef, length(kx))
    uw_vals = Vector{T}(undef, length(kx))
    
    # Process each kx value
    @inbounds for (idx, kx_val) in enumerate(kx)
        # Pre-allocate integration grids
        uu_grid = Matrix{T}(undef, nr, ntheta)
        vv_grid = Matrix{T}(undef, nr, ntheta)
        ww_grid = Matrix{T}(undef, nr, ntheta)
        uw_grid = Matrix{T}(undef, nr, ntheta)
        
        # Fill integration grids
        for (i, r) in enumerate(rs)
            for (j, theta) in enumerate(thetas)
                ky = r * cos(theta)
                kz = r * sin(theta)
                K = [kx_val, ky, kz]
                
                # Get spectral tensor
                tensor_matrix = tensor(tensor_gen, K)
                
                # Extract diagonal and off-diagonal components, multiply by r for Jacobian
                uu_grid[i, j] = r * tensor_matrix[1, 1]
                vv_grid[i, j] = r * tensor_matrix[2, 2]
                ww_grid[i, j] = r * tensor_matrix[3, 3]
                uw_grid[i, j] = r * tensor_matrix[1, 3]
            end
        end
        
        # Perform 2D trapezoidal integration
        uu_vals[idx] = trapezoidal_integral_2d(uu_grid, rs, collect(thetas))
        vv_vals[idx] = trapezoidal_integral_2d(vv_grid, rs, collect(thetas))
        ww_vals[idx] = trapezoidal_integral_2d(ww_grid, rs, collect(thetas))
        uw_vals[idx] = trapezoidal_integral_2d(uw_grid, rs, collect(thetas))
    end
    
    return (uu_vals, vv_vals, ww_vals, uw_vals)
end

"""
    generate_turbulence(params::MannParameters{T}, Lx, Ly, Lz, Nx, Ny, Nz; 
                        seed::Union{Int,Nothing}=nothing, 
                        parallel::Bool=true,
                        use_sinc_correction::Bool=false) -> Tuple{Array{T,3}, Array{T,3}, Array{T,3}} where T<:AbstractFloat

Generate 3D turbulence box using FFT-based method.

# Arguments
- `params`: Mann turbulence parameters
- `Lx, Ly, Lz`: Box dimensions
- `Nx, Ny, Nz`: Grid points in each direction
- `seed`: Random seed for reproducibility (optional)
- `parallel`: Use parallel processing (default: true)
- `use_sinc_correction`: Apply sinc correction for finite box effects (default: false)

# Returns
- Tuple of (U, V, W) velocity field arrays

# Mathematical formulation
Generates turbulence using the spectral method:
1. Create random Fourier coefficients ñ(k) ~ N(0,1)
2. Apply spectral tensor decomposition: û(k) = φ(k) · ñ(k)
3. Scale by volume and energy: û(k) *= √(8αε²/³π³/LxLyLz) * 2N
4. Inverse FFT to get physical space: u(x) = IFFT(û(k))
"""
function generate_turbulence(params::MannParameters{T}, Lx, Ly, Lz, Nx, Ny, Nz; 
                             seed::Union{Int,Nothing}=nothing,
                             parallel::Bool=true,
                             use_sinc_correction::Bool=false) where T<:AbstractFloat
    
    # Set up deterministic random number generation if seed provided
    if seed !== nothing
        Random.seed!(seed)
    end
    
    # Get frequency components
    kx, ky, kz = freq_components(Lx, Ly, Lz, Nx, Ny, Nz)
    
    # Volume scaling factor for FFT normalization
    # Factor of 2 accounts for real FFT, additional factors for energy scaling
    KVolScaleFac = T(2) * T(Nx * Ny * (Nz ÷ 2 + 1)) * sqrt(T(8) * params.ae * T(π)^3 / (Lx * Ly * Lz))
    
    # Generate random Fourier coefficients
    # Use complex Gaussian with σ = 1/√2 for each component to get unit variance
    random_field = random_tensor((Nx, Ny, Nz ÷ 2 + 1, 3), seed=seed)
    
    # Pre-allocate output arrays in Fourier space
    UVW_f = Array{Complex{T},4}(undef, Nx, Ny, Nz ÷ 2 + 1, 3)
    
    # Choose tensor generator based on correction option
    if use_sinc_correction
        tensor_gen_sinc = ShearedSinc(params.ae, params.L, params.Γ, Ly, Lz, T(1), 2)
        tensor_gen = Sheared(params.ae, params.L, params.Γ)
        
        # Apply spectral tensor with sinc correction for low frequencies
        if parallel
            Threads.@threads for i in 1:Nx
                _generate_turbulence_slice_sinc!(UVW_f, random_field, tensor_gen, tensor_gen_sinc,
                                                kx, ky, kz, KVolScaleFac, params.L, i)
            end
        else
            for i in 1:Nx
                _generate_turbulence_slice_sinc!(UVW_f, random_field, tensor_gen, tensor_gen_sinc,
                                                kx, ky, kz, KVolScaleFac, params.L, i)
            end
        end
    else
        tensor_gen = Sheared(params.ae, params.L, params.Γ)
        
        # Apply spectral tensor decomposition
        if parallel
            Threads.@threads for i in 1:Nx
                _generate_turbulence_slice!(UVW_f, random_field, tensor_gen, 
                                          kx, ky, kz, KVolScaleFac, i)
            end
        else
            for i in 1:Nx
                _generate_turbulence_slice!(UVW_f, random_field, tensor_gen, 
                                          kx, ky, kz, KVolScaleFac, i)
            end
        end
    end
    
    # Ensure zero mean (set DC component to zero)
    UVW_f[1, 1, 1, :] .= 0
    
    # Perform inverse FFT to get physical space fields
    U = _irfft3d(view(UVW_f, :, :, :, 1), Nz)
    V = _irfft3d(view(UVW_f, :, :, :, 2), Nz)  
    W = _irfft3d(view(UVW_f, :, :, :, 3), Nz)
    
    return (U, V, W)
end

"""
Helper function to process a single slice in turbulence generation.
"""
function _generate_turbulence_slice!(UVW_f, random_field, tensor_gen, kx, ky, kz, KVolScaleFac, i)
    @inbounds for j in 1:size(UVW_f, 2)
        for k in 1:size(UVW_f, 3)
            K = [kx[i], ky[j], kz[k]]
            
            # Get tensor decomposition
            tensor_decomp = decomp(tensor_gen, K)
            
            # Apply to random field
            n = view(random_field, i, j, k, :)
            UVW_f[i, j, k, :] = tensor_decomp * n
            UVW_f[i, j, k, :] .*= KVolScaleFac
        end
    end
end

"""
Helper function to process a single slice with sinc correction.
"""
function _generate_turbulence_slice_sinc!(UVW_f, random_field, tensor_gen, tensor_gen_sinc, 
                                         kx, ky, kz, KVolScaleFac, L, i)
    @inbounds for j in 1:size(UVW_f, 2)
        for k in 1:size(UVW_f, 3)
            K = [kx[i], ky[j], kz[k]]
            k_norm2 = sum(K .^ 2)
            
            # Use sinc correction for low frequencies
            tensor_decomp = if k_norm2 < 3.0 / L
                decomp(tensor_gen_sinc, K)
            else
                decomp(tensor_gen, K)
            end
            
            # Apply to random field
            n = view(random_field, i, j, k, :)
            UVW_f[i, j, k, :] = tensor_decomp * n
            UVW_f[i, j, k, :] .*= KVolScaleFac
        end
    end
end

"""
    _irfft3d(input::AbstractArray{Complex{T},3}, nz::Int) -> Array{T,3} where T<:AbstractFloat

Perform 3D inverse real FFT using FFTW with optimal performance.
"""
function _irfft3d(input::AbstractArray{Complex{T},3}, nz::Int) where T<:AbstractFloat
    # Create FFTW plans for optimal performance
    # Note: FFTW.jl handles the complex conjugate symmetry automatically
    
    # First, inverse FFT along x and y (complex-to-complex)
    temp1 = ifft(input, (1, 2))
    
    # Then, inverse real FFT along z (complex-to-real)
    result = irfft(temp1, nz, 3)
    
    return result
end

"""
    precompute_frequency_components(Lx, Ly, Lz, Nx, Ny, Nz) -> Tuple

Precompute and cache frequency components for performance optimization.
"""
function precompute_frequency_components(Lx, Ly, Lz, Nx, Ny, Nz)
    return freq_components(Lx, Ly, Lz, Nx, Ny, Nz)
end

"""
    validate_turbulence_statistics(U, V, W; expected_mean=0.0, tolerance=1e-10) -> Dict

Validate statistical properties of generated turbulence.

# Arguments
- `U, V, W`: Velocity field components
- `expected_mean`: Expected mean value (default: 0.0)
- `tolerance`: Tolerance for mean check (default: 1e-10)

# Returns
- Dictionary with statistical properties and validation results
"""
function validate_turbulence_statistics(U, V, W; expected_mean=0.0, tolerance=1e-10)
    stats = Dict{String, Any}()
    
    # Compute basic statistics
    stats["mean_U"] = mean(U)
    stats["mean_V"] = mean(V) 
    stats["mean_W"] = mean(W)
    
    stats["var_U"] = var(U)
    stats["var_V"] = var(V)
    stats["var_W"] = var(W)
    
    stats["std_U"] = std(U)
    stats["std_V"] = std(V)
    stats["std_W"] = std(W)
    
    # Validation checks
    stats["mean_valid"] = all(abs.([stats["mean_U"], stats["mean_V"], stats["mean_W"]]) .< tolerance)
    stats["finite_check"] = all(isfinite.([U; V; W]))
    
    # Energy content
    stats["total_energy"] = 0.5 * (stats["var_U"] + stats["var_V"] + stats["var_W"])
    
    return stats
end

# Export main functions and types
export MannParameters
export spectral_tensor, mann_spectra, generate_turbulence
export trapezoidal_integral_2d
export precompute_frequency_components, validate_turbulence_statistics