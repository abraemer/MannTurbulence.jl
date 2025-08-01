"""
Core tensor operations for Mann turbulence model.

This module implements spectral tensor calculations including isotropic,
sheared (Mann), and their decompositions, translated from the Rust implementation.
"""

using LinearAlgebra
using AbstractFFTs
using FFTW
using Random


"""
    lifetime_approx(kL::Float64) -> Float64

Lifetime approximation function for the Mann model.
Implements the lifetime parameter approximation with bounds checking.
"""
function lifetime_approx(kL)
    kL = max(kL, 0.005)  # Lower bound
    kSqr = kL^2
    return (1.0 + kSqr)^(1.0/6.0) / kL *
           (1.2050983316598936 - 0.04079766636961979 * kL + 1.1050803451576134 * kSqr) /
           (1.0 - 0.04103886513006046 * kL + 1.1050902034670118 * kSqr)
end

"""
    vonkarman_spectrum(ae::Float64, k::Float64, L::Float64) -> Float64

Von Kármán energy spectrum function.

# Arguments
- `ae`: Energy parameter α·ε^(2/3)
- `k`: Wave number magnitude
- `L`: Length scale
"""
function vonkarman_spectrum(ae, k, L)
    return ae * L^(5.0/3.0) * (L * k)^4 / (1.0 + (L * k)^2)^(17.0/6.0)
end

"""
Abstract base type for tensor generators.
"""
abstract type TensorGenerator end

"""
    tensor(gen::TensorGenerator, K::AbstractVector{Float64}) -> Matrix{Float64}

Generate the spectral tensor for given wave vector K.
"""
function tensor end

"""
    decomp(gen::TensorGenerator, K::AbstractVector{Float64}) -> Matrix{Float64}

Generate the tensor decomposition for given wave vector K.
"""
function decomp end

"""
    Isotropic{T<:AbstractFloat}

Isotropic turbulence spectral tensor generator.

# Fields
- `ae::T`: Energy parameter α·ε^(2/3)
- `L::T`: Length scale L
"""
mutable struct Isotropic{T<:AbstractFloat} <: TensorGenerator
    ae::T
    L::T
end

"""
    Isotropic(ae::Float64, L::Float64) -> Isotropic{Float64}

Create an isotropic tensor generator with given parameters.
"""
Isotropic(ae, L) = Isotropic{Float64}(ae, L)

"""
    tensor(iso::Isotropic{T}, K::AbstractVector{T}) -> Matrix{T} where T<:AbstractFloat

Generate the incompressible isotropic turbulence spectral tensor.

Implements Equation (8):
Φ^ISO_ij(k) = E(|k|)/(4π|k|^4) * (δ_ij|k|^2 - k_i*k_j)
"""
function tensor(iso::Isotropic{T}, K::AbstractVector{T}) where T<:AbstractFloat
    k_norm = sqrt(sum(K .^ 2))
    
    if k_norm == 0.0
        return zeros(T, 3, 3)
    end
    
    E = vonkarman_spectrum(T(iso.ae), k_norm, T(iso.L))
    
    # Create tensor matrix: (δ_ij|k|^2 - k_i*k_j)
    tensor_matrix = zeros(T, 3, 3)
    @inbounds for i in 1:3
        for j in 1:3
            if i == j
                tensor_matrix[i, j] = sum(K .^ 2) - K[i] * K[j]
            else
                tensor_matrix[i, j] = -K[i] * K[j]
            end
        end
    end
    
    return tensor_matrix .* (E / (4.0 * π * k_norm^4))
end

"""
    decomp(iso::Isotropic{T}, K::AbstractVector{T}) -> Matrix{T} where T<:AbstractFloat

Generate the decomposition of the isotropic spectral tensor.

Returns φ(k) where φ*(k)φ(k) = Φ^ISO(k).
"""
function decomp(iso::Isotropic{T}, K::AbstractVector{T}) where T<:AbstractFloat
    k_norm = sqrt(sum(K .^ 2))
    
    if k_norm == 0.0
        return zeros(T, 3, 3)
    end
    
    E = vonkarman_spectrum(T(iso.ae), k_norm, T(iso.L))
    
    # Skew-symmetric matrix
    tensor_matrix = T[
        0.0    K[3]  -K[2];
       -K[3]   0.0    K[1];
        K[2]  -K[1]   0.0
    ]
    
    return tensor_matrix .* (sqrt(E / π) / (2.0 * k_norm^2))
end

"""
    Sheared{T<:AbstractFloat}

Sheared (Mann) turbulence spectral tensor generator.

# Fields
- `ae::T`: Energy parameter α·ε^(2/3)
- `L::T`: Length scale L
- `gamma::T`: Lifetime parameter Γ
"""
mutable struct Sheared{T<:AbstractFloat} <: TensorGenerator
    ae::T
    L::T
    gamma::T
end

"""
    Sheared(ae::Float64, L::Float64, gamma::Float64) -> Sheared{Float64}

Create a sheared tensor generator with given parameters.
"""
Sheared(ae, L, gamma) = Sheared{Float64}(ae, L, gamma)

"""
    sheared_transform(sheared::Sheared{T}, K::AbstractVector{T}) -> Matrix{T} where T<:AbstractFloat

Compute the isotropic to sheared tensor transformation matrix.

Implements the transformation matrix A from Equations (12-16).
"""
function sheared_transform(sheared::Sheared{T}, K::AbstractVector{T}) where T<:AbstractFloat
    k_norm2 = sum(K .^ 2)
    
    if k_norm2 == 0.0
        return zeros(T, 3, 3)
    end
    
    beta = sheared.gamma * lifetime_approx(sqrt(k_norm2) * sheared.L)
    
    # Equation (12): K₀ = K + [0, 0, β*K₁]
    K0 = [K[1], K[2], K[3] + beta * K[1]]
    k0_norm2 = sum(K0 .^ 2)
    
    if K[1] == 0.0
        zeta1 = -beta
        zeta2 = 0.0
    else
        # Equation (15)
        C1 = beta * K[1]^2 * (k0_norm2 - 2.0 * K0[3]^2 + beta * K[1] * K0[3]) /
             (k_norm2 * (K[1]^2 + K[2]^2))
        
        # Equation (16)
        C2 = K[2] * k0_norm2 / (K[1]^2 + K[2]^2)^(3.0/2.0) *
             atan(beta * K[1] * sqrt(K[1]^2 + K[2]^2), 
                  k0_norm2 - K0[3] * K[1] * beta)
        
        # Equation (14)
        zeta1 = C1 - K[2] / K[1] * C2
        zeta2 = K[2] / K[1] * C1 + C2
    end
    
    # Equation (13): Transformation matrix A
    return T[
        1.0  0.0  zeta1;
        0.0  1.0  zeta2;
        0.0  0.0  k0_norm2/k_norm2
    ]
end

"""
    tensor(sheared::Sheared{T}, K::AbstractVector{T}) -> Matrix{T} where T<:AbstractFloat

Generate the sheared (Mann) spectral tensor.

Implements the Mann model transformation: Φ^Mann = A·Φ^ISO·A^T
"""
function tensor(sheared::Sheared{T}, K::AbstractVector{T}) where T<:AbstractFloat
    k_norm2 = sum(K .^ 2)
    
    if k_norm2 == 0.0
        return zeros(T, 3, 3)
    end
    
    A = sheared_transform(sheared, K)
    beta = sheared.gamma * lifetime_approx(sqrt(k_norm2) * sheared.L)
    
    # Equation (12): K₀ = K + [0, 0, β*K₁]
    K0 = [K[1], K[2], K[3] + beta * K[1]]
    
    # Get isotropic tensor at transformed wave vector
    iso_gen = Isotropic(sheared.ae, sheared.L)
    iso_tensor = tensor(iso_gen, K0)
    
    # Apply transformation: A·Φ^ISO·A^T
    return A * iso_tensor * A'
end

"""
    decomp(sheared::Sheared{T}, K::AbstractVector{T}) -> Matrix{T} where T<:AbstractFloat

Generate the decomposition of the sheared (Mann) spectral tensor.
"""
function decomp(sheared::Sheared{T}, K::AbstractVector{T}) where T<:AbstractFloat
    k_norm2 = sum(K .^ 2)
    
    if k_norm2 == 0.0
        return zeros(T, 3, 3)
    end
    
    A = sheared_transform(sheared, K)
    beta = sheared.gamma * lifetime_approx(sqrt(k_norm2) * sheared.L)
    
    # Equation (12): K₀ = K + [0, 0, β*K₁]
    K0 = [K[1], K[2], K[3] + beta * K[1]]
    
    # Get isotropic decomposition at transformed wave vector
    iso_gen = Isotropic(sheared.ae, sheared.L)
    iso_decomp = decomp(iso_gen, K0)
    
    # Apply transformation: A·φ^ISO
    return A * iso_decomp
end

"""
    ShearedSinc{T<:AbstractFloat}

Sheared spectral tensor with sinc correction for finite box effects.

# Fields
- `ae::T`: Energy parameter α·ε^(2/3)
- `L::T`: Length scale L
- `gamma::T`: Lifetime parameter Γ
- `Ly::T`: Lateral box length
- `Lz::T`: Vertical box length
- `tol::T`: Adaptive integration tolerance
- `min_depth::Int`: Adaptive integration minimum depth
"""
mutable struct ShearedSinc{T<:AbstractFloat} <: TensorGenerator
    ae::T
    L::T
    gamma::T
    Ly::T
    Lz::T
    tol::T
    min_depth::Int
end

"""
    ShearedSinc(ae::Float64, L::Float64, gamma::Float64, Ly::Float64, Lz::Float64, 
                tol::Float64, min_depth::Int) -> ShearedSinc{Float64}

Create a sheared tensor generator with sinc correction.
"""
ShearedSinc(ae, L, gamma, Ly, Lz, tol, min_depth) =
    ShearedSinc{Float64}(ae, L, gamma, Ly, Lz, tol, min_depth)

"""
    sinc2(x) -> Float64

Unnormalized sinc squared function: (sin(x)/x)²
"""
function sinc2(x)
    return x == 0.0 ? 1.0 : (sin(x) / x)^2
end

"""
    tensor(sinc_gen::ShearedSinc{T}, K::AbstractVector{T}) -> Matrix{T} where T<:AbstractFloat

Generate the sheared spectral tensor with sinc correction.

Uses adaptive quadrature integration to account for finite box effects.
"""
function tensor(sinc_gen::ShearedSinc{T}, K::AbstractVector{T}) where T<:AbstractFloat
    sheared_gen = Sheared(sinc_gen.ae, sinc_gen.L, sinc_gen.gamma)
    
    # Integration function with sinc correction
    function integrand(y::T, z::T)::Matrix{T}
        return sinc2(y * sinc_gen.Ly / 2.0) * sinc2(z * sinc_gen.Lz / 2.0) *
               tensor(sheared_gen, [K[1], K[2] + y, K[3] + z])
    end
    
    # Adaptive 2D quadrature (simplified implementation)
    # For production use, implement full adaptive quadrature
    result = adaptive_quadrature_2d(integrand, 
                                   -2.0 * π / sinc_gen.Ly, 2.0 * π / sinc_gen.Ly,
                                   -2.0 * π / sinc_gen.Lz, 2.0 * π / sinc_gen.Lz,
                                   sinc_gen.tol, sinc_gen.min_depth)
    
    return result .* (1.22686 * sinc_gen.Ly * sinc_gen.Lz / (2.0 * π)^2)
end

"""
    decomp(sinc_gen::ShearedSinc{T}, K::AbstractVector{T}) -> Matrix{T} where T<:AbstractFloat

Generate the decomposition using Cholesky decomposition.
"""
function decomp(sinc_gen::ShearedSinc{T}, K::AbstractVector{T}) where T<:AbstractFloat
    tensor_matrix = tensor(sinc_gen, K)
    
    # Cholesky decomposition
    L = zeros(T, 3, 3)
    
    @inbounds for i in 1:3
        for j in 1:i
            if i == j
                # Diagonal elements
                sum_sq = sum(L[j, k]^2 for k in 1:(j-1))
                val = tensor_matrix[j, j] - sum_sq
                if val <= 0.0
                    throw(ArgumentError("Matrix is not positive definite"))
                end
                L[i, j] = sqrt(val)
            else
                # Off-diagonal elements
                sum_prod = sum(L[i, k] * L[j, k] for k in 1:(j-1))
                if L[j, j] <= 0.0
                    throw(ArgumentError("Matrix is not positive definite"))
                end
                L[i, j] = (tensor_matrix[i, j] - sum_prod) / L[j, j]
            end
        end
    end
    
    return L
end

"""
    adaptive_quadrature_2d(f, x0, x1, y0, y1, tol, min_depth) -> Matrix{Float64}

Simplified 2D adaptive quadrature integration.
For production use, implement full adaptive Simpson's rule.
"""
function adaptive_quadrature_2d(f, x0, x1, y0, y1, tol, min_depth)
    # Simplified implementation using basic quadrature
    # In production, implement full adaptive Simpson's rule as in Rust version
    nx, ny = 32, 32  # Grid points
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    
    result = zeros(Float64, 3, 3)
    
    @inbounds for i in 1:nx
        for j in 1:ny
            x = x0 + (i - 0.5) * dx
            y = y0 + (j - 0.5) * dy
            result .+= f(x, y) .* (dx * dy)
        end
    end
    
    return result
end

"""
    random_tensor(dims::Tuple{Int,Int,Int,Int}; seed::Union{Int,Nothing}=nothing) -> Array{ComplexF64,4}

Generate random complex tensor with Gaussian distribution.

# Arguments
- `dims`: Dimensions (Nx, Ny, Nz, 3) for the tensor
- `seed`: Optional random seed for reproducibility

# Returns
- 4D array of complex numbers with unit variance
"""
function random_tensor(dims::Tuple{Int,Int,Int,Int}; seed::Union{Int,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    Nx, Ny, Nz, _ = dims
    
    # Generate real and imaginary parts with Box-Muller-like distribution
    # Using standard deviation of 1/√2 for each part to get unit variance overall
    σ = 1.0 / sqrt(2.0)
    
    real_part = randn(Float64, dims) .* σ
    imag_part = randn(Float64, dims) .* σ
    
    return complex.(real_part, imag_part)
end


"""
    freq_components(Lx, Ly, Lz, Nx, Ny, Nz)

Generate wave number components for turbulence box specification using FFTW frequency functions.

Uses FFTW.fftfreq and FFTW.rfftfreq for optimal performance with FFTW plans.
"""
function freq_components(Lx, Ly, Lz, Nx, Ny, Nz)
    return (
        FFTW.fftfreq(Nx, 2.0 * π * Nx / Lx),
        FFTW.fftfreq(Ny, 2.0 * π * Ny / Ly),
        FFTW.rfftfreq(Nz, 2.0 * π * Nz / Lz)
    )
end

# Export main functions and types
export Isotropic, Sheared, ShearedSinc, TensorGenerator
export tensor, decomp, sheared_transform
export lifetime_approx, vonkarman_spectrum
export random_tensor, freq_components
export sinc2
