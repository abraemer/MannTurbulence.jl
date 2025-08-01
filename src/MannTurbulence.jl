module MannTurbulence

using FFTW

# Include tensor operations
include("tensors.jl")

# Include spectral generation
include("spectra.jl")

# Re-export main functionality from tensors
export Isotropic, Sheared, ShearedSinc, TensorGenerator
export tensor, decomp, sheared_transform
export lifetime_approx, vonkarman_spectrum
export random_tensor, freq_components
export sinc2

# Re-export main functionality from spectra
export MannParameters
export spectral_tensor, mann_spectra, generate_turbulence
export trapezoidal_integral_2d
export precompute_frequency_components, validate_turbulence_statistics

end
