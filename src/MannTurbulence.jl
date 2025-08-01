module MannTurbulence

using FFTW

# Include tensor operations
include("tensors.jl")

# Re-export main functionality
export Isotropic, Sheared, ShearedSinc, TensorGenerator
export tensor, decomp, sheared_transform
export lifetime_approx, vonkarman_spectrum
export random_tensor, freq_components
export sinc2

end
