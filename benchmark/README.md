# MannTurbulence.jl Benchmark Suite

This directory contains a comprehensive benchmark suite for the MannTurbulence.jl package, designed to measure performance across different operations, grid sizes, and execution modes.

## Overview

The benchmark suite provides:

1. **Performance Metrics**: Comprehensive timing and memory usage measurements
2. **Scalability Analysis**: Performance scaling across different grid sizes
3. **Parallel vs Serial Comparison**: Speedup analysis for parallel execution
4. **Cross-Language Comparison**: Performance comparison with Rust implementation
5. **Automated Reporting**: Markdown reports with tables and SVG plots
6. **Configurable Parameters**: Command-line configuration for flexible testing

## Quick Start

### Basic Usage

Run the benchmark suite with default settings:

```bash
cd MannTurbulence/benchmark
julia benchmarks.jl
```

This will:
- Test grid sizes: 32³, 64³, 128³
- Run 3 warmup iterations and 5 benchmark iterations
- Save results to `results/` directory
- Generate comprehensive report and plots

### Custom Configuration

```bash
# Test larger grid sizes with more iterations
julia benchmarks.jl --grid-sizes "64,128,256" --benchmark-runs 10

# Save results to custom directory
julia benchmarks.jl --output-dir my_benchmark_results

# Quick test with smaller grids
julia benchmarks.jl --grid-sizes "32,64" --warmup-runs 1 --benchmark-runs 3
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--grid-sizes` | Comma-separated grid sizes to test | `32,64,128` |
| `--warmup-runs` | Number of warmup iterations | `3` |
| `--benchmark-runs` | Number of benchmark iterations | `5` |
| `--output-dir` | Output directory for results | `results` |
| `--help`, `-h` | Show help message | - |

## Benchmark Categories

### 1. Tensor Generation Performance

Tests the performance of different spectral tensor types:

- **Isotropic Tensors**: Basic isotropic turbulence spectral tensors
- **Sheared Tensors**: Mann model sheared spectral tensors

**Metrics Measured:**
- Mean computation time per tensor
- Memory allocation patterns
- Scaling with grid size

### 2. Turbulence Generation Performance

Tests full 3D turbulence field generation:

- **Serial Execution**: Single-threaded turbulence generation
- **Parallel Execution**: Multi-threaded turbulence generation using `Threads.@threads`

**Metrics Measured:**
- Total generation time
- Memory usage (scales as O(N³))
- Parallel speedup factors
- Statistical validation of generated fields

### 3. Spectra Computation Performance

Tests Mann velocity spectra computation:

- **Variable Wave Numbers**: Tests with 10, 50, 100, 200 wave numbers
- **Integration Performance**: 2D polar coordinate integration

**Metrics Measured:**
- Computation time vs number of wave numbers
- Memory efficiency
- Numerical accuracy validation

## Output Files

The benchmark suite generates several output files in the specified directory:

### `benchmark_report.md`
Comprehensive markdown report containing:
- Configuration summary
- Performance tables for all benchmark categories
- Cross-language comparison with Rust
- Scaling analysis and recommendations
- Performance interpretation guidelines

### `benchmark_data.json`
Raw benchmark data in JSON format for further analysis:
```json
[
  {
    "name": "turbulence_generation",
    "grid_size": 128,
    "mean_time": 2.345,
    "std_time": 0.123,
    "min_time": 2.201,
    "max_time": 2.567,
    "memory_usage": 134217728,
    "parallel": true,
    "timestamp": "2024-01-01T12:00:00.000",
    "additional_info": {...}
  }
]
```

### SVG Plot Files (if Plots.jl available)
- `turbulence_scaling.svg`: Performance scaling with grid size
- `parallel_speedup.svg`: Serial vs parallel speedup analysis
- `memory_scaling.svg`: Memory usage scaling patterns

## Performance Interpretation

### Expected Scaling Behavior

1. **Turbulence Generation**: Should scale as O(N³ log N) due to FFT operations
2. **Memory Usage**: Scales as O(N³) for 3D grids
3. **Parallel Speedup**: Depends on available CPU cores and memory bandwidth

### Typical Performance Ranges

Based on modern hardware (8-core CPU, 16GB RAM):

| Grid Size | Serial Time | Parallel Time | Memory Usage |
|-----------|-------------|---------------|--------------|
| 32³       | ~0.1s       | ~0.05s        | ~50MB        |
| 64³       | ~0.8s       | ~0.3s         | ~400MB       |
| 128³      | ~6s         | ~2s           | ~3GB         |
| 256³      | ~50s        | ~15s          | ~25GB        |

### Cross-Language Comparison

The benchmark suite includes comparison with the Rust implementation:

- **Spectra Computation**: Julia typically 2-5x slower than Rust
- **Turbulence Generation**: Julia typically 1.5-3x slower than Rust
- **Memory Efficiency**: Similar memory usage patterns

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce grid sizes: `--grid-sizes "32,64"`
   - Use smaller benchmark runs: `--benchmark-runs 3`

2. **Long Execution Times**
   - Start with smaller grids for testing
   - Use fewer warmup runs: `--warmup-runs 1`

3. **Missing Plots**
   - Install Plots.jl: `julia -e 'using Pkg; Pkg.add("Plots")'`
   - Plots are optional; benchmark data is still generated

### Performance Optimization Tips

1. **Julia Optimization**
   - Ensure Julia is compiled with optimizations
   - Use `julia -O3` for maximum performance
   - Pre-compile packages: `julia -e 'using MannTurbulence'`

2. **System Configuration**
   - Close unnecessary applications
   - Ensure sufficient RAM for large grids
   - Use SSD storage for faster I/O

3. **Threading Configuration**
   - Set `JULIA_NUM_THREADS` environment variable
   - Example: `export JULIA_NUM_THREADS=8`

## Extending the Benchmark Suite

### Adding New Benchmarks

To add new benchmark categories:

1. Create a new benchmark function following the pattern:
```julia
function benchmark_new_feature(config::BenchmarkConfig) -> Vector{BenchmarkResult}
    # Implementation
end
```

2. Add the benchmark to the main execution in `main()`:
```julia
new_results = benchmark_new_feature(config)
append!(all_results, new_results)
```

3. Update the report generation to include new results

### Custom Analysis

The JSON output can be loaded for custom analysis:

```julia
using JSON3

# Load benchmark data
data = JSON3.read("results/benchmark_data.json")

# Custom analysis
turbulence_results = filter(r -> r.name == "turbulence_generation", data)
# ... your analysis code
```

## Dependencies

### Required Packages
- `MannTurbulence.jl`: The main package being benchmarked
- `BenchmarkTools.jl`: High-precision timing measurements
- `JSON3.jl`: JSON output generation
- `Statistics.jl`: Statistical analysis
- `Printf.jl`: Formatted output
- `Dates.jl`: Timestamp generation
- `LinearAlgebra.jl`: Linear algebra operations
- `FFTW.jl`: Fast Fourier Transform operations

### Optional Packages
- `Plots.jl`: Plot generation (SVG output)
- `StatsPlots.jl`: Statistical plotting extensions

### Installation

```julia
using Pkg
Pkg.add(["BenchmarkTools", "JSON3", "Plots", "StatsPlots"])
```

## Contributing

To contribute improvements to the benchmark suite:

1. Fork the repository
2. Create a feature branch
3. Add your improvements with tests
4. Submit a pull request

### Guidelines
- Follow existing code style and patterns
- Add documentation for new features
- Include example usage in README updates
- Test with multiple grid sizes and configurations

## License

This benchmark suite is part of MannTurbulence.jl and follows the same license terms.

## References

1. Mann, J. (1994). The spatial structure of neutral atmospheric surface-layer turbulence. Journal of Fluid Mechanics, 273, 141-168.
2. Mann, J. (1998). Wind field simulation. Probabilistic Engineering Mechanics, 13(4), 269-282.
3. Peña, A., Gryning, S. E., & Mann, J. (2010). On the length-scale of the wind profile. Quarterly Journal of the Royal Meteorological Society, 136(653), 2119-2131.

---

For questions or issues, please refer to the main MannTurbulence.jl documentation or open an issue on the project repository.