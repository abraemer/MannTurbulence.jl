# MannTurbulence.jl Benchmark Report

Generated on: 2025-08-01T16:49:12.686

## Configuration

- Grid sizes: [32]
- Warmup runs: 1
- Benchmark runs: 2
- Mann parameters: L=33.6, Γ=3.9, αε²/³=1.0
- Box dimensions: 100.0 × 100.0 × 100.0

## Tensor Generation Performance

| Tensor Type | Grid Size | Mean Time (s) | Std Dev (s) | Min Time (s) | Max Time (s) | Memory (MB) |
|-------------|-----------|---------------|-------------|--------------|--------------|-------------|
| Isotropic | 32 | 0.0003 | 0.0000 | 0.0002 | 0.0003 | 0.70 |
| Sheared | 32 | 0.0006 | 0.0000 | 0.0006 | 0.0006 | 1.65 |

## Turbulence Generation Performance

| Grid Size | Mode | Mean Time (s) | Std Dev (s) | Min Time (s) | Max Time (s) | Memory (GB) | Speedup |
|-----------|------|---------------|-------------|--------------|--------------|-------------|---------|
| 32 | Serial | 0.1502 | 0.1499 | 0.0442 | 0.2563 | 0.04 |  |
| 32 | Parallel | 0.0121 | 0.0007 | 0.0116 | 0.0126 | 0.03 | 12.42x |

## Spectra Computation Performance

| Wave Numbers | Mean Time (s) | Std Dev (s) | Min Time (s) | Max Time (s) | Memory (MB) |
|--------------|---------------|-------------|--------------|--------------|-------------|
| 10 | 0.0257 | 0.0022 | 0.0241 | 0.0272 | 73.50 |
| 50 | 0.1270 | 0.0018 | 0.1257 | 0.1283 | 367.49 |
| 100 | 0.2519 | 0.0020 | 0.2505 | 0.2533 | 734.98 |
| 200 | 0.4988 | 0.0019 | 0.4974 | 0.5002 | 1469.96 |

## Cross-Language Performance Comparison

### Julia vs Rust Performance

**Spectra Computation (100 wave numbers):**
- Julia: 0.2519 seconds
- Rust: 0.0500 seconds
- Ratio (Julia/Rust): 5.04x

## Performance Interpretation

### Scaling Analysis

**Turbulence Generation Scaling:**

### Recommendations

- Use parallel processing for grid sizes ≥ 64³ for optimal performance
- Memory usage scales as O(N³) as expected for 3D turbulence generation
- Tensor generation performance is consistent across different tensor types
- For production use, consider grid sizes up to 256³ depending on available memory
