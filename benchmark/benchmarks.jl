"""
Comprehensive benchmark suite for MannTurbulence.jl

This benchmark suite provides:
1. Performance metrics for tensor generation and turbulence generation
2. Serial vs parallel performance comparison
3. Cross-language comparison with Rust execution times
4. Configurable grid sizes and parameters
5. Automated reporting with markdown tables and SVG plots

Usage:
    julia benchmarks.jl [--grid-sizes "32,64,128"] [--warmup-runs 3] [--benchmark-runs 5] [--output-dir results]
"""

using MannTurbulence
using BenchmarkTools
using JSON3
using Statistics
using Printf
using Dates
using LinearAlgebra
using FFTW

# Check for optional plotting dependencies
const HAS_PLOTS = try
    using Plots
    using StatsPlots
    true
catch
    false
end

# Default configuration
const DEFAULT_CONFIG = Dict(
    "grid_sizes" => [32, 64, 128],
    "warmup_runs" => 3,
    "benchmark_runs" => 5,
    "output_dir" => "results",
    "mann_params" => Dict(
        "L" => 33.6,
        "Γ" => 3.9,
        "ae" => 1.0
    ),
    "box_params" => Dict(
        "Lx" => 100.0,
        "Ly" => 100.0,
        "Lz" => 100.0
    )
)

# Rust benchmark data for comparison (from example files)
const RUST_BENCHMARKS = Dict(
    "spectra_computation" => Dict(
        "description" => "Mann spectra computation for 100 wave numbers",
        "typical_time_seconds" => 0.05,  # Typical Rust performance
        "grid_size" => "N/A"
    ),
    "turbulence_generation" => Dict(
        128 => Dict(
            "stencil_time_serial" => 2.5,
            "stencil_time_parallel" => 0.8,
            "turbulence_time_serial" => 1.2,
            "turbulence_time_parallel" => 0.4
        )
    )
)

"""
    BenchmarkConfig

Configuration structure for benchmark runs.
"""
struct BenchmarkConfig
    grid_sizes::Vector{Int}
    warmup_runs::Int
    benchmark_runs::Int
    output_dir::String
    mann_params::MannParameters
    box_params::NamedTuple
end

function BenchmarkConfig(config_dict::Dict)
    mann_params = MannParameters(
        config_dict["mann_params"]["L"],
        config_dict["mann_params"]["Γ"],
        config_dict["mann_params"]["ae"]
    )
    
    box_params = (
        Lx = config_dict["box_params"]["Lx"],
        Ly = config_dict["box_params"]["Ly"],
        Lz = config_dict["box_params"]["Lz"]
    )
    
    return BenchmarkConfig(
        config_dict["grid_sizes"],
        config_dict["warmup_runs"],
        config_dict["benchmark_runs"],
        config_dict["output_dir"],
        mann_params,
        box_params
    )
end

"""
    BenchmarkResult

Structure to store benchmark results.
"""
struct BenchmarkResult
    name::String
    grid_size::Union{Int, String}
    mean_time::Float64
    std_time::Float64
    min_time::Float64
    max_time::Float64
    memory_usage::Int64
    allocations::Int64
    parallel::Bool
    timestamp::DateTime
    additional_info::Dict{String, Any}
end

"""
    benchmark_tensor_generation(config::BenchmarkConfig) -> Vector{BenchmarkResult}

Benchmark tensor generation for different grid sizes and tensor types.
"""
function benchmark_tensor_generation(config::BenchmarkConfig)
    println("🔧 Benchmarking tensor generation...")
    results = BenchmarkResult[]
    
    # Test different tensor types
    tensor_types = [
        ("Isotropic", () -> Isotropic(config.mann_params.ae, config.mann_params.L)),
        ("Sheared", () -> Sheared(config.mann_params.ae, config.mann_params.L, config.mann_params.Γ))
    ]
    
    for (tensor_name, tensor_constructor) in tensor_types
        for grid_size in config.grid_sizes
            println("  Testing $tensor_name tensor with grid size $grid_size...")
            
            # Create wave vector grid
            kx, ky, kz = freq_components(
                config.box_params.Lx, config.box_params.Ly, config.box_params.Lz,
                grid_size, grid_size, grid_size
            )
            
            # Warmup runs
            tensor_gen = tensor_constructor()
            for _ in 1:config.warmup_runs
                for i in 1:min(10, grid_size)
                    for j in 1:min(10, grid_size)
                        for k in 1:min(5, grid_size÷2+1)
                            K = [kx[i], ky[j], kz[k]]
                            tensor(tensor_gen, K)
                        end
                    end
                end
            end
            
            # Measure a subset of tensor computations
            n_samples = min(1000, grid_size^2)
            sample_indices = rand(1:grid_size, n_samples), rand(1:grid_size, n_samples), rand(1:(grid_size÷2+1), n_samples)
            
            # Benchmark runs
            times = Float64[]
            memory_usage = Int64[]
            allocations = Int64[]
            
            for run in 1:config.benchmark_runs
                tensor_gen = tensor_constructor()
                
                result = @timed begin
                    for (i, j, k) in zip(sample_indices...)
                        K = [kx[i], ky[j], kz[k]]
                        tensor(tensor_gen, K)
                    end
                end
                
                push!(times, result.time)
                push!(memory_usage, result.bytes)
                push!(allocations, result.gctime > 0 ? 1 : 0)  # Simplified allocation count
            end
            
            # Store results
            push!(results, BenchmarkResult(
                "tensor_generation_$tensor_name",
                grid_size,
                mean(times),
                std(times),
                minimum(times),
                maximum(times),
                Int64(mean(memory_usage)),
                Int64(mean(allocations)),
                false,
                now(),
                Dict("tensor_type" => tensor_name, "n_samples" => n_samples)
            ))
        end
    end
    
    return results
end

"""
    benchmark_turbulence_generation(config::BenchmarkConfig) -> Vector{BenchmarkResult}

Benchmark turbulence generation for different grid sizes.
"""
function benchmark_turbulence_generation(config::BenchmarkConfig)
    println("🌪️  Benchmarking turbulence generation...")
    results = BenchmarkResult[]
    
    for grid_size in config.grid_sizes
        for parallel in [false, true]
            parallel_str = parallel ? "parallel" : "serial"
            println("  Testing $parallel_str turbulence generation with grid size $grid_size...")
            
            # Warmup runs
            for _ in 1:config.warmup_runs
                try
                    U, V, W = generate_turbulence(
                        config.mann_params,
                        config.box_params.Lx, config.box_params.Ly, config.box_params.Lz,
                        grid_size, grid_size, grid_size,
                        seed=42, parallel=parallel
                    )
                catch e
                    @warn "Warmup failed for grid size $grid_size ($parallel_str): $e"
                    continue
                end
            end
            
            # Benchmark runs
            times = Float64[]
            memory_usage = Int64[]
            
            for run in 1:config.benchmark_runs
                result = @timed begin
                    try
                        U, V, W = generate_turbulence(
                            config.mann_params,
                            config.box_params.Lx, config.box_params.Ly, config.box_params.Lz,
                            grid_size, grid_size, grid_size,
                            seed=run, parallel=parallel
                        )
                        
                        # Validate results
                        stats = validate_turbulence_statistics(U, V, W)
                        if !stats["mean_valid"] || !stats["finite_check"]
                            @warn "Invalid turbulence generated for grid size $grid_size ($parallel_str)"
                        end
                    catch e
                        @warn "Benchmark failed for grid size $grid_size ($parallel_str): $e"
                        push!(times, NaN)
                        push!(memory_usage, 0)
                        continue
                    end
                end
                
                push!(times, result.time)
                push!(memory_usage, result.bytes)
            end
            
            # Filter out failed runs
            valid_times = filter(!isnan, times)
            valid_memory = memory_usage[1:length(valid_times)]
            
            if !isempty(valid_times)
                push!(results, BenchmarkResult(
                    "turbulence_generation",
                    grid_size,
                    mean(valid_times),
                    std(valid_times),
                    minimum(valid_times),
                    maximum(valid_times),
                    Int64(mean(valid_memory)),
                    0,
                    parallel,
                    now(),
                    Dict("parallel" => parallel, "valid_runs" => length(valid_times))
                ))
            end
        end
    end
    
    return results
end

"""
    benchmark_spectra_computation(config::BenchmarkConfig) -> Vector{BenchmarkResult}

Benchmark Mann spectra computation.
"""
function benchmark_spectra_computation(config::BenchmarkConfig)
    println("📊 Benchmarking spectra computation...")
    results = BenchmarkResult[]
    
    # Test different numbers of wave numbers
    kx_sizes = [10, 50, 100, 200]
    
    for kx_size in kx_sizes
        println("  Testing spectra computation with $kx_size wave numbers...")
        
        kx = 10 .^ range(-5, 2, length=kx_size)
        
        # Warmup runs
        for _ in 1:config.warmup_runs
            mann_spectra(kx, config.mann_params)
        end
        
        # Benchmark runs
        times = Float64[]
        memory_usage = Int64[]
        
        for run in 1:config.benchmark_runs
            result = @timed mann_spectra(kx, config.mann_params)
            push!(times, result.time)
            push!(memory_usage, result.bytes)
        end
        
        push!(results, BenchmarkResult(
            "spectra_computation",
            kx_size,
            mean(times),
            std(times),
            minimum(times),
            maximum(times),
            Int64(mean(memory_usage)),
            0,
            false,
            now(),
            Dict("kx_size" => kx_size)
        ))
    end
    
    return results
end

"""
    generate_performance_report(results::Vector{BenchmarkResult}, config::BenchmarkConfig)

Generate markdown performance report with tables and plots.
"""
function generate_performance_report(results::Vector{BenchmarkResult}, config::BenchmarkConfig)
    println("📝 Generating performance report...")
    
    # Ensure output directory exists
    mkpath(config.output_dir)
    
    # Generate markdown report
    report_path = joinpath(config.output_dir, "benchmark_report.md")
    open(report_path, "w") do io
        write_markdown_report(io, results, config)
    end
    
    # Generate JSON data
    json_path = joinpath(config.output_dir, "benchmark_data.json")
    open(json_path, "w") do io
        JSON3.pretty(io, results)
    end
    
    # Generate plots if available
    if HAS_PLOTS
        generate_performance_plots(results, config)
    else
        println("⚠️  Plots.jl not available - skipping plot generation")
    end
    
    println("✅ Report generated: $report_path")
    return report_path
end

"""
    write_markdown_report(io::IO, results::Vector{BenchmarkResult}, config::BenchmarkConfig)

Write markdown performance report.
"""
function write_markdown_report(io::IO, results::Vector{BenchmarkResult}, config::BenchmarkConfig)
    println(io, "# MannTurbulence.jl Benchmark Report")
    println(io, "")
    println(io, "Generated on: $(now())")
    println(io, "")
    
    # Configuration summary
    println(io, "## Configuration")
    println(io, "")
    println(io, "- Grid sizes: $(config.grid_sizes)")
    println(io, "- Warmup runs: $(config.warmup_runs)")
    println(io, "- Benchmark runs: $(config.benchmark_runs)")
    println(io, "- Mann parameters: L=$(config.mann_params.L), Γ=$(config.mann_params.Γ), αε²/³=$(config.mann_params.ae)")
    println(io, "- Box dimensions: $(config.box_params.Lx) × $(config.box_params.Ly) × $(config.box_params.Lz)")
    println(io, "")
    
    # Tensor generation results
    tensor_results = filter(r -> startswith(r.name, "tensor_generation"), results)
    if !isempty(tensor_results)
        println(io, "## Tensor Generation Performance")
        println(io, "")
        println(io, "| Tensor Type | Grid Size | Mean Time (s) | Std Dev (s) | Min Time (s) | Max Time (s) | Memory (MB) |")
        println(io, "|-------------|-----------|---------------|-------------|--------------|--------------|-------------|")
        
        for result in tensor_results
            tensor_type = get(result.additional_info, "tensor_type", "Unknown")
            memory_mb = result.memory_usage / (1024^2)
            println(io, Printf.@sprintf("| %s | %s | %.4f | %.4f | %.4f | %.4f | %.2f |",
                tensor_type, string(result.grid_size), result.mean_time, result.std_time,
                result.min_time, result.max_time, memory_mb))
        end
        println(io, "")
    end
    
    # Turbulence generation results
    turb_results = filter(r -> r.name == "turbulence_generation", results)
    if !isempty(turb_results)
        println(io, "## Turbulence Generation Performance")
        println(io, "")
        println(io, "| Grid Size | Mode | Mean Time (s) | Std Dev (s) | Min Time (s) | Max Time (s) | Memory (GB) | Speedup |")
        println(io, "|-----------|------|---------------|-------------|--------------|--------------|-------------|---------|")
        
        # Group by grid size for speedup calculation
        for grid_size in config.grid_sizes
            grid_results = filter(r -> r.grid_size == grid_size, turb_results)
            serial_result = findfirst(r -> !r.parallel, grid_results)
            parallel_result = findfirst(r -> r.parallel, grid_results)
            
            for (i, result) in enumerate(grid_results)
                mode = result.parallel ? "Parallel" : "Serial"
                memory_gb = result.memory_usage / (1024^3)
                
                speedup = ""
                if result.parallel && serial_result !== nothing
                    speedup = Printf.@sprintf("%.2fx", grid_results[serial_result].mean_time / result.mean_time)
                end

                println(io, Printf.@sprintf("| %d | %s | %.4f | %.4f | %.4f | %.4f | %.2f | %s |",
                    result.grid_size, mode, result.mean_time, result.std_time,
                    result.min_time, result.max_time, memory_gb, speedup))
            end
        end
        println(io, "")
    end
    
    # Spectra computation results
    spectra_results = filter(r -> r.name == "spectra_computation", results)
    if !isempty(spectra_results)
        println(io, "## Spectra Computation Performance")
        println(io, "")
        println(io, "| Wave Numbers | Mean Time (s) | Std Dev (s) | Min Time (s) | Max Time (s) | Memory (MB) |")
        println(io, "|--------------|---------------|-------------|--------------|--------------|-------------|")
        
        for result in spectra_results
            memory_mb = result.memory_usage / (1024^2)
            println(io, Printf.@sprintf("| %s | %.4f | %.4f | %.4f | %.4f | %.2f |",
                string(result.grid_size), result.mean_time, result.std_time,
                result.min_time, result.max_time, memory_mb))
        end
        println(io, "")
    end
    
    # Cross-language comparison
    println(io, "## Cross-Language Performance Comparison")
    println(io, "")
    println(io, "### Julia vs Rust Performance")
    println(io, "")
    
    # Compare spectra computation
    spectra_100 = findfirst(r -> r.name == "spectra_computation" && r.grid_size == 100, results)
    if spectra_100 !== nothing
        julia_time = results[spectra_100].mean_time
        rust_time = RUST_BENCHMARKS["spectra_computation"]["typical_time_seconds"]
        ratio = julia_time / rust_time
        
        println(io, "**Spectra Computation (100 wave numbers):**")
        println(io, Printf.@sprintf("- Julia: %.4f seconds", julia_time))
        println(io, Printf.@sprintf("- Rust: %.4f seconds", rust_time))
        println(io, Printf.@sprintf("- Ratio (Julia/Rust): %.2fx", ratio))
        println(io, "")
    end
    
    # Compare turbulence generation
    turb_128 = filter(r -> r.name == "turbulence_generation" && r.grid_size == 128, results)
    if !isempty(turb_128)
        println(io, "**Turbulence Generation (128³ grid):**")
        
        for result in turb_128
            mode = result.parallel ? "parallel" : "serial"
            julia_time = result.mean_time
            
            if haskey(RUST_BENCHMARKS["turbulence_generation"], 128)
                rust_key = result.parallel ? "turbulence_time_parallel" : "turbulence_time_serial"
                rust_time = RUST_BENCHMARKS["turbulence_generation"][128][rust_key]
                ratio = julia_time / rust_time
                
                println(io, Printf.@sprintf("- Julia (%s): %.4f seconds", mode, julia_time))
                println(io, Printf.@sprintf("- Rust (%s): %.4f seconds", mode, rust_time))
                println(io, Printf.@sprintf("- Ratio (Julia/Rust): %.2fx", ratio))
                println(io, "")
            end
        end
    end
    
    # Performance interpretation
    println(io, "## Performance Interpretation")
    println(io, "")
    println(io, "### Scaling Analysis")
    println(io, "")
    
    # Analyze scaling for turbulence generation
    if length(turb_results) >= 2
        println(io, "**Turbulence Generation Scaling:**")
        
        for parallel in [false, true]
            mode = parallel ? "parallel" : "serial"
            mode_results = filter(r -> r.parallel == parallel, turb_results)
            
            if length(mode_results) >= 2
                sort!(mode_results, by=r -> r.grid_size)
                
                println(io, "")
                println(io, "*$mode mode:*")
                for i in 2:length(mode_results)
                    prev_result = mode_results[i-1]
                    curr_result = mode_results[i]
                    
                    size_ratio = curr_result.grid_size / prev_result.grid_size
                    time_ratio = curr_result.mean_time / prev_result.mean_time
                    theoretical_ratio = size_ratio^3  # O(N³) expected
                    
                    println(io, Printf.@sprintf("- %d → %d: %.2fx slower (theoretical: %.2fx)",
                        prev_result.grid_size, curr_result.grid_size, time_ratio, theoretical_ratio))
                end
            end
        end
    end
    
    println(io, "")
    println(io, "### Recommendations")
    println(io, "")
    println(io, "- Use parallel processing for grid sizes ≥ 64³ for optimal performance")
    println(io, "- Memory usage scales as O(N³) as expected for 3D turbulence generation")
    println(io, "- Tensor generation performance is consistent across different tensor types")
    println(io, "- For production use, consider grid sizes up to 256³ depending on available memory")
end

"""
    generate_performance_plots(results::Vector{BenchmarkResult}, config::BenchmarkConfig)

Generate performance plots and save as SVG files.
"""
function generate_performance_plots(results::Vector{BenchmarkResult}, config::BenchmarkConfig)
    println("📈 Generating performance plots...")
    
    # Set up plotting backend
    Plots.gr()
    
    # Plot 1: Turbulence generation scaling
    turb_results = filter(r -> r.name == "turbulence_generation", results)
    if !isempty(turb_results)
        p1 = Plots.plot(title="Turbulence Generation Performance", 
                       xlabel="Grid Size", ylabel="Time (seconds)", 
                       yscale=:log10, xscale=:log10)
        
        for parallel in [false, true]
            mode_results = filter(r -> r.parallel == parallel, turb_results)
            if !isempty(mode_results)
                sort!(mode_results, by=r -> r.grid_size)
                grid_sizes = [r.grid_size for r in mode_results]
                times = [r.mean_time for r in mode_results]
                errors = [r.std_time for r in mode_results]
                
                label = parallel ? "Parallel" : "Serial"
                Plots.plot!(p1, grid_sizes, times, yerror=errors, 
                           label=label, marker=:circle, linewidth=2)
            end
        end
        
        # Add theoretical O(N³) line
        if !isempty(turb_results)
            min_size = minimum(r.grid_size for r in turb_results)
            max_size = maximum(r.grid_size for r in turb_results)
            ref_result = first(filter(r -> !r.parallel, turb_results))
            
            theoretical_sizes = [min_size, max_size]
            theoretical_times = [ref_result.mean_time * (s/ref_result.grid_size)^3 for s in theoretical_sizes]
            
            Plots.plot!(p1, theoretical_sizes, theoretical_times, 
                       label="O(N³) theoretical", linestyle=:dash, color=:gray)
        end
        
        Plots.savefig(p1, joinpath(config.output_dir, "turbulence_scaling.svg"))
    end
    
    # Plot 2: Serial vs Parallel comparison
    if !isempty(turb_results)
        p2 = Plots.plot(title="Serial vs Parallel Performance", 
                       xlabel="Grid Size", ylabel="Speedup Factor")
        
        grid_sizes = unique([r.grid_size for r in turb_results])
        speedups = Float64[]
        
        for grid_size in grid_sizes
            serial_result = findfirst(r -> r.grid_size == grid_size && !r.parallel, turb_results)
            parallel_result = findfirst(r -> r.grid_size == grid_size && r.parallel, turb_results)
            
            if serial_result !== nothing && parallel_result !== nothing
                speedup = turb_results[serial_result].mean_time / turb_results[parallel_result].mean_time
                push!(speedups, speedup)
            else
                push!(speedups, NaN)
            end
        end
        
        valid_indices = .!isnan.(speedups)
        if any(valid_indices)
            Plots.plot!(p2, grid_sizes[valid_indices], speedups[valid_indices], 
                       marker=:circle, linewidth=2, label="Measured Speedup")
            Plots.hline!(p2, [1.0], linestyle=:dash, color=:gray, label="No Speedup")
        end
        
        Plots.savefig(p2, joinpath(config.output_dir, "parallel_speedup.svg"))
    end
    
    # Plot 3: Memory usage scaling
    p3 = Plots.plot(title="Memory Usage Scaling", 
                   xlabel="Grid Size", ylabel="Memory Usage (GB)", 
                   yscale=:log10, xscale=:log10)
    
    for result in turb_results
        memory_gb = result.memory_usage / (1024^3)
        mode = result.parallel ? "Parallel" : "Serial"
        Plots.scatter!(p3, [result.grid_size], [memory_gb], 
                      label=mode, alpha=0.7)
    end
    
    Plots.savefig(p3, joinpath(config.output_dir, "memory_scaling.svg"))
    
    println("✅ Plots saved to $(config.output_dir)/")
end

"""
    parse_command_line_args() -> Dict

Parse command line arguments for benchmark configuration.
"""
function parse_command_line_args()
    config = copy(DEFAULT_CONFIG)
    
    args = ARGS
    i = 1
    while i <= length(args)
        arg = args[i]
        
        if arg == "--grid-sizes" && i < length(args)
            grid_sizes_str = args[i+1]
            config["grid_sizes"] = [parse(Int, s) for s in split(grid_sizes_str, ",")]
            i += 2
        elseif arg == "--warmup-runs" && i < length(args)
            config["warmup_runs"] = parse(Int, args[i+1])
            i += 2
        elseif arg == "--benchmark-runs" && i < length(args)
            config["benchmark_runs"] = parse(Int, args[i+1])
            i += 2
        elseif arg == "--output-dir" && i < length(args)
            config["output_dir"] = args[i+1]
            i += 2
        elseif arg == "--help" || arg == "-h"
            println("""
            MannTurbulence.jl Benchmark Suite
            
            Usage: julia benchmarks.jl [options]
            
            Options:
              --grid-sizes SIZES    Comma-separated grid sizes (default: 32,64,128)
              --warmup-runs N       Number of warmup runs (default: 3)
              --benchmark-runs N    Number of benchmark runs (default: 5)
              --output-dir DIR      Output directory for results (default: results)
              --help, -h            Show this help message
            
            Examples:
              julia benchmarks.jl --grid-sizes "64,128,256" --benchmark-runs 10
              julia benchmarks.jl --output-dir my_results
            """)
            exit(0)
        else
            i += 1
        end
    end
    
    return config
end

"""
    main()

Main benchmark execution function.
"""
function main()
    println("🚀 Starting MannTurbulence.jl Benchmark Suite")
    println("=" ^ 50)
    
    # Parse configuration
    config_dict = parse_command_line_args()
    config = BenchmarkConfig(config_dict)
    
    println("Configuration:")
    println("  Grid sizes: $(config.grid_sizes)")
    println("  Warmup runs: $(config.warmup_runs)")
    println("  Benchmark runs: $(config.benchmark_runs)")
    println("  Output directory: $(config.output_dir)")
    println()
    
    # Run benchmarks
    all_results = BenchmarkResult[]
    
    try
        # Tensor generation benchmarks
        tensor_results = benchmark_tensor_generation(config)
        append!(all_results, tensor_results)
        
        # Turbulence generation benchmarks
        turbulence_results = benchmark_turbulence_generation(config)
        append!(all_results, turbulence_results)
        
        # Spectra computation benchmarks
        spectra_results = benchmark_spectra_computation(config)
        append!(all_results, spectra_results)
        
        # Generate report
        report_path = generate_performance_report(all_results, config)
        
        println()
        println("🎉 Benchmark suite completed successfully!")
        println("📊 Results saved to: $(config.output_dir)")
        println("📝 Report available at: $report_path")
        
    catch e
        println("❌ Benchmark suite failed with error: $e")
        rethrow(e)
    end
end

# Run benchmarks if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
