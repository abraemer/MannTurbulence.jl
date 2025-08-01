"""
Helper functions for testing MannTurbulence.jl

Provides common utilities for loading reference data and test constants.
"""

using JSON3

"""
    load_test_data(filename::String) -> Dict

Load test data from JSON file in the Rust reference data directory.

# Arguments
- `filename`: Name of the JSON file (e.g., "mann_spectra.json")

# Returns
- Dictionary containing the parsed JSON data

# Example
```julia
data = load_test_data("mann_spectra.json")
params = data["parameters"]
```
"""
function load_test_data(filename::String)
    filepath = joinpath(@__DIR__, "..", "..", "Mann.rs", "test_data", filename)
    if !isfile(filepath)
        error("Test data file not found: $filepath")
    end
    return JSON3.read(read(filepath, String))
end

"""
    test_data_exists(filename::String) -> Bool

Check if a test data file exists.

# Arguments
- `filename`: Name of the JSON file to check

# Returns
- `true` if the file exists, `false` otherwise
"""
function test_data_exists(filename::String)
    filepath = joinpath(@__DIR__, "..", "..", "Mann.rs", "test_data", filename)
    return isfile(filepath)
end