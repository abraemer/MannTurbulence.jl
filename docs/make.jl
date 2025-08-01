using MannTurbulence
using Documenter

DocMeta.setdocmeta!(MannTurbulence, :DocTestSetup, :(using MannTurbulence); recursive=true)

makedocs(;
    modules=[MannTurbulence],
    authors="Adrian Braemer <adrian.braemer@tngtech.com>",
    sitename="MannTurbulence.jl",
    format=Documenter.HTML(;
        canonical="https://abraemer.github.io/MannTurbulence.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/abraemer/MannTurbulence.jl",
    devbranch="main",
)
