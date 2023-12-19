using Documenter, MDPs

makedocs(sitename="MDPs.jl",
         modules = [MDPs],
         format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
         pages = ["index.md",
                  "simulation.md"]
         )

deploydocs(;
    repo = "github.com/RiskAverseRL/RiskMeasures.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"],
    push_preview=true,
)
