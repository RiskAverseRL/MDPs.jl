using Preferences

# Enable DispatchDoctor's return-type stability checks for the whole package
# during testing. Must be set before MDPs is loaded.
set_preferences!("MDPs", "dispatch_doctor_mode" => "error")

using MDPs
using Test
using DispatchDoctor

@testset "Check type stability" begin
    @test_throws DispatchDoctor.TypeInstabilityError MDPs.test_stability(1)
end

include("src/tabular.jl")
include("src/integral.jl")
include("src/domains/inventory.jl")
include("src/domains/garnet.jl")
include("src/domains/make_domains.jl")
include("src/domains/solvers.jl")
include("src/domains/gridworld.jl")
include("src/domains/simple.jl")
include("src/integral.jl")
include("src/simulate.jl")
include("src/tabular.jl")
include("src/transient.jl")
include("src/average.jl")
