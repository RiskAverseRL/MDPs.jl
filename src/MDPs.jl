module MDPs

using DispatchDoctor: @stable

# The @stable macro checks that the return types of the wrapped methods are
# type stable. It is disabled by default and enabled during the unit tests
# through the "dispatch_doctor_mode" preference (see test/runtests.jl).
@stable default_mode = "disable" begin

    include("objectives.jl")
    export InfiniteH, FiniteH, Markov, Stationary, MarkovDet, StationaryDet
    export TotalReward
    
    include("models/mdp.jl")
    export MDP
    export getnext, transition
    export valuefunction

    include("models/tabular.jl")
    export TabMDP
    export state_count, action_count, states, actions
    export save_mdp

    include("models/integral.jl")
    export IntMDP, IntState, IntAction
    export load_int_mdp, make_int_mdp, compress

    include("valuefunction/valuefunction.jl")
    export make_value

    include("valuefunction/bellman.jl")
    export qvalue, qvalues, qvalues!
    export greedy, greedy!, bellman, bellmangreedy

    include("algorithms/valueiteration.jl")
    export value_iteration, value_iteration!
    
    include("algorithms/mrp.jl")
    export mrp!, mrp, mrp_sparse
    export stationary_matrix, stationary_dist_sparse, decompose_chain
    
    include("algorithms/policyiteration.jl")
    export policy_iteration, policy_iteration_sparse, modified_policy_iteration
    
    include("algorithms/linprogsolve.jl")
    export lp_solve
    
    include("algorithms/transient.jl")
    export lp_solve, anytransient, alltransient
    export isterminal
    
    include("simulation.jl")
    export simulate, random_π
    export Policy, PolicyStationary, PolicyMarkov
    export FPolicyS, FPolicyM, TabPolicySD, TabPolicyMD
    export cumulative
    export Transition

    # a function solely used to check that the stability checks work
    test_stability(x::Integer) = x < 0 ? float(x) : x
    
end # @stable

# methods removed type stability check (DataFramesMeta problems)
include("models/unstable.jl")
export load_mdp

# ----- Domains -------
module Domains
using DispatchDoctor: @stable
@stable default_mode = "disable" begin
    include("domains/simple.jl")
    export Simple
    include("domains/garnet.jl")
    export Garnet
    include("domains/inventory.jl")
    export Inventory
    include("domains/machine.jl")
    export Machine
    include("domains/gambler.jl")
    export Gambler
    include("domains/gridworld.jl")
    export GridWorld
end # @stable
end
export Domains
# --------------------
end # module MDPs
