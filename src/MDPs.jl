module MDPs

include("objectives.jl")
export InfiniteH, FiniteH

include("mdp.jl")
export MDP
export qvalue, qvalues, qvalues!
export greedy, bellman, bellmangreedy
export getnext, transition

include("tabular.jl")
export TabMDP
export state_count, action_count, transition, states, actions
export bellman, greedy, greedy!
export value_iteration, value_iteration!, make_value
export policy_iteration, mrp!, mrp
export policy_iteration_sparse, mrp_sparse
export transform

include("generic.jl")
export GenericMDP, GenericState, GenericAction
export load_mdp, load_generic_mdp, make_generic_mdp, compress, state_count

include("simulation.jl")
export simulate, random_π
export Policy, PolicyStationary, PolicyMarkov
export FPolicyS, FPolicyM, TabPolicySD, TabPolicyMD
export cumulative

# ----- Domains -------
module Domains
include("domains/simple.jl")
export Simple
include("domains/inventory.jl")
export Inventory
include("domains/machine.jl")
export Machine
include("domains/gambler.jl")
export Gambler
end
# --------------------
end # module MDPs
