module MDPs

include("objectives.jl")
export InfiniteH, FiniteH, Markov, Stationary, MarkovDet, StationaryDet

include("models/mdp.jl")
export MDP
export getnext, transition, isterminal
export valuefunction


include("models/tabular.jl")
export TabMDP
export state_count, action_count, states, actions
export save_mdp

include("models/integral.jl")
export IntMDP, IntState, IntAction
export load_mdp, load_int_mdp, make_int_mdp, compress

include("valuefunction/valuefunction.jl")
export make_value

include("valuefunction/bellman.jl")
export qvalue, qvalues, qvalues!
export greedy, greedy!, bellman, bellmangreedy

include("algorithms/valueiteration.jl")
export value_iteration, value_iteration!

include("algorithms/mrp.jl")
export mrp!, mrp, mrp_sparse

include("algorithms/policyiteration.jl")
export policy_iteration, policy_iteration_sparse

include("simulation.jl")
export simulate, random_π
export Policy, PolicyStationary, PolicyMarkov
export FPolicyS, FPolicyM, TabPolicySD, TabPolicyMD
export cumulative
export Transition

# ----- Domains -------
module Domains
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
end
export Domains
# --------------------
end # module MDPs
