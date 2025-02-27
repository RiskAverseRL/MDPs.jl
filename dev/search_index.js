var documenterSearchIndex = {"docs":
[{"location":"simulation/#Simulation","page":"Simulation","title":"Simulation","text":"","category":"section"},{"location":"simulation/","page":"Simulation","title":"Simulation","text":"This will be more extended documentation that will discuss how to simulate policies that are history dependent.","category":"page"},{"location":"#MDPs.jl:-Markov-Decision-Processes","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"","category":"section"},{"location":"#Models","page":"MDPs.jl: Markov Decision Processes","title":"Models","text":"","category":"section"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"This section describes the data structures that can be used to model various types on MDPs.","category":"page"},{"location":"#MDP","page":"MDPs.jl: Markov Decision Processes","title":"MDP","text":"","category":"section"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"This is a general MDP data structure that supports basic functions. See IntMDP and TabMDP below for more models that can be used more directly to model and solve.","category":"page"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs]\nPages = [\"mdp.jl\"]","category":"page"},{"location":"#MDPs.MDP","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.MDP","text":"A general MDP representation with time-independent  transition probabilities and rewards. The model makes no assumption that the states can be efficiently enumerated, but assumes that there is small number of actions\n\nS: state type A: action type\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.getnext-Union{Tuple{A}, Tuple{S}, Tuple{MDP{S, A}, S, A}} where {S, A}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.getnext","text":"getnext(model, s, a)\n\nCompute next states using transition function.\n\nReturns an object that can return a NamedTuple with states, probabilities, and transitions as AbstractArrays. This is a more-efficient version of transition (when supported).\n\nThe standard implementation is not memory efficient.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.transition","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.transition","text":"(sn, p, r) ∈ transition(model, s, a)\n\nReturn an iterator with next states, probabilities, and rewards for model taking an action a in state s.\n\nUse getnext instead, which is more efficient and convenient to use. \n\n\n\n\n\n","category":"function"},{"location":"#MDPs.valuefunction","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.valuefunction","text":"valuefunction(mdp, state, valuefunction)\n\nEvaluates the value function for an mdp in a state\n\n\n\n\n\n","category":"function"},{"location":"#Tabular-MDPs","page":"MDPs.jl: Markov Decision Processes","title":"Tabular MDPs","text":"","category":"section"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"This is an MDP instance that assumes that the states and actions are tabular. ","category":"page"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs]\nPages = [\"tabular.jl\"]","category":"page"},{"location":"#MDPs.TabMDP","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.TabMDP","text":"An abstract tabular Markov Decision Process which is specified by a transition function.  Functions that should be defined for any subtype for value and policy iterations to work are: state_count, states, action_count, actions, and transition.\n\nGenerally, states and actions are 1-based.\n\nThe methods state_count and states should only include non-terminal states\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.save_mdp-Tuple{Type{DataFrames.DataFrame}, TabMDP}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.save_mdp","text":"save_mdp(T::DataFrame, model::TabMDP)\n\nConvert an MDP model to a DataFrame representation with 0-based indices.\n\nImportant: The MDP representation uses 0-based indexes while the output DataFrame is 0-based for backwards compatibility.\n\nThe columns are: idstatefrom, idaction, idstateto, probability, and reward.\n\n\n\n\n\n","category":"method"},{"location":"#Integral-MDPs","page":"MDPs.jl: Markov Decision Processes","title":"Integral MDPs","text":"","category":"section"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"This is a specific MDP instance in which states and actions are specified by integers. ","category":"page"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs]\nPages = [\"integral.jl\"]","category":"page"},{"location":"#MDPs.FormatError","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.FormatError","text":"An incorrect parameter value\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.IntAction","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.IntAction","text":"Represents transitions that follow an action. The lengths nextstate, probability, and reward must be the same.\n\nNextstate may not be unique and each transition can have a different reward associated with the transition. The transitions are not aggregated to allow for comuting the risk of a transition. Aggregating the values by state would change the risk value of the transition. \n\n\n\n\n\n","category":"type"},{"location":"#MDPs.IntMDP","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.IntMDP","text":"MDP with integral states and stationary transitions  State and action indexes are all 1-based integers\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.IntState","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.IntState","text":"Represents a discrete state \n\n\n\n\n\n","category":"type"},{"location":"#MDPs.compress-Tuple{IntAction}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.compress","text":"compress(nextstate, probability, reward)\n\nThe command will combine mulitple transitions to the same state into a single transition. Reward is computed as a weigted average of the individual rewards, assuming expected reward objective.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.load_mdp-Tuple{Any}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.load_mdp","text":"load_mdp(input, idoutcome)\n\nLoad the MDP from input. The function assumes 0-based indexes of states and actions, which is transformed to 1-based index.\n\nInput formats are anything that is supported by DataFrame. Some options are CSV.File(...) or Arrow.Table(...).\n\nStates that have no transition probabilities defined are assumed to be terminal and are set to transition to themselves.\n\nIf docombine is true then the method combines transitions that have the same statefrom, action, stateto. This makes risk-neutral value iteration faster, but may change the value of a risk-averse solution.\n\nThe formulation allows for multiple transitions s,a → s'. When this is the case, the transition probability is assumed to be their sum and the reward is the weighted average of the rewards.\n\nThe method can also process CSV files for MDPO/MMDP, in which case idoutcome specifies a 1-based outcome to load.\n\nExamples\n\nLoad the model from a CSV\n\nusing CSV: File\nusing MDPs\nfilepath = joinpath(dirname(pathof(MDPs)), \"..\",\n                    \"data\", \"riverswim.csv\")\nmodel = load_mdp(File(filepath); idoutcome = 1)\nstate_count(model)\n\n# output\n20\n\nLoad the model from an Arrow file (a binary tabular file format)\n\nusing MDPs, Arrow\nfilepath = joinpath(dirname(pathof(MDPs)), \"..\",\n                    \"data\", \"inventory.arr\")\nmodel = load_mdp(Arrow.Table(filepath))\nstate_count(model)\n\n# output\n21\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.make_int_mdp-Tuple{AbstractVector{<:Matrix}, AbstractVector{<:Array}}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.make_int_mdp","text":"make_int_mdp(Ps, rs)\n\nBuild IntMDP from a list of transition probabilities Ps and reward vectors rs for each action in the MDP. If rs are vectors, then they are assumed to be state action rewards. If rs are matrixes then they are assumed to be state-action-state rewwards. Each row of the transition matrix (and the reward matrix) represents the probabilities of transitioning to next states.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.make_int_mdp-Tuple{TabMDP}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.make_int_mdp","text":"make_int_mdp(mdp::TabMDP, docompress = false)\n\nTransform any tabular MDP mdp to a numeric one. This helps to accelerate operations and value function computation. The actions are also turned into 1-based integer values.\n\nThe option docompress combined transitions to the same state into a single transition. This improves efficiency in risk-neutral settings, but may change the outcome in risk-averse settings.\n\n\n\n\n\n","category":"method"},{"location":"#Objectives","page":"MDPs.jl: Markov Decision Processes","title":"Objectives","text":"","category":"section"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs]\nPages = [\"objectives.jl\"]","category":"page"},{"location":"#MDPs.FiniteH","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.FiniteH","text":"FiniteH(γ, T)\n\nFinite-horizon discounted model. The discount factor γ can be in [0,1] and the horizon T must be a positive integer. The optimal policy is Markov but time dependent.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.InfiniteH","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.InfiniteH","text":"InfiniteH(γ)\n\nInifinite-horizon discounted objective. The discount factor γ can be in [0,1]. The optimal policy is stationary.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.Markov","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Markov","text":"Objective solved by a randomized Markov non-stationary policy. In other words, the solution is time-dependent.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.MarkovDet","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.MarkovDet","text":"Objective solved by a deterministic Markov non-stationary policy. In other words, the solution is time-dependent.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.Objective","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Objective","text":"Abstract objective for an MDP.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.Stationary","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Stationary","text":"Objective that is solved by a randomized stationary policy\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.StationaryDet","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.StationaryDet","text":"Objective that is solved by a randomized stationary policy\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.TotalReward","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.TotalReward","text":"TotalReward()\n\nTotal reward criterion. The objective is to maximize the sum of the undiscounted rewards. \n\nThis objective can generally only be applied to transient states, which have a terminal state; see isterminal for more details.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.horizon","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.horizon","text":"horizon(objective)\n\nReturn the horizon length for objective.\n\n\n\n\n\n","category":"function"},{"location":"#Algorithms","page":"MDPs.jl: Markov Decision Processes","title":"Algorithms","text":"","category":"section"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs]\nPages = [\"valueiteration.jl\"]","category":"page"},{"location":"#MDPs.value_iteration","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.value_iteration","text":"value_iteration(model, objective[, π]; [v_terminal, iterations = 1000, ϵ = 1e-3] )\n\nCompute value function and policy for a tabular MDP model with an objective objective. The time steps go from 1 to T+1, the last decision happens at time T.\n\nThe supported objectives are FiniteH, and InfiniteH. When provided with a a real number γ ∈ [0,1] then the objective is treated as an infinite horizon problem. \n\nFinite Horizon\n\nUse finite-horizon value iteration for a tabular MDP model with  a discount factor γ and horizon T (time steps 1 to T+1) the last decision happens at time T. Returns a vector of value functions for each time step.\n\nThe argument v_terminal represents the terminal value function. It should be provided as a function that maps the state id to its terminal value (at time T+1). If this value is provided, then it is used in place of 0.\n\nIf a policy π is provided, then the algorithm evaluates it. \n\nInfinite Horizon\n\nFor a Bellman error ϵ, the computed value function is quaranteed to be within ϵ ⋅ γ / (1 - γ) of the optimal value function (all in terms of the L_∞ norm).\n\nThe value function is parallelized when parallel is true. This is also known as a Jacobi type of value iteration (as opposed to Gauss-Seidel)\n\nNote that for the purpose of the greedy policy, minimizing the span seminorm is more efficient, but the goal of this function is also to compute the value  function.\n\nThe time steps go from 1 to T+1.\n\nSee also\n\nvalue_iteration!\n\n\n\n\n\n","category":"function"},{"location":"#MDPs.value_iteration!","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.value_iteration!","text":"value_iteration!(v, π, model, objective; [v_terminal] )\n\nRun value iteration using the provided v and π storage for the value function and the policy. See value_iteration for more details.\n\nOnly support FiniteH objective. \n\n\n\n\n\n","category":"function"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs]\nPages = [\"mrp.jl\"]","category":"page"},{"location":"#MDPs.mrp!-Tuple{AbstractMatrix{<:Real}, AbstractVector{<:Real}, TabMDP, AbstractVector{<:Integer}}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.mrp!","text":"mrp!(P_π, r_π, model, π)\n\nSave the transition matrix P_π and reward vector r_π for the  MDP model and policy π. Also supports terminal states.\n\nDoes not support duplicate entries in transition probabilities.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.mrp-Tuple{TabMDP, AbstractVector{<:Integer}}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.mrp","text":"mrp(model, π)\n\nCompute the transition matrix P_π and reward vector r_π for the  MDP model and policy π. See mrp! for more details. \n\n\n\n\n\n","category":"method"},{"location":"#MDPs.mrp_sparse-Tuple{TabMDP, AbstractVector{Int64}}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.mrp_sparse","text":"mrp(model, π)\n\nCompute a sparse transition matrix P_π and reward vector r_π for the  MDP model and policy π.\n\nThis function does not support duplicate entries in transition probabilities.\n\n\n\n\n\n","category":"method"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs]\nPages = [\"policyiteration.jl\"]","category":"page"},{"location":"#MDPs.policy_iteration-Tuple{TabMDP, Real}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.policy_iteration","text":"policy_iteration(model, γ; [iterations=1000])\n\nImplements policy iteration for MDP model with a discount factor γ. The algorithm runs until the policy stops changing or the number of iterations is reached.\n\nDoes not support duplicate entries in transition probabilities.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.policy_iteration_sparse-Tuple{TabMDP, Real}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.policy_iteration_sparse","text":"policy_iteration_sparse(model, γ; iterations)\n\nImplements policy iteration for MDP model with a discount factor γ. The algorithm runs until the policy stops changing or the number of iterations is reached. The value function is computed using sparse linear algebra.\n\nDoes not support duplicate entries in transition probabilities.\n\n\n\n\n\n","category":"method"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs]\nPages = [\"transient.jl\"]","category":"page"},{"location":"#MDPs.alltransient-Tuple{TabMDP, Any}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.alltransient","text":"anytransient(model, lpmf, [silent = true])\n\nChecks if the MDP model has all transient policies. A policy is transient if it is guaranteed to terminate with positive probability after some finite number of steps.\n\nNote that the function returns true only if all policies are transient.\n\nThe parameters match the use in lp_solve.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.anytransient-Tuple{TabMDP, Any}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.anytransient","text":"anytransient(model, lpmf, [silent = true])\n\nChecks if the MDP model has some transient policy. A policy is transient if it is guaranteed to terminate with positive probability after some finite number of steps.\n\nNote that the function returns true even when there are some policies that are not transient.\n\nThe parameters match the use in lp_solve.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.isterminal-Union{Tuple{A}, Tuple{S}, Tuple{MDP{S, A}, S}} where {S, A}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.isterminal","text":"isterminal(model, state)\n\nChecks that the state is terminal in model. A state is terminal if it\n\nhas a single action,\ntransitions to itself,\nhas a reward 0. \n\nExample\n\n    using MDPs\n    model = Domains.Gambler.RuinTransient(0.5, 4, true)\n    isterminal.((model,), states(model))[1:2]\n\n# output\n\n2-element BitVector:\n 1\n 0\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.lp_solve-Tuple{TabMDP, TotalReward, Any}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.lp_solve","text":"lp_solve(model, lpmf, [silent = true])\n\nImplements the linear program primal problem for an MDP model with a discount factor γ. It uses the JuMP model lpm as the linear program solver and returns the state values found found using the solver constructed by JuMP.Model(lpmf).\n\nExamples\n\nExample\n\n    using MDPs, HiGHS\n    model = Domains.Gambler.RuinTransient(0.5, 4, true)\n    lp_solve(model, TotalReward(), HiGHS.Optimizer).policy\n\n# output\n\n5-element Vector{Int64}:\n 1\n 2\n 3\n 2\n 1\n\n\n\n\n\n","category":"method"},{"location":"#Value-Function-Manipulation","page":"MDPs.jl: Markov Decision Processes","title":"Value Function Manipulation","text":"","category":"section"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs]\nPages = [\"valuefunction.jl\"]","category":"page"},{"location":"#MDPs.make_value-Tuple{TabMDP, Markov}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.make_value","text":"make_value(model, objective)\n\nCreates an undefined policy and value function for the model and objective.\n\nSee Also\n\nvalue_iteration!\n\n\n\n\n\n","category":"method"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs]\nPages = [\"bellman.jl\"]","category":"page"},{"location":"#MDPs.bellman-Union{Tuple{A}, Tuple{S}, Tuple{MDP{S, A}, MDPs.Objective, Integer, S, Any}} where {S, A}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.bellman","text":"bellman(model, obj, [t=0,] s, v)\n\nCompute the Bellman operator for state s, and value function v assuming an objective obj.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.bellmangreedy-Union{Tuple{A}, Tuple{S}, Tuple{MDP{S, A}, MDPs.Objective, Integer, S, Any}} where {S, A}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.bellmangreedy","text":"bellmangreedy(model, obj, [t=0,] s, v)\n\nCompute the Bellman operator and greedy action for state s, and value  function v assuming an objective obj. The optional time parameter t allows for time-dependent updates.\n\nThe function uses qvalue to compute the Bellman operator and the greedy policy.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.greedy!-Tuple{Vector{Int64}, TabMDP, MDPs.Objective, Integer, AbstractVector{<:Real}}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.greedy!","text":"greedy!(π, model, obj, v)\n\nUpdate policy π with the greedy policy for value function v and MDP model and an objective obj.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.greedy-Tuple{TabMDP, Stationary, AbstractVector{<:Real}}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.greedy","text":"greedy(model, obj, v)\n\nCompute the greedy action for all states and value function v assuming an objective obj and time t=0.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.greedy-Union{Tuple{A}, Tuple{S}, Tuple{MDP{S, A}, MDPs.Objective, Integer, S, Any}} where {S, A}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.greedy","text":"greedy(model, obj, [t=0,] s, v)\n\nCompute the greedy action for state s and value  function v assuming an objective obj.\n\nIf s is not provided, then computes a value function for all states. The model must support states function.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.qvalue-Union{Tuple{A}, Tuple{S}, Tuple{MDP{S, A}, Union{FiniteH, InfiniteH}, Integer, S, A, Any}} where {S, A}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.qvalue","text":"qvalue(model, objective, [t=0,] s, a, v)\n\nCompute the state-action-values for state s, action a, and value function v for an objective.\n\nThere is no set representation for the value function.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.qvalues!-Union{Tuple{A}, Tuple{S}, Tuple{AbstractVector{<:Real}, MDP{S, A}, MDPs.Objective, Integer, S, Any}} where {S, A}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.qvalues!","text":"qvalues!(qvalues, model, objective, [t=0,] s, v)\n\nCompute the state-action-values for state s, and value function v for the objective.\n\nSaves the values to qvalue which should be at least as long as the number of actions. Values of elements in qvalues that are beyond the action count are set to -Inf.\n\nSee qvalues for more information.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.qvalues-Union{Tuple{A}, Tuple{S}, Tuple{MDP{S, A}, Union{FiniteH, InfiniteH}, Integer, S, Any}} where {S, A}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.qvalues","text":"qvalues(model, objective, [t=0,] s, v)\n\nCompute the state-action-value for state s, and value function v for  objective. There is no set representation of the value function v.\n\nThe function is tractable only if there are a small number of actions and transitions.\n\nThe function is tractable only if there are a small number of actions and transitions.\n\n\n\n\n\n","category":"method"},{"location":"#Simulation","page":"MDPs.jl: Markov Decision Processes","title":"Simulation","text":"","category":"section"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs]\nPages = [\"simulation.jl\"]","category":"page"},{"location":"#MDPs.FPolicyM","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.FPolicyM","text":"General stationary policy specified by a function s,t → a \n\n\n\n\n\n","category":"type"},{"location":"#MDPs.FPolicyS","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.FPolicyS","text":"General stationary policy specified by a function s → a \n\n\n\n\n\n","category":"type"},{"location":"#MDPs.Policy","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Policy","text":"Defines a policy, whether a stationary deterministic, or randomized, Markov, or even history-dependent. The policy should support functions make_internal, append_history that initialize and update the internal state. The function take_action then chooses an action to take.\n\nIt is important that the tracker keeps their own internal states in order to be thread safe.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.TabPolicyMD","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.TabPolicyMD","text":"Markov deterministic policy for tabular MDPs. The policy π has an outer array over time steps and an inner array over states.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.TabPolicySD","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.TabPolicySD","text":"Stationary deterministic policy for tabular MDPs \n\n\n\n\n\n","category":"type"},{"location":"#MDPs.TabPolicyStationary","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.TabPolicyStationary","text":"Generic policy for tabular MDPs \n\n\n\n\n\n","category":"type"},{"location":"#MDPs.Transition","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Transition","text":"Information about a transition from state to nstate after than an action. time is the time at which nstate is observed.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.append_history","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.append_history","text":"append_history(policy, internal, transition) :: internal\n\nUpdate the internal state for a policy by the transition information.\n\n\n\n\n\n","category":"function"},{"location":"#MDPs.cumulative-Tuple{Matrix{<:Number}, Number}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.cumulative","text":"cumulative(rewards, γ)\n\nComputes the cumulative return from rewards returned by the simulation function.\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.make_internal","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.make_internal","text":"make_internal(model, policy, state) -> internal\n\nInitialize the internal state for a policy with the initial state. Returns the initial state.\n\n\n\n\n\n","category":"function"},{"location":"#MDPs.random_π-Tuple{TabMDP}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.random_π","text":"random_π(model)\n\nConstruct a random policy for a tabular MDP\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.simulate-Union{Tuple{A}, Tuple{S}, Tuple{MDP{S, A}, Policy{S, A}, Any, Integer, Integer}} where {S, A}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.simulate","text":"simulate(model, π, initial, horizon, episodes; [stationary = true])\n\nSimulate a policy π in a model and generate states and actions for the horizon decisions and episodes episodes. The initial state is initial.\n\nThe policy π can be a function, or a array, or an array of arrays depending on whether the policy is stationary, Markovian, deterministic, or randomized. When the policy is provided as a function, then the parameter stationary is used.\n\nThere are horizon+1 states generated in every episode including the terminal state at T+1.\n\nThe initial state initial should either be of a type S or can also be a vector that represents the distribution over the states\n\nThe function requires that each state and action transition to a reasonable small number of next states.\n\nSee Also\n\ncumulative to compute the cumulative rewards\n\n\n\n\n\n","category":"method"},{"location":"#MDPs.take_action","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.take_action","text":"take_action(policy, internal, state) -> action\n\nReturn which action to take with the internal state and the MDP state state. \n\n\n\n\n\n","category":"function"},{"location":"#Domains","page":"MDPs.jl: Markov Decision Processes","title":"Domains","text":"","category":"section"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs.Domains]","category":"page"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs.Domains.Gambler]","category":"page"},{"location":"#MDPs.Domains.Gambler.Ruin","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Domains.Gambler.Ruin","text":"Ruin(win, max_capital)\n\nGambler's ruin; the discounted version. Can decide how much to bet at any point in time. With some probability win, the bet is doubled, and with 1-win it is lost. The reward is 1 if it achieves some terminal capital and 0 otherwise. State max_capital+1 is an absorbing win state in which 1 is received forever.\n\nCapital = state - 1\nBet     = action - 1 \n\nAvailable actions are 1, ..., state.\n\nSpecial states: state=1 is broke and state=max_capital+1 is a terminal winning state.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.Domains.Gambler.RuinTransient","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Domains.Gambler.RuinTransient","text":"RuinTransient(win, max_capital, noop[, win_reward = 1.0, lose_reward = 0.0])\n\nGambler's ruin; the transient version. Can decide how much to bet at any point in time. With some probability win, the bet is doubled, and with 1-win it is lost. The reward is 1 if it achieves some terminal capital and 0 otherwise. State max_capital+1 is an absorbing win state in which 1 is received forever.\n\nCapital = state - 1\n\nIf noop = true then the available actions are 1, ..., capital+1 and bet = action - 1. This allows a bet of 0 which is not a transient policy. \n\nIf noop = false then the available actions are 1, ..., capital and bet = action. The MDP is not transient if noop = true, but has some transient policies. When noop = false, the MDP is transient.\n\nSpecial states: state=1 is broke and state=max_capital+1 is maximal capital. Both of the states are absorbing/terminal.\n\nBy default, the reward is 0 when the gambler goes broke and +1 when it achieves the target capital. The difference from Ruin is that no reward received in the terminal state. The rewards for overall win and loss can be adjusted by providing win_reward and lose_reward optional parameters.\n\n\n\n\n\n","category":"type"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs.Domains.Inventory]","category":"page"},{"location":"#MDPs.Domains.Inventory.Demand","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Domains.Inventory.Demand","text":"Models values of demand in values and probabilities in probabilities.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.Domains.Inventory.Model","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Domains.Inventory.Model","text":"An inventory MDP problem simulator\n\nThe states and actions are 1-based integers.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.Domains.Inventory.Parameters","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Domains.Inventory.Parameters","text":"Parameters that define an inventory problem\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.transition-Tuple{MDPs.Domains.Inventory.Parameters, Int64, Int64, Int64}","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.transition","text":"transition(params, stock, order, demand)\n\nUpdate the inventory value and compute the profit.\n\nStarting with a stock number of items, then order of items arrive, after demand of items are sold. Sale price is collected even if it is backlogged (not beyond backlog level). Negative stock means backlog.\n\nStocking costs are asessed after all the orders are fulfilled. \n\nCauses an error when the order is too large, but no error when the demand cannot be satisfied or backlogged.\n\n\n\n\n\n","category":"method"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs.Domains.Machine]","category":"page"},{"location":"#MDPs.Domains.Machine.Replacement","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Domains.Machine.Replacement","text":"Standard machine replacement simulator. See Figure 3 in Delage 2009 for details.\n\nStates are: 1: repair 1 2: repair 2 3 - 10: utility state\n\nActions: 1: Do nothing 2: Repair\n\n\n\n\n\n","category":"type"},{"location":"","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.jl: Markov Decision Processes","text":"Modules = [MDPs.Domains.GridWorld]","category":"page"},{"location":"#MDPs.Domains.GridWorld.Action","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Domains.GridWorld.Action","text":"Models values of demand in values and probabilities in probabilities.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.Domains.GridWorld.Model","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Domains.GridWorld.Model","text":"A GridWorld MDP problem simulator\n\nThe states and actions are 1-based integers.\n\n\n\n\n\n","category":"type"},{"location":"#MDPs.Domains.GridWorld.Parameters","page":"MDPs.jl: Markov Decision Processes","title":"MDPs.Domains.GridWorld.Parameters","text":"Parameters(reward_s, max_side_length, wind)\n\nParameters that define a GridWorld problem\n\nrewards_s: A vector of rewards for each state\nmax_side_length: An integer that represents the maximum side length of the grid\nwind: A float that represents the wind ∈ [0, 1]\n'revolve': Whether or not the agennt can wrap around the grid by moving off the edge and appearing on the other side default True\n'transient': Whether or not there is an absorbing state default False\n\n\n\n\n\n","category":"type"},{"location":"recipes/#Recipes","page":"Recipes","title":"Recipes","text":"","category":"section"},{"location":"recipes/#Converting-a-file-format-of-an-MDP","page":"Recipes","title":"Converting a file format of an MDP","text":"","category":"section"},{"location":"recipes/","page":"Recipes","title":"Recipes","text":"Converting from a CSV to an Arrow file","category":"page"},{"location":"recipes/","page":"Recipes","title":"Recipes","text":"using MDPs\nusing DataFrames\nusing Arrow\nusing CSV\n\nfilein  = joinpath(dirname(pathof(MDPs)), \"..\", \"data\", \"riverswim.csv\")\nfileout = tempname() \nmodel = load_mdp(CSV.File(filein); idoutcome = 1)\noutput = save_mdp(DataFrame, model)\n1\n\n# output\n\n1","category":"page"},{"location":"recipes/","page":"Recipes","title":"Recipes","text":"Converting from an Arrow to a CSV file","category":"page"},{"location":"recipes/","page":"Recipes","title":"Recipes","text":"using MDPs\nusing DataFrames\nusing Arrow\nusing CSV\n\nfilein  = joinpath(dirname(pathof(MDPs)), \"..\", \"data\", \"inventory.arr\")\nfileout = tempname()\nmodel = load_mdp(Arrow.Table(filein))\noutput = save_mdp(DataFrame, model)\nCSV.write(fileout, output)\n1\n\n# output\n\n1","category":"page"},{"location":"recipes/#Making-a-small-MDP","page":"Recipes","title":"Making a small MDP","text":"","category":"section"},{"location":"recipes/","page":"Recipes","title":"Recipes","text":"using MDPs\n\nε = 0.01\nP1 = [1 0 0; 0   1 0; 0 0 1]\nP2 = [0 1 0; 1-ε 0 ε; 0 0 1]\nPs = [P1, P2]\nR = [10 -4 0; -1 -3 0; 0 0 0] # use the same reward for both actions\nRs = [R, R]\n\nM = make_int_mdp(Ps, Rs)\nstate_count(M)\n\n# output\n\n3","category":"page"},{"location":"recipes/#Saving-an-MDP-to-a-file","page":"Recipes","title":"Saving an MDP to a file","text":"","category":"section"},{"location":"recipes/","page":"Recipes","title":"Recipes","text":"using MDPs\nusing DataFrames\nusing MDPs.Domains\nusing CSV\n\nmodel = Gambler.Ruin(0.7, 10)\ndomainoutput = MDPs.save_mdp(DataFrame, model)\nCSV.write(\"output_gambler.csv\", domainoutput)\n\n1\n\n# output\n\n1","category":"page"}]
}
