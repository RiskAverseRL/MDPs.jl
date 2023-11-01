
"""
Defines a policy, whether a stationary deterministic, or randomized, Markov, or
even history-dependent. The policy should support functions `make_internal`, `append_history`
that initialize and update the internal state. The function `take_action` then chooses an action
to take.

It is important that the tracker keeps their own internal states in order to be thread safe.
"""
abstract type Policy{S,A,I} end

"""
Information about a transition from `state` to `nstate` after than an `action`. `time` is the time at which `nstate` is observed.
"""
struct Transition{S,A}
    state :: S
    action :: A
    reward :: Float64
    nstate :: S
    time :: Int
end

"""
    make_internal(model, policy, state) -> internal

Initialize the internal state for a policy with the initial state. Returns the initial state.
"""
function make_internal end

"""
    append_history(policy, internal, transition) :: internal

Update the internal state for a policy by the transition information.
"""
function append_history end

"""
    take_action(policy, internal, state) -> action

Return which action to take with the `internal` state and the MDP state `state`. 
"""
function take_action end

# ------------------------------------------------
# General stationary
# ------------------------------------------------

abstract type PolicyStationary{S,A} <: Policy{S,A,Nothing} end

make_internal(model::MDP{S,A}, π::PolicyStationary{S,A}, state::S) where {S,A} =
    nothing

append_history(π::PolicyStationary{S,A}, _::Nothing, tr::Transition) where {S,A} =
    nothing

take_action(π::PolicyStationary{S,A}, ::Nothing, s::S) where {S,A} =
    take_action(π, s)

# ------------------------------------------------
# General Markov
# ------------------------------------------------

abstract type PolicyMarkov{S,A} <: Policy{S,A,Int} end

# This is the initial time step
make_internal(model::MDP{S,A}, π::PolicyMarkov{S,A}, state::S) where {S,A} = 1

function append_history(π::PolicyMarkov{S,A}, internal::Int, tr::Transition) where {S,A} 
    # make sure that the function advances by 1
    @assert (tr.time == internal + 1) (string(tr.time) * "≠" * string(internal + 1))
    return tr.time
end

# ------------------------------------------------
# Function based
# ------------------------------------------------

""" General stationary policy specified by a function s → a """
struct FPolicyS{S,A} <: PolicyStationary{S,A}
    π :: Function
end

take_action(π::FPolicyS{S,A}, s::S) where {S,A} = π.π(s)

""" General stationary policy specified by a function s,t → a """
struct FPolicyM{S,A} <: PolicyMarkov{S,A}
    π :: Function
end

take_action(π::FPolicyM{S,A}, t::Int, s::S) where {S,A} = π.π(s,t)

# ------------------------------------------------
# Tabular
# ------------------------------------------------

""" Generic policy for tabular MDPs """
const TabPolicyStationary = PolicyStationary{Int,Int}
const TabPolicyMarkov = PolicyMarkov{Int,Int}

""" Stationary deterministic policy for tabular MDPs """
struct TabPolicySD <: TabPolicyStationary
    π :: Vector{Int}
end

take_action(π::TabPolicySD, s::Int) = π[s]

"""
Markov deterministic policy for tabular MDPs. The policy `π` has an outer array over time
steps and an inner array over states.
"""
struct TabPolicyMD <: TabPolicyMarkov
    π :: Vector{Vector{Int}}
end

take_action(π::TabPolicyMD, t::Int, s::Int) = π.π[t][s]

""" 
    simulate(model, π, initial, horizon, episodes; [stationary = true])

Simulate a policy `π` in a `model` and generate states and actions for
the `horizon` decisions and `episodes` episodes. The initial
state is `initial`.

The policy `π` can be a function, or a array, or an array of arrays depending on
whether the policy is stationary, Markovian, deterministic, or randomized. When the policy
is provided as a function, then the parameter `stationary` is used.

There are horizon+1 states generated in every episode including the terminal
state at T+1.

The function requires that each state and action transition
to a reasonable small number of next states.

## See Also
`cumulative` to compute the cumulative rewards
"""
function simulate(model::MDP{S,A}, π::Policy{S,A}, initial::S,
                  horizon::Integer, episodes::Integer) where {S,A}

    states = fill(zero(S), horizon+1, episodes)
    actions = fill(zero(A), horizon, episodes)
    # there is no reward for the last transition
    rewards = fill(NaN, horizon, episodes)

    Threads.@threads for run ∈ 1:episodes
        states[1, run] = initial
        local internal = make_internal(model, π, initial)
        actions[1,run] = take_action(π, internal, initial)
        for t ∈ 2:(horizon+1)
            # a streaming approach to sampling the next state
            prob = rand()            
            tot_prob = 0.
            for (sn,pn,rn) ∈ transition(model, states[t-1,run], actions[t-1,run])
                isterminal(model, sn) && error("Terminal states unsupported.")    
                if prob ≤ (tot_prob += pn) # state sn was sampled
                    # update internal state using the current time step
                    let tr = Transition(states[t-1,run], actions[t-1,run], rn, sn, t)
                        internal = append_history(π, internal, tr)
                    end
                    states[t,run] = copy(sn)
                    rewards[t-1,run] = rn
                    if t ≤ horizon
                        actions[t,run] = take_action(π, internal, sn)
                    end
                    break
                end
            end
        end
    end
    (states = states, actions = actions, rewards = rewards)
end

function simulate(model::TabMDP, π::Vector{Int}, initial::Int,
                  horizon::Integer, episodes::Integer)
    length(π) == state_count(model) || error("Policy length must match state count.")
    simulate(model, TabPolicySD(π), initial, horizon, episodes)
end

# tabular states only
function simulate(model::TabMDP, π::Vector{Vector{Int}}, initial::Int,
                  horizon::Integer, episodes::Integer) 
    horizon ≤ length(π) || error("Horizon must be at most policy length.")
    simulate(model, TabPolicyMD(π), initial, horizon, episodes)
end

function simulate(model::MDP{S,A}, π::Function, initial::S,
                  horizon::Integer, episodes::Integer;
                  stationary = true) where {S,A}
    if stationary
        return simulate(model, FPolicyS{S,A}(π), initial, horizon, episodes)
    else
        return simulate(model, FPolicyM{S,A}(π), initial, horizon, episodes)
    end
end

"""
    random_π(model)

Construct a random policy for a tabular MDP
"""
function random_π(model::TabMDP)
    s -> rand(1:action_count(model, s))
end

"""
    cumulative(rewards, γ)

Computes the cumulative return from rewards returned by the simulation function.
"""
function cumulative(rewards::Matrix{<:Number}, γ::Number)
    horizon = size(rewards,1)
    rweights::Vector{Float64} = γ .^ (0:horizon-1)     # reward weights
    rweights' * rewards |> vec
end
