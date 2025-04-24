## Methods for handling tabular MDPs with a specific integer implementation
import Base
using DataFrames: DataFrame
using DataFramesMeta

"""
An incorrect parameter value
"""
struct FormatError <: Exception
    msg::AbstractString
end

# helper method to check if the arrays have the same length
_same_lengths(x...) = (a = length.(x); all(a .== first(a)))

# ----------------------------------------------------------------
# Action-related data definition
# ----------------------------------------------------------------

"""
Represents transitions that follow an action.
The lengths `nextstate`, `probability`, and `reward` must be the same.

Nextstate may not be unique and each transition can have a different reward
associated with the transition. The transitions are not aggregated to allow for
comuting the risk of a transition. Aggregating the values by state would change
the risk value of the transition.
"""
struct IntAction
    """ 1-based state index of the next state """
    nextstate::Vector{Int}
    probability::Vector{Float64}
    reward::Vector{Float64}

    function IntAction(nextstate, probability, reward)
        # check that the provided values make sense
        (_same_lengths(nextstate, probability, reward)) ||
            error("Argument lengths must match.")
        isempty(nextstate) && error("States cannot be empty.")
        issorted(nextstate) || error("State ids must be sorted increasingly.")
        nextstate[1] ≥ 1 || error("State ids must be positive.") #sorted!
        all(probability .≥ 0.0) || error("Probabilities must be non-negative.")
        ps = sum(probability)
        ps ≈ 1.0 || error("Probabilities must sum to 1 instead of $ps")

        new(nextstate, probability ./ ps, reward)
    end
end

# Prints the action
function Base.show(io::IO, t::MIME"text/plain", a::IntAction)
    show(io, t, collect(zip(a.nextstate, a.probability, a.reward)))
end

# ----------------------------------------------------------------
# State-related data definition
# ----------------------------------------------------------------

""" Represents a discrete state """
struct IntState
    actions::Vector{IntAction}
    function IntState(actions::Vector{IntAction})
        isempty(actions) && throw(ArgumentError("Empty states not allowed"))
        new(actions)
    end
end

"""
MDP with integral states and stationary transitions
State and action indexes are all 1-based integers
"""
struct IntMDP <: TabMDP
    """ States, actions, transitions """
    states::Vector{IntState}
end

# ----------------------------------------------------------------
# TabMDP Interface functions
# ----------------------------------------------------------------

state_count(model::IntMDP) = length(model.states)
states(model::IntMDP) = 1:state_count(model)
action_count(model::IntMDP, s::Int) = length(model.states[s].actions)
actions(model::IntMDP, s::Int) = 1:length(model.states[s].actions)
transition(model::IntMDP, s::Int, a::Int) =
    (x = model.states[s].actions[a]; zip(x.nextstate, x.probability, x.reward))
function getnext(model::IntMDP, s::Int, a::Int)
    x = model.states[s].actions[a]
    (states=x.nextstate, probabilities=x.probability, rewards=x.reward)
end
# ----------------------------------------------------------------
# Loads CSV files
# ----------------------------------------------------------------

"""
    load_mdp(input, idoutcome)

    Load the MDP from `input`. The function **assumes 0-based indexes** (via `zerobased` flag),
of states and actions, which is transformed to 1-based index.

Input formats are anything that is supported by DataFrame. Some
options are `CSV.File(...)` or `Arrow.Table(...)`.

States that have no transition probabilities defined are assumed
to be terminal and are set to transition to themselves.

If `docombine` is true then the method combines transitions that have
the same statefrom, action, stateto. This makes risk-neutral value iteration
faster, but may change the value of a risk-averse solution.

The formulation allows for multiple transitions s,a → s'. When this
is the case, the transition probability is assumed to be their sum
and the reward is the weighted average of the rewards.

The method can also process CSV files for MDPO/MMDP, in which case
`idoutcome` specifies a 1-based outcome to load.

## Examples

Load the model from a CSV
```jldoctest
using CSV: File
using MDPs
filepath = joinpath(dirname(pathof(MDPs)), "..",
                    "data", "riverswim.csv")
model = load_mdp(File(filepath); idoutcome = 1)
state_count(model)

# output
20
```

Load the model from an Arrow file (a binary tabular file format)
```jldoctest
using MDPs, Arrow
filepath = joinpath(dirname(pathof(MDPs)), "..",
                    "data", "inventory.arr")
model = load_mdp(Arrow.Table(filepath))
state_count(model)

# output
21
```
"""
function load_mdp(input; idoutcome=nothing, docompress=false, zerobased=true)
    mdp = DataFrame(input)
    if (idoutcome != nothing)
        mdp = @subset(mdp, :idoutcome .== idoutcome - 1)
    end

    # offset relevant indices by one
    if zerobased
        mdp = @transform(mdp,
            :idstatefrom = :idstatefrom .+ 1,
            :idstateto = :idstateto .+ 1,
            :idaction = :idaction .+ 1)
    end
    if docompress
        mdp = @chain mdp begin
            @transform(:rnew = :probability .* :reward)
            groupby([:idstatefrom, :idaction, :idstateto])
            @combine(:probability = sum(:probability),
                :reward = sum(:rnew) / sum(:probability))
        end
    end
    mdp = @orderby(mdp, :idstatefrom, :idaction, :idstateto)

    statecount = max(maximum(mdp.idstatefrom), maximum(mdp.idstateto))
    states = Vector{IntState}(undef, statecount)
    state_init = BitVector(false for s in 1:statecount)

    for sd ∈ groupby(mdp, :idstatefrom)
        idstate = first(sd.idstatefrom)
        actions = Vector{IntAction}(undef, maximum(sd.idaction))

        action_init = BitVector(false for a in 1:length(actions))
        for ad ∈ groupby(sd, :idaction)
            idaction = first(ad.idaction)
            try
                actions[idaction] = IntAction(ad.idstateto, ad.probability, ad.reward)
            catch e
                error("Error in state $(idstate-1), action $(idaction-1): $e")
            end
            action_init[idaction] = true
        end
        # report an error when there are missing indices
        all(action_init) ||
            throw(FormatError("Actions in state " * string(idstate - 1) *
                              " that were uninitialized " * string(findall(.!action_init) .- 1)))

        states[idstate] = IntState(actions)
        state_init[idstate] = true
    end

    # create transitions to itself for each uninitialized state
    # to simulate a terminal state
    for s ∈ findall(.!state_init)
        states[s] = IntState([IntAction([s], [1.0], [0.0])])
    end
    IntMDP(states)
end


_make_reward(r::Vector{<:Number}, s, n) = repeat([r[s]], n)
_make_reward(R::Matrix{<:Number}, s, n) = R[s, :]


"""
    make_int_mdp(Ps, rs)

Build IntMDP from a list of transition probabilities `Ps` and reward vectors
`rs` for each action in the MDP. If `rs` are vectors, then they are assumed
to be state action rewards. If `rs` are matrixes then they are assumed to be
state-action-state rewwards. Each row of the transition matrix (and the reward
matrix) represents the probabilities of transitioning to next states.
"""
function make_int_mdp(Ps::AbstractVector{<:Matrix}, rs::AbstractVector{<:Array})

    isempty(Ps) && error("Must have at least one action.")
    length(Ps) == length(rs) || error("Ps and rs lengths must match.")

    statecount = size(Ps[1])[1]

    states = Vector{IntState}(undef, statecount)
    for s ∈ 1:statecount
        actions = [
            IntAction(1:statecount, Ps[a][s, :], _make_reward(rs[a], s, statecount))
            for a ∈ eachindex(Ps, rs)]
        states[s] = IntState(actions)
    end
    IntMDP(states)
end


"""
    make_int_mdp(mdp::TabMDP, docompress = false)

Transform any tabular MDP `mdp` to a numeric one. This helps to accelerate
operations and value function computation. The actions are also turned into 1-based integer
values.

The option `docompress` combined transitions to the same state into a single transition.
This improves efficiency in risk-neutral settings, but may change the outcome
in risk-averse settings.
"""
function make_int_mdp(mdp::TabMDP; docompress=false)
    statecount = state_count(mdp)
    states = Vector{IntState}(undef, statecount)

    Threads.@threads for s ∈ 1:statecount
        action_vals = 1:action_count(mdp, s)
        acts = Vector{IntAction}(undef, length(action_vals))
        for (ia, a) ∈ enumerate(action_vals)
            ns = Array{Int}(undef, 0)     # next state
            np = Array{Float64}(undef, 0) # next probalbility
            nr = Array{Float64}(undef, 0) # next reward

            for (nexts, nextp, nextr) ∈ transition(mdp, s, a)
                # check where to insert the next state transition
                i = searchsortedfirst(ns, nexts)
                insert!(ns, i, nexts)
                insert!(np, i, nextp)
                insert!(nr, i, nextr)
            end
            a = IntAction(ns, np, nr)
            acts[ia] = docompress ? compress(a) : a
        end
        states[s] = IntState(acts)
    end
    IntMDP(states)
end

"""
    compress(nextstate, probability, reward)

The command will combine mulitple transitions to the same state into a single transition. Reward
is computed as a weigted average of the individual rewards, assuming expected reward objective.
"""
function compress(a::IntAction)
    nextstate = a.nextstate
    probability = a.probability
    reward = a.reward

    @assert issorted(nextstate)

    # arrays for the new action
    ns = empty(nextstate)
    np = empty(probability)
    nr = empty(reward)

    for pos ∈ eachindex(nextstate, probability, reward)
        if !isempty(ns) && nextstate[pos] == last(ns)
            np[end] += probability[pos]
            nr[end] += probability[pos] * reward[pos] # normalized below
        else
            append!(ns, nextstate[pos])
            append!(np, probability[pos])
            append!(nr, probability[pos] * reward[pos]) # normalized below
        end
    end
    # need to normalize by the probabilities
    for i ∈ eachindex(nr, np)
        if np[i] > 1e-5 # skip ones too close to 0
            @inbounds nr[i] /= np[i]
        end
    end
    IntAction(ns, np, nr)::IntAction
end


