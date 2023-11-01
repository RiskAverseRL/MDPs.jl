## Methods for handling tabular MDPs with a specific integer implementation

import Base
using DataFrames: DataFrame
using DataFramesMeta

"""
An incorrect parameter value
"""
struct FormatError <: Exception
    msg :: AbstractString
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
struct GenericAction
    """ 1-based state index of the next state """
    nextstate :: Vector{Int}
    probability :: Vector{Float64}
    reward :: Vector{Float64}

    function GenericAction(nextstate, probability, reward) 
        # check that the provided values make sense
        (_same_lengths(nextstate, probability, reward)) ||
            error("Argument lengths must match.")
        isempty(nextstate) && error("States cannot be empty.")
        issorted(nextstate) || error("State ids must be sorted increasingly.")
        nextstate[1] ≥ 1 || error("State ids must be positive.") #sorted!
        all(probability .≥ 0.) || error("Probabilities must be non-negative.")
        ps = sum(probability)
        ps ≈ 1.0 || error("Probabilities must sum to 1 instead of $ps")
        
        new(nextstate, probability ./ ps, reward)
    end
end

# Prints the action
function Base.show(io::IO, t::MIME"text/plain", a::GenericAction)
    show(io, t, collect(zip(a.nextstate, a.probability, a.reward)))
end

# ----------------------------------------------------------------
# State-related data definition
# ----------------------------------------------------------------
    
""" Represents a discrete state """
struct GenericState
    actions :: Vector{GenericAction}
    function GenericState(actions :: Vector{GenericAction})
        isempty(actions) && throw(ArgumentError("Empty states not allowed"))
        new(actions)
    end
end

""" 
MDP with integral states and stationary transitions 
State and action indexes are all 1-based integers
"""
struct GenericMDP <: TabMDP
    """ States, actions, transitions """
    states :: Vector{GenericState}
end

# ----------------------------------------------------------------
# TabMDP Interface functions
# ----------------------------------------------------------------

state_count(model::GenericMDP) = length(model.states)
states(model::GenericMDP) = 1:state_count(model)
action_count(model::GenericMDP, s::Int) = length(model.states[s].actions)
actions(model::GenericMDP, s::Int) = 1:length(model.states[s].actions)
reward_T(::GenericMDP, t::Int, s::Int) = 0                   # there is no terminal reward
transition(model::GenericMDP, s::Int, a::Int) =
    (x = model.states[s].actions[a]; zip(x.nextstate, x.probability, x.reward))
function getnext(model::GenericMDP, s::Int, a::Int) 
    x = model.states[s].actions[a]
    (states = x.nextstate, probabilities = x.probability, rewards = x.reward)
end
# ----------------------------------------------------------------
# Loads CSV files
# ----------------------------------------------------------------

load_generic_mdp(input; idoutcome = nothing, docompress = false) =
    load_mdp(input; idoutcome = nothing, docompress = false) 
    
"""
    load_mdp(input, idoutcome)

Load the MDP from `input`. The function **assumes 0-based indexes**
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
    filepath = joinpath(dirname(pathof(MDPs)), "..",
                        "data", "riverswim.csv")
    model = load_mdp(File(filepath); idoutcome = 1)
```
Load the model from an Arrow file (a binary tabular file format)
```jldoctest
    using Arrow
    filepath = joinpath(dirname(pathof(MDPs)), "..",
                        "data", "riverswim.arrow")
    model = load_mdp(Arrow.Table(filepath); idoutcome = 1)
```
"""
function load_mdp(input; idoutcome = nothing, docompress = false)
    mdp = DataFrame(input)
    if (idoutcome != nothing)
        mdp = @subset(mdp, :idoutcome .== idoutcome - 1)
    end

    # offset relevant indices by one
    mdp = @transform(mdp,
              :idstatefrom = :idstatefrom .+ 1,
              :idstateto   = :idstateto.+ 1,
              :idaction    = :idaction .+ 1)
    if docompress
        mdp = @chain mdp begin
            @transform(:rnew = :probability .* :reward)
            groupby([:idstatefrom, :idaction, :idstateto])
            @combine(:probability = sum(:probability),
                 :reward = sum(:rnew)/sum(:probability))
        end
    end
    mdp = @orderby(mdp, :idstatefrom, :idaction, :idstateto)
    
    statecount = max(maximum(mdp.idstatefrom), maximum(mdp.idstateto))
    states = Vector{GenericState}(undef, statecount)
    state_init = BitVector(false for s in 1:statecount)

    for sd ∈ groupby(mdp, :idstatefrom)
        idstate = first(sd.idstatefrom)
        actions = Vector{GenericAction}(undef, maximum(sd.idaction))
       
        action_init = BitVector(false for a in 1:length(actions))
        for ad ∈ groupby(sd, :idaction)
            idaction = first(ad.idaction)
            actions[idaction] = GenericAction(ad.idstateto, ad.probability, ad.reward)
            action_init[idaction] = true
        end
        # report an error when there are missing indices
        all(action_init) ||
            throw(FormatError("Actions in state " * string(idstate - 1) *
                " that were uninitialized " * string(findall(.!action_init) .- 1 ) ))

        states[idstate] = GenericState(actions)
        state_init[idstate] = true
    end

    # create transitions to itself for each uninitialized state
    # to simulate a terminal state
    for s ∈ findall(.!state_init)
        states[s] = GenericState([GenericAction([s], [1.], [0.])])
    end
    GenericMDP(states)
end

"""
    make_generic_mdp(Ps, rs)

Build GenericMDP from a list of transition probabilities `Ps` and reward vectors
`rs` for each action in the MDP. Each row of the transition matrix represents
the probabilities of transitioning to next states.
"""
function make_generic_mdp(Ps::AbstractVector{Matrix{X}},
                          rs::AbstractVector{Vector{Y}}) where {X <: Number, Y <: Number}
    
    isempty(Ps) && error("Must have at least one action.")
    length(Ps) == length(rs) || error("Dimensions must match.")

    statecount = size(Ps[1])[1]
    actioncount = length(Ps)

    states = Vector{GenericState}(undef, statecount)
    for s ∈ 1:statecount
        actions = Vector{GenericAction}(undef, actioncount)
        for a ∈ 1:actioncount
            actions[a] = GenericAction(1:statecount, Ps[a][s,:],
                                       repeat([rs[a][s]], statecount) )
        end
        states[s] = GenericState(actions)
    end
    GenericMDP(states)
end


"""
    make_generic_mdp(mdp::TabMDP, docompress = false)

Transform any tabular MDP `mdp` to a generic one. This helps to accelerate
operations and value function computation. The actions are also turned into 1-based integer
values.

The option `docompress` combined transitions to the same state into a single transition.
This improves efficiency in risk-neutral settings, but may change the outcome
in risk-averse settings.

The function adds one more state at the end which represents a catch-all terminal state
"""
function make_generic_mdp(mdp::TabMDP; docompress = false)
    statecount = state_count(mdp)
    states = Vector{GenericState}(undef, statecount + 1) # + terminal
   
    # add a self-looping state to model a terminal state
    # needed to handle reward_T (termanl)
    states[statecount+1] = GenericState([GenericAction([statecount+1],[1.0],[0.0])])
                          
    Threads.@threads for s ∈ 1:statecount
        action_vals = 1:action_count(mdp, s)
        if isterminal(mdp, s)
            states[s]  = GenericState([GenericAction(
                [statecount+1], [1.0], [reward_T(mdp, s)])])
        else
            acts = Vector{GenericAction}(undef, length(action_vals))
            for (ia,a) ∈ enumerate(action_vals)
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
                a = GenericAction(ns, np, nr)
                acts[ia] = docompress ? compress(a) : a
            end
            states[s] = GenericState(acts)
        end
    end
    GenericMDP(states)
end

"""
    compress(nextstate, probability, reward) 

An generic action can represent the transitions to the same state with multiple entries.
The command will combine transitions to the same state into a single transition. Reward
is computed as a weigted average of the individual rewards, assuming expected reward objective.
"""
function compress(a::GenericAction) 
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
    GenericAction(ns, np, nr):: GenericAction
end


"""
    qvalue(model, γ, s, a, v)

Compute the state-action-values for state `s`, action `a`, and
value function `v` for a discount factor `γ`.

This function overrides the standard definition in the hope of
speeding up the computation.
"""
@inline function qvalue(model::GenericMDP, γ::Real,
                        s::Int, a::Int, v::AbstractVector{<:Real}) 
    x = model.states[s].actions[a]
    val = 0.0
    # much much faster than sum( ... for)
    for i ∈ eachindex(x.nextstate, x.probability, x.reward)
        @inbounds val += x.probability[i] * (x.reward[i] + γ * v[x.nextstate[i]])
    end
    val :: Float64
end
