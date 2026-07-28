## Methods for handling tabular MDPs with a specific integer implementation
import Base
using DataFrames: DataFrame
using DataFramesMeta

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
