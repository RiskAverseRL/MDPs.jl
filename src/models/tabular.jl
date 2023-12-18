import Base
using LinearAlgebra
using SparseArrays

""" 
An abstract tabular Markov Decision Process, time independent.

Default interpretation
- State: Positive integer (>0) is non-terminal, zero or negative integer is terminal
- Action: Positive integer, anything else is invalid

Functions that should be defined for any subtype for value and policy iterations
to work are: `state_count`, `action_count`, `transition`

The methods `state_count` and `states` should only include non-terminal states
"""
abstract type TabMDP <: MDP{Int,Int} end

# ----------------------------------------------------------------
# General MDP interface functions
# ----------------------------------------------------------------

isterminal(::TabMDP, s::Int) = s ≤ 0
valuefunction(::TabMDP, s::Int, v) = v[s]

function state_count end
function action_count end

# enumerates possible states
function states end
# enumerated possible actions
function actions end


# ----------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------

using DataFrames: DataFrame, append!
#import Base: convert

"""
    transform(T::DataFrame, model)

Convert a tabular MDP to a data frame representation
"""
function transform(::Type{DataFrame}, model::TabMDP)
    result = DataFrame()
    for s ∈ states(model)
        for a ∈ actions(model, s)
            for (sn,p,r) ∈ transition(model, s, a)
                newrow = (idstatefrom = s, idaction = a, idstateto = sn,
                          probability = p, reward = r)
                push!(result, newrow)
            end
        end
    end
    result
end
