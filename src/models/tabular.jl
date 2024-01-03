import Base
using LinearAlgebra
using SparseArrays

""" 
An abstract tabular Markov Decision Process which is specified by a transition function. 

Functions that should be defined for any subtype for value and policy iterations
to work are: `state_count`, `states`, `action_count`, `actions`, and `transition`.

Generally, states should be 1-based.

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
states(model::TabMDP) = 1:state_count(model)
# enumerated possible actions
actions(model::TabMDP, s::Int) = 1:action_count(model, s)


# ----------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------

using DataFrames: DataFrame, append!


"""
    save_mdp(T::DataFrame, model::TabMDP)

Convert an MDP `model` to a `DataFrame` representation with 0-based indices.

Important: The MDP representation uses 0-based indexes while the output
DataFrame is 0-based for backwards compatibility.

The columns are: `idstatefrom`, `idaction`, `idstateto`, `probability`,
and `reward`.
"""
function save_mdp(::Type{DataFrame}, model::TabMDP)
    arr_idstatefrom = Vector{Int}()
    arr_idstateto = Vector{Int}()
    arr_idaction = Vector{Int}()
    arr_prob = Vector{Float64}()
    arr_reward = Vector{Float64}()

    for s ∈ states(model)
        for a ∈ actions(model, s)
            for (sn,p,r) ∈ transition(model, s, a)
                push!(arr_idstatefrom, s - 1)
                push!(arr_idaction, a - 1)
                push!(arr_idstateto, sn - 1)
                push!(arr_prob, p)
                push!(arr_reward, r)
            end
        end
    end
    DataFrame(idstatefrom = arr_idstatefrom, idaction = arr_idaction,
              idstateto = arr_idstateto, probability = arr_prob,
              reward = arr_reward)
end
