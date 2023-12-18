
# ----------------------------------------------------------------
# Q-value computation operators
# ----------------------------------------------------------------

"""
    qvalues(model, objective, s, v)

Compute the state-action-value for state `s`, and value function `v` for 
`objective`. There is no set representation of the value function `v`.

The function is tractable only if there are a small number of actions
and transitions.

The function is tractable only if there are a small number of actions and transitions.
"""
function qvalues(model::MDP{S,A}, objective::Objective, s::S, v) where {S,A}
    acts = actions(model, s)
    qvalues = Vector{Float64}(undef, length(acts))
    qvalues!(qvalues, model, objective, s, v)
    return qvalues
end

"""
    qvalues!(qvalues, model, objective, s, v)

Compute the state-action-values for state `s`, and value function `v` for the
`objective`.

Saves the values to `qvalue` which should be at least as long as
the number of actions. Values of elements in `qvalues` that are beyond the action
count are set to `-Inf`.

See `qvalues` for more information.
"""
function qvalues!(qvalues::AbstractVector{<:Real}, model::MDP{S,A},
                  obj::Objective, s::S, v) where {S,A}

    if isterminal(model, s)
        qvalues .= -Inf
        qvalues[1] = 0 
    else
        acts = actions(model, s)
        for (ia,a) ∈ enumerate(acts)
            qvalues[ia] = qvalue(model, obj, s, a, v)
        end
    end
end

"""
    qvalue(model, objective, s, a, v)

Compute the state-action-values for state `s`, action `a`, and
value function `v` for an `objective`.

There is no set representation for the value function.
"""
@inline function qvalue(model::MDP{S,A}, objective::Objective, s::S, a::A, v) where {S,A} 
    val :: Float64 = 0.0
    # much much faster than sum( ... for)
    for (sn, p, r) ∈ transition(model, s, a)
        val += p * (r + discount(objective) * valuefunction(model, sn, v))
    end
    val 
end

# ----------------------------------------------------------------
# Generalized Bellman operators
# ----------------------------------------------------------------

""" 
    bellmangreedy(model, obj, s, v)

Compute the Bellman operator and greedy action for state `s`, and value 
function `v` assuming an objective `obj`.

The function uses `qvalue` to compute the Bellman operator and the greedy policy.
"""
function bellmangreedy(model::MDP{S,A}, obj::Objective, s::S, v) where {S,A}
    if isterminal(model, s)
        (qvalue = 0 :: Float64,
         action = emptyaction(model) :: A) 
    else
        acts = actions(model, s)
        (qval, ia) = findmax(a->qvalue(model, obj, s, a, v), acts) 
        (qvalue = qval :: Float64,
         action = acts[ia] :: A)
    end
end


# ----------------------------------------------------------------
# Greedy policies and Bellman
# ----------------------------------------------------------------

""" 
    greedy(model, obj, s, v)

Compute the greedy action for state `s` and value 
function `v` assuming an objective `obj`.

A real-valued objective `obj` is interpreted as a discount factor. 
"""
greedy(model::MDP{S,A}, obj, s::S, v) :: A where {S,A} =
    bellmangreedy(model, obj, s, v).action 

""" 
    bellman(model, γ, s, v)

Compute the Bellman operator for state `s`, and value 
function `v` assuming an objective `obj`.

A real-valued objective `obj` is interpreted as a discount factor. 
"""
bellman(model::MDP{S,A}, obj, s::S, v) :: Float64 where {S,A} =
    bellmangreedy(model, obj, s, v).qvalue 
