
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


"""
    qvalue(model, γ, s, a, v)

Compute the state-action-values for state `s`, action `a`, and
value function `v` for a discount factor `γ`.

This function is just a more efficient version of the standard definition.
"""
@inline function qvalue(model::IntMDP, objective::Objective,
                        s::Int, a::Int, v::AbstractVector{<:Real}) 
    x = model.states[s].actions[a]
    val = 0.0
    # much much faster than sum( ... for)
    for i ∈ eachindex(x.nextstate, x.probability, x.reward)
        @inbounds val += x.probability[i] *
            (x.reward[i] + discount(objective) * v[x.nextstate[i]])
    end
    val :: Float64
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
    greedy(model, obj, [s,] v)

Compute the greedy action for state `s` and value 
function `v` assuming an objective `obj`.

If `s` is not provided, then computes a value function for all states.
The model must support `states` function.
"""
greedy(model::MDP{S,A}, obj::Objective, s::S, v)  where {S,A}  =
    bellmangreedy(model, obj, s, v).action :: A

#greedy(model::TabMDP{S,A}, obj::Objective, v) where {S,A} =
#    greedy.((model,), (obj,), states(model), (v,))

"""
    greedy!(π, model, obj, v)

Update policy `π` with the greedy policy for value function `v` and MDP `model`
and an objective `obj`.
"""
function greedy!(π::Vector{Int}, model::TabMDP, obj::Objective,
        v::AbstractVector{<:Real})  

    length(π) == state_count(model) ||
        error("Policy π length must be the same as the state count")
    length(v) == state_count(model) ||
        error("Value function length must be the same as the state count")
            
    π .= greedy.((model,), (obj,), states(model), (v,))
end


""" 
    greedy(model, obj, v)

Compute the greedy action for all states and value function `v` assuming
an objective `obj`.
"""
function greedy(model::TabMDP, obj::Stationary, v::AbstractVector{<:Real}) 
    π = Vector{Int}(undef, state_count(model))
    greedy!(π,model,obj,v)
    π
end

greedy(model, γ::Real, v::AbstractVector{<:Real}) = greedy(model, InfiniteH(γ), v)


""" 
    bellman(model, γ, s, v)

Compute the Bellman operator for state `s`, and value function `v` assuming
an objective `obj`.

A real-valued objective `obj` is interpreted as a discount factor. 
"""
bellman(model::MDP{S,A}, obj::Objective, s::S, v) where {S,A} =
    bellmangreedy(model, obj, s, v).qvalue :: Float64
