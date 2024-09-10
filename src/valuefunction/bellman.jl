
# ----------------------------------------------------------------
# Q-value computation operators
# ----------------------------------------------------------------

"""
    qvalues(model, objective, [t=0,] s, v)

Compute the state-action-value for state `s`, and value function `v` for 
`objective`. There is no set representation of the value function `v`.

The function is tractable only if there are a small number of actions
and transitions.

The function is tractable only if there are a small number of actions and transitions.
"""
function qvalues(model::MDP{S,A}, objective::Union{FiniteH, InfiniteH},
                 t::Integer, s::S, v) where {S,A}
    acts = actions(model, s)
    qvalues = Vector{Float64}(undef, length(acts))
    qvalues!(qvalues, model, objective, s, v)
    return qvalues
end

qvalues(model, objective, s, v) = qvalues(model, objective, 0, s, v)

"""
    qvalues!(qvalues, model, objective, [t=0,] s, v)

Compute the state-action-values for state `s`, and value function `v` for the
`objective`.

Saves the values to `qvalue` which should be at least as long as
the number of actions. Values of elements in `qvalues` that are beyond the action
count are set to `-Inf`.

See `qvalues` for more information.
"""
function qvalues!(qvalues::AbstractVector{<:Real}, model::MDP{S,A},
                  obj::Objective, t::Integer, s::S, v) where {S,A}

    acts = actions(model, s)
    for (ia,a) ∈ enumerate(acts)
        qvalues[ia] = qvalue(model, obj, t, s, a, v)
    end
end

qvalues!(qvalues, model, obj, s, v) = qvalues!(qvalues, model, obj, 0, s, v)

"""
    qvalue(model, objective, [t=0,] s, a, v)

Compute the state-action-values for state `s`, action `a`, and
value function `v` for an `objective`.

There is no set representation for the value function.
"""
function qvalue(model::MDP{S,A}, objective::Union{FiniteH, InfiniteH},
                        t::Integer, s::S, a::A, v) where {S,A} 
    val :: Float64 = 0.0
    # much much faster than sum( ... for)
    for (sn, p, r) ∈ transition(model, s, a)
        val += p * (r + discount(objective) * valuefunction(model, sn, v))
    end
    val 
end


# more efficient version for IntMDPs
function qvalue(model::IntMDP, objective::Union{FiniteH, InfiniteH},
                        t::Integer, s::Integer, a::Integer, v::AbstractVector{<:Real}) 
    x = model.states[s].actions[a]
    val = 0.0
    # much much faster than sum( ... for)
    for i ∈ eachindex(x.nextstate, x.probability, x.reward)
        @inbounds val += x.probability[i] *
            (x.reward[i] + discount(objective) * v[x.nextstate[i]])
    end
    val :: Float64
end

qvalue(model, objective, s, a, v) = qvalue(model, objective, 0, s, a, v)

# ----------------------------------------------------------------
# Generalized Bellman operators
# ----------------------------------------------------------------

""" 
    bellmangreedy(model, obj, [t=0,] s, v)

Compute the Bellman operator and greedy action for state `s`, and value 
function `v` assuming an objective `obj`. The optional time parameter `t` allows for
time-dependent updates.

The function uses `qvalue` to compute the Bellman operator and the greedy policy.
"""
function bellmangreedy(model::MDP{S,A}, obj::Objective, t::Integer, s::S, v) where {S,A}
    acts = actions(model, s)
    (qval, ia) = findmax(a->qvalue(model, obj, t, s, a, v), acts) 
    (qvalue = qval :: Float64,
        action = acts[ia] :: A)
end

# default fallback when t is 
bellmangreedy(model, obj, s, v) = bellmangreedy(model, obj, 0, s, v)

# ----------------------------------------------------------------
# Greedy policies and Bellman
# ----------------------------------------------------------------

""" 
    greedy(model, obj, [t=0,] s, v)

Compute the greedy action for state `s` and value 
function `v` assuming an objective `obj`.

If `s` is not provided, then computes a value function for all states.
The model must support `states` function.
"""
greedy(model::MDP{S,A}, obj::Objective, t::Integer, s::S, v)  where {S,A}  =
    bellmangreedy(model, obj, t, s, v).action :: A

greedy(model, obj, s, v) = greedy(model, obj, 0, s, v)


"""
    greedy!(π, model, obj, v)

Update policy `π` with the greedy policy for value function `v` and MDP `model`
and an objective `obj`.
"""
function greedy!(π::Vector{Int}, model::TabMDP, obj::Objective, t::Integer,
        v::AbstractVector{<:Real})  

    length(π) == state_count(model) ||
        error("Policy π length must be the same as the state count")
    length(v) == state_count(model) ||
        error("Value function length must be the same as the state count")
            
    π .= greedy.((model,), (obj,), (t,), states(model), (v,))
end

greedy!(π, model, obj, v) = greedy!(π, model, obj, 0, v)


""" 
    greedy(model, obj, v)

Compute the greedy action for all states and value function `v` assuming
an objective `obj` and time `t=0`.
"""
function greedy(model::TabMDP, obj::Stationary, v::AbstractVector{<:Real}) 
    π = Vector{Int}(undef, state_count(model))
    greedy!(π,model,obj,v)
    π
end


""" 
    bellman(model, obj, [t=0,] s, v)

Compute the Bellman operator for state `s`, and value function `v` assuming
an objective `obj`.
"""
bellman(model::MDP{S,A}, obj::Objective, t::Integer, s::S, v) where {S,A} =
    bellmangreedy(model, obj, t, s, v).qvalue :: Float64

bellman(model, obj, s, v) = bellman(model, obj, 0, s, v)
