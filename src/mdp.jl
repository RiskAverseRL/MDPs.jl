
# ----------------------------------------------------------------
# Type definitions
# ----------------------------------------------------------------

"""
A general MDP representation with time-independent 
transition probabilities and rewards. The model makes no assumption
that the states can be efficiently enumerated, but assumes
that there is small number of actions

S: state type
A: action type
"""
abstract type MDP{S,A} end

# ----------------------------------------------------------------
# Default definition of functions
# ----------------------------------------------------------------

"""
    isterminal(mdp, state)

Return true if the state is terminal
"""
function isterminal end

"""
    make_value(mdp, objective)

Construct a value function for an MDP
"""
function make_value end


# TODO: is this still being used? remove?
function policy end

# TODO: remove and deprecate, it is not well supported and complicates some algorithms
# this is the terminal reward 
function reward_T end

"""
    (sn, p, r) ∈ transition(model, s, a)

Return a list with next states, probabilities, and rewards.
Returns an iterator. 

Use `getnext` instead, which is more efficient and convenient to use. 
"""
function transition end

"""
    valuefunction(mdp, state, valuefunction)

Evaluates the value function for an MDP in a state
"""
function valuefunction end


"""
    getnext(model, s, a)

Returns an object that can return a `NamedTuple` with `states`,
`probabilities`, and `transitions` as `AbstractArrays`. This
is a more-efficient version of transition (when supported).
"""
function getnext end      

# ----------------------------------------------------------------
# Time-dependent definitions
# ----------------------------------------------------------------

reward_T(model::MDP{S,A}, s::S) where {S,A} = 0.::Float64
reward_T(model::MDP{S,A}, t::Integer, s::S) where {S,A} = reward_T(model, s)

# ----------------------------------------------------------------
# Q-value computation operators
# ----------------------------------------------------------------

"""
    qvalues(model, γ, s, v)

Compute the state-action-value for state `s`, and value function `v` for 
a discount factor `γ`. There is no set representation of the value function `v`.

The function is tractable only if there are a small number of actions
and transitions.
"""
function qvalues(model::MDP{S,A}, γ::Real, s::S, v) where {S,A}
    acts = actions(model, s)
    qvalues = Vector{Float64}(undef, length(acts))
    qvalues!(qvalues, model, γ, s, v)
    return qvalues
end

"""
    qvalues!(qvalue, model, γ, s, v)

Compute the state-action-values for state `s`, and value function `v` for 
a discount factor `γ`.

Saves the values to `qvalue` which should be sufficiently long. Elements of `qvalues`
that are beyond the action count are all set to -inf. There is no set representation
of the value function `v`.

A real-valued objective `obj` is interpreted as a discount factor. 

The function is tractable only if there are a small number of actions and transitions.
"""
function qvalues!(qvalues::AbstractVector{<:Real}, model::MDP{S,A},
                  obj, s::S, v) where {S,A}

    if isterminal(model, s)
        qvalues .= -Inf
        qvalues[1] = reward_T(model, s) 
    else
        acts = actions(model, s)
        for (ia,a) ∈ enumerate(acts)
            qvalues[ia] = qvalue(model, obj, s, a, v)
        end
    end
end


"""
    qvalue(model, γ, s, a, v)

Compute the state-action-values for state `s`, action `a`, and
value function `v` for a discount factor `γ`.

There is no set representation for the value function.
"""
@inline function qvalue(model::MDP{S,A}, γ::Real, s::S, a::A, v) where {S,A} 
    val :: Float64 = 0.0
    # much much faster than sum( ... for)
    for (sn, p, r) ∈ transition(model, s, a)
        val += p * (r + γ * valuefunction(model, sn, v))
    end
    val 
end


"""
    qvalue(model, obj, s, a, v)

Compute the state-action-values for state `s`, action `a`, and
value function `v` for a discounted infinite-horizon factor `γ`.

There is no set representation for the value function.
"""
@inline function qvalue(model, obj::Union{InfiniteH,FiniteH}, s, a, v)
    qvalue(model, obj.γ, s, a, v)
end

qvalue(model::MDP{S,A}, t::Integer, obj::FiniteH, s::S, a::A, v) where {S,A} =
    qvalue(model, obj, s, a, v)

# ----------------------------------------------------------------
# Greedy policies and Bellman
# ----------------------------------------------------------------

""" 
    greedy(model, obj, s, v)

Compute the greedy action for state `s` and value 
function `v` assuming an objective `obj`.

A real-valued objective `obj` is interpreted as a discount factor. 
"""
greedy(model::MDP{S,A}, obj, s::S, v) where {S,A} =
    # needs to construct a vector in order to return the max value
    bellmangreedy(model, obj, s, v).action :: A

"""
    greedy(model, obj, s, v)

Compute the greedy action for state `s` and value 
function `v` assuming an objective `obj`.

A real-valued objective `obj` is interpreted as a discount factor. 
"""
greedy(model::MDP{S,A}, t::Integer, obj, s::S, v) where {S,A} =
    bellmangreedy(model, t, obj, v).action :: A

""" 
    bellman(model, γ, s, v)

Compute the Bellman operator for state `s`, and value 
function `v` assuming an objective `obj`.

A real-valued objective `obj` is interpreted as a discount factor. 
"""
bellman(model::MDP{S,A}, obj, s::S, v) where {S,A} =
    bellmangreedy(model, obj, s, v).qvalue :: Float64

bellman(model::MDP{S,A}, t::Integer, obj, s::S, v) where {S,A} =
    bellmangreedy(model, t, obj, s, v).qvalue :: Float64

# ----------------------------------------------------------------
# Generalized Bellman operators
# ----------------------------------------------------------------

""" 
    bellmangreedy(model, obj, s, v)

Compute the Bellman operator and greedy action for state `s`, and value 
function `v` assuming an objective obj.

A real-valued objective `obj` is interpreted as a discount factor. 
"""
function bellmangreedy(model::MDP{S,A}, obj, s::S, v) where {S,A}
    if isterminal(model, s)
        (qvalue = reward_T(model, s) :: Float64,
         action = emptyaction(model) :: A) 
    else
        acts = actions(model, s)
        (qval, ia) = findmax(a->qvalue(model, obj, s, a, v), acts) 
        (qvalue = qval :: Float64,
         action = acts[ia] :: A)
    end
end

""" 
    bellmangreedy(model, t, obj, s, v)

Compute a time-dependent Bellman operator and greedy action for state `s`, and value 
function `v` assuming an objective obj.

A real-valued objective `obj` is interpreted as a discount factor. 
"""
function bellmangreedy(model::MDP{S,A}, t::Integer, obj, s::S, v) where {S,A}
    if isterminal(model, s)
        (qvalue = reward_T(model, s) :: Float64,
         action = emptyaction(model) :: A) 
    else
        acts = actions(model, s)
        (qval, ia) = findmax(a->qvalue(model, t, obj, s, a, v), acts) 
        (qvalue = qval :: Float64,
         action = acts[ia] :: A)
    end
end

# ----------------------------------------------------------------
# General inefficient implementation of getnext
# ----------------------------------------------------------------

"""
General inefficient representation.
"""
function getnext(model::MDP{S,A}, s::S, a::A) where {S,A}
    # TODO: is there any way to generate a warning
    # that this method is being used and it is inefficient?
    rewards = Vector{Float64}()      
    probabilities = Vector{Float64}()
    states = Vector{S}()
    for (sn, p, r) ∈ transition(model, s, a)
        append!(states, sn)
        append!(probabilities, p)
        append!(rewards, r)
    end

    (states = states, probabilities = probabilities, rewards = rewards)
end
