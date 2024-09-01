
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
    (sn, p, r) ∈ transition(model, s, a)

Return an iterator with next states, probabilities, and rewards for
`model` taking an action `a` in state `s`.

Use `getnext` instead, which is more efficient and convenient to use. 
"""
function transition end

"""
    valuefunction(mdp, state, valuefunction)

Evaluates the value function for an `mdp` in a `state`
"""
function valuefunction end

# ----------------------------------------------------------------
# General inefficient implementation of getnext
# ----------------------------------------------------------------

"""
    getnext(model, s, a)

Compute next states using `transition` function.

Returns an object that can return a `NamedTuple` with `states`,
`probabilities`, and `transitions` as `AbstractArrays`. This
is a more-efficient version of transition (when supported).

The standard implementation is not memory efficient.
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
