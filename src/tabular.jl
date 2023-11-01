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
function transition end

states(model::TabMDP) = 1:state_count(model)
actions(model::TabMDP, s::Int) = 1:action_count(model, s)

emptyaction(::TabMDP) = -1

# ----------------------------------------------------------------
# Greedy policies 
# ----------------------------------------------------------------

"""
    greedy!(π, model, objective, v)

Update policy `π` with the greedy policy for value function `v` and MDP `model`
and an objective `objective`. When real number `γ` is used as the objective, it
is interpreted as a discount factor.
"""
function greedy! end

""" 
    greedy(model, objective, v)

Compute the greedy action for all states and value function. See `greedy!` for more
details. 
"""
function greedy end


function greedy!(π::Vector{Int}, model::TabMDP, obj::Objective,
        v::AbstractVector{<:Real})  

    length(π) == state_count(model) ||
        error("Policy π length must be the same as the state count")
    length(v) == state_count(model) ||
        error("Value function length must be the same as the state count")
            
    π .= greedy.((model,), (obj,), 1:state_count(model), (v,))
end


function greedy(model::TabMDP, obj::Stationary, v::AbstractVector{<:Real}) 
    π = Vector{Int}(undef, state_count(model))
    greedy!(π,model,obj,v)
    π
end

@inline greedy!(π, model, γ::Real, v) = greedy!(π, model, InfiniteH(γ), v)
@inline greedy(model, γ::Real, v::AbstractVector{<:Real}) = greedy(model, InfiniteH(γ), v)


# ----------------------------------------------------------------
# Markov reward process and Markov chain
# ----------------------------------------------------------------

"""
    mrp!(P_π, r_π, model, π)

Save the transition matrix `P_π` and reward vector `r_π` for the 
MDP `model` and policy `π`. Also supports terminal states.

Does not support duplicate entries in transition probabilities.
"""
function mrp!(P_π::AbstractMatrix{<:Real}, r_π::AbstractVector{<:Real},
              model::TabMDP, π::AbstractVector{Int})
    S = state_count(model)
    fill!(P_π, 0.); fill!(r_π, 0.)
    for s ∈ 1:S
        if !isterminal(model, s)
            for (sn, p, r) ∈ transition(model, s, π[s])
                P_π[s,sn] ≈ 0. ||
                    error("duplicated transition entries (s1->s2, s1->s2) not allowed")
                P_π[s,sn] += p
                r_π[s] += p * r
            end
        else
            r_π[s] = reward_T(model, s)
        end
    end
end

"""
    mrp(model, π)

Compute the transition matrix `P_π` and reward vector `r_π` for the 
MDP `model` and policy `π`. See mrp! for more details. 
"""
function mrp(model::TabMDP, π::AbstractVector{Int})
    S = state_count(model)
    P_π = Matrix{Float64}(undef,S,S)
    r_π = Vector(undef, S)
    mrp!(P_π, r_π, model, π)
    (P_π, r_π)    
end

"""
    mrp(model, π)

Compute a sparse transition matrix `P_π` and reward vector `r_π` for the 
MDP `model` and policy `π`.

This function does not support duplicate entries in transition probabilities.
"""
function mrp_sparse(model::TabMDP, π::AbstractVector{Int})
    S = state_count(model)
    r_π = zeros(S)

    rows = Vector{Int}(undef, 0)
    columns = Vector{Int}(undef, 0)
    probabilities = Vector{Float64}(undef, 0)
    for s ∈ 1:S
        if !isterminal(model, s)
            for (sn, p, r) ∈ transition(model, s, π[s])
                append!(rows, s)
                append!(columns, sn)
                append!(probabilities, p)
                r_π[s] += p * r
            end
        else
            r_π[s] = reward_T(model, s)
        end
    end
    P_π = sparse(rows, columns, probabilities, S, S, (i,j)->
        error("Duplicate transition entries (s1->s2, s1->s2) are unsupported"))
    (P_π, r_π)
end


# ----------------------------------------------------------------
# Value iteration 
# ----------------------------------------------------------------

"""
    make_value(model, objective)

Creates an *undefined* policy and value function for the
`model` and `objective`.

See Also
--------
`value_iteration!`
"""
function make_value(model::TabMDP, objective::Markov)
    n::Integer = state_count(model)
    T::Integer = horizon(objective) 

    v = Vector{Vector{Float64}}(undef, horizon(objective)+1)
    π = Vector{Vector{Int}}(undef, horizon(objective))

    v[T+1] = Vector{Float64}(undef, n)

    for t ∈ T:-1:1
        # initialize vectors
        v[t] = Vector{Float64}(undef, n)
        π[t] = Vector{Int}(undef, n)
    end
    (policy = π, value = v)
end

"""
    value_iteration(model, γ, T) 

Use finite-horizon value iteration for a tabular MDP `model` with 
a discount factor `γ` and horizon `T` (time steps `1` to `T+1`). Returns 
a vector of value functions for each time step.

The time steps go from 1 to T+1.
"""
value_iteration(model::TabMDP, γ::Real, T::Int) = value_iteration(model, FiniteH(γ, T))


"""
    value_iteration(model, objective; [vterminal]) 

Compute value function and policy for a tabular MDP `model` with 
an objective `objective`. The time steps go from 1 to T+1, the last decision happens at time T.
Calls `value_iteration`.
"""
function value_iteration(model::TabMDP, objective::Markov; v_terminal = nothing)
    vp = make_value(model, objective)
    value_iteration!(vp.value, vp.policy, model, objective; v_terminal = v_terminal)
end

"""
    value_iteration!(v, π, model, objective; [vterminal]) 

Compute value function and policy for a tabular MDP `model` with 
an objective `objective`. The time steps go from 1 to T+1, the last decision happens at time T.
The values and policies are saved to `v` and `π` provided.

The argument `v_terminal` represents the terminal value function. It should be provided as
a function that maps the state id to its terminal value (at time T+1). If this value is provided,
then it is used in place of reward_T.
"""
function value_iteration!(v::Vector{Vector{Float64}}, π::Vector{Vector{Int}},
                          model::TabMDP, objective::Markov; v_terminal = nothing)
    n = state_count(model)

    # final value function
    v[horizon(objective)+1] .= isnothing(v_terminal) ?
        reward_T.((model,), horizon(objective)+1, 1:n) :
        map(v_terminal, 1:n)

    for t ∈ horizon(objective):-1:1
        # initialize vectors
        Threads.@threads for s ∈ 1:n           
            bg = bellmangreedy(model, t, objective, s, v[t+1])
            v[t][s] = bg.qvalue
            π[t][s] = bg.action
        end
    end
    return (policy = π, value = v)
end

"""
    value_iteration(model, γ; iterations, ϵ)

Run infinite horizon discounted value iteration for MDP `model` with a discount 
factor `γ`to compute the value function. The algorithm terminates 
after `iterations` steps or when the Bellman error drops to `ϵ`,
whichever comes first. 

For a Bellman error `ϵ`, the computed value function is quaranteed to be within
ϵ ⋅ γ / (1 - γ) of the optimal value function (all in terms of the L_∞ norm).

The value function is parallelized when `parallel` is true. This is also known
as a Jacobi type of value iteration (as opposed to Gauss-Seidel)

Note that for the purpose of the greedy policy, minimizing the span seminorm
is more efficient, but the goal of this function is also to compute the value 
function.
"""
value_iteration(model::TabMDP, γ::Real; iterations::Integer = 10000, ϵ::Number = 1e-3) = 
    value_iteration(model, InfiniteH(γ); iterations = iterations, ϵ = ϵ)


"""
    value_iteration(model, objective; iterations, ϵ)

Run abstract value iteration for a stationary `objective`. The algorithm terminates 
after `iterations` steps or when the Bellman error drops to `ϵ`. The objective needs
to support a bellman operator and a number of states. 
"""
function value_iteration(model::TabMDP, objective::Stationary;
                         iterations::Integer = 10000, ϵ::Number = 1e-3)
    states = state_count(model)
    vold = zeros(states)  # prior iteration
    vnew = zeros(states)  # current iteration
    residual = 0.         # the bellman residual
    itercount = iterations

    for it ∈ 1:iterations
        Threads.@threads for s ∈ 1:states
            @inbounds vnew[s] = bellman(model, objective, s, vold)
        end
        # compute the bellman residual
        vold .= vnew .- vold
        residual = maximum(abs.(extrema(vold)))
        vold .= vnew
        residual ≤ ϵ && (itercount = it; break)
    end
    return (value = vnew,
            iterations = itercount,
            residual = residual)
end

# ----------------------------------------------------------------
# Policy iteration
# ----------------------------------------------------------------


# adds an identity matrix in-place
@inline function _add_identity!(A)
    size(A,1) == size(A,2) || error("Matrix must be square")
    for i ∈ 1:size(A,1)
        @inbounds A[i,i] += one(A[1,1])
    end
end

"""
    policy_iteration(model, γ; [iterations=1000])

Implements policy iteration for MDP `model` with a discount factor `γ`. The algorithm
runs until the policy stops changing or the number of iterations is reached.

Does not support duplicate entries in transition probabilities.
"""
function policy_iteration(model::TabMDP, γ::Real; iterations::Int = 1000)
    S = state_count(model)
    # preallocate
    v_π = fill(0., S)
    IP_π = zeros(S, S)
    r_π = zeros(S)
    
    policy = fill(-1,S)  # 2 policies to check for change
    policyold = fill(-1,S)
    
    itercount = iterations
    for it ∈ 1:iterations
        policyold .= policy
        greedy!(policy, model, γ, v_π)
        mrp!(IP_π, r_π, model, policy);
        # Solve: v_π .= (I - γ * P_π) \ r_π
        lmul!(-γ, IP_π)
        _add_identity!(IP_π)
        ldiv!(v_π, lu!(IP_π), r_π)
        # check if there was a change
        if all(i->policy[i] == policyold[i], 1:S)
            itercount = it
            break
        end
    end
    (policy = policy, value = v_π, iterations = itercount)
end


"""
    policy_iteration_sparse(model, γ; iterations)

Implements policy iteration for MDP `model` with a discount factor `γ`. The algorithm
runs until the policy stops changing or the number of iterations is reached. The value
function is computed using sparse linear algebra.

Does not support duplicate entries in transition probabilities.
"""
function policy_iteration_sparse(model::TabMDP, γ::Real; iterations::Int = 1000)
    S = state_count(model)
    # preallocate
    v_π = fill(0., S)
    policy = fill(-1, (S,2))  # 2 policies to check for change
    for it ∈ 1:iterations
        (fl,fln) = (it % 2 + 1, (it + 1) % 2+1)
        greedy!(view(policy,:,fl), model, γ, v_π)
        P_π, r_π = mrp_sparse(model, view(policy,:,fl));
        v_π .= (I - γ * P_π) \ r_π  # TODO: eliminate this extra matrix copy 
        if all(policy[:,fl] .== policy[:,fln])
            return (policy = policy[:,fl],
                    value = v_π,
                    iterations = it)
        end
    end
    return (policy = policy[:, iterations % 2 + 1],
            value = v_π,
            iterations = iterations)
end

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
