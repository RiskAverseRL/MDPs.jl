"""
    value_iteration(model, objective; [v_terminal, iterations = 1000, ϵ = 1e-3] ) 


Compute value function and policy for a tabular MDP `model` with an objective
`objective`. The time steps go from 1 to T+1, the last decision happens at time T.

The supported objectives are `FiniteH`, and `InfiniteH`. When provided with a
a real number `γ ∈ [0,1]` then the objective is treated as an infinite
horizon problem. 

Finite Horizon
--------------
Use finite-horizon value iteration for a tabular MDP `model` with 
a discount factor `γ` and horizon `T` (time steps `1` to `T+1`) the last decision
happens at time T. Returns a vector of value functions for each time step.

The argument `v_terminal` represents the terminal value function. It should be provided as
a function that maps the state id to its terminal value (at time T+1). If this value is provided,
then it is used in place of 0.

Infinite Horizon
----------------
For a Bellman error `ϵ`, the computed value function is quaranteed to be within
ϵ ⋅ γ / (1 - γ) of the optimal value function (all in terms of the L_∞ norm).

The value function is parallelized when `parallel` is true. This is also known
as a Jacobi type of value iteration (as opposed to Gauss-Seidel)

Note that for the purpose of the greedy policy, minimizing the span seminorm
is more efficient, but the goal of this function is also to compute the value 
function.

The time steps go from 1 to T+1.
"""
function value_iteration end


"""
    value_iteration!(v, π, model, objective; [v_terminal] )

Run value iteration using the provided `v` and `π` storage for the value function
and the policy. See `value_iteration` for more details.

Only support `FiniteH` objective. 
"""
function value_iteration! end

function value_iteration(model::TabMDP, objective::Markov; v_terminal = nothing)
    vp = make_value(model, objective)
    value_iteration!(vp.value, vp.policy, model, objective; v_terminal = v_terminal)
end

function value_iteration!(v::Vector{Vector{Float64}}, π::Vector{Vector{Int}},
                          model::TabMDP, objective::Markov; v_terminal = nothing)
    n = state_count(model)

    # final value function
    v[horizon(objective)+1] .= isnothing(v_terminal) ? 0 : map(v_terminal, 1:n)

    for t ∈ horizon(objective):-1:1
        # initialize vectors
        Threads.@threads for s ∈ 1:n           
            bg = bellmangreedy(model, objective, s, v[t+1])
            v[t][s] = bg.qvalue
            π[t][s] = bg.action
        end
    end
    return (policy = π, value = v)
end

function value_iteration(model::TabMDP, objective::Stationary;
                         iterations::Integer = 10000, ϵ::Number = 1e-3)
    nstates = state_count(model)
    vold = zeros(nstates)  # prior iteration
    vnew = zeros(nstates)  # current iteration
    residual = 0.         # the bellman residual
    itercount = iterations

    for it ∈ 1:iterations
        Threads.@threads for s ∈ 1:nstates
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
