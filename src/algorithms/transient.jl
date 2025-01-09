using JuMP

# ----------------------------------------------------------------
# Linear Program Solver
# ----------------------------------------------------------------


"""
    isterminal(model, state)

Checks that the `state` is terminal in `model`. A state is terminal if it

1) has a single action,
2) transitions to itself,
3) has a reward 0. 


# Example

```jldoctest
    using MDPs
    model = Domains.Gambler.RuinTransient(0.5, 4, true)
    isterminal.((model,), states(model))[1:2]

# output

2-element BitVector:
 1
 0
```
"""
function isterminal(model::MDP{S,A}, state::S) where {S,A}
    as = actions(model, state)
    length(as) == 1 || return false
    trs = transition(model, state, first(actions(model, state)))
    length(trs) == 1 || return false
    t = first(trs)
    (t[1] == state && t[2] ≈ 1.0 && t[3] ≈ 0.0) || return false
    return true
end


# a helper function used to check for transience
# reward: a function that specifies whether the reward
# from the MDP is used or a custom reward
# the function treats terminal states as having value 0
function _transient_lp(model::TabMDP, reward::Union{Float64, Nothing},
                       lpmf; silent) :: Union{Nothing,NamedTuple}

    @assert minimum(states(model)) == 1 # make sure that the index is 1-based

    lpm = Model(lpmf)
    silent && set_silent(lpm)

    rew(r) = isnothing(reward) ? r :: Float64 : reward :: Float64
    
    n = state_count(model)
    
    @variable(lpm, v[1:n])
    @objective(lpm, Min, sum(v))

    u = Vector{Vector{ConstraintRef}}(undef, n)
    for s ∈ 1:n
        @assert minimum(actions(model,s)) == 1 # make sure that the index is 1-based
        if isterminal(model, s) # set terminal state(s) to 0 value
            u[s] = [@constraint(lpm, v[s] == 0)]
        else
            u[s] = [@constraint(lpm, v[s] ≥ sum(p*(rew(r) + v[sn])
                                                for (sn,p,r) ∈ transition(model,s,a)))
                    for a in actions(model,s)]
        end
    end
    
    optimize!(lpm)

    if is_solved_and_feasible(lpm) 
        (value = value.(v), policy = map(x -> argmax(dual.(x)), u))
    else
        nothing
    end
end


"""
    lp_solve(model, lpmf, [silent = true])

Implements the linear program primal problem for an MDP `model` with a discount factor `γ`.
It uses the JuMP model `lpm` as the linear program solver and returns the state values
found found using the solver constructed by `JuMP.Model(lpmf)`.

## Examples


# Example

```jldoctest
    using MDPs, HiGHS
    model = Domains.Gambler.RuinTransient(0.5, 4, true)
    lp_solve(model, TotalReward(), HiGHS.Optimizer).policy

# output

5-element Vector{Int64}:
 1
 2
 3
 2
 1
```
"""
function lp_solve(model::TabMDP, obj::TotalReward, lpmf; silent = true)
    # nothing => run with the true rewards
    solution = _transient_lp(model, nothing, lpmf; silent = silent)
    if isnothing(solution)
        error("Failed to solve LP formulation. Is MDP transient?")
    else
        solution
    end
end


"""
    anytransient(model, lpmf, [silent = true])

Checks if the MDP `model` has some transient policy. A policy is transient if it
is guaranteed to terminate with positive probability after some finite number of steps.

Note that the function returns true even when there are some policies that are not transient.

The parameters match the use in `lp_solve`.
"""
function anytransient(model::TabMDP, lpmf; silent = true)
    solution = _transient_lp(model, -1., lpmf; silent = silent)
    !isnothing(solution)
end

"""
    anytransient(model, lpmf, [silent = true])

Checks if the MDP `model` has all transient policies. A policy is transient if it
is guaranteed to terminate with positive probability after some finite number of steps.

Note that the function returns true only if all policies are transient.

The parameters match the use in `lp_solve`.
"""
function alltransient(model::TabMDP, lpmf; silent = true)
    solution = _transient_lp(model, 1., lpmf; silent = silent)
    !isnothing(solution)
end
