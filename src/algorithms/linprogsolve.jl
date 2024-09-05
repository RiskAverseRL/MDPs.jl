using JuMP

# ----------------------------------------------------------------
# Linear Program Solver
# ----------------------------------------------------------------


"""
    lp_solve(model, γ, lpmf, [silent = true])

Implements the linear program primal problem for an MDP `model` with a discount factor `γ`.
It uses the JuMP model `lpm` as the linear program solver and returns the state values
found by `lpmf`. The `lpmf` is a factory that can be passed to `JuMP.Model`. 

The function needs to be provided with a solver. See the example below.

# Example

```jldoctest
    using MDPs, HiGHS
    model = Domains.Gambler.Ruin(0.5, 10)
    val = lp_solve(model, 0.9, HiGHS.Optimizer)
    maximum(val.policy)

# output

    6
```
"""

function lp_solve(model::TabMDP, obj::InfiniteH, lpmf; silent = true)
    γ = discount(obj)
    0 ≤ γ < 1 || error("γ must be between 0 and 1.")

    
    lpm = Model(lpmf)
    silent && set_silent(lpm)
    n = state_count(model)
    
    @variable(lpm, v[1:n])
    @objective(lpm, Min, sum(v[1:n]))

    u = Vector{Vector{ConstraintRef}}(undef, n)
    for s ∈ 1:n
        u[s] = [@constraint(lpm, v[s] ≥ sum(sp[2]*(sp[3]+γ*v[sp[1]])
                                        for sp in transition(model,s,a)))
            for a ∈ 1:action_count(model,s)]
    end
    
    optimize!(lpm)
    is_solved_and_feasible(lpm; dual = true) ||
        error("Failed to solve the MDP linear program")
    
    (value = value.(v),
     policy = map(x->argmax(dual.(x)), u))
end

lp_solve(model::TabMDP, γ::Number, lpm; args...) =
    lp_solve(model, InfiniteH(γ), lpm; args...)
    
