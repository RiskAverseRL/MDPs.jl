using JuMP

# ----------------------------------------------------------------
# Linear Program Solver
# ----------------------------------------------------------------


"""
lp_solve(model, γ, lpm)

Implements the linear program primal problem for an MDP `model` with a discount factor `γ`.
It uses the JuMP model `lpm` as the linear program solver and returns the state values
found by `lpm`.
"""

function lp_solve(model::TabMDP, γ::Number, lpm)
    0 ≤ γ < 1 || error("γ must be between 0 and 1")
    set_silent(lpm)
    n = state_count(model)
    @variable(lpm, v[1:n])
    @objective(lpm,Min, sum(v[1:n]))
    for s in 1:n
        m = action_count(model,s)
        for a in 1:m
            snext = transition(model,s,a)
            @constraint(lpm, v[s] ≥ sum(sp[2]*(sp[3]+γ*v[sp[1]]) for sp in snext))
        end
    end
    optimize!(lpm)
    return value.(v)
end