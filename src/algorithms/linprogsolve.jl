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
    π::Vector{Vector{ConstraintRef}} = []
    for s in 1:n
        m = action_count(model,s)
        π_s::Vector{ConstraintRef} = []
        for a in 1:m
            push!(π_s, @constraint(lpm, v[s] ≥ sum(sp[2]*(sp[3]+γ*v[sp[1]]) for sp in transition(model,s,a))))
        end
        push!(π, π_s)
    end
    optimize!(lpm)
    (value = value.(v), policy = map(x->argmax(dual.(x)), π))
end
