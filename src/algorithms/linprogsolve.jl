using JuMP

# ----------------------------------------------------------------
# Linear Program Solver
# ----------------------------------------------------------------


"""
    lp_solve(model, γ, lpm, [silent = true])

Implements the linear program primal problem for an MDP `model` with a discount factor `γ`.
It uses the JuMP model `lpm` as the linear program solver and returns the state values
found by `lpm`. 
"""

function lp_solve(model::TabMDP, obj::InfiniteH, lpm; silent = true)
    γ = discount(obj)
    0 ≤ γ < 1 || error("γ must be between 0 and 1")

    silent && set_silent(lpm)
    n = state_count(model)
    
    @variable(lpm, v[1:n])
    @objective(lpm, Min, sum(v[1:n]))

    π::Vector{Vector{ConstraintRef}} = [] # constraints for recovering the policy
    for s ∈ 1:n
        π_s = [@constraint(lpm, v[s] ≥ sum(sp[2]*(sp[3]+γ*v[sp[1]])
                                        for sp in transition(model,s,a)))
            for a ∈ 1:action_count(model,s)]
        push!(π, π_s)
    end
    
    optimize!(lpm)

    if !is_solved_and_feasible(lpm; dual = true)
        error("could not solve the MDP linear program")
    end
    
    (status = :optimal,
     value = value.(v),
     policy = map(x->argmax(dual.(x)), π))
end

lp_solve(model::TabMDP, γ::Number, lpm; args...) =
    lp_solve(model, InfiniteH(γ), lpm; args...)
    
