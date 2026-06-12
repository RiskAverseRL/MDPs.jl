
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
    model::TabMDP, π::AbstractVector{<:Integer})
    S = state_count(model)
    fill!(P_π, 0.)
    fill!(r_π, 0.)
    for s ∈ 1:S
        for (sn, p, r) ∈ transition(model, s, π[s])
            # P_π[s,sn] ≈ 0. || error("duplicated transition entries (s1->s2, s1->s2) not allowed")
            P_π[s, sn] += p
            r_π[s] += p * r
        end
    end
end

"""
    mrp(model, π)

Compute the transition matrix `P_π` and reward vector `r_π` for the
MDP `model` and policy `π`. See mrp! for more details.
"""
function mrp(model::TabMDP, π::AbstractVector{<:Integer})
    S = state_count(model)
    P_π = Matrix{Float64}(undef, S, S)
    r_π = Vector{Float64}(undef, S)
    mrp!(P_π, r_π, model, π)
    (P_π, r_π)
end

"""
    mrp(model, π)

Compute a sparse transition matrix `P_π` and reward vector `r_π` for the
MDP `model` and policy `π`.
"""
function mrp_sparse(model::TabMDP, π::AbstractVector{Int})
    S = state_count(model)
    r_π = zeros(S)
    P_π = spzeros(S, S)

    for s ∈ 1:S
        for (sn, p, r) ∈ transition(model, s, π[s])
            P_π[s, sn] += p
            r_π[s] += p * r
        end
    end
    (P_π, r_π)
end
