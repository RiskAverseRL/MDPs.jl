
# ----------------------------------------------------------------
# Markov reward process and Markov chain
# ----------------------------------------------------------------
using Graphs

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


function occupancy(model::TabMDP, π::AbstractVector{Int}, μ::AbstractVector{Float64})
    sum(μ) ≈ 1 && μ .≥ 0 || error("Initial distribution must be non-negative and sum to one.")

    S = state_count(model)
    r_π = zeros(S)
    P_π = spzeros(S, S)
    g = SimpleDiGraph(S)
    dist = Vector{Float64}(undef, S)

    for s ∈ 1:S
        for (sn, p, r) ∈ transition(model, s, π[s])
            P_π[s, sn] += p
            add_edge!(g, s, sn)
        end
    end

    comps = attracting_components(g)
    nr = fill(true, S)
    for comp ∈ comps
        nr[comp] .= false
    end
    tr = findall(nr)
    t2r = spzeros(length(tr), length(comps))
    for (i, comp) ∈ enumerate(comps)
        t2r[:, i] = sum(P_π[tr, comp], dims=2)
    end
    nav_probs = t2r' * ((I - P_π[tr, tr])' \ μ[tr])
    for (prob, comp) ∈ zip(nav_probs, comps)
        dist[comp] = (prob + sum(μ[comp])) * (vcat(I - P_π[comp, comp]', ones(1, length(comp))) \ vcat(zeros(length(comp)), 1))
    end
    dist[tr] .= 0
    return dist
end
