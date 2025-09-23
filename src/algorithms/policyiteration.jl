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
        greedy!(policy, model, InfiniteH(γ), v_π)
        mrp!(IP_π, r_π, model, policy)
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
        greedy!(view(policy,:,fl), model, InfiniteH(γ), v_π)
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

function modified_policy_iteration(model::TabMDP, γ::Real; iterations::Int = 1000, inner_iterations = state_count(model))
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
        greedy!(policy, model, InfiniteH(γ), v_π)
        mrp!(IP_π, r_π, model, policy)
        # Solve: v_π .= (I - γ * P_π) \ r_π
        lmul!(-γ, IP_π)
        _add_identity!(IP_π)
        for it2 ∈ 1:inner_iterations
            v_π .= IP_π * v_π .+ r_π
        end
        # check if there was a change
        if all(i->policy[i] == policyold[i], 1:S)
            itercount = it
            break
        end
        
    end
    (policy = policy, value = v_π, iterations = itercount)
end
