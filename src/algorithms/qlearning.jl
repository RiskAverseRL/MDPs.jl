
# ------------------------------------------------------
# Basic q-learning algorithms
# -----------------------------------------------------


"""
    qlearning(trans; α :: Function)    

Runs the q-learning algorithm on a vector of transitions `trans`. The
stepsize is `α`, which can be a constant (number) or a function that maps
the number of samples for the state-action pair to a real number.  
"""
function qlearning(obj, trans :: AbtractVector{Transition{Int,Int}}; α)
    q = make_q(obj)
    c = make_c(obj)
    for tr ∈ trans
        qlearningstep!(obj, q, c, α)
    end
end

function make_q(obj, acounts)
end

function make_c(obj, acounts)
end


"""
    qlearningstep!(obj, q, c, tr)

Updates the q-function estimate `q` and sample-counts `c` using
the single step of following the transition `tr` and the step size `α`.


The q-function is:

* Infinite horizon: a vector of action values for each state
* Finite horizon: a vector of action values for each horizon and state.
         Horizon = number of decisions left  

"""
function qlearningstep! end


@inline function qlearningstep!(obj::InfiniteH, q::Vector{Vector{<:Number}},
                                c::Vector{Vector{<:Integer}}, tr::Transition{Int,Int},
                                α::Number) 
    α ≥ 0 || error("α must be non-negative")

    s = tr.state
    a = tr.action
    sn = tr.nstate
    r = tr.reward
    
    c[s][a] += 1
    q[s][a] = (1-α) * q[s][a] + α * (r + obj.γ * maximum(q[sn]))
end

@inline function qlearningstep!(obj::FiniteH, q::Matrix{Vector{<:Number}},
                                c::Vector{Vector{<:Integer}}, tr::Transition{Int,Int},
                                α::Number) 
    α ≥ 0 || error("α must be non-negative")

    s = tr.state
    a = tr.action
    sn = tr.nstate
    r = tr.reward
    k = tr.time
    t = obj.T - k + 1 # horizon left 

    qn = t ≥ 1 ? maximum(q[t-1,sn]) : zero(first(first(q))
    c[s][a] += 1
    q[t,s][a] = (1-α) * q[t,s][a] + α * (r + obj.γ * qn)
end
