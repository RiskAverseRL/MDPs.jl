MDPs: Markov Decision Processes
===============================

[![Build Status](https://github.com/RiskAverseRL/MDPs/workflows/CI/badge.svg)](https://github.com/RiskAverseRL/MDPs/actions)

[!(https://img.shields.io/badge/docs-latest-blue.svg)](https://riskaverserl.github.io/MDPs.jl/dev/)


**This is experimental software which can fail or change in unpredictable ways.**

A set of tools for solving primarily tabular Markov Decision Processes. The project aims to 

1. Quickly and reliably solve small tabular MDPs with integral states and actions using standard algorithms. See `IntMDP`.
2. Formulate and solve generic MDPs. See `TabMDP`.
3. Analyze MDP value functions and policies. See `value_iteration` and `policy_iteration`.
4. Framework algorithms that easily extend to new objectives. See `InfiniteH` and `FiniteH`.
5. Simulate tabular MDPs. See `Simulate`.
6. Provide a framework for solving general large MDPs. See `MDP`.

The project is build around the following main data structures:

- `MDP`: A structure that supports Bellman updates and value function and qvalue computation. The model assumes that the number of actions is discrete and sufficiently small.
- `TabMDP`: A structure that specialized MDPs to tabular state spaces. The states and actions may have arbitrary data types
- `IntMDP`: A generic tabular MDP which has states and actions indexed by integers. States 0 and below are assumed to be terminal and allow no actions or transitions

One of the main goals of the project is to support exploration of algorithms for solving various reinforcement learning objectives. To achieve this goal, the project involves a number of abstraction. The benefit of this approach is that one can implement new dynamic programming algorithm with a relatively high degree of code reuse.


## Similar packages

[POMDPs](https://github.com/JuliaPOMDP/POMDPs.jl) is a comprehensive package that suports both MDPs and POMDPs and offers a variety of solvers.
