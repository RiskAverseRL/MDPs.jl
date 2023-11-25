MDPs: Markov Decision Processes
===============================

[![Build Status](https://github.com/RiskAverseRL/MDPs/workflows/CI/badge.svg)](https://github.com/RiskAverseRL/MDPs/actions)

**This is experimental software which can fail or change in unpredictable ways.**

A set of tools for solving primarily tabular Markov Decision Processes. The project aims to 

1. Make it easy to quickly and reliably solve tabular MDPs using standard techniques. `GenericMDP` serves this purpose by assuming MDPs in which states and actions are indexed by integers. 
2. Allow formulating MDPs with tabular state spaces but do not require that they map to integers. `TabMDP` serves this purpose and allows easy extensibility with custom state and action types. 
3. Provide tools to analyze MDP value functions and policies.
4. Provide a framework that can used to extent the algorithms to other objectives, including risk-averse and robust MDPs. The package can be extended by defining new objectives `InfiniteH` and `FiniteH` that change how the dynamic program works.
5. Make it easy to simulate a tabular MDP. This functionality can be used to generate training data and evaluate how well one would
be able to learn to act from this data. The module `Simulate` serves this purpose.
6. Provide data structures to allow for algorithms that approximate the value function, such as fitted value iteration. The class `MDP` serves this role. 

The project is build around the following main data structures:

- `MDP`: A structure that supports Bellman updates and value function and qvalue computation. The model assumes that the number of actions is discrete and sufficiently small.
- `TabMDP`: A structure that specialized MDPs to tabular state spaces. The states and actions may have arbitrary data types
- `GenMDP`: A generic tabular MDP which has states and actions indexed by integers. States 0 and below are assumed to be terminal and allow no actions or transitions

One of the main goals of the project is to support exploration of algorithms for solving various reinforcement learning objectives. To achieve this goal, the project involves a number of abstraction. The benefit of this approach is that one can implement new dynamic programming algorithm with a relatively high degree of code reuse.


## Similar packages

[POMDPs](https://github.com/JuliaPOMDP/POMDPs.jl) is a comprehensive package that suports both MDPs and POMDPs and offers a variety of solvers.
