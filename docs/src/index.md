MDPs.jl: Markov Decision Processes
==================================

## Models

This describes the data structures that can be used to model various types on MDPs.

### MDP

This is a general MDP data structure that supports basic functions. See IntMDP and TabMDP below for more models that can be used more directly to model and solve.

```@docs
MDP
```

### Tabular MDPs

This is an MDP instance that assumes that the states and actions are tabular. 

```@docs
TabMDP
```

### Integral MDPs

This is a specific MDP instance in which states and actions are specified by integers. 

```@docs
IntMDP
```

## Objectives

```@docs
FiniteH
```

```@docs
InfiniteH
```

## Algorithms

```@docs
value_iteration
```

```@docs
policy_iteration
```

## Value Function Manipulation


## Simulation

## Domains
