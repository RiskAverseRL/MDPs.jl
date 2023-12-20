MDPs.jl: Markov Decision Processes
==================================

## Models

This section describes the data structures that can be used to model various types on MDPs.

### MDP

This is a general MDP data structure that supports basic functions. See IntMDP and TabMDP below for more models that can be used more directly to model and solve.

```@autodocs
Modules = [MDPs]
Pages = ["mdp.jl"]
```

### Tabular MDPs

This is an MDP instance that assumes that the states and actions are tabular. 

```@autodocs
Modules = [MDPs]
Pages = ["tabular.jl"]
```

### Integral MDPs

This is a specific MDP instance in which states and actions are specified by integers. 

```@autodocs
Modules = [MDPs]
Pages = ["integral.jl"]
```

## Objectives


```@autodocs
Modules = [MDPs]
Pages = ["objectives.jl"]
```

## Algorithms

```@autodocs
Modules = [MDPs]
Pages = ["valueiteration.jl"]
```

```@autodocs
Modules = [MDPs]
Pages = ["mrp.jl"]
```

```@autodocs
Modules = [MDPs]
Pages = ["policyiteration.jl"]
```
## Value Function Manipulation

```@autodocs
Modules = [MDPs]
Pages = ["valuefunction.jl"]
```

```@autodocs
Modules = [MDPs]
Pages = ["bellman.jl"]
```

## Simulation


```@autodocs
Modules = [MDPs]
Pages = ["simulation.jl"]
```

## Domains


```@autodocs
Modules = [MDPs.Domains]
```

```@autodocs
Modules = [MDPs.Domains.Gambler]
```


```@autodocs
Modules = [MDPs.Domains.Inventory]
```

```@autodocs
Modules = [MDPs.Domains.Machine]
```
