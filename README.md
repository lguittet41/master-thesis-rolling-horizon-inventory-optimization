````markdown
# Master Thesis: Rolling-Horizon Inventory Optimization

This repository contains the Julia implementation used for the computational experiments of my master thesis at DTU.

The project focuses on a rolling-horizon inventory optimization framework under demand and lead-time uncertainty. The framework combines deterministic optimization, stochastic simulation, and Monte Carlo evaluation to compare different inventory policies.

## Overview

The code is structured around a main rolling-horizon simulation function, `run_rolling_horizon`, which simulates the evolution of the inventory system over the planning horizon.

The main inputs of the framework include:

- forecast versions,
- lead-time hazard probabilities,
- initial inventory and backorder levels,
- shipments already in transit,
- policy parameters,
- cost and simulation parameters,
- random seed for stochastic simulation.

Different experiments can be selected through the `experiment_type` parameter in the main function.

## Available experiments

The following experiment types are available:

- `:diagnostic` — runs one trajectory for each main policy configuration;
- `:safety_stock` — evaluates different safety stock levels;
- `:demand_uplift` — evaluates different demand uplift values;
- `:worst_case` — evaluates a conservative lead-time policy.

## Requirements

The implementation uses the following Julia packages:

- `Random`
- `JuMP`
- `HiGHS`
- `Statistics`

## Running the code

To run the code, change the experiment type in the final line of the Julia file:

```julia
main(experiment_type = :diagnostic)
````

For example:

```julia
main(experiment_type = :safety_stock)
```

## Confidentiality note

The original thesis case study is based on company-related data. For confidentiality reasons, any public version of this repository should only include anonymized or illustrative input data.

