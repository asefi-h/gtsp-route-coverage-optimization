# Route Coverage Optimization for Autonomous Agricultural Vehicles

[![CI](https://github.com/asefi-h/gtsp-route-coverage-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/asefi-h/gtsp-route-coverage-optimization/actions/workflows/ci.yml)

This repository presents a case study on route coverage optimization for autonomous agricultural vehicles using two solution approaches:

- Mixed-Integer Linear Programming (MILP)
- Hybrid Genetic Algorithm with Dynamic Programming (GA-DP)

Each field cell has multiple feasible coverage options. The objective is to select one option per cell and determine a minimum-cost closed route through the selected options.

## Case Study

This project is part of a broader optimization and decision-support case study portfolio.

More information:

https://www.asefi.consulting/

## Solution Approaches

### MILP

The exact model includes:

- binary node-selection variables,
- binary directed-arc variables,
- MTZ variables for subtour elimination,
- one selected coverage option per cell,
- incoming- and outgoing-degree constraints,
- a minimum-cost objective.

Supported solver interfaces:

- CBC
- GLPK
- HiGHS

### Hybrid GA-DP

The hybrid heuristic separates the problem into two levels:

- the genetic algorithm determines the order of field cells,
- dynamic programming selects the minimum-cost coverage option for each cell for that fixed order.

The DP evaluator includes the return arc from the final selected option to the fixed first option, ensuring a valid closed cycle.

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── HeuSolve/
│   ├── hybrid_ga_dp_engine.py
│   └── run_hybrid_ga_dp.py
├── constraints_catalog.py
├── generate_random_gtsp_data.py
├── generate_toy_gtsp_data.py
├── objective_catalog.py
├── requirements.txt
├── run_milp.py
├── run_optimization_pipeline.py
├── solve_with_cbc.py
├── solve_with_glpk.py
├── solve_with_highs.py
└── solver_catalog.py
```

## Environment Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Tested Python dependencies:

```text
pyomo==6.9.5
matplotlib==3.10.7
highspy==1.12.0
```

CBC and GLPK are optional external solver executables and must be installed separately when those interfaces are used.

## Run the MILP Model

From the repository root:

```bash
python run_milp.py
```

The selected solver and solver parameters are configured in:

```text
solver_catalog.py
```

The runner:

- generates a reproducible GTSP instance or accepts a supplied instance,
- constructs the Pyomo MILP model,
- executes the selected solver,
- reports the selected coverage options and cyclic route,
- records the objective value, optimality gap, solver status, and runtime,
- saves a lightweight run log.

## Run the Hybrid GA-DP Solver

From the repository root:

```bash
python HeuSolve/run_hybrid_ga_dp.py
```

The runner:

- generates a reproducible GTSP instance or accepts a supplied instance,
- configures the genetic algorithm,
- evaluates each cell permutation with dynamic programming,
- reports the best sequence and coverage-option assignment,
- records the objective value, configuration, termination condition, and runtime,
- saves a lightweight run log,
- saves a convergence plot.

## Run the Optimization Pipeline

The repository-level pipeline coordinates individual and comparative optimization runs.

From the repository root:

```bash
python run_optimization_pipeline.py
```

The pipeline can execute:

- the MILP engine,
- the hybrid GA-DP engine,
- or both engines.

Its configuration defines:

- the number of field cells,
- the GTSP instance random seed,
- the selected solution method,
- the MILP solver,
- the GA random seed,
- the GA population size and generation limit,
- crossover, mutation, elitism, and tournament parameters,
- structured experiment-result output.

When both methods are selected, the GTSP instance is generated once and passed unchanged to both engines. This ensures that both methods use identical cells, coverage options, feasible transitions, and cost data.

The pipeline collects a standardized summary from each engine and appends one row per run to a shared CSV file.

## Generated Outputs

Generated files are written under:

```text
outputs/
├── experiment_results.csv
├── heuristic/
├── logs/
└── milp/
```

The experiment CSV records fields such as:

- execution timestamp,
- solution method,
- problem size,
- instance random seed,
- MILP solver or GA configuration,
- objective value,
- solver and total wall times,
- termination condition,
- MILP optimality gap when available,
- run status,
- run-log path,
- convergence-plot path when applicable.

The `outputs/` directory is excluded from version control.

## Reproducibility

The instance generator, MILP runner, hybrid GA-DP runner, and orchestration pipeline support explicit reproducibility settings.

Reproducible execution is supported through:

- fixed or configurable instance random seeds,
- configurable GA random seeds,
- explicit MILP solver selection,
- documented GA hyperparameters,
- shared in-memory instance data for comparative runs,
- standardized CSV result storage,
- pinned Python dependency versions.

When both engines are selected in the pipeline, they receive the same generated GTSP instance. This prevents differences in random data generation from affecting the comparison.

## Example Benchmark

For the tested 8-cell instance with 4 options per cell:

- MILP objective: `2433.010448`
- Hybrid GA-DP objective: `2433.010448`
- MILP relative optimality gap: `0.0%`
- Hybrid GA-DP solver time: less than `1 second`

The hybrid solution matched the MILP closed cycle up to rotation of the reported starting node.