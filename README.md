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
- flow-balance constraints,
- a minimum-cost objective.

Supported solver interfaces:

- CBC
- GLPK
- HiGHS

### Hybrid GA-DP

The hybrid heuristic separates the problem into two levels:

- the genetic algorithm determines the order of field cells,
- dynamic programming selects the minimum-cost coverage option for each cell for that fixed order.

The DP evaluator includes the return arc from the final selected option to the same fixed first option, ensuring a valid closed cycle.

## Repository Structure

```text
.
├── HeuSolve/
│   ├── hybrid_ga_dp_engine.py
│   └── run_hybrid_ga_dp.py
├── constraints_catalog.py
├── generate_random_gtsp_data.py
├── generate_toy_gtsp_data.py
├── objective_catalog.py
├── requirements.txt
├── run_milp.py
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

## Run the Hybrid GA-DP Solver

From the repository root:

```bash
python HeuSolve/run_hybrid_ga_dp.py
```

The runner:

- generates a reproducible GTSP instance,
- configures the genetic algorithm,
- evaluates each permutation with dynamic programming,
- prints the detailed solution,
- saves a lightweight run log,
- saves a convergence plot.

## Generated Outputs

Generated files are written under:

```text
outputs/
├── heuristic/
├── logs/
└── milp/
```

The `outputs/` directory is excluded from version control.

## Reproducibility

The random instance generator and the hybrid GA-DP runner use fixed random seeds by default.

Using the same code, configuration, and dependency versions should reproduce the same generated instance and heuristic trajectory.

## Example Benchmark

For the tested 8-cell instance with 4 options per cell:

- MILP objective: `2433.010448`
- Hybrid GA-DP objective: `2433.010448`
- Hybrid GA-DP solver time: approximately `0.63 seconds`

The heuristic route matched the MILP cycle, up to rotation of the starting node.
