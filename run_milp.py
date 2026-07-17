# -*- coding: utf-8 -*-
"""Build and solve the exact GTSP coverage-routing MILP model."""

from typing import Any

from datetime import datetime
from pathlib import Path
from time import perf_counter

import pyomo.environ as pyo

from constraints_catalog import (
    CONSTRAINT_FUNCTIONS,
    CONSTRAINT_SWITCHES,
)
from generate_random_gtsp_data import (
    DEFAULT_NUM_CELLS,
    generate_random_gtsp_instance,
)
from objective_catalog import (
    OBJECTIVE_FUNCTIONS,
    OBJECTIVE_SWITCHES,
)
from solve_with_cbc import solve_with_cbc
from solve_with_glpk import solve_with_glpk
from solve_with_highs import solve_with_highs
from solver_catalog import SOLVER_PARAMS, SOLVER_SWITCHES


# ------------------------------------------------------------------
# Solver registry
# ------------------------------------------------------------------
# Each catalog key is mapped to its solver-specific execution wrapper.
SOLVER_FUNCTIONS = {
    "cbc": solve_with_cbc,
    "glpk": solve_with_glpk,
    "highs": solve_with_highs,
}


# ------------------------------------------------------------------
# Model construction
# ------------------------------------------------------------------
def build_gtsp_model(data: dict[str, Any]) -> pyo.ConcreteModel:
    """Build the exact GTSP coverage-routing MILP model."""

    model = pyo.ConcreteModel()

    # --------------------------------------------------------------
    # Sets
    # --------------------------------------------------------------
    # N: GTSP nodes representing cell-coverage options.
    model.N = pyo.Set(initialize=data["nodes"])

    # K: decomposed field cells.
    model.K = pyo.Set(initialize=data["cells"])

    # O: candidate coverage options available for each cell.
    model.O = pyo.Set(initialize=data["options"])

    # E: feasible directed transitions between option nodes.
    model.E = pyo.Set(
        within=model.N * model.N,
        initialize=data["edges"],
    )

    # --------------------------------------------------------------
    # Parameters
    # --------------------------------------------------------------
    # Mapping from each option node to its associated field cell.
    model.cell_of = pyo.Param(
        model.N,
        initialize=data["node_to_cell"],
        within=pyo.Any,
    )

    # Intra-cell coverage distance associated with each option node.
    model.L_intra = pyo.Param(
        model.N,
        initialize=data["L_intra"],
        within=pyo.NonNegativeReals,
    )

    # Inter-cell transfer distance for each feasible directed arc.
    model.L_inter = pyo.Param(
        model.E,
        initialize=data["L_inter"],
        within=pyo.NonNegativeReals,
    )

    # Total arc cost:
    # intra-cell coverage distance of the source node
    # plus transfer distance to the target node.
    model.cost = pyo.Param(
        model.E,
        initialize=data["cost"],
        within=pyo.NonNegativeReals,
    )

    # --------------------------------------------------------------
    # Decision variables
    # --------------------------------------------------------------
    # x[i, j] = 1 when directed arc (i, j) is selected.
    model.x = pyo.Var(
        model.E,
        domain=pyo.Binary,
    )

    # y[i] = 1 when coverage-option node i is selected.
    model.y = pyo.Var(
        model.N,
        domain=pyo.Binary,
    )

    # u[k] stores the MTZ route position of cell k.
    model.u = pyo.Var(
        model.K,
        domain=pyo.NonNegativeReals,
    )

    # --------------------------------------------------------------
    # Constraint components
    # --------------------------------------------------------------
    for constraint_name, is_active in CONSTRAINT_SWITCHES.items():
        if not is_active:
            continue

        constraint_builder = CONSTRAINT_FUNCTIONS[constraint_name]
        constraint_component = constraint_builder(model)

        # Constraint builders return a Pyomo component that is attached
        # under the same name used in the activation catalog.
        setattr(
            model,
            constraint_name,
            constraint_component,
        )

    # --------------------------------------------------------------
    # Objective component
    # --------------------------------------------------------------
    active_objectives = [
        name
        for name, is_active in OBJECTIVE_SWITCHES.items()
        if is_active
    ]

    if len(active_objectives) != 1:
        raise RuntimeError(
            "Exactly one objective must be active in "
            "OBJECTIVE_SWITCHES."
        )

    objective_name = active_objectives[0]
    objective_builder = OBJECTIVE_FUNCTIONS[objective_name]

    model.objective = pyo.Objective(
        rule=lambda m: objective_builder(m),
        sense=pyo.minimize,
    )

    return model


# ------------------------------------------------------------------
# Solver selection
# ------------------------------------------------------------------
def select_solver():
    """Return the single active solver wrapper and its catalog name."""

    active_solvers = [
        name
        for name, is_active in SOLVER_SWITCHES.items()
        if is_active
    ]

    if len(active_solvers) != 1:
        raise RuntimeError(
            "Exactly one supported solver must be active in "
            "SOLVER_SWITCHES."
        )

    solver_name = active_solvers[0]

    if solver_name not in SOLVER_FUNCTIONS:
        raise RuntimeError(
            f"No execution wrapper is defined for solver "
            f"'{solver_name}'."
        )

    return solver_name, SOLVER_FUNCTIONS[solver_name]


# ------------------------------------------------------------------
# Solution extraction
# ------------------------------------------------------------------
def extract_selected_nodes(model) -> dict[int, int]:
    """Return the selected GTSP node for each cell."""

    selected_node_by_cell = {}

    for cell in model.K:
        selected_nodes = [
            node
            for node in model.N
            if (
                model.cell_of[node] == cell
                and pyo.value(model.y[node]) > 0.5
            )
        ]

        if len(selected_nodes) != 1:
            raise RuntimeError(
                f"Cell {cell} has an invalid selected-node set: "
                f"{selected_nodes}"
            )

        selected_node_by_cell[cell] = selected_nodes[0]

    return selected_node_by_cell


def extract_selected_arcs(model) -> list[tuple[int, int]]:
    """Return all directed arcs selected in the MILP solution."""

    return [
        (source, target)
        for source, target in model.E
        if (
            pyo.value(model.x[source, target]) is not None
            and pyo.value(model.x[source, target]) > 0.5
        )
    ]


def reconstruct_cycle(
    selected_arcs: list[tuple[int, int]],
) -> list[int]:
    """Reconstruct the selected closed node cycle."""

    if not selected_arcs:
        return []

    next_node = {
        source: target
        for source, target in selected_arcs
    }

    # A closed cycle has no unique natural starting node.
    # The smallest selected node is used only for deterministic reporting.
    start_node = min(next_node)

    route = [start_node]
    current_node = start_node

    while True:
        current_node = next_node[current_node]

        if current_node == start_node:
            break

        if current_node in route:
            raise RuntimeError(
                "Selected arcs do not form one valid closed cycle."
            )

        route.append(current_node)

    if len(route) != len(selected_arcs):
        raise RuntimeError(
            "Selected arcs contain more than one disconnected cycle."
        )

    return route


# ------------------------------------------------------------------
# Solution reporting
# ------------------------------------------------------------------
def print_solution_summary(model) -> None:
    """Print the objective value and selected closed route."""

    objective_value = pyo.value(model.objective)

    if objective_value is None:
        raise RuntimeError(
            "The solver did not return a valid objective value."
        )

    selected_node_by_cell = extract_selected_nodes(model)
    selected_arcs = extract_selected_arcs(model)
    route_nodes = reconstruct_cycle(selected_arcs)

    print("\n========== MILP RESULT ==========")
    print(f"Objective value: {objective_value:.4f}")

    print("\nChosen option per cell:")
    for cell in sorted(model.K):
        node = selected_node_by_cell[cell]
        option = ((node - 1) % len(model.O)) + 1

        print(
            f"  Cell {cell}: option {option} "
            f"(node {node})"
        )

    print("\nRoute nodes:")
    print(f"  {route_nodes}")

    print("\nSelected arcs:")
    for source, target in selected_arcs:
        print(f"  {source} -> {target}")

    print("\nMTZ order values:")
    for cell in sorted(model.K):
        order_value = pyo.value(model.u[cell])
        print(f"  u[{cell}] = {round(order_value)}")

# ------------------------------------------------------------------
# Solver-result and run-log utilities
# ------------------------------------------------------------------
def get_termination_condition(result) -> str:
    """Return a readable solver termination condition."""

    # Standard Pyomo solver-results structure used by CBC and GLPK.
    if hasattr(result, "solver"):
        condition = getattr(result.solver, "termination_condition", None)
        if condition is not None:
            return str(condition)

    # APPSI result structure used by HiGHS.
    condition = getattr(result, "termination_condition", None)
    if condition is not None:
        return str(condition)

    return "unknown"


def get_solver_gap(
    result,
    objective_value,
    termination_condition,
):
    """Return the relative optimality gap when solver bounds are available."""

    if str(termination_condition).lower() == "optimal":
        return 0.0

    lower_bound = None
    upper_bound = None

    # Standard Pyomo result structure.
    if hasattr(result, "problem"):
        try:
            lower_bound = result.problem.lower_bound
            upper_bound = result.problem.upper_bound
        except (AttributeError, IndexError, TypeError):
            pass

    # APPSI result structure.
    if lower_bound is None:
        lower_bound = getattr(result, "best_objective_bound", None)

    if upper_bound is None:
        upper_bound = getattr(result, "best_feasible_objective", None)

    # For a minimization problem, the incumbent objective can be used
    # as the upper bound when the solver does not report it separately.
    if upper_bound is None:
        upper_bound = objective_value

    if lower_bound is None or upper_bound is None:
        return None

    try:
        denominator = max(abs(float(upper_bound)), 1e-10)
        return abs(float(upper_bound) - float(lower_bound)) / denominator
    except (TypeError, ValueError):
        return None


def prepare_output_directories(params) -> None:
    """Create configured output directories when required."""

    if params.get("save_lp"):
        Path(params["lp_file_name"]).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    if params.get("save_solution"):
        Path(params["solution_file_name"]).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    if params.get("save_run_log"):
        Path(params["log_directory"]).mkdir(
            parents=True,
            exist_ok=True,
        )


def write_run_log(log_lines, params, run_start) -> Path | None:
    """Write the lightweight timestamped MILP run log."""

    if not params.get("save_run_log"):
        return None

    timestamp = run_start.strftime("%Y%m%d_%H%M%S")
    log_path = (
        Path(params["log_directory"])
        / f"milp_run_{timestamp}.txt"
    )

    log_path.write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    return log_path

def get_model_statistics(model) -> dict[str, int]:
    """Count active variables and constraints by model type."""

    binary_variables = sum(
        1
        for variable in model.component_data_objects(
            pyo.Var,
            active=True,
        )
        if variable.is_binary()
    )

    integer_variables = sum(
        1
        for variable in model.component_data_objects(
            pyo.Var,
            active=True,
        )
        if variable.is_integer() and not variable.is_binary()
    )

    continuous_variables = sum(
        1
        for variable in model.component_data_objects(
            pyo.Var,
            active=True,
        )
        if not variable.is_integer()
    )

    constraints = sum(
        1
        for _ in model.component_data_objects(
            pyo.Constraint,
            active=True,
        )
    )

    return {
        "binary_variables": binary_variables,
        "integer_variables": integer_variables,
        "continuous_variables": continuous_variables,
        "total_variables": (
            binary_variables
            + integer_variables
            + continuous_variables
        ),
        "constraints": constraints,
    }

# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
def main():
    """Generate an instance, build the MILP, solve it, and report results."""

    run_start = datetime.now()
    total_start_time = perf_counter()

    num_cells = DEFAULT_NUM_CELLS
    log_lines = [
        "GTSP MILP Run Summary",
        f"Run start time: {run_start.isoformat(timespec='seconds')}",
        f"Number of cells: {num_cells}",
    ]

    prepare_output_directories(SOLVER_PARAMS)

    print("\n========== GTSP MILP RUN ==========")
    print(f"Number of cells: {num_cells}")

    try:
        # ----------------------------------------------------------
        # Instance generation
        # ----------------------------------------------------------
        print("1/4 Generating GTSP instance...")

        data = generate_random_gtsp_instance(
            num_cells=num_cells,
        )

        num_nodes = len(data["nodes"])
        num_arcs = len(data["edges"])

        print(
            f"    Generated {num_nodes} nodes "
            f"and {num_arcs} feasible arcs."
        )

        log_lines.extend(
            [
                f"Number of nodes: {num_nodes}",
                f"Number of feasible arcs: {num_arcs}",
                "Instance generation: completed",
            ]
        )

        # ----------------------------------------------------------
        # Model construction
        # ----------------------------------------------------------
        print("2/4 Building Pyomo MILP model...")

        model = build_gtsp_model(data)
        model_stats = get_model_statistics(model)

        print("    Model construction complete.")
        print(
            f"    Variables: {model_stats['total_variables']} total "
            f"({model_stats['binary_variables']} binary, "
            f"{model_stats['integer_variables']} integer, "
            f"{model_stats['continuous_variables']} continuous)"
        )
        print(
            f"    Constraints: {model_stats['constraints']}"
        )
        
        log_lines.extend(
            [
                "Model construction: completed",
                (
                    f"Variables: {model_stats['total_variables']} total "
                    f"({model_stats['binary_variables']} binary, "
                    f"{model_stats['integer_variables']} integer, "
                    f"{model_stats['continuous_variables']} continuous)"
                ),
                f"Constraints: {model_stats['constraints']}",
            ]
        )

        # ----------------------------------------------------------
        # Solver execution
        # ----------------------------------------------------------
        solver_name, solve_function = select_solver()

        print(f"3/4 Solving with {solver_name.upper()}...")
        log_lines.append(f"Selected solver: {solver_name}")

        solve_start_time = perf_counter()

        result = solve_function(
            model,
            SOLVER_PARAMS,
        )

        solve_wall_time = perf_counter() - solve_start_time
        termination_condition = get_termination_condition(result)

        log_lines.extend(
            [
                f"Termination condition: {termination_condition}",
                f"Solver wall time: {solve_wall_time:.4f} seconds",
            ]
        )

        # ----------------------------------------------------------
        # Solution extraction
        # ----------------------------------------------------------
        print("4/4 Extracting and reporting solution...")

        objective_value = None
        route_nodes = []

        try:
            objective_value = pyo.value(
                model.objective,
                exception=False,
            )
        except (ValueError, TypeError):
            objective_value = None

        if objective_value is not None:
            selected_arcs = extract_selected_arcs(model)
            route_nodes = reconstruct_cycle(selected_arcs)

            relative_gap = get_solver_gap(
                result,
                objective_value,
                termination_condition,
            )

            log_lines.append("Feasible solution found: yes")
            log_lines.append(
                f"Objective value: {objective_value:.6f}"
            )

            if relative_gap is not None:
                log_lines.append(
                    f"Relative optimality gap: "
                    f"{relative_gap:.6%}"
                )
            else:
                log_lines.append(
                    "Relative optimality gap: not available"
                )

            log_lines.append(f"Route nodes: {route_nodes}")

            # Print the solver-level status before the detailed solution report.
            print("\n========== SOLVER RESULT ==========")
            print(f"Termination condition: {termination_condition}")
            
            if relative_gap is not None:
                print(f"Relative optimality gap: {relative_gap:.4%}")
            else:
                print("Relative optimality gap: not available")
            
            print(f"Solver wall time: {solve_wall_time:.4f} seconds")
            
            # Print the complete MILP solution summary:
            # objective, selected option per cell, route nodes,
            # selected arcs, and MTZ route-order values.
            print_solution_summary(model)

        else:
            log_lines.extend(
                [
                    "Feasible solution found: no",
                    "Objective value: not available",
                    "Relative optimality gap: not available",
                    "Route nodes: not available",
                ]
            )

            print(
                "No feasible solution was available for reporting."
            )

    except Exception as error:
        log_lines.extend(
            [
                "Run status: failed",
                f"Error type: {type(error).__name__}",
                f"Error message: {error}",
            ]
        )

        print(f"\nMILP run failed: {error}")

    else:
        log_lines.append("Run status: completed")

    finally:
        total_wall_time = perf_counter() - total_start_time

        log_lines.extend(
            [
                f"Total wall time: {total_wall_time:.4f} seconds",
                f"LP file saved: {SOLVER_PARAMS['save_lp']}",
                (
                    f"LP file path: {SOLVER_PARAMS['lp_file_name']}"
                    if SOLVER_PARAMS["save_lp"]
                    else "LP file path: not applicable"
                ),
                (
                    "Solution file saved: "
                    f"{SOLVER_PARAMS['save_solution']}"
                ),
                (
                    "Solution file path: "
                    f"{SOLVER_PARAMS['solution_file_name']}"
                    if SOLVER_PARAMS["save_solution"]
                    else "Solution file path: not applicable"
                ),
            ]
        )

        log_path = write_run_log(
            log_lines,
            SOLVER_PARAMS,
            run_start,
        )

        if log_path is not None:
            print(f"\nRun log saved to: {log_path}")


if __name__ == "__main__":
    main()