# -*- coding: utf-8 -*-
"""Run the hybrid GA-DP solver for a generated GTSP instance."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------
# The runner is stored inside HeuSolve, while the data generator is
# stored in the parent project directory.
# these imports, so noqa: E402 suppresses the intentional import-order warning (linter pupose).
CURRENT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = CURRENT_DIRECTORY.parent

if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.append(str(PROJECT_DIRECTORY))

from generate_random_gtsp_data import (  # noqa: E402
    DEFAULT_NUM_CELLS,
    generate_random_gtsp_instance,
)
from hybrid_ga_dp_engine import (  # noqa: E402
    GAConfig,
    GAResult,
    build_node_index_map,
    run_ga_gtsp,
)


# ------------------------------------------------------------------
# Output configuration
# ------------------------------------------------------------------
OUTPUT_CONFIG = {
    # Lightweight timestamped run summary.
    "save_run_log": True,
    "log_directory": "outputs/logs",

    # Optional convergence-curve export.
    "save_convergence_plot": True,
    "plot_directory": "outputs/heuristic",
}


# ------------------------------------------------------------------
# GA configuration
# ------------------------------------------------------------------
def build_ga_config(number_of_cells: int) -> GAConfig:
    """Build the GA configuration from the current problem size."""

    small_case_threshold = 70
    is_small_case = number_of_cells <= small_case_threshold

    if is_small_case:
        # Moderate population and generation limits for smaller instances.
        population_size = min(
            100,
            max(20, number_of_cells // 2),
        )

        number_of_generations = min(
            100,
            number_of_cells * 10,
        )

        tournament_size = 3
        mutation_rate = 0.20
        number_of_elites = 2

    else:
        # Larger instances use a larger population and stronger mutation.
        population_size = min(
            300,
            max(50, number_of_cells * 2),
        )

        number_of_generations = min(
            400,
            number_of_cells * 3,
        )

        tournament_size = max(
            3,
            int(population_size * 0.10),
        )

        mutation_rate = 0.30
        number_of_elites = 3

    return GAConfig(
        population_size=population_size,
        num_generations=number_of_generations,
        crossover_rate=0.90,
        mutation_rate=mutation_rate,
        num_elites=number_of_elites,
        tournament_size=tournament_size,
        random_seed=1,

        # Print every generation for small test cases. This can be
        # increased later for larger production-size experiments.
        print_progress=True,
        progress_interval=1,
    )


# ------------------------------------------------------------------
# Output-directory utilities
# ------------------------------------------------------------------
def prepare_output_directories(output_config: dict) -> None:
    """Create configured output directories when required."""

    if output_config.get("save_run_log"):
        Path(output_config["log_directory"]).mkdir(
            parents=True,
            exist_ok=True,
        )

    if output_config.get("save_convergence_plot"):
        Path(output_config["plot_directory"]).mkdir(
            parents=True,
            exist_ok=True,
        )


# ------------------------------------------------------------------
# Heuristic solution reporting
# ------------------------------------------------------------------
def print_solution_summary(
    result: GAResult,
    data: dict,
) -> None:
    """Print the detailed hybrid GA-DP solution."""

    node_for = build_node_index_map(data)

    print("\n========== HYBRID GA-DP RESULT ==========")
    print(f"Best objective value: {result.best_cost:.4f}")

    print("\nBest cell permutation:")
    print(f"  {result.best_permutation}")

    print("\nChosen option per cell:")
    for cell in result.best_permutation:
        option = result.chosen_options_per_cell[cell]
        node_id = node_for[cell][option]

        print(
            f"  Cell {cell}: option {option} "
            f"(node {node_id})"
        )

    print("\nRoute nodes:")
    print(f"  {result.route_nodes}")

    print("\nSelected arcs:")
    for source_node, target_node in result.route_arcs:
        print(f"  {source_node} -> {target_node}")


# ------------------------------------------------------------------
# Convergence-plot output
# ------------------------------------------------------------------
def save_convergence_plot(
    result: GAResult,
    config: GAConfig,
    number_of_cells: int,
    run_start: datetime,
    output_config: dict,
) -> Path | None:
    """Save the best-so-far objective value by generation."""

    if not output_config.get("save_convergence_plot"):
        return None

    timestamp = run_start.strftime("%Y%m%d_%H%M%S")

    plot_path = (
        Path(output_config["plot_directory"])
        / f"hybrid_ga_dp_convergence_{timestamp}.png"
    )

    generations = range(
        1,
        len(result.best_cost_history) + 1,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        generations,
        result.best_cost_history,
        marker="o",
        linewidth=1,
    )

    plt.title(
        "Hybrid GA-DP Best Objective by Generation\n"
        f"cells={number_of_cells}, "
        f"population={config.population_size}, "
        f"generations={config.num_generations}"
    )

    plt.xlabel("Generation")
    plt.ylabel("Best objective value")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return plot_path


# ------------------------------------------------------------------
# Lightweight run-log output
# ------------------------------------------------------------------
def write_run_log(
    log_lines: list[str],
    output_config: dict,
    run_start: datetime,
) -> Path | None:
    """Write the timestamped hybrid GA-DP run summary."""

    if not output_config.get("save_run_log"):
        return None

    timestamp = run_start.strftime("%Y%m%d_%H%M%S")

    log_path = (
        Path(output_config["log_directory"])
        / f"hybrid_ga_dp_run_{timestamp}.txt"
    )

    log_path.write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    return log_path


# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
def main() -> None:
    """Generate an instance, run GA-DP, and report the solution."""

    run_start = datetime.now()
    total_start_time = perf_counter()

    number_of_cells = DEFAULT_NUM_CELLS

    log_lines = [
        "Hybrid GA-DP Run Summary",
        f"Run start time: {run_start.isoformat(timespec='seconds')}",
        "Selected method: Hybrid GA-DP",
        f"Number of cells: {number_of_cells}",
    ]

    prepare_output_directories(OUTPUT_CONFIG)

    print("\n========== HYBRID GA-DP RUN ==========")
    print(f"Number of cells: {number_of_cells}")

    try:
        # ----------------------------------------------------------
        # Instance generation
        # ----------------------------------------------------------
        print("1/4 Generating GTSP instance...")

        data = generate_random_gtsp_instance(
            num_cells=number_of_cells,
        )

        number_of_options = len(data["options"])
        number_of_nodes = len(data["nodes"])
        number_of_arcs = len(data["edges"])

        print(
            f"    Generated {number_of_nodes} nodes "
            f"and {number_of_arcs} feasible arcs."
        )

        log_lines.extend(
            [
                f"Number of options per cell: {number_of_options}",
                f"Number of nodes: {number_of_nodes}",
                f"Number of feasible arcs: {number_of_arcs}",
                "Instance generation: completed",
            ]
        )

        # ----------------------------------------------------------
        # GA configuration
        # ----------------------------------------------------------
        print("2/4 Building hybrid GA-DP configuration...")

        config = build_ga_config(number_of_cells)

        print(
            f"    Population: {config.population_size}, "
            f"generations: {config.num_generations}"
        )

        print(
            f"    Crossover: {config.crossover_rate}, "
            f"mutation: {config.mutation_rate}"
        )

        print(
            f"    Elites: {config.num_elites}, "
            f"tournament size: {config.tournament_size}, "
            f"seed: {config.random_seed}"
        )

        log_lines.extend(
            [
                "Configuration: completed",
                f"Population size: {config.population_size}",
                f"Number of generations: {config.num_generations}",
                f"Crossover rate: {config.crossover_rate}",
                f"Mutation rate: {config.mutation_rate}",
                f"Number of elites: {config.num_elites}",
                f"Tournament size: {config.tournament_size}",
                f"Random seed: {config.random_seed}",
            ]
        )

        # ----------------------------------------------------------
        # Hybrid GA-DP execution
        # ----------------------------------------------------------
        print("3/4 Running hybrid GA-DP solver...")

        result = run_ga_gtsp(
            data,
            config,
        )

        log_lines.extend(
            [
                "Termination condition: generation limit reached",
                f"Generations completed: {result.generations_completed}",
                f"Solver wall time: {result.elapsed_time:.4f} seconds",
                "Feasible solution found: yes",
                f"Best objective value: {result.best_cost:.6f}",
                f"Best cell permutation: {result.best_permutation}",
                f"Route nodes: {result.route_nodes}",
            ]
        )

        # ----------------------------------------------------------
        # Result reporting and plot output
        # ----------------------------------------------------------
        print("4/4 Reporting solution and saving outputs...")

        print("\n========== HEURISTIC STATUS ==========")
        print("Method: Hybrid GA-DP")
        print("Termination condition: generation limit reached")
        print(
            f"Generations completed: "
            f"{result.generations_completed}"
        )
        print(
            f"Solver wall time: "
            f"{result.elapsed_time:.4f} seconds"
        )

        print_solution_summary(
            result,
            data,
        )

        plot_path = save_convergence_plot(
            result,
            config,
            number_of_cells,
            run_start,
            OUTPUT_CONFIG,
        )

        log_lines.extend(
            [
                "Run status: completed",
                (
                    "Convergence plot saved: "
                    f"{plot_path is not None}"
                ),
                (
                    f"Convergence plot path: {plot_path}"
                    if plot_path is not None
                    else "Convergence plot path: not applicable"
                ),
            ]
        )

    except Exception as error:
        log_lines.extend(
            [
                "Run status: failed",
                "Feasible solution found: no",
                f"Error type: {type(error).__name__}",
                f"Error message: {error}",
            ]
        )

        print(f"\nHybrid GA-DP run failed: {error}")

    finally:
        total_wall_time = perf_counter() - total_start_time

        log_lines.append(
            f"Total wall time: {total_wall_time:.4f} seconds"
        )

        log_path = write_run_log(
            log_lines,
            OUTPUT_CONFIG,
            run_start,
        )

        if log_path is not None:
            print(f"\nRun log saved to: {log_path}")


if __name__ == "__main__":
    main()