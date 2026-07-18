# -*- coding: utf-8 -*-
"""Coordinate shared GTSP data generation and optimization-engine execution."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from generate_random_gtsp_data import generate_random_gtsp_instance
from run_milp import main as run_milp
from HeuSolve.hybrid_ga_dp_engine import GAConfig
from HeuSolve.run_hybrid_ga_dp import main as run_hybrid_ga_dp

# ------------------------------------------------------------------
# Pipeline configuration
# ------------------------------------------------------------------
# Select which optimization engine to execute:
# "milp", "ga_dp", or "both".
METHOD = "both"

# Shared GTSP instance settings.
NUMBER_OF_CELLS = 5
INSTANCE_SEED = 1234

# Exact MILP solver selection.
MILP_SOLVER = "highs"

# Hybrid GA-DP configuration.
GA_SEED = 1
GA_POPULATION_SIZE = 20
GA_NUMBER_OF_GENERATIONS = 80
GA_CROSSOVER_RATE = 0.90
GA_MUTATION_RATE = 0.20
GA_NUMBER_OF_ELITES = 2
GA_TOURNAMENT_SIZE = 3

# Output controls.
SAVE_EXPERIMENT_RESULTS = True
EXPERIMENT_RESULTS_FILE = Path(
    "outputs/experiment_results.csv"
)

# ------------------------------------------------------------------
# Experiment-result schema
# ------------------------------------------------------------------
# A shared schema allows MILP and GA-DP results to be appended
# to the same CSV file. Fields that do not apply to one method
# remain empty for that row.
RESULT_FIELDS = [
    "run_timestamp",
    "method",
    "number_of_cells",
    "number_of_options",
    "instance_seed",
    "ga_seed",
    "solver",
    "objective_value",
    "solver_wall_time",
    "total_wall_time",
    "termination_condition",
    "relative_gap",
    "population_size",
    "number_of_generations",
    "crossover_rate",
    "mutation_rate",
    "number_of_elites",
    "tournament_size",
    "run_status",
    "log_file",
    "convergence_plot_file",
]

# ------------------------------------------------------------------
# Structured result output
# ------------------------------------------------------------------
def append_result_to_csv(
    run_summary: dict,
    results_file: Path,
) -> None:
    """Append one optimization-run summary to the shared CSV file."""

    # Create the parent output directory before writing the file.
    results_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = results_file.exists()

    # Normalize the runner-specific result dictionary to the common
    # CSV schema. Missing method-specific fields remain blank.
    row = {
        field: run_summary.get(field)
        for field in RESULT_FIELDS
    }

    # Record when this pipeline result row was written.
    row["run_timestamp"] = datetime.now().isoformat(
        timespec="seconds"
    )

    with results_file.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=RESULT_FIELDS,
        )

        # Write the header only when the CSV is created for the first time.
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
        
# ------------------------------------------------------------------
# GA configuration builder
# ------------------------------------------------------------------
def build_pipeline_ga_config() -> GAConfig:
    """Build the GA configuration defined by the pipeline settings."""

    return GAConfig(
        population_size=GA_POPULATION_SIZE,
        num_generations=GA_NUMBER_OF_GENERATIONS,
        crossover_rate=GA_CROSSOVER_RATE,
        mutation_rate=GA_MUTATION_RATE,
        num_elites=GA_NUMBER_OF_ELITES,
        tournament_size=GA_TOURNAMENT_SIZE,
        random_seed=GA_SEED,

        # Keep the pipeline output readable during larger experiment runs.
        print_progress=True,
        progress_interval=1,
    )
        
# ------------------------------------------------------------------
# Pipeline validation
# ------------------------------------------------------------------
def validate_pipeline_configuration() -> None:
    """Validate the selected execution mode and core run settings."""

    supported_methods = {"milp", "ga_dp", "both"}

    if METHOD not in supported_methods:
        raise ValueError(
            f"Unsupported METHOD '{METHOD}'. "
            f"Choose from: {sorted(supported_methods)}."
        )

    if NUMBER_OF_CELLS < 2:
        raise ValueError(
            "NUMBER_OF_CELLS must be at least 2."
        )

    if INSTANCE_SEED < 0:
        raise ValueError(
            "INSTANCE_SEED must be non-negative."
        )

    if GA_SEED < 0:
        raise ValueError(
            "GA_SEED must be non-negative."
        )
        
# ------------------------------------------------------------------
# Pipeline execution
# ------------------------------------------------------------------
def run_pipeline() -> list[dict]:
    """Generate one shared instance and execute the selected engine or engines."""

    validate_pipeline_configuration()

    # Generate the GTSP instance once so both engines receive
    # exactly the same cells, options, feasible arcs, and costs.
    shared_data = generate_random_gtsp_instance(
        num_cells=NUMBER_OF_CELLS,
        random_seed=INSTANCE_SEED,
    )

    run_summaries = []

    # Execute the exact MILP workflow when requested.
    if METHOD in {"milp", "both"}:
        milp_summary = run_milp(
            data=shared_data,
            num_cells=NUMBER_OF_CELLS,
            instance_seed=INSTANCE_SEED,
            solver_name=MILP_SOLVER,
        )

        run_summaries.append(milp_summary)

    # Execute the hybrid GA-DP workflow when requested.
    if METHOD in {"ga_dp", "both"}:
        ga_summary = run_hybrid_ga_dp(
            data=shared_data,
            number_of_cells=NUMBER_OF_CELLS,
            instance_seed=INSTANCE_SEED,
            ga_seed=GA_SEED,
            ga_config=build_pipeline_ga_config(),
        )

        run_summaries.append(ga_summary)

    return run_summaries

# ------------------------------------------------------------------
# Pipeline result handling
# ------------------------------------------------------------------
def save_and_report_results(
    run_summaries: list[dict],
) -> None:
    """Save structured results and print a concise pipeline summary."""

    print("\n========== PIPELINE SUMMARY ==========")

    for run_summary in run_summaries:
        method = run_summary["method"]
        objective_value = run_summary["objective_value"]
        runtime = run_summary["solver_wall_time"]
        status = run_summary["run_status"]

        print(f"\nMethod: {method}")
        print(f"Run status: {status}")

        if objective_value is not None:
            print(f"Objective value: {objective_value:.6f}")
        else:
            print("Objective value: not available")

        if runtime is not None:
            print(f"Solver wall time: {runtime:.4f} seconds")
        else:
            print("Solver wall time: not available")

        # Append one standardized row for every executed engine.
        if SAVE_EXPERIMENT_RESULTS:
            append_result_to_csv(
                run_summary,
                EXPERIMENT_RESULTS_FILE,
            )

    if SAVE_EXPERIMENT_RESULTS:
        print(
            "\nExperiment results saved to: "
            f"{EXPERIMENT_RESULTS_FILE}"
        )
        
# ------------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    summaries = run_pipeline()
    save_and_report_results(summaries)