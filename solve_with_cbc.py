# -*- coding: utf-8 -*-
"""CBC solver wrapper for the GTSP Pyomo model."""

import pyomo.environ as pyo


# ------------------------------------------------------------------
# CBC solver execution
# ------------------------------------------------------------------
def solve_with_cbc(model, params):
    """Solve a Pyomo model with CBC using the supplied configuration."""

    # Create the CBC solver interface and verify that the executable
    # is available in the active environment.
    solver = pyo.SolverFactory("cbc")
    solver.available(exception_flag=True)

    # Translate shared configuration values into CBC option names.
    solver_options = {}

    if params.get("time_limit") is not None:
        solver_options["sec"] = params["time_limit"]

    if params.get("mip_gap") is not None:
        solver_options["ratio"] = params["mip_gap"]

    # Export the algebraic model before optimization when requested.
    if params.get("save_lp"):
        model.write(
            params["lp_file_name"],
            io_options={"symbolic_solver_labels": True},
        )

    # Run CBC and display the solver log in the console.
    result = solver.solve(
        model,
        tee=True,
        options=solver_options,
    )

    # Export model values after optimization when requested.
    if params.get("save_solution"):
        model.write(params["solution_file_name"])

    return result