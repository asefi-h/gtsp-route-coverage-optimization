# -*- coding: utf-8 -*-
"""HiGHS solver wrapper for the GTSP Pyomo model."""

import pyomo.environ as pyo


# ------------------------------------------------------------------
# HiGHS solver execution
# ------------------------------------------------------------------
def solve_with_highs(model, params):
    """Solve a Pyomo model with HiGHS using the supplied configuration."""

    # Create the HiGHS solver interface and verify that the required
    # Python solver package is available in the active environment.
    solver = pyo.SolverFactory("appsi_highs")
    solver.available(exception_flag=True)

    # Configure solver output and shared optimization limits.
    solver.config.stream_solver = True

    if params.get("time_limit") is not None:
        solver.config.time_limit = params["time_limit"]

    if params.get("mip_gap") is not None:
        solver.config.mip_gap = params["mip_gap"]

    # Export the algebraic model before optimization when requested.
    if params.get("save_lp"):
        model.write(
            params["lp_file_name"],
            io_options={"symbolic_solver_labels": True},
        )

    # Run HiGHS and load the resulting variable values into the model.
    result = solver.solve(model)

    # Export model values after optimization when requested.
    if params.get("save_solution"):
        model.write(params["solution_file_name"])

    return result