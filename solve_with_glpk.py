# -*- coding: utf-8 -*-
"""GLPK solver wrapper for the GTSP Pyomo model."""

import pyomo.environ as pyo


# ------------------------------------------------------------------
# GLPK solver execution
# ------------------------------------------------------------------
def solve_with_glpk(model, params):
    """Solve a Pyomo model with GLPK using the supplied configuration."""

    # Create the GLPK solver interface and verify that the executable
    # is available in the active environment.
    solver = pyo.SolverFactory("glpk")
    solver.available(exception_flag=True)

    # Translate shared configuration values into GLPK option names.
    solver_options = {}

    if params.get("time_limit") is not None:
        solver_options["tmlim"] = params["time_limit"]

    if params.get("mip_gap") is not None:
        solver_options["mipgap"] = params["mip_gap"]

    # Export the algebraic model before optimization when requested.
    if params.get("save_lp"):
        model.write(
            params["lp_file_name"],
            io_options={"symbolic_solver_labels": True},
        )

    # Run GLPK and display the solver log in the console.
    result = solver.solve(
        model,
        tee=True,
        options=solver_options,
    )

    # Export model values after optimization when requested.
    if params.get("save_solution"):
        model.write(params["solution_file_name"])

    return result