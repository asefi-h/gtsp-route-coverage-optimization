# -*- coding: utf-8 -*-
"""Solver selection and shared configuration for the GTSP MILP model."""


# ------------------------------------------------------------------
# Solver activation settings
# ------------------------------------------------------------------
# Exactly one solver must be active when the model is executed.
SOLVER_SWITCHES = {
    "cbc": False,
    "glpk": True,
    "gurobi": False,
    "highs": False,
}


# ------------------------------------------------------------------
# Shared solver parameters
# ------------------------------------------------------------------
# Individual solver wrappers translate these generic settings into
# solver-specific option names where supported.
SOLVER_PARAMS = {
    "time_limit": 300,
    "mip_gap": 0.0001,
    "allow_infeasible": False,

    # Optional algebraic-model export.
    "save_lp": False,
    "lp_file_name": "outputs/milp/model.lp",

    # Optional solver-solution export.
    "save_solution": False,
    "solution_file_name": "outputs/milp/solution.sol",

    # Lightweight run-summary log.
    "save_run_log": True,
    "log_directory": "outputs/logs",
}