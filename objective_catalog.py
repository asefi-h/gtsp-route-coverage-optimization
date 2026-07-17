# -*- coding: utf-8 -*-
"""Objective definitions and activation settings for the GTSP MILP model."""


# ------------------------------------------------------------------
# Total route-cost objective
# ------------------------------------------------------------------
def total_route_cost_objective(model):
    """Minimize intra-cell coverage and inter-cell transfer distance."""

    # The cost assigned to arc (i, j) includes:
    #   1. the intra-cell coverage distance of node i, and
    #   2. the transfer distance from node i to node j.
    return sum(
        model.cost[source, target] * model.x[source, target]
        for source, target in model.E
    )


# ------------------------------------------------------------------
# Inter-cell transfer-only objective
# ------------------------------------------------------------------
def inter_region_cost_objective(model):
    """Minimize only the inter-cell transfer distance."""

    # This alternative objective excludes intra-cell coverage distance.
    # It is retained for diagnostic and comparative model runs.
    return sum(
        model.L_inter[source, target] * model.x[source, target]
        for source, target in model.E
    )


# ------------------------------------------------------------------
# Objective builder registry
# ------------------------------------------------------------------
# The main model uses this mapping to attach the selected objective
# without hard-coding a specific objective function.
OBJECTIVE_FUNCTIONS = {
    "total_cost": total_route_cost_objective,
    "inter_only": inter_region_cost_objective,
}


# ------------------------------------------------------------------
# Objective activation settings
# ------------------------------------------------------------------
# Exactly one objective must be active when the model is built.
OBJECTIVE_SWITCHES = {
    "total_cost": True,
    "inter_only": False,
}