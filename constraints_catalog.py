# -*- coding: utf-8 -*-
"""Constraint definitions and activation settings for the GTSP MILP model."""

import pyomo.environ as pyo


# ------------------------------------------------------------------
# Constraint activation settings
# ------------------------------------------------------------------
# Each entry controls whether the corresponding constraint component
# is added to the Pyomo model.
CONSTRAINT_SWITCHES = {
    "cluster_constraint": True,
    "outgoing_degree": True,
    "incoming_degree": True,
    "mtz_bounds_lower": True,
    "mtz_bounds_upper": True,
    "mtz_fix_root": True,
    "mtz_subtour": True,
}


# ------------------------------------------------------------------
# Exactly one coverage option per cell
# ------------------------------------------------------------------
def cluster_constraint(model):
    """Select exactly one GTSP node from each cell."""

    def rule(m, cell):
        # Collect all option nodes associated with the current cell.
        cell_nodes = [
            node
            for node in m.N
            if m.cell_of[node] == cell
        ]

        # Exactly one coverage option must be selected for each cell.
        return sum(m.y[node] for node in cell_nodes) == 1

    return pyo.Constraint(model.K, rule=rule)


# ------------------------------------------------------------------
# Outgoing degree constraint
# ------------------------------------------------------------------
def outgoing_degree_constraint(model):
    """Require one outgoing arc from every selected node."""

    def rule(m, node):
        # Sum all feasible arcs leaving the current node.
        outgoing_arcs = sum(
            m.x[source, target]
            for source, target in m.E
            if source == node
        )

        # A node has one outgoing arc only when it is selected.
        return outgoing_arcs == m.y[node]

    return pyo.Constraint(model.N, rule=rule)


# ------------------------------------------------------------------
# Incoming degree constraint
# ------------------------------------------------------------------
def incoming_degree_constraint(model):
    """Require one incoming arc to every selected node."""

    def rule(m, node):
        # Sum all feasible arcs entering the current node.
        incoming_arcs = sum(
            m.x[source, target]
            for source, target in m.E
            if target == node
        )

        # A node has one incoming arc only when it is selected.
        return incoming_arcs == m.y[node]

    return pyo.Constraint(model.N, rule=rule)


# ------------------------------------------------------------------
# MTZ reference-cell constraint
# ------------------------------------------------------------------
def mtz_fix_root_constraint(model):
    """Fix one reference cell at the first route position."""

    # The smallest cell index is used as the route reference.
    root = min(model.K)

    return pyo.Constraint(expr=model.u[root] == 1)


# ------------------------------------------------------------------
# MTZ lower bounds
# ------------------------------------------------------------------
def mtz_bounds_lower(model):
    """Set the lower MTZ bound for non-reference cells."""

    root = min(model.K)

    def rule(m, cell):
        # The reference cell is already fixed at position 1.
        if cell == root:
            return pyo.Constraint.Skip

        # All other cells must appear after the reference cell.
        return m.u[cell] >= 2

    return pyo.Constraint(model.K, rule=rule)


# ------------------------------------------------------------------
# MTZ upper bounds
# ------------------------------------------------------------------
def mtz_bounds_upper(model):
    """Set the upper MTZ bound for all cells."""

    num_cells = len(model.K)

    def rule(m, cell):
        # Route positions cannot exceed the number of cells.
        return m.u[cell] <= num_cells

    return pyo.Constraint(model.K, rule=rule)


# ------------------------------------------------------------------
# MTZ subtour-elimination constraints
# ------------------------------------------------------------------
def mtz_subtour_constraint(model):
    """Eliminate disconnected cycles using cell-level MTZ constraints."""

    num_cells = len(model.K)
    root = min(model.K)

    def rule(m, source_cell, target_cell):
        # Self-pairs and pairs involving the reference cell are excluded
        # from the standard MTZ ordering relation.
        if (
            source_cell == target_cell
            or source_cell == root
            or target_cell == root
        ):
            return pyo.Constraint.Skip

        # Aggregate selected node-level arcs from source_cell
        # to target_cell.
        selected_transition = sum(
            m.x[source_node, target_node]
            for source_node, target_node in m.E
            if (
                m.cell_of[source_node] == source_cell
                and m.cell_of[target_node] == target_cell
            )
        )

        # If a transition from source_cell to target_cell is selected,
        # target_cell must appear later in the route order.
        return (
            m.u[source_cell]
            - m.u[target_cell]
            + num_cells * selected_transition
            <= num_cells - 1
        )

    return pyo.Constraint(model.K, model.K, rule=rule)


# ------------------------------------------------------------------
# Constraint builder registry
# ------------------------------------------------------------------
# The main model uses this mapping to attach active constraints
# without hard-coding each builder call.
CONSTRAINT_FUNCTIONS = {
    "cluster_constraint": cluster_constraint,
    "outgoing_degree": outgoing_degree_constraint,
    "incoming_degree": incoming_degree_constraint,
    "mtz_bounds_lower": mtz_bounds_lower,
    "mtz_bounds_upper": mtz_bounds_upper,
    "mtz_fix_root": mtz_fix_root_constraint,
    "mtz_subtour": mtz_subtour_constraint,
}