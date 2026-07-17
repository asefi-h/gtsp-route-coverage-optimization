# -*- coding: utf-8 -*-
"""Generate a deterministic toy GTSP instance for route-coverage testing."""

import math
from typing import Any


NUM_OPTIONS = 4


def generate_toy_gtsp_instance(num_cells: int) -> dict[str, Any]:
    """Create a deterministic GTSP instance with four options per cell."""

    cells = list(range(1, num_cells + 1))
    options = list(range(1, NUM_OPTIONS + 1))
    nodes = list(range(1, NUM_OPTIONS * num_cells + 1))

    node_to_cell = {
        node_id: ((node_id - 1) // NUM_OPTIONS) + 1
        for node_id in nodes
    }

    # Intra-cell coverage distance for each node option.
    intra_cost = {}
    for node_id in nodes:
        cell = node_to_cell[node_id]
        option = ((node_id - 1) % NUM_OPTIONS) + 1
        intra_cost[node_id] = 100 + 10 * (cell - 1) + (option % NUM_OPTIONS)

    # Directed transitions are allowed only between different cells.
    edges = [
        (source, target)
        for source in nodes
        for target in nodes
        if node_to_cell[source] != node_to_cell[target]
    ]

    # Synthetic entry and exit coordinates for each coverage option.
    entry_point = {}
    exit_point = {}

    for node_id in nodes:
        cell = node_to_cell[node_id]
        option = ((node_id - 1) % NUM_OPTIONS) + 1

        entry_point[node_id] = (10 * cell, 10 * option)
        exit_point[node_id] = (10 * cell + 5, 10 * option)

    # Inter-cell transfer distance.
    inter_cost = {}
    for source, target in edges:
        dx = exit_point[source][0] - entry_point[target][0]
        dy = exit_point[source][1] - entry_point[target][1]
        inter_cost[source, target] = math.hypot(dx, dy)

    total_cost = {
        edge: intra_cost[edge[0]] + inter_cost[edge]
        for edge in edges
    }

    return {
        "nodes": nodes,
        "cells": cells,
        "options": options,
        "node_to_cell": node_to_cell,
        "edges": edges,
        "L_intra": intra_cost,
        "L_inter": inter_cost,
        "cost": total_cost,
    }


if __name__ == "__main__":
    instance = generate_toy_gtsp_instance(num_cells=5)

    print("Toy GTSP instance generated.")
    print(f"Cells: {len(instance['cells'])}")
    print(f"Nodes: {len(instance['nodes'])}")
    print(f"Edges: {len(instance['edges'])}")