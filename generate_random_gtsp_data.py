# -*- coding: utf-8 -*-
"""Generate reproducible random GTSP instances for computational testing."""

import math
import random
from typing import Any


NUM_OPTIONS = 4
DEFAULT_NUM_CELLS = 8
DEFAULT_RANDOM_SEED = 1234


def generate_random_gtsp_instance(
    num_cells: int = DEFAULT_NUM_CELLS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Create a reproducible random GTSP instance with four options per cell."""

    rng = random.Random(random_seed)

    cells = list(range(1, num_cells + 1))
    options = list(range(1, NUM_OPTIONS + 1))
    nodes = list(range(1, NUM_OPTIONS * num_cells + 1))

    node_to_cell = {
        node_id: ((node_id - 1) // NUM_OPTIONS) + 1
        for node_id in nodes
    }

    # Randomized intra-cell coverage distances.
    intra_cost = {}
    for node_id in nodes:
        cell = node_to_cell[node_id]
        baseline = 100 + 10 * (cell - 1)
        intra_cost[node_id] = baseline + rng.randint(0, 20)

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

        entry_x = 100 * cell + rng.uniform(0, 20)
        entry_y = rng.uniform(0, 40)

        exit_x = entry_x + rng.uniform(3, 10)
        exit_y = entry_y + rng.uniform(-5, 5)

        entry_point[node_id] = (entry_x, entry_y)
        exit_point[node_id] = (exit_x, exit_y)

    # Inter-cell transfer distances.
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
    instance = generate_random_gtsp_instance()

    print("Random GTSP instance generated.")
    print(f"Cells: {len(instance['cells'])}")
    print(f"Nodes: {len(instance['nodes'])}")
    print(f"Edges: {len(instance['edges'])}")