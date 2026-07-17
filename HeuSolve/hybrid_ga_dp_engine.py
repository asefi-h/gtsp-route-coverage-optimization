# -*- coding: utf-8 -*-
"""Hybrid genetic algorithm and dynamic-programming engine for GTSP."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from time import perf_counter
from typing import Any


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------
@dataclass(frozen=True)
class GAConfig:
    """Configuration parameters for the hybrid GA-DP solver."""

    # Number of candidate cell permutations maintained in each generation.
    population_size: int

    # Number of evolutionary iterations performed after initialization.
    num_generations: int

    # Probability of applying order crossover to a selected parent pair.
    crossover_rate: float

    # Probability of applying swap mutation to each offspring permutation.
    mutation_rate: float

    # Number of best individuals copied directly into the next generation.
    num_elites: int

    # Number of competitors sampled during tournament parent selection.
    tournament_size: int

    # Optional seed for reproducible initialization and GA operators.
    random_seed: int | None = None

    # Console progress controls.
    print_progress: bool = True
    progress_interval: int = 1


@dataclass
class Individual:
    """One GA individual representing a permutation of field cells."""

    # The chromosome contains only the visit order of field cells.
    # Coverage options are selected later by dynamic programming.
    permutation: list[int]

    # Minimum cycle cost obtained after DP evaluation of the permutation.
    fitness: float = float("inf")

    # Best coverage option selected for each cell by the DP evaluator.
    chosen_options: dict[int, int] = field(default_factory=dict)

    # Node IDs corresponding to the selected options in route order.
    route_nodes: list[int] = field(default_factory=list)


@dataclass
class GAResult:
    """Final result and execution statistics from the hybrid GA-DP solver."""

    best_permutation: list[int]
    best_cost: float
    chosen_options_per_cell: dict[int, int]
    route_nodes: list[int]
    route_arcs: list[tuple[int, int]]

    # Best-so-far objective value recorded after each generation.
    best_cost_history: list[float]

    generations_completed: int
    elapsed_time: float

    # Compatibility with the earlier runner attribute name.
    @property
    def best_perm(self) -> list[int]:
        return self.best_permutation


# ------------------------------------------------------------------
# GTSP mapping and route utilities
# ------------------------------------------------------------------
def build_node_index_map(
    data: dict[str, Any],
) -> dict[int, dict[int, int]]:
    """Build the mapping node_for[cell][option] -> node ID."""

    node_for: dict[int, dict[int, int]] = {}

    options = list(data["options"])
    number_of_options = len(options)

    # The data generator assigns option nodes in contiguous blocks.
    # For four options, nodes 1-4 belong to cell 1, nodes 5-8 to cell 2,
    # and the option label is recovered from the node position.
    for node_id, cell_id in data["node_to_cell"].items():
        option = ((node_id - 1) % number_of_options) + 1

        if cell_id not in node_for:
            node_for[cell_id] = {}

        node_for[cell_id][option] = node_id

    return node_for


def transition_cost(
    source_cell: int,
    source_option: int,
    target_cell: int,
    target_option: int,
    data: dict[str, Any],
    node_for: dict[int, dict[int, int]],
) -> float:
    """Return the directed cost between two cell-option combinations."""

    # Convert the cell-option representation used by DP into the
    # node-pair representation used by the GTSP data dictionary.
    source_node = node_for[source_cell][source_option]
    target_node = node_for[target_cell][target_option]

    return float(data["cost"][(source_node, target_node)])


def build_route_arcs(
    route_nodes: list[int],
) -> list[tuple[int, int]]:
    """Build all directed route arcs, including cycle closure."""

    if len(route_nodes) < 2:
        return []

    # Add each consecutive transition along the route.
    route_arcs = [
        (route_nodes[index], route_nodes[index + 1])
        for index in range(len(route_nodes) - 1)
    ]

    # Add the final return arc to form a closed cycle.
    route_arcs.append((route_nodes[-1], route_nodes[0]))

    return route_arcs


# ------------------------------------------------------------------
# Dynamic-programming option evaluator
# ------------------------------------------------------------------
def evaluate_options_dp(
    cell_order: list[int],
    data: dict[str, Any],
    node_for: dict[int, dict[int, int]],
) -> tuple[float, dict[int, int], list[int]]:
    """
    Select the minimum-cost coverage option for a fixed cell sequence.

    The GA determines only the order of cells. For that fixed order, this
    evaluator selects one option for each cell such that the complete
    closed-cycle cost is minimized.

    Each candidate option for the first cell is fixed separately. Dynamic
    programming then determines the least-cost option sequence through the
    remaining cells and closes the route to that same first option.
    """

    if len(cell_order) < 2:
        raise ValueError(
            "A closed GTSP route requires at least two cells."
        )

    options = list(data["options"])
    number_of_options = len(options)
    number_of_cells = len(cell_order)

    infinity = float("inf")

    # Best result across all possible first-cell option choices.
    best_total_cost = infinity
    best_option_sequence: list[int] | None = None

    # A cyclic DP cannot initialize all first-cell options together and
    # later close to an unrelated first option. Each first-cell option is
    # therefore evaluated in a separate DP pass.
    for first_option_index, first_option in enumerate(options):

        # dp[position][option_index] stores the minimum accumulated cost
        # through the current position when the current cell uses the
        # specified option.
        dp = [
            [infinity] * number_of_options
            for _ in range(number_of_cells)
        ]

        # parent[position][option_index] stores the previous option index
        # that produced the current minimum cost. It is used to reconstruct
        # the optimal option sequence after the forward recursion.
        parent: list[list[int | None]] = [
            [None] * number_of_options
            for _ in range(number_of_cells)
        ]

        # Only the currently fixed first-cell option is made feasible.
        # No transition cost is charged before the first cell.
        dp[0][first_option_index] = 0.0

        # --------------------------------------------------------------
        # Forward dynamic-programming recursion
        # --------------------------------------------------------------
        for position in range(1, number_of_cells):
            previous_cell = cell_order[position - 1]
            current_cell = cell_order[position]

            # Evaluate every possible option for the current cell.
            for current_index, current_option in enumerate(options):

                # Test every option that could have been selected for
                # the previous cell.
                for previous_index, previous_option in enumerate(options):
                    previous_cost = dp[position - 1][previous_index]

                    # Infinite states were not reachable under the current
                    # fixed first-cell option.
                    if previous_cost == infinity:
                        continue

                    candidate_cost = (
                        previous_cost
                        + transition_cost(
                            previous_cell,
                            previous_option,
                            current_cell,
                            current_option,
                            data,
                            node_for,
                        )
                    )

                    # Retain the least-cost predecessor for this state.
                    if candidate_cost < dp[position][current_index]:
                        dp[position][current_index] = candidate_cost
                        parent[position][current_index] = previous_index

        # --------------------------------------------------------------
        # Cycle closure
        # --------------------------------------------------------------
        last_cell = cell_order[-1]
        first_cell = cell_order[0]

        # Test each possible option on the final cell and add the return
        # transition to the same first option fixed for this DP pass.
        for last_option_index, last_option in enumerate(options):
            path_cost = dp[-1][last_option_index]

            if path_cost == infinity:
                continue

            total_cost = (
                path_cost
                + transition_cost(
                    last_cell,
                    last_option,
                    first_cell,
                    first_option,
                    data,
                    node_for,
                )
            )

            # The option sequence is reconstructed only when the complete
            # closed-cycle cost improves the best result found so far.
            if total_cost >= best_total_cost:
                continue

            # ----------------------------------------------------------
            # DP backtracking
            # ----------------------------------------------------------
            option_indices: list[int | None] = [
                None
            ] * number_of_cells

            option_indices[-1] = last_option_index

            # Follow parent links from the final cell to the first cell.
            for position in range(number_of_cells - 1, 0, -1):
                current_index = option_indices[position]

                if current_index is None:
                    raise RuntimeError(
                        "DP backtracking encountered an invalid state."
                    )

                option_indices[position - 1] = parent[
                    position
                ][current_index]

            # The first option is explicitly fixed for this DP pass.
            option_indices[0] = first_option_index

            if any(index is None for index in option_indices):
                raise RuntimeError(
                    "DP backtracking failed to reconstruct an option sequence."
                )

            best_total_cost = total_cost

            # Convert internal option indices back to the option labels
            # used by the GTSP data dictionary.
            best_option_sequence = [
                options[int(index)]
                for index in option_indices
            ]

    if best_option_sequence is None:
        raise RuntimeError(
            "DP evaluation did not find a feasible option sequence."
        )

    # Store the selected option by cell ID for downstream reporting.
    chosen_options_per_cell = {
        cell: option
        for cell, option in zip(
            cell_order,
            best_option_sequence,
        )
    }

    # Convert the selected cell-option sequence into GTSP node IDs.
    route_nodes = [
        node_for[cell][option]
        for cell, option in zip(
            cell_order,
            best_option_sequence,
        )
    ]

    return (
        best_total_cost,
        chosen_options_per_cell,
        route_nodes,
    )


# ------------------------------------------------------------------
# GA validation and population utilities
# ------------------------------------------------------------------
def validate_ga_config(
    config: GAConfig,
    number_of_cells: int,
) -> None:
    """Validate GA settings before population initialization."""

    if config.population_size < 2:
        raise ValueError("population_size must be at least 2.")

    if config.num_generations < 1:
        raise ValueError("num_generations must be at least 1.")

    if not 0.0 <= config.crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be between 0 and 1.")

    if not 0.0 <= config.mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1.")

    if not 1 <= config.num_elites < config.population_size:
        raise ValueError(
            "num_elites must be at least 1 and smaller than "
            "population_size."
        )

    if not 1 <= config.tournament_size <= config.population_size:
        raise ValueError(
            "tournament_size must be between 1 and population_size."
        )

    if number_of_cells < 2:
        raise ValueError(
            "The hybrid GA-DP solver requires at least two cells."
        )

    if config.progress_interval < 1:
        raise ValueError("progress_interval must be at least 1.")


def initialize_population(
    cells: list[int],
    config: GAConfig,
    random_generator: Random,
) -> list[Individual]:
    """Create the initial population of random cell permutations."""

    population: list[Individual] = []

    for _ in range(config.population_size):
        # Each chromosome must contain every cell exactly once.
        permutation = cells.copy()
        random_generator.shuffle(permutation)

        population.append(
            Individual(permutation=permutation)
        )

    return population


def evaluate_individual(
    individual: Individual,
    data: dict[str, Any],
    node_for: dict[int, dict[int, int]],
) -> None:
    """Evaluate one permutation using the embedded DP option selector."""

    # The GA does not directly encode coverage options. DP determines
    # the optimal option sequence for the individual's cell order.
    (
        cost,
        chosen_options,
        route_nodes,
    ) = evaluate_options_dp(
        individual.permutation,
        data,
        node_for,
    )

    individual.fitness = cost
    individual.chosen_options = chosen_options
    individual.route_nodes = route_nodes


def copy_individual(individual: Individual) -> Individual:
    """Return an independent copy of a GA individual."""

    # Copy mutable fields so later population operations do not alter
    # previously stored elite or best-so-far solutions.
    return Individual(
        permutation=individual.permutation.copy(),
        fitness=individual.fitness,
        chosen_options=individual.chosen_options.copy(),
        route_nodes=individual.route_nodes.copy(),
    )


# ------------------------------------------------------------------
# GA operators
# ------------------------------------------------------------------
def select_parent(
    population: list[Individual],
    config: GAConfig,
    random_generator: Random,
) -> Individual:
    """Select one parent using tournament selection."""

    # Randomly sample competitors and choose the one with the lowest
    # objective value. Lower fitness is better for this minimization case.
    competitors = random_generator.sample(
        population,
        config.tournament_size,
    )

    return min(
        competitors,
        key=lambda individual: individual.fitness,
    )


def order_crossover(
    first_parent: list[int],
    second_parent: list[int],
    random_generator: Random,
) -> tuple[list[int], list[int]]:
    """Apply order crossover to two parent permutations."""

    permutation_size = len(first_parent)

    if permutation_size < 2:
        return first_parent.copy(), second_parent.copy()

    # Two cut points define the segment inherited directly from each parent.
    first_cut, second_cut = sorted(
        random_generator.sample(
            range(permutation_size),
            2,
        )
    )

    first_child: list[int | None] = [None] * permutation_size
    second_child: list[int | None] = [None] * permutation_size

    # Preserve the selected segment in its original position.
    first_child[first_cut : second_cut + 1] = (
        first_parent[first_cut : second_cut + 1]
    )

    second_child[first_cut : second_cut + 1] = (
        second_parent[first_cut : second_cut + 1]
    )

    # Fill the remaining positions with the relative order found in the
    # opposite parent while skipping cells already present in the child.
    first_position = (second_cut + 1) % permutation_size

    for cell in second_parent:
        if cell not in first_child:
            first_child[first_position] = cell
            first_position = (first_position + 1) % permutation_size

    second_position = (second_cut + 1) % permutation_size

    for cell in first_parent:
        if cell not in second_child:
            second_child[second_position] = cell
            second_position = (second_position + 1) % permutation_size

    # At this point each child contains every cell exactly once.
    return (
        [int(cell) for cell in first_child],
        [int(cell) for cell in second_child],
    )


def mutate_permutation(
    permutation: list[int],
    config: GAConfig,
    random_generator: Random,
) -> list[int]:
    """Apply swap mutation and return a new permutation."""

    mutated_permutation = permutation.copy()

    # Mutation is skipped according to the configured probability.
    if (
        len(mutated_permutation) < 2
        or random_generator.random() >= config.mutation_rate
    ):
        return mutated_permutation

    # Swapping two positions preserves permutation feasibility because
    # no cell is added, removed, or duplicated.
    first_position, second_position = random_generator.sample(
        range(len(mutated_permutation)),
        2,
    )

    (
        mutated_permutation[first_position],
        mutated_permutation[second_position],
    ) = (
        mutated_permutation[second_position],
        mutated_permutation[first_position],
    )

    return mutated_permutation


# ------------------------------------------------------------------
# Main hybrid GA-DP engine
# ------------------------------------------------------------------
def run_ga_gtsp(
    data: dict[str, Any],
    config: GAConfig,
) -> GAResult:
    """Run the hybrid GA-DP solver and return the best route found."""

    start_time = perf_counter()

    cells = list(data["cells"])
    validate_ga_config(config, len(cells))

    # A local random generator isolates this run from global random state
    # and provides repeatable results when a seed is supplied.
    random_generator = Random(config.random_seed)

    # Precompute the mapping used repeatedly during DP evaluations.
    node_for = build_node_index_map(data)

    # --------------------------------------------------------------
    # Initial population
    # --------------------------------------------------------------
    population = initialize_population(
        cells,
        config,
        random_generator,
    )

    # Every initial chromosome must be evaluated by DP before selection.
    for individual in population:
        evaluate_individual(
            individual,
            data,
            node_for,
        )

    # Store an independent copy of the best initial solution.
    best_individual = copy_individual(
        min(
            population,
            key=lambda individual: individual.fitness,
        )
    )

    best_cost_history: list[float] = []

    # --------------------------------------------------------------
    # Evolutionary generations
    # --------------------------------------------------------------
    for generation in range(1, config.num_generations + 1):
        new_population: list[Individual] = []

        # Elitism preserves the strongest solutions unchanged and prevents
        # the best known solution from being lost through stochastic operators.
        elites = sorted(
            population,
            key=lambda individual: individual.fitness,
        )[: config.num_elites]

        new_population.extend(
            copy_individual(elite)
            for elite in elites
        )

        # Generate offspring until the next population reaches its target size.
        while len(new_population) < config.population_size:

            # Tournament selection is performed independently for each parent.
            first_parent = select_parent(
                population,
                config,
                random_generator,
            )

            second_parent = select_parent(
                population,
                config,
                random_generator,
            )

            # Apply order crossover according to the crossover probability.
            # Otherwise, carry forward independent copies of the parent routes.
            if random_generator.random() < config.crossover_rate:
                (
                    first_child_permutation,
                    second_child_permutation,
                ) = order_crossover(
                    first_parent.permutation,
                    second_parent.permutation,
                    random_generator,
                )
            else:
                first_child_permutation = (
                    first_parent.permutation.copy()
                )
                second_child_permutation = (
                    second_parent.permutation.copy()
                )

            # Apply swap mutation independently to both offspring.
            child_permutations = (
                mutate_permutation(
                    first_child_permutation,
                    config,
                    random_generator,
                ),
                mutate_permutation(
                    second_child_permutation,
                    config,
                    random_generator,
                ),
            )

            # Each offspring permutation is evaluated by DP before it enters
            # the next generation.
            for child_permutation in child_permutations:
                child = Individual(
                    permutation=child_permutation
                )

                evaluate_individual(
                    child,
                    data,
                    node_for,
                )

                new_population.append(child)

                # The second child may be unnecessary when only one remaining
                # population slot is available.
                if len(new_population) >= config.population_size:
                    break

        # Generational replacement: the completed offspring population becomes
        # the active population for the next iteration.
        population = new_population

        generation_best = min(
            population,
            key=lambda individual: individual.fitness,
        )

        # Track the global best across all generations, not only the current
        # generation's best individual.
        if generation_best.fitness < best_individual.fitness:
            best_individual = copy_individual(generation_best)

        # Record best-so-far cost for convergence analysis and plotting.
        best_cost_history.append(best_individual.fitness)

        should_print = (
            config.print_progress
            and (
                generation == 1
                or generation == config.num_generations
                or generation % config.progress_interval == 0
            )
        )

        if should_print:
            print(
                f"Generation {generation:>4}/"
                f"{config.num_generations}: "
                f"best cost = {best_individual.fitness:.4f}"
            )

    elapsed_time = perf_counter() - start_time

    # Convert the final selected node sequence into explicit directed arcs,
    # including the closing arc from the final node to the first node.
    return GAResult(
        best_permutation=best_individual.permutation.copy(),
        best_cost=best_individual.fitness,
        chosen_options_per_cell=(
            best_individual.chosen_options.copy()
        ),
        route_nodes=best_individual.route_nodes.copy(),
        route_arcs=build_route_arcs(
            best_individual.route_nodes
        ),
        best_cost_history=best_cost_history,
        generations_completed=config.num_generations,
        elapsed_time=elapsed_time,
    )