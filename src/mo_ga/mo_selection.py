"""
mo_ga/mo_selection.py
---------------------
Selection operator for multi-objective genetic algorithm (NSGA-II).

Provides a single concrete implementation: MOTournamentSelection, which
uses the NSGA-II crowded comparison operator instead of raw fitness to
determine tournament winners.

The crowded comparison operator prefers:
    1. The individual with the lower Pareto rank (closer to the Pareto front).
    2. If ranks are equal, the individual with the higher crowding distance
       (more isolated in objective space — preserves diversity).

This means selection naturally promotes both convergence (towards the
Pareto front) and diversity (spread along the front) simultaneously, which
is the key insight of NSGA-II.

Design notes:
    - MOTournamentSelection is interface-compatible with the original
      TournamentSelection (same select() signature). The MOGA engine uses
      it as a drop-in replacement.
    - Pareto ranks and crowding distances MUST be assigned before calling
      select() (i.e. MOPopulation.assign_pareto_ranks() must have been
      called on the current generation).
    - The operator is stateless between calls.

References
----------
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist
multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary
Computation, 6(2), 182–197. (Section III-B: crowded comparison operator)
"""

from __future__ import annotations

from typing import List

import numpy as np

from mo_ga.mo_individual import MOIndividual


class MOTournamentSelection:
    """
    K-way tournament selection using NSGA-II crowded comparison.

    For each parent slot, randomly samples k individuals and returns the
    winner according to the crowded comparison operator:
        - Lower Pareto rank wins.
        - Ties broken by higher crowding distance.

    Parameters
    ----------
    tournament_size : int
        Number of individuals competing in each tournament. Must be >= 2.
        Default 2 (binary tournament — standard for NSGA-II).
    """

    def __init__(self, tournament_size: int = 2) -> None:
        if tournament_size < 2:
            raise ValueError(
                f"Tournament size must be >= 2, got {tournament_size}."
            )
        self.tournament_size = tournament_size

    def select(
        self,
        population: List[MOIndividual],
        n_parents: int,
        rng: np.random.Generator,
    ) -> List[MOIndividual]:
        """
        Run n_parents independent tournaments and return the winners.

        Parameters
        ----------
        population : list of MOIndividual
            Pool from which participants are drawn with replacement.
            All individuals must have been evaluated and ranked before
            calling this method.
        n_parents : int
            Number of parents to select.
        rng : np.random.Generator
            Caller-supplied random generator.

        Returns
        -------
        list of MOIndividual
            The winner of each tournament, selected by crowded comparison.
        """
        selected = []
        pop_size = len(population)

        for _ in range(n_parents):
            indices = rng.integers(0, pop_size, size=self.tournament_size)
            participants = [population[i] for i in indices]
            # min() uses MOIndividual.__lt__ which is the crowded comparison
            winner = min(participants)
            selected.append(winner)

        return selected

    def __repr__(self) -> str:
        return f"MOTournamentSelection(tournament_size={self.tournament_size})"
