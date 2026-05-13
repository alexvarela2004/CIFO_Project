"""
operators/
----------
Genetic operator package for the GA image approximation.

Exports all selection, crossover, and mutation operator classes so
callers can import directly from the package:

    from ga_operators import TournamentSelection, SinglePointCrossover, GaussianMutation
"""

from ga_operators.selection import (
    SelectionOperator,
    TournamentSelection,
    RankSelection,
    RouletteSelection,
)

from ga_operators.crossover import (
    CrossoverOperator,
    SinglePointCrossover,
    UniformCrossover,
    KPointCrossover,
)

from ga_operators.mutation import (
    MutationOperator,
    GaussianMutation,
    ResetMutation,
    SwapMutation,
    CompositeMutation,
    SigmaDecayScheduler,
)

__all__ = [
    # selection
    "SelectionOperator",
    "TournamentSelection",
    "RankSelection",
    "RouletteSelection",
    # crossover
    "CrossoverOperator",
    "SinglePointCrossover",
    "UniformCrossover",
    "KPointCrossover",
    # mutation
    "MutationOperator",
    "GaussianMutation",
    "ResetMutation",
    "SwapMutation",
    "CompositeMutation",
    "SigmaDecayScheduler",
]