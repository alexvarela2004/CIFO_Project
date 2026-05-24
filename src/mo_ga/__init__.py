"""
mo_ga/
------
Multi-objective genetic algorithm (NSGA-II) extension for the triangle
image approximation project.

This package is a self-contained extension that adds multi-objective
optimisation capability without modifying any existing source files.
All existing crossover and mutation operators are reused unchanged.

Contents
--------
mo_individual.py   — MOIndividual: Individual with vector fitness + Pareto rank
mo_population.py   — MOPopulation: Population with NSGA-II ranking + crowding
mo_selection.py    — MOTournamentSelection: crowded comparison tournament
mo_ga.py           — MOGA: NSGA-II engine
runner_mo.py       — runner for multi-objective experiments

Usage
-----
From src/:
    python -m mo_ga.runner_mo --target data/girl_pearl.png

References
----------
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist
multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary
Computation, 6(2), 182-197.
"""

from mo_ga.mo_individual import MOIndividual
from mo_ga.mo_population import MOPopulation
from mo_ga.mo_selection  import MOTournamentSelection
from mo_ga.mo_ga         import MOGA

__all__ = ["MOIndividual", "MOPopulation", "MOTournamentSelection", "MOGA"]
