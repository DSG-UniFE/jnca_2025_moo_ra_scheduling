"""Algorithm configurations for the experiments.

Mirrors MOEA/solve.py (same populations, operators and 50,000 evaluations), so
that the new experiments use exactly the configuration reported in Table 5 of
the paper. Only the seed and the number of worker processes are parameters.
"""
import os
import random
import sys

import numpy as np

# jMetalPy 1.7 still uses the removed aliases np.int and np.float (NSGA-III); restore them on NumPy >= 1.24
for _name, _type in (('int', int), ('float', float)):
    if _name not in np.__dict__:
        setattr(np, _name, _type)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOEA_NAME = os.environ.get("MOEA_DIR", "MOEA_2026")  # "MOEA" reproduces the code used in the paper
MOEA_DIR = os.path.join(ROOT, MOEA_NAME)
RESULTS = os.path.join(ROOT, "experiments", "results", MOEA_NAME)  # one results tree per code version
sys.path.insert(0, MOEA_DIR)

from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.algorithm.multiobjective.nsgaiii import NSGAII, NSGAIII, UniformReferenceDirectionFactory
from jmetal.algorithm.multiobjective.random_search import RandomSearch
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.operator import IntegerPolynomialMutation, IntegerSBXCrossover
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.comparator import DominanceWithConstraintsComparator
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.util.neighborhood import C9
from jmetal.util.termination_criterion import StoppingByEvaluations

ALGORITHMS = ["SPEA2", "RandomSearch", "MOCell", "NSGAII", "NSGAIII", "MSPSO"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def build(name, problem, evals=50_000, processes=8):
    """Return a fresh algorithm instance (same settings as MOEA/solve.py)."""
    term = StoppingByEvaluations(max_evaluations=evals)
    if name == "SPEA2":
        return SPEA2(
            problem=problem, population_size=50, offspring_population_size=50,
            mutation=IntegerPolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
            crossover=IntegerSBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=term)
    if name == "RandomSearch":
        return RandomSearch(problem=problem, termination_criterion=term)
    ev = MultiprocessEvaluator(processes=processes) if processes > 1 else None
    if name == "MOCell":
        kw = {"population_evaluator": ev} if ev else {}
        return MOCell(
            problem=problem, population_size=100, neighborhood=C9(10, 10),
            archive=CrowdingDistanceArchive(100),
            mutation=IntegerPolynomialMutation(probability=0.6, distribution_index=30),
            crossover=IntegerSBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=term, **kw)
    if name == "NSGAII":
        kw = {"population_evaluator": ev} if ev else {}
        return NSGAII(
            problem=problem, population_size=150, offspring_population_size=80,
            mutation=IntegerPolynomialMutation(probability=0.6, distribution_index=30),
            crossover=IntegerSBXCrossover(probability=0.6, distribution_index=15),
            termination_criterion=term, dominance_comparator=DominanceWithConstraintsComparator(), **kw)
    if name == "NSGAIII":
        kw = {"population_evaluator": ev} if ev else {}
        return NSGAIII(
            problem=problem,
            reference_directions=UniformReferenceDirectionFactory(n_dim=problem.number_of_objectives(), n_points=30),
            population_size=150,
            mutation=IntegerPolynomialMutation(probability=0.6, distribution_index=30),
            crossover=IntegerSBXCrossover(probability=0.6, distribution_index=15),
            termination_criterion=term, dominance_comparator=DominanceWithConstraintsComparator(), **kw)
    if name == "MSPSO":
        from mspso import MSPSO
        kw = {"swarm_evaluator": ev} if ev else {}
        return MSPSO(problem=problem, swarm_size=10, termination_criterion=term, **kw)
    raise ValueError(name)
