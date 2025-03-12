from moo_ra_so import MooRaSO

from mspso import MSPSO

import glob
import os
import time

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.util.observer import PrintObjectivesObserver

from jmetal.lab.visualization import Plot
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
)

from multiprocessing import freeze_support
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# For HV calculation
from jmetal.core.quality_indicator import HyperVolume
import numpy as np

if __name__ == "__main__":
    freeze_support()

    # Cycle through service files in services/types
    services_files = glob.glob("../services/types/*.csv")

    for service_file in services_files:
        # folder creation for results storage
        base_name = os.path.splitext(os.path.basename(service_file))[0]
        result_dir = os.path.join("results_so", base_name)
        os.makedirs(result_dir, exist_ok=True)

        print("Processing file: ", service_file)

        # clusters = ["Cloud", "Fog Tier 2", "Fog Tier 1", "Edge Tier 2", "Edge Tier 1"]
        reference_point = [7000, 300, 0.040]

        problem = MooRaSO(service_file)
        problem_name = problem.name()

        algorithms = []

        algorithm = GeneticAlgorithm(
            problem=problem,
            population_size=100,
            offspring_population_size=70,
            mutation=IntegerPolynomialMutation(1.0 / problem.number_of_variables()),
            crossover=IntegerSBXCrossover(probability=0.6, distribution_index=15),
            termination_criterion=StoppingByEvaluations(max_evaluations=50_000),
        )

        algorithm.observable.register(observer=PrintObjectivesObserver(5000))

        algorithm.run()
        
        result = algorithm.result()
        print(f"Results: {result}")
        latency = problem.calculate_max_latency(result)
        total_costs, _, _, _, _ = problem.calculate_costs(result)
        qos = problem.calculate_qos(result)

        print(f"Latency: {latency}, Costs: {total_costs}, QoS: {qos}")
        fun_file = os.path.join(result_dir, f"{algorithm.get_name()}.FUN.{algorithm.label}")
        var_file = os.path.join(result_dir, f"{algorithm.get_name()}.VAR.{algorithm.label}")
        ff = open(fun_file, "w")
        ff.write(f"{latency} {total_costs} {qos}")
        ff.close()
        print_variables_to_file(result, var_file)


        print("Algorithm: {}".format(algorithm.get_name()))
        print("Problem: {}".format(problem.name()))
        print("Solution: {}".format(result.variables))
        print("Fitness: {}".format(result.objectives[0]))
        print("Computing time: {}".format(algorithm.total_computing_time))

    # All metrics can be calulated by running the plot_and_compary.py
