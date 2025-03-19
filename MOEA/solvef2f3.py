from moo_ra_f2f3 import MooRaF2F3

from mspso import MSPSO

from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.algorithm.multiobjective.nsgaiii import NSGAII, NSGAIII
from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory
from jmetal.algorithm.multiobjective.random_search import RandomSearch
from jmetal.algorithm.multiobjective.spea2 import SPEA2

from jmetal.operator import IntegerPolynomialMutation, IntegerSBXCrossover
from jmetal.util.comparator import DominanceWithConstraintsComparator
from jmetal.util.evaluator import DaskEvaluator, MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.neighborhood import C9
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
import os
import glob


def logplot_front(front, algorithm, output_dir, problem_name):
    print(f'Total non-dominated solutions: {len(front)}')
    fun_file = os.path.join(output_dir, f"{algorithm.get_name()}.FUN.{algorithm.label}")
    var_file = os.path.join(output_dir, f"{algorithm.get_name()}.VAR.{algorithm.label}")
    print_function_values_to_file(front, fun_file)
    print_variables_to_file(front, var_file)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")

    front_filename = os.path.join(
        output_dir, f"{algorithm.get_name()}-f2-f3"
    )

    plot_front = Plot(title='Pareto front approximation', axis_labels=['f1', 'f2', 'f3'])
    plot_front.plot(front, label=f"{algorithm.get_name()}-f2-f3", filename=f"{front_filename}", format='png')

def plot_instance_usage(data, filename: str):
    data_centers = []
    instance_sizes = []
    price_types = []
    counts = []
    
    for (dc, instance, price_type), values in data.items():
        data_centers.append(dc)
        instance_sizes.append(instance.split('.')[-1])  # Getting dimension (e.g. large, small, 2xlarge)
        price_types.append(price_type)
        counts.append(values['count'])
    
    plot_data = pd.DataFrame({
        'Data Center': data_centers,
        'Instance Size': instance_sizes,
        'Pricing Type': price_types,
        'Count': counts
    })

    # Mapping numeric pricing types to string labels
    pricing_type_labels = {0: 'On-Demand', 1: 'Reserved', 2: 'Spot'}
    plot_data['Pricing Type'] = plot_data['Pricing Type'].map(pricing_type_labels)
    
    pricing_order = ['On-Demand', 'Reserved', 'Spot']
    colorblind_palette = sns.color_palette("colorblind", len(pricing_order))

    # FacetGrid for Data Center
    g = sns.FacetGrid(plot_data, col='Data Center', col_wrap=3, height=5, sharey=False)
    g.map_dataframe(sns.barplot, x='Instance Size', y='Count', hue='Pricing Type', hue_order=pricing_order, errorbar=None, palette=colorblind_palette)
    g.add_legend(title='Pricing Type')
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels('Instance Size', '# Instances')
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    
    freeze_support()

    # Cycle through service files in services/types
    services_files = glob.glob("../services/types/*.csv")

    for service_file in services_files:
        # folder creation for results storage
        base_name = os.path.splitext(os.path.basename(service_file))[0]
        result_dir = os.path.join("results_f2f3", base_name)
        os.makedirs(result_dir, exist_ok=True)

        print("Processing file: ", service_file)

        problem = MooRaF2F3()
        problem_name = problem.name()


        # for NSGA III
        reference_directions_factory = UniformReferenceDirectionFactory(n_dim=2, n_points=30)

        algorithms = []

        algorithms.append(
        SPEA2(
            problem=problem,
            population_size=50,
            offspring_population_size=50,
            mutation=IntegerPolynomialMutation(
                probability=1.0 / problem.number_of_variables(),
                distribution_index=20,
            ),
            crossover=IntegerSBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=50_000),
        )
        )

        algorithms.append(
            RandomSearch(
                problem=problem,
                termination_criterion=StoppingByEvaluations(max_evaluations=50_000),
            )
        )

        algorithms.append(MOCell(
                problem=problem,
                population_size=100,
                neighborhood=C9(10, 10),
                archive=CrowdingDistanceArchive(100),
                mutation=IntegerPolynomialMutation(probability=0.6, distribution_index=30),
                crossover=IntegerSBXCrossover(probability=1.0, distribution_index=20),
                termination_criterion=StoppingByEvaluations(max_evaluations=50_000),
                population_evaluator=MultiprocessEvaluator(processes=8),
            ))
        
        algorithms.append(NSGAII(
            problem=problem,
            population_evaluator=MultiprocessEvaluator(processes=8),
            population_size=150,
            offspring_population_size=80,
            mutation=IntegerPolynomialMutation(probability=0.6, distribution_index=30),
            crossover=IntegerSBXCrossover(probability=0.6, distribution_index=15),
            termination_criterion=StoppingByEvaluations(max_evaluations=50_000),
            dominance_comparator=DominanceWithConstraintsComparator()
        ))

        
        algorithms.append(NSGAIII(
            problem=problem,
            population_evaluator=MultiprocessEvaluator(processes=8),
            reference_directions=reference_directions_factory,
            population_size=150,
            mutation=IntegerPolynomialMutation(probability=0.6, distribution_index=30),
            crossover=IntegerSBXCrossover(probability=0.6, distribution_index=15),
            termination_criterion=StoppingByEvaluations(max_evaluations=50_000),
            dominance_comparator=DominanceWithConstraintsComparator()
        ))

            
        algorithms.append(MSPSO(
            problem=problem,
            swarm_evaluator=MultiprocessEvaluator(processes=8),
            swarm_size=10,
            termination_criterion=StoppingByEvaluations(max_evaluations=50_000)
        ))

        for algorithm in algorithms:
            algorithm.run()
            front = get_non_dominated_solutions(algorithm.result())    
            logplot_front(front, algorithm)

