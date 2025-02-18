from moo_ra_3 import MooRa3

from mspso import MSPSO

import glob
import os

from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.algorithm.multiobjective.nsgaiii import NSGAII, NSGAIII
from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory
from jmetal.operator import IntegerPolynomialMutation, IntegerSBXCrossover
from jmetal.util.comparator import DominanceWithConstraintsComparator
from jmetal.util.evaluator import DaskEvaluator, MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.operator.crossover import IntegerSBXCrossover
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


def logplot_front(front, algorithm, output_dir, problem_name):

    print(f'Total non-dominated solutions: {len(front)}')
    fun_file = os.path.join(output_dir, f"{algorithm.get_name()}.FUN.{algorithm.label}")
    var_file = os.path.join(output_dir, f"{algorithm.get_name()}.VAR.{algorithm.label}")
    print_function_values_to_file(front, fun_file)
    print_variables_to_file(front, var_file)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem_name}")
    print(f"Computing time: {algorithm.total_computing_time}")

    plot_front = Plot(title='Pareto front approximation', axis_labels=['f1', 'f2', 'f3'])
    front_filename = os.path.join(output_dir, f"{algorithm.get_name()}-MultiCluster-Problem")
    plot_front.plot(front, label=f"{algorithm.get_name()}-MultiCluster-Problem", filename=front_filename, format='png')
    # let's get the front for f1-f2, f1-f3, f2-f3
    front_f1 = [s.objectives[0] for s in front]
    front_f2 = [s.objectives[1] for s in front]
    front_f3 = [s.objectives[2] for s in front]
    from matplotlib import pyplot as plt
    plt.scatter(front_f1, front_f2, color='#236FA4')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f"{algorithm.get_name()}-MCP-f1-f2")
    plt.savefig(os.path.join(output_dir, f"{algorithm.get_name()}-MCP-f1-f2.png"))
    plt.close()
    plt.scatter(front_f1, front_f3, color='#236FA4')
    plt.xlabel('f1')
    plt.ylabel('f3')
    plt.title(f"{algorithm.get_name()}-MCP-f1-f3")
    plt.savefig(os.path.join(output_dir, f"{algorithm.get_name()}-MCP-f1-f2.png"))
    plt.close()
    plt.scatter(front_f2, front_f3, color='#236FA4')
    plt.xlabel('f2')
    plt.ylabel('f3')
    plt.title(f"{algorithm.get_name()}-MCP-f2-f3")
    plt.savefig(os.path.join(output_dir, f"{algorithm.get_name()}-MCP-f1-f2.png"))
    plt.close()


def plot_instance_usage(data, filename: str):
    data_centers = []
    instance_sizes = []
    price_types = []
    counts = []
    
    for (dc, instance, price_type), values in data.items():
        data_centers.append(dc)
        instance_sizes.append(instance)  # Getting dimension (e.g. large, small, 2xlarge)
        price_types.append(price_type)
        counts.append(values['count'])
    
    plot_data = pd.DataFrame({
        'Data Center': data_centers,
        'Instance Type': instance_sizes,
        'Pricing': price_types,
        'Count': counts
    })

    # Mapping numeric pricing types to string labels
    pricing_type_labels = {0: 'On-Demand', 1: 'Reserved', 2: 'Spot'}
    plot_data['Pricing'] = plot_data['Pricing'].map(pricing_type_labels)
    
    pricing_order = ['On-Demand', 'Reserved', 'Spot']
    colorblind_palette = sns.color_palette("colorblind", len(pricing_order))

    #print(plot_data)

    # FacetGrid for Data Center
    g = sns.FacetGrid(plot_data, col='Data Center', col_wrap=3, height=5, sharey=False)
    g.map_dataframe(sns.barplot, x='Instance Type', y='Count', hue='Pricing', hue_order=pricing_order, errorbar=None, palette=colorblind_palette)
    g.add_legend(title='Pricing')
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels('Instance Type', '# Instances')
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    
    freeze_support()

    # Cycle through service files in services/types
    services_files = glob.glob('../services/types/*.csv')
    
    for service_file in services_files:

        #folder creation for results storage
        base_name = os.path.splitext(os.path.basename(service_file))[0]
        result_dir = os.path.join('results', base_name)
        os.makedirs(result_dir, exist_ok=True)

        print("Processing file: ", service_file)

        #clusters = ["Cloud", "Fog Tier 2", "Fog Tier 1", "Edge Tier 2", "Edge Tier 1"]
        reference_point = [7000, 300, 0.040]

        problem = MooRa3(service_file)
        problem_name = problem.name()

        algorithm = MOCell(
            problem=problem,
            population_size=100,
            neighborhood=C9(10, 10),
            archive=CrowdingDistanceArchive(100),
            mutation=IntegerPolynomialMutation(probability=0.6, distribution_index=30),
            crossover=IntegerSBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=50_000),
            population_evaluator=MultiprocessEvaluator(processes=8),
        )

        algorithm.run()

        front = get_non_dominated_solutions(algorithm.result())
        # compute expects a numpy array
        objsnsgaii = [s.objectives for s in front]
        hv = HyperVolume(reference_point)
        hv_value = hv.compute(np.array(objsnsgaii))
        print(f'{algorithm.get_name()}: Hypervolume: {hv_value}')

        for idx,s in enumerate(front):
            print(f'F1: {s.objectives[0]}, F2: {s.objectives[1]}, F3: {s.objectives[2]}')
            s.number_of_objectives = 3
            _, _, _, _, data = problem.calculate_costs(s)
            plot_instance_usage(data, os.path.join(result_dir, f"{algorithm.get_name()}_{idx}.png"))

        logplot_front(front, algorithm, result_dir, problem_name)


        algorithm = NSGAII(
            problem=problem,
            population_evaluator=MultiprocessEvaluator(processes=8),
            population_size=150,
            offspring_population_size=80,
            mutation=IntegerPolynomialMutation(probability=0.6, distribution_index=30),
            crossover=IntegerSBXCrossover(probability=0.6, distribution_index=15),
            termination_criterion=StoppingByEvaluations(max_evaluations=50_000),
            dominance_comparator=DominanceWithConstraintsComparator()
        )


        algorithm.run()

        front = get_non_dominated_solutions(algorithm.result())
        # compute expects a numpy array
        objsnsgaii = [s.objectives for s in front]
        hv = HyperVolume(reference_point)
        hv_value = hv.compute(np.array(objsnsgaii))
        print(f'{algorithm.get_name()}: Hypervolume: {hv_value}')

        for idx,s in enumerate(front):
            print(f'F1: {s.objectives[0]}, F2: {s.objectives[1]}, F3: {s.objectives[2]}')
            s.number_of_objectives = 3
            _, _, _, _, data = problem.calculate_costs(s)
            plot_instance_usage(data, os.path.join(result_dir, f"{algorithm.get_name()}_{idx}.png"))

        logplot_front(front, algorithm, result_dir, problem_name)

    
        # for NSGAIII
        reference_directions_factory = UniformReferenceDirectionFactory(n_dim=3, n_points=30)

        algorithm = NSGAIII(
            problem=problem,
            population_evaluator=MultiprocessEvaluator(processes=8),
            reference_directions=reference_directions_factory,
            population_size=150,
            mutation=IntegerPolynomialMutation(probability=0.6, distribution_index=30),
            crossover=IntegerSBXCrossover(probability=0.6, distribution_index=15),
            termination_criterion=StoppingByEvaluations(max_evaluations=50_000),
            dominance_comparator=DominanceWithConstraintsComparator()
        )


        algorithm.run()

        front = get_non_dominated_solutions(algorithm.result())

        # compute expects a numpy array
        objsnsgaiii = [s.objectives for s in front]
        hv = HyperVolume(reference_point)
        hv_value = hv.compute(np.array(objsnsgaiii))
        print(f'{algorithm.get_name()}: Hypervolume: {hv_value}')


        for idx, s in enumerate(front):
            print(f'F1: {s.objectives[0]}, F2: {s.objectives[1]}, F3: {s.objectives[2]}')
            s.number_of_objectives = 3
            _, _, _, _, data = problem.calculate_costs(s)
            plot_instance_usage(data, os.path.join(result_dir, f"{algorithm.get_name()}_{idx}.png"))

        logplot_front(front, algorithm, result_dir, problem_name)

        algorithm = MSPSO(
            problem=problem,
            swarm_evaluator=MultiprocessEvaluator(processes=8),
            swarm_size=10,
            termination_criterion=StoppingByEvaluations(max_evaluations=50_000)
        )

        algorithm.run()
        front = get_non_dominated_solutions(algorithm.result())

        # compute expects a numpy array
        objsmspso = [s.objectives for s in front]
        hv = HyperVolume(reference_point)
        hv_value = hv.compute(np.array(objsmspso))
        print(f'{algorithm.get_name()}: Hypervolume: {hv_value}')


        for idx, s in enumerate(front):
            print(f'F1: {s.objectives[0]}, F2: {s.objectives[1]}, F3: {s.objectives[2]}')
            s.number_of_objectives = 3
            _, _, _, _, data = problem.calculate_costs(s)
            plot_instance_usage(data, os.path.join(result_dir, f"{algorithm.get_name()}_{idx}.png"))

        logplot_front(front, algorithm, result_dir, problem_name)

        # Put all objs into a numpy array
        objs = np.array(objsnsgaii + objsnsgaiii + objsmspso)
        reference_point = objs.max(axis=0) * 1.1
        reference_point = reference_point.tolist()

        hv = HyperVolume(reference_point)
        hv_nsgaii = hv.compute(np.array(objsnsgaii))
        hv_nsgaiii = hv.compute(np.array(objsnsgaiii))
        hv_mspso = hv.compute(np.array(objsmspso))
        print(f'Reference Point: {reference_point}')
        print(f'NSGAII Hypervolume: {hv_nsgaii}')
        print(f'NSGAIII Hypervolume: {hv_nsgaiii}')
        print(f'MSPSO Hypervolume: {hv_mspso}')
