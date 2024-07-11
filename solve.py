from mc2 import MC2
from mspso import MSPSO

from jmetal.algorithm.multiobjective.nsgaiii import NSGAII, NSGAIII
from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory
from jmetal.operator import IntegerPolynomialMutation, IntegerSBXCrossover
from jmetal.util.comparator import DominanceWithConstraintsComparator
from jmetal.util.evaluator import DaskEvaluator, MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations

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


def logplot_front(front, algorithm):

    print(f'Total non-dominated solutions: {len(front)}')
    print_function_values_to_file(front, f"{algorithm.get_name()}.FUN." + algorithm.label)
    print_variables_to_file(front,f"{algorithm.get_name()}.VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")

    plot_front = Plot(title='Pareto front approximation', axis_labels=['f1', 'f2', 'f3'])
    plot_front.plot(front, label=f"{algorithm.get_name()}-MultiCluster-Problem", filename=f"{algorithm.get_name()}-MultiCluster4", format='png')
    # let's get the front for f1-f2, f1-f3, f2-f3
    front_f1 = [s.objectives[0] for s in front]
    front_f2 = [s.objectives[1] for s in front]
    front_f3 = [s.objectives[2] for s in front]
    from matplotlib import pyplot as plt
    plt.scatter(front_f1, front_f2, color='#236FA4')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f"{algorithm.get_name()}-MCP-f1-f2")
    plt.savefig(f"{algorithm.get_name()}-MCP-f1-f2.png")
    plt.scatter(front_f1, front_f3, color='#236FA4')
    plt.xlabel('f1')
    plt.ylabel('f3')
    plt.title(f"{algorithm.get_name()}-MCP-f1-f3")
    plt.savefig(f"{algorithm.get_name()}-MCP-f1-f3.png")
    plt.scatter(front_f2, front_f3, color='#236FA4')
    plt.xlabel('f2')
    plt.ylabel('f3')
    plt.title(f"{algorithm.get_name()}-MCP-f2-f3")
    plt.savefig(f"{algorithm.get_name()}-MCP-f2-f3.png")


def plot_instance_usage(data, filename: str):
    data_centers = []
    instances = []
    price_types = []
    counts = []
    instance_sizes = []
    
    for (dc, instance, price_type), count in data.items():
        data_centers.append(dc)
        instances.append(instance)
        price_types.append(price_type)
        counts.append(count)
        # Getting dimension (e.g. large, small, 2xlarge)
        instance_size = instance.split('.')[-1]
        instance_sizes.append(instance_size)
    
    plot_data = pd.DataFrame({
        'Data Center': data_centers,
        'Instance Type': instances,
        'Instance Size': instance_sizes,
        'Pricing Type': price_types,
        'Count': counts
    })

    pricing_order = ['On-Demand', 'Reserved', 'Spot']
    colorblind_palette = sns.color_palette("colorblind", len(pricing_order))

    # FacetGrid for Data Center
    g = sns.FacetGrid(plot_data, col='Data Center', col_wrap=3, height=5, sharey=False)
    g.map_dataframe(sns.barplot, x='Instance Size', y='Count', hue='Pricing Type', 
                    hue_order=pricing_order, errorbar=None, palette=colorblind_palette)
    g.add_legend(title='Pricing Type')
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels('Instance Size', 'Count')
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':
    
    freeze_support()

    clusters = ["Cloud", "Fog Tier 2", "Fog Tier 1", "Edge Tier 2", "Edge Tier 1"]

    problem = MC2()
    reference_directions_factory = UniformReferenceDirectionFactory(n_dim=3, n_points=30)

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
    for idx,s in enumerate(front):
        print(f'F1: {s.objectives[0]}, F2: {s.objectives[1]}, F3: {s.objectives[2]}')
        s.number_of_objectives = 3
        plot_instance_usage(problem.calculate_instance_usage(s), f"{algorithm.get_name()}_{idx}.png")

    logplot_front(front, algorithm)

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
    for idx, s in enumerate(front):
        print(f'F1: {s.objectives[0]}, F2: {s.objectives[1]}, F3: {s.objectives[2]}')
        s.number_of_objectives = 3
        plot_instance_usage(problem.calculate_instance_usage(s), f"{algorithm.get_name()}_{idx}.png")
    
    logplot_front(front, algorithm)

    algorithm = MSPSO(
        problem=problem,
        swarm_evaluator=MultiprocessEvaluator(processes=8),
        swarm_size=10,
        termination_criterion=StoppingByEvaluations(max_evaluations=50_000)
    )

    algorithm.run()

    front = get_non_dominated_solutions(algorithm.result())
    for idx, s in enumerate(front):
        print(f'F1: {s.objectives[0]}, F2: {s.objectives[1]}, F3: {s.objectives[2]}')
        s.number_of_objectives = 3
        plot_instance_usage(problem.calculate_instance_usage(s), f"{algorithm.get_name()}_{idx}.png")

    logplot_front(front, algorithm)