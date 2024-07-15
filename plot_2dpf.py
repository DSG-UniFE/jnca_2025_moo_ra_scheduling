from scheduling_ra_f2f3 import SchedulingRAF2F3
from scheduling_ra_f1f2 import SchedulingRAF1F2

from jmetal.algorithm.multiobjective.nsgaiii import NSGAII, NSGAIII
from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory
from jmetal.operator import IntegerPolynomialMutation, IntegerSBXCrossover
from jmetal.util.comparator import DominanceWithConstraintsComparator
from jmetal.util.evaluator import DaskEvaluator, MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions
)

from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# For HV calculation
from jmetal.core.quality_indicator import HyperVolume
import numpy as np

import argparse
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
import glob

# To calculated the non-dominated solutions from
# the merged list

from jmetal.util.solution import get_non_dominated_solutions

'''
Parse the arguments from command line.
Arguments are 
1)--filename, which contains the list of solutions to be plotted 
(This file is the one generated with the JMetal print_function_values_to_file function)
2) --problem, which is the problem to be solved (f1f2, f2f3)
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Plotting the Pareto front')
    parser.add_argument('--filename', type=str, help='File containing the list of solutions to be plotted')
    parser.add_argument('--problem', type=str, help='Problem to be solved (f1f2, f2f3)')
    return parser.parse_args()

'''
Plot the Pareto front starting from the list of solutions
Visualize also the convex hull of the front
'''
def plot_front_hull(front, alg_name, x_label, y_label):
    # Convert front to a numpy array
    front = np.array(front)
    # get x_label and y_label
    plt.title(f'{alg_name}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # for f2-f3
    #plt.xlim(200, 1000)
    #plt.ylim(0, 0.045)
    plt.ylim(100, 1100)
    plt.xlim(70, 170)
    plt.grid(True)
    plt.scatter(front[:,0], front[:,1], color='#236FA4')
    # Create the Convex hull using Scipy
    hull = ConvexHull(front)
    for simplex in hull.simplices:
        plt.plot(front[simplex, 0], front[simplex, 1], 'r--')
    #plt.show()
    plt.savefig(f"Fig-{alg_name}-{x_label}-{y_label}.pdf")
    plt.close()

def plot_2fronts_hull(objs, alg_name, x_label, y_label, fdescr):
    # x-label and y-label formatting
    if x_label == 'f1':
        x_label = 'Avg. Max. Latency (f1)'
    if x_label == 'f2':
        x_label = 'Deployment Costs (f2)'
    elif x_label == 'f3':
        x_label = 'Avg. Interruption Frequency (f3)'
    if y_label == 'f1':
        y_label = 'Avg. Max. Latency (f1)'
    if y_label == 'f2':
        y_label = 'Deployment Costs (f2)'
    elif y_label == 'f3':
        y_label = 'Avg. Interruption Frequency (f3)'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # for f2-f3
    #plt.xlim(200, 1000)
    #plt.ylim(0, 0.045)
    #plt.ylim(100, 1100)
    #plt.xlim(70, 170)
    plt.grid(True)
    colors = ['#236FA4', '#F46036', '#D0D0D0']
    markers = ['o', 's', '^']
    legend = []
    for idx, os in enumerate(objs):
        if len(os) != 0:
            front = np.array(os)
            plt.scatter(front[:,0], front[:,1], color=colors[idx], marker=markers[idx], alpha=0.8)
            legend.append(alg_name[idx])
    
    # Create the Convex hull using Scipy
    #hull = ConvexHull(front)
    #for simplex in hull.simplices:
    #    plt.plot(front[simplex, 0], front[simplex, 1], 'r--')
    plt.legend(legend)
    #plt.show()
    plt.savefig(f"Fig-{fdescr}.pdf")
    plt.close()

'''
Using glob find all the files in the directory that contains
VAR and a string given as argument
'''

def find_files(pattern):
    files = glob.glob(pattern)
    return files    


def main():
    src_str = 'F1F2'
    objs_values = []
    algs = []
    solutions = []
    files = find_files(f'NSGAII.*FUN*{src_str}')
    files.extend(find_files(f'NSGAIII.*FUN*{src_str}'))
    files.extend(find_files(f'MSPSO.*FUN*{src_str}'))
    print(files)
    xlabel = 'f1'
    ylabel = 'f2'
    for filename in files:
        print(filename)
        solution = read_solutions(filename)
        solutions.extend(solution)
        print(f'Number of solutions: {len(solution)}, {type(solution)}')
        # Getting the objective values
        objs = [s.objectives for s in solution]
        objs_values.append(objs)
        algname = filename.split('.')[0]
        algs.append(algname)
    plot_2fronts_hull(objs_values, algs, xlabel, ylabel, 'f1f2-ensemble')

    #plot_2fronts_hull(objs_values, algs, xlabel, ylabel)
    # Then merge the three fronts and get the non-dominated solutions
    snd = get_non_dominated_solutions(solutions)
    print("Number of non-dominated solutions after merge: ", len(snd))

    # Now we can plot the merged front
    nsgaii_front = []
    nsgaiii_front = []
    mspso_front = []
    # get the objective values of the ensemble of non-dominated solutions
    ns = [s.objectives for s in snd]
    for idx, f in enumerate(objs_values):
        for o in f:
            if o in ns:
                if idx == 0:
                    nsgaii_front.append(o)
                elif idx == 1:
                    nsgaiii_front.append(o)
                elif idx == 2:
                    mspso_front.append(o)
    objs_values = [nsgaii_front, nsgaiii_front, mspso_front]
    print(objs_values)
    plot_2fronts_hull(objs_values, algs, xlabel, ylabel, 'f1f2-ensemble-nd')

    src_str = 'F2F3'
    objs_values = []
    algs = []
    solutions = []
    files = find_files(f'NSGAII.*FUN*{src_str}')
    files.extend(find_files(f'NSGAIII.*FUN*{src_str}'))
    files.extend(find_files(f'MSPSO.*FUN*{src_str}'))
    print(files)
    xlabel = 'f2'
    ylabel = 'f3'
    for filename in files:
        print(filename)
        solution = read_solutions(filename)
        solutions.extend(solution)
        print(f'Number of solutions: {len(solution)}, {type(solution)}')
        # Getting the objective values
        objs = [s.objectives for s in solution]
        objs_values.append(objs)
        algname = filename.split('.')[0]
        algs.append(algname)
    plot_2fronts_hull(objs_values, algs, xlabel, ylabel, 'f2f3-ensemble')

    #plot_2fronts_hull(objs_values, algs, xlabel, ylabel)
    # Then merge the three fronts and get the non-dominated solutions
    snd = get_non_dominated_solutions(solutions)
    print("Number of non-dominated solutions after merge: ", len(snd))

    # Now we can plot the merged front
    nsgaii_front = []
    nsgaiii_front = []
    mspso_front = []
    # get the objective values of the ensemble of non-dominated solutions
    ns = [s.objectives for s in snd]
    for idx, f in enumerate(objs_values):
        for o in f:
            if o in ns:
                if idx == 0:
                    nsgaii_front.append(o)
                elif idx == 1:
                    nsgaiii_front.append(o)
                elif idx == 2:
                    mspso_front.append(o)
    objs_values = [nsgaii_front, nsgaiii_front, mspso_front]
    print(objs_values)
    plot_2fronts_hull(objs_values, algs, xlabel, ylabel, 'f2f3-ensemble-nd')


if __name__ == '__main__':
    main()