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

'''
Using glob find all the files in the directory that contains
VAR and a string given as argument
'''
def find_files(pattern):
    files = glob.glob(f'*FUN*{pattern}')
    return files    

def main():
    src_str = 'F1F2'
    files = find_files(src_str)
    for filename in files:
        print(files)
        solutions = read_solutions(filename)
        #print(f'Number of solutions: {len(solutions)}')
        # Getting the objective values
        objective_values = [s.objectives for s in solutions]
        algname = filename.split('.')[0]
        xlabel = src_str[0:2].lower()
        ylabel = src_str[2:4].lower()
        plot_front_hull(objective_values, algname, xlabel, ylabel)

if __name__ == '__main__':
    main()