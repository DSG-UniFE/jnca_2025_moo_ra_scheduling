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
Plot the Pareto front starting from the list of solutions
We have 3 objectives in this case
'''
def plot_3d_front(front, alg_name):
    # Convert front to a numpy array
    front = np.array(front)
    # Instantiate fig 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(f'{alg_name}')
    ax.scatter(front[:,0], front[:,1], front[:,2])
    # Set label for x, y, z
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    # reduce tick font size
    # xlimit 70 175
    # ylimit 150 1200
    # zlimit 0 0.050
    ax.set_xlim(70, 175)
    ax.set_ylim(150, 1200)
    ax.set_zlim(0, 0.050)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.view_init(elev=20, azim=45)
    plt.savefig(f"Fig-3d-{alg_name}.pdf")
    plt.close()

'''
Using glob find all the files in the directory that contains
VAR and a string given as argument
'''
def find_files(pattern):
    files = glob.glob(f'*FUN*{pattern}')
    return files    

def main():
    src_str = 'MultiCluster'
    files = find_files(src_str)
    for filename in files:
        print(files)
        solutions = read_solutions(filename)
        #print(f'Number of solutions: {len(solutions)}')
        # Getting the objective values
        objective_values = [s.objectives for s in solutions]
        algname = filename.split('.')[0]
        #xlabel = src_str[0:2].lower()
        #ylabel = src_str[2:4].lower()
        plot_3d_front(objective_values, algname)#, xlabel, ylabel)

if __name__ == '__main__':
    main()