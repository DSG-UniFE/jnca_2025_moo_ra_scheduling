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
    #font = {'size': 10}
    #plt.rc('font', **font)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    #ax.set_title(f'{alg_name}')
    # increase scatter size
    ax.scatter(front[:,0], front[:,1], front[:,2], alpha=0.8)
    # Set label for x, y, z
    ax.set_xlabel('Avg. Max. Latency (f1)', labelpad=10, fontdict={'fontsize': 12})
    ax.set_ylabel('Deployment Costs (f2)', labelpad=10, fontdict={'fontsize': 12})
    ax.set_zlabel('Avg. Interruption Frequency (f3)', labelpad=8, fontdict={'fontsize': 12})

    # reduce tick font size
    # xlimit 70 175
    # ylimit 150 1200
    # zlimit 0 0.050
    ax.set_xlim(70, 175)
    ax.set_ylim(150, 1200)
    ax.set_zlim(0, 0.050)
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    ax.set_zticklabels(ax.get_zticks(), fontsize=12)
    #ax.tick_params(axis='both', which='major', labelsize=8)
    ax.view_init(elev=20, azim=45) 
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.00)
    #plt.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
    #plt.tight_layout()
    plt.savefig(f"Fig-3d-{alg_name}.pdf", bbox_inches='tight', pad_inches=0.4)
    plt.close()

'''
Using glob find all the files in the directory that contains
VAR and a string given as argument
'''
def find_files(pattern):
    files = glob.glob(f'*FUN*{pattern}')
    return files    

'''
Sparsity calculation according to the paper
 Xu et al., Prediction-guided multi-objective reinforcement 
 learning for continuous robot control.
 Front is the list of solutions and num_objective is the number of objectives
 Sort the lists based on the objectives and calculate the sparsity
'''
def sparsity_calculation(front, num_objective):
    # retrieve the solutions
    objs = [s.objectives for s in front]
    # calculate the sparsity
    sparsity = 0
    for j in range(0, num_objective):
        objs = sorted(objs, key=lambda x: x[j])
        for i in range(0, len(objs) - 1):
            sparsity += (objs[i][j] - objs[i + 1][j]) ** 2
    return sparsity / (len(objs) - 1)

def main():
    src_str = 'MultiCluster'
    files = find_files(src_str)
    objsnsgaii = []
    objsnsgaiii = []
    objsmspso = []
    for filename in files:
        print(filename)
        solutions = read_solutions(filename)
        sparsity = sparsity_calculation(solutions, 3)
        print (f"Sparsity: {sparsity}")
        #print(f'Number of solutions: {len(solutions)}')
        # Getting the objective values
        objective_values = [s.objectives for s in solutions]
        if 'NSGAII.' in filename:
            objsnsgaii = objective_values
        elif 'NSGAIII.' in filename:
            objsnsgaiii = objective_values
        elif 'MSPSO' in filename:
            objsmspso = objective_values
        algname = filename.split('.')[0]
        #xlabel = src_str[0:2].lower()
        #ylabel = src_str[2:4].lower()
        plot_3d_front(objective_values, algname)#, xlabel, ylabel)

        # Put all objs into a numpy array
    objs = np.array(objsnsgaii + objsnsgaiii + objsmspso)
    reference_point = objs.max(axis=0) * 1.1
    reference_point = reference_point.tolist()
    print(f'Len of objsnsgaii: {len(objsnsgaii)} objsnsgaiii: {len(objsnsgaiii)} mspso: {len(objsmspso)}')
    hv = HyperVolume(reference_point)
    hv_nsgaii = hv.compute(np.array(objsnsgaii))
    hv_nsgaiii = hv.compute(np.array(objsnsgaiii))
    hv_mspso = hv.compute(np.array(objsmspso))
    print(f'Reference Point: {reference_point}')
    print(f'NSGAII Hypervolume: {hv_nsgaii}')
    print(f'NSGAIII Hypervolume: {hv_nsgaiii}')
    print(f'MSPSO Hypervolume: {hv_mspso}')

if __name__ == '__main__':
    main()