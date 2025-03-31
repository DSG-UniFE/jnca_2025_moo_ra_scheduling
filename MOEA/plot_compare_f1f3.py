from jmetal.core.quality_indicator import InvertedGenerationalDistance

from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
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
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import glob
import os
import csv
import ast

"""
Retrieve the objectives from the ILP csv file
"""


def retrieve_objectives(filename, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(filename)
    else:
        os.makedirs(output_dir, exist_ok=True)

    with open(filename, newline="") as csvfile:
        print(f"Reading file {filename}")
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Retrieve the gap
            gap = row["MIP_Gap"]

            # Convert the string representation of the lists to actual lists
            try:
                latency = ast.literal_eval(row["Latency"])
                unavailability = ast.literal_eval(row["Unavailability"])
            except Exception as e:
                print(f"Parsing error for gap {gap}: {e}")
                continue

            # Create the output file
            output_filename = os.path.join(output_dir, f"objectives_gap_{gap}.txt")

            # Put the objectives in the file
            with open(output_filename, "w") as outfile:
                for lat, unav in zip(latency, unavailability):
                    outfile.write(f"{lat} {unav}\n")


"""
Plot the Pareto front starting from the list of solutions
We have 3 objectives in this case
"""


def plot_2d_front(front, alg_name, output_dir):
    # Convert front to a numpy array
    front = np.array(front)
    print(f"Alg name: {alg_name}")
    # Instantiate fig
    # font = {'size': 10}
    # plt.rc('font', **font)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    # ax.set_title(f'{alg_name}')
    # increase scatter size
    ax.scatter(front[:, 0], front[:, 1], alpha=0.8)
    # Set the camera view angle to 30, 60
    # Set label for x, y, z
    ax.set_xlabel("Avg. Max. Latency (f1)", labelpad=10, fontdict={"fontsize": 12})
    ax.set_ylabel("Avg. Interruption Frequency (f3)", labelpad=10, fontdict={"fontsize": 12})
    #ax.set_xlim(70, 175)
    #ax.set_ylim(150, 1200)
    #ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    # ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.00)
    # plt.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
    # plt.tight_layout()
    save_path = os.path.join(output_dir, f"Fig-2d-{alg_name}.pdf")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.4)
    plt.close()


"""
Using glob find all the files in the directory that contains
VAR and a string given as argument
"""

def find_files(pattern):
    files = glob.glob(f"*FUN*{pattern}")
    return files


def sparsity_calculation(front, num_objective):
    """
    Sparsity calculation according to the paper
    Xu et al., Prediction-guided multi-objective reinforcement
    learning for continuous robot control.
    Front is the list of solutions and num_objective is the number of objectives
    Sort the lists based on the objectives and calculate the sparsity
    """
    # retrieve the solutions
    objs = [s.objectives for s in front]
    if len(objs) <= 1:
        return 0
    # calculate the sparsity
    sparsity = 0
    for j in range(0, num_objective):
        objs = sorted(objs, key=lambda x: x[j])
        for i in range(0, len(objs) - 1):
            sparsity += (objs[i][j] - objs[i + 1][j]) ** 2
    return sparsity / (len(objs) - 1)


def dominance_based_gap(meta_heuristic_solutions, ilp_solutions):
    """
    Calculate the dominance-based approximation gap for each metaheuristic solution.

    Parameters:
        meta_heuristic_solutions (numpy array): Array of shape (f, m) where f is the number of metaheuristic solutions and m is the number of objectives.
        ilp_solutions (numpy array): Array of shape (n, m) representing ILP solutions.
        f can be greater than n.

    Returns:
        gaps (numpy array): Dominance-based approximation gaps.
    """
    if len(ilp_solutions) == 0:
        return None
    
    # Initialize an array to store the minimum gap for each metaheuristic solution
    gaps = []

    # Iterate over each metaheuristic solution
    for meta_solution in meta_heuristic_solutions:
        # Calculate the distance from the metaheuristic solution to each ILP solution
        # Here, we use a simple distance metric (e.g., Euclidean distance) to find the closest ILP solution
        distances = cdist([meta_solution], ilp_solutions, metric="euclidean")
        min_distance_idx = np.argmin(distances)

        # Calculate the gap to the closest ILP solution
        closest_ilp_solution = ilp_solutions[min_distance_idx]
        #print(f"Closest ILP solution: {closest_ilp_solution}, type: {type(closest_ilp_solution)}")
        #print(f"Metaheuristic solution: {meta_solution}, type: {type(meta_solution)}")
        # for all objective values get the best value
        gap = []
        for i in range(len(closest_ilp_solution)):
            best_value = np.min([closest_ilp_solution[i], meta_solution[i]])
            gap.append(float((meta_solution[i] - closest_ilp_solution[i]) / best_value * 100))
        #print(f"Gap: {gap}")
        gaps.append(gap)
    try:
        min_gap_index = np.argmin(list(map(sum, gaps)), axis=0)
        min_gap = gaps[min_gap_index]
        #print(f"Min gap: {min_gap}")
    except ValueError as e:
        print(f"Error: {e}")
        min_gap = None

    return min_gap


def select_best_reference(ilp_solutions):
    """
    Select the best reference in the ILP solutions based on the ideal point.
    """
    # Assuming minimization problems
    ideal = np.min(ilp_solutions, axis=0)
    distances = np.linalg.norm(ilp_solutions - ideal, axis=1)
    best_index = np.argmin(distances)
    return ilp_solutions[best_index]


def main():
    # Glob for each direcory within results
    dir_usecases_meta = glob.glob("results_f1f3/*/")
    dir_usecases_ilp = glob.glob("../results/usecase/*/")

    # Retrieve the objectives from the ILP csv files
    for usecase in dir_usecases_ilp:
        output_dir = os.path.join(usecase, "objectives_f1_f3")
        os.makedirs(output_dir, exist_ok=True)
        objectives_files_ilp = glob.glob(f"{usecase}f1_f3/results*.csv")
        for objective_file in objectives_files_ilp:
            retrieve_objectives(objective_file, output_dir)

    ##### Metaheuristics metrics computation #####

    # src_str = 'MultiClusterGPU'
    # files = find_files(src_str)

    for usecase in dir_usecases_meta:
        output_dir_sparsities = os.path.dirname("../sparsities_f1f3/")
        usecase_name = usecase.split("/")[1]
        os.makedirs(output_dir_sparsities, exist_ok=True)
        output_dir_hv = os.path.dirname("../hypervolumes_f1f3/")
        os.makedirs(output_dir_hv, exist_ok=True)
        output_dir_igds = os.path.dirname("../igds_f1f3/")
        os.makedirs(output_dir_igds, exist_ok=True)
        # Create directory for dominance approximation gaps
        output_dir_dominance_gaps = os.path.dirname("../dominance_gaps_f1f3/")
        os.makedirs(output_dir_dominance_gaps, exist_ok=True)
        output_file_sparsity = os.path.join(
            output_dir_sparsities, f"sparsity_{usecase_name}.txt"
        )
        output_file_hv = os.path.join(output_dir_hv, f"hypervolume_{usecase_name}.txt")
        output_file_igd = os.path.join(output_dir_igds, f"igd_{usecase_name}.txt")

        output_file_dominance_gap = os.path.join(
            output_dir_dominance_gaps, f"dominance_gaps_{usecase_name}.txt"
        )

        f_sparsity = open(output_file_sparsity, "w")
        usecase = usecase.split("/")[1]
        print(f"Usecase: {usecase}")
        objectives_files_meta = glob.glob(f"results_f1f3/{usecase}/*.FUN.*")
        objectives_files_ilp = glob.glob(
            f"../results/usecase/{usecase}/objectives_f1_f3/*.txt"
        )
        # print(objectives_files_meta)
        # print(objectives_files_ilp)

        objsilpgap00 = []
        objsilpgap005 = []
        objsilpgap01 = []
        objsilpgap025 = []
        objsilpgap050 = []
        objsilpgap075 = []
        objsmocell = []
        objsnsgaii = []
        objsnsgaiii = []
        objsmspso = []
        objsspea2 = []
        objsrandomsearch = []
        objsga = []

        for filename in objectives_files_meta:
            # if '.png' in filename:
            #    continue
            print(filename)

            solutions = read_solutions(filename)
            sparsity = sparsity_calculation(solutions, 2)
            # print(f'Number of solutions: {len(solutions)}')
            # Getting the objective values
            objective_values = [s.objectives for s in solutions]
            # base_name = os.path.splitext(os.path.basename(filename))[0]

            if "MOCell." in filename:
                objsmocell = objective_values
                algname = "MOCell"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")
            elif "NSGAII." in filename:
                objsnsgaii = objective_values
                algname = "NSGAII"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")
            elif "NSGAIII." in filename:
                objsnsgaiii = objective_values
                algname = "NSGAIII"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")
            elif "MSPSO" in filename:
                objsmspso = objective_values
                algname = "MSPSO"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")
            elif "SPEA2" in filename:
                objsspea2 = objective_values
                algname = "SPEA2"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")
            elif "Random Search" in filename:
                objsrandomsearch = objective_values
                algname = "Random Search"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")

            # algname = filename.split('.')[0]
            # xlabel = src_str[0:2].lower()
            # ylabel = src_str[2:4].lower()
            plot_2d_front(objective_values, algname, output_dir)  # , xlabel, ylabel)

        # Add sparsity calculation related to ILP for each gap
        f_sparsity.write(f"\n\n********** ILP **********\n\n")
        for filename in objectives_files_ilp:
            gap = filename.split("_")[-1].split(".t")[0]
            solutions = read_solutions(filename)
            print(f"Filename: {filename} Gap: {gap}: Solutions: {len(solutions)}")
            sparsity = sparsity_calculation(solutions, 2)
            f_sparsity.write(f"ILP Gap {gap} Sparsity: {sparsity}\n")

            if gap == "0.0":
                objsilpgap00 = [s.objectives for s in solutions]
            elif gap == "0.05":
                objsilpgap005 = [s.objectives for s in solutions]
            elif gap == "0.1":
                objsilpgap01 = [s.objectives for s in solutions]
            elif gap == "0.25":
                objsilpgap025 = [s.objectives for s in solutions]
            elif gap == "0.5":
                objsilpgap050 = [s.objectives for s in solutions]
            elif gap == "0.75":
                objsilpgap075 = [s.objectives for s in solutions]

        # Print the dimension of all objs
        print(f"MOCell: {np.shape(objsmocell)} solutions")
        print(f"NSGAII: {np.shape(objsnsgaii)} solutions")
        print(f"NSGAIII: {np.shape(objsnsgaiii)} solutions")
        print(f"MSPSO: {np.shape(objsmspso)} solutions")
        print(f"SPEA2: {np.shape(objsspea2)} solutions")
        print(f"Random Search: {np.shape(objsrandomsearch)} solutions")
        print(f"ILP Gap 0.0: {np.shape(objsilpgap00)} solutions")
        print(f"ILP Gap 0.05: {np.shape(objsilpgap005)} solutions")
        print(f"ILP Gap 0.1: {np.shape(objsilpgap01)} solutions")
        print(f"ILP Gap 0.25: {np.shape(objsilpgap025)} solutions")
        print(f"ILP Gap 0.5: {np.shape(objsilpgap050)} solutions")
        print(f"ILP Gap 0.75: {np.shape(objsilpgap075)} solutions")
        # Put all objs into a numpy array
        objs = np.array(
            objsilpgap00
            + objsilpgap005
            + objsilpgap01
            + objsilpgap025
            + objsilpgap050
            + objsilpgap075
            + objsmocell
            + objsnsgaii
            + objsnsgaiii
            + objsmspso
            + objsspea2
        )

        reference_point = objs.max(axis=0) * 1.1
        reference_point = reference_point.tolist()
        hv = HyperVolume(reference_point)
        hv_mocell = hv.compute(np.array(objsmocell))
        hv_nsgaii = hv.compute(np.array(objsnsgaii))
        hv_nsgaiii = hv.compute(np.array(objsnsgaiii))
        hv_mspso = hv.compute(np.array(objsmspso))
        hv_spea2 = hv.compute(np.array(objsspea2))
        hv_randomsearch = hv.compute(np.array(objsrandomsearch))
        hv_moilp_gap00 = hv.compute(np.array(objsilpgap00))
        hv_moilp_gap005 = hv.compute(np.array(objsilpgap005))
        hv_moilp_gap01 = hv.compute(np.array(objsilpgap01))
        hv_moilp_gap025 = hv.compute(np.array(objsilpgap025))

        igd = InvertedGenerationalDistance(objs)
        igd_mocell = igd.compute(np.array(objsmocell))
        igd_nsgaii = igd.compute(np.array(objsnsgaii))
        igd_nsgaiii = igd.compute(np.array(objsnsgaiii))
        igd_mspso = igd.compute(np.array(objsmspso))
        igd_spea2 = igd.compute(np.array(objsspea2))
        igs_randomsearch = igd.compute(np.array(objsrandomsearch))
        ilp_results_present = True
        try:
            igd_moilp_gap00 = igd.compute(np.array(objsilpgap00))
            igd_moilp_gap005 = igd.compute(np.array(objsilpgap005))
            igd_moilp_gap01 = igd.compute(np.array(objsilpgap01))
            igd_moilp_gap025 = igd.compute(np.array(objsilpgap025))
        except Exception as e:
            print(f"WARNING -- Files are not there!")
            ilp_results_present = False

        with open(output_file_hv, "w") as f:
            # f.write(f"Reference Point: {reference_point}\n")
            f.write(f"MOCell Hypervolume: {hv_mocell}\n")
            f.write(f"NSGAII Hypervolume: {hv_nsgaii}\n")
            f.write(f"NSGAIII Hypervolume: {hv_nsgaiii}\n")
            f.write(f"MSPSO Hypervolume: {hv_mspso}\n")
            f.write(f"SPEA2 Hypervolume: {hv_spea2}\n")
            f.write(f"Random Search Hypervolume: {hv_randomsearch}\n")
            if ilp_results_present:
                f.write(f"\n\n********** ILP **********\n\n")
                f.write(f"ILP Gap 0.00 Hypervolume: {hv_moilp_gap00}\n")
                f.write(f"ILP Gap 0.05 Hypervolume: {hv_moilp_gap005}\n")
                f.write(f"ILP Gap 0.1 Hypervolume: {hv_moilp_gap01}\n")
                f.write(f"ILP Gap 0.25 Hypervolume: {hv_moilp_gap025}\n")

        with open(output_file_igd, "w") as f:
            f.write(f"MOCell IGD: {igd_mocell}\n")
            f.write(f"NSGAII IGD: {igd_nsgaii}\n")
            f.write(f"NSGAIII IGD: {igd_nsgaiii}\n")
            f.write(f"MSPSO IGD: {igd_mspso}\n")
            f.write(f"SPEA2 IGD: {igd_spea2}\n")
            f.write(f"Random Search IGD: {igs_randomsearch}\n")
            if ilp_results_present:
                f.write(f"\n\n********** ILP **********\n\n")
                f.write(f"ILP Gap 0.0 IGD: {igd_moilp_gap00}\n")
                f.write(f"ILP Gap 0.05 IGD: {igd_moilp_gap005}\n")
                f.write(f"ILP Gap 0.1 IGD: {igd_moilp_gap01}\n")
                f.write(f"ILP Gap 0.25 IGD: {igd_moilp_gap025}\n")

        with open(output_file_dominance_gap, "w") as f:
            if True:
                f.write(
                    f"MOCell Dominance Gap 0.00: {dominance_based_gap(np.array(objsmocell), objsilpgap00)}\n"
                    f"MOCell Dominance Gap 0.05: {dominance_based_gap(np.array(objsmocell), objsilpgap005)}\n"
                    f"MOCell Dominance Gap 0.1: {dominance_based_gap(np.array(objsmocell), objsilpgap01)}\n"
                    f"MOCell Dominance Gap 0.25: {dominance_based_gap(np.array(objsmocell), objsilpgap025)}\n"
                    f"MOCell Dominance Gap 0.5: {dominance_based_gap(np.array(objsmocell), objsilpgap050)}\n"
                    f"MOCell Dominance Gap 0.75: {dominance_based_gap(np.array(objsmocell), objsilpgap075)}\n"
                )
                f.write(
                    f"NSGAII Dominance Gap 0.00: {dominance_based_gap(np.array(objsnsgaii), objsilpgap00)}\n"
                    f"NSGAII Dominance Gap 0.05: {dominance_based_gap(np.array(objsnsgaii), objsilpgap005)}\n"
                    f"NSGAII Dominance Gap 0.1: {dominance_based_gap(np.array(objsnsgaii), objsilpgap01)}\n"
                    f"NSGAII Dominance Gap 0.25: {dominance_based_gap(np.array(objsnsgaii), objsilpgap025)}\n"
                    f"NSGAII Dominance Gap 0.5: {dominance_based_gap(np.array(objsnsgaii), objsilpgap050)}\n"
                    f"NSGAII Dominance Gap 0.75: {dominance_based_gap(np.array(objsnsgaii), objsilpgap075)}\n"
                )
                f.write(
                    f"NSGAIII Dominance Gap 0.00: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap00)}\n"
                    f"NSGAIII Dominance Gap 0.05: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap005)}\n"
                    f"NSGAIII Dominance Gap 0.1: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap01)}\n"
                    f"NSGAIII Dominance Gap 0.25: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap025)}\n"
                    f"NSGAIII Dominance Gap 0.5: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap050)}\n"
                    f"NSGAIII Dominance Gap 0.75: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap075)}\n"
                )
                f.write(
                    f"MSPSO Dominance Gap 0.00: {dominance_based_gap(np.array(objsmspso), objsilpgap00)}\n"
                    f"MSPSO Dominance Gap 0.05: {dominance_based_gap(np.array(objsmspso), objsilpgap005)}\n"
                    f"MSPSO Dominance Gap 0.1: {dominance_based_gap(np.array(objsmspso), objsilpgap01)}\n"
                    f"MSPSO Dominance Gap 0.25: {dominance_based_gap(np.array(objsmspso), objsilpgap025)}\n"
                    f"MSPSO Dominance Gap 0.5: {dominance_based_gap(np.array(objsmspso), objsilpgap050)}\n"
                    f"MSPSO Dominance Gap 0.75: {dominance_based_gap(np.array(objsmspso), objsilpgap075)}\n"
                )
                f.write(
                    f"SPEA2 Dominance Gap 0.00: {dominance_based_gap(np.array(objsspea2), objsilpgap00)}\n"
                    f"SPEA2 Dominance Gap 0.05: {dominance_based_gap(np.array(objsspea2), objsilpgap005)}\n"
                    f"SPEA2 Dominance Gap 0.1: {dominance_based_gap(np.array(objsspea2), objsilpgap01)}\n"
                    f"SPEA2 Dominance Gap 0.25: {dominance_based_gap(np.array(objsspea2), objsilpgap025)}\n"
                    f"SPEA2 Dominance Gap 0.5: {dominance_based_gap(np.array(objsspea2), objsilpgap050)}\n"
                    f"SPEA2 Dominance Gap 0.75: {dominance_based_gap(np.array(objsspea2), objsilpgap075)}\n"
                )
                f.write(
                    f"Random Search Dominance Gap 0.00: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap00)}\n"
                    f"Random Search Dominance Gap 0.05: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap005)}\n"
                    f"Random Search Dominance Gap 0.1: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap01)}\n"
                    f"Random Search Dominance Gap 0.25: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap025)}\n"
                    f"Random Search Dominance Gap 0.5: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap050)}\n"
                    f"Random Search Dominance Gap 0.75: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap075)}\n"
                )
                f.write(
                    f"GA Dominance Gap 0.00: {dominance_based_gap(np.array(objsga), objsilpgap00)}\n"
                    f"GA Dominance Gap 0.05: {dominance_based_gap(np.array(objsga), objsilpgap005)}\n"
                    f"GA Dominance Gap 0.1: {dominance_based_gap(np.array(objsga), objsilpgap01)}\n"
                    f"GA Dominance Gap 0.25: {dominance_based_gap(np.array(objsga), objsilpgap025)}\n"
                    f"GA Dominance Gap 0.5: {dominance_based_gap(np.array(objsga), objsilpgap050)}\n"
                    f"GA Dominance Gap 0.75: {dominance_based_gap(np.array(objsga), objsilpgap075)}\n"
                )


if __name__ == "__main__":
    main()
