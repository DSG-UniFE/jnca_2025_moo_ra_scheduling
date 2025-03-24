from jmetal.algorithm.multiobjective.nsgaiii import NSGAII, NSGAIII
from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory
from jmetal.operator import IntegerPolynomialMutation, IntegerSBXCrossover
from jmetal.util.comparator import DominanceWithConstraintsComparator
from jmetal.util.evaluator import DaskEvaluator, MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations
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
                cost = ast.literal_eval(row["Cost"])
                unavailability = ast.literal_eval(row["Unavailability"])
            except Exception as e:
                print(f"Parsing error for gap {gap}: {e}")
                continue

            # Create the output file
            output_filename = os.path.join(output_dir, f"objectives_gap_{gap}.txt")

            # Put the objectives in the file
            with open(output_filename, "w") as outfile:
                for lat, c, unav in zip(latency, cost, unavailability):
                    outfile.write(f"{lat} {c} {unav}\n")


"""
Plot the Pareto front starting from the list of solutions
We have 3 objectives in this case
"""


def plot_3d_front(front, alg_name, output_dir):
    # Convert front to a numpy array
    front = np.array(front)
    print(f"Alg name: {alg_name}")
    # Instantiate fig
    # font = {'size': 10}
    # plt.rc('font', **font)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection="3d")
    # ax.set_title(f'{alg_name}')
    # increase scatter size
    ax.scatter(front[:, 0], front[:, 1], front[:, 2], alpha=0.8)
    # Set the camera view angle to 30, 60
    ax.view_init(30, 60)
    # Set label for x, y, z
    ax.set_xlabel("Avg. Max. Latency (f1)", labelpad=10, fontdict={"fontsize": 12})
    ax.set_ylabel("Deployment Costs (f2)", labelpad=10, fontdict={"fontsize": 12})
    ax.set_zlabel(
        "Avg. Interruption Frequency (f3)", labelpad=8, fontdict={"fontsize": 12}
    )

    # reduce tick font size
    # xlimit 70 175
    # ylimit 150 1200
    # zlimit 0 0.050
    #ax.set_xlim(70, 175)
    #ax.set_ylim(150, 1200)
    #ax.set_zlim(0, 0.050)
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    ax.set_zticklabels(ax.get_zticks(), fontsize=12)
    # ax.tick_params(axis='both', which='major', labelsize=8)
    ax.view_init(elev=20, azim=45)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.00)
    # plt.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
    # plt.tight_layout()
    save_path = os.path.join(output_dir, f"Fig-3d-{alg_name}.pdf")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.4)
    plt.close()


def plot_combined_3d_front(front_dict, save_path):
    """
    Riceve un dizionario front_dict in cui le chiavi sono i nomi degli algoritmi e i valori
    sono liste di punti (liste di obiettivi). Viene calcolato il minimo e massimo globale per
    ciascun asse e impostato:
       xlim = (0.90*min_x, 1.1*max_x)
       ylim = (0.90*min_y, 1.1*max_y)
       zlim = (0.90*min_z, 1.1*max_z)
    Viene poi creato un plot 3D in cui ogni algoritmo viene rappresentato (con marker differenti)
    e il plot viene salvato in save_path.
    """
    # Unisci tutti i punti per calcolare i limiti
    all_points = []
    for points in front_dict.values():
        all_points.extend(points)
    all_points = np.array(all_points)
    if all_points.size == 0:
        print("Nessun punto trovato nel dizionario, nessun plot verrà creato.")
        return

    # Calcola i limiti globali per ciascun obiettivo
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    xlim = (0.90 * mins[0], 1.1 * maxs[0])
    ylim = (0.90 * mins[1], 1.1 * maxs[1])
    zlim = (0.90 * mins[2], 1.1 * maxs[2])

    # Crea la figura 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection="3d")

    # Marker differenti per ogni algoritmo (opzionale)
    markers = {
        "MOCell": "o",
        "NSGAII": "^",
        "NSGAIII": "s",
        "MSPSO": "d",
        "SPEA2": "p",
        "Random Search": "x",
        "GA": "*",
    }

    # Plotta i punti per ciascun algoritmo
    for alg, points in front_dict.items():
        pts = np.array(points)
        marker = markers.get(alg, "o")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, label=alg, marker=marker)

    # Imposta le etichette degli assi
    ax.set_xlabel("Avg. Max. Latency (f1)", labelpad=10, fontdict={"fontsize": 12})
    ax.set_ylabel("Deployment Costs (f2)", labelpad=10, fontdict={"fontsize": 12})
    ax.set_zlabel("Avg. Interruption Frequency (f3)", labelpad=8, fontdict={"fontsize": 12})
    # Imposta i limiti degli assi
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # Aggiungi la legenda
    ax.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.4)
    plt.close()
    print(f"Plot combinato salvato in: {save_path}")


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
        gap = (
            np.abs(meta_solution - closest_ilp_solution)
            / np.abs(closest_ilp_solution)
            * 100
        )
        gaps.append(gap)

    try:
        min_gap = np.min(gaps, axis=0)
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
    dir_usecases_meta = glob.glob("results/*/")
    dir_usecases_ilp = glob.glob("../results/usecase/*/")

    # Retrieve the objectives from the ILP csv files
    for usecase in dir_usecases_ilp:
        output_dir = os.path.join(usecase, "objectives")
        os.makedirs(output_dir, exist_ok=True)
        objectives_files_ilp = glob.glob(f"{usecase}f1_f2_f3/results*.csv")
        for objective_file in objectives_files_ilp:
            retrieve_objectives(objective_file, output_dir)

    ##### Metaheuristics metrics computation #####

    # src_str = 'MultiClusterGPU'
    # files = find_files(src_str)

    for usecase in dir_usecases_meta:
        output_dir_sparsities = os.path.dirname("../sparsities/")
        usecase_name = usecase.split("/")[1]
        usecase_name_path = os.path.basename(os.path.normpath(usecase))
        os.makedirs(output_dir_sparsities, exist_ok=True)
        output_dir_hv = os.path.dirname("../hypervolumes/")
        os.makedirs(output_dir_hv, exist_ok=True)
        output_dir_igds = os.path.dirname("../igds/")
        os.makedirs(output_dir_igds, exist_ok=True)
        # Create directory for dominance approximation gaps
        output_dir_dominance_gaps = os.path.dirname("../dominance_gaps/")
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
        objectives_files_meta = glob.glob(f"results/{usecase}/*.FUN.*")
        objectives_files_ilp = glob.glob(
            f"../results/usecase/{usecase}/objectives/*.txt"
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

        combined_solutions = {}
        for filename in objectives_files_meta:
            # if '.png' in filename:
            #    continue
            print(filename)

            solutions = read_solutions(filename)
            sparsity = sparsity_calculation(solutions, 3)
            # print(f'Number of solutions: {len(solutions)}')
            # Getting the objective values
            objective_values = [s.objectives for s in solutions]
            # base_name = os.path.splitext(os.path.basename(filename))[0]
           

            if "MOCell." in filename:
                objsmocell = objective_values
                algname = "MOCell"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")
                combined_solutions.setdefault(algname, []).extend(objective_values)
            elif "NSGAII." in filename:
                objsnsgaii = objective_values
                algname = "NSGAII"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")
                combined_solutions.setdefault(algname, []).extend(objective_values)
            elif "NSGAIII." in filename:
                objsnsgaiii = objective_values
                algname = "NSGAIII"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")
                combined_solutions.setdefault(algname, []).extend(objective_values)
            elif "MSPSO" in filename:
                objsmspso = objective_values
                algname = "MSPSO"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")
                combined_solutions.setdefault(algname, []).extend(objective_values)
            elif "SPEA2" in filename:
                objsspea2 = objective_values
                algname = "SPEA2"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")
                combined_solutions.setdefault(algname, []).extend(objective_values)
            elif "Random Search" in filename:
                objsrandomsearch = objective_values
                algname = "Random Search"
                f_sparsity.write(f"{algname} Sparsity: {sparsity}\n")
                combined_solutions.setdefault(algname, []).extend(objective_values)

            # algname = filename.split('.')[0]
            # xlabel = src_str[0:2].lower()
            # ylabel = src_str[2:4].lower()
            result_dir = os.path.dirname(filename)
            plot_3d_front(objective_values, algname, result_dir)  # , xlabel, ylabel)

        
        combined_plot_path = os.path.join(result_dir, "combined_plot.pdf")
        plot_combined_3d_front(combined_solutions, combined_plot_path)

        # Add sparsity calculation related to ILP for each gap
        f_sparsity.write(f"\n\n********** ILP **********\n\n")
        for filename in objectives_files_ilp:
            gap = filename.split("_")[-1].split(".t")[0]
            solutions = read_solutions(filename)
            print(f"Filename: {filename} Gap: {gap}: Solutions: {len(solutions)}")
            sparsity = sparsity_calculation(solutions, 3)
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
                    f"NSGAII Dominance Gap: {dominance_based_gap(np.array(objsnsgaii), objsilpgap00)}\n"
                    f"NSGAII Dominance Gap 0.05: {dominance_based_gap(np.array(objsnsgaii), objsilpgap005)}\n"
                    f"NSGAII Dominance Gap 0.1: {dominance_based_gap(np.array(objsnsgaii), objsilpgap01)}\n"
                    f"NSGAII Dominance Gap 0.25: {dominance_based_gap(np.array(objsnsgaii), objsilpgap025)}\n"
                    f"NSGAII Dominance Gap 0.5: {dominance_based_gap(np.array(objsnsgaii), objsilpgap050)}\n"
                    f"NSGAII Dominance Gap 0.75: {dominance_based_gap(np.array(objsnsgaii), objsilpgap075)}\n"
                )
                f.write(
                    f"NSGAIII Dominance Gap: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap00)}\n"
                    f"NSGAIII Dominance Gap 0.05: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap005)}\n"
                    f"NSGAIII Dominance Gap 0.1: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap01)}\n"
                    f"NSGAIII Dominance Gap 0.25: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap025)}\n"
                    f"NSGAIII Dominance Gap 0.5: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap050)}\n"
                    f"NSGAIII Dominance Gap 0.75: {dominance_based_gap(np.array(objsnsgaiii), objsilpgap075)}\n"
                )
                f.write(
                    f"MSPSO Dominance Gap: {dominance_based_gap(np.array(objsmspso), objsilpgap00)}\n"
                    f"MSPSO Dominance Gap 0.05: {dominance_based_gap(np.array(objsmspso), objsilpgap005)}\n"
                    f"MSPSO Dominance Gap 0.1: {dominance_based_gap(np.array(objsmspso), objsilpgap01)}\n"
                    f"MSPSO Dominance Gap 0.25: {dominance_based_gap(np.array(objsmspso), objsilpgap025)}\n"
                    f"MSPSO Dominance Gap 0.5: {dominance_based_gap(np.array(objsmspso), objsilpgap050)}\n"
                    f"MSPSO Dominance Gap 0.75: {dominance_based_gap(np.array(objsmspso), objsilpgap075)}\n"
                )
                f.write(
                    f"SPEA2 Dominance Gap: {dominance_based_gap(np.array(objsspea2), objsilpgap00)}\n"
                    f"SPEA2 Dominance Gap 0.05: {dominance_based_gap(np.array(objsspea2), objsilpgap005)}\n"
                    f"SPEA2 Dominance Gap 0.1: {dominance_based_gap(np.array(objsspea2), objsilpgap01)}\n"
                    f"SPEA2 Dominance Gap 0.25: {dominance_based_gap(np.array(objsspea2), objsilpgap025)}\n"
                    f"SPEA2 Dominance Gap 0.5: {dominance_based_gap(np.array(objsspea2), objsilpgap050)}\n"
                    f"SPEA2 Dominance Gap 0.75: {dominance_based_gap(np.array(objsspea2), objsilpgap075)}\n"
                )
                f.write(
                    f"Random Search Dominance Gap: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap00)}\n"
                    f"Random Search Dominance Gap 0.05: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap005)}\n"
                    f"Random Search Dominance Gap 0.1: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap01)}\n"
                    f"Random Search Dominance Gap 0.25: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap025)}\n"
                    f"Random Search Dominance Gap 0.5: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap050)}\n"
                    f"Random Search Dominance Gap 0.75: {dominance_based_gap(np.array(objsrandomsearch), objsilpgap075)}\n"
                )
                f.write(
                    f"GA Dominance Gap: {dominance_based_gap(np.array(objsga), objsilpgap00)}\n"
                    f"GA Dominance Gap 0.05: {dominance_based_gap(np.array(objsga), objsilpgap005)}\n"
                    f"GA Dominance Gap 0.1: {dominance_based_gap(np.array(objsga), objsilpgap01)}\n"
                    f"GA Dominance Gap 0.25: {dominance_based_gap(np.array(objsga), objsilpgap025)}\n"
                    f"GA Dominance Gap 0.5: {dominance_based_gap(np.array(objsga), objsilpgap050)}\n"
                    f"GA Dominance Gap 0.75: {dominance_based_gap(np.array(objsga), objsilpgap075)}\n"
                )


if __name__ == "__main__":
    main()
