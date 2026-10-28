import numpy as np

def select_best_reference(ilp_solutions):
    """
    Select the best reference in the ILP solutions based on the ideal point.
    """
    # Assuming minimization problems
    ideal = np.min(ilp_solutions, axis=0)
    distances = np.linalg.norm(ilp_solutions - ideal, axis=1)
    best_index = np.argmin(distances)
    return ilp_solutions[best_index]


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

    # Find the best ILP solution 
    sum_objs = []
    for i in range(len(ilp_solutions)):
        sum_objs.append(np.sum(ilp_solutions[i]))

    best_ilp_index = np.argmin(sum_objs)
    best_ilp_solution = ilp_solutions[best_ilp_index]
    best_reference = select_best_reference(ilp_solutions)
    if best_ilp_solution != best_reference:
        print(f"Best ILP solution: {best_ilp_solution}")
        print(f"Reference ILP solution: {best_reference}")
        best_ilp_solution = best_reference
    
    # Iterate over each metaheuristic solution
    for meta_solution in meta_heuristic_solutions:
        gap = []
        for i in range(len(best_ilp_solution)):
            best_value = np.min([best_ilp_solution[i], meta_solution[i]])
            gap.append(float((meta_solution[i] - best_ilp_solution[i]) / max(1e-6, best_value) * 100))
        gaps.append(gap)
    try:
        min_gap_index = np.argmin(list(map(sum, gaps)), axis=0)
        min_gap = gaps[min_gap_index]
    except ValueError as e:
        print(f"Error: {e}")
        min_gap = None

    return min_gap