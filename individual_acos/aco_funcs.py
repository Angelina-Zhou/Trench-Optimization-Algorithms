import numpy as np
import random
import importlib.util
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import shared_vars
import calculate_infiltration as infiltration
import math

V0 = shared_vars.V0

# --------------------------------------
# PARAMETERS
# --------------------------------------
NUM_ANTS = shared_vars.NUM_ANTS
NUM_TIME_STEPS = shared_vars.NUM_TIME_STEPS
NUM_ITERATIONS = shared_vars.NUM_ITERATIONS
num_basins = infiltration.num_basins

rho = 0.1  # Evaporation rate
alpha = 1.0  # Pheromone influence
beta = 0.2  # Heuristic influence
Q = 100  # Pheromone deposit factor

# Pheromone matrix: num_basins x num_steps
# pheromone = np.ones((num_basins, NUM_TIME_STEPS))

# Heuristic information (optional, for initial guidance)
# Can be infiltration rate or inverse of distance if applicable
heuristic_info = np.ones((num_basins, NUM_TIME_STEPS))


# -------------------------------
# V0 INFILTRATION HEURISTIC
# -------------------------------
'''
heuristic_info = [
    [1.96029441], [2.39991768], [0.3201129], [1.0732548374999997], 
    [0.8388136799999999], [0.745382205], [1.6777389150000002], 
    [0.21224421], [1.2866154300000001]
]
flat_heuristic = [value[0] for value in heuristic_info]

min_val = min(flat_heuristic)
max_val = max(flat_heuristic)
normalized_heuristic = [(value - min_val) / (max_val - min_val) for value in flat_heuristic]
heuristic_info = [[value + 1] * NUM_TIME_STEPS for value in normalized_heuristic]'''

# -------------------------------
# GAMMA HEURISTIC
# -------------------------------

heuristic_info = [[math.log(basin['Gamma_xi'] * 1e6) + 1] * NUM_TIME_STEPS for basin in infiltration.basin_params]

# ant decision making
def construct_solution(pheromone, heuristic_info):
    """
    Construct a solution (sequence of basins to fill) based on pheromone and heuristic information.
    """
    solution = []
    for t in range(NUM_TIME_STEPS):
        probabilities = np.zeros(num_basins)

        for b in range(num_basins):
            probabilities[b] = (pheromone[b, t] ** alpha) * (heuristic_info[b][t] ** beta)

        probabilities /= np.sum(probabilities)

        basin_choice = np.random.choice(np.arange(num_basins), p=probabilities)
        solution.append(basin_choice)

    return solution

def update_pheromones(pheromone, solutions, fitnesses):
    """
    Update pheromones based on ant solutions and their fitness.
    """
    pheromone *= (1 - rho)

    for solution, fit in zip(solutions, fitnesses):
        for t, basin in enumerate(solution):
            pheromone[basin, t] += Q * (fit / np.max(fitnesses))

def update_pheromones_eas(pheromone, solutions, fitnesses):
    """
    Update pheromones using elitist strategy where best ant reinforces more.
    """
    pheromone *= (1 - rho)
    best_idx = np.argmax(fitnesses)
    best_solution = solutions[best_idx]
    best_fit = fitnesses[best_idx]

    for t, basin in enumerate(best_solution):
        pheromone[basin, t] += Q * (best_fit / np.max(fitnesses))

def update_pheromones_ras(pheromone, solutions, fitnesses, n_r=10):
    """
    Update pheromones with rank-based pheromone contribution.
    """
    pheromone *= (1 - rho)
    ranked_indices = np.argsort(fitnesses)[::-1]
    for rank, idx in enumerate(ranked_indices[:n_r]):
        solution = solutions[idx]
        for t, basin in enumerate(solution):
            pheromone[basin, t] += (n_r - rank) * Q * (fitnesses[idx] / np.max(fitnesses))

def ant_colony_optimization(pheromone, variant="basic", printing=False):
    """
    Main ACO function to optimize basin filling sequence.
    variant: "basic", "eas", "ras", "mmas", or "acs"
    """
    best_solution = None
    best_fitness = -np.inf
    fitness_history = []

    for iteration in range(NUM_ITERATIONS):
        solutions = []
        fitnesses = []

        for ant in range(NUM_ANTS):
            solution = construct_solution(pheromone, heuristic_info)
            fit = -shared_vars.fitness_func_with_repair(solution)
            solutions.append(solution)
            fitnesses.append(fit)

            if fit > best_fitness:
                best_fitness = fit
                best_solution = solution

        if variant == "eas":
            update_pheromones_eas(pheromone, solutions, fitnesses)
        elif variant == "ras":
            update_pheromones_ras(pheromone, solutions, fitnesses)
        else:
            update_pheromones(pheromone, solutions, fitnesses)
        
        fitness_history.append(best_fitness)
        if printing: print(f"Iteration {iteration+1}/{NUM_ITERATIONS}, Best Infiltration: {best_fitness:.4f}")

    return best_solution, best_fitness, fitness_history, variant + " with trench filling heuristics"