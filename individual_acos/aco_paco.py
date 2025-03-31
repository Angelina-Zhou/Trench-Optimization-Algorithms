import numpy as np
import random
import importlib.util
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import shared_vars
import calculate_infiltration as infiltration
import aco_funcs

V0 = shared_vars.V0

# --------------------------------------
# PARAMETERS
# --------------------------------------
NUM_ANTS = shared_vars.NUM_ANTS
NUM_TIME_STEPS = shared_vars.NUM_TIME_STEPS
NUM_ITERATIONS = shared_vars.NUM_ITERATIONS
num_basins = infiltration.num_basins
rho = 0.1  # Evaporation rate
Q = 100  # Pheromone deposit factor

# Pheromone matrix: num_basins x num_steps
pheromone = np.ones((num_basins, NUM_TIME_STEPS))

# Heuristic information (optional, for initial guidance)
# Can be infiltration rate or inverse of distance if applicable
heuristic_info = np.ones((num_basins, NUM_TIME_STEPS))

'''
# Transpose the heuristic_info list of lists (basin -> row, time step -> column)
heuristic_info = [list(infiltration.compute_dV_dt(V0.copy(), -1)[0])] * NUM_TIME_STEPS

# Step 2: Transpose heuristic_info (swap rows and columns)
heuristic_info = list(zip(*heuristic_info))  # Transpose to basin as rows and time steps as columns

# Step 3: Convert back to list of lists
heuristic_info = [list(row) for row in heuristic_info]
heuristic_info = [[float(abs(value) * 333) for value in row] for row in heuristic_info]

heuristic_info = [
    [1.96029441], [2.39991768], [0.3201129], [1.0732548374999997], 
    [0.8388136799999999], [0.745382205], [1.6777389150000002], 
    [0.21224421], [1.2866154300000001]
]
flat_heuristic = [value[0] for value in heuristic_info]

min_val = min(flat_heuristic)
max_val = max(flat_heuristic)
normalized_heuristic = [(value - min_val) / (max_val - min_val) for value in flat_heuristic]
adjusted_heuristic_info = [[value + 1] * NUM_TIME_STEPS for value in normalized_heuristic]
for value in adjusted_heuristic_info:
    print(value[0])'''

def update_pheromones_paco(pheromone, population, fitnesses):
    """
    Update pheromones based on the population of best solutions.
    """
    pheromone *= (1 - rho)  # Evaporation
    for solution, fit in zip(population, fitnesses):
        for t, basin in enumerate(solution):
            pheromone[basin, t] += Q * (fit / np.max(fitnesses))

def ant_colony_optimization(printing=False,fh=False):
    """
    PACO: Ant Colony Optimization with Population Retention.
    """
    global pheromone
    population = []  # Population of best solutions
    fitnesses = []  # Corresponding fitnesses
    best_solution = None
    best_fitness = -np.inf
    if fh: fitness_history = []

    for iteration in range(NUM_ITERATIONS):
        solutions = []
        new_fitnesses = []

        # Generate solutions with standard ACO
        for ant in range(NUM_ANTS):
            solution = aco_funcs.construct_solution(pheromone, heuristic_info)
            fit = -shared_vars.fitness_func_with_repair(solution)
            solutions.append(solution)
            new_fitnesses.append(fit)

            # Track best solution
            if fit > best_fitness:
                best_fitness = fit
                best_solution = solution

        # Update population with best solutions
        combined = list(zip(solutions, new_fitnesses))
        combined.sort(key=lambda x: x[1], reverse=True)
        population = [sol for sol, fit in combined[:10]]  # Top 10 solutions
        fitnesses = [fit for sol, fit in combined[:10]]

        # Update pheromones based on best solutions
        update_pheromones_paco(pheromone, population, fitnesses)

        if printing: print(f"Iteration {iteration+1}/{NUM_ITERATIONS}, Best Infiltration: {best_fitness:.4f}")
        if fh: fitness_history.append(best_fitness)
    if fh:
        return best_solution, best_fitness, fitness_history
    else:
        return best_solution, best_fitness

if __name__ == "__main__":
    sol, fitness, fitness_history = ant_colony_optimization(True,True)
    shared_vars.plot_fitness_progress(fitness_history, "PACO")
    sol = [int(basin_choice) for basin_choice in sol]
    print(f"paco without info solution:")
    print(sol)
    print(f"paco without info infiltration:")
    print(fitness)