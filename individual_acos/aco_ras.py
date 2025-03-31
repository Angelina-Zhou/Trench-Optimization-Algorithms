import aco_funcs
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import shared_vars

def ant_colony_optimization():
    best_solution, best_fitness, fitness_history, variant = aco_funcs.ant_colony_optimization("ras", True)
    return best_solution, best_fitness, fitness_history

if __name__ == "__main__":
    sol, fitness, fitness_history = ant_colony_optimization()
    shared_vars.plot_fitness_progress(fitness_history, "RAS ACO")
    sol = [int(basin_choice) for basin_choice in sol]
    print(f"{variant} solution:")
    print(sol)
    print(f"{variant} infiltration:")
    print(fitness)