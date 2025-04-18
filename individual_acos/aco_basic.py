import aco_funcs
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import shared_vars
import calculate_infiltration as infiltration
import csv
import math
import matplotlib.pyplot as plt
import numpy as np


def ant_colony_optimization(pheromone, fh=False):
    best_solution, best_fitness, fitness_history, variant = aco_funcs.ant_colony_optimization(pheromone, "basic", True)
    if fh:
        return best_solution, best_fitness, fitness_history
    else:
        return best_solution, best_fitness

'''
if __name__ == "__main__":
    pheromone = np.ones((infiltration.num_basins, shared_vars.NUM_TIME_STEPS))
    sol, fitness, fitness_history = ant_colony_optimization(pheromone,True)
    shared_vars.plot_fitness_progress(fitness_history, "Repaired ACO")
    sol = [int(basin_choice) for basin_choice in sol]
    print(f"{variant} solution:")
    print(sol)
    print(f"{variant} infiltration:")
    print(fitness)'''


if __name__ == "__main__":
    NUM_RUNS = 50
    best_overall_solution = None
    best_overall_score = float('-inf')
    results = []
    for run in range(NUM_RUNS):
        pheromone = np.ones((infiltration.num_basins, shared_vars.NUM_TIME_STEPS))
        best_sequence, best_score = ant_colony_optimization(pheromone)
            
        results.append(best_score)
            
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_solution = best_sequence
        print(f"run {run + 1}: best score: {best_score}")
    
    name = 'ACO_repaired'

    # -------------------------------
    # ANALYSIS AND HISTOGRAMS
    # -------------------------------

    num_cols = 2
    num_rows = len(results)

    plt.figure(figsize=(12, 8))
    plt.hist(results, bins=20, alpha=0.75, color=np.random.rand(3,), label=name)
    plt.title(f"Histogram of Scores - {name}")
    plt.xlabel("Optimal Score")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

    CSV_RESULTS_FILE = f'{name}.csv'

    with open(CSV_RESULTS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([name])

        for i in results:
            row = [i]
            writer.writerow(row)

    # -------------------------------
    # SUMMARY STATISTICS
    # -------------------------------

    print("\nSummary of Results:")
    avg_score = np.mean(results)
    print(f"{name}:")
    print(f"  - Average Optimal Score: {avg_score:.4f}")
    print(f"  - Best Score Found: {best_overall_score:.4f}")
    print(f"  - Best Solution: {[int(i) for i in best_overall_solution]}")
    print("-" * 40)