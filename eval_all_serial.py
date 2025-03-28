import numpy as np
import random
import importlib.util
import matplotlib.pyplot as plt
import csv
import shared_vars
import os
import math


spec = importlib.util.spec_from_file_location(
    "calculate_infiltration", "./calculate_infiltration.py"
)
infiltration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infiltration)

V0 = shared_vars.V0
NUM_PARTICLES = shared_vars.NUM_PARTICLES
NUM_TIME_STEPS = shared_vars.NUM_TIME_STEPS
NUM_ITERATIONS = shared_vars.NUM_ITERATIONS
Q_in = shared_vars.Q_in
TIME_STEP_SECONDS = shared_vars.TIME_STEP_SECONDS
CSV_RESULTS_FILE = 'results.csv'

if __name__ == "__main__":
    individual_psos_dir = './individual_psos'
    pso_variants = {}

    for filename in os.listdir(individual_psos_dir):
        if filename.endswith('.py'):
            module_name = filename[:-3]
            file_path = os.path.join(individual_psos_dir, filename)
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'particle_swarm_optimization'):
                pso_variants[module_name] = module.particle_swarm_optimization


    # To store results for each PSO
    results = {name: [] for name in pso_variants.keys()}
    best_solutions = {}
    best_scores = {}

    # -------------------------------
    # RUN EXPERIMENTS
    # -------------------------------

    NUM_RUNS = 1

    print("Running PSO variants...")

    for name, pso_func in pso_variants.items():
        print(f"running variant {name}...")
        best_overall_solution = None
        best_overall_score = float('-inf')
        for run in range(NUM_RUNS):
            best_sequence, best_score = pso_func()
            
            results[name].append(best_score)
            
            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_solution = best_sequence

        best_solutions[name] = best_overall_solution
        best_scores[name] = best_overall_score
        print(f"{name} - Best Score: {best_overall_score:.4f}")

    # -------------------------------
    # ANALYSIS AND HISTOGRAMS
    # -------------------------------

    num_variants = len(results)
    num_cols = 2
    num_rows = math.ceil(num_variants / num_cols)

    plt.figure(figsize=(12, 8))
    for i, (name, scores) in enumerate(results.items()):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.hist(scores, bins=20, alpha=0.75, color=np.random.rand(3,), label=name)
        plt.title(f"Histogram of Scores - {name}")
        plt.xlabel("Optimal Score")
        plt.ylabel("Frequency")
        plt.legend()

    plt.tight_layout()
    plt.show()

    with open(CSV_RESULTS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(results.keys())

        for i in range(len(next(iter(results.values())))):
            row = [results[key][i] for key in results]
            writer.writerow(row)

    # -------------------------------
    # SUMMARY STATISTICS
    # -------------------------------

    print("\nSummary of Results:")
    for name, scores in results.items():
        avg_score = np.mean(scores)
        best_score = best_scores[name]
        print(f"{name}:")
        print(f"  - Average Optimal Score: {avg_score:.4f}")
        print(f"  - Best Score Found: {best_score:.4f}")
        print(f"  - Best Solution: {best_solutions[name]}")
        print("-" * 40)