from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import importlib.util
import matplotlib.pyplot as plt


spec = importlib.util.spec_from_file_location(
    "calculate_infiltration", "./calculate_infiltration.py"
)
infiltration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infiltration)

V0 = [infiltration.basin_params[i]['A_i'] * 1 for i in range(infiltration.num_basins)]

NUM_PARTICLES = 30
NUM_TIME_STEPS = 100
NUM_ITERATIONS = 500
Q_in = 0.0315  # m³/s
TIME_STEP_SECONDS = 3600

# Objective: maximize total infiltration volume after all time steps are done
# uses Newton approximations
def fitness_function(basin_sequence):
    V = V0
    total_infiltration = 0
    
    for t in range(NUM_TIME_STEPS):
        basin_to_fill = int(basin_sequence[t]) - 1 
        
        if 0 <= basin_to_fill < infiltration.num_basins:
            Q_in_i = Q_in
            dV_dt, vInfiltration = infiltration.compute_dV_dt(V, basin_to_fill)
            

            V = [min(V[i] + float(dV_dt[i]) * TIME_STEP_SECONDS, infiltration.basin_params[i]['vol']) for i in range(infiltration.num_basins)]
            
            total_infiltration += (vInfiltration[basin_to_fill]) * TIME_STEP_SECONDS
    
    return -total_infiltration


def initialize_particles(num_particles, num_time_steps):
    return [np.random.randint(1, infiltration.num_basins + 1, size=num_time_steps) for _ in range(num_particles)]

def mutate_particle(particle, mutation_rate):
    for i in range(len(particle)):
        if np.random.random() < mutation_rate:
            particle[i] = np.random.randint(1, infiltration.num_basins + 1)

def hamming_distance(seq1, seq2):
    return np.sum(seq1 != seq2)


def update_velocity_discrete(particle, best, max_swaps):

    diff_indices = np.where(particle != best)[0]
    num_swaps = min(max_swaps, len(diff_indices))
    
    for _ in range(num_swaps):
        swap_idx = np.random.choice(diff_indices)
        particle[swap_idx] = best[swap_idx]


def pso_basic_euclidean():
    particles = initialize_particles(NUM_PARTICLES, NUM_TIME_STEPS)
    velocities = [np.zeros(NUM_TIME_STEPS) for _ in range(NUM_PARTICLES)]
    personal_best = particles[:]
    personal_best_scores = [fitness_function(p) for p in particles]
    global_best = particles[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)


    for iter in range(NUM_ITERATIONS):
        for i, particle in enumerate(particles):

            inertia = 0.7
            cognitive = 1.5 * np.random.random(NUM_TIME_STEPS) * (personal_best[i] - particle)
            social = 1.5 * np.random.random(NUM_TIME_STEPS) * (global_best - particle)
            velocities[i] = inertia * velocities[i] + cognitive + social


            particle += np.round(velocities[i]).astype(int)
            particle = np.clip(particle, 1, infiltration.num_basins)


            score = fitness_function(particle)
            

            if score < personal_best_scores[i]:
                personal_best[i] = particle
                personal_best_scores[i] = score
            

            if score < global_best_score:
                global_best = particle
                global_best_score = score
        
        # print(f"Iteration {iter + 1}/{NUM_ITERATIONS}, Best Infiltration: {-global_best_score:.4f} m³")

    return global_best, -global_best_score

def pso_euclidean_weak_coeff():

    particles = initialize_particles(NUM_PARTICLES, NUM_TIME_STEPS)
    velocities = [np.zeros(NUM_TIME_STEPS) for _ in range(NUM_PARTICLES)]
    personal_best = particles[:]
    personal_best_scores = [fitness_function(p) for p in particles]
    global_best = particles[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)


    for iter in range(NUM_ITERATIONS):
        for i, particle in enumerate(particles):

            inertia = 0.7
            cognitive = np.random.random(NUM_TIME_STEPS) * (personal_best[i] - particle)
            social = np.random.random(NUM_TIME_STEPS) * (global_best - particle)
            velocities[i] = inertia * velocities[i] + cognitive + social


            particle += np.round(velocities[i]).astype(int)
            particle = np.clip(particle, 1, infiltration.num_basins)


            score = fitness_function(particle)
            

            if score < personal_best_scores[i]:
                personal_best[i] = particle
                personal_best_scores[i] = score
            

            if score < global_best_score:
                global_best = particle
                global_best_score = score
        
        # print(f"Iteration {iter + 1}/{NUM_ITERATIONS}, Best Infiltration: {-global_best_score:.4f} m³")

    return global_best, -global_best_score

def pso_euclidean_strong_coeff():
    particles = initialize_particles(NUM_PARTICLES, NUM_TIME_STEPS)
    velocities = [np.zeros(NUM_TIME_STEPS) for _ in range(NUM_PARTICLES)]
    personal_best = particles[:]
    personal_best_scores = [fitness_function(p) for p in particles]
    global_best = particles[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)


    for iter in range(NUM_ITERATIONS):
        for i, particle in enumerate(particles):

            inertia = 0.7
            cognitive = 2 * np.random.random(NUM_TIME_STEPS) * (personal_best[i] - particle)
            social = 2 * np.random.random(NUM_TIME_STEPS) * (global_best - particle)
            velocities[i] = inertia * velocities[i] + cognitive + social


            particle += np.round(velocities[i]).astype(int)
            particle = np.clip(particle, 1, infiltration.num_basins)


            score = fitness_function(particle)
            

            if score < personal_best_scores[i]:
                personal_best[i] = particle
                personal_best_scores[i] = score
            

            if score < global_best_score:
                global_best = particle
                global_best_score = score
        
        # print(f"Iteration {iter + 1}/{NUM_ITERATIONS}, Best Infiltration: {-global_best_score:.4f} m³")

    return global_best, -global_best_score

def pso_euclidean_random_coeff():
    particles = initialize_particles(NUM_PARTICLES, NUM_TIME_STEPS)
    velocities = [np.zeros(NUM_TIME_STEPS) for _ in range(NUM_PARTICLES)]
    personal_best = particles[:]
    personal_best_scores = [fitness_function(p) for p in particles]
    global_best = particles[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)


    for iter in range(NUM_ITERATIONS):
        for i, particle in enumerate(particles):

            inertia = 0.7
            cognitive_coeff = np.random.uniform(0.5, 2.5)
            social_coeff = np.random.uniform(0.5, 2.5)
            cognitive = cognitive_coeff * np.random.random(NUM_TIME_STEPS) * (personal_best[i] - particle)
            social = np.random.random(NUM_TIME_STEPS) * (global_best - particle)
            velocities[i] = inertia * velocities[i] + cognitive + social


            particle += np.round(velocities[i]).astype(int)
            particle = np.clip(particle, 1, infiltration.num_basins)


            score = fitness_function(particle)
            

            if score < personal_best_scores[i]:
                personal_best[i] = particle
                personal_best_scores[i] = score
            

            if score < global_best_score:
                global_best = particle
                global_best_score = score
        
        # print(f"Iteration {iter + 1}/{NUM_ITERATIONS}, Best Infiltration: {-global_best_score:.4f} m³")

    return global_best, -global_best_score

def pso_euclidean_mutation():
    particles = initialize_particles(NUM_PARTICLES, NUM_TIME_STEPS)
    velocities = [np.zeros(NUM_TIME_STEPS) for _ in range(NUM_PARTICLES)]
    personal_best = particles[:]
    personal_best_scores = [fitness_function(p) for p in particles]
    global_best = particles[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)
    mutation_rate = 0.05

    for iter in range(NUM_ITERATIONS):
        for i, particle in enumerate(particles):

            inertia = 0.7
            cognitive = 1.5 * np.random.random(NUM_TIME_STEPS) * (personal_best[i] - particle)
            social = 1.5 * np.random.random(NUM_TIME_STEPS) * (global_best - particle)
            velocities[i] = inertia * velocities[i] + cognitive + social


            particle += np.round(velocities[i]).astype(int)
            particle = np.clip(particle, 1, infiltration.num_basins)
            mutate_particle(particle, mutation_rate)

            score = fitness_function(particle)
            

            if score < personal_best_scores[i]:
                personal_best[i] = particle
                personal_best_scores[i] = score
            

            if score < global_best_score:
                global_best = particle
                global_best_score = score
        
        # print(f"Iteration {iter + 1}/{NUM_ITERATIONS}, Best Infiltration: {-global_best_score:.4f} m³")

    return global_best, -global_best_score

def pso_hamming():

    particles = initialize_particles(NUM_PARTICLES, NUM_TIME_STEPS)
    personal_best = particles.copy()
    personal_best_scores = np.array([fitness_function(p) for p in particles])
    global_best = particles[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    max_swaps = NUM_TIME_STEPS // 4
    mutation_rate = 0.05

    for iter in range(NUM_ITERATIONS):
        for i, particle in enumerate(particles):

            update_velocity_discrete(particle, personal_best[i], max_swaps)

            update_velocity_discrete(particle, global_best, max_swaps)

            mutate_particle(particle, mutation_rate)

            score = fitness_function(particle)
            
            if score < personal_best_scores[i]:
                personal_best[i] = particle.copy()
                personal_best_scores[i] = score
            
            if score < global_best_score:
                global_best = particle.copy()
                global_best_score = score
        
        # print(f"Iteration {iter + 1}/{NUM_ITERATIONS}, Best Score: {-global_best_score:.4f}")

    return global_best, -global_best_score

'''
pso_variants = {
    "Basic PSO": pso_basic_euclidean,
    "Weak Coefficients PSO": pso_euclidean_weak_coeff,
    "Strong Coefficients PSO": pso_euclidean_strong_coeff,
    "Random Coefficients PSO": pso_euclidean_random_coeff,
    "Euclidean PSO with Mutation": pso_euclidean_mutation,
    "Discrete PSO with Hamming Distance": pso_hamming
}'''


pso_variants = {
    # "Basic PSO": pso_basic_euclidean,
    "Weak Coefficients PSO": pso_euclidean_weak_coeff
}


# To store results for each PSO
results = {name: [] for name in pso_variants.keys()}
best_solutions = {}
best_scores = {}



# To store results for each PSO
results = {name: [] for name in pso_variants.keys()}
best_solutions = {}
best_scores = {}

# -------------------------------
# RUN EXPERIMENTS
# -------------------------------

NUM_RUNS = 10

def run_pso(pso_func, name):
    best_overall_solution = None
    best_overall_score = float('-inf')
    run_results = []

    best_sequence, best_score = pso_func()

    run_results.append(best_score)

    if best_score > best_overall_score:
        best_overall_score = best_score
        best_overall_solution = best_sequence
    print(f"run {run} - best score: {best_score:.4f}")
    
    return name, run_results, best_overall_solution, best_overall_score

def run_pso_with_multiprocessing(pso_func, name, num_runs):
    """
    Runs a PSO algorithm with multiprocessing for the individual runs.
    """
    with ProcessPoolExecutor() as executor:
        # Run multiple PSO instances in parallel
        futures = {executor.submit(run_pso, pso_func, name): i for i in range(num_runs)}
        run_results = []
        for future in futures:
            try:
                result = future.result()
                run_results.append(result)
            except Exception as e:
                print(f"Error during PSO run {futures[future]}: {e}")
        return run_results

def main():
    # Loop through each PSO variant serially
    print("Running PSO variants...")

    for name, pso_func in pso_variants.items():
        print(f"Running {name}...")
        run_results = run_pso_with_multiprocessing(pso_func, name, NUM_RUNS)
        results[name] = run_results
        best_solution, best_score = find_best_solution(run_results)  # Assuming this function exists
        best_solutions[name] = best_solution
        best_scores[name] = best_score

    # -------------------------------
    # ANALYSIS AND HISTOGRAMS
    # -------------------------------

    plt.figure(figsize=(12, 8))
    for i, (name, scores) in enumerate(results.items()):
        plt.subplot(2, 2, i + 1)
        plt.hist(scores, bins=20, alpha=0.75, color=np.random.rand(3,), label=name)
        plt.title(f"Histogram of Scores - {name}")
        plt.xlabel("Optimal Score")
        plt.ylabel("Frequency")
        plt.legend()

    plt.tight_layout()
    plt.show()

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

if __name__ == '__main__':
    main()