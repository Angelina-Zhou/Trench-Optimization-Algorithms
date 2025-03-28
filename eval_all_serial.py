import numpy as np
import random
import importlib.util
import matplotlib.pyplot as plt
import csv
import shared_vars


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
    personal_best_scores = [shared_vars.fitness_function(p) for p in particles]
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


            score = shared_vars.fitness_function(particle)
            

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
    personal_best_scores = [shared_vars.fitness_function(p) for p in particles]
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


            score = shared_vars.fitness_function(particle)
            

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
    personal_best_scores = [shared_vars.fitness_function(p) for p in particles]
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


            score = shared_vars.fitness_function(particle)
            

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
    personal_best_scores = [shared_vars.fitness_function(p) for p in particles]
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


            score = shared_vars.fitness_function(particle)
            

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
    personal_best_scores = [shared_vars.fitness_function(p) for p in particles]
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

            score = shared_vars.fitness_function(particle)
            

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
    personal_best_scores = np.array([shared_vars.fitness_function(p) for p in particles])
    global_best = particles[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    max_swaps = NUM_TIME_STEPS // 4
    mutation_rate = 0.05

    for iter in range(NUM_ITERATIONS):
        for i, particle in enumerate(particles):

            update_velocity_discrete(particle, personal_best[i], max_swaps)

            update_velocity_discrete(particle, global_best, max_swaps)

            mutate_particle(particle, mutation_rate)

            score = shared_vars.fitness_function(particle)
            
            if score < personal_best_scores[i]:
                personal_best[i] = particle.copy()
                personal_best_scores[i] = score
            
            if score < global_best_score:
                global_best = particle.copy()
                global_best_score = score
        
        # print(f"Iteration {iter + 1}/{NUM_ITERATIONS}, Best Score: {-global_best_score:.4f}")

    return global_best, -global_best_score

if __name__ == "__main__":

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
        "Random Coefficients PSO": pso_euclidean_random_coeff,
        "Euclidean PSO with Mutation": pso_euclidean_mutation,
        "Discrete PSO with Hamming Distance": pso_hamming
    }


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
            # print(f"run {run + 1} of {name} running...")
            best_sequence, best_score = pso_func()
            
            results[name].append(best_score)
            
            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_solution = best_sequence
            # print(f"run {run + 1} - best score: {best_score:.4f}")

        best_solutions[name] = best_overall_solution
        best_scores[name] = best_overall_score
        print(f"{name} - Best Score: {best_overall_score:.4f}")

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