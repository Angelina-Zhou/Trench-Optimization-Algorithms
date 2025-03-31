import numpy as np
import random
import importlib.util
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import shared_vars
import calculate_infiltration as infiltration

V0 = shared_vars.V0
NUM_PARTICLES = shared_vars.NUM_PARTICLES
NUM_TIME_STEPS = shared_vars.NUM_TIME_STEPS
NUM_ITERATIONS = shared_vars.NUM_ITERATIONS
Q_in = shared_vars.Q_in
TIME_STEP_SECONDS = shared_vars.TIME_STEP_SECONDS

def particle_swarm_optimization(printing = False):

    particles = shared_vars.initialize_particles(NUM_PARTICLES, NUM_TIME_STEPS)
    personal_best = particles.copy()
    personal_best_scores = np.array([shared_vars.fitness_func(p) for p in particles])
    global_best = particles[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)
    fitness_history = []

    max_swaps = NUM_TIME_STEPS // 4
    mutation_rate = 0.05

    for iter in range(NUM_ITERATIONS):
        for i, particle in enumerate(particles):

            shared_vars.update_velocity_discrete(particle, personal_best[i], max_swaps)

            shared_vars.update_velocity_discrete(particle, global_best, max_swaps)

            shared_vars.mutate_particle(particle, mutation_rate)

            score = shared_vars.fitness_function(particle)
            
            if score < personal_best_scores[i]:
                personal_best[i] = particle.copy()
                personal_best_scores[i] = score
            
            if score < global_best_score:
                global_best = particle.copy()
                global_best_score = score
        
        if printing: print(f"Iteration {iter + 1}/{NUM_ITERATIONS}, Best Score: {-global_best_score:.4f}")
        fitness_history.append(-global_best_score)

    return global_best, -global_best_score, fitness_history

if __name__ == "__main__":
    best_sequence, best_infiltration, fitness_history = particle_swarm_optimization(True)
    shared_vars.plot_fitness_progress(fitness_history, "hamming PSO with repair")

    print("Optimal Basin Sequence:", best_sequence)
    print(f"Maximum Infiltration: {best_infiltration:.4f} m³")
