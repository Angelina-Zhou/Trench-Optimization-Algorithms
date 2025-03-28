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
        
        if printing: print(f"Iteration {iter + 1}/{NUM_ITERATIONS}, Best Infiltration: {-global_best_score:.4f} m³")

    return global_best, -global_best_score

if __name__ == "__main__":
    best_sequence, best_infiltration = particle_swarm_optimization(True)

    print("Optimal Basin Sequence:", best_sequence)
    print(f"Maximum Infiltration: {best_infiltration:.4f} m³")
