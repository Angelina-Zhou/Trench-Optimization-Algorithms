import numpy as np
import random
import importlib.util
import shared_vars

V0 = shared_vars.V0
NUM_PARTICLES = shared_vars.NUM_PARTICLES
NUM_TIME_STEPS = shared_vars.NUM_TIME_STEPS
NUM_ITERATIONS = shared_vars.NUM_ITERATIONS
Q_in = shared_vars.Q_in
TIME_STEP_SECONDS = shared_vars.TIME_STEP_SECONDS


spec = importlib.util.spec_from_file_location(
    "calculate_infiltration_2", "./calculate_infiltration_2.py"
)
infiltration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infiltration)


def particle_swarm_optimization_discrete():

    particles = shared_vars.initialize_particles(NUM_PARTICLES, NUM_TIME_STEPS)
    personal_best = particles.copy()
    personal_best_scores = np.array([shared_vars.fitness_function(p) for p in particles])
    global_best = particles[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

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
        
        print(f"Iteration {iter + 1}/{NUM_ITERATIONS}, Best Score: {-global_best_score:.4f}")

    return global_best, -global_best_score

best_sequence, best_infiltration = particle_swarm_optimization_discrete()

print("Optimal Basin Sequence:", best_sequence)
print(f"Maximum Infiltration: {best_infiltration:.4f} m³")
