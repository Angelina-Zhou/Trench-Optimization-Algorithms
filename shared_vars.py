import numpy as np
import random
import os
import importlib.util

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

spec = importlib.util.spec_from_file_location(
    "calculate_infiltration", os.path.join(parent_dir, "calculate_infiltration.py")
)
infiltration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infiltration)

infiltration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infiltration)

V0 = [infiltration.basin_params[i]['A_i'] * 1 for i in range(infiltration.num_basins)]

NUM_PARTICLES = 30
NUM_TIME_STEPS = 100
NUM_ITERATIONS = 1000
Q_in = 3  # m³/s TODO: try 0.0315, which is the combined infiltration rate of V0
TIME_STEP_SECONDS = 3600


def fitness_function(basin_sequence):
    V = V0.copy()
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