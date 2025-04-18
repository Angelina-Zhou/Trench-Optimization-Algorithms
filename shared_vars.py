import numpy as np
import random
import os
import importlib.util
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

spec = importlib.util.spec_from_file_location(
    "calculate_infiltration", os.path.join(parent_dir, "calculate_infiltration.py")
)
infiltration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infiltration)

infiltration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infiltration)

V0 = [infiltration.basin_params[i]['A_i'] * 1 for i in range(infiltration.num_basins)]

NUM_PARTICLES = 30
NUM_ANTS = 50
NUM_TIME_STEPS = 250
NUM_ITERATIONS = 300
Q_in = 0.5  # m³/s TODO: try 0.0315, which is the combined infiltration rate of V0
TIME_STEP_SECONDS = 3600
PENALTY_COEFF = 0.5

def fitness_func_with_repair(solution):
    V = V0.copy()
    fitness = 0
    penalty = 0

    for t, basin in enumerate(solution):
        dV_dt, vInfiltration = infiltration.compute_dV_dt(V, basin - 1)
        
        vol_basin = V[basin - 1] + dV_dt[basin - 1] * TIME_STEP_SECONDS
        
        if vol_basin > infiltration.basin_params[basin - 1]['vol']:
            overflow_volume = vol_basin - infiltration.basin_params[basin - 1]['vol']
            
            neighbors = [j for (i, j) in infiltration.adjacency if i == basin] + [i for (i, j) in infiltration.adjacency if j == basin]
            redistributed = False
            neighbor_selected = -1

            for neighbor in neighbors:
                neighbor_vol = V[neighbor - 1] + Q_in * TIME_STEP_SECONDS + dV_dt[neighbor - 1] * TIME_STEP_SECONDS

                if neighbor_vol - infiltration.basin_params[neighbor - 1]['vol'] <= overflow_volume:
                    neighbor_selected = neighbor
                    redistributed = True
                    overflow_volume = neighbor_vol - infiltration.basin_params[neighbor - 1]['vol']

            if not redistributed:
                penalty += PENALTY_COEFF * overflow_volume
            else:
                dV_dt, vInfiltration = infiltration.compute_dV_dt(V, neighbor_selected - 1)

        V = [min(V[i] + float(dV_dt[i]) * TIME_STEP_SECONDS, infiltration.basin_params[i]['vol']) for i in range(infiltration.num_basins)]

        fitness += sum(vInfiltration) * TIME_STEP_SECONDS
    return -fitness

def plot_fitness_progress(fitness_history, variant):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, label=f"{variant.upper()}", color='dodgerblue', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Infiltration')
    plt.title(f'Best Infiltration vs Iterations ({variant.upper()})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def fitness_function(basin_sequence):
    V = V0.copy()
    total_infiltration = 0
    
    for t in range(NUM_TIME_STEPS):
        basin_to_fill = int(basin_sequence[t]) - 1
        
        if 0 <= basin_to_fill < infiltration.num_basins:
            Q_in_i = Q_in
            dV_dt, vInfiltration = infiltration.compute_dV_dt(V, basin_to_fill)
            

            V = [min(V[i] + float(dV_dt[i]) * TIME_STEP_SECONDS, infiltration.basin_params[i]['vol']) for i in range(infiltration.num_basins)]
            
            total_infiltration += sum(vInfiltration) * TIME_STEP_SECONDS
    
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