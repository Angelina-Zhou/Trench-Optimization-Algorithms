import numpy as np
import random
import importlib.util

spec = importlib.util.spec_from_file_location(
    "calculate_infiltration", "./calculate_infiltration.py"
)
infiltration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infiltration)

V0 = [infiltration.basin_params[i]['A_i'] * 1 for i in range(infiltration.num_basins)]

NUM_TIME_STEPS = 100
Q_in = 3  # m³/s
TIME_STEP_SECONDS = 3600

# basin_sequence = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
# 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

basin_sequence = [np.int64(1), np.int64(3), np.int64(4), np.int64(1), np.int64(0), np.int64(5), np.int64(2), np.int64(2), np.int64(2), np.int64(7), np.int64(1), np.int64(1), np.int64(2), np.int64(2), np.int64(0), np.int64(4), np.int64(2), np.int64(2), np.int64(6), np.int64(2), np.int64(7), np.int64(2), np.int64(0), np.int64(3), np.int64(1), np.int64(4), np.int64(2), np.int64(8), np.int64(4), np.int64(8), np.int64(1), np.int64(0), np.int64(3), np.int64(5), np.int64(4), np.int64(7), np.int64(6), np.int64(2), np.int64(8), np.int64(2), np.int64(5), np.int64(2), np.int64(4), np.int64(7), np.int64(6), np.int64(2), np.int64(7), np.int64(7), np.int64(1), np.int64(4), np.int64(8), np.int64(5), np.int64(6), np.int64(0), np.int64(1), np.int64(3), np.int64(2), np.int64(1), np.int64(1), np.int64(7), np.int64(6), np.int64(7), np.int64(7), np.int64(2), np.int64(2), np.int64(1), np.int64(1), np.int64(5), np.int64(6), np.int64(5), np.int64(5), np.int64(4), np.int64(2), np.int64(2), np.int64(1), np.int64(5), np.int64(6), np.int64(8), np.int64(6), np.int64(7), np.int64(1), np.int64(4), np.int64(7), np.int64(5), np.int64(6), np.int64(7), np.int64(6), np.int64(2), np.int64(0), np.int64(2), np.int64(7), np.int64(5), np.int64(5), np.int64(0), np.int64(4), np.int64(6), np.int64(5), np.int64(5), np.int64(7), np.int64(2)]

def fitness_function():
    V = V0.copy()
    total_infiltration = 0
    
    for t in range(NUM_TIME_STEPS):
        basin_to_fill = int(basin_sequence[t]) - 1 
        
        if 0 <= basin_to_fill < infiltration.num_basins:
            Q_in_i = Q_in
            dV_dt, vInfiltration = infiltration.compute_dV_dt(V, basin_to_fill)
            

            V = [min(V[i] + float(dV_dt[i]) * TIME_STEP_SECONDS, infiltration.basin_params[i]['vol']) for i in range(infiltration.num_basins)]
            
            total_infiltration += (vInfiltration[basin_to_fill]) * TIME_STEP_SECONDS
    
    return total_infiltration

if __name__ == "__main__":
    print(fitness_function())