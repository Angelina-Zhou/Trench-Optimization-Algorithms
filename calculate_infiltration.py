import numpy as np

# format: x1, x2, y1, y2, Gamma_{xi}, K_{zi}, p_{xi} (m)
# x and y coordinate system is in metres. p_{xi} is the number of metres of the perimeter
# that is outward-facing.
# NOTE: basin 5 has estimated Gamma and K values based on avg of other basins
basin_data = [
    [-115, -90, 96.5, 125, 7.22, 7.72, 53.5],
    [-86, -62, 96.5, 132.5, 49.40, 5.94, 42],
    [-58, -40, 93, 122, 20.12, 0.03, 47],
    [-123.5, -90, 65, 90.5, 5.44, 3.55, 35],
    [-84.5, -61.5, 53.5, 85.5, 18.4825, 3.4225, 0],
    [-56.5, -33.5, 49, 86, 10.57, 2.09, 43.5],
    [-122.5, -90, 38, 61.5, 42.78, 3.46, 56],
    [-83.5, -60.5, 19.5, 50.5, 3.16, 0.71, 41.5],
    [-55, -32, 9, 45, 9.17, 3.88, 71],
]

# format: basin1, basin2, distance between basins (m), proportion of
# shared perimeter (m), delta_{ij} as estimated in the paper.
# NOTE: for adjacency data involving basin 5, the delta values are estimated
# based on the average of other pairs.
adjacency_data = [
    [1, 2, 4, 28.5, 0.05],
    [1, 4, 6, 25, 0.32],
    [2, 3, 4, 25.5, 0.38],
    [3, 6, 7, 16.5, 0.40],
    [4, 7, 3.5, 32.5, 0.33],
    [6, 8, 4, 15, 0.12],
    [6, 9, 4, 21.5, 0.15],
    [7, 8, 6.5, 12.5, 1.53],
    [8, 9, 5.5, 25.5, 0.85],
    [4, 5, 5.5, 20.5, 0.46],
    [5, 7, 5.5, 8, 0.46],
    [5, 8, 3, 23, 0.46],
    [2, 5, 11, 24.5, 0.46],
    [5, 6, 5, 32, 0.46],
]

adjacency = [
    (1, 2),
    (1, 4),
    (2, 3),
    (3, 6),
    (4, 7),
    (6, 8),
    (6, 9),
    (7, 8),
    (8, 9),
    (4, 5),
    (5, 7),
    (5, 8),
    (2, 5),
    (5, 6),
]

# ---------------------------
# BASIN PARAM COMPUTATIONS
# ---------------------------

# NOTE: depth is an estimated value
depth = 4.0
num_basins = len(basin_data)
# NOTE: Q_in (fill rate) is an estimated value
Q_in = 0.0315

basin_params = []
for i, basin in enumerate(basin_data):

    x1, x2, y1, y2, Gamma_xi, K_zi, p_xi = basin
    A_i = (x2 - x1) * (y2 - y1)  # Area of rectangle
    z_i = -depth  # ASSUMPTION: depth is 4m below ground (we don't have the exact numbers)
    vol = A_i * depth

    basin_params.append({
        'A_i': A_i,
        'Gamma_xi': Gamma_xi * 1e-6,
        'K_zi': K_zi * 1e-6,  # Converted to m/s
        'p_xi': p_xi,
        'z_i': z_i,
        'vol': vol
    })

adjacency = {}

for adj in adjacency_data:

    i, j, dist, p_ij, delta_ij = adj

    adjacency[(i-1, j-1)] = {
        'dist': dist,
        'p_ij': p_ij,
        'delta_ij': delta_ij * 1e-6,  # Convert to m/s
    }

# ---------------------------
# FLOW EQUATIONS
# ------------------------------

# horizontal infiltration
def Q_xi(h_i, params):
    return params['Gamma_xi'] * params['p_xi'] * h_i * -1

# vertical infiltration
def Q_zi(h_i, params):
    if h_i > 0:
        return params['K_zi'] * params['A_i'] * -1
    return 0


def Q_ij(h_i, h_j, params_i, params_j, adj):
    z_i, z_j = params_i['z_i'], params_j['z_i']

    # NOTE: interpretation: the paper had a small ambiguity in its equation when it said
    # Q_ij would be 0 if there exists a j for which the below condition holds.
    # the "exists" quantifier sounds like we should check all other basins,
    # but it uses the same j as in Q_ij. Plus, logically, the below would make more
    # sense:
    if (h_j == 0) and z_j > (h_i + z_i):
        return 0
    
    delta_ij, p_ij, dist = adj['delta_ij'], adj['p_ij'], adj['dist']

    # DEBUG
    # print("Q_ij")
    # print(adj)
    res = delta_ij * (p_ij / (2 * dist)) * ((h_j + z_j) ** 2 - (h_i + z_i) ** 2)
    # print(res)

    return res

# dynamic equation 1
def compute_dV_dt(V, fill_basin = -1):

    h = [V[i] / basin_params[i]['A_i'] for i in range(num_basins)]
    dV = np.zeros(num_basins)
    vInfiltration = np.zeros(num_basins)

    for i in range(num_basins):
        params_i = basin_params[i]

        # TODO: this is set to zero for now, fix later
        # estimated Q_in: 0.0315 m^3/s, or 31.5 litres per second
        Q_in_i = Q_in if i == fill_basin else 0

        # Horizontal and vertical infiltration
        dV[i] += Q_in_i
        dV[i] += Q_xi(h[i], params_i)
        vInfiltration[i] -= Q_xi(h[i], params_i)
        dV[i] += Q_zi(h[i], params_i)
        vInfiltration[i] -= Q_zi(h[i], params_i)

        # Q_ij
        for j in range(num_basins):
            if i != j and (i, j) in adjacency:
                params_j = basin_params[j]
                adj = adjacency[(i, j)]
                dV[i] += Q_ij(h[i], h[j], params_i, params_j, adj)

    return dV, vInfiltration


if __name__ == "__main__":
        
    # ---------------------------
    # CALCULATION, INITIALIZATION, SETUP
    # ---------------------------

    # Initial water volumes in cubic metres (as an example, having 1m water in each basin)
    V0 = [basin_params[i]['A_i'] * 1 for i in range(num_basins)]

    # definition: number of seconds in time step for Euler approximation
    TIME_STEP_SECONDS = 3600

    dV_dt, vInfiltration = compute_dV_dt(V0)

    V1 = [min(V0[i] + float(dV_dt[i]) * TIME_STEP_SECONDS, basin_params[i]['vol']) for i in range(num_basins)]

    # ---------------------------
    # OUTPUT
    # ---------------------------
    print("Rate of Change of Water Volume (dV/dt) for each Basin:")
    for i, dV in enumerate(dV_dt):
        print(f"Basin {i+1}: {dV:.4f} m³/s")
    print("old volumes:")
    for i, volume in enumerate(V0):
        print(f"Basin {i + 1}: {volume}m³")
    print("new volumes:")
    for i, volume in enumerate(V1):
        print(f"Basin {i + 1}: {volume}m³")
    print("infiltration rates:")
    for i, rate in enumerate(vInfiltration):
        print(f"Basin {i + 1}: {rate}m³/s")