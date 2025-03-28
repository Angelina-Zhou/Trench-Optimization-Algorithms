import importlib.util

spec = importlib.util.spec_from_file_location(
    "calculate_infiltration", "./calculate_infiltration.py"
)
infiltration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infiltration)

V = [infiltration.basin_params[i]['A_i'] * 1 for i in range(infiltration.num_basins)]

dV_dt, vInfiltration = infiltration.compute_dV_dt(V)

print("Rate of Change of Water Volume (dV/dt) for each Basin:")
for i, dV in enumerate(dV_dt):
    print(f"Basin {i + 1}: {dV:.4f} m³/s")
for i, infiltration in enumerate(vInfiltration):
    print(f"Basin {i + 1}: {infiltration} m³/s")
