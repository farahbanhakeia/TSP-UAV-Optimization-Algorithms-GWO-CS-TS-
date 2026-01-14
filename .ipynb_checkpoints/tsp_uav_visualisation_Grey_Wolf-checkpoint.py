import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# =========================
# PARAMÈTRES  Grey_Wolf
# =========================
N_SENSORS = 100
AREA_SIZE = 100
UAV_SPEED = 5.0
N_WOLVES = 20
MAX_ITER = 50

# Paramètres énergie / temps
E0 = 1.0
ECOL = 2.0
TCOL = 0.5

np.random.seed(42)

# =========================
# GÉNÉRATION DES CAPTEURS
# =========================
sensors = np.random.uniform(0, AREA_SIZE, (N_SENSORS, 2))
BS = np.array([0, 0])

# =========================
# FONCTIONS DISTANCE
# =========================
def dist(a, b):
    return np.linalg.norm(a - b)

def total_distance(path):
    path_full = np.vstack([BS, path, BS])
    return sum(dist(path_full[i], path_full[i+1]) for i in range(len(path_full)-1))

# =========================
# TRAJECTOIRE INITIALE (AVANT OPTIMISATION)
# =========================
initial_idx = np.random.permutation(N_SENSORS)
initial_path = sensors[initial_idx]
initial_distance = total_distance(initial_path)
time_initial = initial_distance / UAV_SPEED + N_SENSORS * TCOL

print(f"Distance initiale (avant GWO): {initial_distance:.2f}, Temps: {time_initial:.2f}")

# =========================
# GWO POUR TSP
# =========================
def gwo_tsp(sensors, n_wolves=10, max_iter=50):
    wolves = [np.random.permutation(len(sensors)) for _ in range(n_wolves)]
    fitness = [total_distance(sensors[w]) for w in wolves]

    sorted_idx = np.argsort(fitness)
    alpha, beta, delta = wolves[sorted_idx[0]], wolves[sorted_idx[1]], wolves[sorted_idx[2]]

    best_history = []
    dstart = fitness[sorted_idx[0]]
    best_prev = dstart
    stagnation = 0

    start_time = time.time()
    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)

        for i in range(n_wolves):
            new_wolf = wolves[i].copy()
            for j in range(len(sensors)):
                if np.random.rand() < 0.5:
                    k = np.random.randint(len(sensors))
                    new_wolf[j], new_wolf[k] = new_wolf[k], new_wolf[j]
            wolves[i] = new_wolf

        fitness = [total_distance(sensors[w]) for w in wolves]
        sorted_idx = np.argsort(fitness)
        alpha, beta, delta = wolves[sorted_idx[0]], wolves[sorted_idx[1]], wolves[sorted_idx[2]]

        best_current = fitness[sorted_idx[0]]
        best_history.append(best_current)

        if abs(best_prev - best_current) < 1e-3:
            stagnation += 1
        else:
            stagnation = 0

        best_prev = best_current
    end_time = time.time()
    return sensors[alpha], best_history, dstart, best_current, stagnation, end_time - start_time

# =========================
# EXÉCUTION GWO
# =========================
best_path_gwo, history, dstart, dend, stagn_iter, calc_time = gwo_tsp(
    sensors, N_WOLVES, MAX_ITER
)

dtotal = total_distance(best_path_gwo)
time_gwo = dtotal / UAV_SPEED + N_SENSORS * TCOL

# =========================
# MÉTRIQUES (5.1 → 5.9)
# =========================
print("\n========= MÉTRIQUES UAV =========")
print(f"5.1 Distance totale UAV (dtotal)       : {dtotal:.2f}")
print(f"5.2 Énergie totale consommée (Etot)   : {E0 * dtotal + N_SENSORS * ECOL:.2f}")
print(f"5.3 Durée totale mission (Tmission)   : {time_gwo:.2f}")
print(f"5.4 Taux de couverture des capteurs   : 100 %")
print(f"5.5 Temps de calcul réel              : {calc_time:.4f} s")
print(f"5.6 Taux de convergence (CR)          : {(dstart - dend) / dstart * 100:.2f} %")
print(f"5.7 Itérations jusqu’à stagnation     : {stagn_iter}")
# =========================
# STABILITÉ (5.8)
# =========================
runs = 10
distances = []
for _ in range(runs):
    _, _, _, d, _, _ = gwo_tsp(sensors, N_WOLVES, MAX_ITER)
    distances.append(d)

print(f"5.8 Moyenne µd                       : {np.mean(distances):.2f}")
print(f"5.8 Variance σ²d                      : {np.var(distances):.2f}")
# =========================
# GAIN RELATIF (5.9) via Nearest Neighbor
# =========================
def nearest_neighbor(sensors):
    visited = [0]
    current = 0
    while len(visited) < len(sensors):
        next_idx = min(
            [i for i in range(len(sensors)) if i not in visited],
            key=lambda i: dist(sensors[current], sensors[i])
        )
        visited.append(next_idx)
        current = next_idx
    return sensors[visited]

dref = total_distance(nearest_neighbor(sensors))
print(f"5.9 Gain relatif par rapport à ref     : {(dref - dtotal) / dref * 100:.2f} %")

# =========================
# FIGURE 1 : COMPARAISON AVANT / APRÈS
# =========================
fig1, axs = plt.subplots(1, 2, figsize=(16, 6))

# Trajectoire AVANT
path_init = np.vstack([BS, initial_path, BS])
axs[0].plot(path_init[:, 0], path_init[:, 1], 'r--', label="Trajectoire")
axs[0].scatter(sensors[:, 0], sensors[:, 1], s=10, label="Capteurs")
axs[0].scatter(BS[0], BS[1], c='red', marker='s', s=80, label="BS")
axs[0].set_title(f"AVANT GWO\nDistance = {initial_distance:.2f}, Durée = {time_initial:.2f}")
axs[0].grid()
axs[0].legend()

# Trajectoire APRÈS
path_opt = np.vstack([BS, best_path_gwo, BS])
axs[1].plot(path_opt[:, 0], path_opt[:, 1], 'purple', label="Trajectoire optimisée")
axs[1].scatter(sensors[:, 0], sensors[:, 1], s=10, label="Capteurs")
axs[1].scatter(BS[0], BS[1], c='red', marker='s', s=80, label="BS")
axs[1].set_title(f"APRÈS GWO\nDistance = {dtotal:.2f}, Durée = {time_gwo:.2f}")
axs[1].grid()
axs[1].legend()

plt.suptitle("Figure 1 : Comparaison des trajectoires UAV")
plt.show(block=False)

# =========================
# FIGURE 2 : CONVERGENCE
# =========================
plt.figure()
plt.plot(history, linewidth=2)
plt.xlabel("Itérations")
plt.ylabel("Distance best")
plt.title("Figure 2 : Courbe de convergence GWO – TSP-UAV")
plt.grid()
plt.show(block=False)

# =========================
# FIGURE 3 : ANIMATION UAV
# =========================
fig3, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, AREA_SIZE + 5)
ax.set_ylim(-5, AREA_SIZE + 5)

ax.scatter(sensors[:, 0], sensors[:, 1], s=20, label="Capteurs")
ax.scatter(BS[0], BS[1], c="red", marker="s", s=100, label="BS")

line, = ax.plot([], [], 'purple', linewidth=2)
uav, = ax.plot([], [], 'ro', markersize=6)

ax.legend()
ax.grid()
ax.set_title("Figure 3 : Animation UAV – GWO TSP")

def update(frame):
    line.set_data(path_opt[:frame + 1, 0], path_opt[:frame + 1, 1])
    uav.set_data([path_opt[frame, 0]], [path_opt[frame, 1]])
    return line, uav

ani = FuncAnimation(fig3, update, frames=len(path_opt), interval=150, repeat=False)
plt.show()
# Exemple pour GWO
import csv

waypoints = np.vstack([BS, best_path_gwo, BS])  # inclut base station au début et fin
times = [0.0]
for i in range(1, len(waypoints)):
    d = np.linalg.norm(waypoints[i] - waypoints[i-1])
    times.append(times[-1] + d / UAV_SPEED)

with open("trajectory_gwo_ns3.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for t, point in zip(times, waypoints):
        writer.writerow([t, point[0], point[1]])


import os
if os.path.exists("trajectory_gwo_ns3.csv"):
    print(" Fichier CSV créé avec succès !")
    print("Chemin complet :", os.path.abspath("trajectory_gwo_ns3.csv"))
else:
    print(" Échec de la création du fichier CSV")
