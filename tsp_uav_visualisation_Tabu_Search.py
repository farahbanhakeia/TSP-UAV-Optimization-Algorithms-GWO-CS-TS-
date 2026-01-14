import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import random

# =========================
# PARAMÈTRES GÉNÉRAUX Tabu_Search.
# =========================
N_SENSORS = 100       # Nombre de capteurs
AREA_SIZE = 100       # Taille de la zone (x,y)
UAV_SPEED = 5.0       # Vitesse UAV
MAX_ITER = 50         # Nombre max d'itérations Tabu Search
TABU_TENURE = 10      # Taille de la liste Tabu
STAGNATION_LIMIT = 10 # Limite de stagnation
N_RUNS = 5            # Nombre d'exécutions pour stabilité
MAX_NEIGHBORS = 50    # Nombre de voisins testés par itération

# Énergie et temps
E0 = 1.0
ECOL = 2.0
TCOL = 0.5

np.random.seed(42)

# =========================
# CAPTEURS & BASE STATION
# =========================
sensors = np.random.uniform(0, AREA_SIZE, (N_SENSORS, 2))
BS = np.array([0, 0])

# =========================
# FONCTIONS DISTANCE
# =========================
def dist(a, b):
    return np.linalg.norm(a - b)

def total_distance(path_indices):
    """Distance totale UAV pour une séquence d'indices"""
    path = np.vstack([BS, sensors[path_indices], BS])
    return sum(dist(path[i], path[i+1]) for i in range(len(path)-1))

# =========================
# TRAJECTOIRE INITIALE (référence simple)
# =========================
initial_idx = np.random.permutation(N_SENSORS)
initial_distance = total_distance(initial_idx)
initial_path = sensors[initial_idx]

# =========================
# VOISINAGE OPTIMISÉ
# =========================
def get_neighbors(solution, max_neighbors):
    neighbors = []
    n = len(solution)
    tested = set()
    while len(neighbors) < max_neighbors:
        i, j = random.sample(range(n), 2)
        if (i, j) not in tested:
            tested.add((i, j))
            neighbor = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append((neighbor, (i, j)))
    return neighbors

# =========================
# TABU SEARCH
# =========================
def tabu_search_tsp(sensors, max_iter, tabu_tenure):
    start_time = time.time()

    current_solution = np.random.permutation(len(sensors))
    best_solution = current_solution.copy()
    best_fitness = total_distance(best_solution)

    tabu_list = []
    history = []

    stagnation_counter = 0
    stagnation_iter = max_iter

    for it in range(max_iter):
        neighbors = get_neighbors(current_solution, MAX_NEIGHBORS)
        neighbors_sorted = sorted(
            neighbors, key=lambda pair: total_distance(pair[0])
        )

        improved = False

        for neighbor, move in neighbors_sorted:
            if move not in tabu_list:
                current_solution = neighbor
                current_fitness = total_distance(current_solution)

                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
                    stagnation_counter = 0
                    improved = True
                else:
                    stagnation_counter += 1

                # Tabu symétrique
                tabu_list.append(move)
                tabu_list.append((move[1], move[0]))
                if len(tabu_list) > tabu_tenure:
                    tabu_list = tabu_list[-tabu_tenure:]
                break

        history.append(best_fitness)

        if not improved and stagnation_counter >= STAGNATION_LIMIT:
            stagnation_iter = it
            break

    exec_time = time.time() - start_time
    return best_solution, history, stagnation_iter, best_fitness, exec_time

# =========================
# EXÉCUTION TABU SEARCH
# =========================
best_path_ts, history_ts, stagnation_iter, dtotal_ts, exec_time = tabu_search_tsp(
    sensors, MAX_ITER, TABU_TENURE
)

# =========================
# STABILITÉ (N_RUNS)
# =========================
distances_runs = []
for _ in range(N_RUNS):
    _, _, _, d, _ = tabu_search_tsp(sensors, MAX_ITER, TABU_TENURE)
    distances_runs.append(d)

mean_distance = np.mean(distances_runs)
variance_distance = np.var(distances_runs)

# =========================
# GAIN RELATIF vs trajectoire initiale
# =========================
gain_relative = (initial_distance - dtotal_ts) / initial_distance * 100

# =========================
# AFFICHAGE DES MÉTRIQUES 5.1 → 5.9
# =========================
print("\n========= MÉTRIQUES TABU SEARCH =========")
print(f"5.1 Distance totale UAV (dtotal)       : {dtotal_ts:.2f}")
print(f"5.2 Énergie totale consommée (Etot)   : {E0*dtotal_ts + N_SENSORS*ECOL:.2f}")
print(f"5.3 Durée totale mission (Tmission)   : {dtotal_ts/UAV_SPEED + N_SENSORS*TCOL:.2f}")
print(f"5.4 Taux de couverture des capteurs   : {N_SENSORS/N_SENSORS*100:.2f} %")
print(f"5.5 Temps de calcul réel              : {exec_time:.4f} s")
print(f"5.6 Taux de convergence (CR)          : {(initial_distance - dtotal_ts)/initial_distance*100:.2f} %")
print(f"5.7 Itérations jusqu’à stagnation     : {stagnation_iter}")
print(f"5.8 Stabilité (moyenne µd)           : {mean_distance:.2f}")
print(f"    Stabilité (variance σ²d)          : {variance_distance:.4f}")
print(f"5.9 Gain relatif par rapport à ref    : {gain_relative:.2f} %")

# =========================
# FIGURE 1 : AVANT / APRÈS
# =========================
time_initial = initial_distance / UAV_SPEED + N_SENSORS * TCOL
time_opt = dtotal_ts / UAV_SPEED + N_SENSORS * TCOL

fig1, axs = plt.subplots(1, 2, figsize=(16, 6))

# Trajectoire initiale
path_init = np.vstack([BS, initial_path, BS])
axs[0].plot(path_init[:, 0], path_init[:, 1], 'r--')
axs[0].scatter(sensors[:, 0], sensors[:, 1], s=10)
axs[0].scatter(BS[0], BS[1], c='red', marker='s', s=80)
axs[0].set_title(f"AVANT TS\nDist={initial_distance:.2f} | T={time_initial:.2f}")
axs[0].grid()

# Trajectoire optimisée
path_opt = np.vstack([BS, sensors[best_path_ts], BS])
axs[1].plot(path_opt[:, 0], path_opt[:, 1], 'purple')
axs[1].scatter(sensors[:, 0], sensors[:, 1], s=10)
axs[1].scatter(BS[0], BS[1], c='red', marker='s', s=80)
axs[1].set_title(f"APRÈS TS\nDist={dtotal_ts:.2f} | T={time_opt:.2f}")
axs[1].grid()

plt.suptitle("Comparaison des trajectoires UAV – Tabu Search")

# =========================
# FIGURE 2 : CONVERGENCE
# =========================
fig2 = plt.figure()
plt.plot(history_ts, linewidth=2)
plt.xlabel("Itérations")
plt.ylabel("Distance")
plt.title("Convergence Tabu Search – TSP-UAV")
plt.grid()

# =========================
# FIGURE 3 : ANIMATION UAV
# =========================
fig3, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, AREA_SIZE + 5)
ax.set_ylim(-5, AREA_SIZE + 5)

ax.scatter(sensors[:, 0], sensors[:, 1], s=20, label="Capteurs")
ax.scatter(BS[0], BS[1], c="red", marker="s", s=100, label="BS")

line, = ax.plot([], [], 'purple', linewidth=2)
uav, = ax.plot([], [], 'ro')

ax.legend()
ax.grid()

def update(frame):
    line.set_data(path_opt[:frame + 1, 0], path_opt[:frame + 1, 1])
    uav.set_data([path_opt[frame, 0]], [path_opt[frame, 1]])
    return line, uav

ani = FuncAnimation(fig3, update, frames=len(path_opt), interval=120, repeat=False)

# =========================
# AFFICHAGE FINAL
# =========================
plt.show()
