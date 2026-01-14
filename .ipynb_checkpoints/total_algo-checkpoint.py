
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import random
import csv
import os

# =========================
# PARAMÈTRES GÉNÉRAUX
# =========================
N_SENSORS = 100
AREA_SIZE = 100
UAV_SPEED = 5.0

# Paramètres GWO
N_WOLVES = 20
MAX_ITER_GWO = 50

# Paramètres Tabu Search
MAX_ITER_TS = 50
TABU_TENURE = 10
MAX_NEIGHBORS = 50
STAGNATION_LIMIT = 10
N_RUNS = 5

# Énergie et temps
E0 = 1.0
ECOL = 2.0
TCOL = 0.5

np.random.seed(42)

# =========================
# CAPTEURS ET STATION
# =========================
sensors = np.random.uniform(0, AREA_SIZE, (N_SENSORS, 2))
BS = np.array([0, 0])

# =========================
# FONCTIONS DISTANCE
# =========================
def dist(a, b):
    return np.linalg.norm(a - b)

def total_distance(path):
    return sum(dist(path[i], path[i+1]) for i in range(len(path)-1))

def path_distance_from_indices(indices):
    path = np.vstack([BS, sensors[indices], BS])
    return sum(dist(path[i], path[i+1]) for i in range(len(path)-1))

# =========================
# 1️⃣ Ordre aléatoire
# =========================
indices_random = np.random.permutation(N_SENSORS)
path_random = sensors[indices_random]
path_random = np.vstack([BS, path_random, BS])
baseline_distance = total_distance(path_random)
baseline_time = baseline_distance / UAV_SPEED + N_SENSORS * TCOL

print("\n========= MÉTRIQUES UAV (Ordre aléatoire) =========")
print(f"Distance totale : {baseline_distance:.2f}, Temps mission : {baseline_time:.2f}")

# =========================
# 2️⃣ GWO TSP
# =========================
def total_distance_path(path):
    path_full = np.vstack([BS, path, BS])
    return sum(dist(path_full[i], path_full[i+1]) for i in range(len(path_full)-1))

def gwo_tsp(sensors, n_wolves=10, max_iter=50):
    wolves = [np.random.permutation(len(sensors)) for _ in range(n_wolves)]
    fitness = [total_distance_path(sensors[w]) for w in wolves]
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
        fitness = [total_distance_path(sensors[w]) for w in wolves]
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

best_path_gwo, history_gwo, dstart, dend, stagn_iter, calc_time = gwo_tsp(sensors, N_WOLVES, MAX_ITER_GWO)
dtotal_gwo = total_distance_path(best_path_gwo)
time_gwo = dtotal_gwo / UAV_SPEED + N_SENSORS * TCOL

print("\n========= MÉTRIQUES UAV (GWO) =========")
print(f"Distance totale : {dtotal_gwo:.2f}, Temps mission : {time_gwo:.2f}, Temps calcul : {calc_time:.4f}s")
print(f"Convergence : {(dstart-dend)/dstart*100:.2f}% , Itérations stagnation : {stagn_iter}")

# =========================
# 3️⃣ Tabu Search TSP
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

def tabu_search_tsp(sensors, max_iter, tabu_tenure):
    start_time = time.time()
    current_solution = np.random.permutation(len(sensors))
    best_solution = current_solution.copy()
    best_fitness = path_distance_from_indices(best_solution)
    tabu_list = []
    history = []
    stagnation_counter = 0
    stagnation_iter = max_iter
    for it in range(max_iter):
        neighbors = get_neighbors(current_solution, MAX_NEIGHBORS)
        neighbors_sorted = sorted(neighbors, key=lambda pair: path_distance_from_indices(pair[0]))
        improved = False
        for neighbor, move in neighbors_sorted:
            if move not in tabu_list:
                current_solution = neighbor
                current_fitness = path_distance_from_indices(current_solution)
                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
                    stagnation_counter = 0
                    improved = True
                else:
                    stagnation_counter += 1
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

best_path_ts, history_ts, stagn_iter_ts, dtotal_ts, exec_time_ts = tabu_search_tsp(sensors, MAX_ITER_TS, TABU_TENURE)

print("\n========= MÉTRIQUES UAV (Tabu Search) =========")
print(f"Distance totale : {dtotal_ts:.2f}, Temps mission : {dtotal_ts/UAV_SPEED + N_SENSORS*TCOL:.2f}, Temps calcul : {exec_time_ts:.4f}s")
print(f"Convergence : {(baseline_distance - dtotal_ts)/baseline_distance*100:.2f}%, Itérations stagnation : {stagn_iter_ts}")

# =========================
# FIGURE COMPARATIVE
# =========================
fig, axs = plt.subplots(1,3, figsize=(24,6))

# Aleatoire
axs[0].plot(path_random[:,0], path_random[:,1], 'orange', label='Random')
axs[0].scatter(sensors[:,0], sensors[:,1], s=10)
axs[0].scatter(BS[0], BS[1], c='red', marker='s', s=80)
axs[0].set_title(f"Random\nDist={baseline_distance:.2f}")

# GWO
path_gwo = np.vstack([BS, best_path_gwo, BS])
axs[1].plot(path_gwo[:,0], path_gwo[:,1], 'purple', label='GWO')
axs[1].scatter(sensors[:,0], sensors[:,1], s=10)
axs[1].scatter(BS[0], BS[1], c='red', marker='s', s=80)
axs[1].set_title(f"GWO\nDist={dtotal_gwo:.2f}")

# Tabu
path_ts = np.vstack([BS, sensors[best_path_ts], BS])
axs[2].plot(path_ts[:,0], path_ts[:,1], 'green', label='Tabu Search')
axs[2].scatter(sensors[:,0], sensors[:,1], s=10)
axs[2].scatter(BS[0], BS[1], c='red', marker='s', s=80)
axs[2].set_title(f"Tabu Search\nDist={dtotal_ts:.2f}")

plt.suptitle("Comparaison des trajectoires UAV")
plt.show()
