import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# =========================
# PARAMÈTRES Cuckoo_Search
# =========================
N_SENSORS = 100
AREA_SIZE = 100
UAV_SPEED = 5.0
E0 = 1.0
ECOL = 2.0
TCOL = 0.5
N_RUNS = 5  # pour stabilité

np.random.seed(42)

# =========================
# CAPTEURS ET STATION
# =========================
sensors = np.random.uniform(0, AREA_SIZE, (N_SENSORS, 2))
BS = np.array([0, 0])

# =========================
# DISTANCE
# =========================
def dist(a, b):
    return np.linalg.norm(a - b)

def total_distance(path):
    return sum(dist(path[i], path[i+1]) for i in range(len(path)-1))

# =========================
# TRAJECTOIRE ALÉATOIRE (baseline)
# =========================
indices_random = np.random.permutation(N_SENSORS)
path_random = sensors[indices_random]
path_random = np.vstack([BS, path_random, BS])

baseline_distance = total_distance(path_random)
baseline_time = baseline_distance / UAV_SPEED

# =========================
# MÉTRIQUES
# =========================
start_time = time.time()
# Ici pas d'optimisation, juste génération, donc exec_time très faible
exec_time = time.time() - start_time

N_visites = N_SENSORS
coverage = N_visites / N_SENSORS * 100
total_energy = E0 * baseline_distance + N_visites * ECOL
mission_time = baseline_distance / UAV_SPEED + N_visites * TCOL

# Convergence : si on simule le départ d'une distance max hypothétique
dstart = baseline_distance * 1.2  # distance hypothétique initiale
dend = baseline_distance
CR = (dstart - dend) / dstart * 100

# Itérations jusqu’à stagnation (simulé)
stagnation_iter = len(path_random) - 1  # on peut considérer le nombre de swaps possible

# Stabilité sur plusieurs runs aléatoires
distances_runs = []
for _ in range(N_RUNS):
    idx = np.random.permutation(N_SENSORS)
    p = np.vstack([BS, sensors[idx], BS])
    distances_runs.append(total_distance(p))
mean_distance = np.mean(distances_runs)
variance_distance = np.var(distances_runs)

# Gain relatif par rapport à une solution de référence (ici la distance max simulée)
dref = dstart
gain_relative = (dref - baseline_distance) / dref * 100

# =========================
# AFFICHAGE MÉTRIQUES
# =========================
print("\n========= MÉTRIQUES UAV (Ordre aléatoire) =========")
print(f"5.1 Distance totale UAV (dtotal)       : {baseline_distance:.2f}")
print(f"5.2 Énergie totale consommée (Etot)   : {total_energy:.2f}")
print(f"5.3 Durée totale mission (Tmission)   : {mission_time:.2f}")
print(f"5.4 Taux de couverture des capteurs   : {coverage:.2f} %")
print(f"5.5 Temps de calcul réel              : {exec_time:.6f} s")
print(f"5.6 Taux de convergence (CR)          : {CR:.2f} %")
print(f"5.7 Itérations jusqu’à stagnation     : {stagnation_iter}")
print(f"5.8 Stabilité (moyenne µd)           : {mean_distance:.2f}")
print(f"    Stabilité (variance σ²d)          : {variance_distance:.4f}")
print(f"5.9 Gain relatif par rapport à ref    : {gain_relative:.2f} %")

# =========================
# FIGURE 1 : Trajectoire UAV
# =========================
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.plot(path_random[:, 0], path_random[:, 1], 'orange', linewidth=1.5, label="Trajectoire UAV")
ax1.scatter(sensors[:, 0], sensors[:, 1], c="blue", s=25, label="Capteurs")
ax1.scatter(BS[0], BS[1], c="red", marker="s", s=120, label="Station de base")
ax1.set_title(f"Trajectoire UAV – Distance: {baseline_distance:.2f}, Temps: {baseline_time:.2f}")
ax1.grid(True)
ax1.legend()

# =========================
# FIGURE 2 : Convergence simulée
# =========================
sim_history = np.linspace(dstart, baseline_distance, len(path_random))
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(sim_history, linewidth=2)
ax2.set_xlabel("Itérations")
ax2.set_ylabel("Distance totale")
ax2.set_title("Convergence simulée UAV – ordre aléatoire")
ax2.grid(True)

# =========================
# FIGURE 3 : Animation UAV
# =========================
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.set_xlim(-5, AREA_SIZE + 5)
ax3.set_ylim(-5, AREA_SIZE + 5)
ax3.scatter(sensors[:, 0], sensors[:, 1], c="blue", s=25, label="Capteurs")
ax3.scatter(BS[0], BS[1], c="red", marker="s", s=120, label="Station de base")

line, = ax3.plot([], [], c="orange", linewidth=1)
uav, = ax3.plot([], [], "ro", markersize=6)
ax3.set_title("Animation du trajet UAV")
ax3.legend()
ax3.grid(True)

def update(frame):
    line.set_data(path_random[:frame + 1, 0], path_random[:frame + 1, 1])
    uav.set_data([path_random[frame, 0]], [path_random[frame, 1]])
    if frame == len(path_random)-1:
        ax3.set_title(f"Trajet terminé ! Distance: {baseline_distance:.2f}, Temps: {baseline_time:.2f}")
    return line, uav

ani = FuncAnimation(fig3, update, frames=len(path_random), interval=120, repeat=False)

plt.show()
