import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# PARAMÈTRES
# =========================
N_SENSORS = 100
AREA_SIZE = 100
UAV_SPEED = 1.0  # vitesse unité par unité de distance
np.random.seed(42)

# =========================
# GÉNÉRATION DES CAPTEURS
# =========================
sensors = np.random.uniform(0, AREA_SIZE, (N_SENSORS, 2))
BS = np.array([0, 0])

# =========================
# DISTANCE
# =========================
def dist(a, b):
    return np.linalg.norm(a - b)

# =========================
# TSP - Nearest Neighbor (NN)
# =========================
visited = [False] * N_SENSORS
path_nn = [BS]
current = BS

for _ in range(N_SENSORS):
    dmin = float("inf")
    idx = -1
    for i, s in enumerate(sensors):
        if not visited[i]:
            d = dist(current, s)
            if d < dmin:
                dmin = d
                idx = i
    visited[idx] = True
    current = sensors[idx]
    path_nn.append(current)

# Retour à la base
path_nn.append(BS)
path_nn = np.array(path_nn)

# =========================
# Baseline aléatoire
# =========================
indices_random = np.random.permutation(N_SENSORS)
path_random = sensors[indices_random]
path_random = np.vstack([BS, path_random, BS])

# =========================
# Fonction pour distance totale
# =========================
def total_distance(path):
    return sum(dist(path[i], path[i+1]) for i in range(len(path)-1))

baseline_distance_nn = total_distance(path_nn)
baseline_distance_random = total_distance(path_random)

# =========================
# Temps total de parcours
# =========================
time_nn = baseline_distance_nn / UAV_SPEED
time_random = baseline_distance_random / UAV_SPEED

print("Distance baseline NN:", baseline_distance_nn)
print("Distance baseline aléatoire:", baseline_distance_random)
print("Temps total NN (aller-retour):", time_nn)
print("Temps total aléatoire (aller-retour):", time_random)

# =========================
# ANIMATION (NN seulement)
# =========================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, AREA_SIZE + 5)
ax.set_ylim(-5, AREA_SIZE + 5)

ax.scatter(sensors[:, 0], sensors[:, 1], c="blue", s=25, label="Capteurs")
ax.scatter(BS[0], BS[1], c="red", marker="s", s=120, label="Station de base")

line, = ax.plot([], [], c="green", linewidth=1)
uav, = ax.plot([], [], "ro", markersize=6)

ax.set_title("Animation du trajet UAV (TSP Nearest Neighbor)")
ax.legend()
ax.grid(True)

def update(frame):
    line.set_data(path_nn[:frame + 1, 0], path_nn[:frame + 1, 1])
    uav.set_data([path_nn[frame, 0]], [path_nn[frame, 1]])
    return line, uav

ani = FuncAnimation(
    fig,
    update,
    frames=len(path_nn),
    interval=120,
    repeat=False
)

plt.show()
