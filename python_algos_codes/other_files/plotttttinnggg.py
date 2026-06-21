# import numpy as np
# import matplotlib.pyplot as plt
#
# algorithms = [
#     "Base RRT",
#     "Hybrid RRT/PRM+A*",
#     "Grid A*",
#     "Grid Dijkstra"
# ]
#
# path_distance = np.array([31.05, 24.31, 25.95, 25.95])
# execution_time = np.array([0.0153, 0.0273, 0.0073, 0.0080])
# nodes_explored = np.array([339, 613, 280, 310])
#
# # Normalize values so different units can be compared
# path_norm = path_distance / max(path_distance)
# time_norm = execution_time / max(execution_time)
# nodes_norm = nodes_explored / max(nodes_explored)
#
# x = np.arange(len(algorithms))
# width = 0.25
#
# plt.figure(figsize=(10, 5))
#
# plt.bar(x - width, path_norm, width, label='Path Distance')
# plt.bar(x, time_norm, width, label='Execution Time')
# plt.bar(x + width, nodes_norm, width, label='Nodes Explored')
#
# plt.xticks(x, algorithms)
# plt.ylabel("Normalized Value")
# plt.title("Overall Algorithm Performance Comparison")
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.6)
#
# plt.tight_layout()
# plt.show()





import matplotlib.pyplot as plt

# Data
algorithms = [
    "RRT",
    "Hybrid RRT/PRM",
    "Grid A*",
    "Grid Dijkstra"
]

path_distance = [31.05, 24.31, 25.95, 25.95]
execution_time = [0.0153, 0.0273, 0.0073, 0.0080]
nodes_explored = [339, 613, 280, 310]

# -------------------------------
# Graph 1: Path Distance
# -------------------------------
plt.figure(figsize=(8, 5))
bars = plt.bar(algorithms, path_distance)

plt.title("Path Distance Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Path Distance")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center',
        va='bottom'
    )

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# -------------------------------
# Graph 2: Execution Time
# -------------------------------
plt.figure(figsize=(8, 5))
bars = plt.bar(algorithms, execution_time)

plt.title("Execution Time Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Execution Time (seconds)")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.4f}",
        ha='center',
        va='bottom'
    )

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# -------------------------------
# Graph 3: Nodes Explored
# -------------------------------
plt.figure(figsize=(8, 5))
bars = plt.bar(algorithms, nodes_explored)

plt.title("Nodes Explored Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Number of Nodes")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{int(height)}",
        ha='center',
        va='bottom'
    )

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()