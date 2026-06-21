import math
import random
import heapq
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from scipy.spatial import KDTree


# ==========================================
# 1. THE ALGORITHM CORE ("My Algorithm")
# ==========================================
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.neighbors = []


class IncrementalHybridAlgorithm:
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=0.5):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.min_rand, self.max_rand = rand_area
        self.expand_dis = expand_dis

        # State variables
        self.node_list = [self.start]
        self.goal_reached = False
        self.goal_node = None

        self.rrt_path = None
        self.astar_path = None
        self.path_length = 0.0

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node.x += min(extend_length, d) * math.cos(theta)
        new_node.y += min(extend_length, d) * math.sin(theta)
        new_node.parent = from_node
        return new_node

    def check_collision(self, from_node, to_node, obstacle_list):
        steps = 10
        for i in range(steps + 1):
            x = from_node.x + (to_node.x - from_node.x) * (i / steps)
            y = from_node.y + (to_node.y - from_node.y) * (i / steps)
            for (ox, oy, w, h) in obstacle_list:
                if ox <= x <= (ox + w) and oy <= y <= (oy + h):
                    return False
        return True

    def add_single_node(self, bias_percentage):
        """Attempts to add a single node. Returns True if a VALID node was added."""
        if random.randint(0, 100) > bias_percentage:
            rnd_node = Node(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand))
        else:
            rnd_node = Node(self.end.x, self.end.y)

        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in self.node_list]
        nearest_node = self.node_list[dlist.index(min(dlist))]
        new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

        if self.check_collision(nearest_node, new_node, self.obstacle_list):
            self.node_list.append(new_node)

            # Check Goal
            if not self.goal_reached and math.hypot(new_node.x - self.end.x,
                                                    new_node.y - self.end.y) <= self.expand_dis:
                final_node = self.steer(new_node, self.end, self.expand_dis)
                if self.check_collision(new_node, final_node, self.obstacle_list):
                    self.node_list.append(final_node)
                    self.goal_reached = True
                    self.goal_node = final_node

            return True  # Successfully added a valid node
        return False  # Node was blocked by obstacle

    def calculate_path_length(self, path):
        """Calculates the Euclidean distance of a given path."""
        if not path or len(path) < 2:
            return 0.0
        length = 0.0
        for i in range(len(path) - 1):
            length += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        return length

    def run_hybrid_prm_astar(self):
        """Builds the PRM web on the current RRT nodes and finds shortest path with A*."""
        for node in self.node_list:
            node.neighbors = []

        # 1. Preserve RRT Tree connections
        for node in self.node_list:
            if node.parent:
                node.neighbors.append(node.parent)
                node.parent.neighbors.append(node)

        # 2. Add PRM 10-Nearest Neighbors
        node_coords = [[n.x, n.y] for n in self.node_list]
        tree = KDTree(node_coords)
        k_query = min(11, len(self.node_list))

        for node in self.node_list:
            distances, indices = tree.query([node.x, node.y], k=k_query)
            if k_query > 1:
                for idx in indices[1:]:
                    neighbor = self.node_list[idx]
                    if neighbor not in node.neighbors:
                        if self.check_collision(node, neighbor, self.obstacle_list):
                            node.neighbors.append(neighbor)
                            neighbor.neighbors.append(node)

        # 3. Run A*
        start_node = self.node_list[0]
        open_set = []
        heapq.heappush(open_set, (0, id(start_node), start_node))
        came_from = {}
        g_score = {start_node: 0}

        def heuristic(n1, n2):
            return math.hypot(n1.x - n2.x, n1.y - n2.y)

        f_score = {start_node: heuristic(start_node, self.goal_node)}

        while open_set:
            current = heapq.heappop(open_set)[2]

            if current == self.goal_node:
                self.astar_path = []
                while current in came_from:
                    self.astar_path.append([current.x, current.y])
                    current = came_from[current]
                self.astar_path.append([start_node.x, start_node.y])

                # Calculate Length
                self.path_length = self.calculate_path_length(self.astar_path)
                break

            for neighbor in current.neighbors:
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, self.goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))

        # Store basic RRT path
        self.rrt_path = []
        curr = self.goal_node
        while curr is not None:
            self.rrt_path.append([curr.x, curr.y])
            curr = curr.parent


# ==========================================
# 2. THE TKINTER GUI APP
# ==========================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("My Algorithm - Interactive Planner")
        self.root.geometry("950x700")

        # New starting coordinates in the middle
        self.start_pos = (5.0, 5.0)
        self.goal_pos = (9.0, 9.0)
        self.area = [0, 10]

        # Obstacle Definitions surrounding the new center Start point
        self.obstacles = {
            "Maze": [
                (2.0, 2.0, 6.0, 1.0),  # Bottom wall
                (2.0, 2.0, 1.0, 6.0),  # Left wall
                (2.0, 7.0, 4.0, 1.0),  # Top wall (gap on top right)
                (7.0, 4.0, 1.0, 4.0)  # Right wall (gap on bottom right)
            ],
            "Concave": [
                (3.0, 3.0, 1.0, 5.0),  # Left
                (3.0, 3.0, 5.0, 1.0),  # Bottom
                (7.0, 3.0, 1.0, 5.0)  # Right
            ],
            "None": []
        }

        self.algo = None
        self.setup_ui()
        self.reset_algorithm()

    def setup_ui(self):
        # --- Left Control Panel ---
        control_frame = tk.Frame(self.root, width=250, padx=15, pady=15, bg="#f0f0f0")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="Algorithm", bg="#f0f0f0", font=("Arial", 9)).pack(anchor="w")
        tk.Label(control_frame, text="My Algorithm (RRT + A*)", bg="#e0e0ff", relief="groove", padx=5, pady=3,
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 15))

        tk.Label(control_frame, text="Obstacle Type", bg="#f0f0f0").pack(anchor="w")
        self.obs_var = tk.StringVar(value="Maze")
        obs_combo = ttk.Combobox(control_frame, textvariable=self.obs_var, values=["Maze", "Concave", "None"],
                                 state="readonly")
        obs_combo.pack(anchor="w", fill=tk.X, pady=(0, 15))
        obs_combo.bind("<<ComboboxSelected>>", lambda e: self.reset_algorithm())

        tk.Label(control_frame, text="Number of Nodes to Add:", bg="#f0f0f0").pack(anchor="w")
        btn_frame = tk.Frame(control_frame, bg="#f0f0f0")
        btn_frame.pack(anchor="w", pady=(0, 15))

        ttk.Button(btn_frame, text="1", width=3, command=lambda: self.step_algo(1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="10", width=3, command=lambda: self.step_algo(10)).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="100", width=4, command=lambda: self.step_algo(100)).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="200", width=4, command=lambda: self.step_algo(200)).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="500", width=4, command=lambda: self.step_algo(500)).pack(side=tk.LEFT, padx=1)

        tk.Label(control_frame, text="Exploration Bias", bg="#f0f0f0").pack(anchor="w")
        self.bias_slider = tk.Scale(control_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                                    bg="#f0f0f0", highlightthickness=0)
        self.bias_slider.set(0.10)
        self.bias_slider.pack(anchor="w", fill=tk.X, pady=(0, 15))

        ttk.Button(control_frame, text="Clear & Reset", command=self.reset_algorithm).pack(anchor="w", pady=20)

        # --- Right Canvas Panel ---
        self.canvas_frame = tk.Frame(self.root, bg="white")
        self.canvas_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def reset_algorithm(self):
        obs_type = self.obs_var.get()
        obs_list = self.obstacles[obs_type]
        self.algo = IncrementalHybridAlgorithm(self.start_pos, self.goal_pos, obs_list, self.area)
        self.update_plot(show_prm=False)

    def step_algo(self, num_nodes):
        bias = self.bias_slider.get()
        bias_percentage = int(bias * 100)

        added = 0
        # While loop guarantees exactly N VALID nodes are added
        while added < num_nodes:
            success = self.algo.add_single_node(bias_percentage)
            if success:
                added += 1
                # Redraw UI every 2 nodes for a fast, animated visual effect without freezing
                if added % 10 == 0 or added == num_nodes:
                    self.update_plot(show_prm=False)
                    self.root.update()

                    # When batch is done, calculate shortest path if goal is reached
        if self.algo.goal_reached:
            self.algo.run_hybrid_prm_astar()
            self.update_plot(show_prm=True)
            self.root.update()

    def update_plot(self, show_prm=False):
        self.ax.clear()

        self.ax.set_xlim(self.area[0], self.area[1])
        self.ax.set_ylim(self.area[0], self.area[1])
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        border = patches.Rectangle((0, 0), 10, 10, linewidth=6, edgecolor='blue', facecolor='none')
        self.ax.add_patch(border)

        # Draw Obstacles as Grey
        obs_type = self.obs_var.get()
        for (ox, oy, w, h) in self.obstacles[obs_type]:
            rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='grey')
            self.ax.add_patch(rect)

        # Draw PRM Web (Cyan) if requested
        if show_prm and self.algo.goal_reached:
            drawn_edges = set()
            for node in self.algo.node_list:
                for neighbor in node.neighbors:
                    # Do not draw cyan over the RRT tree connections
                    if neighbor != node.parent and node != neighbor.parent:
                        edge = tuple(sorted([id(node), id(neighbor)]))
                        if edge not in drawn_edges:
                            self.ax.plot([node.x, neighbor.x], [node.y, neighbor.y], color="cyan", linewidth=0.8,
                                         alpha=0.5, zorder=1)
                            drawn_edges.add(edge)

        # Draw RRT Tree Edges (Black)
        for node in self.algo.node_list:
            if node.parent:
                self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-k", linewidth=1.0, zorder=2)

        # Draw Nodes
        nodes_x = [n.x for n in self.algo.node_list]
        nodes_y = [n.y for n in self.algo.node_list]
        self.ax.scatter(nodes_x, nodes_y, color='red', s=15, zorder=3)

        # Draw Start
        self.ax.plot(self.start_pos[0], self.start_pos[1], marker='o', color='yellow', markersize=14, zorder=5)
        self.ax.plot(self.start_pos[0], self.start_pos[1], marker='+', color='black', markersize=14, zorder=6)

        # Draw Goal
        goal_color = 'lime' if self.algo.goal_reached else 'red'
        goal_size = 14 if self.algo.goal_reached else 6
        self.ax.plot(self.goal_pos[0], self.goal_pos[1], marker='o', color=goal_color, markersize=goal_size, zorder=5)
        if self.algo.goal_reached:
            self.ax.plot(self.goal_pos[0], self.goal_pos[1], marker='+', color='black', markersize=14, zorder=6)

        # Draw Final Paths
        if self.algo.goal_reached and show_prm:
            # A* Optimized Path (Bold Green)
            if self.algo.astar_path:
                ax_p = [p[0] for p in self.algo.astar_path]
                ay_p = [p[1] for p in self.algo.astar_path]
                self.ax.plot(ax_p, ay_p, color="lime", linewidth=4, label="A* Path", zorder=4)

            # Title with Path Length
            self.ax.set_title(
                f"{len(self.algo.node_list)} Nodes, Goal Reached! (Path Length: {self.algo.path_length:.2f})",
                fontsize=14)
        else:
            self.ax.set_title(f"{len(self.algo.node_list)} Nodes, Goal Not Yet Reached", fontsize=14)

        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()