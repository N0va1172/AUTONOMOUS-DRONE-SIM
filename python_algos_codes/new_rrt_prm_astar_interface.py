import math
import random
import heapq
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        # Using a Set ensures we never add duplicate edges
        self.neighbors = set()


class IncrementalHybridAlgorithm:
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=0.5, connection_radius=1.5):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.min_rand, self.max_rand = rand_area
        self.expand_dis = expand_dis
        self.connection_radius = connection_radius

        self.node_list = [self.start]
        self.goal_reached = False
        self.goal_node = None

        self.astar_path = None
        self.path_length = 0.0

    def steer(self, from_node, to_node, extend_length):
        new_node = Node(from_node.x, from_node.y)
        d = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node.x += min(extend_length, d) * math.cos(theta)
        new_node.y += min(extend_length, d) * math.sin(theta)
        new_node.parent = from_node
        return new_node

    def check_collision(self, n1, n2, obstacle_list):
        steps = 10
        for i in range(steps + 1):
            x = n1.x + (n2.x - n1.x) * (i / steps)
            y = n1.y + (n2.y - n1.y) * (i / steps)
            for (ox, oy, w, h) in obstacle_list:
                if ox <= x <= (ox + w) and oy <= y <= (oy + h):
                    return False
        return True

    def add_single_node(self, bias_percentage):
        """Attempts to add a single node. Returns True if a VALID node was added."""
        if random.randint(0, 100) > bias_percentage:
            rnd = Node(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = Node(self.end.x, self.end.y)

        dlist = [(node.x - rnd.x) ** 2 + (node.y - rnd.y) ** 2 for node in self.node_list]
        nearest = self.node_list[dlist.index(min(dlist))]
        new_node = self.steer(nearest, rnd, self.expand_dis)

        if self.check_collision(nearest, new_node, self.obstacle_list):
            self.node_list.append(new_node)

            # 1. Add Original RRT Connection
            new_node.neighbors.add(nearest)
            nearest.neighbors.add(new_node)

            # 2. Add PRM Radius Connections (Cumulative)
            for existing_node in self.node_list[:-1]:  # Don't check against itself
                if math.hypot(existing_node.x - new_node.x, existing_node.y - new_node.y) <= self.connection_radius:
                    if self.check_collision(existing_node, new_node, self.obstacle_list):
                        existing_node.neighbors.add(new_node)
                        new_node.neighbors.add(existing_node)

            # 3. Check Goal
            if not self.goal_reached and math.hypot(new_node.x - self.end.x,
                                                    new_node.y - self.end.y) <= self.expand_dis:
                final_node = self.steer(new_node, self.end, self.expand_dis)
                if self.check_collision(new_node, final_node, self.obstacle_list):
                    self.node_list.append(final_node)

                    # Connect final node to its parent
                    final_node.neighbors.add(new_node)
                    new_node.neighbors.add(final_node)

                    self.goal_reached = True
                    self.goal_node = final_node

            return True  # Node was valid and added
        return False  # Node hit an obstacle

    def run_astar(self):
        """Runs A* on the cumulative neighbor web to find the shortest path."""
        start_node = self.node_list[0]
        open_set = []
        heapq.heappush(open_set, (0, id(start_node), start_node))
        came_from = {}
        g_score = {start_node: 0}

        def heuristic(n1, n2):
            return math.hypot(n1.x - n2.x, n1.y - n2.y)

        while open_set:
            current = heapq.heappop(open_set)[2]

            if current == self.goal_node:
                self.astar_path = []
                while current in came_from:
                    self.astar_path.append([current.x, current.y])
                    current = came_from[current]
                self.astar_path.append([start_node.x, start_node.y])

                # Calculate True Length of the Path
                self.path_length = 0.0
                for i in range(len(self.astar_path) - 1):
                    self.path_length += math.hypot(self.astar_path[i + 1][0] - self.astar_path[i][0],
                                                   self.astar_path[i + 1][1] - self.astar_path[i][1])
                break

            for neighbor in current.neighbors:
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, self.goal_node)
                    heapq.heappush(open_set, (f_score, id(neighbor), neighbor))


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("My Algorithm - Interactive Planner")
        self.root.geometry("950x700")

        self.start_pos = (5.0, 5.0)
        self.goal_pos = (5.0, 1.0)
        self.area = [0, 10]

        self.obstacles = {
            "Maze": [
                (2.0, 2.0, 6.0, 1.0),
                (2.0, 2.0, 1.0, 6.0),
                (2.0, 7.0, 4.0, 1.0),
                (7.0, 4.0, 1.0, 4.0)
            ],
            "Concave": [
                (3.0, 3.0, 1.0, 5.0),
                (3.0, 3.0, 5.0, 1.0),
                (7.0, 3.0, 1.0, 5.0)
            ],
            "None": []
        }

        self.algo = None
        self.setup_ui()
        self.reset_algorithm()

    def setup_ui(self):
        control_frame = tk.Frame(self.root, width=250, padx=15, pady=15, bg="#f0f0f0")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="Algorithm", bg="#f0f0f0", font=("Arial", 9)).pack(anchor="w")
        tk.Label(control_frame, text="My Algorithm (Radius PRM)", bg="#e0e0ff", relief="groove", padx=5, pady=3,
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 15))

        tk.Label(control_frame, text="Obstacle Type:", bg="#f0f0f0").pack(anchor="w")
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

        tk.Label(control_frame, text="Exploration Bias:", bg="#f0f0f0").pack(anchor="w")
        self.bias_slider = tk.Scale(control_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                                    bg="#f0f0f0", highlightthickness=0)
        self.bias_slider.set(0.10)
        self.bias_slider.pack(anchor="w", fill=tk.X, pady=(0, 15))

        ttk.Button(control_frame, text="Clear & Reset", command=self.reset_algorithm).pack(anchor="w", pady=20)

        self.canvas_frame = tk.Frame(self.root, bg="white")
        self.canvas_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def reset_algorithm(self):
        obs_type = self.obs_var.get()
        self.algo = IncrementalHybridAlgorithm(self.start_pos, self.goal_pos, self.obstacles[obs_type], self.area)
        self.update_plot()

    def step_algo(self, num_nodes):
        bias_percentage = int(self.bias_slider.get() * 100)
        added = 0

        while added < num_nodes:
            if self.algo.add_single_node(bias_percentage):
                added += 1
                # # Fast Animation: Update UI every 5 valid nodes
                # if added % 20 == 0 or added == num_nodes:
                #     self.update_plot()
                #     self.root.update()

        if self.algo.goal_reached:
            self.algo.run_astar()
        self.update_plot()
        self.root.update()

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlim(self.area[0], self.area[1])
        self.ax.set_ylim(self.area[0], self.area[1])
        self.ax.set_aspect('equal')
        self.ax.set_xticks([]);
        self.ax.set_yticks([])

        self.ax.add_patch(patches.Rectangle((0, 0), 10, 10, linewidth=6, edgecolor='blue', facecolor='none'))

        # 1. Draw Obstacles (Grey)
        for (ox, oy, w, h) in self.obstacles[self.obs_var.get()]:
            self.ax.add_patch(patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='grey'))

        # 2. Draw PRM Web (Cyan) & RRT Edges (Black)
        drawn_edges = set()
        for node in self.algo.node_list:
            for neighbor in node.neighbors:
                edge = tuple(sorted([id(node), id(neighbor)]))
                if edge not in drawn_edges:
                    # If this is the RRT parent/child link, draw it black. Otherwise, cyan.
                    if neighbor == node.parent or node == neighbor.parent:
                        self.ax.plot([node.x, neighbor.x], [node.y, neighbor.y], color="black", linewidth=1.0, zorder=2)
                    else:
                        self.ax.plot([node.x, neighbor.x], [node.y, neighbor.y], color="cyan", linewidth=0.8, alpha=0.4,
                                     zorder=1)
                    drawn_edges.add(edge)

        # 3. Draw Nodes (Red)
        self.ax.scatter([n.x for n in self.algo.node_list], [n.y for n in self.algo.node_list], color='red', s=15,
                        zorder=3)

        # 4. Draw Start & Goal
        self.ax.plot(self.start_pos[0], self.start_pos[1], marker='o', color='yellow', markersize=14, zorder=5)
        self.ax.plot(self.start_pos[0], self.start_pos[1], marker='+', color='black', markersize=14, zorder=6)

        goal_color = 'lime' if self.algo.goal_reached else 'red'
        self.ax.plot(self.goal_pos[0], self.goal_pos[1], marker='o', color=goal_color,
                     markersize=14 if self.algo.goal_reached else 6, zorder=5)
        if self.algo.goal_reached:
            self.ax.plot(self.goal_pos[0], self.goal_pos[1], marker='+', color='black', markersize=14, zorder=6)

        # 5. Draw Final Path (Lime Green)
        if self.algo.goal_reached and self.algo.astar_path:
            ax_p = [p[0] for p in self.algo.astar_path]
            ay_p = [p[1] for p in self.algo.astar_path]
            self.ax.plot(ax_p, ay_p, color="lime", linewidth=4, label="A* Path", zorder=4)
            self.ax.set_title(
                f"{len(self.algo.node_list)} Nodes | Goal Reached! | Path Length: {self.algo.path_length:.2f}",
                fontsize=14)
        else:
            self.ax.set_title(f"{len(self.algo.node_list)} Nodes | Goal Not Yet Reached", fontsize=14)

        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()