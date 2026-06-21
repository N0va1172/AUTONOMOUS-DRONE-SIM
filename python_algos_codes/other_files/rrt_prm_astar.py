import math
import random
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import KDTree


class Node:
    """A node class for RRT Pathfinding"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.neighbors = []  # List to store ALL valid connections (Graph Edges)


class RRT:
    """
    Class for RRT planning
    """

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=1.0, goal_sample_rate=10, max_iter=2000):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list  # List of (x, y, width, height)
        self.min_rand, self.max_rand = rand_area
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []

        # Counter for random nodes
        self.random_node_count = 0

    def plan(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(nearest_node, new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(self.node_list[-1], final_node, self.obstacle_list):
                    # Append the final goal node so it is part of the valid nodes list for A*
                    self.node_list.append(final_node)
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        new_node.x += min(extend_length, d) * math.cos(theta)
        new_node.y += min(extend_length, d) * math.sin(theta)
        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        self.random_node_count += 1
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(random.uniform(self.min_rand, self.max_rand),
                       random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = Node(self.end.x, self.end.y)
        return rnd

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def check_collision(self, from_node, to_node, obstacle_list):
        steps = 10
        for i in range(steps + 1):
            x = from_node.x + (to_node.x - from_node.x) * (i / steps)
            y = from_node.y + (to_node.y - from_node.y) * (i / steps)
            for (ox, oy, w, h) in obstacle_list:
                if ox <= x <= (ox + w) and oy <= y <= (oy + h):
                    return False
        return True

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path


# --- Example Usage & Visualization ---
def main():
    print("Starting RRT path planning...")

    start_pos = (1.0, 1.0)
    goal_pos = (9.0, 9.0)

    obstacle_list = [
        (3.0, 0.0, 1.0, 7.0),
        (6.0, 3.0, 2.5, 7.0),
        (3.0, 7.0, 2.0, 1.0)
    ]

    rrt = RRT(
        start=start_pos,
        goal=goal_pos,
        rand_area=[0, 10],
        obstacle_list=obstacle_list,
        expand_dis=0.5,
        goal_sample_rate=5,
        max_iter=3000
    )

    rrt_path = rrt.plan()

    if not rrt_path:
        print("Cannot find path. Reached max iterations.")
        return

    # ==============================================================
    # --- PHASE 2: Build the Hybrid Graph (RRT Edges + PRM Edges) ---
    # ==============================================================
    print("Building Probabilistic Map connections...")

    # 1. First, explicitly convert the RRT Tree into bidirectional graph edges
    for node in rrt.node_list:
        if node.parent:
            if node.parent not in node.neighbors:
                node.neighbors.append(node.parent)
            if node not in node.parent.neighbors:
                node.parent.neighbors.append(node)

    # 2. Add the 10-Nearest Neighbors using KDTree
    node_coords = [[n.x, n.y] for n in rrt.node_list]
    tree = KDTree(node_coords)
    k_neighbors = 10
    k_query = min(k_neighbors + 1, len(rrt.node_list))

    for node in rrt.node_list:
        distances, indices = tree.query([node.x, node.y], k=k_query)
        if k_query > 1:
            for idx in indices[1:]:
                neighbor = rrt.node_list[idx]
                if neighbor not in node.neighbors:
                    if rrt.check_collision(node, neighbor, obstacle_list):
                        node.neighbors.append(neighbor)
                        neighbor.neighbors.append(node)

    # ==============================================================
    # --- PHASE 3: Apply A* Algorithm over the Graph ---
    # ==============================================================
    print("Running A* on the generated web...")

    start_node = rrt.node_list[0]
    goal_node = rrt.node_list[-1]  # We appended final_node earlier, so this is the goal

    open_set = []
    # heapq needs a tie-breaker if f_scores are equal. id(node) works perfectly.
    heapq.heappush(open_set, (0, id(start_node), start_node))

    came_from = {}
    g_score = {start_node: 0}

    def heuristic(n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    f_score = {start_node: heuristic(start_node, goal_node)}

    astar_path = None

    while open_set:
        current = heapq.heappop(open_set)[2]

        if current == goal_node:
            # Reconstruct A* path
            astar_path = []
            while current in came_from:
                astar_path.append([current.x, current.y])
                current = came_from[current]
            astar_path.append([start_node.x, start_node.y])
            break

        for neighbor in current.neighbors:
            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))

    # --- Print Data ---
    print("-" * 40)
    print(f"Total random nodes generated: {rrt.random_node_count}")
    print(f"Total valid nodes in Graph: {len(rrt.node_list)}")
    print(f"Number of nodes in RRT path: {len(rrt_path)}")
    if astar_path:
        print(f"Number of nodes in A* path: {len(astar_path)}")
    print("-" * 40)

    # ==============================================================
    # --- PHASE 4: Plotting the Results ---
    # ==============================================================
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw Obstacles
    for (ox, oy, w, h) in obstacle_list:
        rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rect)

    ax.plot(start_pos[0], start_pos[1], "^b", markersize=10, label="Start", zorder=5)
    ax.plot(goal_pos[0], goal_pos[1], "*b", markersize=15, label="Goal", zorder=5)

    # Plot All Nodes and Faint PRM Edges
    ax.scatter([n.x for n in rrt.node_list], [n.y for n in rrt.node_list], color='blue', s=10, zorder=3)
    drawn_edges = set()
    for node in rrt.node_list:
        for neighbor in node.neighbors:
            edge = tuple(sorted([id(node), id(neighbor)]))
            if edge not in drawn_edges:
                ax.plot([node.x, neighbor.x], [node.y, neighbor.y], color="green", alpha=0.1, linewidth=1, zorder=1)
                drawn_edges.add(edge)

    # Plot Original RRT Path (Solid Red)
    rrt_x = [p[0] for p in rrt_path]
    rrt_y = [p[1] for p in rrt_path]
    ax.plot(rrt_x, rrt_y, color="red", linewidth=4, alpha=0.7, label="Original RRT Path")

    # Plot A* Optimized Path (Dashed Cyan)
    if astar_path:
        astar_x = [p[0] for p in astar_path]
        astar_y = [p[1] for p in astar_path]
        ax.plot(astar_x, astar_y, color="cyan", linewidth=3, linestyle="--", label="A* Optimized Path")

    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Hybrid RRT/PRM Pathfinding (RRT vs. A* Comparison)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()