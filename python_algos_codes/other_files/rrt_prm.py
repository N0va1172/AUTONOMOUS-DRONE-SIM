import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import KDTree  # --- NEW: Imported for fast neighbor search ---


class Node:
    """A node class for RRT Pathfinding"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.neighbors = []  # --- NEW: List to store K-nearest connections ---


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
        """
        Plans the path from start to goal.
        Returns a list of (x, y) coordinates for the path.
        """
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # 1. Generate a random node
            rnd_node = self.get_random_node()

            # 2. Find the nearest existing node in the tree
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # 3. Steer from the nearest node toward the random node
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 4. Check if the new path hits an obstacle
            if self.check_collision(nearest_node, new_node, self.obstacle_list):
                self.node_list.append(new_node)

            # 5. Check if we are close enough to the goal
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(self.node_list[-1], final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # Cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """Moves a set distance from the current node toward the target node."""
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.x += min(extend_length, d) * math.cos(theta)
        new_node.y += min(extend_length, d) * math.sin(theta)
        new_node.parent = from_node

        return new_node

    def get_random_node(self):
        """Samples a random point in space, occasionally sampling the goal."""
        self.random_node_count += 1

        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(random.uniform(self.min_rand, self.max_rand),
                       random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = Node(self.end.x, self.end.y)
        return rnd

    def get_nearest_node_index(self, node_list, rnd_node):
        """Finds the index of the closest node in the tree."""
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def check_collision(self, from_node, to_node, obstacle_list):
        """Checks if the line segment intersects any rectangular obstacles."""
        steps = 10  # Number of points to sample along the line segment
        for i in range(steps + 1):
            x = from_node.x + (to_node.x - from_node.x) * (i / steps)
            y = from_node.y + (to_node.y - from_node.y) * (i / steps)

            # Check if this interpolated point falls inside any rectangle
            for (ox, oy, w, h) in obstacle_list:
                if ox <= x <= (ox + w) and oy <= y <= (oy + h):
                    return False  # Collision detected
        return True  # Safe

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
        """Traces back the parents to construct the final path."""
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

    # Start and Goal positions
    start_pos = (1.0, 1.0)
    goal_pos = (9.0, 9.0)

    # Maze Obstacles as (x, y, width, height)
    obstacle_list = [
        (3.0, 0.0, 1.0, 7.0),
        (6.0, 3.0, 2.5, 7.0),
        (3.0, 7.0, 2.0, 1.0)
    ]

    # Initialize RRT
    rrt = RRT(
        start=start_pos,
        goal=goal_pos,
        rand_area=[0, 10],
        obstacle_list=obstacle_list,
        expand_dis=0.5,
        goal_sample_rate=5,
        max_iter=3000
    )

    # Run the algorithm
    path = rrt.plan()

    # --- Print the requested counter ---
    print("-" * 30)
    print(f"Total random nodes generated: {rrt.random_node_count}")
    print(f"Total nodes successfully added to tree: {len(rrt.node_list)}")
    if path:
        print(f"Number of nodes in path: {len(path)}")
    print("-" * 30)

    # ==============================================================
    # --- NEW: Connect valid nodes to 10 nearest neighbors (PRM) ---
    # ==============================================================
    print("Building Probabilistic Map connections...")

    # Extract coordinates for KDTree
    node_coords = [[n.x, n.y] for n in rrt.node_list]
    tree = KDTree(node_coords)
    k_neighbors = 10

    # Ensure we don't request more neighbors than nodes exist
    k_query = min(k_neighbors + 1, len(rrt.node_list))

    for node in rrt.node_list:
        # Query nearest neighbors (includes the node itself)
        distances, indices = tree.query([node.x, node.y], k=k_query)

        # Iterate over neighbors (skip index 0, which is the node itself)
        if k_query > 1:
            for idx in indices[1:]:
                neighbor = rrt.node_list[idx]

                # Check if connection already exists
                if neighbor not in node.neighbors:
                    # Check for collisions using RRT's built-in collision method
                    if rrt.check_collision(node, neighbor, obstacle_list):
                        # Create bidirectional edge
                        node.neighbors.append(neighbor)
                        neighbor.neighbors.append(node)

    # ==============================================================
    # --- Plotting the Results (2 Subplots) ---
    # ==============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Common Elements for both plots ---
    for ax in [ax1, ax2]:
        for (ox, oy, w, h) in obstacle_list:
            rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='gray')
            ax.add_patch(rect)

        # Plot Start and Goal
        ax.plot(start_pos[0], start_pos[1], "^b", markersize=10, label="Start", zorder=5)
        ax.plot(goal_pos[0], goal_pos[1], "*b", markersize=15, label="Goal", zorder=5)

        ax.grid(True)
        ax.set_aspect("equal")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    # --- Plot 1: RRT Valid Nodes and Tree ---
    ax1.set_title("1. Valid Nodes (Original RRT functionality)")
    ax1.scatter([n.x for n in rrt.node_list], [n.y for n in rrt.node_list], color='blue', s=10, zorder=3)

    # Draw original RRT parent-child connections
    for node in rrt.node_list:
        if node.parent:
            ax1.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g", alpha=0.3, zorder=2)

    # Draw final solution path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax1.plot(path_x, path_y, "-r", linewidth=3, label="Final Path", zorder=4)
    ax1.legend()

    # --- Plot 2: Probabilistic Map (Valid Nodes + 10-NN Edges) ---
    ax2.set_title("2. Probabilistic Map (Nodes & Connections)")
    ax2.scatter([n.x for n in rrt.node_list], [n.y for n in rrt.node_list], color='blue', s=10, zorder=3)

    # Draw edges, using a set to avoid drawing bidirectional lines twice
    drawn_edges = set()
    for node in rrt.node_list:
        for neighbor in node.neighbors:
            edge = tuple(sorted([id(node), id(neighbor)]))
            if edge not in drawn_edges:
                ax2.plot([node.x, neighbor.x], [node.y, neighbor.y], "-g", alpha=0.2, linewidth=1, zorder=2)
                drawn_edges.add(edge)

    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()