import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Node:
    """A node class for RRT* Pathfinding"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0  # NEW: Tracks the exact distance traveled from the Start node


class RRTStar:
    """
    Class for RRT* planning
    """

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=0.5, goal_sample_rate=10, max_iter=2000,
                 connect_circle_dist=1.5):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.min_rand, self.max_rand = rand_area
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter

        # NEW: The radius RRT* searches to find neighbors for optimizing paths
        self.connect_circle_dist = connect_circle_dist
        self.node_list = []

    def plan(self, animation=True):
        self.node_list = [self.start]

        for i in range(self.max_iter):
            # 1. Generate random node
            rnd_node = self.get_random_node()

            # 2. Find nearest node and steer
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 3. If no collision, proceed with RRT* optimization
            if self.check_collision(nearest_node, new_node, self.obstacle_list):

                # --- RRT* UPGRADE 1: Find best parent ---
                near_indices = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indices)

                if new_node:
                    self.node_list.append(new_node)
                    # --- RRT* UPGRADE 2: Rewire the tree ---
                    self.rewire(new_node, near_indices)

        # 4. Search for the best path to the goal after all iterations
        best_goal_ind = self.search_best_goal_node()
        if best_goal_ind is not None:
            return self.generate_final_course(best_goal_ind)

        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.x += min(extend_length, d) * math.cos(theta)
        new_node.y += min(extend_length, d) * math.sin(theta)

        # Tentatively set cost and parent
        new_node.cost = from_node.cost + min(extend_length, d)
        new_node.parent = from_node
        return new_node

    def find_near_nodes(self, new_node):
        """Finds all nodes within the search radius of the new node."""
        nnode = len(self.node_list) + 1
        # Dynamic search radius (optional, using fixed radius for simplicity here if preferred, but dynamic scales better)
        r = min(self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode)), self.expand_dis * 3.0)

        dist_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= self.connect_circle_dist ** 2]
        return near_inds

    def choose_parent(self, new_node, near_indices):
        """Evaluates neighbors to find the one that offers the cheapest path to the new node."""
        if not near_indices:
            return new_node

        costs = []
        for i in near_indices:
            near_node = self.node_list[i]
            d, theta = self.calc_distance_and_angle(near_node, new_node)

            if self.check_collision(near_node, new_node, self.obstacle_list):
                costs.append(near_node.cost + d)
            else:
                costs.append(float("inf"))  # Path is blocked

        min_cost = min(costs)
        if min_cost == float("inf"):
            return None  # No safe path found

        min_ind = near_indices[costs.index(min_cost)]
        new_node.cost = min_cost
        new_node.parent = self.node_list[min_ind]

        return new_node

    def rewire(self, new_node, near_indices):
        """Checks if routing existing neighbors through the new node makes their paths shorter."""
        for i in near_indices:
            near_node = self.node_list[i]
            d, theta = self.calc_distance_and_angle(new_node, near_node)

            new_cost = new_node.cost + d

            # If the route through the new node is cheaper, and safe...
            if near_node.cost > new_cost:
                if self.check_collision(new_node, near_node, self.obstacle_list):
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    self.propagate_cost_to_leaves(near_node)

    def propagate_cost_to_leaves(self, parent_node):
        """Recursively updates the cost of all children when a parent is rewired."""
        for node in self.node_list:
            if node.parent == parent_node:
                d, _ = self.calc_distance_and_angle(parent_node, node)
                node.cost = parent_node.cost + d
                self.propagate_cost_to_leaves(node)

    def search_best_goal_node(self):
        """Finds the node closest to the goal that has the absolute lowest cost."""
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_indices = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]

        if not goal_indices:
            return None

        # RRT* doesn't just stop at the first node near the goal; it checks which one has the best cost
        min_cost = min([self.node_list[i].cost for i in goal_indices])
        for i in goal_indices:
            if self.node_list[i].cost == min_cost:
                return i
        return None

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            return Node(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand))
        return Node(self.end.x, self.end.y)

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        return dlist.index(min(dlist))

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
        return math.hypot(x - self.end.x, y - self.end.y)

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

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
    print("Starting RRT* path planning (this may take a few seconds due to rewiring)...")

    start_pos = (1.0, 1.0)
    goal_pos = (9.0, 9.0)

    # Maze Obstacles
    obstacle_list = [
        (3.0, 0.0, 1.0, 7.0),
        (6.0, 3.0, 2.5, 7.0),
        (3.0, 7.0, 2.0, 1.0)
    ]

    # Initialize RRT*
    rrt_star = RRTStar(
        start=start_pos,
        goal=goal_pos,
        rand_area=[0, 10],
        obstacle_list=obstacle_list,
        expand_dis=0.5,
        goal_sample_rate=10,
        max_iter=1500,  # RRT* needs high iterations to optimize
        connect_circle_dist=2.0  # The "look-around" radius for rewiring
    )

    path = rrt_star.plan()

    # --- Plotting the Results ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot rectangular obstacles
    for (ox, oy, w, h) in obstacle_list:
        rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='gray', zorder=2)
        ax.add_patch(rect)

    # Plot the RRT* Tree
    for node in rrt_star.node_list:
        if node.parent:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g", alpha=0.3, zorder=1)

    # Plot the final path
    if path is None:
        print("Cannot find path.")
    else:
        print(f"Optimal path found! Path length: {len(path)} nodes.")
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, "-r", linewidth=4, label="RRT* Optimal Path", zorder=3)

    # Plot Start and Goal
    ax.plot(start_pos[0], start_pos[1], "^b", markersize=10, label="Start", zorder=4)
    ax.plot(goal_pos[0], goal_pos[1], "*b", markersize=15, label="Goal", zorder=4)

    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Rapidly-exploring Random Tree Star (RRT*)")
    ax.legend(loc="upper left")

    plt.show()


if __name__ == '__main__':
    main()