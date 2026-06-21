import math
import random
import heapq

# =====================================================================
# GPS TO METERS TRANSLATOR
# =====================================================================
class CoordinateMapper:
    def __init__(self, origin_lat, origin_lon):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.earth_radius = 6378137.0
        self.lon_scale = math.cos(math.radians(origin_lat))

    def local_to_gps(self, x, y):
        """Converts Local Grid (X, Y) unrounded floats -> Global GPS (Lat, Lon)."""
        delta_lat = y / self.earth_radius
        delta_lon = x / (self.earth_radius * self.lon_scale)
        target_lat = self.origin_lat + math.degrees(delta_lat)
        target_lon = self.origin_lon + math.degrees(delta_lon)

        # ROUNDING HAPPENS ONLY HERE: 7 Decimal places max for ArduPilot
        return round(target_lat, 7), round(target_lon, 7)


# =====================================================================
# PART 1: PURE ALGORITHM (No Interface, No Plotting)
# =====================================================================
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.neighbors = set()


class NewAlgo:
    def __init__(self, start, goal, obstacle_list, x_bounds, y_bounds, circle_list=None, expand_dis=0.5,
                 connection_radius=1.5):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.circle_list = circle_list if circle_list else []

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        self.expand_dis = expand_dis
        self.connection_radius = connection_radius

        self.node_list = [self.start]
        self.goal_reached = False
        self.goal_node = None

        self.astar_path = []
        self.path_length = 0.0

    def steer(self, from_node, to_node, extend_length):
        new_node = Node(from_node.x, from_node.y)
        d = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node.x += min(extend_length, d) * math.cos(theta)
        new_node.y += min(extend_length, d) * math.sin(theta)
        new_node.parent = from_node
        return new_node

    def check_collision(self, n1, n2):
        steps = 10
        for i in range(steps + 1):
            x = n1.x + (n2.x - n1.x) * (i / steps)
            y = n1.y + (n2.y - n1.y) * (i / steps)

            # 1. Check Rectangular Obstacles (x, y, width, height)
            for (ox, oy, w, h) in self.obstacle_list:
                if ox <= x <= (ox + w) and oy <= y <= (oy + h):
                    return False

            # 2. Check Circular Obstacles (center_x, center_y, radius)
            for (cx, cy, r) in self.circle_list:
                if math.hypot(x - cx, y - cy) <= r:
                    return False

        return True

    def add_single_node(self, bias_percentage=10):
        if random.randint(0, 100) > bias_percentage:
            rnd_x = random.uniform(self.x_bounds[0], self.x_bounds[1])
            rnd_y = random.uniform(self.y_bounds[0], self.y_bounds[1])
            rnd = Node(rnd_x, rnd_y)
        else:
            rnd = Node(self.end.x, self.end.y)

        dlist = [(node.x - rnd.x) ** 2 + (node.y - rnd.y) ** 2 for node in self.node_list]
        nearest = self.node_list[dlist.index(min(dlist))]
        new_node = self.steer(nearest, rnd, self.expand_dis)

        if self.check_collision(nearest, new_node):
            self.node_list.append(new_node)

            # RRT Connection
            new_node.neighbors.add(nearest)
            nearest.neighbors.add(new_node)

            # PRM Phase
            for existing_node in self.node_list[:-1]:
                if math.hypot(existing_node.x - new_node.x, existing_node.y - new_node.y) <= self.connection_radius:
                    if self.check_collision(existing_node, new_node):
                        existing_node.neighbors.add(new_node)
                        new_node.neighbors.add(existing_node)

            # Check Goal
            if not self.goal_reached and math.hypot(new_node.x - self.end.x,
                                                    new_node.y - self.end.y) <= self.expand_dis:
                final_node = self.steer(new_node, self.end, self.expand_dis)
                if self.check_collision(new_node, final_node):
                    self.node_list.append(final_node)
                    final_node.neighbors.add(new_node)
                    new_node.neighbors.add(final_node)
                    self.goal_reached = True
                    self.goal_node = final_node
            return True
        return False

    def plan(self, max_iterations=5000):
        iterations = 0
        while not self.goal_reached and iterations < max_iterations:
            self.add_single_node(bias_percentage=10)
            iterations += 1

        if self.goal_reached:
            return self.run_astar()
        return None

    def run_astar(self):
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

                # GUARANTEED NO ROUNDING: Pure floats appended to the array
                while current in came_from:
                    self.astar_path.append([current.x, current.y])
                    current = came_from[current]
                self.astar_path.append([start_node.x, start_node.y])
                self.astar_path.reverse()

                for i in range(len(self.astar_path) - 1):
                    self.path_length += math.hypot(self.astar_path[i + 1][0] - self.astar_path[i][0],
                                                   self.astar_path[i + 1][1] - self.astar_path[i][1])
                return self.astar_path

            for neighbor in current.neighbors:
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, self.goal_node)
                    heapq.heappush(open_set, (f_score, id(neighbor), neighbor))
        return None


if __name__ == "__main__":

    # Initialize the GPS Mapper using Point 1 of your physical boundary
    ORIGIN_LAT = 12.923350
    ORIGIN_LON = 77.501190
    mapper = CoordinateMapper(ORIGIN_LAT, ORIGIN_LON)

    X_BOUNDS = [0.0, 16.3]
    Y_BOUNDS = [0.0, 13.1]
    STEP_SIZE = 0.5

    START_POINT = (1.0, 1.0)
    GOAL_POINT = (14.0, 11.0)

    RECT_OBSTACLES = [
        (4.0, 0.0, 1.0, 8.0),
        (10.0, 5.0, 1.0, 8.1)
    ]

    CIRC_OBSTACLES = [
        (7.0, 10.0, 1.0),
        (13.0, 3.0, 1.5)
    ]

    planner = NewAlgo(
        start=START_POINT,
        goal=GOAL_POINT,
        obstacle_list=RECT_OBSTACLES,
        circle_list=CIRC_OBSTACLES,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        expand_dis=STEP_SIZE,
        connection_radius=1.5
    )

    mi = int(2 * abs((X_BOUNDS[1] - X_BOUNDS[0])) * abs((Y_BOUNDS[1] - Y_BOUNDS[0])) * (1 / STEP_SIZE))
    final_coordinate_list = planner.plan(max_iterations=mi)

    if final_coordinate_list:
        print("\nSUCCESS! Found a path.")

        # Output 1: Local unrounded array
        print("\n--- LOCAL COORDINATES (Unrounded) ---")
        print("local_path_array =", final_coordinate_list)

        # Output 2: GPS Array mapped and rounded to 7 places
        gps_coordinate_list = []
        for point in final_coordinate_list:
            gps_lat, gps_lon = mapper.local_to_gps(point[0], point[1])
            gps_coordinate_list.append([gps_lat, gps_lon])

        print("\n--- GPS COORDINATES (Rounded to 7 decimal places) ---")
        print("gps_path_array =", gps_coordinate_list)

        print(f"\nTotal Distance: {planner.path_length:.2f} meters")

    else:
        print("\nFAILED: Could not find a path to the goal within the iteration limit.")


    def plot_results(algo, x_bounds, y_bounds, obstacles, circles):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        x_range = x_bounds[1] - x_bounds[0]
        y_range = y_bounds[1] - y_bounds[0]
        fig, ax = plt.subplots(figsize=(8 * (x_range / max(x_range, y_range)), 8 * (y_range / max(x_range, y_range))))

        # Add a buffer to the plot limits so we can see the boundary lines clearly
        ax.set_xlim(x_bounds[0] - 1, x_bounds[1] + 1)
        ax.set_ylim(y_bounds[0] - 1, y_bounds[1] + 1)
        ax.set_aspect('equal')

        # 1. Draw the Flight Boundary Box (Thick Blue Rectangle)
        boundary = patches.Rectangle((x_bounds[0], y_bounds[0]), x_range, y_range,
                                     linewidth=3, edgecolor='blue', facecolor='none', zorder=10,
                                     label="Flight Boundary")
        ax.add_patch(boundary)

        # 2. Draw Rectangular Obstacles (Grey)
        for (ox, oy, w, h) in obstacles:
            ax.add_patch(patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='grey'))

        # 3. Draw Circular Obstacles (Dark Red)
        for (cx, cy, r) in circles:
            ax.add_patch(patches.Circle((cx, cy), r, linewidth=1, edgecolor='black', facecolor='darkred'))

        # Draw Nodes and Edges
        drawn_edges = set()
        for node in algo.node_list:
            for neighbor in node.neighbors:
                edge = tuple(sorted([id(node), id(neighbor)]))
                if edge not in drawn_edges:
                    if neighbor == node.parent or node == neighbor.parent:
                        ax.plot([node.x, neighbor.x], [node.y, neighbor.y], color="black", linewidth=1.0, zorder=2)
                    else:
                        ax.plot([node.x, neighbor.x], [node.y, neighbor.y], color="cyan", linewidth=0.8, alpha=0.4,
                                zorder=1)
                    drawn_edges.add(edge)
            ax.scatter(node.x, node.y, color='red', s=10, zorder=3)

        # Highlight Start and Goal
        ax.plot(algo.start.x, algo.start.y, marker='^', color='blue', markersize=12, label="Start", zorder=5)
        ax.plot(algo.end.x, algo.end.y, marker='*', color='gold', markeredgecolor='black', markersize=18, label="Goal",
                zorder=5)

        # Draw Shortest Path
        if algo.astar_path:
            px = [point[0] for point in algo.astar_path]
            py = [point[1] for point in algo.astar_path]
            ax.plot(px, py, color="lime", linewidth=4, label=f"Shortest Path ({algo.path_length:.2f}m)", zorder=4)

        ax.legend(loc="upper left")
        ax.set_title("Hybrid RRT + PRM Navigation Map")
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()


    if final_coordinate_list:
        plot_results(planner, X_BOUNDS, Y_BOUNDS, RECT_OBSTACLES, CIRC_OBSTACLES)

    print()