import math
import random
import heapq
import time
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import KDTree


def calculate_path_length(path):
    if not path or len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(len(path) - 1):
        length += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
    return length


def run_grid_search(start_pos, goal_pos, obstacle_list, bounds, resolution, use_astar=True):
    start_time = time.time()

    movements = [
        (resolution, 0), (-resolution, 0), (0, resolution), (0, -resolution),
        (resolution, resolution), (resolution, -resolution), (-resolution, resolution), (-resolution, -resolution)
    ]

    def is_collision_free(p1, p2):
        steps = 10
        for i in range(steps + 1):
            x = p1[0] + (p2[0] - p1[0]) * (i / steps)
            y = p1[1] + (p2[1] - p1[1]) * (i / steps)
            for (ox, oy, w, h) in obstacle_list:
                if ox <= x <= (ox + w) and oy <= y <= (oy + h):
                    return False
        return True

    def heuristic(p):
        if use_astar:
            return math.hypot(p[0] - goal_pos[0], p[1] - goal_pos[1])
        return 0

    open_set = []
    heapq.heappush(open_set, (0, start_pos))
    came_from = {}
    g_score = {start_pos: 0}
    nodes_explored = 0
    final_path = None

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_explored += 1

        if math.hypot(current[0] - goal_pos[0], current[1] - goal_pos[1]) < 1e-4:
            final_path = []
            curr = current
            while curr in came_from:
                final_path.append([curr[0], curr[1]])
                curr = came_from[curr]
            final_path.append([start_pos[0], start_pos[1]])
            break

        for dx, dy in movements:
            neighbor = (round(current[0] + dx, 3), round(current[1] + dy, 3))
            if not (bounds[0] <= neighbor[0] <= bounds[1] and bounds[0] <= neighbor[1] <= bounds[1]):
                continue
            if not is_collision_free(current, neighbor):
                continue

            tentative_g_score = g_score[current] + math.hypot(dx, dy)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, neighbor))

    end_time = time.time()
    return final_path, nodes_explored, (end_time - start_time)


# ==============================================================
# --- ORIGINAL RRT ALGORITHM CLASSES ---
# ==============================================================
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.neighbors = []


class RRT:
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=1.0, goal_sample_rate=10, max_iter=2000):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.min_rand, self.max_rand = rand_area
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []
        self.random_node_count = 0

    def plan(self):
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


# ==============================================================
# --- SINGLE TRIAL EXECUTION ---
# ==============================================================
def run_single_trial(start_pos, goal_pos, obstacle_list, bounds, resolution, trial_num):
    """Runs all algorithms once and returns a dictionary of the results."""
    trial_results = {"Trial_Number": trial_num}

    # 1. Base RRT Algorithm
    rrt_start_time = time.time()
    rrt = RRT(start_pos, goal_pos, obstacle_list, bounds, expand_dis=resolution, goal_sample_rate=5, max_iter=3000)
    rrt_path = rrt.plan()
    rrt_time = time.time() - rrt_start_time

    trial_results["RRT_Path_Distance"] = round(calculate_path_length(rrt_path), 3) if rrt_path else "FAILED"
    trial_results["RRT_Nodes_in_Tree"] = len(rrt.node_list) if rrt_path else "FAILED"
    trial_results["RRT_Time_Seconds"] = round(rrt_time, 5)

    # 2. Hybrid RRT/PRM + A*
    hybrid_start_time = time.time()

    hybrid_astar_path = None
    hybrid_nodes_explored = 0

    if rrt_path:
        for node in rrt.node_list:
            if node.parent:
                if node.parent not in node.neighbors: node.neighbors.append(node.parent)
                if node not in node.parent.neighbors: node.parent.neighbors.append(node)

        node_coords = [[n.x, n.y] for n in rrt.node_list]
        tree = KDTree(node_coords)
        k_query = min(11, len(rrt.node_list))

        for node in rrt.node_list:
            distances, indices = tree.query([node.x, node.y], k=k_query)
            if k_query > 1:
                for idx in indices[1:]:
                    neighbor = rrt.node_list[idx]
                    if neighbor not in node.neighbors:
                        if rrt.check_collision(node, neighbor, obstacle_list):
                            node.neighbors.append(neighbor)
                            neighbor.neighbors.append(node)

        start_node = rrt.node_list[0]
        goal_node = rrt.node_list[-1]
        open_set = []
        heapq.heappush(open_set, (0, id(start_node), start_node))
        came_from = {}
        g_score = {start_node: 0}

        def heuristic_hybrid(n1, n2):
            return math.hypot(n1.x - n2.x, n1.y - n2.y)

        while open_set:
            current = heapq.heappop(open_set)[2]
            hybrid_nodes_explored += 1
            if current == goal_node:
                hybrid_astar_path = []
                while current in came_from:
                    hybrid_astar_path.append([current.x, current.y])
                    current = came_from[current]
                hybrid_astar_path.append([start_node.x, start_node.y])
                break
            for neighbor in current.neighbors:
                tentative_g_score = g_score[current] + heuristic_hybrid(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic_hybrid(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score, id(neighbor), neighbor))

    hybrid_time = rrt_time + (time.time() - hybrid_start_time)

    trial_results["Hybrid_Path_Distance"] = round(calculate_path_length(hybrid_astar_path),
                                                  3) if hybrid_astar_path else "FAILED"
    trial_results["Hybrid_Nodes_Explored"] = hybrid_nodes_explored if hybrid_astar_path else "FAILED"
    trial_results["Hybrid_Total_Time_Seconds"] = round(hybrid_time, 5)

    # 3. Grid A*
    grid_astar_path, astar_nodes, astar_time = run_grid_search(start_pos, goal_pos, obstacle_list, bounds, resolution,
                                                               use_astar=True)
    trial_results["Grid_AStar_Distance"] = round(calculate_path_length(grid_astar_path),
                                                 3) if grid_astar_path else "FAILED"
    trial_results["Grid_AStar_Explored"] = astar_nodes if grid_astar_path else "FAILED"
    trial_results["Grid_AStar_Time"] = round(astar_time, 5)

    # 4. Grid Dijkstra
    dijkstra_path, dijkstra_nodes, dijkstra_time = run_grid_search(start_pos, goal_pos, obstacle_list, bounds,
                                                                   resolution, use_astar=False)
    trial_results["Grid_Dijkstra_Distance"] = round(calculate_path_length(dijkstra_path),
                                                    3) if dijkstra_path else "FAILED"
    trial_results["Grid_Dijkstra_Explored"] = dijkstra_nodes if dijkstra_path else "FAILED"
    trial_results["Grid_Dijkstra_Time"] = round(dijkstra_time, 5)

    return trial_results


# ==============================================================
# --- BATCH EXECUTION & LOGGING ---
# ==============================================================
def main():
    NUM_RUNS = 25  # <--- CHANGE THIS NUMBER TO RUN IT MORE TIMES
    OUTPUT_FILENAME = "pathfinding_results1.csv"

    start_pos = (1.0, 1.0)
    goal_pos = (9.0, 9.0)
    bounds = [0, 10]
    resolution = 0.5
    obstacle_list = [
        (3.0, 0.0, 1.0, 7.0),
        (6.0, 3.0, 2.5, 7.0),
        (3.0, 7.0, 2.0, 1.0)
    ]

    print(f"Starting {NUM_RUNS} batch runs. Plots are disabled during batch processing to ensure speed.")

    all_results = []

    for i in range(1, NUM_RUNS + 1):
        print(f"Executing Trial {i}/{NUM_RUNS}...")
        trial_data = run_single_trial(start_pos, goal_pos, obstacle_list, bounds, resolution, i)
        all_results.append(trial_data)

    # --- CUSTOM COLUMN ORDER DEFINED HERE ---
    custom_headers = [
        "Trial_Number",
        "Hybrid_Total_Time_Seconds",
        "RRT_Time_Seconds",
        "Grid_AStar_Time",
        "Grid_Dijkstra_Time",
        "Hybrid_Path_Distance",
        "RRT_Path_Distance",
        "Grid_AStar_Distance",
        "Grid_Dijkstra_Distance",
        "Hybrid_Nodes_Explored",
        "RRT_Nodes_in_Tree",
        "Grid_AStar_Explored",
        "Grid_Dijkstra_Explored"
    ]

    # Write all collected data to a CSV (Excel) file
    print(f"\nWriting data to {OUTPUT_FILENAME}...")

    with open(OUTPUT_FILENAME, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=custom_headers)
        writer.writeheader()
        writer.writerows(all_results)

    print("Success! You can now open 'pathfinding_results.csv' in Excel.")


if __name__ == '__main__':
    main()