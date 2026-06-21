import heapq


class Node:
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0  # Distance between current node and start node
        self.h = 0  # Estimated distance from current node to end node (heuristic)
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        # Nodes are equal if they have the same coordinates
        return self.position == other.position

    def __lt__(self, other):
        # Required so the heapq module can compare and sort nodes by their F cost
        return self.f < other.f


def astar(maze, start, end):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze.
    """
    # Create start and end nodes
    start_node = Node(None, start)
    end_node = Node(None, end)

    # Initialize the open and closed lists
    open_list = []
    closed_set = set()  # Using a set for O(1) lookups

    # Add the start node to the priority queue
    heapq.heappush(open_list, start_node)

    # Loop until we find the end node
    while len(open_list) > 0:

        # Pop the node with the lowest f-cost off the queue
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        # Check if we have reached the goal
        if current_node == end_node:
            path = []
            current = current_node
            # Trace back the parents to construct the path
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return the reversed path (start to end)

        # Generate children (Adjacent squares)
        # Using 8-way movement: Up, Down, Left, Right, and Diagonals
        # If you only want 4-way movement, remove the diagonal tuples.
        movements = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for new_position in movements:
            # Calculate new node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range of the maze dimensions
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure it is walkable terrain (0 = path, 1 = wall)
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Make sure the position hasn't already been evaluated
            if node_position in closed_set:
                continue

            # Create new child node
            new_node = Node(current_node, node_position)

            # Calculate f, g, and h values
            new_node.g = current_node.g + 1
            # Using Manhattan distance for the heuristic (h)
            new_node.h = abs(new_node.position[0] - end_node.position[0]) + abs(
                new_node.position[1] - end_node.position[1])
            new_node.f = new_node.g + new_node.h

            # Check if this child is already in the open list with a lower or equal G cost
            skip = False
            for open_node in open_list:
                if new_node == open_node and new_node.g >= open_node.g:
                    skip = True
                    break

            if skip:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, new_node)

    # Return None if no path is found
    return None


# --- Example Usage ---
if __name__ == '__main__':
    # 0 = Walkable path
    # 1 = Wall/Obstacle
    maze = [
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    start_coordinate = (2, 2)
    end_coordinate = (7, 6)

    path = astar(maze, start_coordinate, end_coordinate)

    print("Coordinates of the path:")
    print(path)