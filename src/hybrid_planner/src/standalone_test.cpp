// Standalone sanity test for the ported planner -- no ROS2 dependency.
// Mirrors the Python main(): same start/goal/obstacles, prints the same
// stats your Python script printed (random node count, graph size, path
// lengths). Build/run instructions are in the procedure doc.
#include <iostream>
#include "hybrid_planner/rrt_prm_astar.hpp"

int main()
{
  using namespace hybrid_planner;

  Point2D start{1.0, 1.0};
  Point2D goal{9.0, 9.0};

  std::vector<Obstacle> obstacles = {
    {3.0, 0.0, 1.0, 7.0},
    {6.0, 3.0, 2.5, 7.0},
    {3.0, 7.0, 2.0, 1.0}
  };

  HybridRRTPRMAStar planner(
    start, goal, obstacles,
    /*min_rand=*/0.0, /*max_rand=*/10.0,
    /*expand_dis=*/0.5, /*goal_sample_rate=*/5,
    /*max_iter=*/3000, /*k_neighbors=*/10,
    /*seed=*/42);

  std::cout << "Starting RRT/PRM/A* path planning...\n";
  PlanResult result = planner.plan();

  if (!result.success) {
    std::cout << "Cannot find path. Reached max iterations.\n";
    return 1;
  }

  std::cout << "----------------------------------------\n";
  std::cout << "Total random nodes generated: " << result.random_node_count << "\n";
  std::cout << "Total valid nodes in Graph: " << result.graph.nodes.size() << "\n";
  std::cout << "Number of nodes in RRT path: " << result.rrt_path.size() << "\n";
  if (!result.astar_path.empty()) {
    std::cout << "Number of nodes in A* path: " << result.astar_path.size() << "\n";
  } else {
    std::cout << "A* failed to find a path over the graph.\n";
  }
  std::cout << "----------------------------------------\n";

  std::cout << "\nA* path waypoints:\n";
  for (const auto & p : result.astar_path) {
    std::cout << "  (" << p.x << ", " << p.y << ")\n";
  }

  return 0;
}
