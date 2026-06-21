#include "hybrid_planner/rrt_prm_astar.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <utility>

namespace hybrid_planner
{

int NodeGraph::addNode(double x, double y, int parent_idx)
{
  PlannerNode n;
  n.x = x;
  n.y = y;
  n.parent = parent_idx;
  nodes.push_back(n);
  return static_cast<int>(nodes.size()) - 1;
}

HybridRRTPRMAStar::HybridRRTPRMAStar(
  Point2D start,
  Point2D goal,
  std::vector<Obstacle> obstacles,
  double min_rand,
  double max_rand,
  double expand_dis,
  int goal_sample_rate,
  int max_iter,
  int k_neighbors,
  unsigned int seed,
  double safety_margin)
: start_(start),
  goal_(goal),
  obstacles_(std::move(obstacles)),
  min_rand_(min_rand),
  max_rand_(max_rand),
  expand_dis_(expand_dis),
  goal_sample_rate_(goal_sample_rate),
  max_iter_(max_iter),
  k_neighbors_(k_neighbors),
  rng_(seed == 0 ? std::random_device{}() : seed),
  uniform_dist_(min_rand, max_rand),
  percent_dist_(0, 100)
{
  // Inflate every obstacle by safety_margin on all sides. This gives a real
  // drone (with GPS/EKF position noise and finite control authority) actual
  // clearance, rather than planning paths that mathematically just graze an
  // obstacle edge with zero margin.
  if (safety_margin > 0.0) {
    for (auto & ob : obstacles_) {
      ob.x -= safety_margin;
      ob.y -= safety_margin;
      ob.w += 2.0 * safety_margin;
      ob.h += 2.0 * safety_margin;
    }
  }
}

double HybridRRTPRMAStar::dist(double x1, double y1, double x2, double y2)
{
  return std::hypot(x2 - x1, y2 - y1);
}

double HybridRRTPRMAStar::distToGoal(double x, double y) const
{
  return dist(x, y, goal_.x, goal_.y);
}

Point2D HybridRRTPRMAStar::sampleRandom()
{
  random_node_count_++;
  // Mirrors: if random.randint(0, 100) > goal_sample_rate -> random point,
  // else -> goal point. randint is inclusive on both ends in Python, so we
  // use the same [0, 100] inclusive range here.
  int r = percent_dist_(rng_);
  if (r > goal_sample_rate_) {
    return Point2D{uniform_dist_(rng_), uniform_dist_(rng_)};
  }
  return goal_;
}

int HybridRRTPRMAStar::nearestNodeIndex(const NodeGraph & graph, const Point2D & p) const
{
  int best_idx = 0;
  double best_d = std::numeric_limits<double>::max();
  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    double d = (graph.nodes[i].x - p.x) * (graph.nodes[i].x - p.x) +
               (graph.nodes[i].y - p.y) * (graph.nodes[i].y - p.y);
    if (d < best_d) {
      best_d = d;
      best_idx = static_cast<int>(i);
    }
  }
  return best_idx;
}

PlannerNode HybridRRTPRMAStar::steer(
  const PlannerNode & from, const PlannerNode & to, double extend_length) const
{
  PlannerNode new_node;
  new_node.x = from.x;
  new_node.y = from.y;
  double d = dist(from.x, from.y, to.x, to.y);
  double theta = std::atan2(to.y - from.y, to.x - from.x);
  double step = std::min(extend_length, d);
  new_node.x += step * std::cos(theta);
  new_node.y += step * std::sin(theta);
  // parent is set by caller (needs graph index, not available here)
  return new_node;
}

bool HybridRRTPRMAStar::collisionFree(double x1, double y1, double x2, double y2) const
{
  // The original Python check_collision() sampled 10 fixed points along the
  // edge and tested point-in-box for each. That approach has a fundamental
  // flaw: a segment can clip through a thin sliver or a box corner *between*
  // two consecutive samples without ever landing inside the box, especially
  // once PRM densification adds long edges between nodes that were never
  // adjacent during RRT growth. Increasing the sample count only shrinks the
  // failure window, it never removes it.
  //
  // Fix: use an exact segment-vs-axis-aligned-bounding-box intersection test
  // (slab method) instead of sampling. This is the standard ray/segment-AABB
  // test and has no resolution-dependent blind spot -- if the segment's
  // bounding parametric range [0,1] overlaps the box on both axes
  // simultaneously, the segment truly intersects the box.
  for (const auto & ob : obstacles_) {
    if (segmentIntersectsBox(x1, y1, x2, y2, ob)) {
      return false;
    }
  }
  return true;
}

bool HybridRRTPRMAStar::segmentIntersectsBox(
  double x1, double y1, double x2, double y2, const Obstacle & ob) const
{
  // Slab method: clip the parametric range t in [0,1] of the segment
  // P(t) = (x1,y1) + t * (dx,dy) against each pair of box planes. If the
  // intersection of all clipped ranges is non-empty, the segment crosses
  // the box (this also correctly handles the case where an endpoint starts
  // or ends inside the box).
  double dx = x2 - x1;
  double dy = y2 - y1;

  double t_min = 0.0;
  double t_max = 1.0;

  const double eps = 1e-12;

  // X slab
  if (std::abs(dx) < eps) {
    // Segment is vertical in X: must already be within the box's X range.
    if (x1 < ob.x || x1 > ob.x + ob.w) {
      return false;
    }
  } else {
    double tx1 = (ob.x - x1) / dx;
    double tx2 = (ob.x + ob.w - x1) / dx;
    if (tx1 > tx2) std::swap(tx1, tx2);
    t_min = std::max(t_min, tx1);
    t_max = std::min(t_max, tx2);
    if (t_min > t_max) return false;
  }

  // Y slab
  if (std::abs(dy) < eps) {
    if (y1 < ob.y || y1 > ob.y + ob.h) {
      return false;
    }
  } else {
    double ty1 = (ob.y - y1) / dy;
    double ty2 = (ob.y + ob.h - y1) / dy;
    if (ty1 > ty2) std::swap(ty1, ty2);
    t_min = std::max(t_min, ty1);
    t_max = std::min(t_max, ty2);
    if (t_min > t_max) return false;
  }

  return true;
}

bool HybridRRTPRMAStar::growRRT(
  NodeGraph & graph, int & goal_node_idx, std::vector<int> & rrt_path_indices)
{
  graph.nodes.clear();
  int start_idx = graph.addNode(start_.x, start_.y, -1);
  (void)start_idx;  // always 0

  for (int iter = 0; iter < max_iter_; ++iter) {
    Point2D rnd = sampleRandom();
    PlannerNode rnd_node;
    rnd_node.x = rnd.x;
    rnd_node.y = rnd.y;

    int nearest_idx = nearestNodeIndex(graph, rnd);
    const PlannerNode & nearest = graph.nodes[nearest_idx];

    PlannerNode new_node = steer(nearest, rnd_node, expand_dis_);

    if (collisionFree(nearest.x, nearest.y, new_node.x, new_node.y)) {
      graph.addNode(new_node.x, new_node.y, nearest_idx);
    }

    const PlannerNode & last = graph.nodes.back();
    if (distToGoal(last.x, last.y) <= expand_dis_) {
      PlannerNode goal_node_tmp;
      goal_node_tmp.x = goal_.x;
      goal_node_tmp.y = goal_.y;
      int last_idx = static_cast<int>(graph.nodes.size()) - 1;
      PlannerNode final_node = steer(last, goal_node_tmp, expand_dis_);

      if (collisionFree(last.x, last.y, final_node.x, final_node.y)) {
        goal_node_idx = graph.addNode(final_node.x, final_node.y, last_idx);

        // generate_final_course: walk parent chain from goal back to start
        rrt_path_indices.clear();
        int cur = goal_node_idx;
        while (cur != -1) {
          rrt_path_indices.push_back(cur);
          cur = graph.nodes[cur].parent;
        }
        std::reverse(rrt_path_indices.begin(), rrt_path_indices.end());
        return true;
      }
    }
  }
  return false;
}

void HybridRRTPRMAStar::densifyWithKNN(NodeGraph & graph)
{
  // 1. Convert RRT tree parent links into bidirectional graph edges.
  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    int parent = graph.nodes[i].parent;
    if (parent != -1) {
      auto & node_neighbors = graph.nodes[i].neighbors;
      if (std::find(node_neighbors.begin(), node_neighbors.end(), parent) ==
        node_neighbors.end())
      {
        node_neighbors.push_back(parent);
      }
      auto & parent_neighbors = graph.nodes[parent].neighbors;
      if (std::find(parent_neighbors.begin(), parent_neighbors.end(),
        static_cast<int>(i)) == parent_neighbors.end())
      {
        parent_neighbors.push_back(static_cast<int>(i));
      }
    }
  }

  // 2. Add k-nearest-neighbor edges (brute-force, equivalent to scipy
  // KDTree.query with k = k_neighbors + 1, since the query includes the
  // point itself at distance 0).
  const size_t n = graph.nodes.size();
  int k_query = std::min(static_cast<size_t>(k_neighbors_) + 1, n);

  for (size_t i = 0; i < n; ++i) {
    std::vector<std::pair<double, int>> dists;
    dists.reserve(n);
    for (size_t j = 0; j < n; ++j) {
      if (j == i) continue;
      double d = dist(graph.nodes[i].x, graph.nodes[i].y, graph.nodes[j].x, graph.nodes[j].y);
      dists.emplace_back(d, static_cast<int>(j));
    }
    int take = std::min(static_cast<int>(dists.size()), k_query - 1);
    std::partial_sort(
      dists.begin(), dists.begin() + take, dists.end(),
      [](const auto & a, const auto & b) {return a.first < b.first;});

    for (int idx = 0; idx < take; ++idx) {
      int j = dists[idx].second;
      auto & neighbors_i = graph.nodes[i].neighbors;
      bool already = std::find(neighbors_i.begin(), neighbors_i.end(), j) != neighbors_i.end();
      if (!already) {
        if (collisionFree(graph.nodes[i].x, graph.nodes[i].y, graph.nodes[j].x, graph.nodes[j].y)) {
          graph.nodes[i].neighbors.push_back(j);
          graph.nodes[j].neighbors.push_back(static_cast<int>(i));
        }
      }
    }
  }
}

bool HybridRRTPRMAStar::runAStar(
  const NodeGraph & graph, int start_idx, int goal_idx,
  std::vector<int> & path_indices_out) const
{
  using QueueEntry = std::pair<double, int>;  // (f_score, node_idx)
  auto cmp = [](const QueueEntry & a, const QueueEntry & b) {return a.first > b.first;};
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, decltype(cmp)> open_set(cmp);

  auto heuristic = [&graph](int a, int b) {
      return dist(graph.nodes[a].x, graph.nodes[a].y, graph.nodes[b].x, graph.nodes[b].y);
    };

  std::unordered_map<int, double> g_score;
  std::unordered_map<int, int> came_from;

  g_score[start_idx] = 0.0;
  open_set.push({heuristic(start_idx, goal_idx), start_idx});

  std::set<int> visited;

  while (!open_set.empty()) {
    int current = open_set.top().second;
    open_set.pop();

    if (visited.count(current)) continue;
    visited.insert(current);

    if (current == goal_idx) {
      path_indices_out.clear();
      int node = current;
      while (came_from.count(node)) {
        path_indices_out.push_back(node);
        node = came_from[node];
      }
      path_indices_out.push_back(start_idx);
      std::reverse(path_indices_out.begin(), path_indices_out.end());
      return true;
    }

    for (int neighbor : graph.nodes[current].neighbors) {
      double tentative_g = g_score[current] + heuristic(current, neighbor);
      if (!g_score.count(neighbor) || tentative_g < g_score[neighbor]) {
        came_from[neighbor] = current;
        g_score[neighbor] = tentative_g;
        double f = tentative_g + heuristic(neighbor, goal_idx);
        open_set.push({f, neighbor});
      }
    }
  }
  return false;
}

PlanResult HybridRRTPRMAStar::plan()
{
  PlanResult result;
  NodeGraph graph;
  int goal_idx = -1;
  std::vector<int> rrt_indices;

  bool rrt_ok = growRRT(graph, goal_idx, rrt_indices);
  result.random_node_count = random_node_count_;

  if (!rrt_ok) {
    result.success = false;
    result.graph = std::move(graph);
    return result;
  }

  for (int idx : rrt_indices) {
    result.rrt_path.push_back({graph.nodes[idx].x, graph.nodes[idx].y});
  }

  densifyWithKNN(graph);

  std::vector<int> astar_indices;
  bool astar_ok = runAStar(graph, 0, goal_idx, astar_indices);

  if (astar_ok) {
    for (int idx : astar_indices) {
      result.astar_path.push_back({graph.nodes[idx].x, graph.nodes[idx].y});
    }
  }

  result.success = true;
  result.graph = std::move(graph);
  return result;
}

}  // namespace hybrid_planner
