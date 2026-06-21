#include <chrono>
#include <memory>
#include <vector>
#include <set>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "hybrid_planner/rrt_prm_astar.hpp"

using namespace std::chrono_literals;

namespace hybrid_planner
{

class HybridPlannerRosNode : public rclcpp::Node
{
public:
  HybridPlannerRosNode()
  : Node("hybrid_planner_node")
  {
    // Parameters mirror the Python main(): start/goal/obstacles/rand_area/etc.
    declare_parameter<double>("start_x", 1.0);
    declare_parameter<double>("start_y", 1.0);
    declare_parameter<double>("goal_x", 9.0);
    declare_parameter<double>("goal_y", 9.0);
    declare_parameter<double>("flight_altitude", 3.0);
    declare_parameter<double>("min_rand", 0.0);
    declare_parameter<double>("max_rand", 10.0);
    declare_parameter<double>("expand_dis", 0.5);
    declare_parameter<int>("goal_sample_rate", 5);
    declare_parameter<int>("max_iter", 3000);
    declare_parameter<int>("k_neighbors", 10);
    declare_parameter<double>("safety_margin", 0.3);
    declare_parameter<std::string>("frame_id", "map");

    path_pub_ = create_publisher<nav_msgs::msg::Path>("planned_path", 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "planner_markers", 10);

    // Latched-ish behaviour via a timer: republish periodically so RViz
    // picks it up even if it starts after this node.
    timer_ = create_wall_timer(1s, std::bind(&HybridPlannerRosNode::publishLatest, this));

    runPlanner();
  }

private:
  void runPlanner()
  {
    double sx = get_parameter("start_x").as_double();
    double sy = get_parameter("start_y").as_double();
    double gx = get_parameter("goal_x").as_double();
    double gy = get_parameter("goal_y").as_double();
    alt_ = get_parameter("flight_altitude").as_double();
    frame_id_ = get_parameter("frame_id").as_string();

    // Same obstacle_list as the Python script: (ox, oy, w, h)
    std::vector<Obstacle> obstacles = {
      {3.0, 0.0, 1.0, 7.0},
      {6.0, 3.0, 2.5, 7.0},
      {3.0, 7.0, 2.0, 1.0}
    };
    obstacles_ = obstacles;

    HybridRRTPRMAStar planner(
      Point2D{sx, sy},
      Point2D{gx, gy},
      obstacles,
      get_parameter("min_rand").as_double(),
      get_parameter("max_rand").as_double(),
      get_parameter("expand_dis").as_double(),
      get_parameter("goal_sample_rate").as_int(),
      get_parameter("max_iter").as_int(),
      get_parameter("k_neighbors").as_int(),
      /*seed=*/0,
      get_parameter("safety_margin").as_double());

    RCLCPP_INFO(get_logger(), "Running hybrid RRT/PRM/A* planner...");
    result_ = planner.plan();

    if (!result_.success) {
      RCLCPP_ERROR(get_logger(), "Planner failed: max iterations reached without finding goal.");
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "Planner succeeded. Random nodes sampled: %d | Graph nodes: %zu | RRT path pts: %zu | A* path pts: %zu",
      result_.random_node_count, result_.graph.nodes.size(),
      result_.rrt_path.size(), result_.astar_path.size());

    has_result_ = true;
  }

  void publishLatest()
  {
    if (!has_result_) return;

    auto stamp = now();

    // --- Publish A* path as nav_msgs/Path (this is what the drone follows) ---
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = stamp;
    path_msg.header.frame_id = frame_id_;
    for (const auto & pt : result_.astar_path) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path_msg.header;
      ps.pose.position.x = pt.x;
      ps.pose.position.y = pt.y;
      ps.pose.position.z = alt_;
      ps.pose.orientation.w = 1.0;
      path_msg.poses.push_back(ps);
    }
    path_pub_->publish(path_msg);

    // --- Publish visualization markers: obstacles, graph edges, RRT path, A* path ---
    visualization_msgs::msg::MarkerArray marker_array;
    int id = 0;

    // Obstacles as cubes
    for (const auto & ob : obstacles_) {
      visualization_msgs::msg::Marker m;
      m.header.stamp = stamp;
      m.header.frame_id = frame_id_;
      m.ns = "obstacles";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = ob.x + ob.w / 2.0;
      m.pose.position.y = ob.y + ob.h / 2.0;
      m.pose.position.z = alt_ / 2.0;
      m.pose.orientation.w = 1.0;
      m.scale.x = ob.w;
      m.scale.y = ob.h;
      m.scale.z = alt_;
      m.color.r = 0.5; m.color.g = 0.5; m.color.b = 0.5; m.color.a = 0.6;
      marker_array.markers.push_back(m);
    }

    // Full graph edges (faint green lines), mirrors the matplotlib PRM web
    {
      visualization_msgs::msg::Marker m;
      m.header.stamp = stamp;
      m.header.frame_id = frame_id_;
      m.ns = "graph_edges";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::LINE_LIST;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.scale.x = 0.02;
      m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 0.15;
      m.pose.orientation.w = 1.0;

      std::set<std::pair<int, int>> drawn;
      for (size_t i = 0; i < result_.graph.nodes.size(); ++i) {
        for (int j : result_.graph.nodes[i].neighbors) {
          auto e = std::make_pair(std::min<int>(i, j), std::max<int>(i, j));
          if (drawn.count(e)) continue;
          drawn.insert(e);
          geometry_msgs::msg::Point p1, p2;
          p1.x = result_.graph.nodes[i].x; p1.y = result_.graph.nodes[i].y; p1.z = alt_;
          p2.x = result_.graph.nodes[j].x; p2.y = result_.graph.nodes[j].y; p2.z = alt_;
          m.points.push_back(p1);
          m.points.push_back(p2);
        }
      }
      marker_array.markers.push_back(m);
    }

    // Original RRT path (solid red), mirrors the matplotlib red line
    marker_array.markers.push_back(
      makeLineStrip("rrt_path", id++, result_.rrt_path, 1.0, 0.0, 0.0, 0.08, stamp));

    // A* optimized path (dashed-look cyan -- LINE_STRIP doesn't support
    // dashing natively, so we use a distinct color/width instead, same
    // visual intent as the Python dashed cyan line)
    marker_array.markers.push_back(
      makeLineStrip("astar_path", id++, result_.astar_path, 0.0, 1.0, 1.0, 0.12, stamp));

    marker_pub_->publish(marker_array);
  }

  visualization_msgs::msg::Marker makeLineStrip(
    const std::string & ns, int id, const std::vector<Point2D> & pts,
    double r, double g, double b, double width, const rclcpp::Time & stamp)
  {
    visualization_msgs::msg::Marker m;
    m.header.stamp = stamp;
    m.header.frame_id = frame_id_;
    m.ns = ns;
    m.id = id;
    m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = width;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.9;
    m.pose.orientation.w = 1.0;
    for (const auto & p : pts) {
      geometry_msgs::msg::Point gp;
      gp.x = p.x; gp.y = p.y; gp.z = alt_;
      m.points.push_back(gp);
    }
    return m;
  }

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  PlanResult result_;
  std::vector<Obstacle> obstacles_;
  bool has_result_ = false;
  double alt_ = 3.0;
  std::string frame_id_ = "map";
};

}  // namespace hybrid_planner

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hybrid_planner::HybridPlannerRosNode>());
  rclcpp::shutdown();
  return 0;
}
