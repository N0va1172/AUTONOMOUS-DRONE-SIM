#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "mavros_msgs/msg/state.hpp"
#include "mavros_msgs/srv/command_bool.hpp"
#include "mavros_msgs/srv/set_mode.hpp"
#include "mavros_msgs/srv/command_tol.hpp"

using namespace std::chrono_literals;

namespace hybrid_planner
{

enum class FlightStage
{
  WAIT_PATH,
  WAIT_FCU_CONNECT,
  SET_GUIDED,
  ARM,
  TAKEOFF,
  WAIT_TAKEOFF_ALT,
  FOLLOW_PATH,
  DONE
};

class DroneCommander : public rclcpp::Node
{
public:
  DroneCommander()
  : Node("drone_commander")
  {
    declare_parameter<double>("flight_altitude", 3.0);
    declare_parameter<double>("waypoint_tolerance", 0.4);
    alt_ = get_parameter("flight_altitude").as_double();
    wp_tol_ = get_parameter("waypoint_tolerance").as_double();

    path_sub_ = create_subscription<nav_msgs::msg::Path>(
      "planned_path", 10,
      std::bind(&DroneCommander::pathCallback, this, std::placeholders::_1));

    // MAVROS publishes /mavros/state and /mavros/local_position/pose using
    // SensorDataQoS (best-effort, depth ~5), NOT the default reliable QoS.
    // Subscribing with the default reliable QoS causes a QoS incompatibility
    // -- the subscription is created but ROS2 silently delivers zero
    // messages to it (you'll see a "requesting incompatible QoS" warning,
    // not a crash). Matching QoS here is required for any data to arrive.
    auto sensor_qos = rclcpp::SensorDataQoS();

    state_sub_ = create_subscription<mavros_msgs::msg::State>(
      "mavros/state", sensor_qos,
      std::bind(&DroneCommander::stateCallback, this, std::placeholders::_1));

    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "mavros/local_position/pose", sensor_qos,
      std::bind(&DroneCommander::poseCallback, this, std::placeholders::_1));

    setpoint_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "mavros/setpoint_position/local", 10);

    arming_client_ = create_client<mavros_msgs::srv::CommandBool>("mavros/cmd/arming");
    set_mode_client_ = create_client<mavros_msgs::srv::SetMode>("mavros/set_mode");
    takeoff_client_ = create_client<mavros_msgs::srv::CommandTOL>("mavros/cmd/takeoff");

    timer_ = create_wall_timer(200ms, std::bind(&DroneCommander::controlLoop, this));

    RCLCPP_INFO(get_logger(), "Drone commander started. Waiting for planned path...");
  }

private:
  void pathCallback(const nav_msgs::msg::Path::SharedPtr msg)
  {
    if (path_received_) return;  // only take the first plan, mirrors a one-shot mission
    if (msg->poses.empty()) return;
    path_ = *msg;
    path_received_ = true;
    RCLCPP_INFO(get_logger(), "Received path with %zu waypoints.", path_.poses.size());
  }

  void stateCallback(const mavros_msgs::msg::State::SharedPtr msg)
  {
    current_state_ = *msg;
    fcu_connected_ = msg->connected;
  }

  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    current_pose_ = *msg;
    have_pose_ = true;
  }

  double horizDistance(const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b)
  {
    return std::hypot(a.x - b.x, a.y - b.y);
  }

  void controlLoop()
  {
    switch (stage_) {
      case FlightStage::WAIT_PATH:
        if (path_received_) {
          stage_ = FlightStage::WAIT_FCU_CONNECT;
        }
        break;

      case FlightStage::WAIT_FCU_CONNECT:
        if (fcu_connected_ && have_pose_) {
          RCLCPP_INFO(get_logger(), "FCU connected. Streaming setpoints before switching mode...");
          // ArduPilot/PX4 GUIDED/OFFBOARD typically require a stream of
          // setpoints BEFORE the mode switch is accepted, so we publish a
          // hold-position setpoint here too.
          publishHoldSetpoint();
          setpoint_stream_count_++;
          if (setpoint_stream_count_ > 20) {  // ~4s of streaming at 5Hz minimum
            stage_ = FlightStage::SET_GUIDED;
          }
        }
        break;

      case FlightStage::SET_GUIDED:
        publishHoldSetpoint();
        requestGuidedMode();
        if (current_state_.mode == "GUIDED") {
          stage_ = FlightStage::ARM;
        }
        break;

      case FlightStage::ARM:
        publishHoldSetpoint();
        requestArm();
        if (current_state_.armed) {
          RCLCPP_INFO(get_logger(), "Armed. Commanding takeoff to %.2f m.", alt_);
          stage_ = FlightStage::TAKEOFF;
        }
        break;

      case FlightStage::TAKEOFF:
        // NOTE: we intentionally do NOT call the separate MAV_CMD_NAV_TAKEOFF
        // service here. ArduPilot's NAV_TAKEOFF handler and a continuously
        // streamed GUIDED-mode position setpoint are two competing control
        // paths -- running both at once caused the autopilot to never
        // command real climb thrust (motors stayed pinned at MOT_SPIN_ARM
        // idle, relative_alt stayed at 0, and the land-detector then forced
        // a disarm). publishHoldSetpoint() already targets z = alt_, which
        // is sufficient on its own to command a climb in GUIDED mode -- the
        // exact same mechanism followPath() uses for horizontal waypoints.
        publishHoldSetpoint();
        stage_ = FlightStage::WAIT_TAKEOFF_ALT;
        break;

      case FlightStage::WAIT_TAKEOFF_ALT:
        publishHoldSetpoint();
        if (have_pose_ && current_pose_.pose.position.z >= alt_ - 0.3) {
          RCLCPP_INFO(get_logger(), "Reached takeoff altitude. Following planned path.");
          stage_ = FlightStage::FOLLOW_PATH;
        }
        break;

      case FlightStage::FOLLOW_PATH:
        followPath();
        break;

      case FlightStage::DONE:
        publishHoldSetpoint();  // hold at final waypoint
        break;
    }
  }

  void publishHoldSetpoint()
  {
    geometry_msgs::msg::PoseStamped sp;
    sp.header.stamp = now();
    sp.header.frame_id = "map";
    if (have_pose_) {
      sp.pose.position.x = current_pose_.pose.position.x;
      sp.pose.position.y = current_pose_.pose.position.y;
    }
    sp.pose.position.z = alt_;
    sp.pose.orientation.w = 1.0;
    setpoint_pub_->publish(sp);
  }

  void followPath()
  {
    if (wp_index_ >= path_.poses.size()) {
      stage_ = FlightStage::DONE;
      RCLCPP_INFO(get_logger(), "Path complete. Holding final position.");
      return;
    }

    const auto & target = path_.poses[wp_index_].pose.position;

    geometry_msgs::msg::PoseStamped sp;
    sp.header.stamp = now();
    sp.header.frame_id = "map";
    sp.pose.position.x = target.x;
    sp.pose.position.y = target.y;
    sp.pose.position.z = alt_;
    sp.pose.orientation.w = 1.0;
    setpoint_pub_->publish(sp);

    if (have_pose_) {
      double d = horizDistance(current_pose_.pose.position, target);
      if (d <= wp_tol_) {
        RCLCPP_INFO(
          get_logger(), "Reached waypoint %zu/%zu (x=%.2f y=%.2f)",
          wp_index_ + 1, path_.poses.size(), target.x, target.y);
        wp_index_++;
      }
    }
  }

  void requestGuidedMode()
  {
    if (!set_mode_client_->service_is_ready()) return;
    if (mode_request_in_flight_) return;
    auto req = std::make_shared<mavros_msgs::srv::SetMode::Request>();
    req->custom_mode = "GUIDED";
    mode_request_in_flight_ = true;
    set_mode_client_->async_send_request(
      req,
      [this](rclcpp::Client<mavros_msgs::srv::SetMode>::SharedFuture) {
        mode_request_in_flight_ = false;
      });
  }

  void requestArm()
  {
    if (!arming_client_->service_is_ready()) return;
    if (arm_request_in_flight_) return;
    auto req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
    req->value = true;
    arm_request_in_flight_ = true;
    arming_client_->async_send_request(
      req,
      [this](rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedFuture) {
        arm_request_in_flight_ = false;
      });
  }

  void requestTakeoff()
  {
    if (!takeoff_client_->service_is_ready()) return;
    auto req = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
    req->altitude = static_cast<float>(alt_);
    req->latitude = 0;
    req->longitude = 0;
    req->min_pitch = 0;
    req->yaw = 0;
    takeoff_client_->async_send_request(req);
  }

  // State
  FlightStage stage_ = FlightStage::WAIT_PATH;
  nav_msgs::msg::Path path_;
  bool path_received_ = false;
  size_t wp_index_ = 0;

  mavros_msgs::msg::State current_state_;
  bool fcu_connected_ = false;

  geometry_msgs::msg::PoseStamped current_pose_;
  bool have_pose_ = false;

  int setpoint_stream_count_ = 0;
  bool mode_request_in_flight_ = false;
  bool arm_request_in_flight_ = false;

  double alt_;
  double wp_tol_;

  // ROS interfaces
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr setpoint_pub_;
  rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;
  rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;
  rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedPtr takeoff_client_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace hybrid_planner

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hybrid_planner::DroneCommander>());
  rclcpp::shutdown();
  return 0;
}
