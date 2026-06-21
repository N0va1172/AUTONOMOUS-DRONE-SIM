cmake_minimum_required(VERSION 3.8)
project(hybrid_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)

include_directories(include)

# --- Core planner library (algorithm only, no ROS dependency) ---
add_library(rrt_prm_astar src/rrt_prm_astar.cpp)
target_include_directories(rrt_prm_astar PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# --- Planner ROS2 node ---
add_executable(planner_node src/planner_node.cpp)
target_link_libraries(planner_node rrt_prm_astar)
ament_target_dependencies(planner_node
  rclcpp nav_msgs geometry_msgs visualization_msgs)

# --- Standalone test (no ROS2 dependency, for quick algorithm sanity checks) ---
add_executable(standalone_test src/standalone_test.cpp)
target_link_libraries(standalone_test rrt_prm_astar)

# --- Drone commander ROS2 node (MAVROS interface) ---
add_executable(drone_commander src/drone_commander.cpp)
ament_target_dependencies(drone_commander
  rclcpp nav_msgs geometry_msgs mavros_msgs)

install(TARGETS
  planner_node
  drone_commander
  standalone_test
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS rrt_prm_astar
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})
install(DIRECTORY worlds DESTINATION share/${PROJECT_NAME})
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()
