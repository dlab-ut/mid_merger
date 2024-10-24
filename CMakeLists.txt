cmake_minimum_required(VERSION 3.8)
project(mid_merger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_library(cloud_merger SHARED src/cloud_merger.cpp)
rclcpp_components_register_nodes(cloud_merger "tc_livox_utils::PcdFilter")

target_compile_definitions(cloud_merger
  PRIVATE "PCD_FILTER_BUILDING_LIBRARY"
)

target_include_directories(cloud_merger PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(cloud_merger PUBLIC c_std_99 cxx_std_17) 

ament_target_dependencies(
  cloud_merger
  "rclcpp"
  "sensor_msgs"
  "pcl_ros"
  "rclcpp_components"
)

install(
  TARGETS cloud_merger
  DESTINATION lib
)
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
#install(
 # DIRECTORY config
 # DESTINATION share/${PROJECT_NAME}
#)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
