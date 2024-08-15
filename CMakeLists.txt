cmake_minimum_required(VERSION 3.8)
project(imu_kalman)
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++17")
add_compile_options(-Wall -Wextra -Wpedantic) #-Wall：启用大多数的常规编译警告。-Wextra：启用一些额外的警告（在启用 -Wall 的基础上）。-Wpedantic：启用严格的 ISO C 和 ISO C++ 标准兼容性警告。
#common variables 
#CMAKE_CURRENT_SOURCE_DIR 指的是当前处理的 CMakeLists.txt 所在的路径,比如上面我们提到的 src 子目录。
#CMAKE_SOURCE_DIR 工程顶层目录
#PROJECT_NAME 返回通过 PROJECT 指令定义的项目名称。
#CMAKE_BUILD_TYPE 生成 debug 版和 release 版的程序。可以的取值是 Debug Release RelWithDebInfo 和 MinSizeRel。
#当这个变量值为 Debug 的时候,CMAKE_CXX_FLAGS_DEBUG 和 CMAKE_C_FLAGS_DEBUG 中的字符串作为编译选项生成 Makefile，当这个变量值为 Release 的时候，工程会使用变量CMAKE_CXX_FLAGS_RELEASE 和 CMAKE_C_FLAGS_RELEASE 选项生成 Makefile。
#CMAKE_CXX_FLAGS c++的编译属性 
#LIBRARY_OUTPUT_PATH 生成的库文件放置的路径
#使用ENV{NAME}指令就可以调用系统的环境变量了。比如：MESSAGE(STATUS “HOME dir: ENV{HOME}”)

# find dependencies

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/lib/IMUOrientationEstimator/build")
include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/lib/IMUOrientationEstimator/include) #添加包的路径.cpp中搜索.h就在这路径里面找 gcc和g++ 的标准搜索路径为 /usr/local/include /usr/include /usr/target/include /usr/lib/gcc-lib/target/version/include

link_directories(${CMAKE_SOURCE_DIR}/lib/IMUOrientationEstimator/build)

add_executable(imu_kalman_node src/imu_kalman_node.cpp)
target_link_libraries(imu_kalman_node IMUOrientationEstimator)
ament_target_dependencies(imu_kalman_node rclcpp tf2 tf2_ros sensor_msgs)

add_executable(my_kalman_node src/my_kalman_node.cpp)
ament_target_dependencies(my_kalman_node rclcpp tf2 tf2_ros sensor_msgs nav_msgs)

install(TARGETS
  imu_kalman_node 
  DESTINATION lib/${PROJECT_NAME})


install(TARGETS
  my_kalman_node 
  DESTINATION lib/${PROJECT_NAME})

ament_package()