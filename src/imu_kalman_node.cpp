#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
// ros tf2 相关
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

#include "Estimator.h"

#include <sensor_msgs/msg/imu.hpp>

#include <memory>
using std::placeholders::_1;

// Init OriEst.
const double gyro_noise = 1e-6;
const double gyro_bias_noise = 1e-8;
const double acc_noise = 1e-6;

class ImuKalmanNode : public rclcpp::Node
{
private:
    /* data */
    OriEst::Estimator orientation_estimatro_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_;

    void imuCallback(const sensor_msgs::msg::Imu::UniquePtr imu_in);
    void publishPose();
    Eigen::Matrix3d G_R_I_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    Eigen::Vector3d initial_position_;
    Eigen::Vector3d initial_velocity_;
    rclcpp::Time prev_time_;

public:
    ImuKalmanNode() : Node("imu_kalman_node"), orientation_estimatro_(gyro_noise, gyro_bias_noise, acc_noise)
    {
        sub_ = this->create_subscription<sensor_msgs::msg::Imu>("/Imu_data", 10, std::bind(&ImuKalmanNode::imuCallback, this, _1));
        pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("imu_kalman", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        G_R_I_ = Eigen::Matrix3d::Identity();

        initial_position_ = Eigen::Vector3d{0, 0, 0};
        initial_velocity_ = Eigen::Vector3d{0, 0, 0};
    }
};

void ImuKalmanNode::imuCallback(const sensor_msgs::msg::Imu::UniquePtr imu_in)
{
    static Eigen::Vector3d acc_pre = Eigen::Vector3d(0, 0, 9.73);
    Eigen::Vector3d acc(imu_in->linear_acceleration.x, imu_in->linear_acceleration.y, imu_in->linear_acceleration.z);
    Eigen::Vector3d gyro(imu_in->angular_velocity.x, imu_in->angular_velocity.y, imu_in->angular_velocity.z);
    std::cout << "18:36:20: " << acc << std::endl;

    const double timestamp = imu_in->header.stamp.sec;
    OriEst::Status status = orientation_estimatro_.Estimate(timestamp, gyro, acc, &G_R_I_);

    geometry_msgs::msg::PoseWithCovarianceStamped message = geometry_msgs::msg::PoseWithCovarianceStamped();
    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = "world";

    pose_msg.pose.pose.orientation.x = imu_in->orientation.x;
    pose_msg.pose.pose.orientation.y = imu_in->orientation.y;
    pose_msg.pose.pose.orientation.z = imu_in->orientation.z;
    pose_msg.pose.pose.orientation.w = imu_in->orientation.w;

    Eigen::Vector3d velocity = initial_velocity_ + (acc - Eigen::Vector3d(0.1, 0.1, 9.73)) * 0.01;
    Eigen::Vector3d position = initial_position_ + initial_velocity_ * 0.01 + 0.5 * acc * 0.01 * 0.01;

    initial_velocity_ = velocity;
    initial_position_ = position;

    std::cout << "19:24:30: " << position << std::endl;

    pose_msg.pose.pose.position.x = position[0];
    pose_msg.pose.pose.position.y = position[1];
    pose_msg.pose.pose.position.z = position[2];

    geometry_msgs::msg::TransformStamped t; // 四元数+平移

    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "world";
    t.child_frame_id = "imu";
    t.transform.translation.x = pose_msg.pose.pose.position.x;
    t.transform.translation.y = pose_msg.pose.pose.position.y;
    t.transform.translation.z = pose_msg.pose.pose.position.z;
    t.transform.rotation.x = imu_in->orientation.x;
    t.transform.rotation.y = imu_in->orientation.y;
    t.transform.rotation.z = imu_in->orientation.z;
    t.transform.rotation.w = imu_in->orientation.w;
    // Send the transformation
    tf_broadcaster_->sendTransform(t);

    acc_pre = acc;

    pub_->publish(pose_msg);
    // publishPose();
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Node Ready.");
    rclcpp::spin(std::make_shared<ImuKalmanNode>());
    rclcpp::shutdown();
    return 0;
}