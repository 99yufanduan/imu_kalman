#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
// ros tf2 相关
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
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

const double gyro_noise = 1e-6; // 单位m/s^2
const double acc_noise = 1e-6;  // 单位m/s^2
const double wheel_noise = 0.01;
const double delta_t = 0.01; // 单位 s

const double epsilon = 1e-6; // 无穷小值
const double g = 9.81;       // 单位是 m/s^2，表示重力加速度

const double w_k = 0.1; // 将如果IMU做匀速运动，加速度计测量的就是重力（方向）。当然实际上，并不可能做匀速运动，可以当做过程噪声处理

class ImuKalmanNode : public rclcpp::Node
{
private:
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_twist_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_;

    void imuCallback(const sensor_msgs::msg::Imu::UniquePtr imu_in);
    void wheelCallback(const nav_msgs::msg::Odometry::UniquePtr odom_in);
    void publishPose();
    void predict(Eigen::Vector3d u_k);
    void update(Eigen::Vector3d z_k);
    void positionPredict(Eigen::Vector2d u_position_k);
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    Eigen::Vector3d x_k{
        {0, 0, 0}}; // rpy

    // 状态转移矩阵
    Eigen::Matrix3d A{
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}};

    // 输入矩阵
    Eigen::Matrix3d B{{delta_t, 0, 0},
                      {0, delta_t, 0},
                      {0, 0, delta_t}};

    // 状态协方差，初始为非常小
    Eigen::Matrix3d P_k{{epsilon, 0, 0},
                        {0, epsilon, 0},
                        {0, 0, epsilon}};

    // 输入噪声协方差
    Eigen::Matrix3d W;

    // 观测矩阵线性化后的雅可比矩阵
    Eigen::Matrix3d H;

    // 观测噪声协方差
    Eigen::Matrix3d V;

    Eigen::Matrix3d A_position{
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}};

    // 输入矩阵
    Eigen::Matrix<double, 3, 2> B_position;

    // 输入噪声协方差,假设陀螺仪的的角速度的误差为正负0.01m/s，
    Eigen::Matrix3d W_position{{wheel_noise * wheel_noise, 0, 0},
                               {0, wheel_noise *wheel_noise, 0},
                               {0, 0, 0}};

    Eigen::Vector3d u_position_k; // v,w
    Eigen::Vector3d x_position_k; // xyz
    Eigen::Matrix3d P_position_k;

public:
    ImuKalmanNode(const double gyro_noise, const double acc_noise) : Node("imu_kalman_node")
    {
        x_k = Eigen::Vector3d(0, 0, 0);          // rp
        x_position_k = Eigen::Vector3d(0, 0, 0); // rp
        W = Eigen::Matrix3d{
            {gyro_noise * gyro_noise, 0, 0},
            {0, gyro_noise * gyro_noise, 0},
            {0, 0, gyro_noise * gyro_noise}};
        V = Eigen::Matrix3d{
            {acc_noise * acc_noise, 0, 0},
            {0, acc_noise * acc_noise, 0},
            {0, 0, acc_noise * acc_noise}};

        P_position_k = Eigen::Matrix3d{{epsilon, 0, 0},
                                       {0, epsilon, 0},
                                       {0, 0, epsilon}};

        sub_ = this->create_subscription<sensor_msgs::msg::Imu>("/Imu_data", 10, std::bind(&ImuKalmanNode::imuCallback, this, _1));
        sub_twist_ = this->create_subscription<nav_msgs::msg::Odometry>("/odom", 10, std::bind(&ImuKalmanNode::wheelCallback, this, _1));
        pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("imu_kalman", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }
    void estimator(Eigen::Vector3d gyro, Eigen::Vector3d acc, const double delta_t);
};

Eigen::Quaterniond rpyToQuaternion(double roll_rad, double pitch_rad, double yaw_rad)
{
    // Create rotation matrices for each axis
    Eigen::AngleAxisd rollAngle(roll_rad, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch_rad, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw_rad, Eigen::Vector3d::UnitZ());

    // Combine the rotations
    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;

    return q;
}

void ImuKalmanNode::imuCallback(const sensor_msgs::msg::Imu::UniquePtr imu_in)
{
    Eigen::Vector3d acc(imu_in->linear_acceleration.x, imu_in->linear_acceleration.y, imu_in->linear_acceleration.z); // 单位为m/s^2
    Eigen::Vector3d gyro(imu_in->angular_velocity.x, imu_in->angular_velocity.y, imu_in->angular_velocity.z);         // 单位为rad/s
    static double time_pre = 0;
    const double delta_t = (imu_in->header.stamp.nanosec - time_pre) / 1e9; // 将时间差转换为秒
    estimator(gyro, acc, delta_t);
    time_pre = imu_in->header.stamp.nanosec;

    double timestamp = imu_in->header.stamp.sec;

    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = "world";
    Eigen::Quaterniond q = rpyToQuaternion(x_k[0], x_k[1], x_k[2]);

    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();
    // pose_msg.pose.pose.orientation.x = imu_in->orientation.x;
    // pose_msg.pose.pose.orientation.y = imu_in->orientation.y;
    // pose_msg.pose.pose.orientation.z = imu_in->orientation.z;
    // pose_msg.pose.pose.orientation.w = imu_in->orientation.w;
    pose_msg.pose.pose.position.x = x_position_k[0];
    pose_msg.pose.pose.position.y = x_position_k[1];
    pose_msg.pose.pose.position.z = x_position_k[2];

    // (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis) float64[36] covariance
    // 将位移协方差矩阵赋值给 pose.covariance 的前 3x3 部分
    pose_msg.pose.covariance[0] = P_position_k(0, 0);
    pose_msg.pose.covariance[1] = P_position_k(0, 1);
    pose_msg.pose.covariance[2] = P_position_k(0, 2);
    pose_msg.pose.covariance[6] = P_position_k(1, 0);
    pose_msg.pose.covariance[7] = P_position_k(1, 1);
    pose_msg.pose.covariance[8] = P_position_k(1, 2);
    pose_msg.pose.covariance[12] = P_position_k(2, 0);
    pose_msg.pose.covariance[13] = P_position_k(2, 1);
    pose_msg.pose.covariance[14] = P_position_k(2, 2);

    // 将姿态协方差矩阵赋值给 pose.covariance 的后 3x3 部分
    pose_msg.pose.covariance[21] = P_k(0, 0);
    pose_msg.pose.covariance[22] = P_k(0, 1);
    pose_msg.pose.covariance[23] = P_k(0, 2);
    pose_msg.pose.covariance[27] = P_k(1, 0);
    pose_msg.pose.covariance[28] = P_k(1, 1);
    pose_msg.pose.covariance[29] = P_k(1, 2);
    pose_msg.pose.covariance[33] = P_k(2, 0);
    pose_msg.pose.covariance[34] = P_k(2, 1);
    pose_msg.pose.covariance[35] = P_k(2, 2);

    geometry_msgs::msg::TransformStamped t; // 四元数+平移

    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "world";
    t.child_frame_id = "odom";
    t.transform.translation.x = x_position_k[0];
    t.transform.translation.y = x_position_k[1];
    t.transform.translation.z = x_position_k[2];
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();
    // t.transform.rotation.x = imu_in->orientation.x;
    // t.transform.rotation.y = imu_in->orientation.y;
    // t.transform.rotation.z = imu_in->orientation.z;
    // t.transform.rotation.w = imu_in->orientation.w;
    // Send the transformation
    tf_broadcaster_->sendTransform(t);
    pub_->publish(pose_msg);
}

void ImuKalmanNode::wheelCallback(const nav_msgs::msg::Odometry::UniquePtr odom_in)
{
    Eigen::Vector2d u_position = Eigen::Vector2d(odom_in->twist.twist.linear.x, odom_in->twist.twist.angular.z);
    positionPredict(u_position);
}

void ImuKalmanNode::predict(Eigen::Vector3d gyro) // u_k =gyro.xyz
{
    B = Eigen::Matrix3d{{delta_t, 0, 0},
                        {0, delta_t, 0},
                        {0, 0, delta_t}};
    /**** 状态预测 ****/
    x_k = A * x_k + B * gyro;
    /**** 状态协方差预测 ****/
    P_k = A * P_k * A.transpose() + W;

    /**********************(Debug cout)*********************/
    std::cout << "predict end" << std::endl;
    std::cout << "x_k= " << x_k << std::endl;
    /**********************(Debug END)********************/
}

void ImuKalmanNode::update(Eigen::Vector3d acc) // z_k =acc.xyz
{
    Eigen::Matrix3d E = Eigen::Matrix3d{
        {exp(fabs(acc.norm() - g) / 100000) - 1, 0, 0},
        {0, exp(fabs(acc.norm() - g / 100000)) - 1, 0},
        {0, 0, exp(fabs(acc.norm() - g / 100000)) - 1}};

    H = Eigen::Matrix3d{{0, -1 * g * cos(x_k[1]), 0},
                        {g * cos(x_k[1]) * cos(x_k[0]), -1 * g * sin(x_k[1]) * sin(x_k[0]), 0},
                        {-1 * g * cos(x_k[1]) * sin(x_k[0]), -1 * g * sin(x_k[1]) * cos(x_k[0]), 0}}; // 观测方程线性化的雅可比矩阵
    Eigen::Matrix3d K;
    K = P_k * H.transpose() * (H * P_k * H.transpose() + V + E).inverse(); // 求卡尔曼增益，eigen的求逆会根据矩阵类型自动选择求逆方法
    /**** 状态更新 ****/
    x_k = x_k + K * (acc - Eigen::Vector3d(-1 * g * cos(x_k[0]) * sin(x_k[1]), g * sin(x_k[0]), g * cos(x_k[0]) * cos(x_k[1])));
    /**** 协方差更新 ****/

    P_k = (Eigen::Matrix3d::Identity() - K * H) * P_k;

    /**********************(Debug cout)*********************/
    std::cout << "update end" << std::endl;
    std::cout << "x_k= " << x_k << std::endl;
    /**********************(Debug END)********************/
}

void ImuKalmanNode::estimator(Eigen::Vector3d gyro, Eigen::Vector3d acc, const double delta_t) // gyro.xyz
{
    predict(gyro);
    update(acc);
}

void ImuKalmanNode::positionPredict(Eigen::Vector2d u_position_k)
{

    B_position = Eigen::Matrix<double, 3, 2>{
        {delta_t * 2 * cos(x_k[2]), 0},
        {delta_t * 2 * sin(x_k[2]), 0},
        {0, 0}};
    /**** 状态预测 ****/
    x_position_k = A_position * x_position_k + B_position * u_position_k;
    /**** 状态协方差预测 ****/
    P_position_k = A_position * P_position_k * A_position.transpose() + W_position;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Node Ready.");
    rclcpp::spin(std::make_shared<ImuKalmanNode>(gyro_noise, acc_noise));
    rclcpp::shutdown();
    return 0;
}
