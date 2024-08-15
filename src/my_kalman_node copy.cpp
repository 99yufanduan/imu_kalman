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
const double acc_noise = 1e-8;

const double epsilon = 1e-6; // 无穷小值
const double g = 9.81;       // 单位是 m/s^2，表示重力加速度

const double w_k = 0.1; // 将如果IMU做匀速运动，加速度计测量的就是重力（方向）。当然实际上，并不可能做匀速运动，可以当做过程噪声处理

class ImuKalmanNode : public rclcpp::Node
{
private:
    /* data */
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_;

    void imuCallback(const sensor_msgs::msg::Imu::UniquePtr imu_in);
    void publishPose();
    void predict(Eigen::Vector3d u_k, const double delta_t);
    void update(Eigen::Vector3d z_k);
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    Eigen::Vector3d x_k{
        {0, 0, 0}}; // rpy

    // 状态转移矩阵
    Eigen::Matrix3d A{
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}};

    // 输入矩阵
    Eigen::Matrix3d B;
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

public:
    ImuKalmanNode(const double gyro_noise, const double acc_noise) : Node("imu_kalman_node")
    {
        x_k = Eigen::Vector3d(0, 0, 1); // rpy
        W = Eigen::Matrix3d{
            {gyro_noise * gyro_noise, 0, 0},
            {0, gyro_noise * gyro_noise, 0},
            {0, 0, gyro_noise * gyro_noise}};
        V = Eigen::Matrix3d{
            {acc_noise * acc_noise, 0, 0},
            {0, acc_noise * acc_noise, 0},
            {0, 0, 0}};
        sub_ = this->create_subscription<sensor_msgs::msg::Imu>("/Imu_data", 10, std::bind(&ImuKalmanNode::imuCallback, this, _1));
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
    static Eigen::Vector3d acc_pre = Eigen::Vector3d(0, 0, 9.73);
    Eigen::Vector3d acc(imu_in->linear_acceleration.x, imu_in->linear_acceleration.y, imu_in->linear_acceleration.z);
    Eigen::Vector3d gyro(imu_in->angular_velocity.x, imu_in->angular_velocity.y, imu_in->angular_velocity.z);
    static double time_pre = 0;
    const double delta_t = (imu_in->header.stamp.nanosec - time_pre) / 1e9; // 将时间差转换为秒
    estimator(gyro, acc, 0.01);
    time_pre = imu_in->header.stamp.nanosec;

    std::cout << "hhh " << delta_t << std::endl;
    double timestamp = imu_in->header.stamp.sec;

    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = "world";
    Eigen::Quaterniond q = rpyToQuaternion(x_k[0], x_k[1], x_k[2]);
    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();

    std::cout << "20:16:18: " << x_k[0] << "  " << x_k[1] << "  " << x_k[2] << std::endl;

    geometry_msgs::msg::TransformStamped t; // 四元数+平移

    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "world";
    t.child_frame_id = "imu";
    t.transform.translation.x = 0;
    t.transform.translation.y = 0;
    t.transform.translation.z = 0;
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

void ImuKalmanNode::predict(Eigen::Vector3d gyro, const double delta_t) // gyro.xyz
{
    B = Eigen::Matrix3d{{delta_t, 0, 0},
                        {0, delta_t, 0},
                        {0, 0, delta_t}};
    /**** 状态预测 ****/
    x_k = A * x_k + B * gyro;
    /**** 状态协方差预测 ****/
    P_k = A * P_k * A.transpose() + W;
}

void ImuKalmanNode::update(Eigen::Vector3d acc) // z_k 为acc.xyz
{
    H = Eigen::Matrix3d{{0, -1 * g * cos(x_k[1]), 0},
                        {g * cos(x_k[1]) * cos(x_k[0]), -1 * g * sin(x_k[1]) * sin(x_k[0]), 0},
                        {-1 * g * cos(x_k[1]) * sin(x_k[0]), -1 * g * sin(x_k[1]) * cos(x_k[0]), 0}}; // 观测方程线性化的雅可比矩阵
    Eigen::Matrix3d K;
    Eigen::Matrix3d E = Eigen::Matrix3d{
        {exp(fabs(acc.norm() - g)) - 1, 0, 0},
        {0, exp(fabs(acc.norm() - g)) - 1, 0},
        {0, 0, exp(fabs(acc.norm() - g)) - 1}};
    std::cout << "14:47:01: " << acc.norm() << std::endl;

    K = P_k * H.transpose() * (H * P_k * H.transpose() + V).inverse(); // 求卡尔曼增益，eigen的求逆会根据矩阵类型自动选择求逆方法
    /**** 状态更新 ****/
    x_k = x_k + K * (acc - Eigen::Vector3d(-1 * g * cos(x_k[0]) * sin(x_k[1]), g * sin(x_k[0]), g * cos(x_k[0]) * cos(x_k[1])));
    /**** 协方差更新 ****/
    P_k = (Eigen::Matrix3d::Identity() - K * H) * P_k;
}

void ImuKalmanNode::estimator(Eigen::Vector3d gyro, Eigen::Vector3d acc, const double delta_t) // gyro.xyz
{
    predict(gyro, delta_t);
    update(acc);
}

void positionPredict(const double delta_t)
{
    Eigen::Vector3d x_k(0, 0, 0); // xyz
    Eigen::Vector3d x_k_pre(0, 0, 0);

    const double b = 0.3;   // 轮距
    const double d_r = 0.3; // 右轮的位移
    const double d_l = 0.3; // 左轮的位移
    const double wheel_error = 0.01;

    // 状态转移矩阵
    Eigen::Matrix3d A{
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}};

    // 输入矩阵
    Eigen::Matrix3d B{{delta_t * cos(x_k[2] + (d_r - d_l) / b), 0},
                      {delta_t * cos(x_k[2] + (d_r - d_l) / b), 0},
                      {0, 1}};

    // 状态协方差，初始为非常小
    Eigen::Matrix3d P_k{{epsilon, 0, 0},
                        {0, epsilon, 0},
                        {0, 0, epsilon}};

    // 输入噪声协方差,假设陀螺仪的的角速度的误差为正负0.01m/s，
    Eigen::Matrix3d W{{wheel_error * wheel_error, 0, 0},
                      {0, wheel_error * wheel_error, 0},
                      {0, 0, 0}};
    Eigen::Vector3d u_k(0, 0); // v,w

    /**** 状态预测 ****/
    x_k = A * x_k_pre + B * u_k;
    /**** 状态协方差预测 ****/
    P_k = A * P_k * A.transpose() + W;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Node Ready.");
    rclcpp::spin(std::make_shared<ImuKalmanNode>(gyro_noise, acc_noise));
    rclcpp::shutdown();
    return 0;
}
