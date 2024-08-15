# imu gyro&acc ekf
融合IMU的加速度计和陀螺仪
角速度计积分作为预测，加速度计作为更新
如果IMU做匀速运动，加速度计测量的就是重力（方向）。当然实际上，并不可能做匀速运动，可以当做噪声处理。(所以只能用于低动态的情况下使用)，当晃动越大越相信角速度的值，接近于静止或匀速就相信加速度计的值
## 建模
### 状态量
roll,pitch,yaw
$$
x=
\left[
\begin{matrix}
\phi\\
\theta\\
\psi\\
\end{matrix}
\right]
\tag{1}
$$
### 输入量
$$
u=
\left[
\begin{matrix}
\omega_x\\
\omega_y\\
\omega_z\\
\end{matrix}
\right]
\tag{1}
$$
### 运动方程 
运动方程是线性的
$$
x_k=Ax_{k-1}+Bu_k+w
$$
状态矩阵
$$
A=
\left[
\begin{matrix}
1&0&0\\
0&1&0\\
0&0&1
\end{matrix}
\right]
\tag{1}
$$
输入矩阵
$$
B=
\left[
\begin{matrix}
\Delta_t&0&0\\
0&\Delta_t&0\\
0&0&\Delta_t\\
\end{matrix}
\right]
\tag{1}
$$

### 观测量
$$
z=
\left[
\begin{matrix}
a_x\\
a_y\\
a_z
\end{matrix}
\right]
\tag{1}
$$

### 观测方程
观测方程是非线性的
$$
z_k=
\left[
\begin{matrix}
a_x\\
a_y\\
a_z
\end{matrix}
\right]
=h(x_k)=
\left[
\begin{matrix}
-gsin(\theta)\\
gcos(\theta)sin(\phi)\\
gcos(\theta)cos(\phi)
\end{matrix}
\right]+v
$$

- 线性化观测方程 使用泰勒公式保留一阶雅可比矩阵
$$
H = \frac{\partial h(x)}{\partial x}=
\left[
\begin{matrix}
\frac{\partial a_x}{\partial \phi}&\frac{\partial a_x}{\partial \theta}&\frac{\partial a_x}{\partial \psi}\\
\frac{\partial a_y}{\partial \phi}&\frac{\partial a_y}{\partial \theta}&\frac{\partial a_y}{\partial \psi}\\
\frac{\partial a_z}{\partial \phi}&\frac{\partial a_z}{\partial \theta}&\frac{\partial a_z}{\partial \psi}\\
\end{matrix}
\right]
=
\left[
\begin{matrix}
0&-gsin(\theta&0\\
gcos(\theta)cos(\phi)&-gsin(\theta)sin(\phi)&0\\
gcos(\theta)sin(\phi)&-gsin(\theta)cos(\phi)&0\\
\end{matrix}
\right]
$$

## EKF
### 输入噪声w和观测噪声v
$$
w=
\left[
\begin{matrix}
acc\_noise_x^2&0&0\\
0&acc\_noise_y^2&0\\
0&0&acc\_noise_z^2\\
\end{matrix}
\right]
$$
- acc_noise 标定
acc_noise 为标准差
$$
acc\_noise=
 \sqrt {\frac {1}{N}\sum _ {i=1}^ {n}(x_ {i}-\mu )^ {2}} 
$$

$$
v=
\left[
\begin{matrix}
gyro\_noise_x^2&0&0\\
0&gyro\_noise_y^2&0\\
0&0&gyro\_noise_z^2\\
\end{matrix}
\right]
$$
- gyro_noise 标定
gyro_noise 为标准差
$$
acc\_noise=
 \sqrt {\frac {1}{N}\sum _ {i=1}^ {n}(x_ {i}-\mu )^ {2}} 
$$
### init state
$$
x_0=
\left[
\begin{matrix}
0\\
0\\
0
\end{matrix}
\right]
\tag{1}
$$

$$
P_0
\left[
\begin{matrix}
\epsilon&0&0\\
0&\epsilon&0\\
0&0&\epsilon
\end{matrix}
\right]\\
\epsilon为无穷小值
$$

### predict
$$
 \overline x_ {k} = A_ {k-1}  \widehat {x}_ {k-1} + B_ {k-1}  u_ {k} \\
\overline P_{k} = A_ {k-1}  P_ {k-1}  A_ {k-1}^ {T} + Q_ {k-1} 
$$

### update
$$
 K_ {k} = P_ {k}^ {-}  H_ {k}^ {T} ( H_ {k}  P_ {k}^ {-}  H_ {k}^ {T} + R_ {k} )\\
 将\overline x_k 带入H
$$

$$
 \widehat {x}_ {k} = \overline{x}_{k} + K_ {k} ( z_ {k} -h( \overline {x}_ {k} ))\\
 kf 和ekf不同的地方
$$

$$
 \widehat P_ {k} =(I- K_ {k}  H_ {k} ) \overline P_ {k} \\
 将\overline x_k 带入H
$$
