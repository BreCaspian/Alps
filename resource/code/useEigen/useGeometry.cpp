#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

int main(int argc, char **argv){

    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity(); // 单位阵

    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));
    std::cout.precision(3);

    std::cout << "rotation matrix : " << rotation_vector.matrix() << std::endl;

    rotation_matrix = rotation_vector.toRotationMatrix();

    // Angle Axis
    Eigen::Vector3d v(1, 0, 0);
    Eigen::Vector3d v_rotated = rotation_vector * v;
    std::cout << "(1, 0, 0) after rotation (by angle axis) = " << v_rotated.transpose() << std::endl;

    // Rotation Matrix
    v_rotated = rotation_matrix * v;
    std::cout << "(1, 0, 0) after rotation (by matrix) = " << v_rotated.transpose() << std::endl;

    // EulerAngles
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // Z-Y-X  Yaw-Pitch-Roll 4x4齐次变换矩阵
    std::cout << "Yaw Pitch Roll : " << euler_angles.transpose();

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(rotation_vector);
    T.pretranslate(Eigen::Vector3d(1, 3, 4));
    std::cout << "Transform Matrix = \n" << T.matrix() << std::endl;

    Eigen::Vector3d v_transformed = T * v;  // 平移 + 旋转
    std::cout << "v transform = " << v_transformed.transpose();

    // Quaternion
    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    std::cout << "Quaternion from rotation vector = " << q.coeffs().transpose() << std::endl; // (x, y, z, w)

    q = Eigen::Quaterniond(rotation_matrix);
    std::cout << "Quaternion from rotation matrix = " << q.coeffs().transpose() << std::endl; // (x, y, z, w)

    v_rotated = q * v; // 使用四元数直接旋转 向量v; v‘ = q * v * q.inverse()  q逆矩阵让3D向量从四维空间回到三维空间也就是避免实部产生，也就是群论中共轭作用

    std::cout << "(1, 0, 0) after rotation = " << v_rotated.transpose() << std::endl;

    std::cout << "Equal to " << (q * Eigen::Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << std::endl;


    return 0;




    // YAO YUZHUO 2025-12-05 00:19


}