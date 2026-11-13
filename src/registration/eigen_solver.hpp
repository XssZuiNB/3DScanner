#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

static Eigen::Matrix4f transform_vec6d_to_mat4f(const Eigen::Vector<double, 6> &input)
{
    Eigen::Matrix4d output;
    output.setIdentity();
    output.block<3, 3>(0, 0) = (Eigen::AngleAxisd(input(2), Eigen::Vector3d::UnitZ()) *
                                Eigen::AngleAxisd(input(1), Eigen::Vector3d::UnitY()) *
                                Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitX()))
                                   .matrix();
    output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
    return output.cast<float>();
}

namespace gca
{
inline Eigen::Matrix4f solve_JTJ_JTr(const Eigen::Matrix<float, 6, 6> &JTJ,
                                     const Eigen::Vector<float, 6> &JTr)
{
    const Eigen::Matrix<double, 6, 6> JTJ_d = JTJ.cast<double>();
    const Eigen::Vector<double, 6> JTr_d = JTr.cast<double>();

    // equation 21 in paper of color icp
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(JTJ_d);
    if (ldlt.info() != Eigen::Success)
    {
        return Eigen::Matrix4f::Identity();
    }

    const Eigen::Vector<double, 6> se3_vec = ldlt.solve(-JTr_d);
    return transform_vec6d_to_mat4f(se3_vec);
}
} // namespace gca
