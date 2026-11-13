// SPDX-License-Identifier: 0BSD
/// @file fast_eigen_val_vec.cuh
/// The closed-form eigendecomposition of a symmetric 3x3 matrix in device code, after D. Eberly,
/// "A Robust Eigensolver for 3 x 3 Symmetric Matrices", Geometric Tools,
/// https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf: the matrix scaled
/// by its largest entry, the eigenvalues from the trigonometric solution of its characteristic
/// cubic, the eigenvector of the eigenvalue farthest from the other two from the cross products
/// of the shifted matrix's rows, a second one in the plane orthogonal to it, the third their
/// cross product.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cuda_runtime.h>

namespace gca
{
/// The eigenvalues and unit eigenvectors of a symmetric 3x3 matrix.
struct symmetric_eigen_3x3
{
    /// The eigenvalues in ascending order; for a diagonal matrix, its diagonal entries in their
    /// order, unsorted.
    Eigen::Vector3f values;
    /// Column k: a unit eigenvector of values(k); the columns are orthonormal.
    Eigen::Matrix3f vectors;
};

/// Computes the unit eigenvector of a symmetric matrix for an eigenvalue of multiplicity one.
/// The rows of a - value I then span the plane orthogonal to the eigenvector, so the cross
/// product of two of them is parallel to it; the longest of the three is the most robust [S:
/// Eberly, section 5.2]. Among equally long ones it takes the first, whose length is the one it
/// divides by.
/// @param a the matrix, symmetric.
/// @param value the eigenvalue.
/// @return the eigenvector.
__device__ __forceinline__ Eigen::Vector3f eigenvector_from_rows(const Eigen::Matrix3f &a,
                                                                 float value)
{
    Eigen::Matrix3f shifted = a;
    shifted.diagonal().array() -= value;
    const Eigen::Vector3f row0 = shifted.row(0).transpose();
    const Eigen::Vector3f row1 = shifted.row(1).transpose();
    const Eigen::Vector3f row2 = shifted.row(2).transpose();

    Eigen::Matrix3f products;
    products.col(0) = row0.cross(row1);
    products.col(1) = row0.cross(row2);
    products.col(2) = row1.cross(row2);
    const Eigen::Vector3f squared_lengths(products.col(0).squaredNorm(),
                                          products.col(1).squaredNorm(),
                                          products.col(2).squaredNorm());

    Eigen::Index longest;
    const float longest_squared_length = squared_lengths.maxCoeff(&longest);
    return products.col(longest) / sqrtf(longest_squared_length);
}

/// Computes a unit eigenvector of a symmetric matrix orthogonal to a known one: it lies in the
/// plane spanned by an orthonormal pair u, v orthogonal to the known eigenvector, where it solves
/// the 2x2 eigenproblem of a restricted to that plane.
/// @param a the matrix, symmetric.
/// @param known a unit eigenvector of a.
/// @param value the eigenvalue of the eigenvector sought.
/// @return the eigenvector; u when the restricted problem is zero or NaN.
__device__ __forceinline__ Eigen::Vector3f eigenvector_orthogonal_to(const Eigen::Matrix3f &a,
                                                                     const Eigen::Vector3f &known,
                                                                     float value)
{
    const float abs_x = fabsf(known(0));
    const float abs_y = fabsf(known(1));
    const float larger = fmaxf(abs_x, abs_y);
    const float inv_length = 1.0f / sqrtf(larger * larger + known(2) * known(2));
    Eigen::Vector3f u = abs_x > abs_y ? Eigen::Vector3f(-known(2), 0.0f, known(0))
                                      : Eigen::Vector3f(0.0f, known(2), -known(1));
    u *= inv_length;
    const Eigen::Vector3f v = known.cross(u);
    const Eigen::Vector3f au = a * u;
    const Eigen::Vector3f av = a * v;

    // The restricted problem [m00 m01; m01 m11] (s, t) = 0; the eigenvector is s u + t v.
    float m00 = u.dot(au) - value;
    float m01 = u.dot(av);
    float m11 = v.dot(av) - value;
    const float abs_m00 = fabsf(m00);
    const float abs_m01 = fabsf(m01);
    const float abs_m11 = fabsf(m11);
    if (abs_m00 >= abs_m11)
    {
        if (!(fmaxf(abs_m00, abs_m01) > 0.0f))
        {
            return u;
        }
        // From the row (m00, m01), divided by its larger entry before normalizing.
        if (abs_m00 >= abs_m01)
        {
            m01 /= m00;
            m00 = 1.0f / sqrtf(1.0f + m01 * m01);
            m01 *= m00;
        }
        else
        {
            m00 /= m01;
            m01 = 1.0f / sqrtf(1.0f + m00 * m00);
            m00 *= m01;
        }
        return u * m01 - v * m00;
    }
    if (!(fmaxf(abs_m11, abs_m01) > 0.0f))
    {
        return u;
    }
    // From the row (m01, m11), divided by its larger entry before normalizing.
    if (abs_m11 >= abs_m01)
    {
        m01 /= m11;
        m11 = 1.0f / sqrtf(1.0f + m01 * m01);
        m01 *= m11;
    }
    else
    {
        m11 /= m01;
        m01 = 1.0f / sqrtf(1.0f + m11 * m11);
        m11 *= m01;
    }
    return u * m11 - v * m01;
}

/// Computes the eigenvalues and unit eigenvectors of a symmetric 3x3 matrix in closed form.
/// @param a the matrix, symmetric.
/// @return the eigenvalues and eigenvectors: for the zero matrix zeros and the axes; for a
///     diagonal matrix, and for one with a NaN off-diagonal entry, its diagonal and the axes.
__device__ __forceinline__ symmetric_eigen_3x3 solve_symmetric_eigen_3x3(const Eigen::Matrix3f &a)
{
    symmetric_eigen_3x3 result;
    const float largest = a.cwiseAbs().maxCoeff();
    if (largest == 0.0f)
    {
        result.values.setZero();
        result.vectors.setIdentity();
        return result;
    }

    const Eigen::Matrix3f s = a * (1.0f / largest);
    const float off_diagonal = s(0, 1) * s(0, 1) + s(0, 2) * s(0, 2) + s(1, 2) * s(1, 2);
    if (!(off_diagonal > 0.0f))
    {
        result.values = a.diagonal();
        result.vectors.setIdentity();
        return result;
    }

    // With s = q I + p B, the eigenvalues are q + p beta, beta = 2 cos(angle + 2 pi k / 3) the
    // roots of beta^3 - 3 beta - det(B) = 0.
    const float q = (s(0, 0) + s(1, 1) + s(2, 2)) / 3.0f;
    const float b00 = s(0, 0) - q;
    const float b11 = s(1, 1) - q;
    const float b22 = s(2, 2) - q;
    const float p = sqrtf((b00 * b00 + b11 * b11 + b22 * b22 + off_diagonal * 2.0f) / 6.0f);
    const float c00 = b11 * b22 - s(1, 2) * s(1, 2);
    const float c01 = s(0, 1) * b22 - s(1, 2) * s(0, 2);
    const float c02 = s(0, 1) * s(1, 2) - b11 * s(0, 2);
    const float det = (b00 * c00 - s(0, 1) * c01 + s(0, 2) * c02) / (p * p * p);
    const float half_det = fminf(fmaxf(det * 0.5f, -1.0f), 1.0f);
    const float angle = acosf(half_det) / 3.0f;
    constexpr float two_thirds_pi = 2.09439510239319549f;
    const float beta2 = cosf(angle) * 2.0f;
    const float beta0 = cosf(angle + two_thirds_pi) * 2.0f;
    const float beta1 = -(beta0 + beta2);
    const Eigen::Vector3f values(q + p * beta0, q + p * beta1, q + p * beta2);

    // The eigenvalue farthest from the other two has multiplicity one: the largest when
    // half_det >= 0, else the smallest.
    if (half_det >= 0.0f)
    {
        result.vectors.col(2) = eigenvector_from_rows(s, values(2));
        result.vectors.col(1) = eigenvector_orthogonal_to(s, result.vectors.col(2), values(1));
        result.vectors.col(0) = result.vectors.col(1).cross(result.vectors.col(2));
    }
    else
    {
        result.vectors.col(0) = eigenvector_from_rows(s, values(0));
        result.vectors.col(1) = eigenvector_orthogonal_to(s, result.vectors.col(0), values(1));
        result.vectors.col(2) = result.vectors.col(0).cross(result.vectors.col(1));
    }
    result.values = values * largest;
    return result;
}
} // namespace gca
