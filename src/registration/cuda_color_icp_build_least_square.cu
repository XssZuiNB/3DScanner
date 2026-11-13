#include "cuda_color_icp_build_least_square.cuh"

#include "geometry/float3_eigen.cuh"
#include "geometry/float3_ops.cuh"

#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace gca
{
/// One point's or a sum's J^T J, J^T r and r^2 of the Gauss-Newton step.
using least_squares_terms =
    thrust::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Vector<float, 6>, float>;

struct compute_JTJ_JTr_and_r_gauss_newton_functor
{
    compute_JTJ_JTr_and_r_gauss_newton_functor(
        const thrust::device_vector<gca::point_t> &tgt_points,
        const thrust::device_vector<float3> &tgt_normals,
        const thrust::device_vector<float3> &tgt_color_gradient, const float lambda)
        : m_tgt_points_ptr(thrust::raw_pointer_cast(tgt_points.data()))
        , m_tgt_normals_ptr(thrust::raw_pointer_cast(tgt_normals.data()))
        , m_tgt_color_gradient_ptr(thrust::raw_pointer_cast(tgt_color_gradient.data()))
        , m_sqrt_lambda_geometry(sqrtf(lambda))
        , m_sqrt_lambda_color(sqrtf(1.0f - lambda))
    {
    }

    const gca::point_t *m_tgt_points_ptr;
    const float3 *m_tgt_normals_ptr;
    const float3 *m_tgt_color_gradient_ptr;
    const float m_sqrt_lambda_geometry;
    const float m_sqrt_lambda_color;

    __forceinline__ __device__ least_squares_terms
    operator()(const thrust::tuple<gca::point_t, gca::index_t> &pts_and_nn) const
    {
        auto pts_src = thrust::get<0>(pts_and_nn);
        auto nn_idx_tgt = thrust::get<1>(pts_and_nn);
        if (nn_idx_tgt < 0)
        {
            return least_squares_terms(Eigen::Matrix<float, 6, 6>::Zero(),
                                       Eigen::Vector<float, 6>::Zero(), 0.0f);
        }

        Eigen::Vector<float, 6> J_geometry;
        Eigen::Vector<float, 6> J_color;
        float r_geometry;
        float r_color;

        compute_jacobian_and_residual(J_geometry, J_color, r_geometry, r_color, pts_src,
                                      nn_idx_tgt);

        const Eigen::Matrix<float, 6, 6> JTJ =
            J_geometry * J_geometry.transpose() + J_color * J_color.transpose();
        const Eigen::Vector<float, 6> JTr = J_geometry * r_geometry + J_color * r_color;
        float r2 = r_geometry * r_geometry + r_color * r_color; // r square

        return least_squares_terms(JTJ, JTr, r2);
    }

private:
    __forceinline__ __device__ void compute_jacobian_and_residual(
        Eigen::Vector<float, 6> &J_geometry, Eigen::Vector<float, 6> &J_color, float &r_geometry,
        float &r_color, const gca::point_t &pts_src, const gca::index_t &nn_idx_tgt) const
    {
        const auto &nn_pts_tgt = m_tgt_points_ptr[nn_idx_tgt];
        const auto &normal = m_tgt_normals_ptr[nn_idx_tgt];
        const auto &color_gradient_tgt = m_tgt_color_gradient_ptr[nn_idx_tgt];

        // r_geometry without weight, this is also needed in color part, so compute it here firstly.
        auto r_geometry_ = dot((pts_src.coordinates - nn_pts_tgt.coordinates), normal);

        // geometric
        // jacobian derivation see Master thesis Shengsi Xu
        J_geometry.head<3>() =
            to_eigen(m_sqrt_lambda_geometry * cross(pts_src.coordinates, normal));
        J_geometry.tail<3>() = to_eigen(m_sqrt_lambda_geometry * normal);

        // residual
        r_geometry = m_sqrt_lambda_geometry * r_geometry_;

        // color
        // jacobian
        // paper equation 28, 29
        const Eigen::Vector3f normal_vec = to_eigen(normal);

        // J_f(s)
        // from equation 9: df(s)/ds = I - n * nT
        const Eigen::Matrix3f df_div_ds =
            Eigen::Matrix3f::Identity() - normal_vec * normal_vec.transpose();

        // from equation 8: dC_p(u)/du = d_pT -> equation 29 = d_pT * (I - n * nT) * J_s(xi)
        //  d_pT * (I - n * nT) is called here as dc_x_df
        const Eigen::RowVector3f dc_x_df_row = to_eigen(color_gradient_tgt).transpose() * df_div_ds;
        const float3 dc_x_df = to_float3(dc_x_df_row.transpose());

        // derivation of the formula J_s(xi) see Masterthesis von Shengsi Xu
        /* J_s(xi) = | 0   z  -y   1   0   0 |
                     |-z   0   x   0   1   0 |
                     | y  -x   0   0   0   1 |
                     | 0   0   0   0   0   0 |

           J_color_fir = d_pT * (I - n * nT) * | 0   z  -y |
                                               |-z   0   x |
                                               | y  -x   0 |
                       = | 0   -z  y | * (d_pT * (I - n * nT))T
                         | z   0  -x |
                         |-y   x   0 |
                       = p.coordinates X (d_pT * (I - n * nT))

           J_color_sec = d_pT * (I - n * nT) * | 1   0   0 | = d_pT * (I - n * nT)
                                               | 0   1   0 |
                                               | 0   0   1 |
        */
        J_color.head<3>() = to_eigen(m_sqrt_lambda_color * cross(pts_src.coordinates, dc_x_df));
        J_color.tail<3>() = to_eigen(m_sqrt_lambda_color * dc_x_df);
        // residual
        auto indensity_pts_src = pts_src.color.to_intensity();
        auto indensity_pts_tgt = nn_pts_tgt.color.to_intensity();
        // project src point onto the tangent plane of taget point
        auto proj_coordinates = pts_src.coordinates - r_geometry_ * normal;
        // get projected intensity
        auto indensity_proj = indensity_pts_tgt +
                              dot(color_gradient_tgt, (proj_coordinates - nn_pts_tgt.coordinates));
        r_color = m_sqrt_lambda_color * (indensity_proj - indensity_pts_src); // color
    }
};

struct add_JTJ_JTr_and_RMSE_functor
{
    __forceinline__ __device__ least_squares_terms operator()(const least_squares_terms &fir,
                                                              const least_squares_terms &sec) const
    {
        return least_squares_terms(thrust::get<0>(fir) + thrust::get<0>(sec),
                                   thrust::get<1>(fir) + thrust::get<1>(sec),
                                   thrust::get<2>(fir) + thrust::get<2>(sec));
    }
};

::cudaError_t cuda_build_gauss_newton_color_icp(
    Eigen::Matrix<float, 6, 6> &JTJ, Eigen::Vector<float, 6> &JTr,
    const thrust::device_vector<gca::point_t> &src_points,
    const thrust::device_vector<gca::point_t> &tgt_points,
    const thrust::device_vector<float3> &tgt_normals,
    const thrust::device_vector<float3> &tgt_color_gradient,
    const thrust::device_vector<gca::index_t> &nn_src_tgt, const float lambda)
{
    if (lambda > 1.0f)
    {
        return ::cudaErrorInvalidValue;
    }

    auto n_points_src = src_points.size();
    if (nn_src_tgt.size() != n_points_src)
    {
        return ::cudaErrorInvalidValue;
    }

    if (tgt_points.size() != tgt_normals.size() || tgt_points.size() != tgt_color_gradient.size())
    {
        return ::cudaErrorInvalidValue;
    }

    // for init
    const least_squares_terms zero(Eigen::Matrix<float, 6, 6>::Zero(),
                                   Eigen::Vector<float, 6>::Zero(), 0.0f);

    auto zipped_begin =
        thrust::make_zip_iterator(thrust::make_tuple(src_points.begin(), nn_src_tgt.begin()));

    auto zipped_end =
        thrust::make_zip_iterator(thrust::make_tuple(src_points.end(), nn_src_tgt.end()));

    auto result = thrust::transform_reduce(zipped_begin, zipped_end,
                                           compute_JTJ_JTr_and_r_gauss_newton_functor(
                                               tgt_points, tgt_normals, tgt_color_gradient, lambda),
                                           zero, add_JTJ_JTr_and_RMSE_functor());
    JTJ = thrust::get<0>(result);
    JTr = thrust::get<1>(result);

    return ::cudaSuccess;
}
} // namespace gca
