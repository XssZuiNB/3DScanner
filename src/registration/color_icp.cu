#include "color_icp.hpp"

#include "geometry/float3_ops.cuh"
#include "geometry/geometry_util.cuh"
#include "registration/cuda_color_icp_build_least_square.cuh"
#include "registration/cuda_compute_color_gradient.cuh"
#include "registration/eigen_solver.hpp"
#include "types/console_color.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace gca
{
color_icp::color_icp(size_t maximum_iterations, float max_correspondence_distance,
                     float search_radius_color_gradient)
    : m_max_iter(maximum_iterations)
    , m_max_corr_dist(max_correspondence_distance)
    , m_color_icp_done(false)
    , m_source_pc(nullptr)
    , m_target_pc(nullptr)
    , m_search_radius_color_gradient(search_radius_color_gradient)
    , m_color_gradient_done(false)
    , m_transformation_matrix(Eigen::Matrix4f::Identity())
{
}

void color_icp::set_source_point_cloud(const std::shared_ptr<const gca::point_cloud> new_pc)
{
    m_color_icp_done = false;
    m_transformation_matrix.setIdentity();

    m_source_pc = std::make_shared<gca::point_cloud>(*new_pc);
}

void color_icp::set_target_point_cloud(const std::shared_ptr<const gca::point_cloud> new_pc)
{
    m_color_icp_done = false;
    m_color_gradient_done = false;
    m_transformation_matrix.setIdentity();

    m_target_pc = new_pc;
}

bool color_icp::compute_color_gradient_target()
{
    auto &pts_tgt = m_target_pc->get_points();
    auto &normals_tgt = m_target_pc->get_normals();

    if (!m_color_gradient_done)
    {
        auto min_bound_tgt_with_padding =
            m_target_pc->get_min_bound() - make_float3(1.5f * m_search_radius_color_gradient,
                                                       1.5f * m_search_radius_color_gradient,
                                                       1.5f * m_search_radius_color_gradient);
        auto max_bound_tgt_with_padding =
            m_target_pc->get_max_bound() + make_float3(1.5f * m_search_radius_color_gradient,
                                                       1.5f * m_search_radius_color_gradient,
                                                       1.5f * m_search_radius_color_gradient);

        if (m_search_radius_color_gradient * std::numeric_limits<int>::max() <
            max(max(max_bound_tgt_with_padding.x - min_bound_tgt_with_padding.x,
                    max_bound_tgt_with_padding.y - min_bound_tgt_with_padding.y),
                max_bound_tgt_with_padding.z - min_bound_tgt_with_padding.z))
        {
            std::cout << YELLOW << "Radius is too small!" << std::endl;
            return false;
        }

        auto err = cuda_compute_color_gradient(
            m_target_pc_color_gradient, pts_tgt, normals_tgt, min_bound_tgt_with_padding,
            max_bound_tgt_with_padding, m_search_radius_color_gradient);

        if (err != ::cudaSuccess)
        {
            return false;
        }

        m_color_gradient_done = true;
    }
    return true;
}

bool color_icp::align()
{
    if (m_color_icp_done)
    {
        return true;
    }

    if (!m_source_pc)
    {
        std::cout << YELLOW << "Need to set a valid source point cloud" << std::endl;
        return false;
    }

    if (!m_target_pc)
    {
        std::cout << YELLOW << "Need to set a valid target point cloud" << std::endl;
        return false;
    }

    if (!m_target_pc->has_normals())
    {
        std::cout << YELLOW << "Need to estimate normals of target point cloud" << std::endl;
        return false;
    }

    if (!compute_color_gradient_target())
    {
        std::cout << YELLOW << "Compute color gradient failed!" << std::endl;
        return false;
    }

    Eigen::Matrix<float, 6, 6> JTJ;
    Eigen::Vector<float, 6> JTr;

    for (size_t i = 0; i < m_max_iter; i++)
    {
        auto nn_src_tgt = gca::point_cloud::nn_search(*m_source_pc, *m_target_pc, m_max_corr_dist);
        if (!nn_src_tgt.size())
        {
            break;
        }

        auto err = cuda_build_gauss_newton_color_icp(
            JTJ, JTr, m_source_pc->get_points(), m_target_pc->get_points(),
            m_target_pc->get_normals(), m_target_pc_color_gradient, nn_src_tgt, m_color_icp_lambda);
        if (err != ::cudaSuccess)
        {
            check_cuda_error(err, __FILE__, __LINE__);
            std::cout << YELLOW << "Gauss Newton failed!" << std::endl;
            m_transformation_matrix.setIdentity();
            return false;
        }

        const Eigen::Matrix4f this_transformation_matrix = solve_JTJ_JTr(JTJ, JTr);

        m_source_pc->transform(this_transformation_matrix);

        m_transformation_matrix = this_transformation_matrix * m_transformation_matrix;
        // check convergence
        // Rotation
        float cos_theta =
            0.5 * (this_transformation_matrix(0, 0) + this_transformation_matrix(1, 1) +
                   this_transformation_matrix(2, 2) - 1);
        // Translation square
        float translation_square =
            this_transformation_matrix(0, 3) * this_transformation_matrix(0, 3) +
            this_transformation_matrix(1, 3) * this_transformation_matrix(1, 3) +
            this_transformation_matrix(2, 3) * this_transformation_matrix(2, 3);
        if (cos_theta >= m_rotation_thres && translation_square <= m_translations_thres_square)
        {
            std::cout << GREEN << "Color ICP converged!" << std::endl;
            break;
        }
    }

    m_color_icp_done = true;
    return true;
}

const Eigen::Matrix4f &color_icp::get_final_transformation_matrix() const
{
    if (!m_color_icp_done)
    {
        std::cout << YELLOW << "Color ICP is not done yet! Identity Matrix returned!" << std::endl;
    }
    return m_transformation_matrix;
}
} // namespace gca
