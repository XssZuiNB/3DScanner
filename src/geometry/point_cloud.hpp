#pragma once

#include "cuda_container/cuda_container.hpp"
#include "geometry/type.hpp"

#include <memory>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

namespace gca
{
class point_cloud
{
public:
    point_cloud() = default;
    point_cloud(const point_cloud &other);

    std::vector<point_t> download() const;

    bool has_normals() const
    {
        return m_has_normals;
    }

    size_t points_number() const
    {
        return m_points.size();
    }

    bool estimate_normals(float search_radius);

    const thrust::device_vector<gca::point_t> &get_points() const;
    const thrust::device_vector<float3> &get_normals() const;
    const float3 &get_min_bound() const;
    const float3 &get_max_bound() const;

    std::vector<float3> download_normals() const;

    void transform(const mat4x4 &trans_mat);

    std::shared_ptr<point_cloud> voxel_grid_down_sample(float voxel_size);

    static std::shared_ptr<point_cloud> create_from_rgbd(const gca::cuda_depth_frame &depth,
                                                         const gca::cuda_color_frame &color,
                                                         const gca::cuda_camera_param &param,
                                                         float threshold_min_in_meter = 0.0f,
                                                         float threshold_max_in_meter = 10.0f);

    /// Uploads host points into a new cloud, the counterpart of download(). Every point keeps
    /// its coordinates, color and property as given, in the given order; the cloud has no
    /// normals and no cached bounds (estimate_normals() and the bound getters compute them).
    /// @param points the points.
    /// @return the new cloud.
    static std::shared_ptr<point_cloud> create_from_points(const std::vector<gca::point_t> &points);

    static thrust::device_vector<gca::index_t> nn_search(const gca::point_cloud &query_pc,
                                                         const gca::point_cloud &reference_pc,
                                                         float radius);

    ~point_cloud() = default;

private:
    void compute_min_max_bound() const;

private:
    thrust::device_vector<gca::point_t> m_points;
    thrust::device_vector<float3> m_normals;
    bool m_has_normals = false;
    mutable bool m_has_bound = false;
    mutable float3 m_min_bound;
    mutable float3 m_max_bound;
};
} // namespace gca
