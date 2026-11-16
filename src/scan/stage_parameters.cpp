// SPDX-License-Identifier: 0BSD
/// @file stage_parameters.cpp
/// The stages' parameters of an object class's set (see stage_parameters.hpp).

#include "scan/stage_parameters.hpp"

#include "geometry/frame_pyramid.hpp"

#include <cstddef>

namespace gca
{
registration_parameters registration_parameters_of(const object_class_parameters &set) noexcept
{
    registration_parameters p{};
    for (std::size_t level = 0; level < frame_pyramid::level_count; level++)
        p.levels[level] = {static_cast<float>(set.max_distance_m[level]),
                           set.max_iterations[level]};
    p.huber_threshold_m = static_cast<float>(set.huber_threshold_m);
    p.point_to_plane_weight = static_cast<float>(set.point_to_plane_weight);
    return p;
}

tracking_parameters tracking_parameters_of(const object_class_parameters &set) noexcept
{
    return {{{static_cast<float>(set.min_depth_m), static_cast<float>(set.max_depth_m)},
             {set.normals_radius_px, set.normals_min_points,
              static_cast<float>(set.normals_max_incidence_deg)}},
            registration_parameters_of(set),
            {set.min_fitness, set.max_inlier_rmse_m, set.max_turn_rate_deg_s, set.max_speed_m_s}};
}

sparse_tsdf_parameters sparse_tsdf_parameters_of(const object_class_parameters &set,
                                                 fused_model model) noexcept
{
    const double voxel_size_m =
        model == fused_model::live ? set.live_voxel_size_m : set.full_voxel_size_m;
    return {set.memory_budget_bytes,
            static_cast<float>(voxel_size_m),
            {static_cast<float>(set.depth_noise_per_m),
             static_cast<float>(set.truncation_noise_multiple),
             static_cast<float>(set.truncation_min_voxels)},
            static_cast<float>(set.max_depth_m)};
}
} // namespace gca
