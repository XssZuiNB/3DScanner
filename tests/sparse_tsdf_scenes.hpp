// SPDX-License-Identifier: 0BSD
/// @file sparse_tsdf_scenes.hpp
/// What the sparse TSDF's tests that fuse frames share (fusion/sparse_tsdf.hpp): the budget of a
/// case's sparse TSDF, an object class's model as a case takes it, the fusion of a frame at its
/// pose, the frames of walls facing the camera, the scenes of a sphere resting on a plane with the
/// walks they are seen along, and what a point of such a scene should be: its nearest known
/// surface, its distance, normal and texture there. A failure is a failed check, counted by
/// check.hpp. Header only; each test program is one translation unit.

#pragma once

#include "camera/frame.hpp"
#include "check.hpp"
#include "fusion/sparse_tsdf.hpp"
#include "geometry/organized_points.hpp"
#include "gpu/step_result.hpp"
#include "scan/object_class.hpp"
#include "sparse_tsdf_helpers.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace gca_test
{
/// The blocks of a case's sparse TSDF, and its budget: 65536 x 4100 + 262144 x 16 + 16 bytes
/// [S: computed, docs/methods/sparse-tsdf.md, "The pool and the memory budget"], 273 MB, and a
/// room for one frame of 196608 blocks, above the bound of every case's parameters.
inline constexpr std::uint32_t case_blocks = 65536;
inline constexpr std::uint64_t case_budget = 272'891'920;

/// Gives an object class's model as a case takes it.
struct model_case
{
    std::string name;                       ///< the case, named in the check lines
    gca::sparse_tsdf_parameters tsdf;       ///< the sparse TSDF's parameters
    gca::organized_points_parameters range; ///< the depth range of the organized points
};

/// Gives a model of an object class: its live or its full voxel size and its depth range, the
/// sparse TSDF's far depth the range's or nearer, the range its tests were measured at
/// (measured_range_of()).
/// @param name [in] the case's name.
/// @param kind [in] the object class.
/// @param full_model [in] whether the full models' voxel size, else the live model's.
/// @param nearer_far_m [in] the sparse TSDF's far depth, in m, nearer than the range's; none for
///        the range's own.
/// @return the case.
inline model_case model_of(const std::string &name, gca::object_class kind, bool full_model,
                           std::optional<double> nearer_far_m = std::nullopt)
{
    const gca::object_class_parameters p = gca::parameters_of(kind);
    const double voxel_m = full_model ? p.full_voxel_size_m : p.live_voxel_size_m;
    const depth_range range = measured_range_of(kind);
    return {name,
            {case_budget, static_cast<float>(voxel_m), method_truncation,
             static_cast<float>(nearer_far_m.value_or(range.far_m))},
            {static_cast<float>(range.near_m), static_cast<float>(range.far_m)}};
}

/// Uploads a frame, deprojects it and integrates it at its pose, and checks that the work was
/// queued.
/// @param tsdf [in,out] the sparse TSDF.
/// @param buffers [in,out] the GPU frame and organized points.
/// @param host [in] the frame.
/// @param range [in] the organized points' depth range.
/// @param pose [in] the frame's pose, camera frame to the models' frame.
/// @param stream [in] the stream.
/// @param what [in] the case, named in the check line.
inline void fuse(gca::sparse_tsdf &tsdf, frame_buffers &buffers, const gca::frame &host,
                 const gca::organized_points_parameters &range, const Eigen::Isometry3d &pose,
                 cudaStream_t stream, const std::string &what)
{
    const gca::step_result up = buffers.frame.upload(host, stream);
    const gca::step_result made = buffers.points.deproject(buffers.frame, range, stream);
    const gca::step_result fused =
        tsdf.integrate(buffers.frame, buffers.points, as_camera_pose(pose), stream);
    check(up.done && made.done && fused.done, (what + ": upload, deprojection and integration, " +
                                               up.message + made.message + fused.message)
                                                  .c_str());
}

/// Builds a frame of a wall facing the camera.
/// @param depth_of [in] the depth value of pixel (u, v), in mm.
/// @param color [in] the wall's color.
/// @return the frame, 640x400 with the camera's intrinsics.
template <typename Depth>
gca::frame wall_frame(Depth depth_of, const std::array<std::uint8_t, 3> &color)
{
    return built_frame(small_description, depth_of,
                       [&color](std::uint32_t, std::uint32_t) { return color; });
}

/// Gives a frame of a wall facing the camera at one depth.
/// @param depth_mm [in] the depth of every pixel, in mm.
/// @param color [in] the wall's color.
/// @return the frame, 640x400 with the camera's intrinsics.
inline gca::frame flat_wall(std::uint16_t depth_mm, const std::array<std::uint8_t, 3> &color)
{
    return wall_frame([depth_mm](std::uint32_t, std::uint32_t) { return depth_mm; }, color);
}

/// Gives the walk of a scene's case.
/// @param walk [in] the walk.
/// @return its poses; none, with a failed check, when walk_around() refuses it.
inline std::vector<Eigen::Isometry3d> poses_of(const synthetic_walk &walk)
{
    const std::optional<std::vector<Eigen::Isometry3d>> poses = walk_around(walk);
    check(poses.has_value(), "the walk's poses");
    return poses.value_or(std::vector<Eigen::Isometry3d>{});
}

/// The large scene's sphere: 0.4 m in radius, a large object of 0.5 m to 2 m (GLOSSARY.md),
/// resting on the plane at the world's origin [?: a choice].
inline constexpr double large_sphere_radius_m = 0.4;

/// The large scene's plane under the sphere: 3 m x 3 m [?: a choice, so that the full model's
/// blocks stay within the cases' pool: it takes 54285 of the 65536 (M 2026-11-15:
/// tests/test_sparse_tsdf_fusion.cpp)].
inline constexpr synthetic_plane large_plane = {1.5, 1.5};

/// Gives the walk around the large scene's sphere: 12 poses, 1.5 m to 1.7 m from its centre along
/// the plane, 1.1 m to 1.3 m above it, rolled by up to 10 degrees, looking at its centre moved by
/// up to 5 cm along x and along y [?: a choice, a hand-held walk around a large object].
/// @return the walk, for walk_around().
inline synthetic_walk large_walk()
{
    return {.centre_m = Eigen::Vector3d(0.0, 0.0, large_sphere_radius_m),
            .pose_count = 12,
            .radius_m = 1.6,
            .radius_swing_m = 0.1,
            .height_m = 1.2,
            .height_swing_m = 0.1,
            .roll_swing_deg = 10.0,
            .target_jitter_m = 0.05};
}

/// The tabletop scene with a sphere: the tabletop sphere on the tabletop plane (stage_frames.hpp,
/// synthetic_source.hpp), seen along the walk around the tabletop box, 12 poses from 0.45 m.
/// @return the scene.
inline synthetic_scene tabletop_sphere_scene()
{
    return {.plane = tabletop_plane,
            .sphere = tabletop_sphere,
            .trajectory = poses_of(tabletop_walk())};
}

/// The large scene: the large sphere on the large plane along the large walk.
/// @return the scene.
inline synthetic_scene large_sphere_scene()
{
    const synthetic_walk walk = large_walk();
    return {.plane = large_plane,
            .sphere = synthetic_sphere{walk.centre_m, large_sphere_radius_m},
            .trajectory = poses_of(walk)};
}

/// What the test knows of a point of a scene: its nearest analytic surface, the plane or the
/// sphere, and its distance, normal and color there.
struct expected_point
{
    double distance_m;           ///< the point's distance from the nearest surface, in m
    Eigen::Vector3d normal;      ///< that surface's unit normal at its nearest point, outward
    std::array<double, 3> color; ///< the texture there, 0 to 255
    bool on_sphere;              ///< whether the nearest surface is the sphere
};

/// Finds what a point should be: the nearer of the plane's rectangle and the sphere, the normal
/// up from the plane (the cameras look down on it) or out of the sphere, and the texture at the
/// nearest point.
/// @param scene [in] the scene, with a plane and a sphere.
/// @param surfaces [in] its known surfaces, the plane first.
/// @param p [in] the point, in the world frame, in m.
/// @return what it should be.
inline expected_point expect_at(const synthetic_scene &scene,
                                const std::vector<synthetic_surface> &surfaces,
                                const Eigen::Vector3d &p)
{
    const synthetic_sphere &sphere = *scene.sphere;
    const Eigen::Vector3d out = p - sphere.centre_m;
    const double from_sphere_m = std::abs(out.norm() - sphere.radius_m);
    const double from_plane_m = distance_from(surfaces.front(), p);
    if (from_sphere_m < from_plane_m)
    {
        const Eigen::Vector3d n = out.normalized();
        return {from_sphere_m, n,
                texture_color(synthetic_texture::box, sphere.centre_m + sphere.radius_m * n), true};
    }
    return {from_plane_m, Eigen::Vector3d::UnitZ(),
            texture_color(synthetic_texture::plane, Eigen::Vector3d(p.x(), p.y(), 0.0)), false};
}
} // namespace gca_test
