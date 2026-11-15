// SPDX-License-Identifier: 0BSD
/// @file registration_cases.hpp
/// What the tests that register frames against a surface share (registration/registration.hpp,
/// docs/methods/registration.md): the normals' parameters and the object classes' depth ranges
/// their pyramids are built with, the object classes' starting values of the registration and the
/// parameters with the coarser levels skipped, the tabletop object's and the large object's scenes,
/// the poses they are seen from and their known motions, a pyramid's levels as a surface, and the
/// printing of a result with the check that a converged one carries the known motion. Header only;
/// each test program is one translation unit.

#pragma once

#include "check.hpp"
#include "geometry/frame_pyramid.hpp"
#include "geometry/organized_normals.hpp"
#include "geometry/organized_points.hpp"
#include "registration/registration.hpp"
#include "scan/object_class.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "transform_checks.hpp"

#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>

#include <Eigen/Geometry>

namespace gca_test
{
/// The normals' parameters at every level: the method's starting values, a 7x7 window, 25 points,
/// 75 degrees [S: docs/methods/normals.md, "Input"].
inline constexpr gca::organized_normals_parameters normals_parameters = {3, 25, 75.0f};

/// Gives an object class's depth range for the organized points, each limit half a depth unit
/// inside, so that no depth value lies on a limit.
/// @param kind [in] the object class.
/// @return its range (scan/object_class.hpp), in m, each limit moved inward by half of 1 mm.
inline gca::organized_points_parameters range_of(gca::object_class kind)
{
    const gca::object_class_parameters p = gca::parameters_of(kind);
    const double half_unit_m = 0.5 * camera_depth_unit_m;
    return {static_cast<float>(p.min_depth_m + half_unit_m),
            static_cast<float>(p.max_depth_m - half_unit_m)};
}

/// The tabletop object class's depth range, 0.30 m to 1.0 m [S: src/scan/object_class.cpp].
inline const gca::organized_points_parameters tabletop_range =
    range_of(gca::object_class::tabletop_object);

/// The large object class's depth range, 0.4 m to 3.0 m [S: src/scan/object_class.cpp].
inline const gca::organized_points_parameters large_range =
    range_of(gca::object_class::large_object);

/// The tabletop object class's starting values, the full method: d_max 5, 10 and 20 cm and 20, 10
/// and 10 iterations at levels 0, 1 and 2, δ_G 5 mm, λ 0.968 [?: initial;
/// docs/methods/registration.md, "Parameters"].
inline constexpr gca::registration_parameters tabletop_parameters = {
    {{{0.05f, 20}, {0.10f, 10}, {0.20f, 10}}}, 0.005f, 0.968f};

/// The large object class's starting values, the full method: d_max 10, 20 and 40 cm and 20, 10
/// and 10 iterations at levels 0, 1 and 2, δ_G 30 mm, λ 0.968 [?: initial;
/// docs/methods/registration.md, "Parameters"].
inline constexpr gca::registration_parameters large_parameters = {
    {{{0.10f, 20}, {0.20f, 10}, {0.40f, 10}}}, 0.030f, 0.968f};

/// Gives parameters with the coarser levels skipped: level 0 alone.
/// @param p [in] the parameters.
/// @return p with no iteration at levels 1 and 2.
inline gca::registration_parameters on_level_0(gca::registration_parameters p)
{
    p.levels[1].max_iterations = 0;
    p.levels[2].max_iterations = 0;
    return p;
}

/// Returns the name of how a registration ended, for the printed lines.
/// @param end [in] the end.
/// @return its enumerator's name.
inline const char *end_name(gca::registration_end end)
{
    switch (end)
    {
    case gca::registration_end::converged:
        return "converged";
    case gca::registration_end::iterations_used_up:
        return "iterations used up";
    case gca::registration_end::few_correspondences:
        return "few correspondences";
    case gca::registration_end::degenerate:
        return "degenerate";
    case gca::registration_end::refused:
        return "refused";
    case gca::registration_end::failed:
        return "failed";
    }
    return "unknown";
}

/// The tabletop object's known transform: a turn of 1.5 degrees and a shift of 10.8 mm [?: a
/// choice, a hand's motion in a frame time at 30 fps: 45 degrees and 0.32 m a second].
inline const Eigen::Isometry3d tabletop_delta =
    make_transform(Eigen::Vector3d(0.3, -0.8, 0.5), 1.5, Eigen::Vector3d(0.006, -0.004, 0.008));

/// The large object's known transform: a turn of 0.5 degrees, 24 mm at the box 2.7 m away, and a
/// shift of 14.1 mm [?: a choice].
inline const Eigen::Isometry3d large_delta =
    make_transform(Eigen::Vector3d(-0.4, 0.7, 0.3), 0.5, Eigen::Vector3d(0.008, 0.006, -0.010));

/// The large object's turn too large for level 0 alone: 1.5 degrees about large_delta's axis, 70
/// mm at the box 2.7 m away, and its shift. On level 0 alone the box's faces seen at a slant lie
/// beyond d_max from their partners at the start, and the iterations ended 60 mm off in a
/// degenerate system [M 2026-11-14, point-to-plane-registration's test]: the pyramid's case.
inline const Eigen::Isometry3d large_turn_delta =
    make_transform(Eigen::Vector3d(-0.4, 0.7, 0.3), 1.5, Eigen::Vector3d(0.008, 0.006, -0.010));

/// Gives a level of a pyramid as a level of a surface.
/// @param pyramid [in] the pyramid.
/// @param level [in] the level, below frame_pyramid::level_count.
/// @return its points, marks, normals, intensity and intrinsics.
inline gca::organized_surface level_of(const gca::frame_pyramid &pyramid, std::size_t level)
{
    return {pyramid.points(level).points(), pyramid.points(level).validity_mask(),
            pyramid.normals(level).normals(), pyramid.intensity(level),
            pyramid.level_intrinsics(level)};
}

/// Gives a pyramid's levels as a surface.
/// @param pyramid [in] the pyramid.
/// @return its levels 0, 1 and 2.
inline gca::surface_levels surface_of(const gca::frame_pyramid &pyramid)
{
    return {level_of(pyramid, 0), level_of(pyramid, 1), level_of(pyramid, 2)};
}

/// Prints a result.
/// @param what [in] the case.
/// @param result [in] the result.
inline void print_result(const std::string &what, const gca::registration_result &result)
{
    std::printf("%s: %s after %u iterations, fitness %.4f, inlier RMSE %.3g m, eigenvalues %.3g "
                "%.3g %.3g %.3g %.3g %.3g%s%s\n",
                what.c_str(), end_name(result.end), result.iterations, result.fitness,
                result.inlier_rmse_m, result.eigenvalues[0], result.eigenvalues[1],
                result.eigenvalues[2], result.eigenvalues[3], result.eigenvalues[4],
                result.eigenvalues[5], result.message.empty() ? "" : ", ", result.message.c_str());
}

/// Checks the rule every result keeps: when it says converged, its transform is the known one.
/// Prints how far the transform lies from it either way.
/// @param what [in] the case.
/// @param result [in] the result.
/// @param known [in] the known transform.
/// @param tolerance [in] how far a converged transform may lie from it.
inline void check_converged_is_known(const std::string &what,
                                     const gca::registration_result &result,
                                     const Eigen::Isometry3d &known,
                                     const transform_tolerance &tolerance)
{
    const transform_error error = error_of(result.transform, known);
    std::printf("%s: %.3g deg and %.3g mm from the known transform\n", what.c_str(),
                error.rotation_deg, 1e3 * error.translation_m);
    if (!result.converged())
        return;
    check_near(what, result.transform, known, tolerance);
}

/// The tabletop object: a box of 0.30 m x 0.20 m x 0.25 m turned by 20 degrees and a sphere of
/// 0.10 m in radius beside it, on a plane of 2 m x 2 m, the large object's scene at a quarter of
/// its size [?: a choice; a tabletop object of 10 cm to 50 cm, GLOSSARY.md]. The shared tabletop
/// box alone on its plane holds its weakest motion too weakly for the point-to-plane term
/// (check_degenerate()); its texture holds it.
/// @return the scene, without a trajectory.
inline synthetic_scene tabletop_object_scene()
{
    return {.plane = synthetic_plane{1.0, 1.0},
            .box = synthetic_box{0.0, 0.0, 20.0, 0.30, 0.20, 0.25},
            .sphere = synthetic_sphere{Eigen::Vector3d(0.1425, 0.2625, 0.10), 0.10},
            .trajectory = {}};
}

/// The large object: a box of 1.2 m x 0.8 m x 1.0 m turned by 20 degrees (the sparse TSDF
/// allocation test's) and a sphere of 0.4 m in radius beside it, on a floor of 8 m x 8 m [?: a
/// choice; a large object of 0.5 m to 2 m, GLOSSARY.md].
/// @return the scene, without a trajectory.
inline synthetic_scene large_object_scene()
{
    return {.plane = synthetic_plane{4.0, 4.0},
            .box = synthetic_box{0.0, 0.0, 20.0, 1.2, 0.8, 1.0},
            .sphere = synthetic_sphere{Eigen::Vector3d(0.57, 1.05, 0.40), 0.40},
            .trajectory = {}};
}

/// The pose the tabletop object is seen from: 0.5 m from the point it looks at, from above at 34
/// degrees [?: a choice].
/// @return the pose, camera to world; nothing when look_at() refuses it.
inline std::optional<Eigen::Isometry3d> tabletop_object_pose()
{
    return look_at(Eigen::Vector3d(0.40, -0.225, 0.325), Eigen::Vector3d(0.0, 0.0, 0.125), 0.0);
}

/// The pose the large object is seen from: about 2.7 m from the point it looks at [?: a choice].
/// @return the pose, camera to world; nothing when look_at() refuses it.
inline std::optional<Eigen::Isometry3d> large_object_pose()
{
    return look_at(Eigen::Vector3d(2.2, -1.2, 1.5), Eigen::Vector3d(0.0, 0.0, 0.5), 0.0);
}
} // namespace gca_test
