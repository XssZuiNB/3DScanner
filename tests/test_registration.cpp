// SPDX-License-Identifier: 0BSD
/// @file test_registration.cpp
/// The registration module's registration (registration/registration.hpp) on pairs of frames of the
/// tests' synthetic source whose transform is known; the method is docs/methods/registration.md.
/// Frame A is seen from pose P_A and frame B from P_B = P_A Δ, so that Δ carries B's camera frame
/// into A's. B's pyramid's level 0 is registered against A's level 0 as the surface, from the
/// identity, and the result is compared with Δ. Every result that says converged must carry Δ
/// within its case's tolerance.
/// - Recovered, at both stream sizes: a tabletop object, a box and a sphere on a plane, from about
///   0.5 m; a large object, a box and a sphere on a floor, from about 2.7 m; and the same from
///   about 2 m, the points 2 m away. Converged, within a tolerance of Δ, with the fitness, the RMSE
///   of the depth's rounding and a conditioning above τ. Frame A against itself: converged at once,
///   at the identity exactly.
/// - Degenerate, never converged: a single plane, tilted near and 2 m away, at both stream sizes,
///   with three eigenvalues below τ and three above; and the shared tabletop box on its plane,
///   whose weakest motion its faces hold too weakly for the point-to-plane term alone.
/// - Few correspondences, and none: a surface with depth only in a small window; a surface and a
///   frame without depth; a frame moved out of the surface's view. Not converged, the fitness below
///   its least value.
/// - Patches of frame B the surface does not hold, each held off by its rule: 20 mm deeper, within
///   d_max and beyond δ_G, by Huber's weight, within a tolerance plain least squares exceeds;
///   100 mm deeper by d_max and a patch of ridges by θ_n, each within the tabletop object's
///   tolerance, which the transform exceeds without the rule.
/// - The ends: 2 iterations where 4 are needed end as the iterations used up, never converged; a
///   start whose first system holds every motion and whose last does not ends degenerate.
/// - Refusals as values, with the initial transform: an empty registration, a level that is not
///   one, an empty pyramid, a level larger than the registration's size, a surface's empty images
///   or images of two sizes, intrinsics that cannot be used, an initial transform that is not
///   finite or not rigid, and each parameter outside its range, while the largest of each is
///   taken; CUDA's last error left by an earlier call ends it as failed. CUDA's last error is clear
///   after each.
/// - Every buffer is requested before the pool closes, and the registration asks the closed pool
///   for nothing.

#include "camera/frame.hpp"
#include "check.hpp"
#include "geometry/frame_pyramid.hpp"
#include "geometry/organized_normals.hpp"
#include "geometry/organized_points.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_image.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "gpu_test_helpers.hpp"
#include "registration/normal_equations.hpp"
#include "registration/registration.hpp"
#include "scan/object_class.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_within;

/// The normals' parameters at level 0: the method's starting values, a 7x7 window, 25 points, 75
/// degrees [S: docs/methods/normals.md, "Input"].
constexpr gca::organized_normals_parameters normals_parameters = {3, 25, 75.0f};

/// Gives an object class's depth range for the organized points, each limit half a depth unit
/// inside, so that no depth value lies on a limit.
/// @param kind [in] the object class.
/// @return its range (scan/object_class.hpp), in m, each limit moved inward by half of 1 mm.
gca::organized_points_parameters range_of(gca::object_class kind)
{
    const gca::object_class_parameters p = gca::parameters_of(kind);
    const double half_unit_m = 0.5 * gca_test::camera_depth_unit_m;
    return {static_cast<float>(p.min_depth_m + half_unit_m),
            static_cast<float>(p.max_depth_m - half_unit_m)};
}

/// The tabletop object class's depth range, 0.30 m to 1.0 m [S: src/scan/object_class.cpp].
const gca::organized_points_parameters tabletop_range =
    range_of(gca::object_class::tabletop_object);

/// The large object class's depth range, 0.4 m to 3.0 m [S: src/scan/object_class.cpp].
const gca::organized_points_parameters large_range = range_of(gca::object_class::large_object);

/// The tabletop object class's starting values: d_max 5 cm, δ_G 5 mm, 20 iterations [?: initial;
/// docs/methods/registration.md, "Parameters"].
constexpr gca::registration_parameters tabletop_parameters = {0.05f, 0.005f, 20};

/// The large object class's starting values: d_max 10 cm, δ_G 30 mm, 20 iterations [?: initial;
/// docs/methods/registration.md, "Parameters"].
constexpr gca::registration_parameters large_parameters = {0.10f, 0.030f, 20};

/// How far a recovered tabletop object's transform may lie from Δ: 0.005 degrees and 0.06 mm, about
/// twice the largest errors at both stream sizes, 0.00254 degrees and 0.0311 mm [M 2026-11-14, this
/// test].
constexpr double tabletop_rotation_tolerance_deg = 0.005;
constexpr double tabletop_translation_tolerance_m = 0.06e-3;

/// How far a recovered large object's transform may lie from Δ, from 2.7 m and from 2 m: 0.007
/// degrees and 0.3 mm, about twice the largest errors, 0.00335 degrees and 0.142 mm, both at
/// 640x400 from 2.7 m [M 2026-11-14, this test].
constexpr double large_rotation_tolerance_deg = 0.007;
constexpr double large_translation_tolerance_m = 0.3e-3;

/// The largest inlier RMSE of a recovered transform: 0.5 mm, above the depth's rounding, 1/sqrt(12)
/// of a unit of 1 mm, 0.289 mm [S: the uniform distribution]; the largest is 0.33 mm
/// [M 2026-11-14, this test].
constexpr double recovered_rmse_tolerance_m = 0.5e-3;

/// The least fitness of a recovered transform: 0.95; the least is 0.967 [M 2026-11-14, this test].
constexpr double recovered_least_fitness = 0.95;

/// How far the transform may lie from Δ with a patch of 80 x 80 pixels 20 mm deeper in frame B
/// (check_deeper_patch()): 0.23 degrees and 2 mm, about twice what Huber's weight leaves, 0.115
/// degrees and 1.04 mm, below what plain least squares leaves, 0.339 degrees and 3.04 mm
/// [M 2026-11-14, this test, and with the weight 1].
constexpr double deeper_patch_rotation_tolerance_deg = 0.23;
constexpr double deeper_patch_translation_tolerance_m = 2.0e-3;

/// The depth the patch within d_max and beyond δ_G adds: 20 depth units, 20 mm [?: a choice].
constexpr std::uint16_t huber_patch_depth = 20;

/// The depth the patch beyond d_max adds: 100 depth units, 100 mm, twice the tabletop object's
/// d_max [?: a choice]. With d_max the transform lies 0.0022 degrees and 0.0128 mm from Δ, without
/// it 0.092 degrees and 0.79 mm [M 2026-11-14, this test, and with the distance's test removed].
constexpr std::uint16_t far_patch_depth = 100;

/// The ridges of the patch whose normals face away: a sawtooth along the picture's rows, rising
/// ridge_height depth units over ridge_period_px pixels and dropping back, 20 mm over 16 pixels
/// [?: a choice]. With θ_n the transform lies 0.0015 degrees and 0.0054 mm from Δ, without it 0.084
/// degrees and 0.758 mm [M 2026-11-14, this test, and with the normals' test removed].
constexpr std::uint32_t ridge_height = 20;
constexpr std::uint32_t ridge_period_px = 16;

/// The start of the case where the last system decides: 25 mm up along the table's normal
/// [?: a choice]; its first system's smallest eigenvalue is 0.0385, its last one's 0.00452
/// [M 2026-11-14, this test].
constexpr double start_up_m = 0.025;

/// Returns the name of how a registration ended, for the printed lines.
/// @param end [in] the end.
/// @return its enumerator's name.
const char *end_name(gca::registration_end end)
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

/// Builds a rigid transform: a turn about an axis through the origin, then a shift.
/// @param axis [in] the axis, not zero; normalized here.
/// @param angle_deg [in] the angle, in degrees.
/// @param shift_m [in] the shift, in m.
/// @return the transform p -> R p + shift.
Eigen::Isometry3d make_transform(const Eigen::Vector3d &axis, double angle_deg,
                                 const Eigen::Vector3d &shift_m)
{
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.linear() =
        Eigen::AngleAxisd(angle_deg * std::numbers::pi / 180.0, axis.normalized()).matrix();
    transform.translation() = shift_m;
    return transform;
}

/// The tabletop object's known transform: a turn of 1.5 degrees and a shift of 10.8 mm [?: a
/// choice, a hand's motion in a frame time at 30 fps: 45 degrees and 0.32 m a second].
const Eigen::Isometry3d tabletop_delta =
    make_transform(Eigen::Vector3d(0.3, -0.8, 0.5), 1.5, Eigen::Vector3d(0.006, -0.004, 0.008));

/// The large object's known transform: a turn of 0.5 degrees, 24 mm at the box 2.7 m away, and a
/// shift of 14.1 mm [?: a choice]. A turn of 1.5 degrees, 70 mm at the box, left the box's faces
/// seen at a slant beyond d_max from their partners at the start, and the iterations ended 60 mm
/// off in a degenerate system [M 2026-11-14, this test]: a motion for the pyramid's coarser levels.
const Eigen::Isometry3d large_delta =
    make_transform(Eigen::Vector3d(-0.4, 0.7, 0.3), 0.5, Eigen::Vector3d(0.008, 0.006, -0.010));

/// How far a transform lies from a known one.
struct transform_error
{
    double rotation_deg;  ///< the angle of the turn between them, in degrees
    double translation_m; ///< the length of the shift between them at the camera's centre, in m
};

/// Measures how far a transform lies from a known one: the turn and the shift of known^-1 found.
/// @param found [in] the transform found.
/// @param known [in] the known transform.
/// @return the angle of the turn, in degrees, and the length of the shift, in m.
transform_error error_of(const Eigen::Isometry3d &found, const Eigen::Isometry3d &known)
{
    const Eigen::Isometry3d e = known.inverse() * found;
    const Eigen::AngleAxisd turn(e.linear());
    return {std::abs(turn.angle()) * 180.0 / std::numbers::pi, e.translation().norm()};
}

/// Tolerances of a recovered transform.
struct transform_tolerance
{
    double rotation_deg;  ///< the largest turn from the known transform, in degrees
    double translation_m; ///< the largest shift from it at the camera's centre, in m
};

/// The tolerances of each object class; a case whose transform is not recovered checks a result
/// that says converged against its object class's all the same.
constexpr transform_tolerance tabletop_tolerance = {tabletop_rotation_tolerance_deg,
                                                    tabletop_translation_tolerance_m};
constexpr transform_tolerance large_tolerance = {large_rotation_tolerance_deg,
                                                 large_translation_tolerance_m};
constexpr transform_tolerance deeper_patch_tolerance = {deeper_patch_rotation_tolerance_deg,
                                                        deeper_patch_translation_tolerance_m};

/// What the tests work on: the GPU frames and pyramids at both stream sizes and the registrations,
/// every buffer requested before the pool closes, and the stream the work runs on.
struct stage
{
    gca::gpu_pool pool;               ///< closed once every buffer is granted
    cudaStream_t stream = nullptr;    ///< the stream the work runs on
    gca::gpu_frame small_frame;       ///< a GPU frame of 640x400
    gca::gpu_frame large_frame;       ///< a GPU frame of 1280x800
    gca::frame_pyramid small_surface; ///< frame A's pyramid at 640x400
    gca::frame_pyramid small_moved;   ///< frame B's pyramid at 640x400
    gca::frame_pyramid large_surface; ///< frame A's pyramid at 1280x800
    gca::frame_pyramid large_moved;   ///< frame B's pyramid at 1280x800
    gca::registration registration;   ///< for frames up to 1280x800
    gca::registration small_only;     ///< for frames up to 640x400, for a refusal
};

/// The GPU frame and the two pyramids of a stream size.
struct size_buffers
{
    gca::gpu_frame &frame;       ///< the GPU frame
    gca::frame_pyramid &surface; ///< frame A's pyramid
    gca::frame_pyramid &moved;   ///< frame B's pyramid
};

/// Picks the buffers of a frame's size.
/// @param s [in,out] the stage.
/// @param size [in] the frame's size, 640x400 or 1280x800.
/// @return the buffers of that size.
size_buffers buffers_of(stage &s, const gca::frame_size &size)
{
    if (size == gca_test::large_description.size)
        return {s.large_frame, s.large_surface, s.large_moved};
    return {s.small_frame, s.small_surface, s.small_moved};
}

/// Uploads a frame and builds its pyramid.
/// @param s [in,out] the stage.
/// @param host [in] the frame.
/// @param range [in] the depth range.
/// @param pyramid [out] the pyramid built.
/// @param frame [in,out] the GPU frame of the frame's size.
/// @return whether the upload and the build were queued; failed checks when not.
bool build_pyramid(stage &s, const gca::frame &host, const gca::organized_points_parameters &range,
                   gca::frame_pyramid &pyramid, gca::gpu_frame &frame)
{
    const gca::step_result up = frame.upload(host, s.stream);
    check(up.done, ("the upload, " + up.message).c_str());
    const gca::step_result built = pyramid.build(frame, {range, normals_parameters}, s.stream);
    check(built.done, ("the pyramid's build, " + built.message).c_str());
    return up.done && built.done;
}

/// Gives a pyramid's level 0 as a surface.
/// @param pyramid [in] the pyramid.
/// @return its level 0's points, marks, normals and intrinsics.
gca::organized_surface surface_of(const gca::frame_pyramid &pyramid)
{
    return {pyramid.points(0).points(), pyramid.points(0).validity_mask(),
            pyramid.normals(0).normals(), pyramid.level_intrinsics(0)};
}

/// Prints a result.
/// @param what [in] the case.
/// @param result [in] the result.
void print_result(const std::string &what, const gca::registration_result &result)
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
void check_converged_is_known(const std::string &what, const gca::registration_result &result,
                              const Eigen::Isometry3d &known, const transform_tolerance &tolerance)
{
    const transform_error error = error_of(result.transform, known);
    std::printf("%s: %.3g deg and %.3g mm from the known transform\n", what.c_str(),
                error.rotation_deg, 1e3 * error.translation_m);
    if (!result.converged())
        return;
    check_within((what + ": rotation error (deg)").c_str(), error.rotation_deg,
                 tolerance.rotation_deg);
    check_within((what + ": translation error (m)").c_str(), error.translation_m,
                 tolerance.translation_m);
}

/// Renders frames A and B of a scene, A from a pose and B from pose Δ.
/// @param scene [in] the scene; its trajectory is replaced.
/// @param pose_a [in] frame A's pose, camera to world.
/// @param delta [in] the known transform from B's camera frame into A's.
/// @param description [in] the frames' size, intrinsics and depth unit.
/// @return the frames A and B.
gca_test::rendered render_pair(gca_test::synthetic_scene scene, const Eigen::Isometry3d &pose_a,
                               const Eigen::Isometry3d &delta,
                               const gca::frame_description &description)
{
    scene.trajectory = {pose_a, pose_a * delta};
    return gca_test::render(scene, description);
}

/// Registers frame B against frame A.
/// @param s [in,out] the stage.
/// @param a [in] frame A, the surface's.
/// @param b [in] frame B, registered; of A's size.
/// @param range [in] the depth range.
/// @param parameters [in] the registration's parameters.
/// @param initial [in] the initial transform, B to A.
/// @return the result; refused, with a failed check, when the pyramids could not be built.
gca::registration_result register_frames(stage &s, const gca::frame &a, const gca::frame &b,
                                         const gca::organized_points_parameters &range,
                                         const gca::registration_parameters &parameters,
                                         const Eigen::Isometry3d &initial)
{
    const size_buffers buffers = buffers_of(s, a.description().size);
    if (!build_pyramid(s, a, range, buffers.surface, buffers.frame) ||
        !build_pyramid(s, b, range, buffers.moved, buffers.frame))
        return {};
    return s.registration.align(buffers.moved, 0, surface_of(buffers.surface), initial, parameters,
                                s.stream);
}

/// Registers frame B of a pair against frame A from the identity.
/// @param s [in,out] the stage.
/// @param pair [in] frames A and B.
/// @param range [in] the depth range.
/// @param parameters [in] the registration's parameters.
/// @return the result; refused, with a failed check, when the pair is not two frames.
gca::registration_result register_pair(stage &s, const gca_test::rendered &pair,
                                       const gca::organized_points_parameters &range,
                                       const gca::registration_parameters &parameters)
{
    check(pair.frames.size() == 2, "the pair's two frames");
    if (pair.frames.size() != 2)
        return {};
    return register_frames(s, pair.frames[0], pair.frames[1], range, parameters,
                           Eigen::Isometry3d::Identity());
}

/// The tabletop object: a box of 0.30 m x 0.20 m x 0.25 m turned by 20 degrees and a sphere of 0.10
/// m in radius beside it, on a plane of 2 m x 2 m, the large object's scene at a quarter of its
/// size
/// [?: a choice; a tabletop object of 10 cm to 50 cm, GLOSSARY.md]. The shared tabletop box alone
/// on its plane holds its weakest motion too weakly for the point-to-plane term
/// (check_degenerate()).
/// @return the scene, without a trajectory.
gca_test::synthetic_scene tabletop_object_scene()
{
    return {.plane = {1.0, 1.0},
            .box = gca_test::synthetic_box{0.0, 0.0, 20.0, 0.30, 0.20, 0.25},
            .sphere = gca_test::synthetic_sphere{Eigen::Vector3d(0.1425, 0.2625, 0.10), 0.10},
            .trajectory = {}};
}

/// The large object: a box of 1.2 m x 0.8 m x 1.0 m turned by 20 degrees (the sparse TSDF
/// allocation test's) and a sphere of 0.4 m in radius beside it, on a floor of 8 m x 8 m [?: a
/// choice; a large object of 0.5 m to 2 m, GLOSSARY.md].
/// @return the scene, without a trajectory.
gca_test::synthetic_scene large_object_scene()
{
    return {.plane = {4.0, 4.0},
            .box = gca_test::synthetic_box{0.0, 0.0, 20.0, 1.2, 0.8, 1.0},
            .sphere = gca_test::synthetic_sphere{Eigen::Vector3d(0.57, 1.05, 0.40), 0.40},
            .trajectory = {}};
}

/// The pose the tabletop object is seen from: 0.5 m from the point it looks at, from above at 34
/// degrees [?: a choice].
/// @return the pose, camera to world; nothing when look_at() refuses it.
std::optional<Eigen::Isometry3d> tabletop_object_pose()
{
    return gca_test::look_at(Eigen::Vector3d(0.40, -0.225, 0.325), Eigen::Vector3d(0.0, 0.0, 0.125),
                             0.0);
}

/// Gives a stream size's name for the printed lines.
/// @param description [in] the frames' description, 640x400 or 1280x800.
/// @return " at 640x400" or " at 1280x800".
std::string at_size(const gca::frame_description &description)
{
    return description.size == gca_test::small_description.size ? " at 640x400" : " at 1280x800";
}

/// A case of a pair whose known transform the registration recovers.
struct recovery_case
{
    std::string name;                        ///< the case, for the printed lines
    gca_test::synthetic_scene scene;         ///< the scene
    Eigen::Isometry3d pose_a;                ///< frame A's pose, camera to world
    Eigen::Isometry3d delta;                 ///< the known transform, B to A
    gca::frame_description description;      ///< the frames' size and intrinsics
    gca::organized_points_parameters range;  ///< the depth range
    gca::registration_parameters parameters; ///< the registration's parameters
    transform_tolerance tolerance;           ///< how far the recovered transform may lie
};

/// Checks that a case's known transform is recovered: converged, within its tolerance, with a
/// fitness, RMSE and conditioning of a good registration; and that frame A against itself converges
/// at once, at the identity exactly: the same points give residuals of 0 and a step of 0.
/// @param s [in,out] the stage.
/// @param c [in] the case.
void check_recovery(stage &s, const recovery_case &c)
{
    const gca_test::rendered pair = render_pair(c.scene, c.pose_a, c.delta, c.description);
    const gca::registration_result result = register_pair(s, pair, c.range, c.parameters);
    print_result(c.name, result);
    check(result.converged(), (c.name + ": converged").c_str());
    check_converged_is_known(c.name, result, c.delta, c.tolerance);
    check(result.fitness >= recovered_least_fitness, (c.name + ": the fitness").c_str());
    check_within((c.name + ": inlier RMSE (m)").c_str(), result.inlier_rmse_m,
                 recovered_rmse_tolerance_m);
    check(result.eigenvalues[0] >= gca::registration_degeneracy_threshold,
          (c.name + ": the smallest eigenvalue at least the threshold").c_str());
    if (pair.frames.size() != 2)
        return;
    const gca::registration_result itself = register_frames(
        s, pair.frames[0], pair.frames[0], c.range, c.parameters, Eigen::Isometry3d::Identity());
    const std::string self = c.name + ", frame A against itself";
    print_result(self, itself);
    check(itself.converged() && itself.iterations == 1,
          (self + ": converged after one iteration").c_str());
    check(itself.transform.matrix() == Eigen::Matrix4d::Identity(),
          (self + ": the identity exactly").c_str());
    check(itself.fitness == 1.0 && itself.inlier_rmse_m == 0.0,
          (self + ": fitness 1 and RMSE 0").c_str());
}

/// Runs the cases whose transform is recovered: the tabletop object from about 0.5 m, the large
/// object from about 2.7 m and from about 2 m, each at both stream sizes.
/// @param s [in,out] the stage.
void check_recoveries(stage &s)
{
    const std::optional<Eigen::Isometry3d> tabletop_pose = tabletop_object_pose();
    const std::optional<Eigen::Isometry3d> large_pose =
        gca_test::look_at(Eigen::Vector3d(2.2, -1.2, 1.5), Eigen::Vector3d(0.0, 0.0, 0.5), 0.0);
    const std::optional<Eigen::Isometry3d> near_pose =
        gca_test::look_at(Eigen::Vector3d(1.6, -0.9, 1.3), Eigen::Vector3d(0.0, 0.0, 0.5), 0.0);
    check(tabletop_pose && large_pose && near_pose, "the recoveries' poses");
    if (!tabletop_pose || !large_pose || !near_pose)
        return;
    for (const gca::frame_description &description :
         {gca_test::small_description, gca_test::large_description})
    {
        const std::string at = at_size(description);
        const recovery_case cases[] = {
            {"tabletop object from 0.5 m" + at, tabletop_object_scene(), *tabletop_pose,
             tabletop_delta, description, tabletop_range, tabletop_parameters, tabletop_tolerance},
            {"large object from 2.7 m" + at, large_object_scene(), *large_pose, large_delta,
             description, large_range, large_parameters, large_tolerance},
            {"large object from 2 m" + at, large_object_scene(), *near_pose, large_delta,
             description, large_range, large_parameters, large_tolerance},
        };
        for (const recovery_case &c : cases)
            check_recovery(s, c);
    }
}

/// A case of a pair the registration finds degenerate.
struct degenerate_case
{
    std::string name;                        ///< the case, for the printed lines
    gca_test::synthetic_scene scene;         ///< the scene
    Eigen::Isometry3d pose_a;                ///< frame A's pose, camera to world
    gca::organized_points_parameters range;  ///< the depth range
    gca::registration_parameters parameters; ///< the registration's parameters
    transform_tolerance tolerance;           ///< the tolerance of its object class
    bool plane;                              ///< whether the scene is a single plane
};

/// Runs the degenerate cases: a single plane, tilted near and 2 m away, and the shared tabletop box
/// on its plane, each at both stream sizes, moved by the tabletop object's Δ, which shifts along
/// the plane: degenerate, not converged; a plane with three eigenvalues below τ and three above.
/// @param s [in,out] the stage.
void check_degenerate(stage &s)
{
    const std::optional<Eigen::Isometry3d> tilted_pose =
        gca_test::look_at({0.0, -0.55, 0.35}, {0.0, 0.0, 0.0}, 10.0);
    const double elevation_rad = 70.0 * std::numbers::pi / 180.0;
    const std::optional<Eigen::Isometry3d> far_pose =
        gca_test::look_at({0.0, -2.0 * std::cos(elevation_rad), 2.0 * std::sin(elevation_rad)},
                          {0.0, 0.0, 0.0}, 10.0);
    const std::optional<std::vector<Eigen::Isometry3d>> walk =
        gca_test::walk_around(gca_test::tabletop_walk());
    check(tilted_pose && far_pose && walk, "the degenerate cases' poses");
    if (!tilted_pose || !far_pose || !walk)
        return;
    const gca_test::synthetic_scene plane = {.plane = {3.0, 3.0}, .trajectory = {}};
    const gca_test::synthetic_scene box = {
        .plane = gca_test::tabletop_plane, .box = gca_test::tabletop_box, .trajectory = {}};
    for (const gca::frame_description &description :
         {gca_test::small_description, gca_test::large_description})
    {
        const std::string at = at_size(description);
        const degenerate_case cases[] = {
            {"tilted plane" + at, plane, *tilted_pose, tabletop_range, tabletop_parameters,
             tabletop_tolerance, true},
            {"plane 2 m away" + at, plane, *far_pose, large_range, large_parameters,
             large_tolerance, true},
            {"tabletop box on its plane" + at, box, (*walk)[0], tabletop_range, tabletop_parameters,
             tabletop_tolerance, false},
        };
        for (const degenerate_case &c : cases)
        {
            const gca::registration_result result =
                register_pair(s, render_pair(c.scene, c.pose_a, tabletop_delta, description),
                              c.range, c.parameters);
            print_result(c.name, result);
            check(result.end == gca::registration_end::degenerate,
                  (c.name + ": degenerate").c_str());
            check_converged_is_known(c.name, result, tabletop_delta, c.tolerance);
            const auto below =
                std::count_if(result.eigenvalues.begin(), result.eigenvalues.end(),
                              [](double e) { return e < gca::registration_degeneracy_threshold; });
            if (c.plane)
                check(below == 3, (c.name + ": three eigenvalues below the threshold").c_str());
            else
                check(below >= 1, (c.name + ": an eigenvalue below the threshold").c_str());
        }
    }
}

/// A square window of a picture.
struct pixel_window
{
    std::uint32_t first_u; ///< its first column, in px
    std::uint32_t first_v; ///< its first row, in px
    std::uint32_t side;    ///< its side, in px

    /// Says whether a pixel lies inside the window.
    /// @param u [in] the column, in px.
    /// @param v [in] the row, in px.
    /// @return whether first_u <= u < first_u + side and first_v <= v < first_v + side.
    bool holds(std::uint32_t u, std::uint32_t v) const
    {
        return u >= first_u && u < first_u + side && v >= first_v && v < first_v + side;
    }
};

/// The window of a surface with few correspondences: 16 x 16 pixels at the centre of 640x400
/// [?: a choice]; it leaves a fitness of 0.0016 [M 2026-11-14, this test].
constexpr pixel_window few_window = {312, 192, 16};

/// The window of a patch the surface does not hold: 80 x 80 pixels on the plane near the lower left
/// corner of the tabletop object's frame at 640x400 [?: a choice].
constexpr pixel_window patch_window = {40, 310, 80};

/// Builds a frame with a rendered frame's color and its depth changed: a frame with a changed
/// depth.
/// @param rendered [in] the rendered frame.
/// @param depth_of [in] the new depth value of pixel (u, v) from its rendered one d, in depth
/// units,
///        called as depth_of(u, v, d).
/// @return the frame, index and timestamp 0.
template <typename Depth> gca::frame with_depth(const gca::frame &rendered, Depth depth_of)
{
    const gca::frame_description &d = rendered.description();
    return gca_test::built_frame(
        d,
        [&](std::uint32_t u, std::uint32_t v) -> std::uint16_t {
            return depth_of(u, v, rendered.depth()[static_cast<std::size_t>(v) * d.size.width + u]);
        },
        [&](std::uint32_t u, std::uint32_t v) {
            const std::size_t pixel = 3 * (static_cast<std::size_t>(v) * d.size.width + u);
            return std::array<std::uint8_t, 3>{rendered.color()[pixel], rendered.color()[pixel + 1],
                                               rendered.color()[pixel + 2]};
        });
}

/// Builds a frame with a rendered frame's color and its depth only inside a window.
/// @param rendered [in] the rendered frame.
/// @param window [in] the window; of side 0 for no depth at all.
/// @return the frame, its depth 0 outside the window.
gca::frame windowed(const gca::frame &rendered, const pixel_window &window)
{
    return with_depth(rendered, [&](std::uint32_t u, std::uint32_t v, std::uint16_t z) {
        return window.holds(u, v) ? z : std::uint16_t{0};
    });
}

/// Builds a frame with a rendered frame's color and its depth raised inside the patch window, where
/// it has depth.
/// @param rendered [in] the rendered frame.
/// @param raise_of [in] the depth added at pixel (u, v) of the window, in depth units, called as
///        raise_of(u, v).
/// @return the frame.
template <typename Raise> gca::frame patched(const gca::frame &rendered, Raise raise_of)
{
    return with_depth(rendered, [&](std::uint32_t u, std::uint32_t v, std::uint16_t z) {
        return patch_window.holds(u, v) && z != 0 ? static_cast<std::uint16_t>(z + raise_of(u, v))
                                                  : z;
    });
}

/// Checks a case with few correspondences or none: not converged, ended for too few after one
/// iteration, its fitness below the least.
/// @param what [in] the case.
/// @param result [in] its result.
/// @param none [in] whether it has no correspondence at all: then its fitness is 0.
void check_few(const std::string &what, const gca::registration_result &result, bool none)
{
    print_result(what, result);
    check(result.end == gca::registration_end::few_correspondences,
          (what + ": ended for too few correspondences").c_str());
    check(result.iterations == 1, (what + ": after one iteration").c_str());
    if (none)
        check(result.fitness == 0.0, (what + ": fitness 0").c_str());
    else
        check(result.fitness > 0.0 && result.fitness < gca::registration_min_fitness,
              (what + ": a fitness above 0 below the least").c_str());
}

/// Runs the cases with few correspondences and none, on the tabletop object's pair at 640x400: a
/// surface with depth only in a window of 16 x 16 pixels at the picture's centre; a surface without
/// depth; a frame without depth; and the frame moved 5 m to the side of the surface's view.
/// @param s [in,out] the stage.
void check_few_correspondences(stage &s)
{
    const std::optional<Eigen::Isometry3d> pose = tabletop_object_pose();
    check(pose.has_value(), "the few correspondences' pose");
    if (!pose)
        return;
    const gca_test::rendered pair =
        render_pair(tabletop_object_scene(), *pose, tabletop_delta, gca_test::small_description);
    if (pair.frames.size() != 2)
        return;
    const gca::frame &a = pair.frames[0];
    const gca::frame &b = pair.frames[1];
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    check_few("a surface with depth in 16 x 16 pixels",
              register_frames(s, windowed(a, few_window), b, tabletop_range, tabletop_parameters,
                              identity),
              false);
    check_few("a surface without depth",
              register_frames(s, windowed(a, {0, 0, 0}), b, tabletop_range, tabletop_parameters,
                              identity),
              true);
    check_few("a frame without depth",
              register_frames(s, a, windowed(b, {0, 0, 0}), tabletop_range, tabletop_parameters,
                              identity),
              true);
    check_few("the frame 5 m beside the surface's view",
              register_frames(
                  s, a, b, tabletop_range, tabletop_parameters,
                  make_transform(Eigen::Vector3d::UnitZ(), 0.0, Eigen::Vector3d(5.0, 0.0, 0.0))),
              true);
}

/// Runs the cases of a patch of frame B the surface does not hold, on the tabletop object's pair at
/// 640x400, each a rule of the association or of the weight at work: the patch 20 mm deeper, within
/// d_max and beyond δ_G, whose pull Huber's weight caps; the patch 100 mm deeper, beyond d_max,
/// whose pairs the largest distance cuts; and a patch of ridges within d_max of the plane, whose
/// slopes' pairs the largest angle between normals cuts (ridge_height).
/// @param s [in,out] the stage.
void check_patches(stage &s)
{
    const std::optional<Eigen::Isometry3d> pose = tabletop_object_pose();
    check(pose.has_value(), "the patches' pose");
    if (!pose)
        return;
    const gca_test::rendered pair =
        render_pair(tabletop_object_scene(), *pose, tabletop_delta, gca_test::small_description);
    if (pair.frames.size() != 2)
        return;
    const gca::frame &a = pair.frames[0];
    const gca::frame &b = pair.frames[1];
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const struct
    {
        std::string name;
        gca::frame frame;
        transform_tolerance tolerance;
    } cases[] = {
        {"a patch 20 mm deeper",
         patched(b, [](std::uint32_t, std::uint32_t) { return huber_patch_depth; }),
         deeper_patch_tolerance},
        {"a patch 100 mm deeper",
         patched(b, [](std::uint32_t, std::uint32_t) { return far_patch_depth; }),
         tabletop_tolerance},
        {"a patch of ridges",
         patched(b,
                 [](std::uint32_t u, std::uint32_t) {
                     return static_cast<std::uint16_t>(ridge_height * (u % ridge_period_px) /
                                                       ridge_period_px);
                 }),
         tabletop_tolerance},
    };
    for (const auto &c : cases)
    {
        const gca::registration_result result =
            register_frames(s, a, c.frame, tabletop_range, tabletop_parameters, identity);
        print_result(c.name, result);
        check(result.converged(), (c.name + ": converged").c_str());
        check_converged_is_known(c.name, result, tabletop_delta, c.tolerance);
    }
}

/// Runs the cases of the two ends a registration that steps can come to besides convergence. The
/// tabletop object's pair at 640x400 with 2 iterations, where it needs 4: the iterations used up,
/// never converged. And the tabletop object's box without its sphere from 0.67 m, the frame against
/// itself from a start 25 mm up along the table's normal: its first system holds every motion,
/// its last at the identity does not; with one iteration the iterations are used up and the
/// smallest eigenvalue lies above τ, with 20 the registration ends degenerate, the last system
/// deciding.
/// @param s [in,out] the stage.
void check_ends(stage &s)
{
    const std::optional<Eigen::Isometry3d> pose = tabletop_object_pose();
    const std::optional<Eigen::Isometry3d> far_pose = gca_test::look_at(
        Eigen::Vector3d(0.55, -0.30, 0.375), Eigen::Vector3d(0.0, 0.0, 0.125), 0.0);
    check(pose && far_pose, "the ends' poses");
    if (!pose || !far_pose)
        return;
    const gca_test::rendered pair =
        render_pair(tabletop_object_scene(), *pose, tabletop_delta, gca_test::small_description);
    gca_test::synthetic_scene box = tabletop_object_scene();
    box.sphere = std::nullopt;
    box.trajectory = {*far_pose};
    const gca_test::rendered alone = gca_test::render(box, gca_test::small_description);
    if (pair.frames.size() != 2 || alone.frames.size() != 1)
        return;

    gca::registration_parameters two = tabletop_parameters;
    two.max_iterations = 2;
    const gca::registration_result short_of = register_frames(
        s, pair.frames[0], pair.frames[1], tabletop_range, two, Eigen::Isometry3d::Identity());
    print_result("2 iterations of 4", short_of);
    check(short_of.end == gca::registration_end::iterations_used_up && short_of.iterations == 2,
          "2 iterations of 4: the iterations used up, not converged");

    Eigen::Isometry3d up = Eigen::Isometry3d::Identity();
    up.translation() = far_pose->linear().transpose() * Eigen::Vector3d(0.0, 0.0, start_up_m);
    gca::registration_parameters one = tabletop_parameters;
    one.max_iterations = 1;
    const gca::registration_result first =
        register_frames(s, alone.frames[0], alone.frames[0], tabletop_range, one, up);
    print_result("the box alone, its first system", first);
    check(first.end == gca::registration_end::iterations_used_up &&
              first.eigenvalues[0] >= gca::registration_degeneracy_threshold,
          "the box alone: its first system holds every motion, the iterations used up");
    const gca::registration_result last = register_frames(s, alone.frames[0], alone.frames[0],
                                                          tabletop_range, tabletop_parameters, up);
    print_result("the box alone, its last system", last);
    check(last.end == gca::registration_end::degenerate &&
              last.eigenvalues[0] < gca::registration_degeneracy_threshold,
          "the box alone: its last system decides, degenerate");
    check_converged_is_known("the box alone", last, Eigen::Isometry3d::Identity(),
                             tabletop_tolerance);
}

/// Checks that an alignment was refused with a message that holds a text, and left the initial
/// transform.
/// @param what [in] the case.
/// @param result [in] its result.
/// @param initial [in] the initial transform it was given.
/// @param expected [in] a text the message holds.
void check_align_refused(const std::string &what, const gca::registration_result &result,
                         const Eigen::Isometry3d &initial, const std::string &expected)
{
    gca_test::check_refusal(what, result.end == gca::registration_end::refused, result.message,
                            result.message.find(expected) != std::string::npos);
    check(result.iterations == 0 && result.transform.matrix() == initial.matrix(),
          (what + ": no iteration, the initial transform").c_str());
    gca_test::check_no_cuda_error(what);
}

/// Runs the refusals, on the tabletop object's frame at both sizes, then the largest parameters,
/// which are taken, and CUDA's last error left by an earlier call.
/// @param s [in,out] the stage.
void check_refusals(stage &s)
{
    const std::optional<Eigen::Isometry3d> pose = tabletop_object_pose();
    check(pose.has_value(), "the refusals' pose");
    if (!pose)
        return;
    gca_test::synthetic_scene scene = tabletop_object_scene();
    scene.trajectory = {*pose};
    const gca_test::rendered small = gca_test::render(scene, gca_test::small_description);
    const gca_test::rendered large = gca_test::render(scene, gca_test::large_description);
    if (small.frames.size() != 1 || large.frames.size() != 1 ||
        !build_pyramid(s, small.frames[0], tabletop_range, s.small_surface, s.small_frame) ||
        !build_pyramid(s, large.frames[0], tabletop_range, s.large_surface, s.large_frame))
        return;
    const gca::frame_pyramid &frame = s.small_surface;
    const gca::organized_surface surface = surface_of(s.small_surface);
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const gca::registration_parameters usable = tabletop_parameters;
    gca::registration &r = s.registration;
    cudaStream_t stream = s.stream;

    check_align_refused("an empty registration",
                        gca::registration{}.align(frame, 0, surface, identity, usable, stream),
                        identity, "holds no buffers");
    check_align_refused("level 3", r.align(frame, 3, surface, identity, usable, stream), identity,
                        "level 3 is not one of the frame pyramid's 3 levels");
    check_align_refused("an empty pyramid",
                        r.align(gca::frame_pyramid{}, 0, surface, identity, usable, stream),
                        identity, "holds no levels");
    check_align_refused(
        "a level of 1280x800 for a registration of 640x400",
        s.small_only.align(s.large_surface, 0, surface, identity, usable, stream), identity,
        "the frame's level of 1280x800 px is larger than the registration's 640x400 px");
    const gca::gpu_image<float3> no_points;
    const gca::gpu_image<std::uint8_t> no_marks;
    check_align_refused("a surface of empty images",
                        r.align(frame, 0, {no_points, no_marks, no_points, surface.intrinsics},
                                identity, usable, stream),
                        identity, "the surface's images are empty");
    check_align_refused("a surface of two sizes",
                        r.align(frame, 0,
                                {surface.points, s.small_surface.points(1).validity_mask(),
                                 surface.normals, surface.intrinsics},
                                identity, usable, stream),
                        identity, "not of one size: 640x400 px, 320x200 px and 640x400 px");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::pair<const char *, gca::intrinsics> intrinsics[] = {
        {"fx 0", {0.0f, 306.14f, 317.41f, 199.13f}},
        {"fy -1", {306.18f, -1.0f, 317.41f, 199.13f}},
        {"cx NaN", {306.18f, 306.14f, nan, 199.13f}},
    };
    for (const auto &[name, k] : intrinsics)
        check_align_refused(std::string("intrinsics with ") + name,
                            r.align(frame, 0,
                                    {surface.points, surface.validity_mask, surface.normals, k},
                                    identity, usable, stream),
                            identity, "intrinsics are not finite");
    Eigen::Isometry3d not_finite = identity;
    not_finite.translation().x() = std::numeric_limits<double>::infinity();
    Eigen::Isometry3d scaled = identity;
    scaled.linear() *= 1.00001;
    Eigen::Isometry3d mirrored = identity;
    mirrored.linear()(2, 2) = -1.0;
    const std::pair<const char *, Eigen::Isometry3d> initials[] = {
        {"an initial transform not finite", not_finite},
        {"an initial transform scaled by 1.00001", scaled},
        {"an initial transform mirrored", mirrored},
    };
    for (const auto &[name, initial] : initials)
        check_align_refused(name, r.align(frame, 0, surface, initial, usable, stream), initial,
                            "the initial transform is not finite, or its linear part is no "
                            "rotation");
    const struct
    {
        const char *name;
        gca::registration_parameters parameters;
        const char *expected;
    } parameters[] = {
        {"d_max 0", {0.0f, 0.005f, 20}, "the largest distance"},
        {"d_max a float step above 0.5 m",
         {std::nextafter(0.5f, 1.0f), 0.005f, 20},
         "the largest distance"},
        {"d_max NaN", {nan, 0.005f, 20}, "the largest distance"},
        {"Huber's threshold 0", {0.05f, 0.0f, 20}, "Huber's threshold"},
        {"Huber's threshold a float step above d_max",
         {0.05f, std::nextafter(0.05f, 1.0f), 20},
         "Huber's threshold"},
        {"0 iterations", {0.05f, 0.005f, 0}, "the most iterations 0"},
        {"101 iterations", {0.05f, 0.005f, 101}, "the most iterations 101"},
    };
    for (const auto &p : parameters)
        check_align_refused(p.name, r.align(frame, 0, surface, identity, p.parameters, stream),
                            identity, p.expected);
    // The largest values are taken: d_max 0.5 m, δ_G = d_max, 100 iterations.
    const gca::registration_result largest =
        r.align(frame, 0, surface, identity, {0.5f, 0.5f, 100}, stream);
    print_result("the largest parameters", largest);
    check(largest.converged(), "the largest parameters are taken");

    // cudaMalloc's refusal of 2^52 bytes, 4 PiB on a board of 7.4 GiB [S: docs/HARDWARE.md],
    // leaves its error as CUDA's last error.
    void *never_granted = nullptr;
    check(cudaMalloc(&never_granted, std::size_t{1} << 52) != cudaSuccess,
          "cudaMalloc refuses 2^52 bytes");
    const gca::registration_result failed = r.align(frame, 0, surface, identity, usable, stream);
    print_result("CUDA's last error left by an earlier call", failed);
    check(failed.end == gca::registration_end::failed &&
              failed.message.find("left by an earlier call") != std::string::npos &&
              failed.transform.matrix() == identity.matrix(),
          "CUDA's earlier error: failed with its message and the initial transform");
    gca_test::check_no_cuda_error("CUDA's earlier error");
}
} // namespace

/// Requests every buffer, closes the pool, and runs the cases.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    stage s;
    if (!gca_test::create_stream(s.stream))
        return gca_test::check_summary("test_registration");
    const gca::frame_size small = gca_test::small_description.size;
    const gca::frame_size large = gca_test::large_description.size;
    s.small_frame = gca_test::granted("a GPU frame", gca::gpu_frame::request(s.pool, small));
    s.large_frame = gca_test::granted("a GPU frame", gca::gpu_frame::request(s.pool, large));
    s.small_surface = gca_test::granted("a pyramid", gca::frame_pyramid::request(s.pool, small));
    s.small_moved = gca_test::granted("a pyramid", gca::frame_pyramid::request(s.pool, small));
    s.large_surface = gca_test::granted("a pyramid", gca::frame_pyramid::request(s.pool, large));
    s.large_moved = gca_test::granted("a pyramid", gca::frame_pyramid::request(s.pool, large));
    s.registration = gca_test::granted("a registration", gca::registration::request(s.pool, large));
    s.small_only = gca_test::granted("a registration", gca::registration::request(s.pool, small));
    s.pool.close();

    check_recoveries(s);
    check_degenerate(s);
    check_few_correspondences(s);
    check_patches(s);
    check_ends(s);
    check_refusals(s);

    check(s.pool.refused_while_closed() == 0, "the closed pool was asked for nothing");
    gca_test::check_no_cuda_error("the end");
    cudaStreamDestroy(s.stream);
    return gca_test::check_summary("test_registration");
}
