// SPDX-License-Identifier: 0BSD
/// @file test_registration.cpp
/// The registration module's registration (registration/registration.hpp) on pairs of frames of the
/// tests' synthetic source whose transform is known; the method is docs/methods/registration.md,
/// which also keeps the agreement with the prototype, measured before it was retired. Frame A is
/// seen from pose P_A and frame B from P_B = P_A Δ, so that Δ carries B's camera frame into A's.
/// B's pyramid is registered against A's pyramid's levels as the surface, from the identity, and
/// the result is compared with Δ. Every result that says converged must carry Δ within its case's
/// tolerance. "The full method" is colored ICP coarse to fine with the object class's starting
/// values; "the point-to-plane term on level 0" is the registration as point-to-plane-registration
/// built it, the coarser levels skipped and λ = 1, whose systems' sums are that registration's bit
/// for bit (docs/methods/registration.md, "Accuracy").
/// - Recovered by the full method, at both stream sizes: a tabletop object, a box and a sphere on a
///   plane, from about 0.5 m; a large object, a box and a sphere on a floor, from about 2.7 m; and
///   the same from about 2 m, the points 2 m away. Converged, within a tolerance of Δ, with the
///   fitness, the RMSE of the depth's rounding and a conditioning above τ. Frame A against itself:
///   converged at once at every level, at the identity exactly.
/// - The color term: a textured plane moved within itself is recovered by the full method, and is
///   degenerate with the color term off (λ = 1), at both stream sizes; the same plane without
///   texture is degenerate with the color term on. Where a weak texture's color disagrees with the
///   geometry, λ decides, and the geometry wins at the starting value. A bright patch the surface
///   does not hold is held off by Huber's weight on the color term.
/// - Coarse to fine: the large box turned by 1.5 degrees, too much for level 0 alone, is recovered
///   through the pyramid; the large box shifted by 147 mm, beyond level 0's d_max, is recovered
///   with d_max per level and not with level 0's at every level.
/// - The point-to-plane term on level 0: degenerate, never converged, a single plane, tilted near
///   and 2 m away, at both stream sizes, with three eigenvalues below τ, and the shared tabletop
///   box on its plane; the same with λ = 1 through the pyramid. Few correspondences, and none,
///   there and by the full method. Patches of frame B the surface does not hold, each held off by
///   its rule. The ends: the iterations used up, and the last system deciding.
/// - Refusals as values, with the initial transform: an empty registration, an empty pyramid, a
///   frame larger than the registration's size, a level of the surface with empty images, images
///   of two sizes, a size larger than the registration's or intrinsics that cannot be used, an
///   initial transform that is not finite or not rigid, and each parameter outside its range,
///   while the largest of each is taken; CUDA's last error left by an earlier call ends it as
///   failed. CUDA's last error is clear after each.
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
#include "registration_cases.hpp"
#include "scan/object_class.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "transform_checks.hpp"
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
using gca_test::check_converged_is_known;
using gca_test::check_near;
using gca_test::check_within;
using gca_test::error_of;
using gca_test::large_delta;
using gca_test::large_object_pose;
using gca_test::large_object_scene;
using gca_test::large_parameters;
using gca_test::large_range;
using gca_test::large_turn_delta;
using gca_test::make_transform;
using gca_test::normals_parameters;
using gca_test::on_level_0;
using gca_test::print_result;
using gca_test::surface_of;
using gca_test::tabletop_delta;
using gca_test::tabletop_object_pose;
using gca_test::tabletop_object_scene;
using gca_test::tabletop_parameters;
using gca_test::tabletop_range;
using gca_test::transform_error;
using gca_test::transform_tolerance;

/// Gives parameters with the color term off: λ = 1.
/// @param p [in] the parameters.
/// @return p with λ = 1.
gca::registration_parameters without_color(gca::registration_parameters p)
{
    p.point_to_plane_weight = 1.0f;
    return p;
}

/// Gives the parameters of the registration as point-to-plane-registration built it: the
/// point-to-plane term on level 0 alone, with d_max, δ_G and level 0's iterations of p.
/// @param p [in] the parameters.
/// @return p on level 0 alone, with λ = 1.
gca::registration_parameters point_to_plane_on_level_0(const gca::registration_parameters &p)
{
    return without_color(on_level_0(p));
}

/// How far the tabletop object's transform may lie from Δ by the point-to-plane term on level 0:
/// 0.005 degrees and 0.06 mm, about twice the largest errors at both stream sizes, 0.00254 degrees
/// and 0.0311 mm [M 2026-11-14, point-to-plane-registration's test]; the patches' and the ends'
/// cases.
constexpr double level_0_tabletop_rotation_tolerance_deg = 0.005;
constexpr double level_0_tabletop_translation_tolerance_m = 0.06e-3;

/// How far a tabletop object's transform recovered by the full method may lie from Δ: 0.02 degrees
/// and 0.15 mm, about twice the largest errors at both stream sizes, 0.00991 degrees and 0.0761 mm
/// at 1280x800 [M 2026-11-14, this test]. The color term holds about ten times the point-to-plane
/// term's information here and moves the minimum by what its linear model of the texture misses
/// within a pixel: the point-to-plane term alone came within 0.00254 degrees and 0.0311 mm.
constexpr double tabletop_rotation_tolerance_deg = 0.02;
constexpr double tabletop_translation_tolerance_m = 0.15e-3;

/// How far a large object's transform recovered by the full method may lie from Δ, from 2.7 m and
/// from 2 m and turned by 1.5 degrees: 0.03 degrees and 1.4 mm, about twice the largest errors,
/// 0.0136 degrees and 0.671 mm, from 2.7 m at 640x400, where the box's texture has 5 to 9 pixels
/// to a wave [M 2026-11-14, this test; S: computed, waves of 45 to 80 mm, 8.8 mm a pixel]; the
/// point-to-plane term alone came within 0.00335 degrees and 0.142 mm there.
constexpr double large_rotation_tolerance_deg = 0.03;
constexpr double large_translation_tolerance_m = 1.4e-3;

/// The largest inlier RMSE of a recovered transform: 0.5 mm, above the depth's rounding, 1/sqrt(12)
/// of a unit of 1 mm, 0.289 mm [S: the uniform distribution]; the largest is 0.33 mm
/// [M 2026-11-14, this test].
constexpr double recovered_rmse_tolerance_m = 0.5e-3;

/// The least fitness of a recovered transform: 0.95; the least is 0.967 [M 2026-11-14, this test].
constexpr double recovered_least_fitness = 0.95;

/// How far the transform may lie from Δ with a patch of 80 x 80 pixels 20 mm deeper in frame B
/// (check_patches()): 0.23 degrees and 2 mm, about twice what Huber's weight leaves, 0.115
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

/// How far the textured plane's recovered transform may lie from its in-plane motion
/// (check_textured_plane()): 0.002 degrees and 0.07 mm, about twice the largest errors, 0.000907
/// degrees and 0.0355 mm, the tilted plane at 640x400 [M 2026-11-14, this test].
constexpr double plane_rotation_tolerance_deg = 0.002;
constexpr double plane_translation_tolerance_m = 0.07e-3;

/// The bright patch on the textured plane (check_color_patch()): 150 levels added to every channel
/// of frame B's color inside the patch window, at most 255, up to 0.59 of the intensity's range,
/// six times δ_C [?: a choice, a highlight].
constexpr int bright_patch_levels = 150;

/// How far the textured plane's transform may lie from its motion with the bright patch in frame B:
/// 0.007 degrees and 0.23 mm, about twice what Huber's weight on the color term leaves, 0.00359
/// degrees and 0.116 mm, below what the color term's plain least squares leaves, 0.0103 degrees and
/// 0.332 mm [M 2026-11-14, this test, and with the weight 1].
constexpr double bright_patch_rotation_tolerance_deg = 0.007;
constexpr double bright_patch_translation_tolerance_m = 0.23e-3;

/// The weak texture whose color disagrees with the geometry (check_color_against_geometry()): a
/// grey ramp along the picture's rows from 32 to 224 levels over its width, 0.75 of the intensity's
/// range, in frame B shifted by ramp_shift_px against frame A [?: a choice].
constexpr double ramp_low_levels = 32.0;
constexpr double ramp_high_levels = 224.0;
constexpr std::uint32_t ramp_shift_px = 20;

/// How far the geometry lets the ramp's color move the transform at λ = 0.968: 4 mm, about twice
/// the 1.94 mm it moves, of the about 33 mm the color alone asks for, 20 px at 0.5 m; at λ = 0.5
/// it moves 21.1 mm [M 2026-11-14, this test; S: computed, 20 · 0.5 / 306.18 m].
constexpr double ramp_translation_tolerance_m = 4.0e-3;

/// The tolerances of each object class and case; a case whose transform is not recovered checks a
/// result that says converged against its object class's all the same.
constexpr transform_tolerance level_0_tabletop_tolerance = {
    level_0_tabletop_rotation_tolerance_deg, level_0_tabletop_translation_tolerance_m};
constexpr transform_tolerance tabletop_tolerance = {tabletop_rotation_tolerance_deg,
                                                    tabletop_translation_tolerance_m};
constexpr transform_tolerance large_tolerance = {large_rotation_tolerance_deg,
                                                 large_translation_tolerance_m};
constexpr transform_tolerance deeper_patch_tolerance = {deeper_patch_rotation_tolerance_deg,
                                                        deeper_patch_translation_tolerance_m};
constexpr transform_tolerance plane_tolerance = {plane_rotation_tolerance_deg,
                                                 plane_translation_tolerance_m};
constexpr transform_tolerance bright_patch_tolerance = {bright_patch_rotation_tolerance_deg,
                                                        bright_patch_translation_tolerance_m};

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
    return s.registration.align(buffers.moved, surface_of(buffers.surface), initial, parameters,
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

/// Counts the levels a registration's parameters iterate.
/// @param parameters [in] the parameters.
/// @return the levels with at least one iteration.
std::uint32_t levels_iterated(const gca::registration_parameters &parameters)
{
    return static_cast<std::uint32_t>(std::count_if(
        parameters.levels.begin(), parameters.levels.end(),
        [](const gca::registration_level_parameters &l) { return l.max_iterations > 0; }));
}

/// Gives a stream size's name for the printed lines.
/// @param description [in] the frames' description, 640x400 or 1280x800.
/// @return " at 640x400" or " at 1280x800".
std::string at_size(const gca::frame_description &description)
{
    return description.size == gca_test::small_description.size ? " at 640x400" : " at 1280x800";
}

/// The stream sizes every size-dependent case runs at.
const std::array<gca::frame_description, 2> both_sizes = {gca_test::small_description,
                                                          gca_test::large_description};

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
/// at once at every level, at the identity exactly: the same points and intensities give residuals
/// of 0 and a step of 0.
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
    check(itself.converged() && itself.iterations == levels_iterated(c.parameters),
          (self + ": converged after one iteration at each level").c_str());
    check(itself.transform.matrix() == Eigen::Matrix4d::Identity(),
          (self + ": the identity exactly").c_str());
    check(itself.fitness == 1.0 && itself.inlier_rmse_m == 0.0,
          (self + ": fitness 1 and RMSE 0").c_str());
}

/// Runs the cases whose transform the full method recovers: the tabletop object from about 0.5 m,
/// the large object from about 2.7 m and from about 2 m, each at both stream sizes.
/// @param s [in,out] the stage.
void check_recoveries(stage &s)
{
    const std::optional<Eigen::Isometry3d> tabletop_pose = tabletop_object_pose();
    const std::optional<Eigen::Isometry3d> large_pose = large_object_pose();
    const std::optional<Eigen::Isometry3d> near_pose =
        gca_test::look_at(Eigen::Vector3d(1.6, -0.9, 1.3), Eigen::Vector3d(0.0, 0.0, 0.5), 0.0);
    check(tabletop_pose && large_pose && near_pose, "the recoveries' poses");
    if (!tabletop_pose || !large_pose || !near_pose)
        return;
    for (const gca::frame_description &description : both_sizes)
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

/// Counts a result's eigenvalues below τ.
/// @param result [in] the result.
/// @return how many of its six eigenvalues lie below registration_degeneracy_threshold.
long count_below_threshold(const gca::registration_result &result)
{
    return std::count_if(result.eigenvalues.begin(), result.eigenvalues.end(),
                         [](double e) { return e < gca::registration_degeneracy_threshold; });
}

/// A case of a pair the point-to-plane term finds degenerate.
struct degenerate_case
{
    std::string name;                        ///< the case, for the printed lines
    gca_test::synthetic_scene scene;         ///< the scene
    Eigen::Isometry3d pose_a;                ///< frame A's pose, camera to world
    gca::organized_points_parameters range;  ///< the depth range
    gca::registration_parameters parameters; ///< the object class's full method
    transform_tolerance tolerance;           ///< the tolerance of its object class
    bool plane;                              ///< whether the scene is a single plane
};

/// The tilted plane's pose: about 0.65 m from the point of the plane it looks at, rolled by 10
/// degrees [?: a choice]; the plane's depth runs from about 0.4 m to 0.8 m.
/// @return the pose, camera to world; nothing when look_at() refuses it.
std::optional<Eigen::Isometry3d> tilted_plane_pose()
{
    return gca_test::look_at({0.0, -0.55, 0.35}, {0.0, 0.0, 0.0}, 10.0);
}

/// The pose of the plane 2 m away: 2 m from the point it looks at, from 70 degrees above the
/// plane, rolled by 10 degrees [?: a choice].
/// @return the pose, camera to world; nothing when look_at() refuses it.
std::optional<Eigen::Isometry3d> far_plane_pose()
{
    const double elevation_rad = 70.0 * std::numbers::pi / 180.0;
    return gca_test::look_at({0.0, -2.0 * std::cos(elevation_rad), 2.0 * std::sin(elevation_rad)},
                             {0.0, 0.0, 0.0}, 10.0);
}

/// The single plane of the plane cases: 6 m x 6 m [?: a choice], larger than every view of it.
const gca_test::synthetic_scene single_plane = {.plane = gca_test::synthetic_plane{3.0, 3.0},
                                                .trajectory = {}};

/// Runs the degenerate cases: a single plane, tilted near and 2 m away, and the shared tabletop box
/// on its plane, each at both stream sizes, moved by the tabletop object's Δ, which shifts along
/// the plane: degenerate, not converged; a plane with three eigenvalues below τ and three above.
/// Each by the point-to-plane term on level 0, as point-to-plane-registration measured them, and by
/// the point-to-plane term through the pyramid (λ = 1). The textured box on its plane is recovered
/// by the full method, its texture holding what its faces hold too weakly.
/// @param s [in,out] the stage.
void check_degenerate(stage &s)
{
    const std::optional<Eigen::Isometry3d> tilted_pose = tilted_plane_pose();
    const std::optional<Eigen::Isometry3d> far_pose = far_plane_pose();
    const std::optional<std::vector<Eigen::Isometry3d>> walk =
        gca_test::walk_around(gca_test::tabletop_walk());
    check(tilted_pose && far_pose && walk, "the degenerate cases' poses");
    if (!tilted_pose || !far_pose || !walk)
        return;
    const gca_test::synthetic_scene box = {
        .plane = gca_test::tabletop_plane, .box = gca_test::tabletop_box, .trajectory = {}};
    for (const gca::frame_description &description : both_sizes)
    {
        const std::string at = at_size(description);
        const degenerate_case cases[] = {
            {"tilted plane" + at, single_plane, *tilted_pose, tabletop_range, tabletop_parameters,
             tabletop_tolerance, true},
            {"plane 2 m away" + at, single_plane, *far_pose, large_range, large_parameters,
             large_tolerance, true},
            {"tabletop box on its plane" + at, box, (*walk)[0], tabletop_range, tabletop_parameters,
             tabletop_tolerance, false},
        };
        for (const degenerate_case &c : cases)
        {
            const gca_test::rendered pair =
                render_pair(c.scene, c.pose_a, tabletop_delta, description);
            const std::pair<std::string, gca::registration_parameters> ways[] = {
                {c.name, point_to_plane_on_level_0(c.parameters)},
                {c.name + ", the point-to-plane term through the pyramid",
                 without_color(c.parameters)},
            };
            for (const auto &[name, parameters] : ways)
            {
                const gca::registration_result result = register_pair(s, pair, c.range, parameters);
                print_result(name, result);
                check(result.end == gca::registration_end::degenerate,
                      (name + ": degenerate").c_str());
                check_converged_is_known(name, result, tabletop_delta, c.tolerance);
                const long below = count_below_threshold(result);
                if (c.plane)
                    check(below == 3, (name + ": three eigenvalues below the threshold").c_str());
                else
                    check(below >= 1, (name + ": an eigenvalue below the threshold").c_str());
            }
            if (c.plane)
                continue;
            // The box's texture holds the motion its faces hold too weakly.
            const std::string colored = c.name + ", the full method";
            const gca::registration_result result = register_pair(s, pair, c.range, c.parameters);
            print_result(colored, result);
            check(result.converged(), (colored + ": converged").c_str());
            check_converged_is_known(colored, result, tabletop_delta, c.tolerance);
        }
    }
}

/// Builds a frame with a rendered frame's depth and color changed.
/// @param rendered [in] the rendered frame.
/// @param depth_of [in] the new depth value of pixel (u, v) from its rendered one d, in depth
///        units, called as depth_of(u, v, d).
/// @param color_of [in] the new color of pixel (u, v) from its rendered one c, red, green and
///        blue, called as color_of(u, v, c).
/// @return the frame, index and timestamp 0.
template <typename Depth, typename Color>
gca::frame rebuilt(const gca::frame &rendered, Depth depth_of, Color color_of)
{
    const gca::frame_description &d = rendered.description();
    return gca_test::built_frame(
        d,
        [&](std::uint32_t u, std::uint32_t v) -> std::uint16_t {
            return depth_of(u, v, rendered.depth()[static_cast<std::size_t>(v) * d.size.width + u]);
        },
        [&](std::uint32_t u, std::uint32_t v) {
            const std::size_t pixel = 3 * (static_cast<std::size_t>(v) * d.size.width + u);
            return color_of(u, v,
                            std::array<std::uint8_t, 3>{rendered.color()[pixel],
                                                        rendered.color()[pixel + 1],
                                                        rendered.color()[pixel + 2]});
        });
}

/// Builds a frame with a rendered frame's color and its depth changed.
/// @param rendered [in] the rendered frame.
/// @param depth_of [in] the new depth value of pixel (u, v) from its rendered one d, in depth
///        units, called as depth_of(u, v, d).
/// @return the frame, index and timestamp 0.
template <typename Depth> gca::frame with_depth(const gca::frame &rendered, Depth depth_of)
{
    return rebuilt(
        rendered, depth_of,
        [](std::uint32_t, std::uint32_t, const std::array<std::uint8_t, 3> &c) { return c; });
}

/// Builds a frame with a rendered frame's depth and its color changed.
/// @param rendered [in] the rendered frame.
/// @param color_of [in] the new color of pixel (u, v) from its rendered one c, called as
///        color_of(u, v, c).
/// @return the frame, index and timestamp 0.
template <typename Color> gca::frame with_color(const gca::frame &rendered, Color color_of)
{
    return rebuilt(
        rendered, [](std::uint32_t, std::uint32_t, std::uint16_t z) { return z; }, color_of);
}

/// Builds a frame with a rendered frame's depth and grey 128 for every color: no texture.
/// @param rendered [in] the rendered frame.
/// @return the frame, index and timestamp 0.
gca::frame untextured(const gca::frame &rendered)
{
    return with_color(rendered,
                      [](std::uint32_t u, std::uint32_t v, const std::array<std::uint8_t, 3> &) {
                          return gca_test::grey(u, v);
                      });
}

/// The textured plane's motion within itself: a turn of 0.5 degrees about the plane's normal
/// through the point the views look at, the world's origin, then 3 mm along the world's x and -2 mm
/// along its y, the motion of the prototype's test of a textured plane [?: a choice]. Frame B is
/// seen from this motion applied to frame A's pose, so that the plane holds both views alike and
/// only its texture moves.
/// @return the motion, in the world frame.
Eigen::Isometry3d in_plane_motion()
{
    return make_transform(Eigen::Vector3d::UnitZ(), 0.5, Eigen::Vector3d(0.003, -0.002, 0.0));
}

/// Gives the transform from frame B's camera frame into frame A's when B is seen from a world
/// motion of A's pose.
/// @param pose_a [in] frame A's pose, camera to world.
/// @param motion [in] the motion of the camera in the world frame: P_B = motion P_A.
/// @return Δ = P_A^-1 P_B.
Eigen::Isometry3d delta_of(const Eigen::Isometry3d &pose_a, const Eigen::Isometry3d &motion)
{
    return pose_a.inverse() * motion * pose_a;
}

/// Runs the cases of a single textured plane moved within itself, the tilted plane and the plane
/// 2 m away at both stream sizes: the full method recovers the motion, converged, within
/// plane_tolerance, with every eigenvalue at least τ; with the color term off it is degenerate,
/// three eigenvalues below τ, as point-to-plane-registration found the plane; and the same plane
/// without texture is degenerate with the color term on.
/// @param s [in,out] the stage.
void check_textured_plane(stage &s)
{
    const std::optional<Eigen::Isometry3d> tilted_pose = tilted_plane_pose();
    const std::optional<Eigen::Isometry3d> far_pose = far_plane_pose();
    check(tilted_pose && far_pose, "the textured planes' poses");
    if (!tilted_pose || !far_pose)
        return;
    for (const gca::frame_description &description : both_sizes)
    {
        const std::string at = at_size(description);
        const struct
        {
            std::string name;
            Eigen::Isometry3d pose;
            gca::organized_points_parameters range;
            gca::registration_parameters parameters;
        } cases[] = {
            {"textured tilted plane" + at, *tilted_pose, tabletop_range, tabletop_parameters},
            {"textured plane 2 m away" + at, *far_pose, large_range, large_parameters},
        };
        for (const auto &c : cases)
        {
            const Eigen::Isometry3d delta = delta_of(c.pose, in_plane_motion());
            const gca_test::rendered pair = render_pair(single_plane, c.pose, delta, description);
            if (pair.frames.size() != 2)
                continue;
            const gca::registration_result colored = register_pair(s, pair, c.range, c.parameters);
            print_result(c.name, colored);
            check(colored.converged(), (c.name + ": converged").c_str());
            check_converged_is_known(c.name, colored, delta, plane_tolerance);
            check(colored.eigenvalues[0] >= gca::registration_degeneracy_threshold,
                  (c.name + ": the smallest eigenvalue at least the threshold").c_str());

            const std::string off = c.name + ", the color term off";
            const gca::registration_result geometric =
                register_pair(s, pair, c.range, without_color(c.parameters));
            print_result(off, geometric);
            check(geometric.end == gca::registration_end::degenerate &&
                      count_below_threshold(geometric) == 3,
                  (off + ": degenerate, three eigenvalues below the threshold").c_str());
            check_converged_is_known(off, geometric, delta, plane_tolerance);

            const std::string grey = c.name + " without texture";
            const gca::registration_result plain =
                register_frames(s, untextured(pair.frames[0]), untextured(pair.frames[1]), c.range,
                                c.parameters, Eigen::Isometry3d::Identity());
            print_result(grey, plain);
            check(plain.end == gca::registration_end::degenerate &&
                      count_below_threshold(plain) == 3,
                  (grey + ": degenerate, three eigenvalues below the threshold").c_str());
            check_converged_is_known(grey, plain, delta, plane_tolerance);
        }
    }
}

/// Runs the case of a transform too large for level 0 alone: the large box without its sphere from
/// about 2.7 m, turned by 1.5 degrees (large_turn_delta), at both stream sizes. Level 0 alone does
/// not converge to it; the full method recovers it through the pyramid within the large object's
/// tolerance.
/// @param s [in,out] the stage.
void check_large_turn(stage &s)
{
    const std::optional<Eigen::Isometry3d> pose = large_object_pose();
    check(pose.has_value(), "the large turn's pose");
    if (!pose)
        return;
    gca_test::synthetic_scene box = large_object_scene();
    box.sphere = std::nullopt;
    for (const gca::frame_description &description : both_sizes)
    {
        const std::string name = "large box turned by 1.5 deg" + at_size(description);
        const gca_test::rendered pair = render_pair(box, *pose, large_turn_delta, description);
        const gca::registration_result alone =
            register_pair(s, pair, large_range, on_level_0(large_parameters));
        print_result(name + ", level 0 alone", alone);
        check(!alone.converged(), (name + ", level 0 alone: not converged").c_str());
        check_converged_is_known(name + ", level 0 alone", alone, large_turn_delta,
                                 large_tolerance);
        const gca::registration_result pyramid =
            register_pair(s, pair, large_range, large_parameters);
        print_result(name, pyramid);
        check(pyramid.converged(), (name + ": converged through the pyramid").c_str());
        check_converged_is_known(name, pyramid, large_turn_delta, large_tolerance);
    }
}

/// Gives parameters with level 0's d_max at every level: one largest distance for the pyramid.
/// @param p [in] the parameters.
/// @return p with each level's max_distance_m level 0's.
gca::registration_parameters one_max_distance(gca::registration_parameters p)
{
    for (gca::registration_level_parameters &level : p.levels)
        level.max_distance_m = p.levels[0].max_distance_m;
    return p;
}

/// The large object's shift beyond level 0's d_max of 0.10 m: 0.5 degrees about large_delta's
/// axis and a shift of 147 mm [?: a choice].
const Eigen::Isometry3d large_shift_delta =
    make_transform(Eigen::Vector3d(-0.4, 0.7, 0.3), 0.5, Eigen::Vector3d(0.09, 0.06, -0.10));

/// Runs the case of a misalignment beyond level 0's d_max: the large box without its sphere from
/// about 2.7 m, shifted by 147 mm (large_shift_delta), at 640x400. With d_max per level, 2^l times
/// level 0's, the full method recovers it within the large object's tolerance; with level 0's
/// d_max at every level it does not converge. At 1280x800, whose level 2 is 640x400's level 1,
/// neither recovers it (docs/methods/registration.md, "Coarse to fine").
/// @param s [in,out] the stage.
void check_large_shift(stage &s)
{
    const std::optional<Eigen::Isometry3d> pose = large_object_pose();
    check(pose.has_value(), "the large shift's pose");
    if (!pose)
        return;
    gca_test::synthetic_scene box = large_object_scene();
    box.sphere = std::nullopt;
    const std::string name = "large box shifted by 147 mm at 640x400";
    const gca_test::rendered pair =
        render_pair(box, *pose, large_shift_delta, gca_test::small_description);
    const gca::registration_result per_level =
        register_pair(s, pair, large_range, large_parameters);
    print_result(name, per_level);
    check(per_level.converged(), (name + ": converged with d_max per level").c_str());
    check_converged_is_known(name, per_level, large_shift_delta, large_tolerance);
    const gca::registration_result one =
        register_pair(s, pair, large_range, one_max_distance(large_parameters));
    print_result(name + ", one d_max", one);
    check(!one.converged(), (name + ", one d_max: not converged").c_str());
    check_converged_is_known(name + ", one d_max", one, large_shift_delta, large_tolerance);
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

/// Checks a case with few correspondences or none: not converged, ended for too few, its fitness
/// below the least, each level that iterates after its first system.
/// @param what [in] the case.
/// @param result [in] its result.
/// @param none [in] whether it has no correspondence at all: then its fitness is 0.
/// @param levels [in] the levels the case's parameters iterate.
void check_few(const std::string &what, const gca::registration_result &result, bool none,
               std::uint32_t levels)
{
    print_result(what, result);
    check(result.end == gca::registration_end::few_correspondences,
          (what + ": ended for too few correspondences").c_str());
    check(result.iterations == levels, (what + ": after one iteration at each level").c_str());
    if (none)
        check(result.fitness == 0.0, (what + ": fitness 0").c_str());
    else
        check(result.fitness > 0.0 && result.fitness < gca::registration_min_fitness,
              (what + ": a fitness above 0 below the least").c_str());
}

/// Runs the cases with few correspondences and none, on the tabletop object's pair at 640x400, by
/// the point-to-plane term on level 0 and by the full method: a surface with depth only in a window
/// of 16 x 16 pixels at the picture's centre; a surface without depth; a frame without depth; and
/// the frame moved 5 m to the side of the surface's view.
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
    const Eigen::Isometry3d beside =
        make_transform(Eigen::Vector3d::UnitZ(), 0.0, Eigen::Vector3d(5.0, 0.0, 0.0));
    const std::pair<std::string, gca::registration_parameters> ways[] = {
        {"", point_to_plane_on_level_0(tabletop_parameters)},
        {", the full method", tabletop_parameters},
    };
    for (const auto &[way, parameters] : ways)
    {
        const std::uint32_t levels = levels_iterated(parameters);
        check_few(
            "a surface with depth in 16 x 16 pixels" + way,
            register_frames(s, windowed(a, few_window), b, tabletop_range, parameters, identity),
            false, levels);
        check_few(
            "a surface without depth" + way,
            register_frames(s, windowed(a, {0, 0, 0}), b, tabletop_range, parameters, identity),
            true, levels);
        check_few(
            "a frame without depth" + way,
            register_frames(s, a, windowed(b, {0, 0, 0}), tabletop_range, parameters, identity),
            true, levels);
        check_few("the frame 5 m beside the surface's view" + way,
                  register_frames(s, a, b, tabletop_range, parameters, beside), true, levels);
    }
}

/// Runs the cases of a patch of frame B the surface does not hold, on the tabletop object's pair at
/// 640x400, by the point-to-plane term on level 0, each a rule of the association or of the weight
/// at work: the patch 20 mm deeper, within d_max and beyond δ_G, whose pull Huber's weight caps;
/// the patch 100 mm deeper, beyond d_max, whose pairs the largest distance cuts; and a patch of
/// ridges within d_max of the plane, whose slopes' pairs the largest angle between normals cuts
/// (ridge_height).
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
         level_0_tabletop_tolerance},
        {"a patch of ridges",
         patched(b,
                 [](std::uint32_t u, std::uint32_t) {
                     return static_cast<std::uint16_t>(ridge_height * (u % ridge_period_px) /
                                                       ridge_period_px);
                 }),
         level_0_tabletop_tolerance},
    };
    for (const auto &c : cases)
    {
        const gca::registration_result result =
            register_frames(s, a, c.frame, tabletop_range,
                            point_to_plane_on_level_0(tabletop_parameters), identity);
        print_result(c.name, result);
        check(result.converged(), (c.name + ": converged").c_str());
        check_converged_is_known(c.name, result, tabletop_delta, c.tolerance);
    }
}

/// Runs the cases of the two ends a registration that steps can come to besides convergence, by the
/// point-to-plane term on level 0. The tabletop object's pair at 640x400 with 2 iterations, where
/// it needs 4: the iterations used up, never converged. And the tabletop object's box without its
/// sphere from 0.67 m, the frame against itself from a start 25 mm up along the table's normal: its
/// first system holds every motion, its last at the identity does not; with one iteration the
/// iterations are used up and the smallest eigenvalue lies above τ, with 20 the registration ends
/// degenerate, the last system deciding.
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
    const gca::registration_parameters level_0 = point_to_plane_on_level_0(tabletop_parameters);

    gca::registration_parameters two = level_0;
    two.levels[0].max_iterations = 2;
    const gca::registration_result short_of = register_frames(
        s, pair.frames[0], pair.frames[1], tabletop_range, two, Eigen::Isometry3d::Identity());
    print_result("2 iterations of 4", short_of);
    check(short_of.end == gca::registration_end::iterations_used_up && short_of.iterations == 2,
          "2 iterations of 4: the iterations used up, not converged");

    Eigen::Isometry3d up = Eigen::Isometry3d::Identity();
    up.translation() = far_pose->linear().transpose() * Eigen::Vector3d(0.0, 0.0, start_up_m);
    gca::registration_parameters one = level_0;
    one.levels[0].max_iterations = 1;
    const gca::registration_result first =
        register_frames(s, alone.frames[0], alone.frames[0], tabletop_range, one, up);
    print_result("the box alone, its first system", first);
    check(first.end == gca::registration_end::iterations_used_up &&
              first.eigenvalues[0] >= gca::registration_degeneracy_threshold,
          "the box alone: its first system holds every motion, the iterations used up");
    const gca::registration_result last =
        register_frames(s, alone.frames[0], alone.frames[0], tabletop_range, level_0, up);
    print_result("the box alone, its last system", last);
    check(last.end == gca::registration_end::degenerate &&
              last.eigenvalues[0] < gca::registration_degeneracy_threshold,
          "the box alone: its last system decides, degenerate");
    check_converged_is_known("the box alone", last, Eigen::Isometry3d::Identity(),
                             level_0_tabletop_tolerance);
}

/// Gives the ramp's grey at a column, shifted along the picture's rows.
/// @param u [in] the column, in px.
/// @param shift_px [in] the shift, in px: the grey of column u + shift_px.
/// @param width [in] the picture's width, in px.
/// @return red, green and blue, each the ramp's grey rounded.
std::array<std::uint8_t, 3> ramp_grey(std::uint32_t u, std::uint32_t shift_px, std::uint32_t width)
{
    const double level =
        ramp_low_levels + (ramp_high_levels - ramp_low_levels) * (u + shift_px) / width;
    const auto g = static_cast<std::uint8_t>(std::lround(std::min(level, 255.0)));
    return {g, g, g};
}

/// Runs the case where the color disagrees with the geometry, on the tabletop object's frame A at
/// 640x400 against itself, the geometry holding every motion at the identity: frame A's color
/// becomes a weak ramp along the picture's rows (ramp_grey()), frame B's the same ramp shifted by
/// ramp_shift_px, so that the color term alone would move B along the rows by that shift. λ weighs
/// the terms: at the starting value 0.968 the geometry wins and the transform stays within
/// ramp_translation_tolerance_m of the identity; at λ = 0.5 the color moves it further.
/// @param s [in,out] the stage.
void check_color_against_geometry(stage &s)
{
    const std::optional<Eigen::Isometry3d> pose = tabletop_object_pose();
    check(pose.has_value(), "the ramp's pose");
    if (!pose)
        return;
    gca_test::synthetic_scene scene = tabletop_object_scene();
    scene.trajectory = {*pose};
    const gca_test::rendered rendered = gca_test::render(scene, gca_test::small_description);
    if (rendered.frames.size() != 1)
        return;
    const std::uint32_t width = gca_test::small_description.size.width;
    const gca::frame a = with_color(rendered.frames[0], [&](std::uint32_t u, std::uint32_t,
                                                            const std::array<std::uint8_t, 3> &) {
        return ramp_grey(u, 0, width);
    });
    const gca::frame b = with_color(rendered.frames[0], [&](std::uint32_t u, std::uint32_t,
                                                            const std::array<std::uint8_t, 3> &) {
        return ramp_grey(u, ramp_shift_px, width);
    });
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    gca::registration_parameters even = tabletop_parameters;
    even.point_to_plane_weight = 0.5f;
    const gca::registration_result weighed =
        register_frames(s, a, b, tabletop_range, tabletop_parameters, identity);
    const gca::registration_result halved =
        register_frames(s, a, b, tabletop_range, even, identity);
    print_result("a ramp against the geometry, λ 0.968", weighed);
    print_result("a ramp against the geometry, λ 0.5", halved);
    const transform_error at_start = error_of(weighed.transform, identity);
    const transform_error at_half = error_of(halved.transform, identity);
    std::printf("a ramp against the geometry: %.3g mm from the identity at λ 0.968, %.3g mm at "
                "λ 0.5\n",
                1e3 * at_start.translation_m, 1e3 * at_half.translation_m);
    check(weighed.converged(), "a ramp against the geometry, λ 0.968: converged");
    check_within("a ramp against the geometry, λ 0.968: translation from the identity (m)",
                 at_start.translation_m, ramp_translation_tolerance_m);
    check(at_half.translation_m > at_start.translation_m,
          "a ramp against the geometry: λ 0.5 moves the transform further than λ 0.968");
}

/// Runs the case of a bright patch frame B shows and the surface does not hold, a highlight: the
/// textured tilted plane moved within itself at 640x400, frame B's color raised by
/// bright_patch_levels inside the patch window. Huber's weight on the color term caps the patch's
/// pull: converged, within bright_patch_tolerance of the motion.
/// @param s [in,out] the stage.
void check_color_patch(stage &s)
{
    const std::optional<Eigen::Isometry3d> pose = tilted_plane_pose();
    check(pose.has_value(), "the bright patch's pose");
    if (!pose)
        return;
    const Eigen::Isometry3d delta = delta_of(*pose, in_plane_motion());
    const gca_test::rendered pair =
        render_pair(single_plane, *pose, delta, gca_test::small_description);
    if (pair.frames.size() != 2)
        return;
    const gca::frame bright = with_color(
        pair.frames[1], [](std::uint32_t u, std::uint32_t v, const std::array<std::uint8_t, 3> &c) {
            if (!patch_window.holds(u, v))
                return c;
            std::array<std::uint8_t, 3> raised{};
            for (std::size_t k = 0; k < 3; k++)
                raised[k] = static_cast<std::uint8_t>(std::min(c[k] + bright_patch_levels, 255));
            return raised;
        });
    const gca::registration_result result =
        register_frames(s, pair.frames[0], bright, tabletop_range, tabletop_parameters,
                        Eigen::Isometry3d::Identity());
    print_result("a bright patch", result);
    check(result.converged(), "a bright patch: converged");
    check_converged_is_known("a bright patch", result, delta, bright_patch_tolerance);
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

/// Gives parameters with one level's values replaced.
/// @param level [in] the level, below frame_pyramid::level_count.
/// @param max_distance_m [in] its d_max, in m.
/// @param max_iterations [in] its most iterations.
/// @return the tabletop object's parameters with that level's values replaced.
gca::registration_parameters with_level(std::size_t level, float max_distance_m,
                                        std::uint32_t max_iterations)
{
    gca::registration_parameters p = tabletop_parameters;
    p.levels[level] = {max_distance_m, max_iterations};
    return p;
}

/// Gives the tabletop object's parameters with δ_G and λ replaced.
/// @param huber_threshold_m [in] δ_G, in m.
/// @param point_to_plane_weight [in] λ.
/// @return the parameters.
gca::registration_parameters with_weights(float huber_threshold_m, float point_to_plane_weight)
{
    gca::registration_parameters p = tabletop_parameters;
    p.huber_threshold_m = huber_threshold_m;
    p.point_to_plane_weight = point_to_plane_weight;
    return p;
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
    const gca::surface_levels surface = surface_of(s.small_surface);
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const gca::registration_parameters usable = tabletop_parameters;
    gca::registration &r = s.registration;
    cudaStream_t stream = s.stream;

    check_align_refused("an empty registration",
                        gca::registration{}.align(frame, surface, identity, usable, stream),
                        identity, "holds no buffers");
    check_align_refused("an empty pyramid",
                        r.align(gca::frame_pyramid{}, surface, identity, usable, stream), identity,
                        "holds no levels");
    check_align_refused(
        "a frame of 1280x800 for a registration of 640x400",
        s.small_only.align(s.large_surface, surface, identity, usable, stream), identity,
        "the frame's level 0 of 1280x800 px is larger than the registration's 640x400 px");
    const gca::gpu_image<float3> no_points;
    const gca::gpu_image<std::uint8_t> no_marks;
    const gca::gpu_image<float> no_intensity;
    const gca::organized_surface &level_0 = surface[0];
    const gca::organized_surface &level_1 = surface[1];
    const gca::organized_surface &level_2 = surface[2];
    const gca::organized_surface empty = {no_points, no_marks, no_points, no_intensity,
                                          level_1.intrinsics};
    check_align_refused("a surface's level 1 of empty images",
                        r.align(frame, {level_0, empty, level_2}, identity, usable, stream),
                        identity, "level 1 of the surface: the surface's images are empty");
    check_align_refused(
        "a surface's level 0 of two sizes",
        r.align(frame,
                {gca::organized_surface{level_0.points, level_1.validity_mask, level_0.normals,
                                        level_0.intensity, level_0.intrinsics},
                 level_1, level_2},
                identity, usable, stream),
        identity,
        "level 0 of the surface: the surface's points, marks, normals and intensity are not of one "
        "size: 640x400 px, 320x200 px, 640x400 px and 640x400 px");
    check_align_refused(
        "a surface's level 2 with an intensity of another size",
        r.align(frame,
                {level_0, level_1,
                 gca::organized_surface{level_2.points, level_2.validity_mask, level_2.normals,
                                        level_1.intensity, level_2.intrinsics}},
                identity, usable, stream),
        identity, "level 2 of the surface: the surface's points, marks, normals and intensity");
    const gca::surface_levels large_surface = surface_of(s.large_surface);
    check_align_refused(
        "a surface's level 1 of 640x400 for a registration of 640x400",
        s.small_only.align(frame, {level_0, large_surface[1], level_2}, identity, usable, stream),
        identity,
        "level 1 of the surface: the surface's 640x400 px are larger than the "
        "registration's 320x200 px");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::pair<const char *, gca::intrinsics> intrinsics[] = {
        {"fx 0", {0.0f, 76.535f, 79.3525f, 49.7825f}},
        {"fy -1", {76.545f, -1.0f, 79.3525f, 49.7825f}},
        {"cx NaN", {76.545f, 76.535f, nan, 49.7825f}},
    };
    for (const auto &[name, k] : intrinsics)
        check_align_refused(std::string("level 2's intrinsics with ") + name,
                            r.align(frame,
                                    {level_0, level_1,
                                     gca::organized_surface{level_2.points, level_2.validity_mask,
                                                            level_2.normals, level_2.intensity, k}},
                                    identity, usable, stream),
                            identity,
                            "level 2 of the surface: the surface's intrinsics are not finite");
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
        check_align_refused(name, r.align(frame, surface, initial, usable, stream), initial,
                            "the initial transform is not finite, or its linear part is no "
                            "rotation");
    const struct
    {
        const char *name;
        gca::registration_parameters parameters;
        const char *expected;
    } parameters[] = {
        {"level 0's d_max 0", with_level(0, 0.0f, 20), "level 0's largest distance"},
        {"level 2's d_max a float step above 0.5 m", with_level(2, std::nextafter(0.5f, 1.0f), 10),
         "level 2's largest distance"},
        {"level 1's d_max NaN", with_level(1, nan, 10), "level 1's largest distance"},
        {"level 0's 0 iterations", with_level(0, 0.05f, 0),
         "level 0's most iterations 0 are not within 1 to 100"},
        {"level 0's 101 iterations", with_level(0, 0.05f, 101),
         "level 0's most iterations 101 are not within 1 to 100"},
        {"level 2's 101 iterations", with_level(2, 0.20f, 101),
         "level 2's most iterations 101 are not within 0 to 100"},
        {"Huber's threshold 0", with_weights(0.0f, 0.968f), "Huber's threshold"},
        {"Huber's threshold a float step above the smallest d_max",
         with_weights(std::nextafter(0.05f, 1.0f), 0.968f), "Huber's threshold"},
        {"Huber's threshold above level 1's d_max, the smallest, below level 0's",
         {{{{0.10f, 20}, {0.05f, 10}, {0.20f, 10}}}, 0.06f, 0.968f},
         "Huber's threshold"},
        {"λ 0", with_weights(0.005f, 0.0f), "the point-to-plane term's weight 0"},
        {"λ a float step above 1", with_weights(0.005f, std::nextafter(1.0f, 2.0f)),
         "the point-to-plane term's weight"},
        {"λ NaN", with_weights(0.005f, nan), "the point-to-plane term's weight nan"},
    };
    for (const auto &p : parameters)
        check_align_refused(p.name, r.align(frame, surface, identity, p.parameters, stream),
                            identity, p.expected);
    // The largest values are taken: d_max 0.5 m and 100 iterations at every level, δ_G = d_max,
    // λ = 1.
    const gca::registration_result largest = r.align(
        frame, surface, identity, {{{{0.5f, 100}, {0.5f, 100}, {0.5f, 100}}}, 0.5f, 1.0f}, stream);
    print_result("the largest parameters", largest);
    check(largest.converged(), "the largest parameters are taken");

    // cudaMalloc's refusal of 2^52 bytes, 4 PiB on a board of 7.4 GiB [S: docs/HARDWARE.md],
    // leaves its error as CUDA's last error.
    void *never_granted = nullptr;
    check(cudaMalloc(&never_granted, std::size_t{1} << 52) != cudaSuccess,
          "cudaMalloc refuses 2^52 bytes");
    const gca::registration_result failed = r.align(frame, surface, identity, usable, stream);
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
    check_textured_plane(s);
    check_large_turn(s);
    check_large_shift(s);
    check_few_correspondences(s);
    check_patches(s);
    check_ends(s);
    check_color_against_geometry(s);
    check_color_patch(s);
    check_refusals(s);

    check(s.pool.refused_while_closed() == 0, "the closed pool was asked for nothing");
    gca_test::check_no_cuda_error("the end");
    cudaStreamDestroy(s.stream);
    return gca_test::check_summary("test_registration");
}
