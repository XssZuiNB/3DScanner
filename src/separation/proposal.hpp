// SPDX-License-Identifier: 0BSD
/// @file proposal.hpp
/// Separation's proposal (R15): from the full model, its trajectory and gravity's up, the support
/// plane, the dominant plane under the object, and a box that stands upright on it around what
/// stands on the plane where the views of the trajectory meet; on the host in double. Gravity's up
/// comes from the camera's accelerometer (separation/gravity.hpp), the trajectory tells where the
/// views meet. The plane is the one with the most points near the views among the planes whose
/// normal lies near up with the meeting point above them; after its fits it is checked against
/// both conditions again, and its points must lie flat on it. The box is the least rectangle
/// around the footprint of the piece standing nearest to the meeting point, grown by a margin. The
/// result is a value for every call: proposed, no plane, nothing on the plane, the views do not
/// meet, no gravity, or refused, each with its reason. The constants stand here, each
/// [?: initial]. The method, its sources and its measurements: docs/methods/separation.md.

#pragma once

#include "types/camera_pose.hpp"
#include "types/gravity_estimate.hpp"
#include "types/model_point.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gca
{
/// The least eigenvalue of (1/N) Σ (I - z z^T) over the cameras' optical axes z at which the views
/// meet: sin² 5 deg [?: initial; docs/methods/separation.md, "The views' meeting point"].
inline constexpr double separation_least_axis_spread = 0.007596123493895969;

/// The points of the window the plane's hypotheses are scored on: 20 000 [?: initial].
inline constexpr std::size_t separation_sample_points = 20000;

/// The plane's hypotheses, each through three sampled points: 4000 [?: initial; it misses a plane
/// of 12 % of the window's points with the chance 0.1 %, docs/methods/separation.md].
inline constexpr std::uint32_t separation_hypotheses = 4000;

/// The largest angle of the support plane's normal from gravity's up, in degrees: 45
/// [?: initial: far above gravity's error and a support's tilt, far below a wall's 90].
inline constexpr double separation_largest_tilt_deg = 45.0;

/// d_in, the half thickness of the plane's band, in voxels: 3 [?: initial; the fused model's
/// points lie within a 99th percentile of 0.6 voxels of a plane at 1 mm and 1.2 voxels at 4 mm,
/// docs/methods/sparse-tsdf.md, "What the extraction gives"].
inline constexpr double separation_band_voxels = 3.0;

/// The plane's fits to its band's points after the hypotheses: 3 [?: initial].
inline constexpr int separation_plane_fits = 3;

/// The largest root mean square of the last band's points' distances from the fitted plane, in
/// voxels: 1, between the fitted planes' 0.30 to 0.46 voxels in tests/test_separation.cpp, noise of
/// 0.25 voxels and the feet of what stands on them, and the d_in / sqrt 3 = 1.73 voxels of points
/// spread evenly across the band, as where a plane cuts across surfaces at a slant [?: initial;
/// docs/methods/separation.md, "No plane"].
inline constexpr double separation_largest_band_spread_voxels = 1.0;

/// The least points of the window in the plane's band: 10 000, a patch of 100 x 100 voxels at one
/// point per voxel column [?: initial].
inline constexpr std::size_t separation_least_plane_points = 10000;

/// c, the edge of a cell of the footprint, in voxels: 5 [?: initial].
inline constexpr double separation_cell_voxels = 5.0;

/// μ, the box's margin on each side and on top, in voxels: 5 [?: initial].
inline constexpr double separation_margin_voxels = 5.0;

/// The largest farthest depth, in voxels: 10^5, so that the footprint's cells lie within 4·10^4 of
/// the meeting point's foot along each axis [?: a choice; S: computed, 2 D / c].
inline constexpr double separation_largest_depth_voxels = 1e5;

/// The proposal's parameters, both the scan's object class's (scan/object_class.hpp).
struct separation_parameters
{
    /// v, the voxel size the full model was fused at (the object class's full_voxel_size_m), in m,
    /// finite and > 0: it scales the plane's band, the footprint's cell and the margin.
    double voxel_size_m;
    /// D, the farthest depth a frame contributes (the object class's max_depth_m), in m, finite,
    /// > 0 and at most separation_largest_depth_voxels v: the window never reaches farther from the
    /// views' meeting point.
    double max_depth_m;
};

/// The support plane: the points p of the models' frame with normal · p = offset_m.
struct support_plane
{
    /// n, its unit normal in the models' frame, pointing to the side the object stands on.
    Eigen::Vector3d normal;
    double offset_m; ///< d, in m: the plane's signed distance from the models' origin along n

    /// Gives a point's height above the plane.
    /// @param point_m [in] the point, in the models' frame, in m.
    /// @return normal · point_m - offset_m, in m: > 0 on the side the object stands on.
    double height_m(const Eigen::Vector3d &point_m) const noexcept
    {
        return normal.dot(point_m) - offset_m;
    }
};

/// The box: upright on the support plane, its bottom face in the plane.
struct separation_box
{
    /// From the box's frame into the models' frame: its origin the centre of the box's bottom face,
    /// which lies in the plane; its z axis the plane's normal; its x axis along the box's longer
    /// side, of either sign, L_x >= L_y; a rotation and a translation in m.
    Eigen::Isometry3d pose;
    /// L_x, L_y and H, the edge lengths along its x, y and z axes, in m, >= 0; H its height above
    /// the plane.
    Eigen::Vector3d size_m;

    /// Says whether a point lies in the box, its faces included.
    /// @param point_m [in] the point, in the models' frame, in m.
    /// @return whether q = pose^-1 point_m has |q_x| <= L_x / 2, |q_y| <= L_y / 2 and
    ///         0 <= q_z <= H.
    bool holds(const Eigen::Vector3d &point_m) const noexcept
    {
        const Eigen::Vector3d q = pose.inverse() * point_m;
        return std::abs(q.x()) <= 0.5 * size_m.x() && std::abs(q.y()) <= 0.5 * size_m.y() &&
               q.z() >= 0.0 && q.z() <= size_m.z();
    }
};

/// How a proposal ended (docs/methods/separation.md, "The ends"); only proposed gives a plane and
/// a box.
enum class separation_end
{
    proposed, ///< the plane and the box
    /// No plane under the views' meeting point: no hypothesis met both conditions; or the band held
    /// fewer than separation_least_plane_points points before a fit or after the last; or the
    /// fitted plane left the conditions; or the last band's points lie farther from it than
    /// separation_largest_band_spread_voxels v in the root mean square.
    no_plane,
    nothing_on_plane,  ///< a plane, but nothing stands on it within the window
    views_do_not_meet, ///< the tracked frames' optical axes are near parallel
    /// Gravity gives no up (types/gravity_estimate.hpp): not estimated, no accelerometer sample,
    /// too few, their mean 0, their spread too wide, or its estimate refused; the operator sets
    /// plane and box.
    no_gravity,
    refused, ///< the inputs, before anything was computed; the message says why
};

/// A proposal's result, a value for every call.
struct separation_proposal
{
    separation_end end = separation_end::refused; ///< how it ended
    std::string message; ///< why it proposed nothing, in plain English; empty when proposed
    support_plane plane{Eigen::Vector3d::Zero(), 0.0}; ///< the support plane when proposed
    /// The box when proposed.
    separation_box box{Eigen::Isometry3d::Identity(), Eigen::Vector3d::Zero()};
    /// Gravity's up as given, a unit vector in the models' frame; zero when refused or without
    /// gravity.
    Eigen::Vector3d up = Eigen::Vector3d::Zero();
    /// The views' meeting point, in the models' frame, in m; zero when refused, without gravity
    /// or when the views do not meet.
    Eigen::Vector3d meeting_point_m = Eigen::Vector3d::Zero();
    /// W, how far from the meeting point the proposal looked, in m; 0 when refused, without
    /// gravity or when the views do not meet.
    double window_m = 0.0;
    /// The window's points in the plane's band: of the proposed plane, of the plane nothing stands
    /// on, or for no plane of the last plane counted; 0 when no hypothesis was a candidate or
    /// before the plane.
    std::size_t plane_points = 0;
    std::size_t object_points = 0; ///< the points of the piece the box is around; 0 unless proposed

    /// Says whether the proposal has a plane and a box.
    /// @return end == proposed.
    bool proposed() const noexcept
    {
        return end == separation_end::proposed;
    }
};

/// Proposes the separation of a full model (docs/methods/separation.md): checks the inputs;
/// gravity's up, or the end without it; the views' meeting point m, nearest to the tracked frames'
/// optical axes; the window, the model's points within
/// min(the median camera's distance from m, D) of m; the support plane by random sample consensus
/// among the planes whose normal lies within 45 degrees of up with m above them, the most points
/// within d_in winning, then fitted by orthogonal least squares, both conditions checked again and
/// its band's points within an RMS distance of v; the points higher than d_in above it in cells of
/// c on the plane, joined to their eight neighbours; the piece nearest to m's foot; the least
/// rectangle around its footprint, grown by μ, its height the piece's highest point plus μ.
/// Deterministic: the samples come from a generator of a fixed seed.
/// @param full_model [in] the full model's points in the models' frame, in m; their positions
///        finite, else refused; normals and colors not read; host memory, read during the call.
/// @param trajectory [in] the trajectory: the camera's pose at each tracked frame, camera to model,
///        at least one, else refused; each position finite and each quaternion of norm 1 within
///        pose_norm_tolerance (types/camera_pose.hpp, fault_of()), else refused; frames in tracking
///        loss left out; host memory, read during the call.
/// @param gravity [in] gravity's estimate of the scan (separation/gravity.hpp, estimate_gravity(),
///        or the scan's parameters): an outcome of the closed set, else refused; when estimated, up
///        in the models' frame, finite and of length 1 within gravity_up_tolerance (up_is_unit()),
///        else refused; any other outcome ends the proposal without gravity.
/// @param parameters [in] v and D (separation_parameters), else refused; the model's points lie
///        within D of a camera, as a scan fuses them, else the window leaves out what lies farther.
/// @return the proposal (separation_proposal): proposed, with the plane, the box and the evidence;
///         no gravity, with gravity's reason (reason_of()), before the views; no plane, nothing on
///         the plane or the views do not meet, each with the reason and the evidence found before
///         it; or refused, with the reason, when an input breaks one of the conditions above.
/// @note Host only; allocates the window's points in double, the sample and the footprint's
///       cells, and throws std::bad_alloc when an allocation fails. Holds no state: two threads may
///       propose at once.
separation_proposal propose_separation(std::span<const model_point> full_model,
                                       std::span<const camera_pose> trajectory,
                                       const gravity_estimate &gravity,
                                       const separation_parameters &parameters);
} // namespace gca
