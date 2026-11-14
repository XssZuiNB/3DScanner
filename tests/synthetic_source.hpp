// SPDX-License-Identifier: 0BSD
/// @file synthetic_source.hpp
/// The tests' synthetic source: a frame source (camera/frame_source.hpp) that renders a known
/// scene by the one pinhole convention (types/camera_param.hpp) and tells a test what it
/// rendered. The scene is a textured support plane, a textured box of known size standing on it
/// or none, and a known trajectory: frame i of an opening is seen from the trajectory's pose i.
/// The source tells a test the trajectory and the known surfaces, textured rectangles in the
/// world frame, which are exactly what it renders. It also builds the trajectories its scenes are
/// seen along: a camera's upright pose looking at a point, and a walk once around a point, each
/// refused as a value where the view is not defined. This is the
/// one renderer of the tests' frames: a test whose scene it lacks adds the scene here as one more
/// option. A test builds the source to deliver its trajectory's frames and then to report the
/// end, a timeout or a failure, or to refuse its opening, each with the message it is given.
/// No camera and no OrbbecSDK; host only.

#pragma once

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "types/frame_description.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace gca_test
{
/// The support plane: the world's plane z = 0, the rectangle |x| <= half_size_x_m,
/// |y| <= half_size_y_m around the world's origin, textured by synthetic_texture::plane and seen
/// from either side.
struct synthetic_plane
{
    double half_size_x_m; ///< half the rectangle's extent along the world's x axis, in m, > 0
    double half_size_y_m; ///< half the rectangle's extent along the world's y axis, in m, > 0
};

/// A box standing on the support plane, textured by synthetic_texture::box. Its own axes are the
/// world's x and y turned by yaw_deg about the world's z axis; its edges lie along them and along
/// the world's z. Its bottom face lies in the plane and is not rendered: the plane hides it.
struct synthetic_box
{
    double centre_x_m; ///< its footprint's centre in the plane: x in the world frame, in m
    double centre_y_m; ///< its footprint's centre in the plane: y in the world frame, in m
    double yaw_deg;    ///< the turn of its axes about the world's z axis, from x toward y, in deg
    double size_x_m;   ///< its edge length along its own x axis, in m, > 0
    double size_y_m;   ///< its edge length along its own y axis, in m, > 0
    double height_m;   ///< its edge length along the world's z axis, up from the plane, in m, > 0
};

/// The scene the synthetic source renders. The world frame is the scene's own, in m: x and y in
/// the support plane, z up from it.
struct synthetic_scene
{
    synthetic_plane plane;            ///< the support plane
    std::optional<synthetic_box> box; ///< the box standing on the plane, or none
    /// The trajectory: pose i maps a point from the camera frame of frame i of an opening (x
    /// right, y down, z forward, in m) into the world frame. The source delivers one frame for
    /// each pose.
    std::vector<Eigen::Isometry3d> trajectory;
};

/// The textures of the scene's surfaces.
enum class synthetic_texture
{
    /// The support plane's: three sine waves in the plane, one per channel, along x, along y and
    /// diagonally, each with its own wavelength, so that the color changes along every direction
    /// of the plane and has a gradient everywhere [?: a choice].
    plane,
    /// The box's, painted through the box like a solid: three sine waves in space, one per
    /// channel, along directions that each rise or fall by 35 to 50 degrees, with wavelengths and
    /// phases other than the plane's [?: a choice; the tables in synthetic_source.cpp]. On a face
    /// of any turn every channel changes along the face, by at least 0.57 of its wave's slope
    /// [S: computed, sin 35 degrees on a side, cos 50 degrees on the top].
    box,
};

/// Returns a texture's color at a point of a surface it covers.
/// @param texture [in] the texture.
/// @param point_m [in] the point, in the world frame, in m.
/// @return red, green and blue, unrounded, each within 28 to 228 of the 8-bit range [S: the
///         waves' mean 128 and amplitude 100, synthetic_source.cpp].
std::array<double, 3> texture_color(synthetic_texture texture, const Eigen::Vector3d &point_m);

/// A known surface of the scene: a textured rectangle, seen from either side.
struct synthetic_surface
{
    Eigen::Vector3d centre_m; ///< its centre, in the world frame, in m
    Eigen::Vector3d axis_u;   ///< its first axis, a unit vector in the world frame
    /// Its second axis, a unit vector in the world frame, perpendicular to axis_u. axis_u x axis_v
    /// is the surface's normal: up for the plane, out of the box for a face of the box.
    Eigen::Vector3d axis_v;
    double half_u_m;           ///< half its extent along axis_u, in m, > 0
    double half_v_m;           ///< half its extent along axis_v, in m, > 0
    synthetic_texture texture; ///< its texture
};

/// Returns a camera's pose looking from a position toward a target, upright, then rolled about
/// its viewing axis. Before the roll its z axis points to the target, its x axis is level and to
/// the right of the view (z x up), and its y axis, z x x, points down in the world.
/// @param position [in] the camera's position in the world frame, in m.
/// @param target [in] the point it looks at, in the world frame, in m.
/// @param roll_deg [in] the roll about the viewing axis, in degrees, positive from the camera's x
///        toward its y; within +-90 degrees its y axis still points down in the world and its x
///        axis to the right of the view.
/// @return the pose, camera frame to world frame. Nothing when the target lies within 1e-9 m of
///         the position [?: a choice, far below a scene's size], when the view lies within 1e-4
///         rad of straight up or down, where the level x axis is not defined [?: a choice: there
///         the inputs' rounding, 2.2e-16 of them, turns the x axis by up to 2.2e-12 rad
///         (S: computed)], or when an input is not finite.
std::optional<Eigen::Isometry3d> look_at(const Eigen::Vector3d &position,
                                         const Eigen::Vector3d &target, double roll_deg);

/// A walk once around a point, as a hand-held camera goes around an object: pose k of
/// pose_count stands at the azimuth 360 k / pose_count degrees around the centre, from the
/// world's x axis toward its y, and looks toward the centre (look_at). Its distance from the
/// centre, its height, its roll and the point it looks at swing along the walk, each at its own
/// rate, so that no two views differ by a turn about the centre alone [?: a choice]:
/// distance radius_m + radius_swing_m sin 2a, height height_m + height_swing_m cos 3a, roll
/// roll_swing_deg sin 5a, the point looked at the centre moved by target_jitter_m
/// (sin 3a, cos 5a, 0), at the pose's azimuth a. Every field is the test's to set: the tabletop
/// box's walk of today's test and the large box's walk of the scan-level tests (spec, "Testing
/// Decisions") take other values.
struct synthetic_walk
{
    Eigen::Vector3d centre_m; ///< the point walked around, in the world frame, in m
    std::size_t pose_count;   ///< the poses of the walk, >= 0
    /// The camera's mean distance from the centre along the plane, in m, > 0. Above
    /// radius_swing_m + 1.5 target_jitter_m it keeps every camera off the vertical through the
    /// point it looks at: that point lies within sqrt(2) target_jitter_m, less than
    /// 1.5 target_jitter_m, of the centre [S: its offset target_jitter_m (sin 3a, cos 5a, 0)], and
    /// the camera lies at least radius_m - radius_swing_m from the centre. walk_around() refuses
    /// a walk with a view nearer the vertical than look_at() allows.
    double radius_m;
    double radius_swing_m;  ///< the amplitude of that distance's swing, in m, >= 0
    double height_m;        ///< the camera's mean height above the plane z = 0, in m
    double height_swing_m;  ///< the amplitude of the height's swing, in m, >= 0
    double roll_swing_deg;  ///< the amplitude of the roll about the viewing axis, in deg, >= 0
    double target_jitter_m; ///< how far the point looked at moves along x and along y, in m, >= 0
};

/// Builds the trajectory of a walk once around a point.
/// @param walk [in] the walk.
/// @return pose_count poses, camera frame to world frame, in the walk's order; nothing when
///         look_at() refuses one of them.
std::optional<std::vector<Eigen::Isometry3d>> walk_around(const synthetic_walk &walk);

/// What the synthetic source reports once it has delivered its frames.
enum class synthetic_ending
{
    end,     ///< next_status::end, as a sequence at its end
    timeout, ///< next_status::timeout, as a camera that sends nothing
    failure, ///< next_status::failure, as a camera that fails
};

/// How a test builds the synthetic source.
struct synthetic_source_options
{
    /// The size, intrinsics and depth unit of every frame; the only size the source opens for.
    gca::frame_description description;
    synthetic_scene scene; ///< what the frames show, one frame for each pose of its trajectory
    synthetic_ending ending = synthetic_ending::end; ///< what next() reports after the frames
    std::string ending_message;                      ///< the message next() reports with it
    std::string open_refusal; ///< when not empty, open() refuses with this reason
};

/// A frame source that renders the scene of its options along its trajectory.
class synthetic_source final : public gca::frame_source
{
public:
    /// Builds the source, closed, and the scene's known surfaces.
    /// @param options [in] the frames, the scene and the ending; taken over.
    explicit synthetic_source(synthetic_source_options options);

    /// Opens the source; its next frame has index 0.
    /// @param size [in] the size asked for, in px.
    /// @return opened; or refused with open_refusal when the options give one, with both sizes
    ///         when size is not the description's, and when the source is open already.
    gca::open_result open(const gca::frame_size &size) noexcept override;

    /// Renders the next frame: frame i of an opening has index i and timestamp i * 33333 µs
    /// (30 fps [?: a choice, the camera's rate]) and shows the known surfaces from the
    /// trajectory's pose i. Each pixel shows the nearest surface its ray meets in front of the
    /// camera: depth is the hit's z in the camera frame, rounded to the nearest depth value;
    /// color is the surface's texture there, rounded. A pixel whose ray meets no surface in front
    /// of the camera, or whose depth rounds below 1 or above the largest 16-bit value, 65535 [S],
    /// has depth 0 and color black. Frames come as fast as they are asked for.
    /// @param out [out] the frame, filled when the status is frame, else left as it was.
    /// @return frame for the first calls after an opening, one for each pose of the trajectory,
    ///         then the ending with its message at every call; a failure when the source is not
    ///         open.
    gca::next_result next(gca::frame &out) noexcept override;

    /// Closes the source.
    void close() noexcept override;

    /// Describes the source.
    /// @return the kind of source, no firmware and no SDK version; the same before, during and
    ///         after an opening.
    gca::source_description describe() const noexcept override;

    /// Tells a test what the source renders.
    /// @return the scene: the plane, the box and the trajectory, whose pose i is that of frame i
    ///         of an opening; valid as long as the source.
    const synthetic_scene &scene() const noexcept;

    /// Tells a test the known surfaces, the ones every frame is rendered from.
    /// @return the plane first, then, when the scene has a box, its top and its sides on its own
    ///         +x, +y, -x and -y; valid as long as the source.
    const std::vector<synthetic_surface> &surfaces() const noexcept;

private:
    synthetic_source_options m_options;
    std::vector<synthetic_surface> m_surfaces; ///< the scene's known surfaces
    bool m_open = false;
    std::uint64_t m_next_index = 0; ///< the index of the next frame of this opening
};
} // namespace gca_test
