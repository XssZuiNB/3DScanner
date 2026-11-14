// SPDX-License-Identifier: 0BSD
/// @file synthetic_source.hpp
/// The tests' synthetic source: a frame source (camera/frame_source.hpp) that renders a known
/// scene by the one pinhole convention (types/camera_param.hpp) and tells a test what it
/// rendered. The scene is a textured support plane seen by the camera at a known pose. This is
/// the one renderer of the tests' frames: a test whose scene it lacks adds the scene here as one
/// more option. A test builds the source to deliver a number of frames and then to report the
/// end, a timeout or a failure, or to refuse its opening, each with the message it is given.
/// No camera and no OrbbecSDK; host only.

#pragma once

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "types/frame_description.hpp"

#include <array>
#include <cstdint>
#include <string>

#include <Eigen/Geometry>

namespace gca_test
{
/// The support plane: the world's plane z = 0, the rectangle |x| <= half_size_x_m,
/// |y| <= half_size_y_m around the world's origin, textured by plane_color() and seen from
/// either side.
struct synthetic_plane
{
    double half_size_x_m; ///< half the rectangle's extent along the world's x axis, in m, > 0
    double half_size_y_m; ///< half the rectangle's extent along the world's y axis, in m, > 0
};

/// The scene the synthetic source renders. The world frame is the scene's own, in m: x and y in
/// the support plane, z up from it.
struct synthetic_scene
{
    synthetic_plane plane; ///< the support plane
    /// The camera's pose: maps a point from the camera frame (x right, y down, z forward, in m)
    /// into the world frame.
    Eigen::Isometry3d camera_to_world;
};

/// Returns the support plane's texture at a point of it: three sine waves, one per channel, in
/// three directions of the plane, so that the color changes along every direction of it and has
/// a gradient everywhere [?: a choice].
/// @param x_m [in] the point's x in the world frame, in m.
/// @param y_m [in] the point's y in the world frame, in m.
/// @return red, green and blue, unrounded, each within 28 to 228 of the 8-bit range [S: the
///         waves' mean 128 and amplitude 100, synthetic_source.cpp].
std::array<double, 3> plane_color(double x_m, double y_m);

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
    synthetic_scene scene;         ///< what every frame shows
    std::uint64_t frame_count = 0; ///< the frames delivered after each opening, then the ending
    synthetic_ending ending = synthetic_ending::end; ///< what next() reports after them
    std::string ending_message;                      ///< the message next() reports with it
    std::string open_refusal; ///< when not empty, open() refuses with this reason
};

/// A frame source that renders the scene of its options.
class synthetic_source final : public gca::frame_source
{
public:
    /// Builds the source, closed.
    /// @param options [in] the frames, the scene and the ending; taken over.
    explicit synthetic_source(synthetic_source_options options);

    /// Opens the source; its next frame has index 0.
    /// @param size [in] the size asked for, in px.
    /// @return opened; or refused with open_refusal when the options give one, with both sizes
    ///         when size is not the description's, and when the source is open already.
    gca::open_result open(const gca::frame_size &size) noexcept override;

    /// Renders the next frame: frame i of an opening has index i and timestamp i * 33333 µs
    /// (30 fps [?: a choice, the camera's rate]); every frame shows the scene from the same pose.
    /// Each pixel shows the plane where its ray meets it: depth is the hit's z in the camera
    /// frame, rounded to the nearest depth value; color is plane_color() there, rounded. A pixel
    /// whose ray misses the plane or meets it behind the camera, or whose depth rounds below 1
    /// or above the largest 16-bit value, 65535 [S], has depth 0 and color black. Frames come
    /// as fast as they are asked for.
    /// @param out [out] the frame, filled when the status is frame, else left as it was.
    /// @return frame for the first frame_count calls after an opening, then the ending with its
    ///         message at every call; a failure when the source is not open.
    gca::next_result next(gca::frame &out) noexcept override;

    /// Closes the source.
    void close() noexcept override;

    /// Describes the source.
    /// @return the kind of source, no firmware and no SDK version; the same before, during and
    ///         after an opening.
    gca::source_description describe() const noexcept override;

    /// Tells a test what the source renders.
    /// @return the scene: the plane and the camera's pose, valid as long as the source.
    const synthetic_scene &scene() const noexcept;

private:
    synthetic_source_options m_options;
    bool m_open = false;
    std::uint64_t m_next_index = 0; ///< the index of the next frame of this opening
};
} // namespace gca_test
