// SPDX-License-Identifier: 0BSD
/// @file test_frame_source.cpp
/// The frame source seam, driven by the tests' synthetic source:
/// - every pixel of a synthetic frame holds the depth and the color that the known plane, its
///   texture and the camera's pose give, computed here by the pinhole formula on a path of its
///   own (the plane taken into the camera frame), apart from the renderer's code: with a depth
///   unit that holds the whole plane, with one so fine that part of it lies beyond the largest
///   depth value, and with the camera looking away from the plane, which then lies behind it;
/// - a frame's size, intrinsics and depth unit are those the source was built with, and its
///   indices and timestamps increase strictly; an opening after a close starts again at index 0
///   and renders the same frame;
/// - the end, a timeout and a failure each come back as a value with its message, again at the
///   next call, and leave the caller's frame as it was, every pixel of it;
/// - a refused opening comes back as a value with its reason: a source built to refuse, a size
///   it does not render, a source open already; next() on a source that is not open is a failure.
/// No exception can leave the seam: its functions are noexcept, so one would end this program.

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "check.hpp"
#include "synthetic_source.hpp"
#include "types/frame_description.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_max_error;
using gca_test::check_refusal;
using gca_test::check_within;
using gca_test::max_error;

/// The frames' size: small, so that the test checks every pixel at once [?: a choice].
constexpr gca::frame_size test_size = {160, 120};

/// The intrinsics: fx and fy differ, and the principal point lies off the picture's centre at
/// no whole or half pixel, so that a swap or a shift of either shows [?: a choice].
constexpr gca::intrinsics test_intrinsics = {150.0f, 140.0f, 83.3f, 56.6f};

/// The depth unit that holds the whole plane: 0.1 mm, one of the Gemini 2 L's depth precisions
/// [?: a choice]. On this scene a depth step of half a pixel is 0.5 mm near the picture's centre
/// [S: computed, 2026-11-14], five depth units.
constexpr float fine_depth_unit_m = 0.0001f;

/// The depth unit so fine that the plane's far part has no depth: the largest 16-bit value,
/// 65535 units, is 0.655 m, and the plane's depths run from 0.40 m to 0.80 m [S: computed,
/// 2026-11-14].
constexpr float finest_depth_unit_m = 0.00001f;

/// The tolerances of a pixel's depth, in depth units, and of its color, in 8-bit levels: half a
/// step from rounding to the nearest value [S], and 1e-6 for a value that lies within the two
/// paths' double rounding of half a step, about 1e-11 of a step at 65535 depth units [S: double's
/// relative step 2.2e-16]. The largest errors over the three pictures are 0.49990 units and
/// 0.49999 levels [M 2026-11-14]; rounding down instead gives errors up to one step.
constexpr double depth_tolerance_units = 0.5 + 1e-6;
constexpr double color_tolerance_levels = 0.5 + 1e-6;

/// The support plane: 0.6 m x 0.5 m [?: a choice], of which the picture shows part; the upper
/// rows look past its far edge [S: computed, 2026-11-14].
constexpr gca_test::synthetic_plane test_plane = {0.30, 0.25};

/// Returns a camera's pose looking from a position toward a target, its x axis level, then
/// rolled about its viewing axis.
/// @param position [in] the camera's position in the world frame, in m.
/// @param target [in] the point it looks at, in the world frame, in m; not straight above or
///        below the position.
/// @param roll_deg [in] the roll about the viewing axis, in degrees.
/// @return the pose, camera frame to world frame.
Eigen::Isometry3d look_at(const Eigen::Vector3d &position, const Eigen::Vector3d &target,
                          double roll_deg)
{
    // The camera's axes in the world: z toward the target, x level and to the right, y = z x x,
    // which points down when z points down.
    const Eigen::Vector3d z = (target - position).normalized();
    const Eigen::Vector3d x = z.cross(Eigen::Vector3d::UnitZ()).normalized();
    const Eigen::Vector3d y = z.cross(x);
    Eigen::Matrix3d level;
    level.col(0) = x;
    level.col(1) = y;
    level.col(2) = z;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = level * Eigen::AngleAxisd(roll_deg * M_PI / 180.0, Eigen::Vector3d::UnitZ());
    pose.translation() = position;
    return pose;
}

/// Returns the camera's pose that sees the plane: 0.40 m above it, looking down at about 40
/// degrees from (0.04, -0.45) toward the point (0.02, 0.03) of the plane, rolled by 12 degrees,
/// so that the depth changes along both image axes [?: a choice].
/// @return the pose, camera frame to world frame.
Eigen::Isometry3d seeing_pose()
{
    return look_at({0.04, -0.45, 0.40}, {0.02, 0.03, 0.0}, 12.0);
}

/// Returns a camera's pose that looks away from the plane: 0.40 m above it, looking up, so that
/// the plane lies behind the camera along most pixels' rays [?: a choice].
/// @return the pose, camera frame to world frame.
Eigen::Isometry3d looking_away_pose()
{
    return look_at({0.04, -0.05, 0.40}, {0.06, 0.02, 1.40}, 12.0);
}

/// Builds the options of a source of the test scene.
/// @param depth_unit_m [in] the frames' depth unit, in m per value, > 0.
/// @param camera_to_world [in] the camera's pose, camera frame to world frame.
/// @param frame_count [in] the frames delivered before the ending.
/// @param ending [in] what next() reports after them.
/// @param ending_message [in] its message.
/// @return the options.
gca_test::synthetic_source_options make_options(float depth_unit_m,
                                                const Eigen::Isometry3d &camera_to_world,
                                                std::uint64_t frame_count,
                                                gca_test::synthetic_ending ending,
                                                const std::string &ending_message)
{
    gca_test::synthetic_source_options options;
    options.description = {test_size, test_intrinsics, depth_unit_m};
    options.scene = {test_plane, camera_to_world};
    options.frame_count = frame_count;
    options.ending = ending;
    options.ending_message = ending_message;
    return options;
}

/// Builds the options of a source of the test scene from the seeing pose, at the fine depth unit.
/// @param frame_count [in] the frames delivered before the ending.
/// @param ending [in] what next() reports after them.
/// @param ending_message [in] its message.
/// @return the options.
gca_test::synthetic_source_options make_options(std::uint64_t frame_count,
                                                gca_test::synthetic_ending ending,
                                                const std::string &ending_message)
{
    return make_options(fine_depth_unit_m, seeing_pose(), frame_count, ending, ending_message);
}

/// A frame's index, timestamp and pixels, copied to compare the frame with later; the frame
/// itself cannot be copied.
struct frame_copy
{
    std::uint64_t index;              ///< the frame's index
    std::uint64_t timestamp_us;       ///< its timestamp, in µs
    std::vector<std::uint8_t> color;  ///< its color picture
    std::vector<std::uint16_t> depth; ///< its depth picture
};

/// Copies a frame's index, timestamp and pixels.
/// @param frame [in] the frame.
/// @return the copy.
frame_copy copy_of(const gca::frame &frame)
{
    return {frame.index(), frame.timestamp_us(),
            std::vector<std::uint8_t>(frame.color().begin(), frame.color().end()),
            std::vector<std::uint16_t>(frame.depth().begin(), frame.depth().end())};
}

/// Returns whether a frame holds what a copy holds.
/// @param frame [in] the frame.
/// @param copy [in] the copy.
/// @return whether the index, the timestamp and every pixel of color and depth are equal.
bool holds(const gca::frame &frame, const frame_copy &copy)
{
    return frame.index() == copy.index && frame.timestamp_us() == copy.timestamp_us &&
           std::ranges::equal(frame.color(), copy.color) &&
           std::ranges::equal(frame.depth(), copy.depth);
}

/// What a pixel should hold.
struct expected_pixel
{
    bool seen = false;             ///< whether it has depth
    double depth_m = 0.0;          ///< its depth, the camera-frame z of what it sees, in m
    std::array<double, 3> color{}; ///< its color, unrounded
};

/// The pixels of a picture by what they see.
struct pixel_counts
{
    int seen = 0;         ///< pixels that see the plane within the depth range
    int off_plane = 0;    ///< pixels whose ray, forward or backward, misses the plane's rectangle
    int behind = 0;       ///< pixels whose ray meets the plane behind the camera, or nearer than
                          ///< half a depth unit
    int beyond_range = 0; ///< pixels that see the plane farther than the largest depth value
};

/// Computes what a pixel should hold, by the one pinhole convention, on a path apart from the
/// renderer's: the plane z = 0 is taken into the camera frame, n . p = -t_z with n the world's
/// z axis seen from the camera, and the ray z * ((u - cx) / fx, (v - cy) / fy, 1) meets it at
/// z = -t_z / (n . ray).
/// @param scene [in] the plane and the camera's pose.
/// @param description [in] the frame's intrinsics and depth unit.
/// @param u [in] the pixel's column, in px.
/// @param v [in] the pixel's row, in px.
/// @param counts [in,out] the pixel is counted by what it sees.
/// @return its depth and color, or not seen.
expected_pixel expect_pixel(const gca_test::synthetic_scene &scene,
                            const gca::frame_description &description, int u, int v,
                            pixel_counts &counts)
{
    const gca::intrinsics &k = description.intrinsics;
    const Eigen::Vector3d ray((u - double{k.cx}) / k.fx, (v - double{k.cy}) / k.fy, 1.0);
    const Eigen::Matrix3d world_from_camera = scene.camera_to_world.linear();
    const Eigen::Vector3d plane_normal_camera = world_from_camera.transpose().col(2);
    const double z = -scene.camera_to_world.translation().z() / plane_normal_camera.dot(ray);
    const Eigen::Vector3d hit_world = scene.camera_to_world * Eigen::Vector3d(z * ray);
    const double depth_value = std::round(z / description.depth_unit_m);
    expected_pixel pixel;
    if (!(std::abs(hit_world.x()) <= scene.plane.half_size_x_m &&
          std::abs(hit_world.y()) <= scene.plane.half_size_y_m))
        counts.off_plane++;
    else if (depth_value < 1.0)
        counts.behind++;
    else if (depth_value > std::numeric_limits<std::uint16_t>::max())
        counts.beyond_range++;
    else
    {
        counts.seen++;
        pixel.seen = true;
        pixel.depth_m = z;
        pixel.color = gca_test::plane_color(hit_world.x(), hit_world.y());
    }
    return pixel;
}

/// Renders one frame of the test scene and checks every pixel: a pixel that sees the plane holds
/// its depth and its color within their tolerances, and a pixel that does not holds depth 0 and
/// black.
/// @param name [in] the case, printed in each line.
/// @param depth_unit_m [in] the frames' depth unit, in m per value.
/// @param camera_to_world [in] the camera's pose, camera frame to world frame.
/// @return the pixels by what they should see; all zero when the source gave no frame.
pixel_counts check_picture(const std::string &name, float depth_unit_m,
                           const Eigen::Isometry3d &camera_to_world)
{
    gca_test::synthetic_source source(
        make_options(depth_unit_m, camera_to_world, 1, gca_test::synthetic_ending::end, "end"));
    check(source.open(test_size).opened, (name + ": opened").c_str());
    gca::frame frame;
    pixel_counts counts;
    if (source.next(frame).status != gca::next_status::frame)
    {
        check(false, (name + ": a frame").c_str());
        return counts;
    }
    const gca::frame_description &description = frame.description();
    max_error depth_error_units;
    max_error color_error_levels;
    int wrongly_shown = 0;
    int wrongly_unseen = 0;
    for (int v = 0; v < static_cast<int>(test_size.height); v++)
    {
        for (int u = 0; u < static_cast<int>(test_size.width); u++)
        {
            const expected_pixel expected = expect_pixel(source.scene(), description, u, v, counts);
            const std::size_t pixel = static_cast<std::size_t>(v) * test_size.width + u;
            const std::uint16_t depth = frame.depth()[pixel];
            const std::uint8_t *color = &frame.color()[3 * pixel];
            if (!expected.seen)
            {
                if (depth != 0 || color[0] != 0 || color[1] != 0 || color[2] != 0)
                    wrongly_shown++;
                continue;
            }
            if (depth == 0)
            {
                wrongly_unseen++;
                continue;
            }
            depth_error_units.add(std::abs(depth - expected.depth_m / depth_unit_m));
            for (int c = 0; c < 3; c++)
                color_error_levels.add(std::abs(color[c] - expected.color[c]));
        }
    }
    std::printf("%s: %d pixels see the plane, %d miss it, %d have it behind, %d see it beyond "
                "the depth range\n",
                name.c_str(), counts.seen, counts.off_plane, counts.behind, counts.beyond_range);
    check_within((name + ": pixels with depth or color that should have neither").c_str(),
                 wrongly_shown, 0);
    check_within((name + ": pixels without depth that should have it").c_str(), wrongly_unseen, 0);
    check_max_error(name + ": depth error, in depth units", depth_error_units,
                    depth_tolerance_units);
    check_max_error(name + ": color error, in levels", color_error_levels, color_tolerance_levels);
    return counts;
}

/// Checks every pixel of three pictures, and that each shows the kinds of pixels it is meant to:
/// the seeing pose at the fine unit sees the plane and misses it; at the finest unit part of the
/// plane lies beyond the depth range; the pose looking away has the plane behind it.
void check_pictures()
{
    const pixel_counts fine = check_picture("0.1 mm unit", fine_depth_unit_m, seeing_pose());
    check(fine.seen > 0 && fine.off_plane > 0 && fine.behind == 0 && fine.beyond_range == 0,
          "0.1 mm unit: pixels on and off the plane, none behind or beyond the depth range");
    const pixel_counts finest = check_picture("0.01 mm unit", finest_depth_unit_m, seeing_pose());
    check(finest.seen > 0 && finest.beyond_range > 0,
          "0.01 mm unit: pixels on the plane and beyond the depth range");
    const pixel_counts away = check_picture("looking away", fine_depth_unit_m, looking_away_pose());
    check(away.behind > 0 && away.seen == 0, "looking away: the plane behind, nothing seen");
}

/// Checks that each frame has the size, intrinsics and depth unit the source was built with,
/// that indices and timestamps increase strictly, and that the synthetic source's frame i of an
/// opening has index i.
void check_description_and_order()
{
    constexpr int frames = 4;
    gca_test::synthetic_source source(make_options(frames, gca_test::synthetic_ending::end, "end"));
    check(source.open(test_size).opened, "description: opened");
    gca::frame frame;
    std::uint64_t last_index = 0;
    std::uint64_t last_timestamp_us = 0;
    for (int i = 0; i < frames; i++)
    {
        check(source.next(frame).status == gca::next_status::frame, "description: a frame");
        const gca::frame_description &d = frame.description();
        check(d.size == test_size, "description: the size the source was built with");
        check(d.intrinsics.fx == test_intrinsics.fx && d.intrinsics.fy == test_intrinsics.fy &&
                  d.intrinsics.cx == test_intrinsics.cx && d.intrinsics.cy == test_intrinsics.cy,
              "description: the intrinsics the source was built with");
        check(d.depth_unit_m == fine_depth_unit_m,
              "description: the depth unit the source was built with");
        check(frame.color().size() == 3u * test_size.width * test_size.height &&
                  frame.depth().size() == std::size_t{test_size.width} * test_size.height,
              "description: buffers of the frame's size");
        std::printf("frame %d: index %llu, timestamp %llu us\n", i,
                    static_cast<unsigned long long>(frame.index()),
                    static_cast<unsigned long long>(frame.timestamp_us()));
        check(frame.index() == static_cast<std::uint64_t>(i),
              "order: the synthetic source's frame i of an opening has index i");
        check(i == 0 || frame.index() > last_index, "order: indices increase strictly");
        check(i == 0 || frame.timestamp_us() > last_timestamp_us,
              "order: timestamps increase strictly");
        last_index = frame.index();
        last_timestamp_us = frame.timestamp_us();
    }
}

/// Checks that an opening after a close starts again: its first frame has index 0 and is the
/// first opening's first frame, pixel for pixel.
void check_reopening()
{
    gca_test::synthetic_source source(make_options(3, gca_test::synthetic_ending::end, "end"));
    gca::frame frame;
    check(source.open(test_size).opened, "reopening: opened");
    check(source.next(frame).status == gca::next_status::frame, "reopening: a first frame");
    const frame_copy first = copy_of(frame);
    check(source.next(frame).status == gca::next_status::frame, "reopening: a second frame");
    source.close();
    check(source.open(test_size).opened, "reopening: opened again after the close");
    check(source.next(frame).status == gca::next_status::frame,
          "reopening: a frame of the new opening");
    std::printf("reopening: the new opening's first frame has index %llu\n",
                static_cast<unsigned long long>(frame.index()));
    check(holds(frame, first),
          "reopening: the new opening's first frame is the first opening's, index 0");
}

/// Checks that a source built to end in a way reports it as a value with its message, at the
/// call after its frames and at the next call, and leaves the caller's frame as it was: index,
/// timestamp and every pixel.
/// @param name [in] the case, printed in each line.
/// @param ending [in] how the source ends.
/// @param status [in] the status that ending gives.
/// @param message [in] the message the source is built with.
void check_ending(const std::string &name, gca_test::synthetic_ending ending,
                  gca::next_status status, const std::string &message)
{
    constexpr int frames = 2;
    gca_test::synthetic_source source(make_options(frames, ending, message));
    check(source.open(test_size).opened, (name + ": opened").c_str());
    gca::frame frame;
    for (int i = 0; i < frames; i++)
        check(source.next(frame).status == gca::next_status::frame, (name + ": a frame").c_str());
    const frame_copy before = copy_of(frame);
    for (int call = 0; call < 2; call++)
    {
        const gca::next_result result = source.next(frame);
        std::printf("%s, call %d after the frames: %s\n", name.c_str(), call + 1,
                    result.message.c_str());
        check(result.status == status && result.message == message,
              (name + ": reported with its message").c_str());
    }
    check(holds(frame, before), (name + ": the frame left as it was, every pixel").c_str());
}

/// Checks that a refused opening comes back as a value with its reason, and that next() on a
/// source that is not open is a failure.
void check_refusals()
{
    const std::string held = "the camera is held by another program";
    gca_test::synthetic_source_options refusing =
        make_options(1, gca_test::synthetic_ending::end, "end");
    refusing.open_refusal = held;
    gca_test::synthetic_source refuser(refusing);
    const gca::open_result refused = refuser.open(test_size);
    check_refusal("a source built to refuse", !refused.opened, refused.refusal,
                  refused.refusal == held);
    gca::frame frame;
    const gca::next_result unopened = refuser.next(frame);
    check_refusal("next() on a source that refused to open",
                  unopened.status == gca::next_status::failure, unopened.message,
                  unopened.message == "the synthetic source is not open");

    gca_test::synthetic_source source(make_options(1, gca_test::synthetic_ending::end, "end"));
    const gca::open_result wrong_size = source.open({320, 240});
    check_refusal("a size the source does not render", !wrong_size.opened, wrong_size.refusal,
                  wrong_size.refusal == "the synthetic source renders 160x120 px, not 320x240 px");
    check(source.open(test_size).opened, "refusals: the source's own size opens");
    const gca::open_result again = source.open(test_size);
    check_refusal("a source open already", !again.opened, again.refusal,
                  again.refusal == "the synthetic source is open already");
    check(source.next(frame).status == gca::next_status::frame,
          "refusals: the source refused a second opening and stays open");
    source.close();
    const gca::next_result closed = source.next(frame);
    check_refusal("next() on a closed source", closed.status == gca::next_status::failure,
                  closed.message, closed.message == "the synthetic source is not open");
}

/// Checks that the source describes itself, the same before and during an opening.
void check_description_of_source()
{
    gca_test::synthetic_source source(make_options(1, gca_test::synthetic_ending::end, "end"));
    const gca::source_description before = source.describe();
    std::printf("describe(): %s, firmware '%s', SDK '%s'\n", before.camera.c_str(),
                before.firmware_version.c_str(), before.sdk_version.c_str());
    check(!before.camera.empty() && before.firmware_version.empty() && before.sdk_version.empty(),
          "describe(): the kind of source, no versions");
    check(source.open(test_size).opened, "describe(): opened");
    const gca::source_description open = source.describe();
    check(open.camera == before.camera && open.firmware_version.empty() && open.sdk_version.empty(),
          "describe(): the same during an opening");
}
} // namespace

int main()
{
    check_pictures();
    check_description_and_order();
    check_reopening();
    check_ending("end", gca_test::synthetic_ending::end, gca::next_status::end,
                 "the synthetic sequence has ended");
    check_ending("timeout", gca_test::synthetic_ending::timeout, gca::next_status::timeout,
                 "no frame from the camera for 5 s");
    check_ending("failure", gca_test::synthetic_ending::failure, gca::next_status::failure,
                 "the camera left the USB bus");
    check_refusals();
    check_description_of_source();
    return gca_test::check_summary("test_frame_source");
}
