// SPDX-License-Identifier: 0BSD
/// @file test_frame_source.cpp
/// The frame source seam's contract, driven by the tests' synthetic source (what each pixel of a
/// synthetic frame holds is the renderer's, checked in test_synthetic_source.cpp):
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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_refusal;

/// The frames' size: small, as the seam's contract does not depend on it [?: a choice].
constexpr gca::frame_size test_size = {64, 48};

/// The intrinsics: fx and fy differ and the principal point lies at no whole or half pixel, so
/// that a swap or a shift shows in the description's check [?: a choice].
constexpr gca::intrinsics test_intrinsics = {60.0f, 56.0f, 31.7f, 23.4f};

/// The depth unit: 0.1 mm, one of the Gemini 2 L's depth precisions [?: a choice].
constexpr float test_depth_unit_m = 0.0001f;

/// Builds the options of a source of the support plane, 0.6 m x 0.5 m, seen in every frame from
/// 0.40 m above it and 0.40 m off its centre, so that the frames hold depth and color that an
/// ending has to leave as they were [?: a choice].
/// @param frame_count [in] the frames delivered before the ending.
/// @param ending [in] what next() reports after them.
/// @param ending_message [in] its message.
/// @return the options; with no frame and a failed check when look_at() refuses the pose.
gca_test::synthetic_source_options make_options(std::uint64_t frame_count,
                                                gca_test::synthetic_ending ending,
                                                const std::string &ending_message)
{
    const std::optional<Eigen::Isometry3d> pose =
        gca_test::look_at({0.0, -0.40, 0.40}, {0.0, 0.0, 0.0}, 0.0);
    check(pose.has_value(), "the scene: look_at() gives the camera's pose");
    gca_test::synthetic_source_options options;
    options.description = {test_size, test_intrinsics, test_depth_unit_m};
    options.scene = {{0.30, 0.25},
                     std::nullopt,
                     std::vector<Eigen::Isometry3d>(pose ? frame_count : 0,
                                                    pose.value_or(Eigen::Isometry3d::Identity()))};
    options.ending = ending;
    options.ending_message = ending_message;
    return options;
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
        check(d.depth_unit_m == test_depth_unit_m,
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
                  wrong_size.refusal == "the synthetic source renders 64x48 px, not 320x240 px");
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
