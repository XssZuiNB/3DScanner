// SPDX-License-Identifier: 0BSD
/// @file test_orbbec_rules.cpp
/// The Orbbec camera's decisions that need no camera (camera/orbbec_rules.hpp), fed with plain
/// data as OrbbecSDK reports it:
/// - the color stream is chosen by size, rate and the order of formats; the depth stream is one
///   aligned in hardware at the rate, else one of the asked size and rate, never another size;
/// - an opening's report becomes an opening or a refusal with its own reason: no camera, the
///   camera held by another program, another exception, no such stream on a USB 2 or a USB 3
///   link;
/// - a frameset that passes its checks becomes the caller's frame, with the opening's size and
///   intrinsics, its depth unit, index and timestamp, every pixel copied;
/// - a timeout and a failure come back as values and leave the caller's frame as it was;
/// - a frameset of a wrong size, depth format or depth unit is refused with its reason, the first
///   frame and every later one alike; one with an index or timestamp out of order is skipped and
///   counted with its reason; either leaves the frame and the opening's state as they were, so the
///   next frame is taken and the refused or skipped one's index stays a gap;
/// - a jump of OrbbecSDK's timestamps by 2^32 µs either way is taken out, also with a rest of a
///   few ms; the clocks may differ by up to 10 s either way, as a backlog in OrbbecSDK's queue
///   makes them, and no further; a long pause is no jump.
/// The test links no OrbbecSDK (ldd shows none).

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/orbbec_rules.hpp"
#include "check.hpp"
#include "frame_copy.hpp"
#include "types/camera_param.hpp"
#include "types/frame_description.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace
{
using gca_test::check;
using gca_test::check_refusal;
using gca_test::copy_of;
using gca_test::frame_copy;
using gca_test::holds;

/// The opened stream size: small, as the rules do not depend on it [?: a choice].
constexpr gca::frame_size test_size = {8, 6};

/// The intrinsics: fx and fy differ and the principal point lies at no whole or half pixel, so
/// that a swap shows [?: a choice].
constexpr gca::intrinsics test_intrinsics = {7.0f, 6.5f, 3.6f, 2.7f};

/// The jump of OrbbecSDK's timestamps, 2^32 µs [M 2026-11-13, docs/HARDWARE.md].
constexpr std::uint64_t jump_us = std::uint64_t{1} << 32;

/// The time between two frames at 30 fps, in µs [S: 1 s / 30].
constexpr std::uint64_t frame_step_us = 33'333;

/// A frameset with its own pixels, as the camera source hands it to the gate.
struct sample
{
    std::vector<std::uint8_t> color;  ///< its RGB8 pixels
    std::vector<std::uint16_t> depth; ///< its depth values
    gca::orbbec_frameset frameset;    ///< the plain data, pointing at the two above
};

/// Builds a frameset of a size whose pixels depend on a seed, so that two framesets differ in
/// every pixel.
/// @param size [in] the size of both pictures, in px.
/// @param seed [in] the pixels' seed.
/// @param index [in] the color frame's number.
/// @param camera_time_us [in] the timestamp on the camera's clock, in µs.
/// @param host_time_us [in] the host's time, in µs.
/// @return the frameset with 1 mm per depth value and 16-bit depth; its spans point at its own
///         vectors, so it must not be copied, only moved or returned.
sample make_sample(gca::frame_size size, std::uint32_t seed, std::uint64_t index,
                   std::uint64_t camera_time_us, std::uint64_t host_time_us)
{
    sample s;
    const std::size_t pixels = std::size_t{size.width} * size.height;
    s.color.resize(3 * pixels);
    s.depth.resize(pixels);
    for (std::size_t i = 0; i < s.color.size(); i++)
        s.color[i] = static_cast<std::uint8_t>(seed * 31 + i * 7);
    for (std::size_t i = 0; i < s.depth.size(); i++)
        s.depth[i] = static_cast<std::uint16_t>(500 + seed * 13 + i * 3);
    s.frameset.color_size = size;
    s.frameset.depth_size = size;
    s.frameset.depth_16_bit = true;
    s.frameset.depth_scale_mm = 1.0f;
    s.frameset.index = index;
    s.frameset.camera_time_us = camera_time_us;
    s.frameset.host_time_us = host_time_us;
    return s;
}

/// Points a sample's frameset at its own pixels; called after the sample stands in its final
/// place.
/// @param s [in,out] the sample.
void point_at_pixels(sample &s)
{
    s.frameset.color_rgb = s.color;
    s.frameset.depth = s.depth;
}

/// Hands a sample to the gate as a wait that gave it.
/// @param gate [in,out] the gate.
/// @param s [in,out] the sample, pointed at its pixels first.
/// @param out [out] the caller's frame.
/// @return the gate's result; nothing when it skipped the sample.
std::optional<gca::next_result> take_sample(gca::orbbec_frame_gate &gate, sample &s,
                                            gca::frame &out)
{
    point_at_pixels(s);
    gca::orbbec_wait wait;
    wait.status = gca::orbbec_wait_status::frameset;
    wait.frameset = s.frameset;
    return gate.take(wait, out);
}

/// Says whether a result of the gate is a frame.
/// @param result [in] the result; nothing for a skip.
/// @return whether it is a result with the status frame.
bool taken(const std::optional<gca::next_result> &result)
{
    return result.has_value() && result->status == gca::next_status::frame;
}

/// Checks that a frame holds a sample: the opening's size and intrinsics, the sample's depth
/// unit and index, a timestamp, and every pixel.
/// @param what [in] the case, named in the check lines.
/// @param out [in] the frame.
/// @param s [in] the sample.
/// @param timestamp_us [in] the timestamp the frame should carry, in µs.
void check_holds_sample(const std::string &what, const gca::frame &out, const sample &s,
                        std::uint64_t timestamp_us)
{
    const gca::frame_description &d = out.description();
    check(d.size == test_size, (what + ": the opening's size").c_str());
    check(d.intrinsics.fx == test_intrinsics.fx && d.intrinsics.fy == test_intrinsics.fy &&
              d.intrinsics.cx == test_intrinsics.cx && d.intrinsics.cy == test_intrinsics.cy,
          (what + ": the opening's intrinsics").c_str());
    check(d.depth_unit_m == s.frameset.depth_scale_mm * 0.001f,
          (what + ": the depth unit in m, the frameset's in mm times 0.001").c_str());
    check(out.index() == s.frameset.index, (what + ": the frameset's index").c_str());
    std::printf("%s: index %llu, timestamp %llu µs\n", what.c_str(),
                static_cast<unsigned long long>(out.index()),
                static_cast<unsigned long long>(out.timestamp_us()));
    check(out.timestamp_us() == timestamp_us, (what + ": the expected timestamp").c_str());
    check(std::ranges::equal(out.color(), s.color) && std::ranges::equal(out.depth(), s.depth),
          (what + ": every pixel of color and depth").c_str());
}

/// Checks that the gate refuses a sample with the expected reason and leaves the caller's frame
/// as it was.
/// @param what [in] the case, printed and named in the check lines.
/// @param gate [in,out] the gate.
/// @param s [in,out] the sample.
/// @param out [in,out] the caller's frame.
/// @param reason [in] the expected reason.
void check_refused(const std::string &what, gca::orbbec_frame_gate &gate, sample &s,
                   gca::frame &out, const std::string &reason)
{
    const frame_copy before = copy_of(out);
    const std::optional<gca::next_result> result = take_sample(gate, s, out);
    const bool failed = result.has_value() && result->status == gca::next_status::failure;
    check_refusal(what, failed, failed ? result->message : "", failed && result->message == reason);
    check(holds(out, before), (what + ": the frame left as it was, every pixel").c_str());
}

/// Checks that the gate skips a sample with the expected reason, counts it, and leaves the
/// caller's frame as it was.
/// @param what [in] the case, printed and named in the check lines.
/// @param gate [in,out] the gate.
/// @param s [in,out] the sample.
/// @param out [in,out] the caller's frame.
/// @param reason [in] the expected reason.
void check_skipped(const std::string &what, gca::orbbec_frame_gate &gate, sample &s,
                   gca::frame &out, const std::string &reason)
{
    const frame_copy before = copy_of(out);
    const std::uint64_t count_before = gate.skips().count;
    const std::optional<gca::next_result> result = take_sample(gate, s, out);
    const bool skipped = !result.has_value();
    std::printf("%s: %s\n", what.c_str(),
                skipped ? ("skipped, " + gate.skips().last_reason).c_str() : "not skipped");
    check(skipped && gate.skips().count == count_before + 1 && gate.skips().last_reason == reason,
          ("skipped and counted with the expected reason: " + what).c_str());
    check(holds(out, before), (what + ": the frame left as it was, every pixel").c_str());
}

/// Builds an opening's report of a camera that opened.
/// @return one camera on USB3.0, both streams found, no exception.
gca::orbbec_open_report opened_report()
{
    gca::orbbec_open_report report;
    report.camera_count = 1;
    report.connection_type = "USB3.0";
    report.color_stream_offered = true;
    report.depth_stream_offered = true;
    return report;
}

/// Checks one opening's report against the refusal it should give.
/// @param what [in] the case, printed and named in the check line.
/// @param report [in] the report.
/// @param reason [in] the expected reason.
void check_opening_refused(const std::string &what, const gca::orbbec_open_report &report,
                           const std::string &reason)
{
    const std::optional<std::string> refusal = gca::opening_refusal(report, {1280, 800}, 30);
    check_refusal(what, refusal.has_value(), refusal.value_or(""), refusal == reason);
}

/// Checks the choice of the color and the depth stream.
void check_stream_choice()
{
    using gca::orbbec_format;
    const std::vector<gca::orbbec_profile> colors = {
        {640, 400, 30, orbbec_format::mjpg}, {640, 400, 15, orbbec_format::bgr},
        {640, 400, 30, orbbec_format::rgb},  {640, 400, 30, orbbec_format::yuyv},
        {1280, 800, 30, orbbec_format::rgb}, {1280, 800, 30, orbbec_format::bgr}};
    check(gca::choose_color_profile(colors, {640, 400}, 30) == std::optional<std::size_t>{2},
          "color: 640x400 at 30 fps is RGB, before MJPG and YUYV; BGR only at 15 fps");
    check(gca::choose_color_profile(colors, {1280, 800}, 30) == std::optional<std::size_t>{5},
          "color: 1280x800 at 30 fps is BGR, before RGB");
    check(!gca::choose_color_profile(colors, {800, 600}, 30).has_value(), "color: no 800x600");
    check(!gca::choose_color_profile(colors, {1280, 800}, 15).has_value(),
          "color: no 1280x800 at 15 fps");

    const std::vector<gca::orbbec_profile> hardware = {{640, 400, 30, orbbec_format::rle},
                                                       {320, 200, 30, orbbec_format::y14},
                                                       {640, 400, 15, orbbec_format::y16}};
    // The Gemini 2 L on USB 2 [M 2026-11-13, docs/HARDWARE.md]: Y14 and RLE, 1280x800 at 10 fps.
    const std::vector<gca::orbbec_profile> usb_2 = {{640, 400, 30, orbbec_format::rle},
                                                    {640, 400, 30, orbbec_format::y14},
                                                    {1280, 800, 10, orbbec_format::y14}};
    const auto depth = gca::choose_depth_profile(hardware, usb_2, {640, 400}, 30);
    check(depth && depth->hardware && depth->position == 1,
          "depth: aligned in hardware at the rate, of any size, Y14 before RLE");
    const auto software = gca::choose_depth_profile({}, usb_2, {640, 400}, 30);
    check(software && !software->hardware && software->position == 1,
          "depth: none in hardware, then the asked size and rate, Y14 before RLE");
    check(!gca::choose_depth_profile({}, usb_2, {1280, 800}, 30).has_value(),
          "depth: none in hardware and none of 1280x800 at 30 fps: no 640x400 instead");
    const std::vector<gca::orbbec_profile> sixteen = {{1280, 800, 30, orbbec_format::rle},
                                                      {1280, 800, 30, orbbec_format::z16},
                                                      {1280, 800, 30, orbbec_format::y16}};
    const auto y16 = gca::choose_depth_profile({}, sixteen, {1280, 800}, 30);
    check(y16 && !y16->hardware && y16->position == 2, "depth: Y16 before Z16 before RLE");
}

/// Checks that each kind of opening report gives its own refusal, and an opened camera none.
void check_openings()
{
    check(!gca::opening_refusal(opened_report(), {1280, 800}, 30).has_value(),
          "opening: a camera with both streams and no exception opens");

    gca::orbbec_open_report missing;
    check_opening_refused("opening: no camera", missing, "no Orbbec camera is connected");

    gca::orbbec_open_report busy = opened_report();
    busy.error = gca::orbbec_error{gca::orbbec_exception::resource_busy, "the device is busy"};
    check_opening_refused("opening: OrbbecSDK says the resource is busy", busy,
                          "the camera is held by another program (OrbbecSDK: the device is busy)");

    const std::string claim_busy = "uvc_open failed: [Path: 2-1.4-6, Return Code: -6]";
    gca::orbbec_open_report claimed = opened_report();
    claimed.error = gca::orbbec_error{gca::orbbec_exception::io, claim_busy};
    check_opening_refused("opening: libusb refused the interface's claim as busy", claimed,
                          "the camera is held by another program (OrbbecSDK: " + claim_busy + ")");

    const std::string claim_other = "uvc_open failed: [Path: 2-1.4-6, Return Code: -4]";
    gca::orbbec_open_report io_other = opened_report();
    io_other.error = gca::orbbec_error{gca::orbbec_exception::io, claim_other};
    check_opening_refused("opening: another input or output error", io_other,
                          "OrbbecSDK failed while opening the camera: " + claim_other);

    gca::orbbec_open_report other_type = opened_report();
    other_type.error = gca::orbbec_error{gca::orbbec_exception::other, claim_busy};
    check_opening_refused("opening: the busy claim's message with another exception type",
                          other_type, "OrbbecSDK failed while opening the camera: " + claim_busy);

    gca::orbbec_open_report error_first;
    error_first.error = gca::orbbec_error{gca::orbbec_exception::other, "libusb_init failed"};
    check_opening_refused("opening: an exception before any camera was counted", error_first,
                          "OrbbecSDK failed while opening the camera: libusb_init failed");

    gca::orbbec_open_report usb2_color = opened_report();
    usb2_color.connection_type = "USB2.0";
    usb2_color.color_stream_offered = false;
    usb2_color.depth_stream_offered = false;
    check_opening_refused("opening: no color stream on USB 2", usb2_color,
                          "the camera's link is USB2.0, which offers no color stream of "
                          "1280x800 at 30 fps; its USB 3 link did not come up (AGENTS.md §6)");

    gca::orbbec_open_report usb2_depth = opened_report();
    usb2_depth.connection_type = "USB2.1";
    usb2_depth.depth_stream_offered = false;
    check_opening_refused("opening: no depth stream on USB 2", usb2_depth,
                          "the camera's link is USB2.1, which offers no depth stream aligned to "
                          "color of 1280x800 at 30 fps; its USB 3 link did not come up "
                          "(AGENTS.md §6)");

    gca::orbbec_open_report usb3_depth = opened_report();
    usb3_depth.depth_stream_offered = false;
    check_opening_refused("opening: no depth stream on USB 3", usb3_depth,
                          "the camera offers no depth stream aligned to color of 1280x800 at "
                          "30 fps");

    gca::orbbec_open_report unnamed_link = opened_report();
    unnamed_link.connection_type = "USB";
    unnamed_link.color_stream_offered = false;
    check_opening_refused("opening: no color stream on a link of no named speed", unnamed_link,
                          "the camera offers no color stream of 1280x800 at 30 fps");
}

/// Checks that framesets in order become frames, and that a timeout and a failure come back as
/// values that leave the frame as it was and the opening's state too.
void check_frames_timeout_and_failure()
{
    gca::orbbec_frame_gate gate;
    gate.start(test_size, test_intrinsics);
    gca::frame out;
    sample first = make_sample(test_size, 1, 5, 1'000'000, 70'000'000);
    first.frameset.depth_scale_mm = 0.1f;
    check(taken(take_sample(gate, first, out)), "frames: the first");
    check_holds_sample("frames: the first", out, first, 1'000'000);
    sample second =
        make_sample(test_size, 2, 6, 1'000'000 + frame_step_us, 70'000'000 + frame_step_us);
    second.frameset.depth_scale_mm = 0.1f;
    check(taken(take_sample(gate, second, out)), "frames: the second");
    check_holds_sample("frames: the second", out, second, 1'000'000 + frame_step_us);

    const frame_copy before = copy_of(out);
    gca::orbbec_wait timeout;
    timeout.status = gca::orbbec_wait_status::timeout;
    const gca::next_result timed_out =
        gate.take(timeout, out).value_or(gca::next_result{gca::next_status::frame, "skipped"});
    std::printf("timeout: %s\n", timed_out.message.c_str());
    check(timed_out.status == gca::next_status::timeout &&
              timed_out.message == "no frame from the camera within 5000 ms",
          "timeout: reported as a timeout with its message, no frame");
    check(holds(out, before), "timeout: the frame left as it was, every pixel");

    gca::orbbec_wait failure;
    failure.status = gca::orbbec_wait_status::failure;
    failure.message = "OrbbecSDK: the device was disconnected";
    const gca::next_result failed =
        gate.take(failure, out).value_or(gca::next_result{gca::next_status::frame, "skipped"});
    std::printf("failure: %s\n", failed.message.c_str());
    check(failed.status == gca::next_status::failure && failed.message == failure.message,
          "failure: reported as a failure with OrbbecSDK's message");
    check(holds(out, before), "failure: the frame left as it was, every pixel");

    sample third =
        make_sample(test_size, 3, 9, 1'000'000 + 4 * frame_step_us, 70'000'000 + 4 * frame_step_us);
    third.frameset.depth_scale_mm = 0.1f;
    check(taken(take_sample(gate, third, out)), "frames: taken after the timeout and the failure");
    check_holds_sample("frames: after the timeout and the failure, index 9 after 6", out, third,
                       1'000'000 + 4 * frame_step_us);
}

/// Checks each refusal of a frameset's pictures and depth unit, on the first frame of an
/// opening and on a later one, and that the next good frame is taken with its index's gap.
void check_picture_refusals()
{
    struct defect
    {
        std::string name;        ///< the case
        void (*apply)(sample &); ///< makes the sample defective
        std::string reason;      ///< the expected reason
    };
    const std::vector<defect> defects = {
        {"a color picture of another size",
         [](sample &s) {
             s.color.resize(3 * 6 * 6);
             s.frameset.color_size = {6, 6};
         },
         "the color picture is 6x6, the opening's size is 8x6"},
        {"a depth picture of another size",
         [](sample &s) {
             s.depth.resize(6 * 6);
             s.frameset.depth_size = {6, 6};
         },
         "the depth picture is 6x6, the opening's size is 8x6"},
        {"color shorter than its size", [](sample &s) { s.color.pop_back(); },
         "the color picture holds 143 bytes, 8x6 in RGB8 needs 144"},
        {"depth not 16 bit",
         [](sample &s) {
             s.depth.clear();
             s.frameset.depth_16_bit = false;
             s.frameset.depth_format = "RLE";
         },
         "depth arrives as RLE; 16 bit per pixel (Y16 or Z16) is needed"},
        {"depth shorter than its size", [](sample &s) { s.depth.pop_back(); },
         "the depth picture holds 47 values, 8x6 needs 48"},
        {"a depth unit of 0", [](sample &s) { s.frameset.depth_scale_mm = 0.0f; },
         "the depth unit, 0 mm per value, is not a positive number"},
        {"a depth unit that is not a number",
         [](sample &s) { s.frameset.depth_scale_mm = std::numeric_limits<float>::quiet_NaN(); },
         "the depth unit, nan mm per value, is not a positive number"},
    };
    for (const defect &d : defects)
    {
        for (const bool later : {false, true})
        {
            const std::string what =
                std::string(later ? "a later frame: " : "the first frame: ") + d.name;
            gca::orbbec_frame_gate gate;
            gate.start(test_size, test_intrinsics);
            gca::frame out;
            std::uint64_t time_us = 2'000'000;
            if (later)
            {
                sample good = make_sample(test_size, 1, 1, time_us, time_us);
                check(taken(take_sample(gate, good, out)),
                      (what + ": the first frame taken").c_str());
            }
            sample bad =
                make_sample(test_size, 2, 2, time_us + frame_step_us, time_us + frame_step_us);
            d.apply(bad);
            check_refused(what, gate, bad, out, d.reason);
            sample next = make_sample(test_size, 3, 3, time_us + 2 * frame_step_us,
                                      time_us + 2 * frame_step_us);
            check(taken(take_sample(gate, next, out)), (what + ": the next frame taken").c_str());
            check(out.index() == 3, (what + ": the refused frame's index 2 stays a gap").c_str());
        }
    }

    gca::orbbec_frame_gate gate;
    gate.start(test_size, test_intrinsics);
    gca::frame out;
    sample first = make_sample(test_size, 1, 1, 3'000'000, 3'000'000);
    check(taken(take_sample(gate, first, out)), "depth unit: the first frame taken, 1 mm");
    sample changed =
        make_sample(test_size, 2, 2, 3'000'000 + frame_step_us, 3'000'000 + frame_step_us);
    changed.frameset.depth_scale_mm = 0.1f;
    check_refused("a later frame: a depth unit other than the opening's", gate, changed, out,
                  "the depth unit changed within the opening, from 1 to 0.1 mm per value");

    gca::orbbec_frame_gate unstarted;
    sample any = make_sample(test_size, 1, 1, 1'000, 1'000);
    check_refused("a gate not started", unstarted, any, out,
                  "the color picture is 8x6, the opening's size is 0x0");
}

/// Checks that an index not after the last taken one is refused, and that a later one is taken.
void check_indices()
{
    gca::orbbec_frame_gate gate;
    gate.start(test_size, test_intrinsics);
    gca::frame out;
    sample first = make_sample(test_size, 1, 10, 5'000'000, 5'000'000);
    check(taken(take_sample(gate, first, out)), "index: frame 10");
    sample same =
        make_sample(test_size, 2, 10, 5'000'000 + frame_step_us, 5'000'000 + frame_step_us);
    check_skipped("index: the same index again", gate, same, out,
                  "the frame's index 10 is not after the last taken frame's 10");
    sample lower =
        make_sample(test_size, 3, 9, 5'000'000 + 2 * frame_step_us, 5'000'000 + 2 * frame_step_us);
    check_skipped("index: a lower index", gate, lower, out,
                  "the frame's index 9 is not after the last taken frame's 10");
    sample later =
        make_sample(test_size, 4, 13, 5'000'000 + 3 * frame_step_us, 5'000'000 + 3 * frame_step_us);
    check(taken(take_sample(gate, later, out)) && out.index() == 13,
          "index: frame 13 taken after 10, a gap of two");
}

/// One step of a timestamp case: what the camera and the host report, and what should come out.
struct time_step
{
    std::uint64_t camera_us;   ///< the camera's timestamp, in µs
    std::uint64_t host_us;     ///< the host's time, in µs
    std::uint64_t expected_us; ///< the delivered timestamp, in µs; 0 when it is to be skipped
    std::string reason;        ///< the expected reason of a skip
};

/// Runs one timestamp case through a gate, a new index for every step.
/// @param name [in] the case, printed and named in the check lines.
/// @param steps [in] its steps.
void check_time_case(const std::string &name, const std::vector<time_step> &steps)
{
    gca::orbbec_frame_gate gate;
    gate.start(test_size, test_intrinsics);
    gca::frame out;
    std::uint32_t index = 1;
    for (const time_step &step : steps)
    {
        const std::string what = name + ", step " + std::to_string(index);
        sample s = make_sample(test_size, index, index, step.camera_us, step.host_us);
        index++;
        if (step.expected_us == 0)
        {
            check_skipped(what, gate, s, out, step.reason);
            continue;
        }
        check(taken(take_sample(gate, s, out)), (what + ": taken").c_str());
        check_holds_sample(what, out, s, step.expected_us);
    }
}

/// Checks the rule for timestamps: jumps of 2^32 µs either way taken out, also with a rest of a
/// few ms; the tolerance of 10 s either way, at its bounds, behind a backlog and with a lagging
/// camera clock; a long pause no jump; and each skip, after which the next frame is taken
/// against the last taken one.
void check_timestamps()
{
    constexpr std::uint64_t t = 7'000'000'000;  // the camera's clock, in µs [?: a choice]
    constexpr std::uint64_t h = 90'000'000'000; // the host's clock, in µs [?: a choice]
    constexpr std::uint64_t s = frame_step_us;
    check_time_case("a jump up by 2^32 µs", {{t, h, t, ""},
                                             {t + s + jump_us, h + s, t + s, ""},
                                             {t + 2 * s + jump_us, h + 2 * s, t + 2 * s, ""}});
    check_time_case("a jump down by 2^32 µs", {{t, h, t, ""},
                                               {t + s - jump_us, h + s, t + s, ""},
                                               {t + 2 * s - jump_us, h + 2 * s, t + 2 * s, ""}});
    check_time_case("a jump up and back down", {{t, h, t, ""},
                                                {t + s + jump_us, h + s, t + s, ""},
                                                {t + 2 * s, h + 2 * s, t + 2 * s, ""}});
    check_time_case(
        "a jump up across a pause of 10 s",
        {{t, h, t, ""}, {t + 10'000'000 + jump_us, h + 10'000'000, t + 10'000'000, ""}});
    // An hour, more than 2^31 µs: a step that only a rule without the host's clock would take
    // for a jump [S: 3.6e9 > 2.15e9].
    check_time_case("a pause of an hour",
                    {{t, h, t, ""}, {t + 3'600'000'000, h + 3'600'000'000, t + 3'600'000'000, ""}});
    constexpr std::uint64_t ms = 1'000; // 1 ms, in µs
    // A jump with a rest: the nearest whole number of jumps is taken out, not fewer.
    check_time_case("a jump up with a rest of +3 ms",
                    {{t, h, t, ""}, {t + s + 3 * ms + jump_us, h + s, t + s + 3 * ms, ""}});
    check_time_case("a jump up with a rest of -3 ms",
                    {{t, h, t, ""}, {t + s + jump_us, h + s + 3 * ms, t + s, ""}});
    check_time_case("a jump down with a rest of +3 ms",
                    {{t, h, t, ""}, {t + s + 3 * ms - jump_us, h + s, t + s + 3 * ms, ""}});
    check_time_case("a jump down with a rest of -3 ms",
                    {{t, h, t, ""}, {t + s - jump_us, h + s + 3 * ms, t + s, ""}});
    check_time_case("the camera's clock lagging the host's by 5 s",
                    {{t, h, t, ""}, {t + s, h + s + 5'000'000, t + s, ""}});
    constexpr std::uint64_t tolerance_us = 10'000'000;
    check_time_case(
        "the clocks apart by the tolerance, 10 s, either way",
        {{t, h, t, ""},
         {t + s + tolerance_us, h + s, t + s + tolerance_us, ""},
         {t + 2 * s + tolerance_us, h + 2 * s + tolerance_us, t + 2 * s + tolerance_us, ""}});
    check_time_case("the clocks apart by 1 µs more than the tolerance, either way",
                    {{t, h, t, ""},
                     {t + s + tolerance_us + 1, h + s, 0,
                      "the camera's clock stepped by 10033334 µs while the host's stepped by "
                      "33333 µs, which is no jump of 2^32 µs"},
                     {t + s, h + s + tolerance_us + 1, 0,
                      "the camera's clock stepped by 33333 µs while the host's stepped by "
                      "10033334 µs, which is no jump of 2^32 µs"},
                     {t + s, h + s, t + s, ""}});
    check_time_case("a step of 1 s against 30 s on the host",
                    {{t, h, t, ""},
                     {t + 1'000'000, h + 30'000'000, 0,
                      "the camera's clock stepped by 1000000 µs while the host's stepped by "
                      "30000000 µs, which is no jump of 2^32 µs"},
                     {t + 30'000'000, h + 30'000'000 + s, t + 30'000'000, ""}});
    // A backlog: the caller stalls for 2 s while the camera sends at 5 fps, then takes the 10
    // framesets OrbbecSDK's queue holds [S: Pipeline.cpp, OrbbecSDK v2.10.6] 1 ms apart.
    std::vector<time_step> backlog = {{t, h, t, ""}};
    for (std::uint64_t k = 1; k <= 10; k++)
        backlog.push_back({t + k * 200 * ms, h + 2'000 * ms + k * ms, t + k * 200 * ms, ""});
    check_time_case("a backlog of 10 frames at 5 fps", backlog);
    check_time_case("a step back by 1 ms",
                    {{t, h, t, ""},
                     {t - 1'000, h + s, 0,
                      "the frame's timestamp, " + std::to_string(t - 1'000) +
                          " µs, is not after the last taken frame's, " + std::to_string(t) + " µs"},
                     {t + s, h + s, t + s, ""}});
    check_time_case("the same timestamp again",
                    {{t, h, t, ""},
                     {t, h + s, 0,
                      "the frame's timestamp, " + std::to_string(t) +
                          " µs, is not after the last taken frame's, " + std::to_string(t) + " µs"},
                     {t + s, h + 2 * s, t + s, ""}});
    check_time_case("the camera's clock 20 s ahead of the host's",
                    {{t, h, t, ""},
                     {t + 20'000'000, h + s, 0,
                      "the camera's clock stepped by 20000000 µs while the host's stepped by "
                      "33333 µs, which is no jump of 2^32 µs"},
                     {t + s, h + s, t + s, ""}});
    check_time_case("a timestamp of 0",
                    {{t, h, t, ""},
                     {0, h + s, 0,
                      "the frame's timestamp on the camera's clock, 0 µs, is 0 or 2^60 µs or "
                      "more"},
                     {t + s, h + 2 * s, t + s, ""}});
    check_time_case("a timestamp of 2^60 µs",
                    {{std::uint64_t{1} << 60, h, 0,
                      "the frame's timestamp on the camera's clock, 1152921504606846976 µs, is 0 "
                      "or 2^60 µs or more"},
                     {t, h, t, ""}});
    constexpr std::uint64_t last_below_bound_us = (std::uint64_t{1} << 60) - 1;
    check_time_case("a jump down that takes the timestamp to 2^60 µs",
                    {{last_below_bound_us, h, last_below_bound_us, ""},
                     {last_below_bound_us + s - jump_us, h + s, 0,
                      "the frame's timestamp, 1152921504606880308 µs, is 2^60 µs or more"}});
    check_time_case("a host time of 2^60 µs",
                    {{t, std::uint64_t{1} << 60, 0,
                      "the host's time, 1152921504606846976 µs, is 2^60 µs or more"},
                     {t, h, t, ""}});
}

/// Checks that start() begins a new opening: a lower index and timestamp and another depth unit
/// than the last opening's are taken.
void check_new_opening()
{
    gca::orbbec_frame_gate gate;
    gate.start(test_size, test_intrinsics);
    gca::frame out;
    sample old = make_sample(test_size, 1, 500, 9'000'000, 9'000'000);
    check(taken(take_sample(gate, old, out)), "new opening: the last opening's frame");
    sample skipped = make_sample(test_size, 3, 400, 9'100'000, 9'100'000);
    check_skipped("new opening: the last opening's skipped frame", gate, skipped, out,
                  "the frame's index 400 is not after the last taken frame's 500");
    gate.start(test_size, test_intrinsics);
    check(gate.skips().count == 0 && gate.skips().last_reason.empty(),
          "new opening: no skipped frame counted");
    sample fresh = make_sample(test_size, 2, 1, 1'000, 9'500'000);
    fresh.frameset.depth_scale_mm = 0.1f;
    check(taken(take_sample(gate, fresh, out)),
          "new opening: its first frame taken, index, timestamp and unit lower");
    check_holds_sample("new opening", out, fresh, 1'000);
}
} // namespace

/// Runs every check of the Orbbec camera's rules.
/// @return 0 when every check passed, 1 otherwise (check.hpp, check_summary()).
int main()
{
    check_stream_choice();
    check_openings();
    check_frames_timeout_and_failure();
    check_picture_refusals();
    check_indices();
    check_timestamps();
    check_new_opening();
    return gca_test::check_summary("test_orbbec_rules");
}
