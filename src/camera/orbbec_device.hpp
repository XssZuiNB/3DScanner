// SPDX-License-Identifier: 0BSD
/// @file orbbec_device.hpp
/// The Orbbec RGB-D camera (Gemini 2 L) as a frame source (camera/frame_source.hpp), through
/// OrbbecSDK v2. Depth is aligned to the color camera, so both pictures share one size and one
/// set of intrinsics. Each frameset passes the checks of camera/orbbec_rules.hpp: its size, its
/// 16-bit depth and its depth unit on every frame, its index and its timestamp, with OrbbecSDK's
/// jump of 2^32 µs taken out. A frameset with a bad timestamp or index is skipped and counted; a
/// timeout, an exception of OrbbecSDK or OpenCV and a frameset of a wrong size, depth format or
/// depth unit come back as values, and the caller's frame stays as it was. Color is delivered as
/// RGB8. Besides the seam it tells its caller what an opening read and chose (opening()) and the
/// frames it skipped (skips()), which orbbec_viewer prints.

#pragma once

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/orbbec_rules.hpp"
#include "types/frame_description.hpp"

#include <libobsensor/ObSensor.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace gca
{
/// What an opening of the camera read from it and chose, for orbbec_viewer's lines.
struct orbbec_opening
{
    std::string serial_number; ///< the camera's serial number; empty when no camera was reached
    /// OrbbecSDK's name of its link, "USB3.0" at 5000M [M 2026-11-13, docs/HARDWARE.md]; empty
    /// when no camera was reached.
    std::string connection_type;
    /// The streams chosen, as "Color: <w>x<h>@<fps> <format> | Depth: <w>x<h>@<fps> <format> |
    /// D2C: <hardware or software>" [S: orbbec_device.cpp, select_profiles()], "Color:
    /// 640x400@30 RGB | Depth: 640x400@30 Y14 | D2C: hardware" at 640x400 [M 2026-11-13,
    /// docs/HARDWARE.md]; empty when the camera offered no pair of streams for the size.
    std::string streams;
};

/// The Orbbec camera behind the frame source seam.
class orbbec_device final : public frame_source
{
public:
    /// Prepares a camera; nothing is opened until open().
    /// @param fps [in] the frame rate of both streams, in frames per second, for every opening.
    explicit orbbec_device(std::uint32_t fps);

    /// Closes the camera when it is open.
    ~orbbec_device() override;

    /// Prints the first connected device's name, serial number, firmware, USB connection and
    /// every color and depth stream profile to stdout. Opens the device, starts no stream, and
    /// prints a firmware that open() refuses without a word about it. An exception is printed as
    /// "Orbbec error: <message>".
    static void print_stream_profiles();

    /// Opens the first connected camera and starts its color and aligned depth streams.
    /// @param size [in] the stream size, in px: a color profile of that size and the
    ///        constructor's frame rate, then a depth profile aligned to it, as
    ///        camera/orbbec_rules.hpp's choose_color_profile() and choose_depth_profile() choose.
    /// @return opened, or not opened with the reason of camera/orbbec_rules.hpp,
    ///         opening_refusal(): held by another program, another exception of OrbbecSDK, no
    ///         camera, a firmware older than 1.4.53 or a version that is not three numbers
    ///         (firmware_refusal(), checked before any stream starts), no such stream (on a USB 2
    ///         link named as the cause); or the camera is open already, and stays open as it was.
    ///         A refused opening releases what it opened.
    /// @note Allocates OrbbecSDK's context, device and pipeline, and the reason of a refusal. Every
    ///       exception thrown inside becomes a refusal. One thread at a time.
    open_result open(const frame_size &size) noexcept override;

    /// Waits up to orbbec_frame_time_limit_ms for the next frame and gives it. A frameset the
    /// gate skips for its timestamp or index is not delivered; the wait goes on within the same
    /// time limit, and skips() counts it.
    /// @param out [out] filled with the frame when the result is a frame; left as it was on every
    ///            other result. Owned by the caller.
    /// @return frame; timeout; failure with OrbbecSDK's or OpenCV's message, or with the reason
    ///         camera/orbbec_rules.hpp's orbbec_frame_gate refused the frameset; failure when the
    ///         camera is not open. A failure ends nothing: the camera stays open.
    ///         An error of OrbbecSDK during the wait, a camera unplugged among them, comes back as
    ///         a timeout: the installed Pipeline.hpp returns nullptr before it hands the error to
    ///         Error::handle() [S: OrbbecSDK v2.10.6, Pipeline.hpp, waitForFrameset].
    ///         The frame's index is OrbbecSDK's number of the color frame, so a frame dropped
    ///         after libuvc handed it on leaves a gap. OrbbecSDK hands out the oldest of up to 10
    ///         queued framesets and drops the oldest when its queue is full [S: OrbbecSDK v2.10.6,
    ///         Pipeline.cpp], so a caller slower than the camera gets frames up to 10 frames old.
    /// @note Allocates, unlike the seam's note: OrbbecSDK's wrapper objects of each frameset
    ///       (make_shared in Pipeline.hpp, waitForFrameset, and Frame.hpp, getColorFrame and
    ///       getDepthFrame [S: OrbbecSDK v2.10.6]), the message of a result other than a frame,
    ///       the reason of a skip, out's buffers when they grow, and the decoder's buffer for a
    ///       color format other than RGB when it grows. Every exception thrown while waiting and
    ///       decoding becomes a failure; a failed allocation of out's buffers ends the program
    ///       (std::terminate). One thread at a time.
    next_result next(frame &out) noexcept override;

    /// Stops the streams and releases the camera. A closed camera stays closed; an exception of
    /// OrbbecSDK while stopping is dropped, as close() has no result, and the camera is released
    /// either way.
    void close() noexcept override;

    /// Gives the frames the current or the last opening skipped for their timestamp or index.
    /// @return their count and the last one's reason, valid until the next next() or open().
    const orbbec_skips &skips() const noexcept;

    /// Tells what the last opening read from the camera and chose, whether it opened or not.
    /// @return the camera's serial number, its link and the streams chosen, each empty where the
    ///         opening did not get that far; valid until the next open().
    const orbbec_opening &opening() const noexcept;

    /// Describes the camera.
    /// @return OrbbecSDK's name of the camera and its firmware's version as the last opening read
    ///         them (the camera "Orbbec camera" and no firmware before an opening reached the
    ///         device), and OrbbecSDK's version, major.minor.patch, which is the library's own.
    /// @note Allocates for the strings. One thread at a time.
    source_description describe() const noexcept override;

private:
    /// Picks the color and the depth profile for a stream size with choose_color_profile() and
    /// choose_depth_profile() and records in the report which were found.
    /// @param size [in] the stream size, in px.
    /// @param report [in,out] gains whether a color and a depth profile were found.
    /// @pre m_pipeline exists. OrbbecSDK's exceptions pass through to open().
    void select_profiles(const frame_size &size, orbbec_open_report &report);

    /// Starts the pipeline with the selected profiles, depth aligned to color, both frames
    /// required in every frameset.
    /// @pre select_profiles() found both profiles. OrbbecSDK's exceptions pass through to open().
    void start_pipeline();

    /// Waits for the next frameset and turns it into plain data for orbbec_frame_gate.
    /// @param time_limit_ms [in] how long to wait, in ms, > 0.
    /// @return the wait's status; with a frameset its pixels, which m_color_frame, m_depth_frame
    ///         and m_color_rgb keep valid until the next call or release_frames().
    /// @pre The camera is open. OrbbecSDK's and OpenCV's exceptions pass through to next().
    orbbec_wait wait_for_frameset(std::uint32_t time_limit_ms);

    /// Gives a color frame's pixels as RGB8: RGB as it is, BGR and YUYV converted, MJPG decoded
    /// and converted, the last three into m_color_rgb.
    /// @param color [in] the color frame.
    /// @param frameset [in,out] gains the picture's size (the frame's, or the decoded one for
    ///        MJPG) and its pixels, valid while color and m_color_rgb are.
    /// @return nothing when it decoded; otherwise why not: a format the camera source does not
    ///         decode, raw data of another length than its size needs, an MJPG frame that does not
    ///         decode.
    /// @note Allocates m_color_rgb when it grows, OpenCV's decoded picture of MJPG, and the
    ///       reason. OpenCV's exceptions pass through.
    std::optional<std::string> decode_color(const std::shared_ptr<ob::ColorFrame> &color,
                                            orbbec_frameset &frameset);

    /// Drops the frames the last frameset left referenced.
    void release_frames() noexcept;

    /// Stops the pipeline when it runs and releases OrbbecSDK's pipeline, device and context.
    void release() noexcept;

    std::shared_ptr<ob::Context> m_context;
    std::shared_ptr<ob::Device> m_device;
    std::shared_ptr<ob::Pipeline> m_pipeline;
    std::shared_ptr<ob::Align> m_sw_align; ///< set when hardware alignment is not offered
    std::shared_ptr<ob::VideoStreamProfile> m_color_profile;
    std::shared_ptr<ob::VideoStreamProfile> m_depth_profile;
    bool m_hw_d2c = false;           ///< whether the camera aligns depth in hardware
    bool m_pipeline_started = false; ///< whether the pipeline runs
    bool m_open = false;             ///< whether an opening succeeded and was not closed
    const std::uint32_t m_fps;       ///< the frame rate of both streams, in frames per second
    std::string m_camera_name;       ///< OrbbecSDK's name of the camera, from the last opening
    std::string m_firmware_version;  ///< the camera's firmware, from the last opening
    orbbec_opening m_opening;        ///< what the last opening read and chose
    orbbec_frame_gate m_gate;        ///< the checks of the current opening

    // The last frameset's frames stay referenced until the next wait, so that the pixels handed
    // to the gate stay valid.
    std::shared_ptr<ob::ColorFrame> m_color_frame;
    std::shared_ptr<ob::DepthFrame> m_depth_frame;
    std::string m_depth_format;            ///< the name of a depth format the gate refuses
    std::vector<std::uint8_t> m_color_rgb; ///< decoded color of a format other than RGB
    cv::Mat m_decoded_bgr;                 ///< OpenCV's decoded picture of an MJPG frame
};
} // namespace gca
