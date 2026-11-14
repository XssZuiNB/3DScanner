// SPDX-License-Identifier: 0BSD
/// @file orbbec_device.hpp
/// The Orbbec RGB-D camera (Gemini 2 L) as a frame source (camera/frame_source.hpp), through
/// OrbbecSDK v2. Depth is aligned to the color camera, so both pictures share one size and one
/// set of intrinsics. Each frameset passes the checks of camera/orbbec_rules.hpp: its size, its
/// 16-bit depth and its depth unit on every frame, its index and its timestamp, with OrbbecSDK's
/// jump of 2^32 µs taken out. A frameset with a bad timestamp or index is skipped and counted; a
/// timeout, an exception of OrbbecSDK or OpenCV and a frameset of a wrong size, depth format or
/// depth unit come back as values, and the caller's frame stays as it was. Color is delivered as
/// RGB8.
///
/// orbbec_viewer still reaches the camera through gca::device (camera/camera_device.hpp): its
/// private functions here are an adapter over open(), next() and close() that converts each frame
/// back to BGR8, the order the prototype's GPU code and OpenCV's windows and files take. The
/// adapter lasts until the viewer moves onto the frame source seam, which deletes gca::device.

#pragma once

#include "camera/camera_device.hpp"
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
/// The Orbbec camera behind the frame source seam, and behind gca::device for orbbec_viewer
/// until the viewer moves onto the seam.
class orbbec_device final : public frame_source, public device
{
public:
    /// Prepares a camera; nothing is opened until open(), or gca::device::device_start().
    /// @param width [in] the color (and aligned depth) width gca::device::device_start() opens
    ///        for, in px; open() takes its own size.
    /// @param height [in] the color (and aligned depth) height gca::device::device_start() opens
    ///        for, in px; open() takes its own size.
    /// @param fps [in] the frame rate of both streams, in frames per second, for every opening.
    orbbec_device(std::uint32_t width, std::uint32_t height, std::uint8_t fps);

    /// Closes the camera when it is open.
    ~orbbec_device() override;

    /// Prints the first connected device's name, serial number, firmware, USB connection and
    /// every color and depth stream profile to stdout. Opens the device, starts no stream. An
    /// exception is printed as "Orbbec error: <message>".
    static void print_stream_profiles();

    /// Opens the first connected camera and starts its color and aligned depth streams.
    /// @param size [in] the stream size, in px: a color profile of that size and the
    ///        constructor's frame rate, then a depth profile aligned to it, as
    ///        camera/orbbec_rules.hpp's choose_color_profile() and choose_depth_profile() choose.
    /// @return opened, or not opened with the reason of camera/orbbec_rules.hpp,
    ///         opening_refusal(): held by another program, another exception of OrbbecSDK, no
    ///         camera, no such stream (on a USB 2 link named as the cause); or the camera is open
    ///         already, and stays open as it was.
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

    /// gca::device's adapter: opens the camera for the constructor's size, warns "Warning: camera
    /// is not connected via USB3 (<link>). ..." on a slower link, and names the camera for
    /// gca::device's "Found camera!" line: its name, serial number and link, then describe()'s
    /// two versions.
    /// @return true when it opened; otherwise prints the reason and returns false.
    bool find_device() final;

    /// gca::device's adapter: prints the selected streams and takes the first frame, whose depth
    /// unit and intrinsics gca::device reads next. Timeouts within the first 20 s of waiting are
    /// waited through, as the prototype did.
    /// @return true when the first frame came; otherwise prints why not ("Timeout while waiting
    ///         for first frames." after 20 s, or the failure's message) and returns false.
    bool start_stream() final;

    /// gca::device's adapter: the depth unit of the current frame.
    /// @return meters per depth value.
    float read_depth_scale() final;

    /// gca::device's adapter: the intrinsics of the current frame, the color camera's.
    /// @return the color camera's intrinsics at the opened size.
    gca::intrinsics read_color_intrinsics() final;

    /// gca::device's adapter: the depth intrinsics, the color camera's, as depth is aligned to
    /// color.
    /// @return the color camera's intrinsics.
    gca::intrinsics read_depth_intrinsics() final;

    /// gca::device's adapter: the color-to-depth extrinsics, the identity, as depth is aligned to
    /// color.
    /// @return the identity rotation and a zero translation.
    gca::extrinsics read_color_to_depth_extrinsics() final;

    /// gca::device's adapter: the depth-to-color extrinsics, the identity, as depth is aligned to
    /// color.
    /// @return the identity rotation and a zero translation.
    gca::extrinsics read_depth_to_color_extrinsics() final;

    /// gca::device's adapter: takes the next frame and converts its color to BGR8. It prints the
    /// frames skipped since the last call with the last reason; on a timeout "No frames received
    /// from Orbbec camera for 5s!" (AGENTS.md §3 reads it), on a failure its message, and keeps
    /// the current frame, as the viewer did before the seam.
    void receive_data_from_device() final;

    /// gca::device's adapter: hands the current color picture to gca::device.
    /// @return the current color picture (BGR8), sharing the adapter's buffer.
    cv::Mat set_color_to_cv_mat() final;

    /// gca::device's adapter: wraps the current depth picture as an image for gca::device.
    /// @return the current aligned depth picture (CV_16UC1), sharing the frame's buffer.
    cv::Mat set_depth_to_cv_mat() final;

    /// gca::device's adapter: closes the camera.
    void stop_stream() final;

    /// gca::device's adapter: converts the current frame's color to BGR8 and points gca::device's
    /// raw data at the current pictures.
    void use_current_frame();

    std::shared_ptr<ob::Context> m_context;
    std::shared_ptr<ob::Device> m_device;
    std::shared_ptr<ob::Pipeline> m_pipeline;
    std::shared_ptr<ob::Align> m_sw_align; ///< set when hardware alignment is not offered
    std::shared_ptr<ob::VideoStreamProfile> m_color_profile;
    std::shared_ptr<ob::VideoStreamProfile> m_depth_profile;
    bool m_hw_d2c = false;           ///< whether the camera aligns depth in hardware
    bool m_pipeline_started = false; ///< whether the pipeline runs
    bool m_open = false;             ///< whether an opening succeeded and was not closed
    std::string m_camera_name;       ///< OrbbecSDK's name of the camera, from the last opening
    std::string m_firmware_version;  ///< the camera's firmware, from the last opening
    std::string m_serial_number;     ///< the adapter's: the camera's serial number, this opening
    std::string m_connection_type;   ///< the adapter's: the camera's link, this opening
    std::string m_streams;           ///< the selected streams, for the adapter's line
    orbbec_frame_gate m_gate;        ///< the checks of the current opening

    // The last frameset's frames stay referenced until the next wait, so that the pixels handed
    // to the gate stay valid.
    std::shared_ptr<ob::ColorFrame> m_color_frame;
    std::shared_ptr<ob::DepthFrame> m_depth_frame;
    std::string m_depth_format;            ///< the name of a depth format the gate refuses
    std::vector<std::uint8_t> m_color_rgb; ///< decoded color of a format other than RGB
    cv::Mat m_decoded_bgr;                 ///< OpenCV's decoded picture of an MJPG frame

    frame m_frame;                   ///< gca::device's adapter: the current frame
    std::uint64_t m_skips_shown = 0; ///< gca::device's adapter: the skipped frames it printed
    cv::Mat m_color_bgr;             ///< gca::device's adapter: the current color picture as BGR8
};
} // namespace gca
