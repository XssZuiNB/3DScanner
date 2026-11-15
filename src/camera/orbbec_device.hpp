// SPDX-License-Identifier: 0BSD
/// @file orbbec_device.hpp
/// The Orbbec RGB-D camera (Gemini 2 L) as a frame source (camera/frame_source.hpp), through
/// OrbbecSDK v2. Depth is aligned to the color camera, so both pictures share one size and one
/// set of intrinsics. Each frameset passes the checks of camera/orbbec_rules.hpp: its size, its
/// 16-bit depth and its depth unit on every frame, its index and its timestamp, with OrbbecSDK's
/// jump of 2^32 µs taken out. A frameset with a bad timestamp or index is skipped and counted; a
/// timeout, an exception of OrbbecSDK or OpenCV and a frameset of a wrong size, depth format or
/// depth unit come back as values, and the caller's frame stays as it was. Color is delivered as
/// RGB8. The IMU's accelerometer and gyroscope stream beside depth and color, each through
/// OrbbecSDK's sensor of its own into a queue (camera/orbbec_imu.hpp), and each frame taken
/// carries their samples (docs/methods/imu-samples.md); when they cannot be started the camera
/// opens without them, and its frames carry none. Besides the seam it tells its caller what an
/// opening read and chose (opening()), the frames it skipped (skips()) and the IMU's readings it
/// lost (imu_skips()), which orbbec_viewer prints, and while it is open it reads and writes the
/// color camera's run-time properties of camera/camera_properties.hpp, which camera/color_hold.hpp
/// holds for a scan.

#pragma once

#include "camera/camera_properties.hpp"
#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/orbbec_imu.hpp"
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
    /// The IMU's streams started, as "Accel: <rate> <range> | Gyro: <rate> <range>" in OrbbecSDK's
    /// names of the rate and the full-scale range [S: orbbec_device.cpp, start_imu()]; empty when
    /// they were not started.
    std::string imu_streams;
    /// Why the opening went on without the IMU (camera/orbbec_imu.hpp, imu_refusal()); empty when
    /// the IMU's streams started, or the opening did not get that far.
    std::string imu_refusal;
};

/// The Orbbec camera behind the frame source seam and the seam of its run-time properties.
class orbbec_device final : public frame_source, public camera_properties
{
public:
    /// Prepares a camera; nothing is opened until open().
    /// @param fps [in] the frame rate of both streams, in frames per second, for every opening.
    /// @note Allocates the IMU's two queues of orbbec_imu_queue_readings readings, 26 KB each
    ///       [S: computed, 32 bytes a reading]; throws std::bad_alloc when that fails.
    explicit orbbec_device(std::uint32_t fps);

    /// Closes the camera when it is open.
    ~orbbec_device() override;

    /// Prints the first connected device's name, serial number, firmware, USB connection and
    /// every color and depth stream profile to stdout. Opens the device, starts no stream, and
    /// prints a firmware that open() refuses without a word about it. An exception is printed as
    /// "Orbbec error: <message>".
    static void print_stream_profiles();

    /// Opens the first connected camera and starts its color and aligned depth streams, then its
    /// accelerometer and gyroscope at orbbec_imu_rate_hz (start_imu()), or none of them, which
    /// refuses nothing: opening() says why, and the frames then carry no samples.
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
    /// time limit, and skips() counts it. A frame taken carries the IMU's samples the queues give
    /// it (orbbec_imu_queue::deliver()); next() waits for no sample.
    /// @param out [out] filled with the frame and its samples when the result is a frame; left as
    ///            it was on every other result. Owned by the caller.
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
    ///       the reason of a skip, out's buffers when they grow, its lists of samples beyond
    ///       frame_imu_samples_reserved, and the decoder's buffer for a color format other than
    ///       RGB when it grows. Every exception thrown while waiting and decoding becomes a
    ///       failure; a failed allocation of out's buffers ends the program (std::terminate).
    ///       Takes each IMU queue's lock for its walk, after the wait. One thread at a time.
    next_result next(frame &out) noexcept override;

    /// Stops the streams, the IMU's first, and releases the camera. A closed camera stays closed;
    /// an exception of OrbbecSDK while stopping is dropped, as close() has no result, and the
    /// camera is released either way.
    void close() noexcept override;

    /// Gives the frames the current or the last opening skipped for their timestamp or index.
    /// @return their count and the last one's reason, valid until the next next() or open().
    const orbbec_skips &skips() const noexcept;

    /// Gives the IMU's readings the current or the last opening lost: dropped from a full queue,
    /// unreadable, not placed on the frames' clock, out of order (orbbec_imu_queue::deliver()).
    /// @return their count and the last one's reason, valid until the next next() or open().
    const orbbec_skips &imu_skips() const noexcept;

    /// Tells what the last opening read from the camera and chose, whether it opened or not.
    /// @return the camera's serial number, its link, the streams chosen and the IMU's streams or
    ///         why it went on without them, each empty where the opening did not get that far;
    ///         valid until the next open().
    const orbbec_opening &opening() const noexcept;

    /// Describes the camera.
    /// @return OrbbecSDK's name of the camera and its firmware's version as the last opening read
    ///         them (the camera "Orbbec camera" and no firmware before an opening reached the
    ///         device), and OrbbecSDK's version, major.minor.patch, which is the library's own.
    /// @note Allocates for the strings. One thread at a time.
    source_description describe() const noexcept override;

    /// Reads one of the color camera's properties through OrbbecSDK: a switch with
    /// getBoolProperty(), the others with getIntProperty().
    /// @param property [in] the property.
    /// @return its value, a switch 0 or 1; or nothing with the reason: "the camera is not open",
    ///         or "OrbbecSDK: <message>" for the exception OrbbecSDK threw.
    /// @note Allocates inside OrbbecSDK and for a refusal's reason. One thread at a time, the
    ///       thread that calls next(). Whether a read while the streams run delays a frame is
    ///       unmeasured [?].
    property_reading read_property(color_property property) noexcept override;

    /// Writes one of the color camera's switches through OrbbecSDK's setBoolProperty(), at run
    /// time; every other property is refused before OrbbecSDK sees it, by is_switch(), the rule of
    /// AGENTS.md §3's list. Whether the camera keeps a written switch in its flash is unverified
    /// [?: AGENTS.md §3].
    /// @param property [in] the property, a switch.
    /// @param value [in] 0 (off) or 1 (on).
    /// @return nothing when OrbbecSDK took it; otherwise the reason: "<name> (<number>) is no
    ///         switch, which the scanner may not write (AGENTS.md §3)", "the camera is not open",
    ///         or "OrbbecSDK: <message>" for the exception OrbbecSDK threw.
    /// @note Allocates inside OrbbecSDK and for a refusal's reason. One thread at a time, the
    ///       thread that calls next().
    std::optional<std::string> write_property(color_property property,
                                              std::int32_t value) noexcept override;

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

    /// Starts the accelerometer and the gyroscope at orbbec_imu_rate_hz, each through OrbbecSDK's
    /// sensor of its own, whose callback puts every reading into the sensor's queue, which it
    /// empties first. Both start or neither: when the camera does not offer both at the rate, or
    /// OrbbecSDK throws, the one started is stopped. Records in m_opening the streams started, or
    /// imu_refusal()'s reason.
    /// @pre The pipeline runs (start_pipeline()); the IMU is not started.
    /// @note Allocates the sensors, their profile lists and callbacks, and the opening's strings.
    ///       Every exception of OrbbecSDK becomes the IMU's refusal; none leaves. The callbacks run
    ///       on OrbbecSDK's threads until stop_imu().
    void start_imu() noexcept;

    /// Stops the IMU's sensors that run and releases them, so that no callback runs after it
    /// returns [?: OrbbecSDK's stop is taken to wait for its callback], then stops both queues,
    /// which empties them and makes them take no reading until start_imu() starts them again. An
    /// exception of OrbbecSDK is dropped, and the sensor released either way.
    void stop_imu() noexcept;

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

    // The IMU: its two sensors, alive while they stream, since OrbbecSDK's callback calls into
    // the sensor object, and their queues, which OrbbecSDK's threads fill and next() empties.
    std::shared_ptr<ob::Sensor> m_accel_sensor; ///< set while start_imu() runs or it streams
    std::shared_ptr<ob::Sensor> m_gyro_sensor;  ///< set while start_imu() runs or it streams
    bool m_accel_started = false;               ///< whether the accelerometer streams
    bool m_gyro_started = false;                ///< whether the gyroscope streams
    orbbec_imu_queue m_accel_queue{"accelerometer", orbbec_imu_queue_readings};
    orbbec_imu_queue m_gyro_queue{"gyroscope", orbbec_imu_queue_readings};
    orbbec_skips m_imu_skips; ///< the IMU's readings this opening lost
};
} // namespace gca
