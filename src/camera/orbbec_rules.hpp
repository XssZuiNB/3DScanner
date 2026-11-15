// SPDX-License-Identifier: 0BSD
/// @file orbbec_rules.hpp
/// The Orbbec camera's decisions that need no camera, as functions of plain data: the profiles
/// OrbbecSDK offers become the chosen color and depth streams, what OrbbecSDK reported while the
/// camera was opened, its firmware's version among it, becomes an opening or a refusal with its
/// reason, and what a wait for the next frameset gave becomes the frame source's result
/// (camera/frame_source.hpp): a frame that passed its checks, a frame skipped for its timestamp or
/// index, a timeout, or a failure with its message. The checks of a frame are those of one
/// opening: its size, its depth format and depth unit, its index and its timestamp, where the jump
/// of exactly 2^32 µs that OrbbecSDK's timestamps make [M 2026-11-13, docs/HARDWARE.md] is taken
/// out; by the same rule a time of the camera's clock, an IMU reading's, is placed on the clock of
/// the frames delivered. camera/orbbec_device.cpp feeds these functions with what OrbbecSDK
/// reports; they build and link without OrbbecSDK, so the offline tests drive them
/// (tests/test_orbbec_rules.cpp, tests/test_orbbec_imu.cpp).

#pragma once

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "types/camera_param.hpp"
#include "types/frame_description.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace gca
{
/// How long the camera's source waits for a frame before next() reports a timeout, in ms
/// [?: a choice: the prototype's 5 s, which AGENTS.md §3's stop condition "No frames received
/// from Orbbec camera for 5s!" names].
constexpr std::uint32_t orbbec_frame_time_limit_ms = 5000;

/// How far the camera's clock may step otherwise than the host's between two frames of an
/// opening, once the jumps of 2^32 µs are taken out, in µs, either way [?: a choice: five times
/// the 2 s that OrbbecSDK's frameset queue of 10 holds at 5 fps (S: Pipeline.cpp, OrbbecSDK
/// v2.10.6), and far below the 2^31 µs at which a jump could not be told from a step].
constexpr std::int64_t orbbec_clock_tolerance_us = 10'000'000;

/// The kinds of OrbbecSDK's exceptions that the rules tell apart; camera/orbbec_device.cpp maps
/// OrbbecSDK's OBExceptionType to them by name.
enum class orbbec_exception
{
    io,            ///< an input or output error, OB_EXCEPTION_TYPE_IO
    resource_busy, ///< a resource busy or locked by another operation, ..._RESOURCE_BUSY
    other,         ///< any other type, or an exception that is not an ob::Error
};

/// An exception OrbbecSDK threw, as plain data.
struct orbbec_error
{
    orbbec_exception type = orbbec_exception::other; ///< its kind
    std::string message;                             ///< its message, what()
};

/// The stream formats that the choice of streams tells apart; camera/orbbec_device.cpp maps
/// OrbbecSDK's OBFormat to them by name.
enum class orbbec_format
{
    bgr,   ///< color, 8 bits each of blue, green, red
    rgb,   ///< color, 8 bits each of red, green, blue
    mjpg,  ///< color, Motion JPEG
    yuyv,  ///< color, YUV 4:2:2
    y16,   ///< depth, 16 bit
    z16,   ///< depth, 16 bit
    y14,   ///< depth, 14 bit, which arrives as 16 bit [M 2026-11-13, AGENTS.md §6]
    rle,   ///< depth, run-length coded
    other, ///< any other format
};

/// A stream profile OrbbecSDK offers, as plain data.
struct orbbec_profile
{
    std::uint32_t width;  ///< its pictures' width, in px
    std::uint32_t height; ///< its pictures' height, in px
    std::uint32_t fps;    ///< its frame rate, in frames per second
    orbbec_format format; ///< its format
};

/// Chooses the color stream: a profile of the asked size and rate, in the order of formats BGR,
/// RGB, MJPG, YUYV, the one needing the least conversion first [S: the prototype's order], the
/// list's first of a format.
/// @param profiles [in] the color profiles OrbbecSDK offers, in its order.
/// @param size [in] the stream size asked for, in px.
/// @param fps [in] the frame rate asked for, in frames per second.
/// @return the chosen profile's position in profiles, or nothing when none fits.
std::optional<std::size_t> choose_color_profile(std::span<const orbbec_profile> profiles,
                                                const frame_size &size, std::uint32_t fps);

/// The chosen depth stream.
struct orbbec_depth_choice
{
    std::size_t position; ///< its position in the list it was chosen from
    bool hardware;        ///< chosen from the hardware alignment's list; else from all profiles
};

/// Chooses the depth stream aligned to the chosen color stream, in the order of formats Y16, Z16,
/// Y14, RLE [S: the prototype's order; the gate refuses a frame that does not arrive as 16 bit]:
/// a profile of the asked rate, of any size, that the camera aligns in hardware; else a profile of
/// the asked size and rate that OrbbecSDK's software filter aligns, so that no depth of another
/// size is scaled to the color's [?: the software filter is unmeasured].
/// @param hardware [in] the depth profiles the camera aligns in hardware to the chosen color
///        profile (Pipeline::getD2CDepthProfileList()), in OrbbecSDK's order; empty when none.
/// @param all [in] every depth profile OrbbecSDK offers, in its order.
/// @param size [in] the stream size asked for, in px.
/// @param fps [in] the frame rate asked for, in frames per second.
/// @return the chosen profile and its list, or nothing when none fits.
std::optional<orbbec_depth_choice> choose_depth_profile(std::span<const orbbec_profile> hardware,
                                                        std::span<const orbbec_profile> all,
                                                        const frame_size &size, std::uint32_t fps);

/// What OrbbecSDK reported while the camera was being opened, step by step. A step the opening
/// did not reach keeps its default.
struct orbbec_open_report
{
    std::uint32_t camera_count = 0; ///< the cameras ob::Context::queryDeviceList() found
    /// The first camera's link as ob::DeviceInfo::getConnectionType() names it: "USB2.0",
    /// "USB3.0" and the like [S: Device.hpp, UsbTypes.cpp, OrbbecSDK v2.10.6]; empty when not read.
    std::string connection_type;
    /// The first camera's firmware version as ob::DeviceInfo::getFirmwareVersion() gives it: the
    /// camera's own string, which OrbbecSDK reads from the camera and passes on unchanged
    /// [S: OrbbecSDK v2.10.6, DeviceBase.cpp, fetchDeviceInfo], "1.5.02" [M 2026-11-15,
    /// docs/HARDWARE.md]; empty when not read.
    std::string firmware_version;
    bool color_stream_offered = false; ///< choose_color_profile() found a profile
    bool depth_stream_offered = false; ///< choose_depth_profile() found a profile
    std::optional<orbbec_error> error; ///< the exception that ended the opening, if one did
};

/// Decides whether a camera's firmware is too old for the scanner, or cannot be read. The least
/// version is OrbbecSDK v2's least firmware for the Gemini 2 L, 1.4.53 [S: OrbbecSDK v2's README,
/// web, read 2026-11-15]; the tree serves only that camera, so the rule is applied to whatever
/// camera is connected, without a check of its name [?: a choice]. On an older firmware, 1.4.38,
/// every frame's timestamp stood still and every frame was skipped [M 2026-11-15,
/// docs/HARDWARE.md, "What the camera is and reports"].
///
/// The version is read as three numbers major.minor.patch, each one or more ASCII digits read
/// whole into 32 bits ("02" is 2), separated by dots, and compared part by part as numbers, major
/// first: 1.10.0 is above 1.4.53. Anything else cannot be read: an empty string or part, a missing
/// or a fourth part, a sign, a space, a letter or a suffix anywhere, a part beyond 32 bits. This is
/// stricter than OrbbecSDK's own reading, DeviceBase::getFirmwareVersionInt(), which skips every
/// character but digits and dots and refuses a part above 99 [S: OrbbecSDK v2.10.6,
/// DeviceBase.cpp]: a string of another form is refused with the string shown, rather than read as
/// a version it may not be [?: a choice, the orchestrator's].
/// @param version [in] the camera's firmware version as OrbbecSDK gives it, "1.5.02"
///        [M 2026-11-15].
/// @return nothing when the version is 1.4.53 or later. Otherwise the reason: the version as found
///         (in quotes when it cannot be read), the least, and that Orbbec's firmware tool updates
///         it on the user's word only, as it writes the camera's flash (AGENTS.md §3).
/// @note Allocates for the reason.
std::optional<std::string> firmware_refusal(std::string_view version);

/// Decides whether an opening of the camera is refused, and why.
/// @param report [in] what OrbbecSDK reported while the camera was opened.
/// @param size [in] the stream size asked for, in px.
/// @param fps [in] the frame rate asked for, in frames per second.
/// @return nothing when the camera opened: no exception, a camera, a firmware the firmware's rule
///         accepts, both streams. Otherwise the reason, in this order: an exception that says the
///         camera is held by another program (resource busy; or an input or output error whose
///         message holds "uvc_open failed" and "Return Code: -6", which is how OrbbecSDK's LibUVC
///         backend reports a claim of the camera's interface that libusb refused as busy,
///         LIBUSB_ERROR_BUSY
///         [S: OrbbecSDK v2.10.6, ObLibuvcDevicePort.cpp; its libuvc, device.c, uvc_claim_if;
///         OrbbecSDKConfig.xml gives the Gemini 2 L the LibUVC backend]); any other exception,
///         with its message; no camera; the firmware's refusal, firmware_refusal(), which comes
///         before the streams as orbbec_device starts none for a firmware it refuses; no stream
///         of that size and rate, where a link named "USB1..." or "USB2..." is named as the cause
///         (AGENTS.md §6). Each recognition is unverified on the camera [?].
/// @note Allocates for the reason.
std::optional<std::string> opening_refusal(const orbbec_open_report &report, const frame_size &size,
                                           std::uint32_t fps);

/// One frameset as OrbbecSDK delivered it, its color already decoded to RGB8, as plain data.
/// The pixels are only read; they belong to OrbbecSDK or to the camera's decoder and must stay
/// valid until orbbec_frame_gate::take() returns.
struct orbbec_frameset
{
    frame_size color_size{};                 ///< the color picture's size, in px
    std::span<const std::uint8_t> color_rgb; ///< its pixels, RGB8 in camera/frame.hpp's order
    frame_size depth_size{};                 ///< the aligned depth picture's size, in px
    bool depth_16_bit = false;     ///< whether depth arrived as 16 bit per pixel (Y16 or Z16)
    std::string_view depth_format; ///< the depth format's name, for the reason of a refusal
    std::span<const std::uint16_t> depth; ///< its pixels; empty when depth_16_bit is false
    float depth_scale_mm = 0.0f;          ///< mm per depth value, DepthFrame::getValueScale()
    /// The color frame's number, ob::Frame::getIndex(). OrbbecSDK numbers each frame its LibUVC
    /// backend hands on, from 1 at the stream's start [S: OrbbecSDK v2.10.6,
    /// ObLibuvcDevicePort.cpp; Frame.cpp, copyInfoFromOther keeps it]; a frame dropped after that
    /// (by OrbbecSDK's queues, its aggregator or its timestamp check, or skipped here) leaves a
    /// gap, one libuvc never handed on leaves none [?].
    std::uint64_t index = 0;
    /// The color frame's timestamp on the camera's clock, ob::Frame::getTimeStampUs(), in µs.
    std::uint64_t camera_time_us = 0;
    /// When the source received it, on the host's steady clock, in µs.
    std::uint64_t host_time_us = 0;
};

/// What a wait for the next frameset gave.
enum class orbbec_wait_status
{
    frameset, ///< a frameset arrived
    timeout,  ///< none arrived within the time left of orbbec_frame_time_limit_ms
    failure,  ///< OrbbecSDK or the decoder failed; the message says how
};

/// What a wait for the next frameset gave, as plain data.
struct orbbec_wait
{
    orbbec_wait_status status = orbbec_wait_status::failure; ///< which of the three
    orbbec_frameset frameset;                                ///< the frameset, with frameset
    std::string message; ///< what failed, in plain English, with failure; empty otherwise
};

/// Whether orbbec_frame_gate::place() placed a time on the frames' clock, and why not.
enum class orbbec_placement
{
    placed,                   ///< the time is placed
    no_frame_taken,           ///< no frame of this opening has been taken to place it against
    camera_time_out_of_range, ///< its camera time is 0 or 2^60 µs or more, as take() refuses
    host_time_out_of_range,   ///< its host time is 2^60 µs or more, as take() refuses
    off_clock, ///< the rest lies beyond orbbec_clock_tolerance_us: not on the frames' clock
};

/// The frames an opening skipped for their timestamp or index, and why the last one was.
struct orbbec_skips
{
    std::uint64_t count = 0; ///< the frames skipped since the opening's start
    std::string last_reason; ///< why the last one was skipped; empty before the first
};

/// The checks of one opening of the camera, frame by frame. A frameset that passes becomes the
/// caller's frame. A frameset with a bad timestamp or index is a bad frame, not a failed device:
/// it is skipped (not delivered, its index a gap) and counted with its reason, and the source
/// waits on within the same time limit. A frameset of a wrong size, depth format or depth unit,
/// and an exception, are faults of the device and come back as a failure with the reason. A
/// skipped frame and a failure leave the caller's frame and the opening's state as they were. A
/// failure ends nothing by itself: the source stays open, and its caller decides.
class orbbec_frame_gate
{
public:
    /// Starts the checks of an opening: forgets the previous opening's depth unit, index,
    /// timestamps and skipped frames.
    /// @param size [in] the stream size the camera was opened for, in px.
    /// @param intrinsics [in] the color camera's intrinsics at that size; copied into every frame.
    void start(const frame_size &size, const intrinsics &intrinsics) noexcept;

    /// Turns what a wait for the next frameset gave into next()'s result, or skips the frameset.
    /// @param wait [in] what the wait gave; its pixels are read during the call only.
    /// @param out [out] filled when the result is a frame: prepared with the opening's size and
    ///            intrinsics, the frameset's depth unit, its index and its timestamp, then every
    ///            pixel copied. Left as it was on every other result and on a skip.
    /// @return nothing when the frameset is skipped: the caller waits for the next one within
    ///         the time left, and skips() counts it. Otherwise frame; timeout with its message;
    ///         or failure with the wait's message or the reason the frameset is refused: its color
    ///         or depth picture is not of the opening's size, its depth is not 16 bit, or its depth
    ///         unit is not a positive number or differs from the opening's first taken frame's.
    ///         Every frameset is checked, the first and every later one alike.
    ///
    ///         A frameset is skipped when its index is not after the last taken one's, or its
    ///         timestamp fails this rule. The delivered timestamp is the camera's plus an offset,
    ///         0 at the opening's start. Between the last taken frame and this one the camera's
    ///         clock stepped by d and the host's by h; d - h is split into k 2^32 µs, k the
    ///         nearest whole number, plus a rest within +-2^31 µs. A jump of OrbbecSDK's
    ///         timestamps by k 2^32 µs is taken out of the offset (OrbbecSDK joins the high 32
    ///         bits of the camera's time to each frame's low 32 bits, so a misjudged high part
    ///         jumps either way [S: OrbbecSDK v2.10.6, FrameTimestampCalculator.cpp]). The frame
    ///         is skipped when its camera timestamp is 0 (OrbbecSDK's mark of a frame without one
    ///         [S: G2FrameTimestampCalculator.cpp]) or 2^60 µs or more, its host time 2^60 µs or
    ///         more, the rest beyond orbbec_clock_tolerance_us either way, or the delivered
    ///         timestamp not after the last one or 2^60 µs or more. The bound of 2^60 µs, 36 000
    ///         years [S: computed], keeps every sum within 64 bits.
    /// @pre start() was called for this opening; before it every frameset is refused for its
    ///      size, as the opening's size is 0 x 0.
    /// @note Allocates for the message of a result other than a frame, the reason of a skip, and
    ///       out's buffers when they grow (frame::prepare()); a taken frame of the size out
    ///       already has allocates nothing. A failed allocation ends the program
    ///       (std::terminate): the function is noexcept. One thread at a time.
    std::optional<next_result> take(const orbbec_wait &wait, frame &out) noexcept;

    /// Gives the frames this opening skipped.
    /// @return their count and the last one's reason, valid until the next take() or start().
    const orbbec_skips &skips() const noexcept;

    /// Places a time of the camera's clock, an IMU reading's, on the clock of the frames take()
    /// delivers, by take()'s rule measured against the last taken frame
    /// (docs/methods/imu-samples.md, "The clock"): the camera's clock stepped by d from that
    /// frame's camera time to this one and the host's by h; d - h is split into k 2^32 µs, k the
    /// nearest whole number, plus a rest within +-2^31 µs, and the time is the frame's delivered
    /// timestamp plus d less k 2^32 µs. So a jump of OrbbecSDK's timestamps between the frame and
    /// the reading is taken out either way, whether the frame's or the reading's.
    /// @param camera_time_us [in] the time on the camera's clock, ob::Frame::getTimeStampUs(), in
    ///        µs.
    /// @param host_time_us [in] when the source received it, on the host's steady clock, in µs.
    /// @param timestamp_us [out] its time on the frames' clock, in µs, which may lie before 0 or
    ///        after the last taken frame's; set when it is placed.
    /// @return placed; or why not (orbbec_placement), which placement_refusal() puts into words.
    /// @note Allocates nothing. One thread at a time, the one that calls take().
    orbbec_placement place(std::uint64_t camera_time_us, std::uint64_t host_time_us,
                           std::int64_t &timestamp_us) const noexcept;

    /// Tells why place() does not place a time, while the gate has taken no frame since.
    /// @param camera_time_us [in] the time on the camera's clock, in µs, as place() took it.
    /// @param host_time_us [in] the host's time, in µs, as place() took it.
    /// @return the reason: no frame of this opening has been taken; the camera time is 0 or 2^60
    ///         µs or more, or the host time 2^60 µs or more; the camera's and the host's steps
    ///         from the last taken frame, which differ by no jump of 2^32 µs. Empty when place()
    ///         places the time.
    /// @note Allocates the reason. One thread at a time, the one that calls take().
    std::string placement_refusal(std::uint64_t camera_time_us, std::uint64_t host_time_us) const;

private:
    /// Checks a frameset's pictures and depth unit against the opening.
    /// @param frameset [in] the frameset.
    /// @return the reason it is refused, or nothing when its pictures pass.
    std::optional<std::string> picture_refusal(const orbbec_frameset &frameset) const;

    /// Checks a frameset's index and timestamp against the last taken frame.
    /// @param frameset [in] the frameset.
    /// @param timestamp_us [out] the timestamp to deliver, in µs; set when it passes.
    /// @param offset_us [out] the offset after this frameset's jumps, in µs; set when it passes.
    /// @return the reason it is skipped, or nothing when it passes.
    std::optional<std::string> order_refusal(const orbbec_frameset &frameset,
                                             std::int64_t &timestamp_us,
                                             std::int64_t &offset_us) const;

    frame_size m_size{};                  ///< the opening's stream size, in px
    intrinsics m_intrinsics{};            ///< the opening's color intrinsics
    bool m_has_taken = false;             ///< whether a frame of this opening was taken
    float m_depth_scale_mm = 0.0f;        ///< the first taken frame's depth unit, in mm per value
    std::uint64_t m_last_index = 0;       ///< the last taken frame's index
    std::int64_t m_last_camera_us = 0;    ///< its camera timestamp, in µs
    std::int64_t m_last_host_us = 0;      ///< its host time, in µs
    std::int64_t m_last_timestamp_us = 0; ///< its delivered timestamp, in µs
    std::int64_t m_offset_us = 0; ///< delivered minus camera timestamp, a multiple of 2^32 µs
    orbbec_skips m_skips;         ///< the frames this opening skipped
};
} // namespace gca
