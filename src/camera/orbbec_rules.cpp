// SPDX-License-Identifier: 0BSD
/// @file orbbec_rules.cpp
/// The Orbbec camera's decisions that need no camera: the firmware's rule, an opening's refusal,
/// a frameset's checks and the rule for its timestamp, which also places an IMU reading's time
/// (see orbbec_rules.hpp).

#include "camera/orbbec_rules.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <system_error>
#include <utility>

namespace gca
{
namespace
{
/// The jump of OrbbecSDK's timestamps, 2^32 µs [M 2026-11-13, docs/HARDWARE.md].
constexpr std::int64_t jump_us = std::int64_t{1} << 32;

/// The bound of a timestamp and a host time, 2^60 µs: below it every sum of the timestamp rule
/// stays within 64 bits [S: computed in orbbec_frame_gate::take()].
constexpr std::int64_t time_bound_us = std::int64_t{1} << 60;

/// A step of the camera's clock against the host's, split into OrbbecSDK's jumps and the rest.
struct clock_split
{
    std::int64_t jumps;   ///< k, the nearest whole number of 2^32 µs in the difference
    std::int64_t rest_us; ///< the difference less k 2^32 µs, within +-2^31 µs
};

/// Splits the difference of a camera's step and the host's into whole jumps of 2^32 µs and a rest,
/// the one copy of the split take() and place() share.
/// @param camera_step_us [in] the camera clock's step, in µs, within +-2^60.
/// @param host_step_us [in] the host clock's step, in µs, within +-2^60.
/// @return k and the rest: camera_step_us - host_step_us = k 2^32 + rest.
clock_split split_steps(std::int64_t camera_step_us, std::int64_t host_step_us)
{
    const std::int64_t difference_us = camera_step_us - host_step_us;
    // The nearest multiple of 2^32 µs; the right shift of a negative number rounds toward minus
    // infinity in C++20.
    const std::int64_t jumps = (difference_us + jump_us / 2) >> 32;
    return {jumps, difference_us - jumps * jump_us};
}

/// Says whether the rest of a split lies beyond the clocks' tolerance.
/// @param rest_us [in] the rest, in µs.
/// @return whether it lies beyond orbbec_clock_tolerance_us either way.
bool beyond_tolerance(std::int64_t rest_us)
{
    return rest_us > orbbec_clock_tolerance_us || rest_us < -orbbec_clock_tolerance_us;
}

/// Writes a size for a message.
/// @param size [in] the size, in px.
/// @return "<width>x<height>".
std::string size_text(const frame_size &size)
{
    return std::to_string(size.width) + "x" + std::to_string(size.height);
}

/// Writes a number for a message, with up to six significant digits.
/// @param value [in] the number.
/// @return the number as printf's %g writes it.
std::string number_text(double value)
{
    char text[32];
    std::snprintf(text, sizeof(text), "%g", value);
    return text;
}

/// Decides whether an exception of OrbbecSDK says the camera is held by another program.
/// @param error [in] the exception.
/// @return whether its type is resource busy, or it is an input or output error from OrbbecSDK's
///         LibUVC backend whose claim of the camera's interface libusb refused as busy (-6)
///         [S: OrbbecSDK v2.10.6, ObLibuvcDevicePort.cpp, "uvc_open failed: [Path: ..., Return
///         Code: ...]"; its libuvc, device.c, uvc_claim_if; libusb.h, LIBUSB_ERROR_BUSY = -6].
bool says_held(const orbbec_error &error)
{
    if (error.type == orbbec_exception::resource_busy)
        return true;
    return error.type == orbbec_exception::io &&
           error.message.find("uvc_open failed") != std::string::npos &&
           error.message.find("Return Code: -6") != std::string::npos;
}

/// Decides whether a link is slower than USB 3.
/// @param connection_type [in] the link as OrbbecSDK names it.
/// @return whether the name starts with "USB1" or "USB2" [S: UsbTypes.cpp, OrbbecSDK v2.10.6].
bool slower_than_usb_3(const std::string &connection_type)
{
    return connection_type.starts_with("USB1") || connection_type.starts_with("USB2");
}

/// A firmware version's numbers: major, minor, patch.
using firmware_numbers = std::array<std::uint32_t, 3>;

/// OrbbecSDK v2's least firmware for the Gemini 2 L [S: OrbbecSDK v2's README, web, read
/// 2026-11-15].
constexpr firmware_numbers least_firmware = {1, 4, 53};

/// Reads a firmware version as three numbers, as firmware_refusal() says.
/// @param version [in] the version as OrbbecSDK gives it.
/// @return its numbers major, minor, patch; nothing when it is not three parts separated by dots,
///         each one or more ASCII digits that fit in 32 bits.
std::optional<firmware_numbers> read_firmware(std::string_view version)
{
    firmware_numbers numbers{};
    for (std::size_t i = 0; i < numbers.size(); i++)
    {
        const bool last = i + 1 == numbers.size();
        const std::size_t end = last ? version.size() : version.find('.');
        if (end == std::string_view::npos)
            return std::nullopt;
        const char *const first = version.data();
        const char *const past = first + end;
        // from_chars takes no sign, space or prefix for an unsigned number; an empty part or one
        // beyond 32 bits is an error, and a part not read to its end has more than digits.
        const auto [stop, error] = std::from_chars(first, past, numbers[i]);
        if (error != std::errc{} || stop != past)
            return std::nullopt;
        if (!last)
            version.remove_prefix(end + 1);
    }
    return numbers;
}

/// Writes a firmware version's numbers for a message.
/// @param numbers [in] the numbers.
/// @return "<major>.<minor>.<patch>".
std::string firmware_text(const firmware_numbers &numbers)
{
    return std::to_string(numbers[0]) + "." + std::to_string(numbers[1]) + "." +
           std::to_string(numbers[2]);
}
} // namespace

std::optional<std::string> firmware_refusal(std::string_view version)
{
    const std::optional<firmware_numbers> numbers = read_firmware(version);
    // Compared part by part as numbers, major first: std::array's < is lexicographic.
    if (numbers && *numbers >= least_firmware)
        return std::nullopt;
    const std::string tail =
        firmware_text(least_firmware) +
        ", the least OrbbecSDK v2 supports for the Gemini 2 L; Orbbec's firmware tool updates it, "
        "on the user's word only, as it writes the camera's flash (AGENTS.md §3)";
    if (!numbers)
        return "the camera's firmware version \"" + std::string(version) +
               "\" is not three numbers major.minor.patch, so it cannot be compared with " + tail;
    return "the camera's firmware " + std::string(version) + " is older than " + tail;
}

std::optional<std::size_t> choose_color_profile(std::span<const orbbec_profile> profiles,
                                                const frame_size &size, std::uint32_t fps)
{
    for (const orbbec_format format :
         {orbbec_format::bgr, orbbec_format::rgb, orbbec_format::mjpg, orbbec_format::yuyv})
    {
        for (std::size_t i = 0; i < profiles.size(); i++)
        {
            const orbbec_profile &p = profiles[i];
            if (p.format == format && p.width == size.width && p.height == size.height &&
                p.fps == fps)
                return i;
        }
    }
    return std::nullopt;
}

std::optional<orbbec_depth_choice> choose_depth_profile(std::span<const orbbec_profile> hardware,
                                                        std::span<const orbbec_profile> all,
                                                        const frame_size &size, std::uint32_t fps)
{
    constexpr orbbec_format depth_formats[] = {orbbec_format::y16, orbbec_format::z16,
                                               orbbec_format::y14, orbbec_format::rle};
    for (const orbbec_format format : depth_formats)
    {
        for (std::size_t i = 0; i < hardware.size(); i++)
        {
            if (hardware[i].format == format && hardware[i].fps == fps)
                return orbbec_depth_choice{i, true};
        }
    }
    for (const orbbec_format format : depth_formats)
    {
        for (std::size_t i = 0; i < all.size(); i++)
        {
            const orbbec_profile &p = all[i];
            if (p.format == format && p.width == size.width && p.height == size.height &&
                p.fps == fps)
                return orbbec_depth_choice{i, false};
        }
    }
    return std::nullopt;
}

std::optional<std::string> opening_refusal(const orbbec_open_report &report, const frame_size &size,
                                           std::uint32_t fps)
{
    if (report.error)
    {
        if (says_held(*report.error))
            return "the camera is held by another program (OrbbecSDK: " + report.error->message +
                   ")";
        return "OrbbecSDK failed while opening the camera: " + report.error->message;
    }
    if (report.camera_count == 0)
        return "no Orbbec camera is connected";
    // The firmware's rule gives the reason here; orbbec_device::open() calls it too, to start no
    // stream for a firmware it refuses, so the firmware's reason comes before the streams'.
    if (std::optional<std::string> refusal = firmware_refusal(report.firmware_version))
        return refusal;
    if (report.color_stream_offered && report.depth_stream_offered)
        return std::nullopt;
    const std::string stream =
        (report.color_stream_offered ? "depth stream aligned to color" : "color stream") +
        std::string(" of ") + size_text(size) + " at " + std::to_string(fps) + " fps";
    if (slower_than_usb_3(report.connection_type))
        return "the camera's link is " + report.connection_type + ", which offers no " + stream +
               "; its USB 3 link did not come up (AGENTS.md §6)";
    return "the camera offers no " + stream;
}

void orbbec_frame_gate::start(const frame_size &size, const intrinsics &intrinsics) noexcept
{
    *this = orbbec_frame_gate{};
    m_size = size;
    m_intrinsics = intrinsics;
}

std::optional<std::string> orbbec_frame_gate::picture_refusal(const orbbec_frameset &frameset) const
{
    if (frameset.color_size != m_size)
        return "the color picture is " + size_text(frameset.color_size) +
               ", the opening's size is " + size_text(m_size);
    if (frameset.depth_size != m_size)
        return "the depth picture is " + size_text(frameset.depth_size) +
               ", the opening's size is " + size_text(m_size);
    const std::size_t pixels = std::size_t{m_size.width} * m_size.height;
    if (frameset.color_rgb.size() != 3 * pixels)
        return "the color picture holds " + std::to_string(frameset.color_rgb.size()) + " bytes, " +
               size_text(m_size) + " in RGB8 needs " + std::to_string(3 * pixels);
    if (!frameset.depth_16_bit)
        return "depth arrives as " + std::string(frameset.depth_format) +
               "; 16 bit per pixel (Y16 or Z16) is needed";
    if (frameset.depth.size() != pixels)
        return "the depth picture holds " + std::to_string(frameset.depth.size()) + " values, " +
               size_text(m_size) + " needs " + std::to_string(pixels);
    if (!std::isfinite(frameset.depth_scale_mm) || frameset.depth_scale_mm <= 0.0f)
        return "the depth unit, " + number_text(frameset.depth_scale_mm) +
               " mm per value, is not a positive number";
    if (m_has_taken && frameset.depth_scale_mm != m_depth_scale_mm)
        return "the depth unit changed within the opening, from " + number_text(m_depth_scale_mm) +
               " to " + number_text(frameset.depth_scale_mm) + " mm per value";
    return std::nullopt;
}

std::optional<std::string> orbbec_frame_gate::order_refusal(const orbbec_frameset &frameset,
                                                            std::int64_t &timestamp_us,
                                                            std::int64_t &offset_us) const
{
    if (m_has_taken && frameset.index <= m_last_index)
        return "the frame's index " + std::to_string(frameset.index) +
               " is not after the last taken frame's " + std::to_string(m_last_index);
    if (frameset.camera_time_us == 0 ||
        frameset.camera_time_us >= static_cast<std::uint64_t>(time_bound_us))
        return "the frame's timestamp on the camera's clock, " +
               std::to_string(frameset.camera_time_us) + " µs, is 0 or 2^60 µs or more";
    if (frameset.host_time_us >= static_cast<std::uint64_t>(time_bound_us))
        return "the host's time, " + std::to_string(frameset.host_time_us) +
               " µs, is 2^60 µs or more";

    // Every value below 2^60 in magnitude: the steps below 2^60, their difference below 2^61,
    // the jumps below 2^61 + 2^31, the offset below 2^60 (a taken timestamp and a camera timestamp
    // both lie in [0, 2^60)), so every sum stays below 2^63 [S: computed].
    const std::int64_t camera_us = static_cast<std::int64_t>(frameset.camera_time_us);
    offset_us = m_offset_us;
    if (m_has_taken)
    {
        const std::int64_t camera_step_us = camera_us - m_last_camera_us;
        const std::int64_t host_step_us =
            static_cast<std::int64_t>(frameset.host_time_us) - m_last_host_us;
        const clock_split split = split_steps(camera_step_us, host_step_us);
        if (beyond_tolerance(split.rest_us))
            return "the camera's clock stepped by " + std::to_string(camera_step_us) +
                   " µs while the host's stepped by " + std::to_string(host_step_us) +
                   " µs, which is no jump of 2^32 µs";
        offset_us -= split.jumps * jump_us;
    }
    timestamp_us = camera_us + offset_us;
    if (m_has_taken && timestamp_us <= m_last_timestamp_us)
        return "the frame's timestamp, " + std::to_string(timestamp_us) +
               " µs, is not after the last taken frame's, " + std::to_string(m_last_timestamp_us) +
               " µs";
    if (timestamp_us >= time_bound_us)
        return "the frame's timestamp, " + std::to_string(timestamp_us) + " µs, is 2^60 µs or more";
    return std::nullopt;
}

std::optional<next_result> orbbec_frame_gate::take(const orbbec_wait &wait, frame &out) noexcept
{
    switch (wait.status)
    {
    case orbbec_wait_status::timeout:
        return next_result{next_status::timeout, "no frame from the camera within " +
                                                     std::to_string(orbbec_frame_time_limit_ms) +
                                                     " ms"};
    case orbbec_wait_status::failure:
        return next_result{next_status::failure, wait.message};
    case orbbec_wait_status::frameset:
        break;
    }
    const orbbec_frameset &frameset = wait.frameset;
    if (const std::optional<std::string> refusal = picture_refusal(frameset))
        return next_result{next_status::failure, *refusal};
    std::int64_t timestamp_us = 0;
    std::int64_t offset_us = 0;
    if (std::optional<std::string> skip = order_refusal(frameset, timestamp_us, offset_us))
    {
        m_skips.count++;
        m_skips.last_reason = std::move(*skip);
        return std::nullopt;
    }

    out.prepare({m_size, m_intrinsics, frameset.depth_scale_mm * 0.001f}, frameset.index,
                static_cast<std::uint64_t>(timestamp_us));
    std::ranges::copy(frameset.color_rgb, out.color().begin());
    std::ranges::copy(frameset.depth, out.depth().begin());

    m_has_taken = true;
    m_depth_scale_mm = frameset.depth_scale_mm;
    m_last_index = frameset.index;
    m_last_camera_us = static_cast<std::int64_t>(frameset.camera_time_us);
    m_last_host_us = static_cast<std::int64_t>(frameset.host_time_us);
    m_last_timestamp_us = timestamp_us;
    m_offset_us = offset_us;
    return next_result{next_status::frame, {}};
}

const orbbec_skips &orbbec_frame_gate::skips() const noexcept
{
    return m_skips;
}

orbbec_placement orbbec_frame_gate::place(std::uint64_t camera_time_us, std::uint64_t host_time_us,
                                          std::int64_t &timestamp_us) const noexcept
{
    if (!m_has_taken)
        return orbbec_placement::no_frame_taken;
    if (camera_time_us == 0 || camera_time_us >= static_cast<std::uint64_t>(time_bound_us))
        return orbbec_placement::camera_time_out_of_range;
    if (host_time_us >= static_cast<std::uint64_t>(time_bound_us))
        return orbbec_placement::host_time_out_of_range;
    // Every value below 2^60 in magnitude as in order_refusal(); the frame's delivered timestamp
    // lies in [0, 2^60), so the sum below stays within +-2^62 [S: computed].
    const std::int64_t camera_step_us =
        static_cast<std::int64_t>(camera_time_us) - m_last_camera_us;
    const std::int64_t host_step_us = static_cast<std::int64_t>(host_time_us) - m_last_host_us;
    const clock_split split = split_steps(camera_step_us, host_step_us);
    if (beyond_tolerance(split.rest_us))
        return orbbec_placement::off_clock;
    timestamp_us = m_last_timestamp_us + camera_step_us - split.jumps * jump_us;
    return orbbec_placement::placed;
}

std::string orbbec_frame_gate::placement_refusal(std::uint64_t camera_time_us,
                                                 std::uint64_t host_time_us) const
{
    std::int64_t timestamp_us = 0;
    switch (place(camera_time_us, host_time_us, timestamp_us))
    {
    case orbbec_placement::placed:
        return {};
    case orbbec_placement::no_frame_taken:
        return "no frame of this opening has been taken to place it against";
    case orbbec_placement::camera_time_out_of_range:
        return "its time on the camera's clock, " + std::to_string(camera_time_us) +
               " µs, is 0 or 2^60 µs or more";
    case orbbec_placement::host_time_out_of_range:
        return "the host's time, " + std::to_string(host_time_us) + " µs, is 2^60 µs or more";
    case orbbec_placement::off_clock:
        break;
    }
    return "its time on the camera's clock stepped by " +
           std::to_string(static_cast<std::int64_t>(camera_time_us) - m_last_camera_us) +
           " µs from the last taken frame's while the host's stepped by " +
           std::to_string(static_cast<std::int64_t>(host_time_us) - m_last_host_us) +
           " µs, which is no jump of 2^32 µs";
}
} // namespace gca
