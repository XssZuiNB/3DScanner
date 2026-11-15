// SPDX-License-Identifier: 0BSD
/// @file color_hold.hpp
/// The color camera's exposure and white balance held for a scan (R13; spec, "The camera seam",
/// run-time properties), as a frame source in front of the camera. When the camera has opened, it
/// reads the five properties of camera/camera_properties.hpp and switches both automatic controls
/// on, the camera's defaults [M 2026-11-13, docs/HARDWARE.md]. After the color_settle_frames-th
/// frame the camera delivers it reads the values the controls reached and switches each control
/// off, so that the camera holds them, and reads the five again. At the close, and when it is
/// destroyed open, it switches both controls on again, whatever happened before, and reads the
/// five once more. Only the two switches are written (AGENTS.md §3); nothing is written to the
/// camera's flash [?: unverified on the camera, AGENTS.md §3]. A property the camera refuses does
/// not stop the frames, and the report names it: a control is switched off only when the camera
/// took every read and write of its properties until then, so a refused read, or a refused switch
/// off, leaves the control automatic, and a refused switch on at the start leaves it as the camera
/// had it. A read refused while held or at the end only goes into the report: the control stays
/// as it is. Only delivered frames count; a timeout, a failure and the end count none. The frames
/// pass through unchanged. orbbec_viewer --hold-color prints the report's lines; it builds without
/// OrbbecSDK, and tests/test_color_hold.cpp drives it.

#pragma once

#include "camera/camera_properties.hpp"
#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "types/frame_description.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gca
{
/// How many delivered frames of an opening the automatic controls get to settle before they are
/// held [?: initial, the spec's 30; `first-trial-of-held-exposure` measures it].
constexpr std::uint64_t color_settle_frames = 30;

/// When a reading of the five properties was taken.
enum class color_moment
{
    at_start,    ///< at the opening, before the controls are switched on
    before_hold, ///< after the settling, before the controls are switched off
    while_held,  ///< right after the controls were switched off
    after,       ///< at the close, after the controls were switched on again
};

/// The step a refusal came in.
enum class color_step
{
    start, ///< the opening's reading and the switches on
    hold,  ///< the readings around the switches off
    end,   ///< the switches on again and the last reading
};

/// One reading of the five properties.
struct color_reading
{
    color_moment moment = color_moment::at_start; ///< when it was taken
    /// Each property's value, in the order of color_property_table; nothing where the read was
    /// refused.
    std::array<std::optional<std::int32_t>, color_property_count> values;
};

/// A read or a write the camera refused.
struct color_refusal
{
    color_property property = color_property::auto_exposure; ///< the property
    color_step step = color_step::start;                     ///< the step it came in
    /// The value written, a switch's 0 or 1; nothing for a read.
    std::optional<std::int32_t> written;
    std::string reason; ///< the camera's reason
};

/// What the hold did in the current or the last opening, in the order it happened.
struct color_hold_report
{
    std::vector<color_reading> readings; ///< the readings taken
    std::vector<color_refusal> refusals; ///< the reads and writes the camera refused
};

/// Builds a reading's line.
/// @param reading [in] the reading.
/// @return "Color <moment>: auto exposure <on|off>, exposure <n>, gain <n>, auto white balance
///         <on|off>, white balance <n>", the moment "at the start", "before the hold", "while
///         held" or "after", a switch on for every value but 0, a refused value "unread".
/// @note Allocates the line. Two threads may call it at once: it reads only its argument and
///       constants.
std::string reading_line(const color_reading &reading);

/// Builds a refusal's line.
/// @param refusal [in] the refusal.
/// @return "Color: the camera refused to read <property> (<number>) <step>: <reason>", or "...
///         refused to switch <property> (<number>) <on|off> <step>: <reason>", the step "at the
///         start", "at the hold" or "at the end".
/// @note Allocates the line. Two threads may call it at once: it reads only its argument and
///       constants.
std::string refusal_line(const color_refusal &refusal);

/// A frame source that holds the color camera's exposure and white balance while it is open. One
/// thread at a time, as the frame source seam says: its calls run the camera's frame and property
/// calls on the calling thread.
class color_hold final : public frame_source
{
public:
    /// Puts the hold in front of a camera; nothing is read or written until open().
    /// @param camera [in,out] the camera's frames; it must outlive the hold, and only the hold
    ///        opens and closes it while the hold is used.
    /// @param properties [in,out] the same camera's properties; it must outlive the hold.
    color_hold(frame_source &camera, camera_properties &properties) noexcept;

    /// Closes the hold, and with it the camera, when it is open, so that the controls are
    /// switched on again also when the program ends without a close, an exception unwinding the
    /// stack. An exception inside the hold's or the camera's calls, which are noexcept, ends the
    /// program through std::terminate instead, and nothing switches the controls on then.
    ~color_hold() override;

    /// Opens the camera; when it opened, starts a new report, reads the five properties and
    /// switches both automatic controls on.
    /// @param size [in] the stream size, in px.
    /// @return the camera's result; a refused opening reads and writes nothing.
    /// @note Allocates the report's entries and what the camera's open() allocates. One thread at
    ///       a time.
    open_result open(const frame_size &size) noexcept override;

    /// Takes the camera's next frame; after the color_settle_frames-th frame delivered since the
    /// opening it reads the five properties, switches off each control whose properties the
    /// camera took every read and write of so far, and reads the five again.
    /// @param out [out] as the camera's next() fills it.
    /// @return the camera's result, unchanged; a timeout, a failure or the end counts no frame.
    /// @note Allocates the report's entries at the hold, and what the camera's next() allocates.
    ///       One thread at a time.
    next_result next(frame &out) noexcept override;

    /// Switches both automatic controls on and reads the five properties, when the hold is open,
    /// then closes the camera.
    /// @note Allocates the report's entries. One thread at a time.
    void close() noexcept override;

    /// Describes the camera.
    /// @return the camera's description.
    /// @note Allocates for the strings. One thread at a time, the thread of the other calls.
    source_description describe() const noexcept override;

    /// Tells what the hold read and the camera refused.
    /// @return the report of the current or the last opening, valid as long as the hold; open()
    ///         starts it anew, next() and close() add to it.
    /// @note One thread at a time, the thread of the other calls.
    const color_hold_report &report() const noexcept;

private:
    /// Reads the five properties into the report, a refused read into its refusals.
    /// @param moment [in] when.
    /// @param step [in] the step, for a refusal.
    void read_all(color_moment moment, color_step step);

    /// Writes a control's switch, the switch color_property_table gives it, a refused write into
    /// the report's refusals.
    /// @param control [in] the control.
    /// @param on [in] whether to switch it on.
    /// @param step [in] the step, for a refusal.
    void switch_control(color_control control, bool on, color_step step);

    /// Records a refusal; one at the start or the hold marks its control as not to be held.
    /// @param refusal [in] the refusal; taken over.
    void refuse(color_refusal refusal);

    frame_source &m_camera;
    camera_properties &m_properties;
    bool m_open = false;        ///< whether the hold opened the camera and did not close it
    std::uint64_t m_frames = 0; ///< the frames delivered since the opening
    /// Whether the camera took every read and write of each control's properties so far in this
    /// opening, before its switch-off.
    std::array<bool, color_control_count> m_accepted = {true, true};
    color_hold_report m_report;
};
} // namespace gca
