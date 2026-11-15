// SPDX-License-Identifier: 0BSD
/// @file color_hold.cpp
/// The color camera's exposure and white balance held for a scan: the steps at the opening, after
/// the settling and at the close, the rule of which control is held, and the report's lines (see
/// color_hold.hpp).

#include "color_hold.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace gca
{
namespace
{
/// The controls, in the order of their indices.
constexpr std::array<color_control, color_control_count> color_controls = {
    color_control::exposure, color_control::white_balance};

/// Finds a control's switch in color_property_table.
/// @param control [in] the control.
/// @return its switch; auto_exposure for a control without one, which the static_assert below
///         rules out.
constexpr color_property switch_of(color_control control) noexcept
{
    for (const color_property_row &row : color_property_table)
    {
        if (row.is_switch && row.control == control)
            return row.property;
    }
    return color_property::auto_exposure;
}

// Each control has exactly one switch in the table, the one switch_of() gives.
static_assert(
    [] {
        for (const color_control control : color_controls)
        {
            std::size_t switches = 0;
            for (const color_property_row &row : color_property_table)
                switches += row.is_switch && row.control == control ? 1 : 0;
            if (switches != 1)
                return false;
        }
        return true;
    }(),
    "each control has one switch in color_property_table");

/// Gives a control's index into the hold's array of accepted controls.
/// @param control [in] the control.
/// @return 0 for exposure, 1 for white balance.
std::size_t index_of(color_control control) noexcept
{
    return static_cast<std::size_t>(control);
}

/// Names a moment for a reading's line.
/// @param moment [in] the moment.
/// @return "at the start", "before the hold", "while held" or "after"; a static string.
const char *moment_name(color_moment moment) noexcept
{
    switch (moment)
    {
    case color_moment::at_start:
        return "at the start";
    case color_moment::before_hold:
        return "before the hold";
    case color_moment::while_held:
        return "while held";
    case color_moment::after:
        return "after";
    }
    return "at an unknown moment";
}

/// Names a step for a refusal's line.
/// @param step [in] the step.
/// @return "at the start", "at the hold" or "at the end"; a static string.
const char *step_name(color_step step) noexcept
{
    switch (step)
    {
    case color_step::start:
        return "at the start";
    case color_step::hold:
        return "at the hold";
    case color_step::end:
        return "at the end";
    }
    return "at an unknown step";
}
} // namespace

std::string reading_line(const color_reading &reading)
{
    std::string line = std::string("Color ") + moment_name(reading.moment) + ":";
    for (std::size_t i = 0; i < color_property_count; i++)
    {
        const color_property_row &row = color_property_table[i];
        const std::optional<std::int32_t> &value = reading.values[i];
        line += std::string(i == 0 ? " " : ", ") + row.name + " ";
        if (!value)
            line += "unread";
        else if (row.is_switch)
            line += *value != 0 ? "on" : "off";
        else
            line += std::to_string(*value);
    }
    return line;
}

std::string refusal_line(const color_refusal &refusal)
{
    const std::string property = std::string(property_name(refusal.property)) + " (" +
                                 std::to_string(static_cast<std::uint32_t>(refusal.property)) + ")";
    const std::string action = refusal.written
                                   ? "switch " + property + (*refusal.written != 0 ? " on" : " off")
                                   : "read " + property;
    return "Color: the camera refused to " + action + " " + step_name(refusal.step) + ": " +
           refusal.reason;
}

color_hold::color_hold(frame_source &camera, camera_properties &properties) noexcept
    : m_camera(camera)
    , m_properties(properties)
{
}

color_hold::~color_hold()
{
    close();
}

open_result color_hold::open(const frame_size &size) noexcept
{
    const open_result opened = m_camera.open(size);
    if (!opened.opened)
        return opened;
    m_open = true;
    m_frames = 0;
    m_accepted = {true, true};
    m_report = {};
    read_all(color_moment::at_start, color_step::start);
    for (const color_control control : color_controls)
        switch_control(control, true, color_step::start);
    return opened;
}

next_result color_hold::next(frame &out) noexcept
{
    const next_result result = m_camera.next(out);
    if (!m_open || result.status != next_status::frame)
        return result;
    m_frames++;
    if (m_frames == color_settle_frames)
    {
        read_all(color_moment::before_hold, color_step::hold);
        for (const color_control control : color_controls)
        {
            // A control is switched off only when the camera took every read and write of its
            // properties so far, the reading just taken included.
            if (m_accepted[index_of(control)])
                switch_control(control, false, color_step::hold);
        }
        read_all(color_moment::while_held, color_step::hold);
    }
    return result;
}

void color_hold::close() noexcept
{
    if (m_open)
    {
        // Both, whatever happened before: a control the camera refused to switch off, or one it
        // refused to switch on at the start, is written on again too.
        for (const color_control control : color_controls)
            switch_control(control, true, color_step::end);
        read_all(color_moment::after, color_step::end);
        m_open = false;
    }
    m_camera.close();
}

source_description color_hold::describe() const noexcept
{
    return m_camera.describe();
}

const color_hold_report &color_hold::report() const noexcept
{
    return m_report;
}

void color_hold::read_all(color_moment moment, color_step step)
{
    color_reading reading;
    reading.moment = moment;
    for (std::size_t i = 0; i < color_property_count; i++)
    {
        const color_property property = color_property_table[i].property;
        property_reading read = m_properties.read_property(property);
        if (read.value)
            reading.values[i] = read.value;
        else
            refuse({property, step, std::nullopt, std::move(read.refusal)});
    }
    m_report.readings.push_back(std::move(reading));
}

void color_hold::switch_control(color_control control, bool on, color_step step)
{
    const color_property property = switch_of(control);
    const std::int32_t value = on ? 1 : 0;
    std::optional<std::string> refusal = m_properties.write_property(property, value);
    if (refusal)
        refuse({property, step, value, std::move(*refusal)});
}

void color_hold::refuse(color_refusal refusal)
{
    const color_property_row *row = row_of(refusal.property);
    if (row != nullptr && refusal.step != color_step::end)
        m_accepted[index_of(row->control)] = false;
    m_report.refusals.push_back(std::move(refusal));
}
} // namespace gca
