// SPDX-License-Identifier: 0BSD
/// @file camera_properties.hpp
/// The seam of the camera's run-time properties (R13): the color camera's properties a scan may
/// touch, under OrbbecSDK's numbers, and the interface through which they are read and written,
/// a refusal coming back as a value. Of these only the two automatic controls' switches are ever
/// written, AGENTS.md §3's list; the values are only read. The Orbbec camera implements the
/// interface (camera/orbbec_device.hpp), the tests a table (tests/test_color_hold.cpp), and
/// camera/color_hold.hpp holds the steps that hold exposure and white balance for a scan. This
/// header builds without OrbbecSDK (src/CMakeLists.txt, the library camera).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace gca
{
/// The color camera's properties a scan touches, each under its number in OrbbecSDK's
/// OBPropertyID [S: OrbbecSDK v2.10.6, libobsensor/h/Property.h]; camera/orbbec_device.cpp checks
/// each number against OrbbecSDK's name when it builds, and color_property_table below says what
/// each is.
enum class color_property : std::uint32_t
{
    auto_exposure = 2000,      ///< OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
    exposure = 2001,           ///< OB_PROP_COLOR_EXPOSURE_INT
    gain = 2002,               ///< OB_PROP_COLOR_GAIN_INT
    auto_white_balance = 2003, ///< OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL
    white_balance = 2004,      ///< OB_PROP_COLOR_WHITE_BALANCE_INT
};

/// The color camera's two automatic controls, each a switch and the values it sets
/// (color_property_table says which).
enum class color_control : std::size_t
{
    exposure = 0,      ///< the automatic exposure
    white_balance = 1, ///< the automatic white balance
};

/// The number of the automatic controls.
inline constexpr std::size_t color_control_count = 2;

/// One property in the table of the properties.
struct color_property_row
{
    color_property property; ///< the property
    const char *name;        ///< its name for the console's and a scan's lines, a static string
    /// Whether it is a switch, one of OrbbecSDK's _BOOL properties, 0 (off) or 1 (on): the one
    /// rule of what may be written, AGENTS.md §3's list; every other property is only read.
    bool is_switch;
    color_control control; ///< the automatic control it belongs to
};

/// The one table of the properties: their names, which of them are switches and which control
/// each belongs to, in the order of their numbers, which is the order a reading lists them in.
inline constexpr std::array color_property_table = std::to_array<color_property_row>({
    {color_property::auto_exposure, "auto exposure", true, color_control::exposure},
    {color_property::exposure, "exposure", false, color_control::exposure},
    {color_property::gain, "gain", false, color_control::exposure},
    {color_property::auto_white_balance, "auto white balance", true, color_control::white_balance},
    {color_property::white_balance, "white balance", false, color_control::white_balance},
});

/// The number of the properties, the length of a reading of all of them.
inline constexpr std::size_t color_property_count = color_property_table.size();

// Row i holds the property numbered 2000 + i and the last row white balance, so that no property
// is missing, doubled or out of order.
static_assert(
    [] {
        for (std::size_t i = 0; i < color_property_count; i++)
        {
            if (static_cast<std::size_t>(color_property_table[i].property) !=
                static_cast<std::size_t>(color_property::auto_exposure) + i)
                return false;
        }
        return color_property_table.back().property == color_property::white_balance;
    }(),
    "color_property_table lists 2000 to 2004 once each, in the order of their numbers");

/// Finds a property's row in the table.
/// @param property [in] the property.
/// @return its row, in the table, which lives as long as the program; nullptr for a value that
///         names none of the five.
constexpr const color_property_row *row_of(color_property property) noexcept
{
    for (const color_property_row &row : color_property_table)
    {
        if (row.property == property)
            return &row;
    }
    return nullptr;
}

/// Tells whether a property is a switch, read and written as 0 (off) or 1 (on), and so whether
/// it may be written (color_property_row::is_switch).
/// @param property [in] the property.
/// @return true for auto_exposure and auto_white_balance; false for the others and for a value
///         that names none of the five.
constexpr bool is_switch(color_property property) noexcept
{
    const color_property_row *row = row_of(property);
    return row != nullptr && row->is_switch;
}

/// Names a property for the console's and a scan's lines.
/// @param property [in] the property.
/// @return "auto exposure", "exposure", "gain", "auto white balance" or "white balance"; "an
///         unknown property" for a value that names none of the five; a static string.
constexpr const char *property_name(color_property property) noexcept
{
    const color_property_row *row = row_of(property);
    return row != nullptr ? row->name : "an unknown property";
}

/// What a read of a property gave: its value, or the reason the camera refused it.
struct property_reading
{
    /// The value in the camera's own unit (OrbbecSDK's integer; a switch 0 or 1); nothing when
    /// the read was refused.
    std::optional<std::int32_t> value;
    std::string refusal; ///< why the read was refused, in plain English; empty with a value
};

/// The seam through which the camera's properties are read and written. One thread at a time.
class camera_properties
{
public:
    /// Builds the seam's part, which holds nothing.
    camera_properties() = default;

    /// The properties stand for one camera; they are not copied.
    camera_properties(const camera_properties &) = delete;
    camera_properties &operator=(const camera_properties &) = delete;

    /// Ends the seam's part.
    virtual ~camera_properties() = default;

    /// Reads a property from the camera.
    /// @param property [in] the property.
    /// @return its value; or nothing with the reason: the camera is not open, or it refused.
    /// @note Allocates for a refusal's reason; an implementation may allocate inside a vendor's
    ///       SDK. A failed allocation ends the program (std::terminate): the function is noexcept.
    virtual property_reading read_property(color_property property) noexcept = 0;

    /// Writes a switch of the camera, at run time: the camera keeps it until it is written again
    /// or loses power [?: for 2000 and 2003 unverified on the camera, AGENTS.md §3]. Only a switch
    /// (is_switch()) is written, AGENTS.md §3's list.
    /// @param property [in] the property, a switch.
    /// @param value [in] 0 (off) or 1 (on).
    /// @return nothing when the camera took it; otherwise the reason: the property is no switch,
    ///         the camera is not open, or it refused.
    /// @note Allocates for a refusal's reason; an implementation may allocate inside a vendor's
    ///       SDK. A failed allocation ends the program (std::terminate): the function is noexcept.
    virtual std::optional<std::string> write_property(color_property property,
                                                      std::int32_t value) noexcept = 0;
};
} // namespace gca
