// SPDX-License-Identifier: 0BSD
/// @file stand_in_camera.hpp
/// The tests' stand-in for the camera's run-time properties (camera/camera_properties.hpp): a
/// frame source in front of another, the synthetic source or one in front of it, whose properties
/// are a table under OrbbecSDK's numbers 2000 to 2004, written here as numbers. The table's
/// automatic controls move their values with every frame delivered while their switch is on, as
/// the camera's do. Every access is logged; one while closed, of a number not in the table, a
/// write of a property that is not writable and one the table refuses are refused with their
/// reasons. The hold's test (test_color_hold.cpp) and the service's (test_scanner_service.cpp)
/// stand it in for the Orbbec camera. One thread at a time calls it, as the frame source seam says,
/// and a test reads and sets the table and the log directly only while no other thread uses the
/// stand-in; set_refuse_reads() alone may be called while another thread uses it, as its calls
/// take the stand-in's lock. Header only; each test program is one translation unit.

#pragma once

#include "camera/camera_properties.hpp"
#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "types/frame_description.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace gca_test
{
/// One property of the stand-in's table.
struct table_entry
{
    std::int32_t value;         ///< its value now
    bool writable;              ///< whether a write is taken (only the two switches)
    bool refuse_reads = false;  ///< whether every read is refused
    bool refuse_writes = false; ///< whether every write is refused
};

/// One access of the stand-in's properties, in the order they came.
struct property_access
{
    std::uint32_t number;              ///< the property's number
    std::optional<std::int32_t> write; ///< the value written; nothing for a read
};

/// A camera whose frames are another source's and whose properties are a table: the automatic
/// exposure (2000) moves exposure (2001) by 100 and gain (2002) by 3 with every frame delivered
/// while it is on, the automatic white balance (2003) white balance (2004) by 50.
class stand_in_camera final : public gca::frame_source, public gca::camera_properties
{
public:
    /// Builds the camera, closed, its table at its starting values, both switches off.
    /// @param frames [in,out] the source of its frames, closed; owned by the caller, which keeps
    ///        it alive as long as the stand-in.
    explicit stand_in_camera(gca::frame_source &frames)
        : m_frames(frames)
    {
    }

    /// Opens the source of the frames.
    /// @param size [in] the size asked for.
    /// @return the source's result.
    gca::open_result open(const gca::frame_size &size) noexcept override
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        const gca::open_result opened = m_frames.open(size);
        m_open = m_open || opened.opened;
        return opened;
    }

    /// Gives the source's next frame and, with a frame, moves the values of each control whose
    /// switch is on.
    /// @param out [out] as the source fills it.
    /// @return the source's result.
    gca::next_result next(gca::frame &out) noexcept override
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        const gca::next_result result = m_frames.next(out);
        if (result.status == gca::next_status::frame)
        {
            if (table[2000].value != 0)
            {
                table[2001].value += 100;
                table[2002].value += 3;
            }
            if (table[2003].value != 0)
                table[2004].value += 50;
        }
        return result;
    }

    /// Closes the source of the frames.
    void close() noexcept override
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        m_frames.close();
        m_open = false;
    }

    /// Describes the stand-in.
    /// @return "stand-in camera", firmware "9.9.99", SDK "0.0.0".
    gca::source_description describe() const noexcept override
    {
        return {"stand-in camera", "9.9.99", "0.0.0"};
    }

    /// Reads a property from the table and logs the read.
    /// @param property [in] the property.
    /// @return its value; or nothing with the reason: closed, no such number, reads refused.
    gca::property_reading read_property(gca::color_property property) noexcept override
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        const auto number = static_cast<std::uint32_t>(property);
        log.push_back({number, std::nullopt});
        if (const std::optional<std::string> reason = refusal_of(number))
            return {std::nullopt, *reason};
        if (table.at(number).refuse_reads)
            return {std::nullopt, "the stand-in refuses reads of " + std::to_string(number)};
        return {table.at(number).value, {}};
    }

    /// Writes a property into the table and logs the write.
    /// @param property [in] the property.
    /// @param value [in] the value.
    /// @return nothing when taken; otherwise the reason: closed, no such number, read only, writes
    ///         refused.
    std::optional<std::string> write_property(gca::color_property property,
                                              std::int32_t value) noexcept override
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        const auto number = static_cast<std::uint32_t>(property);
        log.push_back({number, value});
        if (const std::optional<std::string> reason = refusal_of(number))
            return reason;
        table_entry &entry = table.at(number);
        if (!entry.writable)
            return "the stand-in's " + std::to_string(number) + " is read only";
        if (entry.refuse_writes)
            return "the stand-in refuses writes of " + std::to_string(number);
        entry.value = value;
        return std::nullopt;
    }

    /// Sets whether every read of a property is refused, from any thread, between two of its
    /// openings.
    /// @param number [in] the property's number, one of the table's.
    /// @param refuse [in] whether to refuse its reads.
    void set_refuse_reads(std::uint32_t number, bool refuse)
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        table.at(number).refuse_reads = refuse;
    }

    /// Tells whether the stand-in is open.
    /// @return whether an opening took and no close followed.
    bool is_open() const noexcept
    {
        return m_open;
    }

    /// The properties under OrbbecSDK's numbers [S: OrbbecSDK v2.10.6, Property.h], at values of
    /// the camera's ranges [M 2026-11-13, docs/HARDWARE.md], both switches off.
    std::map<std::uint32_t, table_entry> table = {
        {2000, {0, true}}, {2001, {5000, false}}, {2002, {64, false}},
        {2003, {0, true}}, {2004, {4000, false}},
    };
    std::vector<property_access> log;   ///< every access, in order
    std::vector<std::uint32_t> refused; ///< the numbers refused for being closed or unknown

private:
    /// Refuses an access while closed or of a number not in the table, and records it.
    /// @param number [in] the property's number.
    /// @return the reason, or nothing when the access may go on.
    std::optional<std::string> refusal_of(std::uint32_t number)
    {
        if (!m_open || table.count(number) == 0)
        {
            refused.push_back(number);
            return m_open ? "the stand-in has no property " + std::to_string(number)
                          : std::string("the stand-in is not open");
        }
        return std::nullopt;
    }

    std::mutex
        m_mutex; ///< guards the table, the log and the source's calls against set_refuse_reads()
    gca::frame_source &m_frames;
    bool m_open = false;
};
} // namespace gca_test
