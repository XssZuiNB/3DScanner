// SPDX-License-Identifier: 0BSD
/// @file test_color_hold.cpp
/// The hold of the color camera's exposure and white balance (camera/color_hold.hpp), in front of
/// a stand-in camera whose frames are the synthetic source's and whose properties are a table
/// under OrbbecSDK's numbers 2000 to 2004, written here as numbers. The table's automatic
/// controls move their values with every frame delivered while their switch is on, as the
/// camera's do, and both switches start off, so that switching them on shows:
/// - at the opening both controls are switched on; after the 29th frame they are still on, after
///   the 30th both are off and their values those of the 30th frame, which stay; a second opening
///   holds again after its own 30th frame;
/// - after each ending both controls are on again: a close after the frames (a stop, or the
///   program asked to stop, which ends the viewer's loop), a close after the camera failed while
///   held or before the settling, and the hold destroyed open, as an exception unwinding the
///   program destroys it;
/// - only 2000 and 2003 are written, and no other property than the five is touched, none while
///   the camera is closed (the table refuses any other, and the test sees none);
/// - each of the five refused in turn: the frames go on, its control stays automatic and the other
///   is held, and the report names that property and no other; a refused switch-off leaves its
///   control automatic, the other held;
/// - only delivered frames count: after frames and then timeouts, next() called on, neither
///   control is switched off before the 30th frame;
/// - the report's readings and refusals as lines, against lines written here;
/// - the rule of what may be written, is_switch(), holds for 2000 and 2003 and no other number.
/// Needs no GPU and no camera.

#include "camera/camera_properties.hpp"
#include "camera/color_hold.hpp"
#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "check.hpp"
#include "synthetic_source.hpp"
#include "types/frame_description.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;

/// The frames' size: small, as the hold does not look at the pixels [?: a choice].
constexpr gca::frame_size test_size = {32, 24};

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

/// A camera whose frames are the synthetic source's and whose properties are a table: the
/// automatic exposure (2000) moves exposure (2001) by 100 and gain (2002) by 3 with every frame
/// delivered while it is on, the automatic white balance (2003) white balance (2004) by 50. Every
/// access is logged; one while closed, of a number not in the table, a write of a property that
/// is not writable and a refused one are refused with their reasons.
class stand_in_camera final : public gca::frame_source, public gca::camera_properties
{
public:
    /// Builds the camera, closed, its table at its starting values.
    /// @param frames [in] the frames delivered before the ending.
    /// @param ending [in] what next() reports after them.
    stand_in_camera(std::uint64_t frames, gca_test::synthetic_ending ending)
        : m_frames(options(frames, ending))
    {
    }

    /// Opens the synthetic source.
    /// @param size [in] the size asked for, test_size.
    /// @return the synthetic source's result.
    gca::open_result open(const gca::frame_size &size) noexcept override
    {
        const gca::open_result opened = m_frames.open(size);
        m_open = m_open || opened.opened;
        return opened;
    }

    /// Gives the synthetic source's next frame and, with a frame, moves the values of each control
    /// whose switch is on.
    /// @param out [out] as the synthetic source fills it.
    /// @return the synthetic source's result.
    gca::next_result next(gca::frame &out) noexcept override
    {
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

    /// Closes the synthetic source.
    void close() noexcept override
    {
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

    /// The properties under OrbbecSDK's numbers [S: OrbbecSDK v2.10.6, Property.h], at values of
    /// the camera's ranges [M 2026-11-13, docs/HARDWARE.md], both switches off.
    std::map<std::uint32_t, table_entry> table = {
        {2000, {0, true}}, {2001, {5000, false}}, {2002, {64, false}},
        {2003, {0, true}}, {2004, {4000, false}},
    };
    std::vector<property_access> log;   ///< every access, in order
    std::vector<std::uint32_t> refused; ///< the numbers refused for being closed or unknown

private:
    /// Builds the synthetic source's options: frames of nothing, at the identity pose.
    /// @param frames [in] the frames before the ending.
    /// @param ending [in] the ending.
    /// @return the options.
    static gca_test::synthetic_source_options options(std::uint64_t frames,
                                                      gca_test::synthetic_ending ending)
    {
        gca_test::synthetic_source_options o;
        o.description = {test_size, {30.0f, 30.0f, 15.5f, 11.5f}, 0.001f};
        o.scene.trajectory.assign(frames, Eigen::Isometry3d::Identity());
        o.ending = ending;
        o.ending_message = "the stand-in's frames ended";
        return o;
    }

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

    gca_test::synthetic_source m_frames;
    bool m_open = false;
};

/// What a case expects of the stand-in's table.
struct table_state
{
    bool exposure_on;           ///< auto exposure (2000) on
    bool white_balance_on;      ///< auto white balance (2003) on
    std::int32_t exposure;      ///< exposure (2001)
    std::int32_t gain;          ///< gain (2002)
    std::int32_t white_balance; ///< white balance (2004)
};

/// Takes frames until a count is reached or the camera gives no frame.
/// @param hold [in,out] the hold.
/// @param count [in] the frames to take.
/// @return the frames taken.
std::uint64_t take_frames(gca::color_hold &hold, std::uint64_t count)
{
    gca::frame frame;
    std::uint64_t taken = 0;
    while (taken < count && hold.next(frame).status == gca::next_status::frame)
        taken++;
    return taken;
}

/// Checks the stand-in's switches and values.
/// @param camera [in] the stand-in.
/// @param what [in] the case, for the check line.
/// @param expected [in] the table expected.
void check_table(const stand_in_camera &camera, const std::string &what,
                 const table_state &expected)
{
    const auto &t = camera.table;
    std::printf("%s: switches %d %d, exposure %d, gain %d, white balance %d\n", what.c_str(),
                t.at(2000).value, t.at(2003).value, t.at(2001).value, t.at(2002).value,
                t.at(2004).value);
    check(t.at(2000).value == (expected.exposure_on ? 1 : 0) &&
              t.at(2003).value == (expected.white_balance_on ? 1 : 0),
          (what + ": the switches").c_str());
    check(t.at(2001).value == expected.exposure && t.at(2002).value == expected.gain &&
              t.at(2004).value == expected.white_balance,
          (what + ": the values").c_str());
}

/// Checks that both switches are on.
/// @param camera [in] the stand-in.
/// @param what [in] the case, for the check line.
void check_both_on(const stand_in_camera &camera, const std::string &what)
{
    std::printf("%s: switches %d %d\n", what.c_str(), camera.table.at(2000).value,
                camera.table.at(2003).value);
    check(camera.table.at(2000).value == 1 && camera.table.at(2003).value == 1,
          (what + ": both switches on").c_str());
}

/// Checks that only the switches 2000 and 2003 were written and every number touched lies in the
/// table, none of them while closed.
/// @param camera [in] the stand-in.
/// @param what [in] the case, for the check line.
void check_only_listed_touched(const stand_in_camera &camera, const std::string &what)
{
    bool listed = camera.refused.empty();
    for (const property_access &a : camera.log)
        listed = listed && (a.write ? a.number == 2000 || a.number == 2003
                                    : a.number >= 2000 && a.number <= 2004);
    check(listed, (what + ": only 2000 and 2003 written, nothing outside 2000 to 2004 read, "
                          "nothing touched while closed")
                      .c_str());
}

/// Tells whether a switch was ever written off.
/// @param camera [in] the stand-in.
/// @param number [in] the switch's number.
/// @return whether the log holds a write of 0 to it.
bool written_off(const stand_in_camera &camera, std::uint32_t number)
{
    for (const property_access &a : camera.log)
    {
        if (a.number == number && a.write && *a.write == 0)
            return true;
    }
    return false;
}

/// Checks a line against the one expected and prints it.
/// @param what [in] the case, for the check line.
/// @param line [in] the line built.
/// @param expected [in] the line expected.
void check_line(const std::string &what, const std::string &line, const std::string &expected)
{
    std::printf("%s: %s\n", what.c_str(), line.c_str());
    check(line == expected, (what + ": the line").c_str());
}

/// The hold after the settling: on after 29 frames and moving, off after the 30th with frame
/// 30's values, which stay; a close switches both on again, and the four readings say so; a
/// second opening holds after its own 30th frame.
void test_hold_after_settling()
{
    stand_in_camera camera(40, gca_test::synthetic_ending::end);
    gca::color_hold hold(camera, camera);
    check(hold.open(test_size).opened, "settling: opened");
    check_table(camera, "settling: at the opening", {true, true, 5000, 64, 4000});
    check(take_frames(hold, 29) == 29, "settling: 29 frames");
    check_table(camera, "settling: after 29 frames", {true, true, 7900, 151, 5450});
    check(take_frames(hold, 1) == 1, "settling: the 30th frame");
    check_table(camera, "settling: held after the 30th frame", {false, false, 8000, 154, 5500});
    check(take_frames(hold, 10) == 10, "settling: 10 frames more");
    check_table(camera, "settling: still held after 40 frames", {false, false, 8000, 154, 5500});
    hold.close();
    check_table(camera, "settling: after the close", {true, true, 8000, 154, 5500});
    check_only_listed_touched(camera, "settling");
    check(hold.describe().camera == "stand-in camera", "settling: the camera's description");

    const gca::color_hold_report &report = hold.report();
    check(report.refusals.empty(), "settling: nothing refused");
    check(report.readings.size() == 4, "settling: four readings");
    const char *const expected[] = {
        "Color at the start: auto exposure off, exposure 5000, gain 64, auto white balance off, "
        "white balance 4000",
        "Color before the hold: auto exposure on, exposure 8000, gain 154, auto white balance on, "
        "white balance 5500",
        "Color while held: auto exposure off, exposure 8000, gain 154, auto white balance off, "
        "white balance 5500",
        "Color after: auto exposure on, exposure 8000, gain 154, auto white balance on, white "
        "balance 5500"};
    for (std::size_t i = 0; i < report.readings.size() && i < 4; i++)
        check_line("settling: reading " + std::to_string(i), gca::reading_line(report.readings[i]),
                   expected[i]);

    check(hold.open(test_size).opened, "second opening: opened");
    check(hold.report().readings.size() == 1, "second opening: a new report");
    check(take_frames(hold, 29) == 29, "second opening: 29 frames");
    check_table(camera, "second opening: after 29 frames", {true, true, 10900, 241, 6950});
    check(take_frames(hold, 1) == 1, "second opening: the 30th frame");
    check_table(camera, "second opening: held after the 30th frame",
                {false, false, 11000, 244, 7000});
    hold.close();
    check_both_on(camera, "second opening: after the close");
}

/// The endings: a close after the camera failed while held, a close after it failed before the
/// settling, and the hold destroyed open after the hold each leave both switches on.
void test_endings()
{
    {
        stand_in_camera camera(35, gca_test::synthetic_ending::failure);
        gca::color_hold hold(camera, camera);
        check(hold.open(test_size).opened, "failure while held: opened");
        check(take_frames(hold, 40) == 35, "failure while held: 35 frames, then the failure");
        check_table(camera, "failure while held: held", {false, false, 8000, 154, 5500});
        hold.close();
        check_both_on(camera, "failure while held: after the close");
        check_only_listed_touched(camera, "failure while held");
        check(hold.report().refusals.empty(), "failure while held: nothing refused");
    }
    {
        stand_in_camera camera(10, gca_test::synthetic_ending::failure);
        gca::color_hold hold(camera, camera);
        check(hold.open(test_size).opened, "failure before the settling: opened");
        check(take_frames(hold, 40) == 10, "failure before the settling: 10 frames, the failure");
        hold.close();
        check_table(camera, "failure before the settling: after the close",
                    {true, true, 6000, 94, 4500});
        check(!written_off(camera, 2000) && !written_off(camera, 2003),
              "failure before the settling: no switch written off");
        check_only_listed_touched(camera, "failure before the settling");
    }
    {
        stand_in_camera camera(40, gca_test::synthetic_ending::end);
        {
            gca::color_hold hold(camera, camera);
            check(hold.open(test_size).opened, "destroyed open: opened");
            check(take_frames(hold, 40) == 40, "destroyed open: 40 frames");
            check_table(camera, "destroyed open: held", {false, false, 8000, 154, 5500});
        }
        check_both_on(camera, "destroyed open: after the hold's end");
        check_only_listed_touched(camera, "destroyed open");
    }
}

/// Each of the five refused in turn, its reads and writes alike, from the opening on: the 40
/// frames come, the refused property's control is never switched off and the other is held, the
/// report names the refused property in every refusal, one of them at the start; after the close
/// the other control is on again.
void test_each_property_refused()
{
    for (const std::uint32_t number : {2000u, 2001u, 2002u, 2003u, 2004u})
    {
        const std::string what = "refused " + std::to_string(number);
        const bool exposure_refused = number <= 2002;
        const std::uint32_t refused_switch = exposure_refused ? 2000 : 2003;
        const std::uint32_t other_switch = exposure_refused ? 2003 : 2000;
        stand_in_camera camera(40, gca_test::synthetic_ending::end);
        camera.table.at(number).refuse_reads = true;
        camera.table.at(number).refuse_writes = true;
        gca::color_hold hold(camera, camera);
        check(hold.open(test_size).opened, (what + ": opened").c_str());
        check(take_frames(hold, 40) == 40, (what + ": 40 frames").c_str());
        check(!written_off(camera, refused_switch),
              (what + ": its control never switched off").c_str());
        check(camera.table.at(other_switch).value == 0,
              (what + ": the other control held").c_str());
        hold.close();
        check(camera.table.at(other_switch).value == 1,
              (what + ": the other control on after the close").c_str());
        check_only_listed_touched(camera, what);

        const std::vector<gca::color_refusal> &refusals = hold.report().refusals;
        bool named = !refusals.empty() && refusals.front().step == gca::color_step::start;
        for (const gca::color_refusal &r : refusals)
        {
            std::printf("%s: %s\n", what.c_str(), gca::refusal_line(r).c_str());
            named = named && static_cast<std::uint32_t>(r.property) == number;
        }
        check(named, (what + ": every refusal names it, the first at the start").c_str());
    }

    stand_in_camera camera(40, gca_test::synthetic_ending::end);
    camera.table.at(2001).refuse_reads = true;
    gca::color_hold hold(camera, camera);
    check(hold.open(test_size).opened, "refused reading: opened");
    const gca::color_hold_report &report = hold.report();
    check(report.readings.size() == 1 && report.refusals.size() == 1,
          "refused reading: one reading, one refusal");
    if (report.readings.size() == 1 && report.refusals.size() == 1)
    {
        check_line("refused reading", gca::reading_line(report.readings[0]),
                   "Color at the start: auto exposure off, exposure unread, gain 64, auto white "
                   "balance off, white balance 4000");
        check_line("refused reading", gca::refusal_line(report.refusals[0]),
                   "Color: the camera refused to read exposure (2001) at the start: the stand-in "
                   "refuses reads of 2001");
    }
}

/// Only delivered frames count: 20 frames, then 20 calls of next() that each give a timeout, then
/// the close; neither switch is written off, and the only readings are those at the start and
/// after the close.
void test_only_delivered_frames_count()
{
    stand_in_camera camera(20, gca_test::synthetic_ending::timeout);
    gca::color_hold hold(camera, camera);
    check(hold.open(test_size).opened, "timeouts: opened");
    gca::frame frame;
    std::uint64_t frames = 0;
    std::uint64_t timeouts = 0;
    for (int i = 0; i < 40; i++)
    {
        const gca::next_status status = hold.next(frame).status;
        frames += status == gca::next_status::frame ? 1 : 0;
        timeouts += status == gca::next_status::timeout ? 1 : 0;
    }
    std::printf("timeouts: %llu frames, %llu timeouts\n", static_cast<unsigned long long>(frames),
                static_cast<unsigned long long>(timeouts));
    check(frames == 20 && timeouts == 20, "timeouts: 20 frames, then 20 timeouts");
    check(!written_off(camera, 2000) && !written_off(camera, 2003),
          "timeouts: no switch written off");
    check_table(camera, "timeouts: after 40 calls", {true, true, 7000, 124, 5000});
    hold.close();
    check(hold.report().readings.size() == 2, "timeouts: the readings at the start and after only");
    check_both_on(camera, "timeouts: after the close");
    check_only_listed_touched(camera, "timeouts");
}

/// The rule of what may be written: is_switch() is true for the numbers 2000 and 2003, written
/// here, and false for 2001, 2002 and 2004.
void test_switches_rule()
{
    for (const std::uint32_t number : {2000u, 2001u, 2002u, 2003u, 2004u})
    {
        const bool expected = number == 2000 || number == 2003;
        const bool rule = gca::is_switch(static_cast<gca::color_property>(number));
        std::printf("is_switch(%u): %d\n", number, rule);
        check(rule == expected,
              ("the rule of what may be written: " + std::to_string(number)).c_str());
    }
}

/// The camera refuses to switch auto exposure off at the hold: exposure stays automatic and its
/// values go on moving, white balance is held; the report holds that one refusal; after the
/// close both are on.
void test_switch_off_refused()
{
    stand_in_camera camera(40, gca_test::synthetic_ending::end);
    gca::color_hold hold(camera, camera);
    check(hold.open(test_size).opened, "switch-off refused: opened");
    check(take_frames(hold, 29) == 29, "switch-off refused: 29 frames");
    camera.table.at(2000).refuse_writes = true;
    check(take_frames(hold, 1) == 1, "switch-off refused: the 30th frame");
    camera.table.at(2000).refuse_writes = false;
    check(take_frames(hold, 10) == 10, "switch-off refused: 10 frames more");
    check_table(camera, "switch-off refused: after 40 frames", {true, false, 9000, 184, 5500});
    hold.close();
    check_both_on(camera, "switch-off refused: after the close");
    check_only_listed_touched(camera, "switch-off refused");
    const std::vector<gca::color_refusal> &refusals = hold.report().refusals;
    check(refusals.size() == 1, "switch-off refused: one refusal");
    if (refusals.size() == 1)
        check_line("switch-off refused", gca::refusal_line(refusals[0]),
                   "Color: the camera refused to switch auto exposure (2000) off at the hold: the "
                   "stand-in refuses writes of 2000");
}
} // namespace

/// Runs the hold's checks and prints the tables, the lines and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    test_hold_after_settling();
    test_endings();
    test_each_property_refused();
    test_switch_off_refused();
    test_only_delivered_frames_count();
    test_switches_rule();
    return gca_test::check_summary("test_color_hold");
}
