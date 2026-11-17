// SPDX-License-Identifier: 0BSD
/// @file parameters_file.cpp
/// A scan's parameters as JSON through OpenCV's FileStorage on memory (see
/// parameters_file.hpp). Each function's comment names the trap of FileStorage it guards against.

#include "scan/parameters_file.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace gca
{
namespace
{
/// The format version this writer writes and this reader reads (docs/SCAN_FILES.md).
constexpr int format_version = 1;

/// The largest whole number JSON holds exactly, the largest budget and count of frames: 2^53, up
/// to which every whole number is a double [S: IEEE 754 binary64, a 53-bit significand].
constexpr std::uint64_t largest_exact = std::uint64_t{1} << 53;

/// The largest count of iterations or points and the largest width or height: 2^32 - 1,
/// std::uint32_t's.
constexpr std::uint64_t largest_32 = 4294967295u;

/// The longest string FileStorage reads back: it refuses one of 4096 characters when it reads it
/// and throws when it writes one of 4097 [M 2026-11-14: OpenCV 4.8.0].
constexpr std::size_t longest_string = 4095;

/// The keys of the top level.
constexpr std::string_view version_key = "format_version";
constexpr std::string_view commit_key = "commit";
constexpr std::string_view class_key = "object_class";
constexpr std::string_view set_key = "parameters";
constexpr std::string_view stream_key = "stream";
constexpr std::string_view source_key = "source";
constexpr std::string_view refused_key = "refused_properties";
constexpr std::string_view start_key = "start_utc";
constexpr std::string_view stop_key = "stop_utc";
constexpr std::string_view frames_key = "frames";
constexpr std::string_view end_key = "end";

/// The keys of the top level, in the order the writer writes them.
constexpr std::array<std::string_view, 11> top_keys = {
    version_key, commit_key, class_key, set_key,    stream_key, source_key,
    refused_key, start_key,  stop_key,  frames_key, end_key};

/// The keys of "source", in the order the writer writes them.
constexpr std::array<std::string_view, 3> source_keys = {"camera", "firmware_version",
                                                         "sdk_version"};

/// What a number of the file may be (docs/SCAN_FILES.md, "Parameters").
enum class value_rule
{
    length,           ///< a length or a threshold: finite and > 0
    share,            ///< a share: finite, > 0 and at most 1
    budget,           ///< the memory budget: a whole number from 1 to 2^53
    count,            ///< a count of iterations or points: a whole number from 0 to 2^32 - 1
    size,             ///< a width or a height: a whole number from 1 to 2^32 - 1
    tally,            ///< a count of frames: a whole number from 0 to 2^53
    float_length,     ///< a float's length: finite and > 0 as a float
    float_coordinate, ///< a float's coordinate: finite as a float
};

/// A number of one of the file's maps: its key, its rule, and how it is read from its struct and
/// written into it, as a double. A whole number is exact as a double: a count, a width and a
/// height are at most 2^32 - 1, a budget and a count of frames at most 2^53 once its rule admits
/// it [S: IEEE 754 binary64, a 53-bit significand]; read_numbers() checks the rule before it writes
/// one, and a float's number lies within the float's range then. A budget or a count of frames
/// above 2^53 rounds as a double, so set_refusal() and frames_refusal() check those as integers.
/// @tparam Struct the struct the map holds.
template <typename Struct> struct number_entry
{
    std::string_view key;          ///< the key in the map
    value_rule rule;               ///< what its value may be
    double (*get)(const Struct &); ///< the member's value
    void (*put)(Struct &, double); ///< sets the member to a value its rule admits
};

/// Gives the struct a pointer to a member belongs to.
/// @tparam Pointer the member pointer's type.
template <typename Pointer> struct struct_of_member;

/// Gives the struct a pointer to a member belongs to.
/// @tparam Member the member's type.
/// @tparam Struct the struct.
template <typename Member, typename Struct> struct struct_of_member<Member Struct::*>
{
    using type = Struct; ///< the struct
};

/// Gives the entry of a member.
/// @tparam member the member, a double, a float or an integer.
/// @param key [in] its key.
/// @param rule [in] its rule, a whole rule for an integer member.
/// @return the entry, which reads and writes that member.
template <auto member> constexpr auto entry_of(std::string_view key, value_rule rule)
{
    using Struct = typename struct_of_member<decltype(member)>::type;
    using Member = std::remove_reference_t<decltype(std::declval<Struct &>().*member)>;
    return number_entry<Struct>{
        key, rule, [](const Struct &s) { return static_cast<double>(s.*member); },
        [](Struct &s, double value) { s.*member = static_cast<Member>(value); }};
}

/// Gives the entry of one level's element of an array member of the set.
/// @tparam member the array member, of doubles or of integers.
/// @tparam level the element, a level of the pyramid.
/// @param key [in] its key.
/// @param rule [in] its rule.
/// @return the entry, which reads and writes that element.
template <auto member, std::size_t level>
constexpr number_entry<object_class_parameters> level_entry_of(std::string_view key,
                                                               value_rule rule)
{
    using Element = std::remove_reference_t<decltype((std::declval<object_class_parameters &>().*
                                                      member)[level])>;
    return {key, rule,
            [](const object_class_parameters &set) {
                return static_cast<double>((set.*member)[level]);
            },
            [](object_class_parameters &set, double value) {
                (set.*member)[level] = static_cast<Element>(value);
            }};
}

/// The keys the depths' order names.
constexpr std::string_view min_depth_key = "min_depth_m";
constexpr std::string_view budget_key = "memory_budget_bytes";
constexpr std::string_view max_depth_key = "max_depth_m";
constexpr std::string_view too_close_key = "too_close_depth_m";

/// The set's type, by a short name for the table below.
using class_set = object_class_parameters;

/// The set of parameters' one table: each key with its rule and its member, in the order the
/// writer writes them.
constexpr std::array<number_entry<class_set>, 24> set_entries = {
    entry_of<&class_set::live_voxel_size_m>("live_voxel_size_m", value_rule::length),
    entry_of<&class_set::full_voxel_size_m>("full_voxel_size_m", value_rule::length),
    entry_of<&class_set::min_depth_m>(min_depth_key, value_rule::length),
    entry_of<&class_set::max_depth_m>(max_depth_key, value_rule::length),
    entry_of<&class_set::memory_budget_bytes>(budget_key, value_rule::budget),
    entry_of<&class_set::too_close_depth_m>(too_close_key, value_rule::length),
    entry_of<&class_set::depth_noise_per_m>("depth_noise_per_m", value_rule::length),
    entry_of<&class_set::truncation_noise_multiple>("truncation_noise_multiple",
                                                    value_rule::length),
    entry_of<&class_set::truncation_min_voxels>("truncation_min_voxels", value_rule::length),
    entry_of<&class_set::normals_radius_px>("normals_radius_px", value_rule::count),
    entry_of<&class_set::normals_min_points>("normals_min_points", value_rule::count),
    entry_of<&class_set::normals_max_incidence_deg>("normals_max_incidence_deg",
                                                    value_rule::length),
    level_entry_of<&class_set::max_distance_m, 0>("max_distance_level_0_m", value_rule::length),
    level_entry_of<&class_set::max_distance_m, 1>("max_distance_level_1_m", value_rule::length),
    level_entry_of<&class_set::max_distance_m, 2>("max_distance_level_2_m", value_rule::length),
    level_entry_of<&class_set::max_iterations, 0>("max_iterations_level_0", value_rule::count),
    level_entry_of<&class_set::max_iterations, 1>("max_iterations_level_1", value_rule::count),
    level_entry_of<&class_set::max_iterations, 2>("max_iterations_level_2", value_rule::count),
    entry_of<&class_set::huber_threshold_m>("huber_threshold_m", value_rule::length),
    entry_of<&class_set::point_to_plane_weight>("point_to_plane_weight", value_rule::share),
    entry_of<&class_set::min_fitness>("min_fitness", value_rule::share),
    entry_of<&class_set::max_inlier_rmse_m>("max_inlier_rmse_m", value_rule::length),
    entry_of<&class_set::max_turn_rate_deg_s>("max_turn_rate_deg_s", value_rule::length),
    entry_of<&class_set::max_speed_m_s>("max_speed_m_s", value_rule::length),
};

/// The stream's one table: its size, intrinsics and depth unit, each with its rule and member, in
/// the order the writer writes them. The members lie in the description's size and intrinsics,
/// so each entry is written out.
constexpr std::array<number_entry<frame_description>, 7> stream_entries = {{
    {"width_px", value_rule::size,
     [](const frame_description &d) { return static_cast<double>(d.size.width); },
     [](frame_description &d, double v) { d.size.width = static_cast<std::uint32_t>(v); }},
    {"height_px", value_rule::size,
     [](const frame_description &d) { return static_cast<double>(d.size.height); },
     [](frame_description &d, double v) { d.size.height = static_cast<std::uint32_t>(v); }},
    {"fx_px", value_rule::float_length,
     [](const frame_description &d) { return static_cast<double>(d.intrinsics.fx); },
     [](frame_description &d, double v) { d.intrinsics.fx = static_cast<float>(v); }},
    {"fy_px", value_rule::float_length,
     [](const frame_description &d) { return static_cast<double>(d.intrinsics.fy); },
     [](frame_description &d, double v) { d.intrinsics.fy = static_cast<float>(v); }},
    {"cx_px", value_rule::float_coordinate,
     [](const frame_description &d) { return static_cast<double>(d.intrinsics.cx); },
     [](frame_description &d, double v) { d.intrinsics.cx = static_cast<float>(v); }},
    {"cy_px", value_rule::float_coordinate,
     [](const frame_description &d) { return static_cast<double>(d.intrinsics.cy); },
     [](frame_description &d, double v) { d.intrinsics.cy = static_cast<float>(v); }},
    {"depth_unit_m", value_rule::float_length,
     [](const frame_description &d) { return static_cast<double>(d.depth_unit_m); },
     [](frame_description &d, double v) { d.depth_unit_m = static_cast<float>(v); }},
}};

/// The keys of "frames", which frames_refusal() names too.
constexpr std::string_view taken_key = "taken";
constexpr std::string_view fused_key = "fused";
constexpr std::string_view lost_key = "lost";
constexpr std::string_view not_taken_key = "not_taken";

/// The frames' one table: the counts, each with its rule and member, in the order the writer
/// writes them.
constexpr std::array<number_entry<scan_frame_counts>, 4> frame_entries = {
    entry_of<&scan_frame_counts::taken>(taken_key, value_rule::tally),
    entry_of<&scan_frame_counts::fused>(fused_key, value_rule::tally),
    entry_of<&scan_frame_counts::lost>(lost_key, value_rule::tally),
    entry_of<&scan_frame_counts::not_taken>(not_taken_key, value_rule::tally),
};

/// Gives the keys of a table, in the writer's order.
/// @param entries [in] the table.
/// @return its keys.
template <typename Struct, std::size_t N>
constexpr std::array<std::string_view, N> keys_of(
    const std::array<number_entry<Struct>, N> &entries)
{
    std::array<std::string_view, N> keys{};
    for (std::size_t k = 0; k < N; k++)
        keys[k] = entries[k].key;
    return keys;
}

/// The table of why a scan ended: one case for each, so -Wswitch names an enumerator added
/// without its key (tools/verify.sh fails on the warning).
/// @param end [in] why the scan ended.
/// @return its key; nothing for a value that is no enumerator.
constexpr std::optional<std::string_view> key_of(scan_end end) noexcept
{
    switch (end)
    {
    case scan_end::stopped:
        return "stopped";
    case scan_end::source_timeout:
        return "source_timeout";
    case scan_end::source_failure:
        return "source_failure";
    case scan_end::source_end:
        return "source_end";
    case scan_end::tracking_failed:
        return "tracking_failed";
    case scan_end::writer_failed:
        return "writer_failed";
    case scan_end::shut_down:
        return "shut_down";
    }
    return std::nullopt;
}

/// Quotes a key for a reason.
/// @param key [in] the key.
/// @return the key in double quotes.
std::string in_quotes(std::string_view key)
{
    return "\"" + std::string(key) + "\"";
}

/// Names a key of a map for a reason.
/// @param key [in] the key.
/// @param map [in] the map's key at the top level.
/// @return the key in double quotes, "in", and the map's key in double quotes.
std::string key_in(std::string_view key, std::string_view map)
{
    return in_quotes(key) + " in " + in_quotes(map);
}

/// Writes a double for a reason, in its shortest exact form.
/// @param value [in] the number.
/// @return its digits, "nan" or "inf" included.
std::string number_text(double value)
{
    std::string text;
    append_number(text, value);
    return text;
}

/// Checks the numbers of a struct against their rules, as a struct the scanner can have run
/// with: a length finite and > 0, a share finite, > 0 and at most 1, a coordinate finite, a width
/// or height from 1, each as its member holds it; a count is any std::uint32_t, and a budget and
/// a count of frames are checked as integers by set_refusal() and frames_refusal().
/// @param entries [in] the struct's table.
/// @param values [in] the struct.
/// @param map [in] the map's key at the top level, for the reason.
/// @return the reason it is not, naming the key; empty when it is.
template <typename Struct, std::size_t N>
std::string rules_refusal(const std::array<number_entry<Struct>, N> &entries, const Struct &values,
                          std::string_view map)
{
    for (const number_entry<Struct> &entry : entries)
    {
        const double value = entry.get(values);
        const bool positive = std::isfinite(value) && value > 0.0;
        std::string not_one;
        switch (entry.rule)
        {
        case value_rule::length:
        case value_rule::float_length:
            if (!positive)
                not_one = number_text(value) + ", not a finite number > 0";
            break;
        case value_rule::share:
            if (!(positive && value <= 1.0))
                not_one = number_text(value) + ", not a number in (0, 1]";
            break;
        case value_rule::float_coordinate:
            if (!std::isfinite(value))
                not_one = number_text(value) + ", not a finite number";
            break;
        case value_rule::size:
            // Exact: a width or height is a std::uint32_t.
            if (value < 1.0)
                not_one = number_text(value) + ", not a whole number from 1 to 4294967295";
            break;
        case value_rule::budget:
        case value_rule::tally:
        case value_rule::count:
            break;
        }
        if (!not_one.empty())
            return key_in(entry.key, map) + " is " + not_one;
    }
    return {};
}

/// Checks that a whole number lies at most at 2^53, as an integer: 2^53 + 1 converted to a double
/// rounds to 2^53.
/// @param key [in] its key, for the reason.
/// @param map [in] its map's key at the top level, for the reason.
/// @param value [in] the number.
/// @param least [in] the least it may be, 0 or 1.
/// @return the reason it does not lie from least to 2^53, naming the key; empty when it does.
std::string exact_refusal(std::string_view key, std::string_view map, std::uint64_t value,
                          std::uint64_t least)
{
    if (value >= least && value <= largest_exact)
        return {};
    return key_in(key, map) + " is " + std::to_string(value) + ", not a whole number from " +
           std::to_string(least) + " to 2^53";
}

/// Checks that a set of parameters is one the scanner can run with: each value by its rule
/// (rules_refusal()), the budget from 1 to 2^53, the depths in order and the "too close" threshold
/// between them; the stages' own ranges are checked when a scan requests its buffers.
/// @param set [in] the set.
/// @return the reason it is not, naming the key; empty when it is.
std::string set_refusal(const object_class_parameters &set)
{
    std::string reason = rules_refusal(set_entries, set, set_key);
    if (reason.empty())
        reason = exact_refusal(budget_key, set_key, set.memory_budget_bytes, 1);
    if (!reason.empty())
        return reason;
    if (!(set.min_depth_m < set.max_depth_m))
        return in_quotes(min_depth_key) + " " + number_text(set.min_depth_m) +
               " is not less than " + in_quotes(max_depth_key) + " " + number_text(set.max_depth_m);
    if (!(set.min_depth_m < set.too_close_depth_m && set.too_close_depth_m < set.max_depth_m))
        return in_quotes(too_close_key) + " " + number_text(set.too_close_depth_m) +
               " does not lie between " + in_quotes(min_depth_key) + " " +
               number_text(set.min_depth_m) + " and " + in_quotes(max_depth_key) + " " +
               number_text(set.max_depth_m);
    return {};
}

/// Checks that the counts of frames are ones a scan can have: each from 0 to 2^53, and the frames
/// taken those fused and those lost together, and the frame tracking failed on when the scan
/// ended because of it (docs/SCAN_FILES.md, "Parameters").
/// @param frames [in] the counts.
/// @param end [in] why the scan ended.
/// @return the reason they are not, naming the key; empty when they are.
std::string frames_refusal(const scan_frame_counts &frames, scan_end end)
{
    std::string reason = exact_refusal(taken_key, frames_key, frames.taken, 0);
    if (reason.empty())
        reason = exact_refusal(fused_key, frames_key, frames.fused, 0);
    if (reason.empty())
        reason = exact_refusal(lost_key, frames_key, frames.lost, 0);
    if (reason.empty())
        reason = exact_refusal(not_taken_key, frames_key, frames.not_taken, 0);
    if (!reason.empty())
        return reason;
    // Each is at most 2^53 here, so the sum does not wrap.
    const bool failed = end == scan_end::tracking_failed;
    if (frames.taken != frames.fused + frames.lost + (failed ? 1 : 0))
        return key_in(taken_key, frames_key) + " " + std::to_string(frames.taken) +
               " is not \"fused\" " + std::to_string(frames.fused) +
               (failed ? ", \"lost\" " : " and \"lost\" ") + std::to_string(frames.lost) +
               (failed ? " and the frame tracking failed on" : "") + " together, for " +
               in_quotes(end_key) + " " + in_quotes(name_of(end));
    return {};
}

/// Checks that a string is one the file holds. OpenCV's JSON writer gives text that both its own
/// reader and Python's json.loads take back unchanged only for some strings [M 2026-11-14:
/// OpenCV 4.8.0]: it writes an apostrophe as \' (no JSON escape), a string that starts and ends
/// with a double quote raw (OpenCV's reader then drops the quotes), a control byte raw (no JSON),
/// and a string of more than 4095 characters does not read back. So the file holds strings of
/// printable ASCII (0x20 to 0x7e) without the apostrophe, not bounded by double quotes, of at
/// most 4095 characters; a byte above 0x7e is refused too, as the reader checks no UTF-8.
/// @param named [in] the key as a reason names it, in double quotes.
/// @param value [in] the string.
/// @return the reason it is not, naming the key; empty when it is.
std::string string_refusal(const std::string &named, std::string_view value)
{
    static const char hex[] = "0123456789abcdef";
    if (value.size() > longest_string)
        return named + " holds " + std::to_string(value.size()) + " characters, more than 4095";
    for (std::size_t k = 0; k < value.size(); k++)
    {
        const unsigned char byte = static_cast<unsigned char>(value[k]);
        if (byte < 0x20 || byte > 0x7e || byte == '\'')
            return named + " holds the byte 0x" + hex[byte >> 4] + hex[byte & 0xf] + " at " +
                   std::to_string(k) +
                   ", not printable ASCII without the apostrophe, which OpenCV does not write as "
                   "JSON";
    }
    if (!value.empty() && value.front() == '"' && value.back() == '"')
        return named + " starts and ends with a double quote, which OpenCV does not write as JSON";
    return {};
}

/// Gives the strings of a source's description in source_keys' order, the one copy of that order.
/// @tparam Source source_description, const for reading the strings.
/// @param source [in] the description; its strings are written through the pointers when it is
///        not const.
/// @return pointers to its camera, firmware version and SDK version, valid as long as it is.
template <typename Source>
std::array<decltype(&std::declval<Source &>().camera), 3> strings_of(Source &source)
{
    return {&source.camera, &source.firmware_version, &source.sdk_version};
}

/// Gives the text of a time for the file, or why it has none.
/// @param key [in] the time's key, for the reason.
/// @param time [in] the time.
/// @param reason [out] why the file cannot hold it, naming the key; set only then.
/// @return its text (scan/utc_time.hpp); nothing outside 1970 to 9999.
std::optional<std::string> time_text(std::string_view key, utc_time time, std::string &reason)
{
    std::optional<std::string> text = utc_text(time);
    if (!text)
        reason = in_quotes(key) + " is " + std::to_string(time.time_since_epoch().count()) +
                 " us after 1970-01-01T00:00:00Z, not a time from 1970 to 9999";
    return text;
}

/// Checks that a map holds each of its keys once and no other. FileStorage keeps a key that
/// stands twice in a map twice, and its lookup gives the first [M 2026-11-14: OpenCV 4.8.0], so
/// this walks the map's keys itself.
/// @param map [in] the map, read by FileStorage.
/// @param keys [in] the keys it holds.
/// @param where [in] where the map stands, for the reason ("at the top level").
/// @return the reason it does not, naming the key; empty when it does.
template <std::size_t N>
std::string keys_refusal(const cv::FileNode &map, const std::array<std::string_view, N> &keys,
                         const std::string &where)
{
    std::vector<std::string> seen;
    for (const cv::FileNode &child : map)
    {
        const std::string key = child.name();
        if (std::find(keys.begin(), keys.end(), key) == keys.end())
            return "unknown key " + in_quotes(key) + " " + where;
        if (std::find(seen.begin(), seen.end(), key) != seen.end())
            return "key " + in_quotes(key) + " twice " + where;
        seen.push_back(key);
    }
    for (const std::string_view key : keys)
    {
        if (std::find(seen.begin(), seen.end(), key) == seen.end())
            return "no key " + in_quotes(key) + " " + where;
    }
    return {};
}

/// Reads the numbers of a map into its struct, the map's keys checked first (keys_refusal()).
/// FileStorage reads a string asked for as a number as DBL_MAX, and an integer beyond 2147483647
/// wrapped to 32 bits, without a word [M 2026-11-14: OpenCV 4.8.0]; so each number is taken only
/// from a real number, which FileStorage writes with a decimal point or an exponent and reads back
/// bit for bit, the whole numbers too. A whole number is checked before it is converted, and a
/// float's number against the float's range, as a double outside its type's range does not
/// convert [S: C++20, [conv.fpint] and [conv.double]]; rules_refusal() checks the rest after.
/// @param root [in] the top level, read by FileStorage.
/// @param map_key [in] the map's key at the top level; the map is a map.
/// @param entries [in] the struct's table.
/// @param reason [out] why the map was refused, naming the key; set only on a refusal.
/// @return the struct, or nothing when the map's keys are not the table's, a value is no real
///         number, a whole number is not one of its rule's, or a float's number is not finite or
///         lies outside the float's range.
template <typename Struct, std::size_t N>
std::optional<Struct> read_numbers(const cv::FileNode &root, std::string_view map_key,
                                   const std::array<number_entry<Struct>, N> &entries,
                                   std::string &reason)
{
    const cv::FileNode map = root[std::string(map_key)];
    reason = keys_refusal(map, keys_of(entries), "in " + in_quotes(map_key));
    if (!reason.empty())
        return std::nullopt;
    Struct values{};
    for (const number_entry<Struct> &entry : entries)
    {
        const cv::FileNode node = map[std::string(entry.key)];
        if (!node.isReal())
        {
            reason =
                key_in(entry.key, map_key) + " is no number with a decimal point or an exponent";
            return std::nullopt;
        }
        const double value = static_cast<double>(node);
        const bool whole = value == std::floor(value);
        const char *not_one = nullptr;
        switch (entry.rule)
        {
        case value_rule::budget:
            if (!(value >= 1.0 && value <= static_cast<double>(largest_exact) && whole))
                not_one = "a whole number from 1 to 2^53";
            break;
        case value_rule::count:
            if (!(value >= 0.0 && value <= static_cast<double>(largest_32) && whole))
                not_one = "a whole number from 0 to 4294967295";
            break;
        case value_rule::size:
            if (!(value >= 1.0 && value <= static_cast<double>(largest_32) && whole))
                not_one = "a whole number from 1 to 4294967295";
            break;
        case value_rule::tally:
            if (!(value >= 0.0 && value <= static_cast<double>(largest_exact) && whole))
                not_one = "a whole number from 0 to 2^53";
            break;
        case value_rule::float_length:
        case value_rule::float_coordinate:
            if (!(std::abs(value) <= static_cast<double>(std::numeric_limits<float>::max())))
                not_one = "a finite number that a float holds";
            break;
        case value_rule::length:
        case value_rule::share:
            break;
        }
        if (not_one != nullptr)
        {
            reason = key_in(entry.key, map_key) + " is " + number_text(value) + ", not " + not_one;
            return std::nullopt;
        }
        entry.put(values, value);
    }
    return values;
}

/// Reads a string that the file holds (string_refusal()).
/// @param node [in] its node.
/// @param named [in] the key as a reason names it, in double quotes.
/// @param reason [out] why it was refused, naming the key; set only on a refusal.
/// @return the string, or nothing when the node is no string or string_refusal() refuses it.
std::optional<std::string> read_string(const cv::FileNode &node, const std::string &named,
                                       std::string &reason)
{
    if (!node.isString())
    {
        reason = named + " is no string";
        return std::nullopt;
    }
    std::string value = node.string();
    reason = string_refusal(named, value);
    if (!reason.empty())
        return std::nullopt;
    return value;
}

/// Reads the camera's run-time properties refused: an array of names of color_property_table. A
/// name is compared with the table's own, so that only its five names, which string_refusal()
/// admits, are read.
/// @param root [in] the top level, read by FileStorage.
/// @param reason [out] why it was refused, naming the key; set only on a refusal.
/// @return one flag for each row of the table, true for a name the array holds; nothing when the
///         value is no array, or holds a value that is no string, no name of the table, or a name
///         twice.
std::optional<std::array<bool, color_property_count>> read_refused(const cv::FileNode &root,
                                                                   std::string &reason)
{
    const cv::FileNode names = root[std::string(refused_key)];
    if (!names.isSeq())
    {
        reason = in_quotes(refused_key) + " is no JSON array";
        return std::nullopt;
    }
    std::array<bool, color_property_count> refused{};
    for (const cv::FileNode &node : names)
    {
        if (!node.isString())
        {
            reason = in_quotes(refused_key) + " holds a value that is no string";
            return std::nullopt;
        }
        const std::string name = node.string();
        std::size_t k = 0;
        while (k < color_property_count && name != color_property_table[k].name)
            k++;
        if (k == color_property_count)
        {
            reason = in_quotes(refused_key) + " holds \"" + name.substr(0, 40) +
                     "\", no property of the camera the scanner touches";
            return std::nullopt;
        }
        if (refused[k])
        {
            reason = in_quotes(refused_key) + " holds \"" + name + "\" twice";
            return std::nullopt;
        }
        refused[k] = true;
    }
    return refused;
}

/// Reads a time of the top level.
/// @param root [in] the top level, read by FileStorage.
/// @param key [in] the time's key.
/// @param reason [out] why it was refused, naming the key; set only on a refusal.
/// @return the time, or nothing when its value is no string or not a time's text
///         (scan/utc_time.hpp).
std::optional<utc_time> read_time(const cv::FileNode &root, std::string_view key,
                                  std::string &reason)
{
    const cv::FileNode node = root[std::string(key)];
    if (!node.isString())
    {
        reason = in_quotes(key) + " is no string";
        return std::nullopt;
    }
    const std::string text = node.string();
    const std::optional<utc_time> time = utc_time_of(text);
    if (!time)
        reason = in_quotes(key) + " is \"" + text.substr(0, 40) +
                 "\", not a time in UTC as YYYYMMDDTHHMMSS.ffffffZ from 1970 to 9999";
    return time;
}

/// Reads why a scan ended from the top level.
/// @param root [in] the top level, read by FileStorage.
/// @param reason [out] why it was refused; set only on a refusal.
/// @return why it ended, or nothing when its value is no string or no key of the closed set.
std::optional<scan_end> read_end(const cv::FileNode &root, std::string &reason)
{
    const cv::FileNode node = root[std::string(end_key)];
    if (!node.isString())
    {
        reason = in_quotes(end_key) + " is no string";
        return std::nullopt;
    }
    const std::string name = node.string();
    const std::optional<scan_end> end = scan_end_named(name);
    if (!end)
    {
        // The keys from the one table, so that a new key changes one place.
        std::string keys;
        for (int k = 0; key_of(static_cast<scan_end>(k)); k++)
            keys += (k == 0 ? "" : ", ") + std::string(*key_of(static_cast<scan_end>(k)));
        reason = in_quotes(end_key) + " is \"" + name.substr(0, 40) +
                 "\", no key of why a scan ended (" + keys + ")";
    }
    return end;
}

/// Reads the parameters from the file's text. FileStorage reads YAML and XML too, whatever
/// format it is told, and refuses JSON that starts with white space [M 2026-11-14: OpenCV
/// 4.8.0]; so the text has to start with '{' after any white space, which is skipped.
/// FileStorage reads "format_version" as a 32-bit int: 4294967297 and 0x1 read as 1
/// (docs/SCAN_FILES.md lists the parser's deviations).
/// @param text [in] the file's content.
/// @param reason [out] why the file was refused; set only on a refusal.
/// @return the parameters, or nothing on a refusal (read_parameters() lists them).
/// @note Throws cv::Exception when FileStorage cannot parse the text.
std::optional<scan_parameters> parse(const std::string &text, std::string &reason)
{
    const std::size_t start = text.find_first_not_of(" \t\r\n");
    if (start == std::string::npos || text[start] != '{')
    {
        reason = "no JSON object: the first byte after any white space is not '{'";
        return std::nullopt;
    }
    const cv::FileStorage storage(text.substr(start), cv::FileStorage::READ |
                                                          cv::FileStorage::MEMORY |
                                                          cv::FileStorage::FORMAT_JSON);
    const cv::FileNode root = storage.root();
    // The version first: a file of another version is refused as such, whatever its keys.
    const cv::FileNode version = root[std::string(version_key)];
    if (!version.isInt())
    {
        reason = "no integer " + in_quotes(version_key) + " at the top level";
        return std::nullopt;
    }
    if (static_cast<int>(version) != format_version)
    {
        reason = "format version " + std::to_string(static_cast<int>(version)) +
                 ", this reader reads version " + std::to_string(format_version);
        return std::nullopt;
    }
    reason = keys_refusal(root, top_keys, "at the top level");
    if (!reason.empty())
        return std::nullopt;
    for (const std::string_view map_key : {set_key, stream_key, source_key, frames_key})
    {
        if (!root[std::string(map_key)].isMap())
        {
            reason = in_quotes(map_key) + " is no JSON object";
            return std::nullopt;
        }
    }

    scan_parameters read{};
    std::optional<std::string> commit =
        read_string(root[std::string(commit_key)], in_quotes(commit_key), reason);
    if (!commit)
        return std::nullopt;
    read.commit = std::move(*commit);
    const cv::FileNode class_name = root[std::string(class_key)];
    if (!class_name.isString())
    {
        reason = in_quotes(class_key) + " is no string";
        return std::nullopt;
    }
    const std::optional<gca::object_class> kind = object_class_named(class_name.string());
    if (!kind)
    {
        reason = in_quotes(class_key) + " is " + in_quotes(class_name.string()) +
                 ", no object class (tabletop_object, large_object)";
        return std::nullopt;
    }
    read.object_class = *kind;

    const std::optional<object_class_parameters> set =
        read_numbers(root, set_key, set_entries, reason);
    if (!set)
        return std::nullopt;
    reason = set_refusal(*set);
    if (!reason.empty())
        return std::nullopt;
    read.parameters = *set;
    const std::optional<frame_description> described =
        read_numbers(root, stream_key, stream_entries, reason);
    if (!described)
        return std::nullopt;
    reason = rules_refusal(stream_entries, *described, stream_key);
    if (!reason.empty())
        return std::nullopt;
    read.stream = *described;

    const cv::FileNode source = root[std::string(source_key)];
    reason = keys_refusal(source, source_keys, "in " + in_quotes(source_key));
    if (!reason.empty())
        return std::nullopt;
    const std::array<std::string *, 3> strings = strings_of(read.source);
    for (std::size_t k = 0; k < source_keys.size(); k++)
    {
        std::optional<std::string> value = read_string(source[std::string(source_keys[k])],
                                                       key_in(source_keys[k], source_key), reason);
        if (!value)
            return std::nullopt;
        *strings[k] = std::move(*value);
    }
    const std::optional<std::array<bool, color_property_count>> refused =
        read_refused(root, reason);
    if (!refused)
        return std::nullopt;
    read.refused_properties = *refused;

    const std::optional<utc_time> started = read_time(root, start_key, reason);
    if (!started)
        return std::nullopt;
    const std::optional<utc_time> stopped = read_time(root, stop_key, reason);
    if (!stopped)
        return std::nullopt;
    read.start = *started;
    read.stop = *stopped;
    const std::optional<scan_frame_counts> frames =
        read_numbers(root, frames_key, frame_entries, reason);
    if (!frames)
        return std::nullopt;
    const std::optional<scan_end> end = read_end(root, reason);
    if (!end)
        return std::nullopt;
    reason = frames_refusal(*frames, *end);
    if (!reason.empty())
        return std::nullopt;
    read.frames = *frames;
    read.end = *end;
    return read;
}

/// Writes the numbers of a struct as a map.
/// @param storage [in,out] the storage written.
/// @param map_key [in] the map's key at the top level.
/// @param entries [in] the struct's table.
/// @param values [in] the struct, its numbers admitted by rules_refusal().
/// @note Throws cv::Exception when FileStorage cannot write.
template <typename Struct, std::size_t N>
void write_numbers(cv::FileStorage &storage, std::string_view map_key,
                   const std::array<number_entry<Struct>, N> &entries, const Struct &values)
{
    storage.startWriteStruct(std::string(map_key), cv::FileNode::MAP);
    // Exact: a whole number is at most 2^53 once its rule admits it, and a float is a double.
    for (const number_entry<Struct> &entry : entries)
        cv::write(storage, std::string(entry.key), entry.get(values));
    storage.endWriteStruct();
}

/// Checks that a scan's parameters are ones the file holds, in the order the reader reads them.
/// @param parameters [in] the parameters.
/// @param start [out] the start's text, when the parameters are admitted.
/// @param stop [out] the stop's text, when the parameters are admitted.
/// @return the reason they are not, naming the key; empty when they are.
std::string parameters_refusal(const scan_parameters &parameters, std::string &start,
                               std::string &stop)
{
    std::string reason = string_refusal(in_quotes(commit_key), parameters.commit);
    if (reason.empty())
        reason = set_refusal(parameters.parameters);
    if (reason.empty())
        reason = rules_refusal(stream_entries, parameters.stream, stream_key);
    if (reason.empty())
        reason = source_refusal(parameters.source);
    if (!reason.empty())
        return reason;
    const std::optional<std::string> started = time_text(start_key, parameters.start, reason);
    if (!started)
        return reason;
    const std::optional<std::string> stopped = time_text(stop_key, parameters.stop, reason);
    if (!stopped)
        return reason;
    start = *started;
    stop = *stopped;
    if (!key_of(parameters.end))
        return in_quotes(end_key) + " is " + std::to_string(static_cast<int>(parameters.end)) +
               ", no enumerator of why a scan ended";
    return frames_refusal(parameters.frames, parameters.end);
}
} // namespace

std::string_view name_of(scan_end end) noexcept
{
    return key_of(end).value_or(std::string_view{});
}

std::optional<scan_end> scan_end_named(std::string_view name) noexcept
{
    // The enumerators run from 0 without a gap (parameters_file.hpp): the first value without a
    // key ends them.
    for (int k = 0;; k++)
    {
        const scan_end end = static_cast<scan_end>(k);
        const std::optional<std::string_view> key = key_of(end);
        if (!key)
            return std::nullopt;
        if (*key == name)
            return end;
    }
}

std::string source_refusal(const source_description &source) noexcept
{
    const std::array<const std::string *, 3> strings = strings_of(source);
    for (std::size_t k = 0; k < source_keys.size(); k++)
    {
        std::string reason = string_refusal(key_in(source_keys[k], source_key), *strings[k]);
        if (!reason.empty())
            return reason;
    }
    return {};
}

write_result write_parameters(const std::filesystem::path &path,
                              const scan_parameters &parameters) noexcept
{
    std::string start;
    std::string stop;
    const std::string refusal = parameters_refusal(parameters, start, stop);
    if (!refusal.empty())
        return {false, path.string() + ": " + refusal};
    std::string text;
    try
    {
        cv::FileStorage storage(".json", cv::FileStorage::WRITE | cv::FileStorage::MEMORY |
                                             cv::FileStorage::FORMAT_JSON);
        // Each string is one string_refusal() admits, which OpenCV writes as JSON; a time's text
        // and an end's key hold digits, letters, '.' and '_' only.
        cv::write(storage, std::string(version_key), format_version);
        cv::write(storage, std::string(commit_key), parameters.commit);
        cv::write(storage, std::string(class_key), std::string(name_of(parameters.object_class)));
        write_numbers(storage, set_key, set_entries, parameters.parameters);
        write_numbers(storage, stream_key, stream_entries, parameters.stream);
        storage.startWriteStruct(std::string(source_key), cv::FileNode::MAP);
        const std::array<const std::string *, 3> strings = strings_of(parameters.source);
        for (std::size_t k = 0; k < source_keys.size(); k++)
            cv::write(storage, std::string(source_keys[k]), *strings[k]);
        storage.endWriteStruct();
        // The table's names, printable ASCII without the apostrophe, which OpenCV writes as JSON;
        // an empty array as "[" and "]" on lines of their own, which it reads back as an empty
        // sequence [M 2026-11-17: OpenCV 4.8.0, python's cv2].
        storage.startWriteStruct(std::string(refused_key), cv::FileNode::SEQ);
        for (std::size_t k = 0; k < color_property_count; k++)
        {
            if (parameters.refused_properties[k])
                cv::write(storage, std::string(), std::string(color_property_table[k].name));
        }
        storage.endWriteStruct();
        cv::write(storage, std::string(start_key), start);
        cv::write(storage, std::string(stop_key), stop);
        write_numbers(storage, frames_key, frame_entries, parameters.frames);
        cv::write(storage, std::string(end_key), std::string(name_of(parameters.end)));
        text = storage.releaseAndGetString();
    }
    catch (const cv::Exception &e)
    {
        return {false, path.string() + ": OpenCV could not write the JSON: " + e.err};
    }
    return write_file(path, text);
}

read_result<scan_parameters> read_parameters(const std::filesystem::path &path) noexcept
{
    read_result<std::string> file = read_file(path);
    if (!file.value)
        return {std::nullopt, file.reason};
    std::string reason;
    std::optional<scan_parameters> parameters;
    try
    {
        parameters = parse(*file.value, reason);
    }
    catch (const cv::Exception &e)
    {
        reason = "no JSON that OpenCV reads (" + e.err + ")";
    }
    if (!parameters)
        return {std::nullopt, path.string() + ": " + reason};
    return {std::move(parameters), {}};
}
} // namespace gca
