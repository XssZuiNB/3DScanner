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

/// The largest budget JSON holds exactly: 2^53 bytes, up to which every whole number is a double
/// [S: IEEE 754 binary64, a 53-bit significand].
constexpr std::uint64_t largest_budget = std::uint64_t{1} << 53;

/// The longest string FileStorage reads back: it refuses one of 4096 characters when it reads it
/// and throws when it writes one of 4097 [M 2026-11-14: OpenCV 4.8.0].
constexpr std::size_t longest_string = 4095;

/// The keys of the top level.
constexpr std::string_view version_key = "format_version";
constexpr std::string_view commit_key = "commit";
constexpr std::string_view class_key = "object_class";
constexpr std::string_view set_key = "parameters";

/// The keys of the top level, in the order the writer writes them.
constexpr std::array<std::string_view, 4> top_keys = {version_key, commit_key, class_key, set_key};

/// What a value of the set of parameters may be (docs/SCAN_FILES.md, "Parameters").
enum class value_rule
{
    length, ///< a length or a threshold: finite and > 0
    share,  ///< a share: finite, > 0 and at most 1
    budget, ///< the memory budget: a whole number from 1 to 2^53
    count,  ///< a count of iterations: a whole number from 0 to 2^32 - 1
};

/// A value of the set of parameters: its key, its rule, and how it is read from a set and written
/// into one, as a double: the budget and a count exactly, the budget being at most 2^53 once
/// set_refusal() admits it and a count at most 2^32 - 1, which read_set() checks before it writes
/// one [S: IEEE 754 binary64, a 53-bit significand].
struct set_entry
{
    std::string_view key;                           ///< the key in "parameters"
    value_rule rule;                                ///< what its value may be
    double (*get)(const object_class_parameters &); ///< the member's value
    void (*put)(object_class_parameters &, double); ///< sets the member to a value its rule admits
};

/// Gives the entry of a member of the set.
/// @tparam member the member, a double or an integer.
/// @param key [in] its key.
/// @param rule [in] its rule.
/// @return the entry, which reads and writes that member.
template <auto member> constexpr set_entry entry_of(std::string_view key, value_rule rule)
{
    return {key, rule,
            [](const object_class_parameters &set) { return static_cast<double>(set.*member); },
            [](object_class_parameters &set, double value) {
                set.*member = static_cast<std::remove_reference_t<decltype(set.*member)>>(value);
            }};
}

/// Gives the entry of one level's element of an array member of the set.
/// @tparam member the array member, of doubles or of integers.
/// @tparam level the element, a level of the pyramid.
/// @param key [in] its key.
/// @param rule [in] its rule.
/// @return the entry, which reads and writes that element.
template <auto member, std::size_t level>
constexpr set_entry level_entry_of(std::string_view key, value_rule rule)
{
    return {key, rule,
            [](const object_class_parameters &set) {
                return static_cast<double>((set.*member)[level]);
            },
            [](object_class_parameters &set, double value) {
                (set.*member)[level] =
                    static_cast<std::remove_reference_t<decltype((set.*member)[level])>>(value);
            }};
}

/// The keys the depths' order and the budget's check name.
constexpr std::string_view min_depth_key = "min_depth_m";
constexpr std::string_view max_depth_key = "max_depth_m";
constexpr std::string_view budget_key = "memory_budget_bytes";
constexpr std::string_view too_close_key = "too_close_depth_m";

/// The set's type, by a short name for the table below.
using class_set = object_class_parameters;

/// The set of parameters' one table: each key with its rule and its member, in the order the
/// writer writes them.
constexpr std::array<set_entry, 18> set_entries = {
    entry_of<&class_set::live_voxel_size_m>("live_voxel_size_m", value_rule::length),
    entry_of<&class_set::full_voxel_size_m>("full_voxel_size_m", value_rule::length),
    entry_of<&class_set::min_depth_m>(min_depth_key, value_rule::length),
    entry_of<&class_set::max_depth_m>(max_depth_key, value_rule::length),
    entry_of<&class_set::memory_budget_bytes>(budget_key, value_rule::budget),
    entry_of<&class_set::too_close_depth_m>(too_close_key, value_rule::length),
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

/// The largest count of iterations the set holds: 2^32 - 1, std::uint32_t's.
constexpr double largest_count = 4294967295.0;

/// The values of a set, in set_entries' order.
using set_values = std::array<double, set_entries.size()>;

/// Gives the keys of the set of parameters, in the writer's order.
/// @return set_entries' keys.
constexpr std::array<std::string_view, set_entries.size()> keys_of_set()
{
    std::array<std::string_view, set_entries.size()> keys{};
    for (std::size_t k = 0; k < set_entries.size(); k++)
        keys[k] = set_entries[k].key;
    return keys;
}

/// The keys of the set of parameters, in the order the writer writes them.
constexpr std::array<std::string_view, set_entries.size()> parameter_keys = keys_of_set();

/// Gives a set's values in set_entries' order.
/// @param set [in] the set.
/// @return its values, each as set_entry::get gives it.
set_values values_of(const object_class_parameters &set)
{
    set_values values{};
    for (std::size_t k = 0; k < set_entries.size(); k++)
        values[k] = set_entries[k].get(set);
    return values;
}

/// Gives the set of values in set_entries' order, the inverse of values_of().
/// @param values [in] the values; the budget a whole number from 1 to 2^53 and each count a whole
///        number from 0 to 2^32 - 1, which read_set() checks before.
/// @return the set.
object_class_parameters set_of(const set_values &values)
{
    object_class_parameters set{};
    for (std::size_t k = 0; k < set_entries.size(); k++)
        set_entries[k].put(set, values[k]);
    return set;
}

/// Quotes a key for a reason.
/// @param key [in] the key.
/// @return the key in double quotes.
std::string in_quotes(std::string_view key)
{
    return "\"" + std::string(key) + "\"";
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

/// Checks that a set of parameters is one the scanner can run with: each length and threshold
/// finite and > 0, each share finite, > 0 and at most 1, the depths in order and the budget in
/// range; the stages' own ranges are checked when a scan requests its buffers.
/// @param set [in] the set.
/// @return the reason it is not, naming the key; empty when it is.
std::string set_refusal(const object_class_parameters &set)
{
    const set_values values = values_of(set);
    for (std::size_t k = 0; k < set_entries.size(); k++)
    {
        const double value = values[k];
        const bool positive = std::isfinite(value) && value > 0.0;
        if (set_entries[k].rule == value_rule::length && !positive)
            return in_quotes(set_entries[k].key) + " is " + number_text(value) +
                   ", not a finite number > 0";
        if (set_entries[k].rule == value_rule::share && !(positive && value <= 1.0))
            return in_quotes(set_entries[k].key) + " is " + number_text(value) +
                   ", not a number in (0, 1]";
    }
    if (!(set.min_depth_m < set.max_depth_m))
        return in_quotes(min_depth_key) + " " + number_text(set.min_depth_m) +
               " is not less than " + in_quotes(max_depth_key) + " " + number_text(set.max_depth_m);
    if (!(set.min_depth_m < set.too_close_depth_m && set.too_close_depth_m < set.max_depth_m))
        return in_quotes(too_close_key) + " " + number_text(set.too_close_depth_m) +
               " does not lie between " + in_quotes(min_depth_key) + " " +
               number_text(set.min_depth_m) + " and " + in_quotes(max_depth_key) + " " +
               number_text(set.max_depth_m);
    // Compared as integers: 2^53 + 1 converted to a double rounds to 2^53.
    if (set.memory_budget_bytes < 1 || set.memory_budget_bytes > largest_budget)
        return in_quotes(budget_key) + " is " + std::to_string(set.memory_budget_bytes) +
               ", not a whole number from 1 to 2^53";
    return {};
}

/// Checks that a string is one the file holds. OpenCV's JSON writer gives text that both its own
/// reader and Python's json.loads take back unchanged only for some strings [M 2026-11-14:
/// OpenCV 4.8.0]: it writes an apostrophe as \' (no JSON escape), a string that starts and ends
/// with a double quote raw (OpenCV's reader then drops the quotes), a control byte raw (no JSON),
/// and a string of more than 4095 characters does not read back. So the file holds strings of
/// printable ASCII (0x20 to 0x7e) without the apostrophe, not bounded by double quotes, of at
/// most 4095 characters; a byte above 0x7e is refused too, as the reader checks no UTF-8.
/// @param key [in] the key, for the reason.
/// @param value [in] the string.
/// @return the reason it is not, naming the key; empty when it is.
std::string string_refusal(std::string_view key, std::string_view value)
{
    static const char hex[] = "0123456789abcdef";
    if (value.size() > longest_string)
        return in_quotes(key) + " holds " + std::to_string(value.size()) +
               " characters, more than 4095";
    for (std::size_t k = 0; k < value.size(); k++)
    {
        const unsigned char byte = static_cast<unsigned char>(value[k]);
        if (byte < 0x20 || byte > 0x7e || byte == '\'')
            return in_quotes(key) + " holds the byte 0x" + hex[byte >> 4] + hex[byte & 0xf] +
                   " at " + std::to_string(k) +
                   ", not printable ASCII without the apostrophe, which OpenCV does not write as "
                   "JSON";
    }
    if (!value.empty() && value.front() == '"' && value.back() == '"')
        return in_quotes(key) +
               " starts and ends with a double quote, which OpenCV does not write as JSON";
    return {};
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

/// Reads the set of parameters from its map, whose keys keys_refusal() has checked. FileStorage
/// reads a string asked for as a number as DBL_MAX, and an integer beyond 2147483647 wrapped to
/// 32 bits, without a word [M 2026-11-14: OpenCV 4.8.0]; so each parameter is taken only from a
/// real number, which FileStorage writes with a decimal point or an exponent and reads back bit
/// for bit, the budget and the counts too.
/// @param map [in] the map "parameters".
/// @param reason [out] why the set was refused, naming the key; set only on a refusal.
/// @return the set, or nothing when a value is no real number, the budget is no whole number
///         from 1 to 2^53, a count no whole number from 0 to 2^32 - 1, or set_refusal() refuses
///         the set.
std::optional<object_class_parameters> read_set(const cv::FileNode &map, std::string &reason)
{
    set_values values;
    for (std::size_t k = 0; k < set_entries.size(); k++)
    {
        const cv::FileNode node = map[std::string(set_entries[k].key)];
        if (!node.isReal())
        {
            reason = in_quotes(set_entries[k].key) + " in " + in_quotes(set_key) +
                     " is no number with a decimal point or an exponent";
            return std::nullopt;
        }
        values[k] = static_cast<double>(node);
    }
    // The budget and the counts are checked before they are converted: a double outside the
    // integer's range does not convert.
    for (std::size_t k = 0; k < set_entries.size(); k++)
    {
        const double value = values[k];
        const bool whole = value == std::floor(value);
        if (set_entries[k].rule == value_rule::budget &&
            !(value >= 1.0 && value <= static_cast<double>(largest_budget) && whole))
        {
            reason = in_quotes(set_entries[k].key) + " is " + number_text(value) +
                     ", not a whole number from 1 to 2^53";
            return std::nullopt;
        }
        if (set_entries[k].rule == value_rule::count &&
            !(value >= 0.0 && value <= largest_count && whole))
        {
            reason = in_quotes(set_entries[k].key) + " is " + number_text(value) +
                     ", not a whole number from 0 to 4294967295";
            return std::nullopt;
        }
    }
    const object_class_parameters set = set_of(values);
    reason = set_refusal(set);
    if (!reason.empty())
        return std::nullopt;
    return set;
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
    const cv::FileNode commit = root[std::string(commit_key)];
    const cv::FileNode class_name = root[std::string(class_key)];
    const cv::FileNode set_map = root[std::string(set_key)];
    if (!commit.isString())
        reason = in_quotes(commit_key) + " is no string";
    else if (!class_name.isString())
        reason = in_quotes(class_key) + " is no string";
    else if (!set_map.isMap())
        reason = in_quotes(set_key) + " is no JSON object";
    else
        reason = string_refusal(commit_key, commit.string());
    if (!reason.empty())
        return std::nullopt;
    const std::optional<gca::object_class> kind = object_class_named(class_name.string());
    if (!kind)
    {
        reason = in_quotes(class_key) + " is " + in_quotes(class_name.string()) +
                 ", no object class (tabletop_object, large_object)";
        return std::nullopt;
    }
    reason = keys_refusal(set_map, parameter_keys, "in " + in_quotes(set_key));
    if (!reason.empty())
        return std::nullopt;
    const std::optional<object_class_parameters> set = read_set(set_map, reason);
    if (!set)
        return std::nullopt;
    return scan_parameters{commit.string(), *kind, *set};
}
} // namespace

write_result write_parameters(const std::filesystem::path &path,
                              const scan_parameters &parameters) noexcept
{
    const object_class_parameters &set = parameters.parameters;
    std::string refusal = set_refusal(set);
    if (refusal.empty())
        refusal = string_refusal(commit_key, parameters.commit);
    if (!refusal.empty())
        return {false, path.string() + ": " + refusal};
    std::string text;
    try
    {
        cv::FileStorage storage(".json", cv::FileStorage::WRITE | cv::FileStorage::MEMORY |
                                             cv::FileStorage::FORMAT_JSON);
        // Each string is one string_refusal() admits, which OpenCV writes as JSON.
        cv::write(storage, std::string(version_key), format_version);
        cv::write(storage, std::string(commit_key), parameters.commit);
        cv::write(storage, std::string(class_key), std::string(name_of(parameters.object_class)));
        storage.startWriteStruct(std::string(set_key), cv::FileNode::MAP);
        // Exact: set_refusal() keeps the budget at most 2^53, and a count is at most 2^32 - 1.
        const set_values values = values_of(set);
        for (std::size_t k = 0; k < set_entries.size(); k++)
            cv::write(storage, std::string(set_entries[k].key), values[k]);
        storage.endWriteStruct();
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
