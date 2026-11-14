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

/// A length of the set of parameters: its key and its member.
struct length_parameter
{
    std::string_view key;                    ///< the key in "parameters"
    double object_class_parameters::*member; ///< the member it is read into
};

/// The lengths of the set, in the order the writer writes them; the budget follows them.
constexpr std::array<length_parameter, 4> lengths = {{
    {"live_voxel_size_m", &object_class_parameters::live_voxel_size_m},
    {"full_voxel_size_m", &object_class_parameters::full_voxel_size_m},
    {"min_depth_m", &object_class_parameters::min_depth_m},
    {"max_depth_m", &object_class_parameters::max_depth_m},
}};

/// The budget's key in "parameters".
constexpr std::string_view budget_key = "memory_budget_bytes";

/// The keys of the set of parameters, in the order the writer writes them.
constexpr std::array<std::string_view, 5> parameter_keys = {
    lengths[0].key, lengths[1].key, lengths[2].key, lengths[3].key, budget_key};

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

/// Checks that a set of parameters is one the scanner can run with.
/// @param set [in] the set.
/// @return the reason it is not, naming the key; empty when it is.
std::string set_refusal(const object_class_parameters &set)
{
    for (const length_parameter &length : lengths)
    {
        const double value = set.*length.member;
        if (!(std::isfinite(value) && value > 0.0))
            return in_quotes(length.key) + " is " + number_text(value) +
                   ", not a finite number > 0";
    }
    if (!(set.min_depth_m < set.max_depth_m))
        return in_quotes(lengths[2].key) + " " + number_text(set.min_depth_m) +
               " is not less than " + in_quotes(lengths[3].key) + " " +
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
/// for bit.
/// @param map [in] the map "parameters".
/// @param reason [out] why the set was refused, naming the key; set only on a refusal.
/// @return the set, or nothing when a value is no real number, the budget is no whole number
///         from 1 to 2^53, or set_refusal() refuses the set.
std::optional<object_class_parameters> read_set(const cv::FileNode &map, std::string &reason)
{
    std::array<double, parameter_keys.size()> values;
    for (std::size_t k = 0; k < parameter_keys.size(); k++)
    {
        const cv::FileNode node = map[std::string(parameter_keys[k])];
        if (!node.isReal())
        {
            reason = in_quotes(parameter_keys[k]) + " in " + in_quotes(set_key) +
                     " is no number with a decimal point or an exponent";
            return std::nullopt;
        }
        values[k] = static_cast<double>(node);
    }
    // The budget is checked before it is converted: a double outside the integer's range does
    // not convert.
    const double budget = values.back();
    if (!(budget >= 1.0 && budget <= static_cast<double>(largest_budget) &&
          budget == std::floor(budget)))
    {
        reason = in_quotes(budget_key) + " is " + number_text(budget) +
                 ", not a whole number from 1 to 2^53";
        return std::nullopt;
    }
    object_class_parameters set{};
    for (std::size_t k = 0; k < lengths.size(); k++)
        set.*lengths[k].member = values[k];
    set.memory_budget_bytes = static_cast<std::uint64_t>(budget);
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
        for (const length_parameter &length : lengths)
            cv::write(storage, std::string(length.key), set.*length.member);
        // Exact: set_refusal() keeps the budget at most 2^53.
        cv::write(storage, std::string(budget_key), static_cast<double>(set.memory_budget_bytes));
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
