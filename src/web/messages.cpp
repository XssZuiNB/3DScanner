// SPDX-License-Identifier: 0BSD
/// @file messages.cpp
/// The web layer's messages: the state and a command's answer as JSON text, the live model as
/// bytes (see messages.hpp; docs/ROUTES.md, "Messages").

#include "web/messages.hpp"

#include "scan/file_io.hpp"
#include "scan/parameters_file.hpp"
#include "tracking/tracker.hpp"
#include "types/little_endian.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace gca
{
namespace
{
/// The bytes of the live model message's header, before its points: 80 (docs/ROUTES.md, "The live
/// model").
constexpr std::size_t live_model_header_bytes = 80;

/// The bytes the live model message takes for each point: its position, three floats, and its
/// color, three bytes: 15 (docs/ROUTES.md, "The live model").
constexpr std::size_t live_model_point_bytes = 15;

/// Appends a text as a JSON string (RFC 8259, section 7): in quotes, with the quote and the
/// backslash escaped, the five control characters JSON names (\b, \f, \n, \r, \t) by their
/// letter, every other control character below 0x20 as \u00XX. Every other byte is copied as it
/// is, so a text in UTF-8 stays UTF-8. OpenCV's JSON reader, the test's, reads the letter escapes
/// back and refuses \u [M 2026-11-17: cv2 4.8.0 on the board]; no message the scan interface writes
/// holds such a control character [S: its messages, plain English].
/// @param json [in,out] the JSON text appended to.
/// @param text [in] the text.
void append_string(std::string &json, std::string_view text)
{
    static constexpr char hex[] = "0123456789abcdef";
    json += '"';
    for (const char c : text)
    {
        const auto byte = static_cast<unsigned char>(c);
        switch (c)
        {
        case '"':
            json += "\\\"";
            break;
        case '\\':
            json += "\\\\";
            break;
        case '\b':
            json += "\\b";
            break;
        case '\f':
            json += "\\f";
            break;
        case '\n':
            json += "\\n";
            break;
        case '\r':
            json += "\\r";
            break;
        case '\t':
            json += "\\t";
            break;
        default:
            if (byte < 0x20)
            {
                json += "\\u00";
                json += hex[byte >> 4];
                json += hex[byte & 0xf];
            }
            else
                json += c;
        }
    }
    json += '"';
}

/// Appends a key, after a comma unless it is the object's first, and the colon its value follows.
/// @param json [in,out] the JSON text appended to; it ends with the object's "{" or a value.
/// @param key [in] the key, plain ASCII without a quote or a backslash.
void append_key(std::string &json, std::string_view key)
{
    if (json.back() != '{')
        json += ',';
    append_string(json, key);
    json += ':';
}

/// Appends a key and its string value.
/// @param json [in,out] the JSON text appended to; it ends with the object's "{" or a value.
/// @param key [in] the key, as append_key() takes it.
/// @param value [in] the value.
void append_field(std::string &json, std::string_view key, std::string_view value)
{
    append_key(json, key);
    append_string(json, value);
}

/// Appends a key and its whole number.
/// @param json [in,out] the JSON text appended to; it ends with the object's "{" or a value.
/// @param key [in] the key, as append_key() takes it.
/// @param value [in] the value, written in decimal; at most 2^53 to be read exactly by the page,
///        whose numbers are doubles [S: IEEE 754 binary64, a 53-bit significand].
void append_field(std::string &json, std::string_view key, std::uint64_t value)
{
    append_key(json, key);
    append_number(json, value);
}

/// Appends a key and its truth value.
/// @param json [in,out] the JSON text appended to; it ends with the object's "{" or a value.
/// @param key [in] the key, as append_key() takes it.
/// @param value [in] the value, written as true or false.
void append_flag(std::string &json, std::string_view key, bool value)
{
    append_key(json, key);
    json += value ? "true" : "false";
}
} // namespace

std::string state_message(const scan_state &state)
{
    std::string json = "{";
    append_field(json, "version", state.version);
    append_field(json, "phase", name_of(state.phase));
    append_field(json, "scan", state.scan);
    append_field(json, "tracking", state.tracking ? name_of(*state.tracking) : std::string_view());
    append_field(json, "lost_cause", name_of(state.lost_cause));
    append_flag(json, "too_close", state.too_close);
    append_key(json, "frames");
    json += '{';
    append_field(json, "taken", state.frames.taken);
    append_field(json, "fused", state.frames.fused);
    append_field(json, "lost", state.frames.lost);
    append_field(json, "not_taken", state.frames.not_taken);
    json += '}';
    append_field(json, "end", state.end ? name_of(*state.end) : std::string_view());
    append_field(json, "end_message", state.end_message);
    append_field(json, "outcome", name_of(state.outcome));
    append_field(json, "outcome_message", state.outcome_message);
    append_field(json, "refusal", state.refusal);
    append_field(json, "live_model_failure", state.live_model_failure);
    json += '}';
    return json;
}

std::string reply_message(const command_reply &reply)
{
    std::string json = "{";
    append_flag(json, "accepted", reply.accepted);
    append_field(json, "refusal", reply.refusal);
    json += '}';
    return json;
}

std::string live_model_message(const live_model &model)
{
    // The header: version at 0, frame at 8, has_frame at 16, n at 20, the pose's seven doubles at
    // 24; the positions at 80, three floats a point; the colors after them, three bytes a point.
    static_assert(24 + 7 * 8 == live_model_header_bytes);
    static_assert(3 * 4 + 3 == live_model_point_bytes);
    const std::size_t n = model.points.size();
    std::string message(live_model_header_bytes + live_model_point_bytes * n, '\0');
    // A char's bytes may be written through an unsigned char's pointer [S: C++20, [basic.lval]].
    auto *const bytes = reinterpret_cast<std::uint8_t *>(message.data());
    put_little_endian(bytes, model.version);
    put_little_endian(bytes + 8, model.frame.value_or(0));
    put_little_endian(bytes + 16, std::uint32_t{model.frame ? 1u : 0u});
    put_little_endian(bytes + 20, static_cast<std::uint32_t>(n));
    const Eigen::Vector3d &t = model.pose.position_m;
    const Eigen::Quaterniond &q = model.pose.orientation;
    const double pose[7] = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
    for (std::size_t k = 0; k < 7; k++)
        put_double(bytes + 24 + 8 * k, pose[k]);
    std::uint8_t *position = bytes + live_model_header_bytes;
    std::uint8_t *color = position + 12 * n;
    for (const model_point &p : model.points)
    {
        put_float(position, p.position_m.x());
        put_float(position + 4, p.position_m.y());
        put_float(position + 8, p.position_m.z());
        position += 12;
        color[0] = p.color[0];
        color[1] = p.color[1];
        color[2] = p.color[2];
        color += 3;
    }
    return message;
}
} // namespace gca
