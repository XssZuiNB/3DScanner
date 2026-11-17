// SPDX-License-Identifier: 0BSD
/// @file messages.cpp
/// The web layer's messages as JSON text (see messages.hpp; docs/ROUTES.md, "Messages").

#include "web/messages.hpp"

#include "scan/file_io.hpp"
#include "scan/parameters_file.hpp"
#include "tracking/tracker.hpp"

#include <cstdint>
#include <string_view>

namespace gca
{
namespace
{
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
} // namespace gca
