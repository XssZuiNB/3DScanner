// SPDX-License-Identifier: 0BSD
/// @file messages.hpp
/// The web layer's messages (docs/ROUTES.md, "Messages"): the scan interface's state and a
/// command's answer as the JSON text the server writes, and the live model as the binary message
/// the WebSocket pushes; the layouts the page reads. The server only writes JSON; it reads none (a
/// command and its arguments come in the request line). Each JSON message is one object on one
/// line, its keys in the document's order; a string is the scan interface's text as it stands,
/// escaped as JSON asks; a value the state does not hold is the empty string, never null.

#pragma once

#include "scan/live_model.hpp"
#include "scan/scan_interface.hpp"

#include <string>

namespace gca
{
/// Writes the state message (docs/ROUTES.md, "The state").
/// @param state [in] the state, a snapshot of the scan interface's.
/// @return the message: every field of the state, a missing tracking state and a missing end as
///         the empty string; an enumerator that is no value of its enumeration as the empty
///         string too.
/// @note Allocates the text; a failed allocation throws std::bad_alloc. Any thread may call it.
std::string state_message(const scan_state &state);

/// Writes the answer message of a command (docs/ROUTES.md, "A command's answer").
/// @param reply [in] the answer, the scan interface's or the web layer's own refusal.
/// @return the message: whether the command was taken and the refusal, empty when it was.
/// @note As state_message().
std::string reply_message(const command_reply &reply);

/// Lays out the live model message (docs/ROUTES.md, "The live model"), little-endian: its version,
/// its frame, whether it has one, the count of its points, the pose, then every point's position,
/// then every point's color, in the order of the live model's points. Its normals, its spacing and
/// when it was taken are not sent.
/// @param model [in] the live model, a snapshot of the scan interface's; at most 2^32 - 1 points,
///        which its limit keeps (scan_options::live_model_points).
/// @return the message, 80 bytes and 15 bytes a point: 3.0 MB for 200000 points [S: computed]. A
/// live model without a frame (the empty one a scan starts
///         with) has the frame 0, the identity pose as it holds it, and no point.
/// @note Allocates the message; a failed allocation throws std::bad_alloc. Any thread may call it.
std::string live_model_message(const live_model &model);
} // namespace gca
