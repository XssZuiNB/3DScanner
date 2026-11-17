// SPDX-License-Identifier: 0BSD
/// @file messages.hpp
/// The web layer's messages (docs/ROUTES.md, "Messages"): the scan interface's state and a
/// command's answer as the JSON text the server writes, the layout the page reads. The server
/// only writes JSON; it reads none (a command and its arguments come in the request line). Each
/// message is one JSON object on one line, its keys in the document's order; a string is the scan
/// interface's text as it stands, escaped as JSON asks; a value the state does not hold is the
/// empty string, never null.

#pragma once

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
} // namespace gca
