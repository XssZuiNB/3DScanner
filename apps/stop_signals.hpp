// SPDX-License-Identifier: 0BSD
/// @file stop_signals.hpp
/// The signals that stop the programs that hold the camera, orbbec_viewer and the scanner service,
/// with their names: Ctrl-C (SIGINT), a kill or systemd's stop (SIGTERM) and a closed terminal or
/// SSH session (SIGHUP). Each program installs its own way of taking them (orbbec_viewer.cpp,
/// service.hpp); the set is one. Header only.

#pragma once

#include <array>
#include <csignal>

namespace gca
{
/// A stop signal and its name.
struct stop_signal
{
    int number;       ///< the signal's number
    const char *name; ///< its name for a program's lines, a static string
};

/// The stop signals, the one set both programs take.
inline constexpr std::array<stop_signal, 3> stop_signals = {{
    {SIGINT, "SIGINT"},
    {SIGTERM, "SIGTERM"},
    {SIGHUP, "SIGHUP"},
}};
} // namespace gca
