// SPDX-License-Identifier: 0BSD
/// @file console_color.hpp
/// The colors of the programs' console lines: ANSI escape sequences, each a Select Graphic
/// Rendition (SGR) code [S: ECMA-48, SGR], written before a line's text and RESET after it. The
/// offline tests take them out before they compare a line. Header only.

#pragma once

/// Ends a colored text: SGR 0, every attribute back to its default.
#define RESET "\033[0m"
/// A failure, a refusal or a stop condition: SGR 31, red foreground.
#define RED "\033[31m"
/// A file saved: SGR 32, green foreground.
#define GREEN "\033[32m"
/// A warning the run goes on after: SGR 33, yellow foreground.
#define YELLOW "\033[33m"
/// What the camera found and chose: SGR 36, cyan foreground.
#define D_GREEN "\033[36m"
