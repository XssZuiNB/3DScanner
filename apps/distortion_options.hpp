// SPDX-License-Identifier: 0BSD
/// @file distortion_options.hpp
/// distortion_check's command line: seven options, each required and each with a value, and the
/// two PNG files of an orbbec_viewer snapshot; the reader refuses an argument it does not know,
/// a value it cannot read and a missing part with the reason and the usage. Host only, without
/// the camera and the GPU, so that the offline tests drive it.

#pragma once

#include "checkerboard_measure.hpp"

#include <optional>
#include <ostream>
#include <string>

namespace gca
{
/// distortion_check's options; no default, read_distortion_options() needs each.
struct distortion_options
{
    intrinsics camera{};        ///< --fx, --fy, --cx, --cy, px
    double depth_scale_m = 0.0; ///< --depth-scale, m per raw depth value
    checkerboard board{};       ///< --square-mm (mm) and --corners (COLSxROWS)
    std::string color_path;     ///< the first path: the snapshot's color PNG
    std::string depth_path;     ///< the second path: its aligned depth PNG
};

/// Builds the usage, which names every option with its unit and where its value comes from;
/// read_distortion_options() prints it with a refusal.
/// @return the usage, lines joined by "\n", without a line end at the end.
std::string distortion_usage();

/// Reads the command line: --fx, --fy, --cx, --cy (px), --depth-scale (m/unit), --square-mm
/// (mm), each a finite number, and --corners COLSxROWS (two whole numbers joined by an x), in
/// any order, and the color and the depth PNG, in this order. The values' ranges are
/// measure_board()'s to check.
/// @param argc [in] the argument count.
/// @param argv [in] the arguments, argv[0] the program; the caller owns them.
/// @param err [in,out] where a refusal is printed: "distortion_check: <reason>", then the usage.
/// @return the options; nothing when the command line is refused: an argument it does not know
///     (one that starts with "-", --help among them, an empty one, a third path), an option
///     without its value, a value it cannot read, a missing option, a missing path.
std::optional<distortion_options> read_distortion_options(int argc, const char *const argv[],
                                                          std::ostream &err);
} // namespace gca
