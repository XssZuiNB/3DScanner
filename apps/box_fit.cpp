// SPDX-License-Identifier: 0BSD
/// @file box_fit.cpp
/// The acceptance tool (R21): reads an object model's PLY (docs/SCAN_FILES.md, "Models"), fits
/// a box to it (box_measure.hpp) and prints its length, width and height in mm and, for each of
/// its five faces, the points and their spread about the face's plane; the hardware acceptance
/// reads these lines. Opens no camera and no GPU.

#include "box_fit_options.hpp"
#include "box_measure.hpp"
#include "scan/model_file.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

/// Measures the box in an object model: reads the command line, then the model, fits the box and
/// prints its lines.
/// @param argc [in] the argument count.
/// @param argv [in] the arguments (box_fit_options.hpp).
/// @return 0 after the box's lines; 1 when the model cannot be read ("box_fit: <the reader's
///     reason>") or is no box ("box_fit: <path>: no box: <reason>"), each on stderr; 2 when the
///     command line is refused.
int main(int argc, char *argv[])
{
    const std::optional<std::filesystem::path> path =
        gca::read_box_fit_options(argc, argv, std::cerr);
    if (!path)
        return 2;
    const gca::read_result<std::vector<gca::model_point>> model = gca::read_model(*path);
    if (!model.value)
    {
        std::cerr << "box_fit: " << model.reason << std::endl;
        return 1;
    }
    const gca::box_result result = gca::measure_box(*model.value);
    if (!result.measurement)
    {
        std::cerr << "box_fit: " << path->string() << ": no box: " << result.refusal << std::endl;
        return 1;
    }
    std::cout << gca::box_report(path->string(), *result.measurement) << std::endl;
    return 0;
}
