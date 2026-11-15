// SPDX-License-Identifier: 0BSD
/// @file depth_noise.cpp
/// The depth noise's c from a still sequence (docs/methods/depth-noise.md): reads
/// DIR/frames.sequence (docs/SEQUENCE_FILE.md), taken with the camera at rest, measures it
/// (depth_noise_measure.hpp) and prints c of sigma(z) = c z^2 with its spread over depth bins, each
/// bin's noise and the spatial error; the measurement on the camera reads these lines. Opens no
/// camera and no GPU.

#include "depth_noise_measure.hpp"
#include "depth_noise_options.hpp"

#include <filesystem>
#include <iostream>
#include <optional>

/// Measures the depth noise of a still sequence: reads the command line, then the sequence, and
/// prints its lines.
/// @param argc [in] the argument count.
/// @param argv [in] the arguments (depth_noise_options.hpp).
/// @return 0 after the measurement's lines; 1 when the sequence is refused, "depth_noise: " and
///     the refusal, naming the sequence, on stderr (depth_noise_measure.hpp); 2 when the command
///     line is refused.
int main(int argc, char *argv[])
{
    const std::optional<std::filesystem::path> directory =
        gca::read_depth_noise_options(argc, argv, std::cerr);
    if (!directory)
        return 2;
    const gca::depth_noise_result result = gca::measure_depth_noise(*directory / "frames.sequence");
    if (!result.measurement)
    {
        std::cerr << "depth_noise: " << result.refusal << std::endl;
        return 1;
    }
    std::cout << gca::depth_noise_report(directory->string(), *result.measurement) << std::endl;
    return 0;
}
