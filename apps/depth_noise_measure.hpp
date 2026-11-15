// SPDX-License-Identifier: 0BSD
/// @file depth_noise_measure.hpp
/// depth_noise's measurement (docs/methods/depth-noise.md): the depth noise's coefficient c of
/// sigma(z) = c z^2, which sets the sparse TSDF's truncation (docs/methods/sparse-tsdf.md, "The
/// truncation distance"), from a sequence taken with the camera at rest (docs/SEQUENCE_FILE.md).
/// It reads the sequence twice, sums each pixel's depth over the frames, keeps the pixels with
/// depth in nearly every frame, sorts them into depth bins by their mean depth, takes each bin's
/// noise with the depth unit's rounding removed, fits c over the bins and gives the mean frame's
/// offset from a plane fitted to it, the spatial error; it refuses a sequence it cannot read, a
/// depth unit or intrinsics that cannot be used, too few pixels, a camera that was not still and
/// noise below the depth unit's rounding. Host only, in double, without the camera and the GPU,
/// so that the offline tests and the measurement on the camera use it alike.

#pragma once

#include "types/frame_description.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gca
{
/// One depth bin that holds enough pixels to count: at least 1000 kept pixels.
struct depth_noise_bin
{
    double near_m;      ///< the depth its pixels' mean depths reach from, in m, a multiple of 0.1
    double far_m;       ///< the depth they stay below, near_m + 0.1, in m
    std::size_t pixels; ///< its kept pixels, >= 1000
    /// Its noise sigma_b, the root of its pixels' mean variance over the frames less the depth
    /// unit's rounding share u^2 / 12, in mm, >= 0; 0 when the rounding's share is the larger.
    double noise_mm;
    /// Its pixels' offset from the plane fitted to the mean frame, along their rays, the root
    /// mean square with the temporal noise's share removed, in mm, >= 0.
    double spatial_mm;
    /// Its own c_b, in 1/m, >= 0; nothing when its noise lies below half the depth unit, where the
    /// rounding's share cannot be told from the noise, and the fit leaves it out.
    std::optional<double> noise_per_m;
};

/// What a still sequence gives.
struct depth_noise_measurement
{
    std::uint64_t frames;          ///< the frames read, N
    frame_description description; ///< the frames' size, intrinsics and depth unit, from the header
    /// The pixels kept: with depth in at least 90 % of the frames and in both halves.
    std::size_t pixels_kept;
    /// The root mean square over the kept pixels of the second half's mean depth less the
    /// first's, in mm, >= 0.
    double drift_mm;
    /// That drift over its standard error for a still camera, T, >= 0: near 1 when still, at most 3
    /// in a measurement.
    double drift_ratio;
    /// The mean frame's offset from the plane fitted to it, over every kept pixel, as a bin's.
    double spatial_mm;
    std::vector<depth_noise_bin> bins; ///< every bin that counts, nearest first
    std::size_t bins_fitted;           ///< the bins with a c_b, which the fit takes, >= 3
    /// c, in 1/m, >= 0: the least-squares fit of sigma = c z^2 to the bins with a c_b.
    double noise_per_m;
    /// The root mean square of c_b - c over the bins with a c_b, in 1/m, >= 0.
    double spread_per_m;
};

/// What measure_depth_noise() gave.
struct depth_noise_result
{
    std::optional<depth_noise_measurement> measurement; ///< the measurement; nothing when refused
    /// Why the sequence is refused, in plain English, naming the sequence; empty when measured.
    std::string refusal;
};

/// Measures the depth noise of a still sequence (the method: docs/methods/depth-noise.md). Reads
/// the sequence twice, through the sequence's reader (camera/sequence_reader.hpp): once to count
/// its frames, once to sum them.
/// @param sequence [in] the sequence's path, a frames.sequence file (docs/SEQUENCE_FILE.md).
/// @return the measurement; or the refusal, "<path>: " and the reason, in this order: the
///     sequence cannot be read, with the reader's reason (a sequence that ends early included);
///     "its depth unit, <u> m per value, is no positive number"; "its intrinsics, fx <fx> fy <fy>
///     cx <cx> cy <cy> px, are no pinhole camera's"; "too few pixels: <B> depth bins of 0.1 m
///     hold at least 1000 of its <K> pixels kept, with depth in at least 90 % of the <N> frames
///     and in both halves; the fit needs 3"; "not still: the halves differ by <D> mm rms, <T>
///     times their standard error; a still sequence stays within 3 times"; "no plane fits the
///     mean frame's <K> points", which the bins that count, at least 3000 points, leave no room
///     for; "below the rounding: the noise lies below 0.5 depth units, <u/2> mm, in <k> of the <B>
///     depth bins that hold enough pixels; the fit needs 3 above it".
/// @note Host only. Allocates about 101 bytes for each pixel of the frames [S: computed from the
///       vectors it holds at once]: a frame (5), each pixel's sums (40), and each kept pixel's
///       values (32) and point (24); 103 MB at 1280x800. A failed allocation ends the program
///       (std::terminate): the function is noexcept. Holds no state: two threads may measure at
///       once.
depth_noise_result measure_depth_noise(const std::filesystem::path &sequence) noexcept;

/// Writes the measurement's lines, which the measurement on the camera reads (AGENTS.md §7
/// rule 5):
///     depth_noise: <NAME>: <N> frames of <W>x<H>, depth unit <u> mm
///     pixels: <K> of <P> kept, with depth in at least 90 % of the frames and in both halves
///     still: the halves differ by <D> mm rms, <T> times their standard error
///     spatial: <S> mm rms from the plane fitted to the mean frame
/// then for each bin "bin <a> to <b> m: <n> pixels, noise <sigma> mm, spatial <s> mm, c <c_b> /m",
/// or "below the rounding" in place of its c, and last "c: <c> /m, spread <r> /m over <B> bins";
/// lengths in mm and T to 0.01, the bins' ends to 0.1 m, c, c_b and the spread to 0.000001 /m.
/// @param name [in] the sequence's name, its directory as depth_noise was given it.
/// @param measurement [in] the measurement.
/// @return the lines, joined by "\n", without a line end at the end.
std::string depth_noise_report(const std::string &name, const depth_noise_measurement &measurement);
} // namespace gca
