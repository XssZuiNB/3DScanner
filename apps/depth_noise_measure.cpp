// SPDX-License-Identifier: 0BSD
/// @file depth_noise_measure.cpp
/// The depth noise of a still sequence: the frames read twice, each pixel's sums, the pixels kept,
/// the depth bins, the drift between the halves, the plane fitted to the mean frame, the fit of c,
/// the refusals and the report's lines (see depth_noise_measure.hpp).

#include "depth_noise_measure.hpp"

#include "camera/frame.hpp"
#include "camera/sequence_format.hpp"
#include "camera/sequence_reader.hpp"
#include "separation/plane_fit.hpp"
#include "types/camera_param.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <span>
#include <sstream>

#include <Eigen/Core>

namespace gca
{
namespace
{
/// The least share of the frames in which a kept pixel has depth, as the fraction
/// kept_share_tenths / 10 [?: initial]: a pixel that loses its depth often lies on an edge, a dark
/// spot or a speckle the camera's own filter removes, whose noise is not the surface's [?].
constexpr std::uint64_t kept_share_tenths = 9;

/// A depth bin's width, in m [?: initial]: 26 bins over 0.4 m to 3 m [S: computed].
constexpr double bin_width_m = 0.1;

/// The least kept pixels a bin holds to count [?: initial]: one pixel's variance over 300 frames
/// has a relative deviation of 8.2 % [S: the chi-square distribution, (2 / 299)^(1/2)], so a bin's
/// mean of 1000 independent pixels deviates by about 0.26 % [S: computed, 8.2 % / 1000^(1/2)].
constexpr std::size_t bin_least_pixels = 1000;

/// The fewest bins the fit of c takes [?: initial]: three depths show whether sigma follows
/// c z^2.
constexpr std::size_t fit_least_bins = 3;

/// The largest drift ratio T of a still sequence [?: initial]: T lies near 1 when the frames'
/// noise is independent; at 3 with 300 frames, a drift alike at every pixel spread evenly over
/// the sequence raises sigma by 2 % [S: computed, docs/methods/depth-noise.md, "Still"].
constexpr double still_limit = 3.0;

/// The least noise of a bin the fit takes, in depth units [?: a choice]: from u / 2 on, the
/// rounding's share u^2 / 12 holds the rounded variance within 3.2 % of sigma^2 at any offset of
/// the true depth within its unit [S: computed, docs/methods/depth-noise.md, "The depth unit's
/// rounding"].
constexpr double least_noise_units = 0.5;

/// One pixel's sums over the frames, centred on its first depth value, so that the sums are
/// exact in 64-bit integers: a value is at most 65535, its square below 2^32.
struct pixel_sums
{
    std::array<std::int64_t, 2> sum{};    ///< the values less the first, in each half
    std::uint64_t squares = 0;            ///< their squares, over both halves
    std::array<std::uint32_t, 2> count{}; ///< the frames with depth in each half
    std::uint16_t first = 0;              ///< the first value with depth; 0 before it
};

/// One kept pixel's values over the frames.
struct kept_pixel
{
    double depth_m;       ///< its mean depth z, in m
    double variance_m2;   ///< its depth's variance over the frames s^2, in m^2
    std::uint32_t index;  ///< its index in the picture, v * width + u
    std::uint32_t frames; ///< the frames with depth n
    std::int64_t bin;     ///< its depth bin, floor(z / bin_width_m)
};

/// One depth bin's sums over its kept pixels.
struct bin_sums
{
    std::size_t pixels = 0;
    double variance_m2 = 0.0;   ///< the sum of s^2
    double noise_squared = 0.0; ///< the sum of (s^2 - u^2 / 12) / z^4, in 1/m^2
    double depth_m = 0.0;       ///< the sum of z
    double spatial_m2 = 0.0;    ///< the sum of e^2 - s^2 / n
};

/// Gives the ray of a pixel by the one pinhole convention (types/camera_param.hpp).
/// @param k [in] the intrinsics, in px, usable.
/// @param width [in] the picture's width, in px.
/// @param index [in] the pixel's index, v * width + u.
/// @return ((u - cx) / fx, (v - cy) / fy, 1), in the camera frame.
Eigen::Vector3d pixel_ray(const intrinsics &k, std::uint32_t width, std::uint32_t index)
{
    const double u = static_cast<double>(index % width);
    const double v = static_cast<double>(index / width);
    return Eigen::Vector3d((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0);
}

/// Reads the description from a sequence's header, so that its reader can be opened for its size.
/// @param path [in] the sequence's path.
/// @return the description; nothing when the header cannot be read or is refused, which the
///     reader's opening then names.
/// @note Allocates the file's stream; std::ifstream reports a failure to open or read by its state,
///       and a failed allocation ends the program (std::terminate), as its caller is noexcept.
std::optional<frame_description> header_description(const std::filesystem::path &path)
{
    std::array<std::uint8_t, sequence_header_bytes> bytes{};
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    const std::size_t read = file ? bytes.size() : static_cast<std::size_t>(file.gcount());
    return decode_sequence_header(std::span<const std::uint8_t>(bytes.data(), read)).description;
}

/// Reads a sequence once, to its end, handing each frame to a function.
/// @param reader [in,out] the sequence's reader, closed; open while it reads, closed after.
/// @param size [in] the frames' size, in px.
/// @param buffer [in,out] the frame the reader fills, reused across the readings.
/// @param take [in] what is done with each frame and its place k in the file's order, from 0.
/// @return empty when the reading reached the sequence's end; otherwise the reader's reason.
template <typename take_frame>
std::string read_all(sequence_reader &reader, const frame_size &size, frame &buffer,
                     take_frame &&take)
{
    const open_result opened = reader.open(size);
    if (!opened.opened)
        return opened.refusal;
    std::string reason;
    for (std::uint64_t k = 0;; k++)
    {
        const next_result next = reader.next(buffer);
        if (next.status == next_status::frame)
        {
            take(buffer, k);
            continue;
        }
        if (next.status != next_status::end)
            reason = next.message;
        break;
    }
    reader.close();
    return reason;
}

/// Formats a number as a stream does by default, six significant digits.
/// @param value [in] the number.
/// @return its text.
std::string number_text(double value)
{
    std::ostringstream out;
    out << value;
    return out.str();
}

/// Names the least share of the frames in which a kept pixel has depth.
/// @return the share as a percentage, "90 %".
std::string kept_share_text()
{
    return std::to_string(10 * kept_share_tenths) + " %";
}

/// Formats a number with a fixed count of decimals.
/// @param value [in] the number.
/// @param decimals [in] the decimals.
/// @return its text.
std::string fixed_text(double value, int decimals)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(decimals) << value;
    return out.str();
}
} // namespace

depth_noise_result measure_depth_noise(const std::filesystem::path &sequence) noexcept
{
    const auto refuse = [&](const std::string &reason) {
        return depth_noise_result{std::nullopt, sequence.string() + ": " + reason};
    };
    sequence_reader reader(sequence);
    const std::optional<frame_description> header = header_description(sequence);
    // Without a header, the reader's opening names what is wrong with the file.
    if (!header)
        return {std::nullopt, reader.open(frame_size{0, 0}).refusal};
    const frame_description description = *header;
    const frame_size size = description.size;
    frame buffer;
    std::uint64_t frames = 0;
    std::string reason =
        read_all(reader, size, buffer, [&](const frame &, std::uint64_t) { frames++; });
    if (!reason.empty())
        return {std::nullopt, reason};
    const double unit_m = description.depth_unit_m;
    if (!std::isfinite(unit_m) || unit_m <= 0.0)
        return refuse("its depth unit, " + number_text(unit_m) +
                      " m per value, is no positive number");
    const intrinsics &k = description.intrinsics;
    if (!usable_intrinsics(k))
        return refuse("its intrinsics, fx " + number_text(k.fx) + " fy " + number_text(k.fy) +
                      " cx " + number_text(k.cx) + " cy " + number_text(k.cy) +
                      " px, are no pinhole camera's");

    // Each pixel's sums, the frames before frames / 2 in the first half.
    const std::size_t pixel_count = std::size_t{size.width} * size.height;
    std::vector<pixel_sums> sums(pixel_count);
    reason = read_all(reader, size, buffer, [&](const frame &f, std::uint64_t index) {
        const std::size_t half = index < frames / 2 ? 0 : 1;
        const std::span<const std::uint16_t> depth = f.depth();
        for (std::size_t p = 0; p < pixel_count; p++)
        {
            const std::uint16_t value = depth[p];
            if (value == 0)
                continue;
            pixel_sums &s = sums[p];
            if (s.first == 0)
                s.first = value;
            const std::int64_t centred = std::int64_t{value} - s.first;
            s.sum[half] += centred;
            s.squares += static_cast<std::uint64_t>(centred * centred);
            s.count[half]++;
        }
    });
    if (!reason.empty())
        return {std::nullopt, reason};

    // The kept pixels, with the drift's sums over them and the bins' counts.
    std::vector<kept_pixel> kept;
    kept.reserve(pixel_count);
    double drift_squares = 0.0;
    double drift_variance = 0.0;
    std::map<std::int64_t, bin_sums> bins;
    for (std::size_t p = 0; p < pixel_count; p++)
    {
        const pixel_sums &s = sums[p];
        const std::uint64_t n = std::uint64_t{s.count[0]} + s.count[1];
        if (s.count[0] == 0 || s.count[1] == 0 || 10 * n < kept_share_tenths * frames)
            continue;
        const double total = static_cast<double>(s.sum[0] + s.sum[1]);
        const double nd = static_cast<double>(n);
        const double depth_m = unit_m * (s.first + total / nd);
        const double variance_m2 =
            unit_m * unit_m * (static_cast<double>(s.squares) - total * total / nd) / (nd - 1.0);
        const double drift_m = unit_m * (static_cast<double>(s.sum[1]) / s.count[1] -
                                         static_cast<double>(s.sum[0]) / s.count[0]);
        drift_squares += drift_m * drift_m;
        drift_variance += variance_m2 * (1.0 / s.count[0] + 1.0 / s.count[1]);
        const std::int64_t bin = static_cast<std::int64_t>(std::floor(depth_m / bin_width_m));
        kept.push_back({depth_m, variance_m2, static_cast<std::uint32_t>(p),
                        static_cast<std::uint32_t>(n), bin});
        bins[bin].pixels++;
    }
    std::size_t bins_counting = 0;
    for (const auto &[index, bin] : bins)
        if (bin.pixels >= bin_least_pixels)
            bins_counting++;
    if (bins_counting < fit_least_bins)
        return refuse("too few pixels: " + std::to_string(bins_counting) + " depth bins of " +
                      fixed_text(bin_width_m, 1) + " m hold at least " +
                      std::to_string(bin_least_pixels) + " of its " + std::to_string(kept.size()) +
                      " pixels kept, with depth in at least " + kept_share_text() + " of the " +
                      std::to_string(frames) + " frames and in both halves; the fit needs " +
                      std::to_string(fit_least_bins));

    const double drift_mm = 1000.0 * std::sqrt(drift_squares / static_cast<double>(kept.size()));
    const double drift_ratio =
        drift_variance > 0.0
            ? std::sqrt(drift_squares / drift_variance)
            : (drift_squares > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
    if (drift_ratio > still_limit)
        return refuse("not still: the halves differ by " + fixed_text(drift_mm, 2) + " mm rms, " +
                      fixed_text(drift_ratio, 2) +
                      " times their standard error; a still sequence stays within " +
                      number_text(still_limit) + " times");

    // The plane through the mean frame's points, and each pixel's offset from it along its ray.
    std::vector<Eigen::Vector3d> points;
    points.reserve(kept.size());
    for (const kept_pixel &pixel : kept)
        points.push_back(pixel.depth_m * pixel_ray(k, size.width, pixel.index));
    const std::optional<plane_fit> plane = fit_plane(points);
    // The bins that count hold at least 3000 points, so the fit has its three.
    if (!plane)
        return refuse("no plane fits the mean frame's " + std::to_string(points.size()) +
                      " points");
    const double rounding_m2 = unit_m * unit_m / 12.0;
    double spatial_m2 = 0.0;
    for (const kept_pixel &pixel : kept)
    {
        const Eigen::Vector3d ray = pixel_ray(k, size.width, pixel.index);
        const double offset_m =
            pixel.depth_m - plane->normal.dot(plane->centroid) / plane->normal.dot(ray);
        const double spatial = offset_m * offset_m - pixel.variance_m2 / pixel.frames;
        const double z2 = pixel.depth_m * pixel.depth_m;
        bin_sums &bin = bins[pixel.bin];
        bin.variance_m2 += pixel.variance_m2;
        bin.noise_squared += (pixel.variance_m2 - rounding_m2) / (z2 * z2);
        bin.depth_m += pixel.depth_m;
        bin.spatial_m2 += spatial;
        spatial_m2 += spatial;
    }

    // Each bin that counts, and the fit of c over those above the rounding.
    depth_noise_measurement m{};
    m.frames = frames;
    m.description = description;
    m.pixels_kept = kept.size();
    m.drift_mm = drift_mm;
    m.drift_ratio = drift_ratio;
    m.spatial_mm =
        1000.0 * std::sqrt(std::fmax(0.0, spatial_m2 / static_cast<double>(kept.size())));
    double weighted = 0.0;
    double weights = 0.0;
    for (const auto &[index, sums_of_bin] : bins)
    {
        if (sums_of_bin.pixels < bin_least_pixels)
            continue;
        const double pixels = static_cast<double>(sums_of_bin.pixels);
        const double noise_m =
            std::sqrt(std::fmax(0.0, sums_of_bin.variance_m2 / pixels - rounding_m2));
        depth_noise_bin bin{static_cast<double>(index) * bin_width_m,
                            static_cast<double>(index + 1) * bin_width_m,
                            sums_of_bin.pixels,
                            1000.0 * noise_m,
                            1000.0 * std::sqrt(std::fmax(0.0, sums_of_bin.spatial_m2 / pixels)),
                            std::nullopt};
        if (noise_m >= least_noise_units * unit_m)
        {
            const double c = std::sqrt(std::fmax(0.0, sums_of_bin.noise_squared / pixels));
            const double z = sums_of_bin.depth_m / pixels;
            bin.noise_per_m = c;
            weighted += c * z * z * z * z;
            weights += z * z * z * z;
        }
        m.bins.push_back(bin);
    }
    for (const depth_noise_bin &bin : m.bins)
        m.bins_fitted += bin.noise_per_m ? 1 : 0;
    if (m.bins_fitted < fit_least_bins)
        return refuse("below the rounding: the noise lies below " + number_text(least_noise_units) +
                      " depth units, " + fixed_text(1000.0 * least_noise_units * unit_m, 2) +
                      " mm, in " + std::to_string(m.bins.size() - m.bins_fitted) + " of the " +
                      std::to_string(m.bins.size()) +
                      " depth bins that hold enough pixels; the fit needs " +
                      std::to_string(fit_least_bins) + " above it");
    m.noise_per_m = weighted / weights;
    double spread = 0.0;
    for (const depth_noise_bin &bin : m.bins)
        if (bin.noise_per_m)
            spread += (*bin.noise_per_m - m.noise_per_m) * (*bin.noise_per_m - m.noise_per_m);
    m.spread_per_m = std::sqrt(spread / static_cast<double>(m.bins_fitted));
    return {m, {}};
}

std::string depth_noise_report(const std::string &name, const depth_noise_measurement &m)
{
    const frame_size &size = m.description.size;
    std::ostringstream out;
    out << "depth_noise: " << name << ": " << m.frames << " frames of " << size.width << "x"
        << size.height << ", depth unit " << number_text(1000.0 * m.description.depth_unit_m)
        << " mm\n";
    out << "pixels: " << m.pixels_kept << " of " << std::size_t{size.width} * size.height
        << " kept, with depth in at least " << kept_share_text()
        << " of the frames and in both halves\n";
    out << std::fixed << std::setprecision(2);
    out << "still: the halves differ by " << m.drift_mm << " mm rms, " << m.drift_ratio
        << " times their standard error\n";
    out << "spatial: " << m.spatial_mm << " mm rms from the plane fitted to the mean frame\n";
    for (const depth_noise_bin &bin : m.bins)
    {
        out << std::setprecision(1) << "bin " << bin.near_m << " to " << bin.far_m
            << " m: " << bin.pixels << " pixels, noise " << std::setprecision(2) << bin.noise_mm
            << " mm, spatial " << bin.spatial_mm << " mm, ";
        if (bin.noise_per_m)
            out << "c " << std::setprecision(6) << *bin.noise_per_m << " /m\n";
        else
            out << "below the rounding\n";
    }
    out << std::setprecision(6) << "c: " << m.noise_per_m << " /m, spread " << m.spread_per_m
        << " /m over " << m.bins_fitted << " bins";
    return out.str();
}
} // namespace gca
