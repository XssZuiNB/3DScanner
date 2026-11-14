// SPDX-License-Identifier: 0BSD
/// @file synthetic_source.cpp
/// The tests' synthetic source: the scene's texture, the renderer and the source's states (see
/// synthetic_source.hpp).

#include "synthetic_source.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>

namespace gca_test
{
namespace
{
/// The time between two frames, in µs: 30 fps [?: a choice, the camera's rate].
constexpr std::uint64_t frame_period_us = 33333;

/// One sine wave of the plane's texture.
struct texture_wave
{
    double direction_deg; ///< its direction in the plane, from the world's x toward its y
    double wavelength_m;  ///< its wavelength, in m
    double phase_rad;     ///< its phase at the world's origin
};

/// The texture's waves, red, green and blue: along x, along y and diagonally, each with its own
/// wavelength [?: a choice].
constexpr texture_wave texture_waves[3] = {{0.0, 0.10, 0.0}, {90.0, 0.07, 0.5}, {60.0, 0.05, 1.0}};

/// Each channel's mean and amplitude: the channel stays within 28 to 228 [S: computed].
constexpr double texture_mean = 128.0;
constexpr double texture_amplitude = 100.0;

/// Maps what the options' ending asks for to the status next() reports.
/// @param ending [in] the ending.
/// @return the status of the same name.
gca::next_status ending_status(synthetic_ending ending)
{
    switch (ending)
    {
    case synthetic_ending::end:
        return gca::next_status::end;
    case synthetic_ending::timeout:
        return gca::next_status::timeout;
    case synthetic_ending::failure:
        return gca::next_status::failure;
    }
    // Not reached: the switch names every ending, and -Wswitch says when one is added.
    return gca::next_status::failure;
}

/// Renders the scene as the camera sees it into a prepared frame, by the one pinhole
/// convention: the ray of pixel (u, v) is ((u - cx) / fx, (v - cy) / fy, 1) in the camera frame.
/// Its point at parameter s has the camera-frame depth s, so the ray's parameter where it meets
/// the plane is the pixel's depth.
/// @param scene [in] the plane and the camera's pose.
/// @param out [in,out] prepared for its description (frame::prepare()); every pixel of its
///            color and depth is written.
void render(const synthetic_scene &scene, gca::frame &out)
{
    const gca::frame_description &description = out.description();
    const gca::intrinsics &k = description.intrinsics;
    const Eigen::Matrix3d rotation = scene.camera_to_world.linear();
    const Eigen::Vector3d origin = scene.camera_to_world.translation();
    const std::span<std::uint8_t> color = out.color();
    const std::span<std::uint16_t> depth = out.depth();
    for (std::uint32_t v = 0; v < description.size.height; v++)
    {
        for (std::uint32_t u = 0; u < description.size.width; u++)
        {
            const Eigen::Vector3d ray_camera((u - double{k.cx}) / k.fx, (v - double{k.cy}) / k.fy,
                                             1.0);
            const Eigen::Vector3d ray_world = rotation * ray_camera;
            // Where origin + s * ray_world meets z = 0. A plane behind the camera gives s < 0,
            // a ray parallel to it an infinite or NaN s, and the comparisons below refuse each.
            const double s = -origin.z() / ray_world.z();
            const Eigen::Vector3d hit = origin + s * ray_world;
            const double value = std::round(s / description.depth_unit_m);
            const bool seen = value >= 1.0 && value <= std::numeric_limits<std::uint16_t>::max() &&
                              std::abs(hit.x()) <= scene.plane.half_size_x_m &&
                              std::abs(hit.y()) <= scene.plane.half_size_y_m;
            const std::size_t pixel = static_cast<std::size_t>(v) * description.size.width + u;
            depth[pixel] = seen ? static_cast<std::uint16_t>(value) : 0;
            const std::array<double, 3> rgb =
                seen ? plane_color(hit.x(), hit.y()) : std::array<double, 3>{};
            for (std::size_t c = 0; c < 3; c++)
                color[3 * pixel + c] = static_cast<std::uint8_t>(std::lround(rgb[c]));
        }
    }
}
} // namespace

std::array<double, 3> plane_color(double x_m, double y_m)
{
    std::array<double, 3> rgb{};
    for (std::size_t c = 0; c < 3; c++)
    {
        const texture_wave &wave = texture_waves[c];
        const double a = wave.direction_deg * M_PI / 180.0;
        const double along_m = x_m * std::cos(a) + y_m * std::sin(a);
        rgb[c] =
            texture_mean +
            texture_amplitude * std::sin(2.0 * M_PI * along_m / wave.wavelength_m + wave.phase_rad);
    }
    return rgb;
}

synthetic_source::synthetic_source(synthetic_source_options options)
    : m_options(std::move(options))
{
}

gca::open_result synthetic_source::open(const gca::frame_size &size) noexcept
{
    if (m_open)
        return {false, "the synthetic source is open already"};
    if (!m_options.open_refusal.empty())
        return {false, m_options.open_refusal};
    const gca::frame_size &own = m_options.description.size;
    if (size != own)
        return {false, "the synthetic source renders " + std::to_string(own.width) + "x" +
                           std::to_string(own.height) + " px, not " + std::to_string(size.width) +
                           "x" + std::to_string(size.height) + " px"};
    m_open = true;
    m_next_index = 0;
    return {true, {}};
}

gca::next_result synthetic_source::next(gca::frame &out) noexcept
{
    if (!m_open)
        return {gca::next_status::failure, "the synthetic source is not open"};
    if (m_next_index >= m_options.frame_count)
        return {ending_status(m_options.ending), m_options.ending_message};
    out.prepare(m_options.description, m_next_index, m_next_index * frame_period_us);
    render(m_options.scene, out);
    m_next_index++;
    return {gca::next_status::frame, {}};
}

void synthetic_source::close() noexcept
{
    m_open = false;
}

gca::source_description synthetic_source::describe() const noexcept
{
    return {"synthetic source: one textured plane", {}, {}};
}

const synthetic_scene &synthetic_source::scene() const noexcept
{
    return m_options.scene;
}
} // namespace gca_test
