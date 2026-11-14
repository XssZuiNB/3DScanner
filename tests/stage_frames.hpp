// SPDX-License-Identifier: 0BSD
/// @file stage_frames.hpp
/// The frames the per-frame path's stages are tested on: the camera's two stream sizes with its
/// intrinsics and depth unit, two sizes that test a stage's kernel at its limits, the tabletop
/// scene's sphere, and the rendering of a scene's frames by the tests' synthetic source. Header
/// only; each test program is one translation unit.

#pragma once

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "check.hpp"
#include "synthetic_source.hpp"
#include "types/frame_description.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace gca_test
{
/// The camera's depth unit: 1 mm, its default [S: AGENTS.md §5].
inline constexpr float camera_depth_unit_m = 0.001f;

/// The smaller stream size, 640x400, with the camera's intrinsics there: fx and fy differ, and the
/// principal point lies off the picture's centre at no whole or half pixel [M 2026-11-13:
/// docs/HARDWARE.md, the camera's sizes at 30 fps].
inline constexpr gca::frame_description small_description = {
    {640, 400}, {306.18f, 306.14f, 317.41f, 199.13f}, camera_depth_unit_m};

/// The larger stream size, 1280x800, with the camera's intrinsics there [M 2026-11-13:
/// docs/HARDWARE.md, the camera's sizes at 30 fps].
inline constexpr gca::frame_description large_description = {
    {1280, 800}, {612.37f, 612.29f, 635.81f, 398.27f}, camera_depth_unit_m};

/// A size that is no multiple of a kernel's block of 32 x 8: one column past 20 blocks and one row
/// past 50, so that the last column and row of blocks each hold one pixel of the picture and most
/// of their threads lie outside it [?: a choice]; with the smaller size's intrinsics.
inline constexpr gca::frame_description odd_description = {
    {641, 401}, small_description.intrinsics, camera_depth_unit_m};

/// A picture of one column and 600000 rows: its rows need 75000 blocks of 8 rows, more than the
/// 65535 a grid holds along y [S: CUDA C++ Programming Guide, "Compute Capabilities"], so that
/// CUDA refuses a stage kernel's launch.
inline constexpr gca::frame_size tall_size = {1, 600000};

/// The tabletop scene's sphere: 8 cm in radius, resting on tabletop_plane off its centre
/// [?: a choice, a tabletop object's size].
inline const synthetic_sphere tabletop_sphere = {Eigen::Vector3d(0.03, -0.02, 0.08), 0.08};

/// A scene's frames and the known surfaces they are rendered from.
struct rendered
{
    std::vector<gca::frame> frames;          ///< one for each pose, in the trajectory's order
    std::vector<synthetic_surface> surfaces; ///< the plane, then the box's faces (surfaces())
};

/// Renders the frames of a scene from the synthetic source.
/// @param scene [in] the scene, one frame for each pose of its trajectory.
/// @param description [in] the frames' size, intrinsics and depth unit.
/// @return the frames and the known surfaces; fewer frames, with a failed check, when the source
///         refuses or ends early.
inline rendered render(const synthetic_scene &scene, const gca::frame_description &description)
{
    synthetic_source_options options;
    options.description = description;
    options.scene = scene;
    synthetic_source source(std::move(options));
    rendered result{{}, source.surfaces()};
    check(source.open(description.size).opened, "the synthetic source opened");
    for (std::size_t i = 0; i < scene.trajectory.size(); i++)
    {
        gca::frame frame;
        if (source.next(frame).status != gca::next_status::frame)
        {
            check(false, "the synthetic source gave every frame of its trajectory");
            break;
        }
        result.frames.push_back(std::move(frame));
    }
    return result;
}
} // namespace gca_test
