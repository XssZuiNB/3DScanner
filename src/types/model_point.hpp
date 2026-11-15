// SPDX-License-Identifier: 0BSD
/// @file model_point.hpp
/// One point of a model, the full model or the object model (GLOSSARY.md): its position, its unit
/// normal and its color in the models' frame. The model's file (scan/model_file.hpp,
/// docs/SCAN_FILES.md) writes and reads it, and separation (separation/proposal.hpp) proposes the
/// support plane and the box from it. A plain struct of Eigen's value types, header only.

#pragma once

#include <array>
#include <cstdint>

#include <Eigen/Core>

namespace gca
{
/// One point of a model, in the models' frame: the camera frame of the first fused frame (x
/// right, y down, z forward, in m).
struct model_point
{
    Eigen::Vector3f position_m;        ///< the point, in m
    Eigen::Vector3f normal;            ///< the surface's unit normal at the point, its length
                                       ///< within 1e-6 of 1
    std::array<std::uint8_t, 3> color; ///< red, green, blue, 0 to 255
};
} // namespace gca
