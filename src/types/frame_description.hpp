// SPDX-License-Identifier: 0BSD
/// @file frame_description.hpp
/// What every frame of one opening of a frame source shares: the size of its pictures, the
/// camera's intrinsics and the depth unit. A frame source is opened for a stream size; each frame
/// it delivers carries the whole description (camera/frame.hpp). Plain structs, header only.

#pragma once

#include "types/camera_param.hpp"

#include <cstdint>

namespace gca
{
/// The stream size: the size of the pictures a frame source is opened for, color and aligned
/// depth alike.
struct frame_size
{
    std::uint32_t width;  ///< pixels in a row, in px
    std::uint32_t height; ///< rows, in px

    /// Compares two sizes.
    /// @param other [in] the other size.
    /// @return whether width and height are equal.
    bool operator==(const frame_size &other) const = default;
};

/// What the pixels of every frame of one opening mean.
struct frame_description
{
    frame_size size;            ///< the size of the color and the depth picture
    gca::intrinsics intrinsics; ///< the camera's intrinsics at this size
    float depth_unit_m;         ///< meters per depth value, > 0: depth in m = value * depth_unit_m
};
} // namespace gca
