// SPDX-License-Identifier: 0BSD
/// @file frame_copy.hpp
/// A copy of a frame's index, timestamp and pixels, for the tests that check a frame source left
/// the caller's frame as it was: a frame itself cannot be copied (camera/frame.hpp). Header only.

#pragma once

#include "camera/frame.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace gca_test
{
/// A frame's index, timestamp and pixels, copied to compare the frame with later.
struct frame_copy
{
    std::uint64_t index;              ///< the frame's index
    std::uint64_t timestamp_us;       ///< its timestamp, in µs
    std::vector<std::uint8_t> color;  ///< its color picture
    std::vector<std::uint16_t> depth; ///< its depth picture
};

/// Copies a frame's index, timestamp and pixels.
/// @param frame [in] the frame.
/// @return the copy.
inline frame_copy copy_of(const gca::frame &frame)
{
    return {frame.index(), frame.timestamp_us(),
            std::vector<std::uint8_t>(frame.color().begin(), frame.color().end()),
            std::vector<std::uint16_t>(frame.depth().begin(), frame.depth().end())};
}

/// Returns whether a frame holds what a copy holds.
/// @param frame [in] the frame.
/// @param copy [in] the copy.
/// @return whether the index, the timestamp and every pixel of color and depth are equal.
inline bool holds(const gca::frame &frame, const frame_copy &copy)
{
    return frame.index() == copy.index && frame.timestamp_us() == copy.timestamp_us &&
           std::ranges::equal(frame.color(), copy.color) &&
           std::ranges::equal(frame.depth(), copy.depth);
}
} // namespace gca_test
