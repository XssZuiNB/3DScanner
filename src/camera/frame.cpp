// SPDX-License-Identifier: 0BSD
/// @file frame.cpp
/// A frame: its buffers sized for its description, and its accessors (see frame.hpp).

#include "camera/frame.hpp"

#include <cstddef>

namespace gca
{
void frame::prepare(const frame_description &description, std::uint64_t index,
                    std::uint64_t timestamp_us)
{
    const std::size_t pixels =
        static_cast<std::size_t>(description.size.width) * description.size.height;
    m_color.resize(3 * pixels);
    m_depth.resize(pixels);
    m_description = description;
    m_index = index;
    m_timestamp_us = timestamp_us;
}

const frame_description &frame::description() const noexcept
{
    return m_description;
}

std::uint64_t frame::index() const noexcept
{
    return m_index;
}

std::uint64_t frame::timestamp_us() const noexcept
{
    return m_timestamp_us;
}

std::span<std::uint8_t> frame::color() noexcept
{
    return m_color;
}

std::span<const std::uint8_t> frame::color() const noexcept
{
    return m_color;
}

std::span<std::uint16_t> frame::depth() noexcept
{
    return m_depth;
}

std::span<const std::uint16_t> frame::depth() const noexcept
{
    return m_depth;
}
} // namespace gca
