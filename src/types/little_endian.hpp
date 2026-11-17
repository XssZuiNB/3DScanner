// SPDX-License-Identifier: 0BSD
/// @file little_endian.hpp
/// The bytes of a number in a binary layout the tree defines, the least significant byte first,
/// whatever the host's order: an unsigned integer, and a float (IEEE 754 binary32) and a double
/// (binary64) by their bits, every bit kept. The sequence's file (camera/sequence_format.cpp,
/// docs/SEQUENCE_FILE.md) and the web layer's live model message (web/messages.cpp,
/// docs/ROUTES.md) are written with them. Header only.

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace gca
{
/// Writes an unsigned number's bytes, the least significant first.
/// @tparam T std::uint16_t, std::uint32_t or std::uint64_t.
/// @param at [out] where its sizeof(T) bytes go; owned by the caller.
/// @param value [in] the number.
template <typename T> void put_little_endian(std::uint8_t *at, T value) noexcept
{
    static_assert(std::is_unsigned_v<T>);
    for (std::size_t k = 0; k < sizeof(T); k++)
        at[k] = static_cast<std::uint8_t>(value >> (8 * k));
}

/// Reads an unsigned number whose least significant byte comes first.
/// @tparam T std::uint16_t, std::uint32_t or std::uint64_t.
/// @param at [in] its sizeof(T) bytes; owned by the caller.
/// @return the number.
template <typename T> T get_little_endian(const std::uint8_t *at) noexcept
{
    static_assert(std::is_unsigned_v<T>);
    T value = 0;
    for (std::size_t k = 0; k < sizeof(T); k++)
        value = static_cast<T>(value | static_cast<T>(at[k]) << (8 * k));
    return value;
}

/// Writes a float's IEEE 754 binary32 bits, the least significant byte first.
/// @param at [out] where its 4 bytes go; owned by the caller.
/// @param value [in] the float; every bit is kept, NaN's payload and -0 among them.
inline void put_float(std::uint8_t *at, float value) noexcept
{
    put_little_endian(at, std::bit_cast<std::uint32_t>(value));
}

/// Reads a float written by put_float().
/// @param at [in] its 4 bytes; owned by the caller.
/// @return the float, bit for bit.
inline float get_float(const std::uint8_t *at) noexcept
{
    return std::bit_cast<float>(get_little_endian<std::uint32_t>(at));
}

/// Writes a double's IEEE 754 binary64 bits, the least significant byte first.
/// @param at [out] where its 8 bytes go; owned by the caller.
/// @param value [in] the double; every bit is kept.
inline void put_double(std::uint8_t *at, double value) noexcept
{
    put_little_endian(at, std::bit_cast<std::uint64_t>(value));
}

/// Reads a double written by put_double().
/// @param at [in] its 8 bytes; owned by the caller.
/// @return the double, bit for bit.
inline double get_double(const std::uint8_t *at) noexcept
{
    return std::bit_cast<double>(get_little_endian<std::uint64_t>(at));
}
} // namespace gca
