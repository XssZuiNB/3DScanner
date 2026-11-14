// SPDX-License-Identifier: 0BSD
/// @file sequence_format.cpp
/// The sequence's header and a frame's lead, encoded and decoded byte by byte, least significant
/// byte first, whatever the host's order (see sequence_format.hpp).

#include "camera/sequence_format.hpp"

#include <algorithm>
#include <string_view>
#include <type_traits>

namespace gca
{
namespace
{
/// The header's first bytes, which say that the file is a sequence.
constexpr std::string_view magic = "3dscanner sequence";

/// The format's version, which the header holds after the magic.
constexpr std::uint32_t version = 1;

/// The bytes of the magic and the version, which decode_sequence_header() checks first.
constexpr std::size_t magic_and_version_bytes = 22;

/// The kind bytes of a frame's record: with a pose, in tracking loss, not tracked.
constexpr std::uint8_t pose_kind = 'P';
constexpr std::uint8_t loss_kind = 'L';
constexpr std::uint8_t untracked_kind = 'N';

/// The bytes of a frame's lead after its kind: the index and the timestamp, and the pose's seven
/// numbers.
constexpr std::size_t index_and_timestamp_bytes = 16;
constexpr std::size_t pose_bytes = 56;

static_assert(magic.size() + 4 == magic_and_version_bytes);
static_assert(magic_and_version_bytes + 2 * 4 + 5 * 4 == sequence_header_bytes);
static_assert(1 + index_and_timestamp_bytes + pose_bytes == frame_lead_max_bytes);

/// Writes an unsigned number's bytes, the least significant first.
/// @tparam T std::uint32_t or std::uint64_t.
/// @param at [out] where its sizeof(T) bytes go; owned by the caller.
/// @param value [in] the number.
template <typename T> void put_little_endian(std::uint8_t *at, T value)
{
    static_assert(std::is_unsigned_v<T>);
    for (std::size_t k = 0; k < sizeof(T); k++)
        at[k] = static_cast<std::uint8_t>(value >> (8 * k));
}

/// Reads an unsigned number whose least significant byte comes first.
/// @tparam T std::uint32_t or std::uint64_t.
/// @param at [in] its sizeof(T) bytes; owned by the caller.
/// @return the number.
template <typename T> T get_little_endian(const std::uint8_t *at)
{
    static_assert(std::is_unsigned_v<T>);
    T value = 0;
    for (std::size_t k = 0; k < sizeof(T); k++)
        value |= static_cast<T>(at[k]) << (8 * k);
    return value;
}

/// Writes a float's IEEE 754 binary32 bits, the least significant byte first.
/// @param at [out] where its 4 bytes go; owned by the caller.
/// @param value [in] the float; every bit is kept, NaN's payload and -0 among them.
void put_float(std::uint8_t *at, float value)
{
    put_little_endian(at, std::bit_cast<std::uint32_t>(value));
}

/// Reads a float written by put_float().
/// @param at [in] its 4 bytes; owned by the caller.
/// @return the float, bit for bit.
float get_float(const std::uint8_t *at)
{
    return std::bit_cast<float>(get_little_endian<std::uint32_t>(at));
}

/// Writes a double's IEEE 754 binary64 bits, the least significant byte first.
/// @param at [out] where its 8 bytes go; owned by the caller.
/// @param value [in] the double; every bit is kept.
void put_double(std::uint8_t *at, double value)
{
    put_little_endian(at, std::bit_cast<std::uint64_t>(value));
}

/// Reads a double written by put_double().
/// @param at [in] its 8 bytes; owned by the caller.
/// @return the double, bit for bit.
double get_double(const std::uint8_t *at)
{
    return std::bit_cast<double>(get_little_endian<std::uint64_t>(at));
}
} // namespace

std::array<std::uint8_t, sequence_header_bytes> encode_sequence_header(
    const frame_description &description) noexcept
{
    std::array<std::uint8_t, sequence_header_bytes> bytes{};
    std::copy(magic.begin(), magic.end(), bytes.begin());
    put_little_endian(&bytes[18], version);
    put_little_endian(&bytes[22], description.size.width);
    put_little_endian(&bytes[26], description.size.height);
    put_float(&bytes[30], description.intrinsics.fx);
    put_float(&bytes[34], description.intrinsics.fy);
    put_float(&bytes[38], description.intrinsics.cx);
    put_float(&bytes[42], description.intrinsics.cy);
    put_float(&bytes[46], description.depth_unit_m);
    return bytes;
}

sequence_header_result decode_sequence_header(std::span<const std::uint8_t> bytes) noexcept
{
    const std::size_t compared = std::min(bytes.size(), magic.size());
    if (!std::equal(bytes.begin(), bytes.begin() + compared, magic.begin()))
        return {std::nullopt,
                "is no sequence: it does not start with \"" + std::string(magic) + "\""};
    if (bytes.size() >= magic_and_version_bytes)
    {
        const std::uint32_t found = get_little_endian<std::uint32_t>(&bytes[18]);
        if (found != version)
            return {std::nullopt, "is of the sequence format's version " + std::to_string(found) +
                                      "; this reader knows version " + std::to_string(version) +
                                      " only"};
    }
    if (bytes.size() < sequence_header_bytes)
        return {std::nullopt, "ends early: within its header, after " +
                                  std::to_string(bytes.size()) + " of its " +
                                  std::to_string(sequence_header_bytes) + " bytes"};
    frame_description description{};
    description.size.width = get_little_endian<std::uint32_t>(&bytes[22]);
    description.size.height = get_little_endian<std::uint32_t>(&bytes[26]);
    description.intrinsics.fx = get_float(&bytes[30]);
    description.intrinsics.fy = get_float(&bytes[34]);
    description.intrinsics.cx = get_float(&bytes[38]);
    description.intrinsics.cy = get_float(&bytes[42]);
    description.depth_unit_m = get_float(&bytes[46]);
    return {description, {}};
}

frame_lead_bytes encode_frame_lead(const frame_lead &lead) noexcept
{
    frame_lead_bytes bytes{};
    const camera_pose *pose = std::get_if<camera_pose>(&lead.tracking);
    bytes.bytes[0] =
        pose ? pose_kind
             : (std::holds_alternative<tracking_loss>(lead.tracking) ? loss_kind : untracked_kind);
    put_little_endian(&bytes.bytes[1], lead.index);
    put_little_endian(&bytes.bytes[9], lead.timestamp_us);
    bytes.size = 1 + index_and_timestamp_bytes;
    if (!pose)
        return bytes;
    const Eigen::Vector3d &t = pose->position_m;
    const Eigen::Quaterniond &q = pose->orientation;
    const double values[7] = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
    for (std::size_t k = 0; k < 7; k++)
        put_double(&bytes.bytes[bytes.size + 8 * k], values[k]);
    bytes.size += pose_bytes;
    return bytes;
}

std::optional<std::size_t> frame_lead_bytes_after(std::uint8_t kind) noexcept
{
    if (kind == pose_kind)
        return index_and_timestamp_bytes + pose_bytes;
    if (kind == loss_kind || kind == untracked_kind)
        return index_and_timestamp_bytes;
    return std::nullopt;
}

std::optional<frame_lead> decode_frame_lead(
    std::uint8_t kind, std::span<const std::uint8_t, frame_lead_max_bytes - 1> rest) noexcept
{
    frame_lead lead{get_little_endian<std::uint64_t>(&rest[0]),
                    get_little_endian<std::uint64_t>(&rest[8]), not_tracked{}};
    if (kind == untracked_kind)
        return lead;
    if (kind == loss_kind)
    {
        lead.tracking = tracking_loss{};
        return lead;
    }
    if (kind != pose_kind)
        return std::nullopt;
    double values[7];
    for (std::size_t k = 0; k < 7; k++)
        values[k] = get_double(&rest[index_and_timestamp_bytes + 8 * k]);
    // Eigen's quaternion constructor takes w first; the file holds x y z w.
    lead.tracking = camera_pose{Eigen::Vector3d(values[0], values[1], values[2]),
                                Eigen::Quaterniond(values[6], values[3], values[4], values[5])};
    return lead;
}

std::string order_refusal(std::uint64_t index, std::uint64_t timestamp_us,
                          std::uint64_t index_before, std::uint64_t timestamp_before_us) noexcept
{
    if (index <= index_before)
        return "frame index " + std::to_string(index) +
               " does not increase; the frame before has index " + std::to_string(index_before);
    if (timestamp_us <= timestamp_before_us)
        return "timestamp " + std::to_string(timestamp_us) +
               " µs does not increase; the frame before was taken at " +
               std::to_string(timestamp_before_us) + " µs";
    return {};
}

bool same_description(const frame_description &a, const frame_description &b) noexcept
{
    const auto bits = [](float value) { return std::bit_cast<std::uint32_t>(value); };
    return a.size == b.size && bits(a.intrinsics.fx) == bits(b.intrinsics.fx) &&
           bits(a.intrinsics.fy) == bits(b.intrinsics.fy) &&
           bits(a.intrinsics.cx) == bits(b.intrinsics.cx) &&
           bits(a.intrinsics.cy) == bits(b.intrinsics.cy) &&
           bits(a.depth_unit_m) == bits(b.depth_unit_m);
}
} // namespace gca
