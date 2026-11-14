// SPDX-License-Identifier: 0BSD
/// @file sequence_reader.cpp
/// The sequence's reader: the header checked at open(), then one record after another read with
/// POSIX calls into a spare frame, which changes places with the caller's (see
/// sequence_reader.hpp).

#include "camera/sequence_reader.hpp"

#include "camera/posix_file.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <span>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace gca
{
namespace
{
/// Names a size.
/// @param size [in] the size, in px.
/// @return "<width>x<height> px".
std::string size_name(const frame_size &size)
{
    return std::to_string(size.width) + "x" + std::to_string(size.height) + " px";
}
} // namespace

sequence_reader::sequence_reader(std::filesystem::path path)
    : m_path(std::move(path))
{
}

sequence_reader::~sequence_reader()
{
    close();
}

open_result sequence_reader::open(const frame_size &size) noexcept
{
    if (m_fd >= 0)
        return {false, m_path.string() + ": the reader is open already"};
    const int fd = ::open(m_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        const std::string message = system_message();
        return {false, m_path.string() + ": cannot open: " + message};
    }
    std::array<std::uint8_t, sequence_header_bytes> bytes{};
    const std::optional<std::size_t> read = read_up_to(fd, bytes.data(), bytes.size());
    const std::string message = read ? std::string() : system_message();
    const auto refuse = [&](const std::string &reason) {
        ::close(fd);
        return open_result{false, m_path.string() + reason};
    };
    if (!read)
        return refuse(": cannot read: " + message);
    const sequence_header_result header =
        decode_sequence_header(std::span<const std::uint8_t>(bytes.data(), *read));
    if (!header.description)
        return refuse(" " + header.reason);
    if (header.description->size != size)
        return refuse(": its frames are " + size_name(header.description->size) + ", not the " +
                      size_name(size) + " asked");
    m_fd = fd;
    m_description = *header.description;
    // The spare frame's buffers exist before the first frame.
    m_spare.prepare(m_description, 0, 0);
    m_tracking = not_tracked{};
    m_records = 0;
    m_finished = false;
    m_ended_early = false;
    m_ending = {};
    return {true, {}};
}

next_result sequence_reader::next(frame &out) noexcept
{
    if (m_fd < 0)
        return {next_status::failure, m_path.string() + ": the sequence reader is not open"};
    if (m_finished)
        return m_ending;
    // A record's name is built only for a failure: a frame given allocates nothing.
    const auto record = [this] { return "record " + std::to_string(m_records + 1); };
    const auto unreadable = [this] {
        return finish(next_status::failure, "cannot read: " + system_message());
    };
    const auto early = [this](const std::string &where) {
        m_ended_early = true;
        return finish(next_status::failure, "ends early: " + where);
    };
    const auto unknown_kind = [&](std::uint8_t kind) {
        char hex[8];
        std::snprintf(hex, sizeof(hex), "0x%02x", kind);
        return finish(next_status::failure, record() + " has the unknown kind " + hex);
    };

    std::uint8_t kind = 0;
    std::optional<std::size_t> read = read_up_to(m_fd, &kind, 1);
    if (!read)
        return unreadable();
    if (*read == 0)
        return early(
            "after " +
            (m_records == 0 ? std::string("its header") : "record " + std::to_string(m_records)) +
            ", without the sequence's end");
    if (kind == sequence_end_kind)
    {
        std::uint8_t extra = 0;
        read = read_up_to(m_fd, &extra, 1);
        if (!read)
            return unreadable();
        if (*read != 0)
            return finish(next_status::failure, "bytes follow the sequence's end, " + record() +
                                                    "; the end is the file's last byte");
        return finish(next_status::end,
                      "the sequence's end, after " + std::to_string(m_records) + " frames");
    }
    const std::optional<std::size_t> lead_size = frame_lead_bytes_after(kind);
    if (!lead_size)
        return unknown_kind(kind);
    std::array<std::uint8_t, frame_lead_max_bytes - 1> rest{};
    read = read_up_to(m_fd, rest.data(), *lead_size);
    if (!read)
        return unreadable();
    if (*read < *lead_size)
        return early("within " + record() + ", before the sequence's end");
    const std::optional<frame_lead> lead = decode_frame_lead(kind, rest);
    if (!lead)
        return unknown_kind(kind);
    if (m_records > 0)
    {
        const std::string out_of_order =
            order_refusal(lead->index, lead->timestamp_us, m_last_index, m_last_timestamp_us);
        if (!out_of_order.empty())
            return finish(next_status::failure, record() + ": " + out_of_order);
    }

    m_spare.prepare(m_description, lead->index, lead->timestamp_us);
    const std::span<std::uint16_t> depth = m_spare.depth();
    const std::span<std::uint8_t> color = m_spare.color();
    for (const std::span<std::uint8_t> picture :
         {std::span<std::uint8_t>(reinterpret_cast<std::uint8_t *>(depth.data()),
                                  depth.size_bytes()),
          color})
    {
        read = read_up_to(m_fd, picture.data(), picture.size());
        if (!read)
            return unreadable();
        if (*read < picture.size())
            return early("within " + record() + ", before the sequence's end");
    }
    std::swap(out, m_spare);
    m_tracking = lead->tracking;
    m_records++;
    m_last_index = lead->index;
    m_last_timestamp_us = lead->timestamp_us;
    return {next_status::frame, {}};
}

void sequence_reader::close() noexcept
{
    if (m_fd < 0)
        return;
    ::close(m_fd);
    m_fd = -1;
}

source_description sequence_reader::describe() const noexcept
{
    return {"sequence " + m_path.string(), {}, {}};
}

const frame_tracking &sequence_reader::tracking() const noexcept
{
    return m_tracking;
}

bool sequence_reader::ended_early() const noexcept
{
    return m_ended_early;
}

next_result sequence_reader::finish(next_status status, const std::string &message)
{
    m_finished = true;
    m_ending = {status, m_path.string() + ": " + message};
    return m_ending;
}
} // namespace gca
