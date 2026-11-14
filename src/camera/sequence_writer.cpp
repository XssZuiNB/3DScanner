// SPDX-License-Identifier: 0BSD
/// @file sequence_writer.cpp
/// The sequence's writer: a ring of frames filled by swaps under one lock, and a thread that
/// writes each frame's record with POSIX calls (see sequence_writer.hpp).

#include "camera/sequence_writer.hpp"

#include <bit>
#include <cerrno>
#include <string_view>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace gca
{
namespace
{
/// The bytes of a picture, as they lie in memory.
/// @tparam T the picture's element, std::uint8_t or std::uint16_t.
/// @param picture [in] the picture; owned by its frame, which outlives the view.
/// @return a view of its bytes.
template <typename T> std::string_view bytes_of(std::span<const T> picture)
{
    return {reinterpret_cast<const char *>(picture.data()), picture.size_bytes()};
}

/// Whether two descriptions are the same bit for bit, so that a NaN equals itself and -0 differs
/// from 0.
/// @param a [in] one description.
/// @param b [in] the other.
/// @return whether size, intrinsics and depth unit have the same bits.
bool same_description(const frame_description &a, const frame_description &b)
{
    const auto bits = [](float value) { return std::bit_cast<std::uint32_t>(value); };
    return a.size == b.size && bits(a.intrinsics.fx) == bits(b.intrinsics.fx) &&
           bits(a.intrinsics.fy) == bits(b.intrinsics.fy) &&
           bits(a.intrinsics.cx) == bits(b.intrinsics.cx) &&
           bits(a.intrinsics.cy) == bits(b.intrinsics.cy) &&
           bits(a.depth_unit_m) == bits(b.depth_unit_m);
}
} // namespace

sequence_writer::~sequence_writer()
{
    if (m_open)
        close();
}

open_result sequence_writer::open(const std::filesystem::path &path,
                                  const frame_description &description,
                                  std::size_t queue_frames) noexcept
{
    if (m_open)
        return {false, path.string() + ": the writer is open already, writing " + m_path.string()};
    if (queue_frames == 0)
        return {false, path.string() + ": a queue of 0 frames takes no frame"};
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0)
    {
        const std::string message = system_message();
        return {false, path.string() + ": cannot create: " + message};
    }
    const auto refuse = [&](const std::string &what) {
        const std::string message = system_message();
        const std::string reason = path.string() + ": " + what + ": " + message;
        ::close(fd);
        ::unlink(path.c_str());
        return open_result{false, reason};
    };
    const std::array<std::uint8_t, sequence_header_bytes> header =
        encode_sequence_header(description);
    if (!write_all(fd, {reinterpret_cast<const char *>(header.data()), header.size()}))
        return refuse("the header's write failed");
    if (!sync_directory(path.parent_path()))
        return refuse("the sync of its directory failed");

    m_slots.resize(queue_frames);
    for (slot &place : m_slots)
        place.queued_frame.prepare(description, 0, 0);
    m_path = path;
    m_description = description;
    m_fd = fd;
    m_head = 0;
    m_count = 0;
    m_closing = false;
    m_failure.clear();
    m_accepted_any = false;
    try
    {
        m_thread = std::thread(&sequence_writer::write_queue, this);
    }
    catch (const std::system_error &error)
    {
        m_slots.clear();
        m_fd = -1;
        errno = error.code().value();
        return refuse("the writer's thread could not be started");
    }
    m_open = true;
    return {true, {}};
}

push_result sequence_writer::push(frame &in_out, const frame_tracking &tracking) noexcept
{
    if (!m_open)
        return {push_status::failure, "the sequence writer is not open"};
    // A reason is built only for a refusal: an accepted push allocates nothing.
    if (!same_description(in_out.description(), m_description))
        return {push_status::failure,
                m_path.string() + ": frame index " + std::to_string(in_out.index()) +
                    ": its size, intrinsics or depth unit are not the sequence's"};
    if (m_accepted_any)
    {
        const std::string out_of_order =
            order_refusal(in_out.index(), in_out.timestamp_us(), m_last_index, m_last_timestamp_us);
        if (!out_of_order.empty())
            return {push_status::failure, m_path.string() + ": " + out_of_order};
    }
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_failure.empty())
            return {push_status::failure, m_failure};
        if (m_count == m_slots.size())
            return {push_status::full, {}};
        slot &place = m_slots[(m_head + m_count) % m_slots.size()];
        m_last_index = in_out.index();
        m_last_timestamp_us = in_out.timestamp_us();
        m_accepted_any = true;
        std::swap(place.queued_frame, in_out);
        place.tracking = tracking;
        m_count++;
    }
    m_wake.notify_one();
    return {push_status::accepted, {}};
}

bool sequence_writer::has_room() noexcept
{
    if (!m_open)
        return false;
    const std::lock_guard<std::mutex> lock(m_mutex);
    return m_count < m_slots.size();
}

write_result sequence_writer::close() noexcept
{
    if (!m_open)
        return {false, "the sequence writer is not open"};
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        m_closing = true;
    }
    m_wake.notify_one();
    m_thread.join();
    m_open = false;

    // The writer's thread has ended: what it set is this thread's to read.
    std::string failure = m_failure;
    const auto failed = [this](const char *what) {
        const std::string message = system_message();
        return m_path.string() + ": " + what + ": " + message;
    };
    const char end = static_cast<char>(sequence_end_kind);
    if (failure.empty() && !write_all(m_fd, {&end, 1}))
        failure = failed("the end's write failed");
    else if (failure.empty() && ::fsync(m_fd) != 0)
        failure = failed("the sync to disk failed");
    if (::close(m_fd) != 0 && failure.empty())
        failure = failed("the close failed");
    m_fd = -1;
    m_slots.clear();
    if (!failure.empty())
        return {false, failure};
    return {true, {}};
}

void sequence_writer::write_queue() noexcept
{
    std::unique_lock<std::mutex> lock(m_mutex);
    for (;;)
    {
        m_wake.wait(lock, [this] { return m_count > 0 || m_closing; });
        if (m_count == 0)
            return;
        const slot &queued = m_slots[m_head];
        lock.unlock();
        const bool written = write_record(queued);
        // The system's message is taken before anything else can change errno.
        const std::string message = written ? std::string() : system_message();
        lock.lock();
        if (!written)
        {
            // The file holds the frames before this one; the frames queued after it cannot follow.
            m_failure = m_path.string() + ": the write of frame index " +
                        std::to_string(queued.queued_frame.index()) + " failed: " + message;
            m_count = 0;
            return;
        }
        m_head = (m_head + 1) % m_slots.size();
        m_count--;
    }
}

bool sequence_writer::write_record(const slot &queued) const noexcept
{
    const frame &held = queued.queued_frame;
    const frame_lead_bytes lead =
        encode_frame_lead({held.index(), held.timestamp_us(), queued.tracking});
    return write_all(m_fd, {reinterpret_cast<const char *>(lead.bytes.data()), lead.size}) &&
           write_all(m_fd, bytes_of(held.depth())) && write_all(m_fd, bytes_of(held.color()));
}
} // namespace gca
