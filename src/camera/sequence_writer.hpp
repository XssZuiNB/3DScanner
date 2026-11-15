// SPDX-License-Identifier: 0BSD
/// @file sequence_writer.hpp
/// The sequence's writer (GLOSSARY.md, "Sequence"; docs/SEQUENCE_FILE.md): it takes frames, each
/// with its IMU samples, with what tracking said about each from a bounded queue and writes them,
/// uncompressed, on a thread of its own, so that the per-frame path does not wait for the disk. A
/// frame enters the queue by a swap of buffers, samples included, so a push neither copies nor
/// allocates. Whether the queue has room is
/// told before a frame's stages run (has_room()); when the queue is full the push says so too,
/// and the writer loses no frame it accepted: close() writes every one of them, in the order
/// accepted, before the end. A failure is a value with its reason (CODING_STANDARDS.md,
/// "The rewrite's code"); no exception leaves the writer.

#pragma once

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/posix_file.hpp"
#include "camera/sequence_format.hpp"
#include "types/frame_description.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gca
{
/// The queue's size for a scan, in frames [?: a choice]: 16 frames of 1280x800 hold 82 MB
/// [S: computed, 5 120 000 bytes each] and cover a writer that falls 0.53 s behind at 30 fps
/// [S: computed, 16 / 30 s]. On the board the writer takes a frame of 1280x800 in 7.8 ms, the
/// longest wait for room in a full queue of 16 was 46 ms, and at 30 fps for 30 s it took every
/// frame [M 2026-11-14, docs/HARDWARE.md, "Memory, storage, network"]. Not measured: a walk of 5
/// minutes, about 46 GB, past the drive's own cache, and the per-frame path running beside the
/// writer.
inline constexpr std::size_t sequence_queue_frames = 16;

/// Whether push() took the frame.
enum class push_status
{
    accepted, ///< the frame is in the queue, and close() writes it
    full,     ///< every slot of the queue holds a frame not yet written; the frame was not taken
    failure,  ///< the frame was not taken: the writer failed or is not open, or the frame does not
              ///< fit the sequence; the reason says which
};

/// What push() gave.
struct push_result
{
    push_status status = push_status::failure; ///< taken, or why not
    std::string reason; ///< why not, in plain English, naming the file; empty when the frame was
                        ///< accepted or the queue was full
};

/// Writes one sequence at a time. One thread at a time calls open(), push() and close(); the
/// writer's own thread writes the frames between open() and close().
class sequence_writer
{
public:
    /// Builds a writer that is not open and holds nothing.
    sequence_writer() = default;

    /// A writer is not copied: it stands for one file and its thread.
    sequence_writer(const sequence_writer &) = delete;
    sequence_writer &operator=(const sequence_writer &) = delete;

    /// Closes the writer when it is open (close()), so that its thread ends; what close() gives
    /// is lost then, so a caller that needs it calls close() first.
    ~sequence_writer();

    /// Creates a sequence and starts the writer's thread. The file is created under its name,
    /// never over a file that stands there, and its header is written; its directory is synced,
    /// so that the name survives a power cut. The queue's frames are allocated here, before the
    /// first frame.
    /// @param path [in] the sequence's path; its directory exists, and no file stands there.
    /// @param description [in] the size, intrinsics and depth unit of every frame it takes.
    /// @param queue_frames [in] the queue's size, in frames, >= 1; a scan takes
    ///        sequence_queue_frames.
    /// @return opened; or not, with the reason: the writer is open already, the queue's size is
    ///         0, the file cannot be created (it exists, its directory is missing or cannot be
    ///         written) with the system's message, the header's write or the directory's sync
    ///         failed (the file is removed then), the thread could not be started (the file is
    ///         removed then).
    /// @note Allocates queue_frames frames of the description's size. A failed allocation ends
    ///       the program (std::terminate): the function is noexcept.
    open_result open(const std::filesystem::path &path, const frame_description &description,
                     std::size_t queue_frames) noexcept;

    /// Tells whether the queue has room for a frame. Room seen stays until the caller's next
    /// push(), because only the caller fills the queue and the writer's thread only empties it.
    /// So the per-frame path keeps this order, and no frame is fused that is not written:
    /// has_room(); the frame's upload (gpu/gpu_frame.hpp, which stages it on the host); the
    /// stages, of which tracking gives the frame its pose or a mark; push() with that pose or
    /// mark. A frame that finds no room is not taken.
    /// @return true when a slot is free, also after a write failure, which the next push()
    ///         reports; false when every slot holds a frame not yet written, or the writer is not
    ///         open.
    /// @note Allocates nothing. Takes the queue's lock; waits for nothing else.
    bool has_room() noexcept;

    /// Puts a frame into the queue by swapping buffers with a free slot: the caller's frame goes
    /// into the queue as it is, and the caller gets the slot's frame in exchange, of the same
    /// description, its index, timestamp and pixels unspecified. A push after has_room() said
    /// true is accepted or fails; it never finds the queue full.
    /// @param in_out [in,out] the frame, of the sequence's description, its index and its
    ///        timestamp greater than those of the frame accepted before, at most
    ///        imu_samples_per_record_max samples of each sensor, each after the sensor's sample
    ///        before and none after the frame's timestamp; owned by the caller. Swapped when
    ///        accepted, as it was otherwise.
    /// @param tracking [in] the frame's pose, bit for bit, or its mark; copied.
    /// @return accepted; full, with no reason; or failure with the reason: the writer is not
    ///         open, it failed to write (the same reason at every later call), the frame's
    ///         description is not the sequence's, its index or timestamp does not increase, it
    ///         holds more samples of a sensor than a record holds, a sample does not follow its
    ///         sensor's last or lies after the frame's timestamp (imu_sample_order).
    /// @note Allocates nothing when the frame is accepted or the queue is full; allocates for a
    ///       failure's reason. A failed allocation ends the program (std::terminate): the
    ///       function is noexcept. Takes the queue's lock for the swap; waits for nothing else.
    push_result push(frame &in_out, const frame_tracking &tracking) noexcept;

    /// Writes every frame still in the queue, in order, then the end of the sequence, syncs the
    /// file to disk and ends the writer's thread. The writer is not open afterwards, and its
    /// queue's frames are released.
    /// @return written; or the reason, naming the file: the writer is not open; a write failed
    ///         (the frames written before it stand in the file, which then has no end, and the
    ///         frames queued after it are lost); the end's write, the sync or the close failed.
    /// @note Waits for the writer's thread to write the queue. Allocates for a reason. A failed
    ///       allocation ends the program (std::terminate): the function is noexcept.
    write_result close() noexcept;

    /// Tells how many frames stand whole in the file: the records the writer's thread has written
    /// since the last open(), in the order accepted. After close() they are the frames the file
    /// holds; after a failed write, the frames before the one that failed, so that the frames
    /// accepted and not counted here were lost with the failure.
    /// @return the count; 0 before the first open().
    /// @note Allocates nothing. Takes the queue's lock; waits for nothing else.
    std::uint64_t frames_written() noexcept;

private:
    /// A place in the queue: a frame and what tracking said about it.
    struct slot
    {
        frame queued_frame;      ///< the frame, owned by the queue
        frame_tracking tracking; ///< its pose or mark
    };

    /// The writer's thread: writes the frames in the order queued, until close() has been asked
    /// and the queue is empty, or a write fails.
    void write_queue() noexcept;

    /// Writes one frame's record: its lead, its samples, its depth, its color.
    /// @param queued [in] the frame and its tracking; the writer's thread owns it during the call.
    /// @return whether every byte was written; on false errno holds the reason.
    bool write_record(const slot &queued) const noexcept;

    // Set by open() and close(), read by the writer's thread only between them.
    std::filesystem::path m_path;      ///< the sequence's path
    frame_description m_description{}; ///< the description of every frame
    int m_fd = -1;                     ///< the open file's descriptor; -1 when not open
    std::vector<slot> m_slots;         ///< the queue's places, a ring
    std::thread m_thread;              ///< the writer's thread while open

    // The caller's thread's own.
    bool m_open = false;                   ///< whether open() succeeded and close() has not run
    bool m_accepted_any = false;           ///< whether a frame was accepted since open()
    std::uint64_t m_last_index = 0;        ///< the index of the frame accepted last
    std::uint64_t m_last_timestamp_us = 0; ///< its timestamp, in µs
    imu_sample_order m_sample_order;       ///< each sensor's last sample accepted

    // The lock guards the ring's state, the request to close, the failure and the count written,
    // and with them the hand-over of a slot: the caller fills only a slot outside
    // [head, head + count), the writer's thread writes only the slot at head while it is counted.
    std::mutex m_mutex;
    std::condition_variable m_wake; ///< wakes the writer's thread: a frame queued, or close()
    std::size_t m_head = 0;         ///< the slot written next
    std::size_t m_count = 0;        ///< the slots that hold a frame not yet written
    bool m_closing = false;         ///< whether close() has been asked
    std::string m_failure;          ///< why a write failed; empty while none has
    std::uint64_t m_written = 0;    ///< the records written whole since open()
};
} // namespace gca
