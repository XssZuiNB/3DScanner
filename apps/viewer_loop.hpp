// SPDX-License-Identifier: 0BSD
/// @file viewer_loop.hpp
/// orbbec_viewer's loop on a frame source (camera/frame_source.hpp): it takes the first frame,
/// prints the intrinsics line and sets up the GPU for frames of its size, then grabs frame after
/// frame, makes each one's organized points on the GPU (geometry/organized_points.hpp), reports the
/// frame rate and the timings, shows color | depth | overlay and saves snapshots, the points with
/// their normals (geometry/organized_normals.hpp) as a model's PLY (scan/model_file.hpp), until
/// quit, the exit flag, the frame limit, the source's end or a failure of the GPU. With --sequence
/// it writes every frame it takes, with its IMU samples, into a sequence (docs/SEQUENCE_FILE.md),
/// each marked as tracked by nothing, and counts the frames the writer's queue had no room for. The
/// viewer runs it on the Orbbec camera, the offline tests on their synthetic source; it links no
/// OrbbecSDK.
///
/// The frame is RGB8 (camera/frame.hpp), as the GPU frame uploads it; OpenCV's window and its PNG
/// files take BGR8, so the loop converts each frame's color once, into one picture both read.

#pragma once

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/posix_file.hpp"
#include "camera/sequence_writer.hpp"
#include "types/frame_description.hpp"
#include "viewer_options.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <ostream>

namespace gca_viewer
{
/// How often the viewer prints its report line [?: a choice, the prototype's once a second].
constexpr std::chrono::seconds report_period{1};

/// How long the viewer waits for an opening's first frame before it gives up [S: the
/// prototype's start waited up to 100 times 200 ms for its first frameset].
constexpr std::chrono::seconds first_frame_time_limit{20};

/// The name of the file the viewer writes its sequence to, in the directory --sequence names
/// [?: a choice; the ending is the one ctest `sequence` gives its files].
inline constexpr const char *sequence_file_name = "frames.sequence";

/// Where the loop writes the frames it takes when --sequence names a directory. The seam exists
/// so that the offline tests run the loop with a scripted sink, which answers chosen pushes with
/// a full queue or a failure: a real writer is not full at a test's frame size and pace, since a
/// record of 160x120 takes about 0.1 ms to write [S: computed, 96 017 bytes at docs/HARDWARE.md's
/// 5.9 ms for 5.12 MB] and the loop about 5 ms per frame [M 2026-11-14, ctest viewer_loop]. The
/// program has one implementation, sequence_file over gca::sequence_writer; the tests script
/// their own. The loop calls open() once, after the first frame; push() once for each frame it
/// takes, in the order taken, after the frame's last use, until a push fails; close() once at its
/// end when open() succeeded, then frames_written(). One thread calls them.
class sequence_sink
{
public:
    /// Builds the seam's part of a sink, which holds nothing.
    sequence_sink() = default;

    /// A sink is not copied: it stands for one file.
    sequence_sink(const sequence_sink &) = delete;
    sequence_sink &operator=(const sequence_sink &) = delete;

    /// Ends the sink; an implementation's destructor closes what is open.
    virtual ~sequence_sink() = default;

    /// Creates the sequence for the frames of one opening.
    /// @param description [in] the size, intrinsics and depth unit of every frame it takes.
    /// @return opened; or not, with the reason, naming the file (gca::sequence_writer::open()).
    virtual gca::open_result open(const gca::frame_description &description) noexcept = 0;

    /// Puts a frame into the sequence, marked as tracked by nothing.
    /// @param in_out [in,out] the frame, of the opening's description, its index and timestamp
    ///        greater than those of the frame accepted before. Accepted: swapped for a frame of
    ///        the same description, its pixels unspecified. Full or failure: as it was.
    /// @return accepted; full, the queue having no room, with no reason; or failure with the
    ///         reason, naming the file (gca::sequence_writer::push()).
    virtual gca::push_result push(gca::frame &in_out) noexcept = 0;

    /// Writes every frame accepted and the end, and syncs the file.
    /// @return written; or the reason, naming the file (gca::sequence_writer::close()).
    virtual gca::write_result close() noexcept = 0;

    /// Tells how many frames stand whole in the file (gca::sequence_writer::frames_written()).
    /// @return the count; after a failed write, the frames before the one that failed.
    virtual std::uint64_t frames_written() noexcept = 0;

    /// Gives the sequence's file, for the line that closes it.
    /// @return its path, valid as long as the sink.
    virtual const std::filesystem::path &path() const noexcept = 0;
};

/// The sequence the viewer writes: the file sequence_file_name in the directory --sequence names,
/// written by gca::sequence_writer with a queue of gca::sequence_queue_frames, every frame marked
/// gca::not_tracked.
class sequence_file final : public sequence_sink
{
public:
    /// Names the file; nothing is created until open().
    /// @param directory [in] the directory, which read_options() found empty and writable.
    explicit sequence_file(const std::filesystem::path &directory);

    /// Creates the file, never over one that stands there, and starts the writer.
    /// @param description [in] the frames' description.
    /// @return as gca::sequence_writer::open() gives it.
    /// @note Allocates the writer's queue, gca::sequence_queue_frames frames of the description's
    ///       size, and starts its thread.
    gca::open_result open(const gca::frame_description &description) noexcept override;

    /// Pushes the frame with the mark gca::not_tracked.
    /// @param in_out [in,out] the frame; as gca::sequence_writer::push() leaves it.
    /// @return as gca::sequence_writer::push() gives it.
    /// @note Allocates nothing when the frame is accepted or the queue is full. Takes the
    ///       writer's lock; waits for nothing else.
    gca::push_result push(gca::frame &in_out) noexcept override;

    /// Closes the writer.
    /// @return as gca::sequence_writer::close() gives it.
    /// @note Waits for the writer's thread to write its queue, and for the sync to disk.
    gca::write_result close() noexcept override;

    /// Tells how many frames the writer wrote whole.
    /// @return as gca::sequence_writer::frames_written() gives it.
    /// @note Allocates nothing. Takes the writer's lock; waits for nothing else.
    std::uint64_t frames_written() noexcept override;

    /// Gives the file's path.
    /// @return the directory and sequence_file_name.
    const std::filesystem::path &path() const noexcept override;

private:
    std::filesystem::path m_path;  ///< the sequence's file
    gca::sequence_writer m_writer; ///< its writer, open between open() and close()
};

/// Where the viewer prints, how often it reports, what stops it and where the frames it takes go:
/// the viewer's console, its stop signals and the sequence of --sequence, or a test's stream, flag
/// and sink.
struct run_context
{
    /// The least time between two report lines: report_period in the viewer, zero for a report
    /// after every frame.
    std::chrono::steady_clock::duration report_period;
    /// Read before each next() of the loop: true ends the loop. The viewer's handler of SIGINT,
    /// SIGTERM and SIGHUP sets it, on whichever thread the signal reaches; it outlives the run.
    const std::atomic<bool> &exit_requested;
    /// Where the lines are printed, with console_color.hpp's color codes; it outlives the run.
    std::ostream &out;
    /// Where the frames taken are written, not opened yet; nullptr for no sequence. It outlives
    /// the run.
    sequence_sink *sequence = nullptr;
};

/// Runs the viewer on an open frame source. Steps and lines:
/// - the first frame: timeouts are waited through for first_frame_time_limit, then "Timeout
///   while waiting for first frames."; a failure or the end prints the source's message; either
///   way "Couldn't start stream!" follows and nothing else runs. The first frame prints
///   "Intrinsics fx=<fx> fy=<fy> cx=<cx> cy=<cy>  depth scale=<m> m/unit", sets up the GPU for
///   frames of its size (a failure prints CUDA's or the pool's reason, then "Couldn't set up the
///   GPU!", and nothing else runs) and opens the sequence when there is one (a refusal prints its
///   reason, and nothing else runs); it is not shown or counted, but it is the sequence's first
///   frame.
/// - the loop, until 'q' or ESC in the window, the exit flag, opt.max_frames delivered frames, the
///   source's end (its message printed), a push to the sequence that fails or a GPU step that
///   fails (its reason printed): each delivered frame is converted, uploaded, made organized points
///   on the GPU within the depth range 0.15 m to 3.0 m, its validity mask read back and its valid
///   points counted, shown unless headless, and counted. A timeout prints "No frames received from
///   Orbbec camera for 5s!" (AGENTS.md §3 reads it as a stop condition), a failure the source's
///   message; neither counts a frame, and the loop goes on. In the window the keys keep working
///   through them: the window stays on the last delivered frame, which 's' saves.
/// - the sequence: each frame taken, the first included, is pushed once its successor has come,
///   or after the loop for the last one, so that the frame stays whole while it is shown and
///   saved. A push that finds the queue full is counted, the frame not written; it was shown like
///   any other.
/// - the report, once the report period has passed since the last: "fps <frames per second> |
///   grab+decode <ms> ms | gpu point cloud <ms> ms | points <count>", the frames per second
///   counting delivered frames only, the times the means over the frames since the last report,
///   the GPU's from the upload to the count, the points the last frame's valid ones
///   (docs/HARDWARE.md's numbers come from it). With a sequence the line goes on with " | queue
///   full <count>", the pushes since the last report that found the queue full.
/// - a snapshot, on 's' in the window and on the last of opt.max_frames frames when headless: the
///   frame's normals estimated on the GPU, then cloud_NNN.ply, the frame's valid points that have a
///   normal as a model's PLY (docs/SCAN_FILES.md, "Models"), in the frame's camera frame, with
///   their pixels' colors, and "Saved <count> points to cloud_NNN.ply", or the writer's reason when
///   it fails, the loop going on; then cloud_NNN.ply.color.png (8-bit color) and
///   cloud_NNN.ply.depth.png (16-bit depth values), all in the current directory, NNN counting
///   from 000. A GPU step of the snapshot that fails prints its reason and ends the loop, with
///   nothing written.
/// - the sequence's close, after the loop: "Wrote <accepted> frames to <file>, <full> not written
///   because the writer's queue was full", the totals of the run; or, when a push or the close
///   failed, once: "<reason, naming the file>; <written> of <taken> frames taken written, <lost>
///   lost from the writer's queue, <full> not written because the writer's queue was full", the
///   frames taken counting the first and the one that came with a failed push, those lost the
///   frames the writer accepted and did not write, and the rest of the frames taken never pushed.
/// @param source [in,out] the frame source, open; left open.
/// @param opt [in] the options; the loop reads max_frames and headless, the opening took the
///        rest, and the sequence comes in the context.
/// @param context [in] the report period, the exit flag, the stream the lines go to and the
///        sequence.
/// @return true when the loop ran, no GPU step failed and the sequence, if any, was written whole;
///         false when no first frame came, the GPU could not be set up, a GPU step failed, or the
///         sequence could not be created or written.
/// @note Allocates the two frames and the BGR picture once, and on the GPU, for the first frame's
///       size, a CUDA stream of its own and from a pool (gpu/gpu_pool.hpp) the GPU frame, the
///       organized points and normals and a pinned buffer of the validity mask, 36 bytes a pixel
///       [S: computed, gpu_frame.hpp, organized_points.hpp, organized_normals.hpp], 37 MB at
///       1280x800; a frame allocates nothing on the GPU. Each frame's work runs on that stream, and
///       the loop waits for it once per frame, after the mask's copy, so that the report counts the
///       frame's points: a camera test's choice, not the scanner's per-frame path, which reads back
///       only what its next step needs (CODING_STANDARDS.md, "The rewrite's code"). A snapshot
///       allocates its points' host copies and waits for the stream once more. With a sequence,
///       its open allocates the writer's queue, gca::sequence_queue_frames frames, 16, 82 MB at
///       1280x800 [S: sequence_writer.hpp], and starts the writer's thread, which writes the
///       frames beside the loop until the close waits for it. Runs on one thread besides the
///       writer's; only the exit flag may be set from another.
bool run_viewer(gca::frame_source &source, const options &opt, const run_context &context);
} // namespace gca_viewer
