// SPDX-License-Identifier: 0BSD
/// @file live_view.hpp
/// The live view (GLOSSARY.md; docs/methods/tracking.md, "The live view"): the picture of the live
/// model's surface as the camera sees it from a tracked frame's pose, after that frame's fusion, in
/// the model's colors lit lightly by the surface's normal, as a JPEG file; the newest one, which a
/// reader copies under a lock of its own; and the encoder that makes them on a thread of its own,
/// one at a time, so that a frame whose encoder is busy has no view and the scan thread never waits
/// for one. The surface is the tracker's, cast after the fusion (tracking/tracker.hpp, surface()).
#pragma once

#include "fusion/sparse_tsdf.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace gca
{
/// The shading's share of a pixel's color that does not depend on the light, a: a pixel with a
/// normal shows its color times a + (1 - a) cos(theta), theta the angle between the normal and the
/// ray back to the camera [?: initial, the user's choice of 2026-11-17, "模型颜色加轻度光照": the
/// model's color lit lightly by the normal; docs/methods/tracking.md, "The live view"].
inline constexpr float live_view_ambient = 0.6f;

/// The grey a pixel whose ray met no surface shows, each of red, green and blue [?: initial: a dark
/// grey, so that a black object stands out from it].
inline constexpr std::uint8_t live_view_background = 48;

/// The JPEG's quality, OpenCV's IMWRITE_JPEG_QUALITY, 0 to 100 [?: a choice; its bytes and time in
/// docs/HARDWARE.md, "The live view"].
inline constexpr int live_view_jpeg_quality = 85;

/// The live view: a value, copied out as a snapshot.
struct live_view
{
    /// Rises by one with each view published, the empty one a scan starts with too.
    std::uint64_t version = 0;
    /// The index of the tracked frame it shows, from the pose tracking gave it, after its fusion;
    /// none for the empty view of a scan before its first.
    std::optional<std::uint64_t> frame;
    /// That frame's timestamp, in µs on its source's clock; 0 without a frame.
    std::uint64_t timestamp_us = 0;
    /// The picture's width and height, in px, the frames'; 0 x 0 without a frame.
    frame_size size{0, 0};
    /// The picture as a JPEG file's bytes (JFIF, baseline, live_view_jpeg_quality), rows top to
    /// bottom as the frame's; empty without a frame.
    std::vector<std::uint8_t> jpeg;
};

/// The newest live view, under a lock of its own, and its listener: the scan interface's for its
/// lifetime. A scan's start clears it (clear(), on the scan thread); during a scan its encoder
/// publishes each view (live_view_encoder, on the encoder's thread), so that one thread at a time
/// publishes. Not copied and not moved: the encoder keeps a reference to it.
class newest_live_view
{
public:
    /// Builds the newest live view, empty, version 0.
    /// @param size [in] the frames' size, in px: the newest view's bytes are reserved for 3 bytes a
    ///        pixel, more than a JPEG of the picture takes [?: a JPEG of quality below 100 is
    ///        smaller than the picture's raw bytes].
    /// @param on_view [in] called after each view is published, with it, outside the lock, on the
    ///        thread that published it; never two calls at once. Empty: no call. Copied.
    /// @note Allocates the reserved bytes; a failed allocation throws std::bad_alloc.
    newest_live_view(const frame_size &size, std::function<void(const live_view &)> on_view);

    /// The newest live view is not copied or moved: an encoder refers to it.
    newest_live_view(const newest_live_view &) = delete;
    newest_live_view &operator=(const newest_live_view &) = delete;

    /// Gives the newest live view.
    /// @return a copy of the view last published, a snapshot: empty, without a frame, before the
    ///         first view of the scan in progress, or before the first scan.
    /// @note Takes the lock for the copy, which allocates for its bytes; any thread may call it,
    ///       the listener too.
    live_view latest() const;

    /// Publishes an empty view, without a frame, its version the last one's + 1, and calls the
    /// listener: a scan's start.
    /// @pre No encoder of it is running a view: none exists, as between two scans. When one is, two
    ///      threads publish at once: the listener may be called on both at once, and a call may see
    ///      its view change under it (a data race).
    /// @note Allocates nothing; takes the lock for the clearing.
    void clear() noexcept;

private:
    friend class live_view_encoder;

    /// Publishes a frame's view, its version the last one's + 1, under the lock; its bytes are
    /// swapped in, so that nothing is allocated.
    /// @param frame [in] the frame's index.
    /// @param timestamp_us [in] the frame's timestamp, in µs on its source's clock.
    /// @param size [in] the picture's size, in px.
    /// @param jpeg [in,out] the JPEG's bytes; given the last view's bytes in exchange, which the
    ///        encoder fills next.
    /// @note On the encoder's thread; allocates nothing.
    void publish(std::uint64_t frame, std::uint64_t timestamp_us, const frame_size &size,
                 std::vector<std::uint8_t> &jpeg) noexcept;

    /// Calls the listener with the newest view, outside the lock.
    /// @note On the thread that published it.
    void notify() noexcept;

    std::function<void(const live_view &)> m_on_view; ///< the listener
    mutable std::mutex m_mutex;                       ///< guards m_published
    live_view m_published;                            ///< the newest view
};

/// What offering a frame's view to its encoder gave.
enum class live_view_offer_status
{
    queued, ///< the encoder was free: the shading and the copy are queued, the frame handed over
    busy,   ///< the encoder was busy with a view: this frame has none, nothing was queued
    failed, ///< CUDA refused the work, or the encoder's last view failed; the message says why
};

/// What offering a frame's view gave, with the reason when it failed.
struct live_view_offer
{
    live_view_offer_status status = live_view_offer_status::failed; ///< queued, busy or failed
    std::string message; ///< why it failed, in plain English; empty otherwise
};

/// The encoder of a scan's live views (docs/methods/tracking.md, "The live view"): the picture on
/// the GPU, its pinned copy, the event after the copy, and the thread that waits for the event,
/// encodes the picture as a JPEG, publishes it into the newest live view and calls its listener.
/// One view at a time: the encoder is busy from the offer that hands a frame over until that
/// frame's view is published and its listener has returned, or the view has failed, and a frame
/// offered while it is busy has no view; so a view is never older than its frame's offer by more
/// than one view's making. Its thread is started by the request and ended by finish() or the
/// encoder's end, after the view it was handed. Move-only; the scan thread alone offers.
class live_view_encoder
{
public:
    /// An empty encoder: no picture, no thread; an offer to it fails.
    live_view_encoder() noexcept;

    /// Requests the encoder's memory from the pool, creates its event and starts its thread.
    /// @param pool [in,out] the pool; open, else the request is refused as closed.
    /// @param size [in] the pictures' size, the frames', in px, each > 0.
    /// @param views [in,out] the newest live view the views are published into; owned by the
    ///        caller, which keeps it alive as long as the encoder.
    /// @return granted, with an encoder that is free; or the first of the requests that was
    ///         refused, with its status and message (gpu_pool::request()), or failure when CUDA
    ///         could not create the event, with CUDA's error, then cleared.
    /// @note Allocates device memory and page-locked host memory, 3 bytes a pixel each, 0.77 MB
    ///       at 640x400 [S: computed], an event (cudaEventCreateWithFlags, blocking and without
    ///       timing) and host memory for the JPEG's bytes, 3 bytes a pixel, so it is called before
    ///       the pool is closed; whether CUDA's allocators wait for the GPU is not stated by CUDA
    ///       [?]. Starts a thread. A failed allocation, and a thread that cannot be started, end
    ///       the program (std::terminate): the function is noexcept. Two threads may not request
    ///       from one pool at once.
    static request_result<live_view_encoder> request(gpu_pool &pool, const frame_size &size,
                                                     newest_live_view &views) noexcept;

    /// An encoder is not copied: it stands for one thread.
    live_view_encoder(const live_view_encoder &) = delete;
    live_view_encoder &operator=(const live_view_encoder &) = delete;

    /// Takes over another encoder's memory, event and thread without allocating.
    /// @param other [in,out] the encoder moved from; left empty.
    live_view_encoder(live_view_encoder &&other) noexcept;

    /// Takes over another encoder; this one is ended first, as its end ends it.
    /// @param other [in,out] the encoder moved from; left empty. Moving onto itself changes
    ///        nothing.
    /// @return this encoder.
    /// @pre No work queued on the GPU uses this one's picture or its pinned copy any more but the
    ///      copy its thread waits for; when some does, it may use freed memory (gpu_image's and
    ///      pinned_buffer's move assignments).
    live_view_encoder &operator=(live_view_encoder &&other) noexcept;

    /// Ends the encoder: its thread finishes the view it was handed, publishes it, and ends; then
    /// its memory and event are freed.
    /// @pre No work queued on the GPU uses its picture or its pinned copy any more but the copy its
    ///      thread waits for; when some does, it may use freed memory (gpu_image's and
    ///      pinned_buffer's ends).
    /// @note Waits for the thread (join), which waits for the GPU's copy of the view handed last
    ///       and its encoding; frees with cudaFree, cudaFreeHost and cudaEventDestroy.
    ~live_view_encoder();

    /// Offers a tracked frame's view (docs/methods/tracking.md, "The live view"): when the encoder
    /// is free, queues on the stream the shading of the surface's level 0 into the picture
    /// (live_view_ambient, live_view_background), in OpenCV's order, blue, green, red, its copy
    /// into the pinned picture and the event after it, then hands the frame over to the thread and
    /// returns; when it is busy, does nothing.
    /// @param surface [in] the live model's surface cast from the frame's pose after its fusion, of
    ///        the encoder's size at level 0 (tracking/tracker.hpp, surface()); owned by the caller,
    ///        which holds it unchanged until the work queued here has run: the next cast is
    ///        queued on the same stream.
    /// @param frame [in] the frame's index.
    /// @param timestamp_us [in] the frame's timestamp, in µs on its source's clock.
    /// @param stream [in] the CUDA stream the surface's cast was queued on; owned by the caller.
    /// @return queued; busy, with nothing queued; failed with the reason, nothing handed over:
    ///         the encoder is empty or ended (finish()), the surface is empty or not of its size,
    ///         the encoder's last view failed (the wait for its copy, or its encoding; the reason
    ///         it left is taken, so the offer after this one tries again), or CUDA refused the
    ///         shading's launch, the copy or the event, CUDA's error then cleared and the picture
    ///         unspecified.
    /// @note Queues a kernel of one thread a pixel in the stages' block of 32 x 8, a
    ///       cudaMemcpyAsync into page-locked memory and a cudaEventRecord; waits for nothing;
    ///       takes the encoder's lock twice, which its thread holds only to take a frame or to
    ///       become free. Allocates nothing when queued or busy; allocates for the message when
    ///       failed, and a failed allocation ends the program (std::terminate): the function is
    ///       noexcept. Not while another thread offers.
    live_view_offer offer(const model_surface &surface, std::uint64_t frame,
                          std::uint64_t timestamp_us, cudaStream_t stream) noexcept;

    /// Ends the encoder at a scan's end, so that the newest view shows the last tracked frame
    /// (docs/methods/tracking.md, "The live view"): waits until the encoder is free; when that
    /// frame was not the last one handed over (its offer found the encoder busy, or failed),
    /// queues its view as offer() does and hands it over together with the end, in one step of the
    /// lock; then waits for the thread, which makes that view before it ends.
    /// @param surface [in] the live model's surface as the frame's pose sees it, unchanged since
    ///        the frame's fusion (a lost frame casts nothing), as for offer(); owned by the caller.
    /// @param frame [in] the last tracked frame's index.
    /// @param timestamp_us [in] its timestamp, in µs on its source's clock.
    /// @param stream [in] the CUDA stream the surface's cast was queued on; owned by the caller.
    /// @return done when the frame's view is the newest one published; not done with the reasons,
    ///         joined by "; ": the encoder is empty or ended, the surface is empty or not of its
    ///         size, CUDA refused the work, or a view the encoder made failed (the wait for its
    ///         copy, or its encoding): the one in flight at the call, or the frame's.
    /// @note Waits for the encoder's view in flight, its listener's return and, when it hands the
    ///       frame over, for that view's copy, encoding and listener (the thread's join); queues
    ///       as offer() does. Allocates for a message only. After it the encoder offers nothing.
    ///       Not while another thread offers.
    step_result finish(const model_surface &surface, std::uint64_t frame,
                       std::uint64_t timestamp_us, cudaStream_t stream) noexcept;

private:
    struct state; ///< the picture, the event, the thread and what it shares with the offers

    /// Ends the thread and frees the memory and the event: the end's and the move assignment's.
    void release() noexcept;

    std::unique_ptr<state> m_state; ///< on the heap, so that the thread's reference survives a move
};
} // namespace gca
