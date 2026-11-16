// SPDX-License-Identifier: 0BSD
/// @file live_model.hpp
/// The live model as the scan interface publishes it (R10; the spec, "The scan interface";
/// docs/methods/tracking.md, "The live model"): the colored points of the live model's sparse TSDF
/// with the pose of the frame they were taken at, and the taking of them on the scan thread. A
/// live model holds at most a limit of points, thinned on the GPU by each edge's place in the grid
/// (sparse_tsdf::extract()'s spacing), so that a surface keeps the same points from one live model
/// to the next. The spacing rises within a scan and never falls. Only the first live model of a
/// scan and the first after the spacing rose extract every block; the others extract the blocks
/// touched since the last one (the slots' stamps) and replace those blocks' points in the last
/// live model on the host.

#pragma once

#include "fusion/sparse_tsdf.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/pinned_buffer.hpp"
#include "types/camera_pose.hpp"
#include "types/model_point.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace gca
{
/// The live model: a value, copied out as a snapshot.
struct live_model
{
    /// Rises by one with each live model published, the empty one a scan starts with too.
    std::uint64_t version = 0;
    /// The index of the frame it was taken at, after that frame's fusion; none for the empty live
    /// model of a scan before its first.
    std::optional<std::uint64_t> frame;
    /// The pose tracking gave that frame, its camera frame into the models' frame
    /// (types/camera_pose.hpp); the identity while there is no frame.
    camera_pose pose{Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity()};
    /// When the scan thread decided to take it, by the clock its pacing runs on.
    std::chrono::steady_clock::time_point taken{};
    /// The grid's thinning its points were extracted at, in voxels, a power of two.
    std::uint32_t spacing = every_crossing;
    /// Its points, in the models' frame, at most the scan interface's limit, in no stated order.
    std::vector<model_point> points;
};

/// A live model as the scan thread holds it: the value it publishes, and each point's block's slot
/// in the hash, which the next take's merge reads. The two travel together, so that one swap moves
/// both.
struct held_live_model
{
    live_model value;                 ///< the live model; its points as readers get them
    std::vector<std::uint32_t> slots; ///< the k-th the slot of value.points' k-th point's block
};

/// The share of the limit above which the spacing rises for the next live model: 3/4
/// [?: a choice, room for the model to grow by a third until the next live model].
inline constexpr double live_model_fill = 0.75;

/// Gives the spacing for the next live model: the same while a live model's points stay within
/// live_model_fill of the limit, else the least power-of-two multiple of it at which they would,
/// a doubling taking a surface's points to a quarter (docs/methods/tracking.md, "The live model").
/// @param points [in] the points the live model held, or found, at the spacing.
/// @param spacing [in] the spacing they were taken at, a power of two.
/// @param limit [in] the most points a live model holds, > 0.
/// @return the spacing, never less than spacing.
/// @note Host only; allocates nothing.
std::uint32_t spacing_after(std::uint64_t points, std::uint32_t spacing,
                            std::uint32_t limit) noexcept;

/// What taking a live model gave.
enum class live_take_status
{
    taken,    ///< the points are in place, at most the limit
    too_many, ///< more points than the limit: nothing is taken, the spacing rose and the next take
              ///< extracts every block
    failed,   ///< CUDA failed or refused; the message says why
};

/// What taking a live model gave, with the extraction's counts.
struct live_take
{
    live_take_status status = live_take_status::failed; ///< taken, too many or failed
    bool every_block = false;  ///< whether the extraction took every block, not those touched
    std::uint32_t found = 0;   ///< the points the extraction found
    std::uint32_t blocks = 0;  ///< the blocks it went through
    std::uint32_t spacing = 0; ///< the spacing it extracted at
    std::string message;       ///< why it failed, in plain English; empty otherwise
};

/// The taking of a scan's live models: the extraction's buffer of the limit's points and of a
/// list of every block, their pinned copies, the marks of the touched blocks on the host, and the
/// spacing and the stamp the next take starts from. One for each scan, as the live model's sparse
/// TSDF. Move-only. Not synchronized: the scan thread alone uses it.
class live_model_taker
{
public:
    /// An empty taker; take() with it fails.
    live_model_taker() noexcept = default;

    /// Requests the taker's memory from the pool, for a sparse TSDF of a layout: the extraction's
    /// buffer of the limit's points, their slots, and a list of every block the TSDF holds; their
    /// pinned copies; and on the host one mark for each slot of the hash.
    /// @param pool [in,out] the pool; open, else the request is refused as closed.
    /// @param limit [in] the most points a live model holds, > 0.
    /// @param layout [in] the live model's sparse TSDF's layout (sparse_tsdf::layout()).
    /// @return granted, with a taker whose first take extracts every block at spacing 1; or the
    ///         first request that was refused, with its status and message.
    /// @note Allocates device memory (32 bytes a point and 4 bytes a block, and 8 bytes), pinned
    ///       host memory (as much and 8 bytes) and host memory (1 bit a slot): 8.3 MB, 8.3 MB and
    ///       256 KB for 200000 points and the 2.0 GB TSDF's 479620 blocks and 2^21 slots
    ///       [S: computed], outside the TSDF's memory budget as the tracker's buffers are; so it is
    ///       called before the pool is closed. Allocates for the message when refused; a failed
    ///       allocation ends the program (std::terminate). Two threads may not request from one
    ///       pool at once.
    static request_result<live_model_taker> request(gpu_pool &pool, std::uint32_t limit,
                                                    const sparse_tsdf_layout &layout) noexcept;

    /// A taker is not copied: a copy allocates.
    live_model_taker(const live_model_taker &) = delete;
    live_model_taker &operator=(const live_model_taker &) = delete;

    /// Takes over another taker's memory and state without allocating.
    /// @param other [in,out] the taker moved from; left empty.
    live_model_taker(live_model_taker &&other) noexcept = default;

    /// Takes over another taker's memory and state; this one's are freed first.
    /// @param other [in,out] the taker moved from; left empty.
    /// @return this taker.
    /// @pre No work queued on the GPU uses this one's memory any more.
    live_model_taker &operator=(live_model_taker &&other) noexcept = default;

    /// Frees the memory.
    /// @pre No work queued on the GPU uses it any more.
    ~live_model_taker() = default;

    /// Takes the live model's points at the taker's spacing: every block's when the take is the
    /// first or the spacing rose since the last, else the blocks the allocations touched since the
    /// last take, whose points replace those blocks' points of the last live model; the blocks
    /// not touched keep theirs. Then raises the spacing for the next take when the points exceed
    /// live_model_fill of the limit (spacing_after()), which then extracts every block.
    /// @param model [in] the live model's sparse TSDF, the one the taker was requested for, its
    ///        integrations queued on stream.
    /// @param last [in] the last live model's points with their blocks' slots, as an earlier take
    /// of
    ///        this taker gave them; ignored when every block is taken.
    /// @param next [out] the live model's points and their slots, resized within the capacity
    ///        reserved for the limit; left as they were unless taken. Not last. Its other fields
    ///        are the caller's to set.
    /// @param stream [in] the CUDA stream the live model's work is queued on and waited for.
    /// @return taken; too many, with nothing taken and the spacing raised, when the extraction or
    ///         the points kept and new together exceed the limit; failed with the reason when the
    ///         taker is empty, or CUDA refused or failed (sparse_tsdf::extract(), the copy or the
    ///         wait), with nothing taken.
    /// @note The extraction (sparse_tsdf::extract(), with its one wait), one copy of the points,
    ///       their slots and the list of blocks into the pinned buffers, and one written wait,
    ///       cudaStreamSynchronize(stream); then the conversion and the merge on the host.
    ///       Allocates nothing when taken or too many; allocates for the message when failed, and
    ///       a failed allocation ends the program (std::terminate): the function is noexcept.
    live_take take(const sparse_tsdf &model, const held_live_model &last, held_live_model &next,
                   cudaStream_t stream) noexcept;

private:
    /// One mark a slot of the hash, on the host: the blocks a take's extraction went through.
    class slot_marks
    {
    public:
        /// Gives a mark to each slot of a hash, none set.
        /// @param slot_count [in] the hash's slots.
        /// @note Allocates (1 bit a slot); a failed allocation ends the program (std::terminate).
        void resize(std::uint32_t slot_count) noexcept
        {
            m_words.assign((std::size_t{slot_count} + 63) / 64, 0);
        }

        /// Marks slots.
        /// @param slots [in] the slots, each below the hash's slot count.
        /// @param count [in] how many.
        void mark(const std::uint32_t *slots, std::uint32_t count) noexcept
        {
            for (std::uint32_t k = 0; k < count; k++)
                m_words[slots[k] / 64] |= std::uint64_t{1} << (slots[k] % 64);
        }

        /// Says whether a slot is marked.
        /// @param slot [in] the slot, below the hash's slot count.
        /// @return whether it is.
        bool marked(std::uint32_t slot) const noexcept
        {
            return (m_words[slot / 64] >> (slot % 64) & 1u) != 0;
        }

        /// Takes the marks of slots back, and with them every mark sharing their words, so that
        /// after the slots last marked are cleared no mark is set.
        /// @param slots [in] the slots, each below the hash's slot count.
        /// @param count [in] how many.
        void clear(const std::uint32_t *slots, std::uint32_t count) noexcept
        {
            for (std::uint32_t k = 0; k < count; k++)
                m_words[slots[k] / 64] = 0;
        }

    private:
        std::vector<std::uint64_t>
            m_words; ///< 64 marks a word, slot s at bit s % 64 of word s / 64
    };

    std::uint32_t m_limit = 0;                ///< the most points a live model holds
    surface_points m_extracted;               ///< the extraction's buffer of the limit and its list
    pinned_buffer m_points_on_host;           ///< the points as the last take copied them
    pinned_buffer m_slots_on_host;            ///< their slots
    pinned_buffer m_blocks_on_host;           ///< the blocks the last extraction went through
    slot_marks m_touched;                     ///< the blocks of a merge's extraction
    std::uint32_t m_spacing = every_crossing; ///< the spacing of the next take
    std::uint32_t m_since = 0;                ///< the first stamp the next take's blocks need
    bool m_every_block = true;                ///< whether the next take extracts every block
};
} // namespace gca
