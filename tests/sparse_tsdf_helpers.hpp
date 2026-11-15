// SPDX-License-Identifier: 0BSD
/// @file sparse_tsdf_helpers.hpp
/// The sparse TSDF tests' shared helpers (fusion/sparse_tsdf.hpp): the method's starting values of
/// the truncation's rule, a pose as the sparse TSDF takes it, a frame's buffers on the GPU, a
/// lookup of block coordinates on the GPU with its upload and its download of the numbers found,
/// the extraction of the surface's points with their download, and the reading, printing and
/// comparison of the usage. A failure is a failed check, counted by check.hpp. Header only; each
/// test program is one translation unit.

#pragma once

#include "check.hpp"
#include "fusion/sparse_tsdf.hpp"
#include "geometry/organized_points.hpp"
#include "gpu/gpu_array.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "gpu_test_helpers.hpp"
#include "types/camera_pose.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace gca_test
{
/// The truncation's rule at the method's starting values: c = 0.01 /m [?: the datasheet's bound,
/// until the camera's depth noise is measured], k = 3 [?: initial], mu_min = 4 voxels
/// [?: initial] (docs/methods/sparse-tsdf.md, "The truncation distance").
inline constexpr gca::truncation_parameters method_truncation = {0.01f, 3.0f, 4.0f};

/// Gives a pose of the camera as the sparse TSDF takes it.
/// @param pose [in] camera frame to the models' frame.
/// @return the same pose as position and orientation.
inline gca::camera_pose as_camera_pose(const Eigen::Isometry3d &pose)
{
    return {pose.translation(), Eigen::Quaterniond(pose.linear())};
}

/// A frame's buffers on the GPU, from the pool: the GPU frame and its organized points.
struct frame_buffers
{
    gca::gpu_frame frame;         ///< the GPU frame the frame is uploaded into
    gca::organized_points points; ///< the organized points it is deprojected into
};

/// Requests a frame's buffers from the pool.
/// @param pool [in,out] the pool, open.
/// @param size [in] the frame's size.
/// @param what [in] the case, named in the check lines.
/// @return the buffers; empty ones and a failed check when the pool refused.
inline frame_buffers request_buffers(gca::gpu_pool &pool, const gca::frame_size &size,
                                     const std::string &what)
{
    return {granted(what + ": the GPU frame", gca::gpu_frame::request(pool, size)),
            granted(what + ": the organized points", gca::organized_points::request(pool, size))};
}

/// Block coordinates on the GPU with room for their block numbers, and the same coordinates on the
/// host.
struct lookup
{
    std::vector<gca::block_coordinate> host;           ///< the coordinates
    gca::gpu_array<gca::block_coordinate> coordinates; ///< the coordinates on the GPU
    gca::gpu_array<std::uint32_t> numbers;             ///< room for their block numbers
};

/// Requests the arrays of a lookup from the pool and uploads the coordinates on the stream the
/// kernels run on, waiting for the copy, so that no kernel reads coordinates not yet copied.
/// @param pool [in,out] the pool, open.
/// @param host [in] the coordinates, not empty; taken over.
/// @param stream [in] the stream of the kernels.
/// @param what [in] the case, named in the check lines.
/// @return the lookup; empty arrays and a failed check when the pool refused.
inline lookup make_lookup(gca::gpu_pool &pool, std::vector<gca::block_coordinate> host,
                          cudaStream_t stream, const std::string &what)
{
    lookup made{std::move(host), {}, {}};
    made.coordinates = granted(what + ": the coordinates' array",
                               pool.request_array<gca::block_coordinate>(made.host.size()));
    made.numbers =
        granted(what + ": the numbers' array", pool.request_array<std::uint32_t>(made.host.size()));
    if (!made.coordinates.empty())
        check(cudaMemcpyAsync(made.coordinates.data(), made.host.data(),
                              made.coordinates.size_bytes(), cudaMemcpyHostToDevice,
                              stream) == cudaSuccess &&
                  cudaStreamSynchronize(stream) == cudaSuccess,
              (what + ": the coordinates uploaded").c_str());
    return made;
}

/// Finds a lookup's coordinates and downloads their block numbers on the same stream.
/// @param tsdf [in] the sparse TSDF.
/// @param found [in,out] the coordinates and the room for their numbers, which are written.
/// @param stream [in] the stream, which the download waits for.
/// @param what [in] the case, named in the check lines.
/// @return each coordinate's block number or no_block; empty when the lookup failed.
inline std::vector<std::uint32_t> find(const gca::sparse_tsdf &tsdf, lookup &found,
                                       cudaStream_t stream, const std::string &what)
{
    const gca::step_result queued = tsdf.find_blocks(found.coordinates, found.numbers, stream);
    check(queued.done, (what + ": lookup queued, " + queued.message).c_str());
    std::vector<std::uint32_t> numbers(found.host.size(), 0);
    const bool downloaded =
        queued.done &&
        cudaMemcpyAsync(numbers.data(), found.numbers.data(), found.numbers.size_bytes(),
                        cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    check(downloaded, (what + ": the numbers downloaded").c_str());
    if (!downloaded)
        numbers.clear();
    return numbers;
}

/// Extracts a sparse TSDF's points and downloads them.
/// @param tsdf [in] the sparse TSDF.
/// @param buffer [in,out] the buffer of points.
/// @param stream [in] the stream.
/// @param what [in] the case, named in the check lines.
/// @return the points written; fewer, with a failed check, when the extraction or the download
///         failed.
inline std::vector<gca::surface_point> extract_points(const gca::sparse_tsdf &tsdf,
                                                      gca::surface_points &buffer,
                                                      cudaStream_t stream, const std::string &what)
{
    const gca::extraction_result extracted = tsdf.extract(buffer, stream);
    check(extracted.done && extracted.message.empty() && extracted.written == extracted.found,
          (what + ": extraction done, " + extracted.message).c_str());
    std::vector<gca::surface_point> points(extracted.written);
    const bool downloaded =
        points.empty() || (cudaMemcpyAsync(points.data(), buffer.points().data(),
                                           points.size() * sizeof(gca::surface_point),
                                           cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                           cudaStreamSynchronize(stream) == cudaSuccess);
    check(downloaded, (what + ": the points downloaded").c_str());
    if (!downloaded)
        points.clear();
    return points;
}

/// Reads the usage and checks that it was read.
/// @param tsdf [in,out] the sparse TSDF.
/// @param stream [in] the stream.
/// @param what [in] the case, named in the check line.
/// @return the usage; all 0, which no usage read is, and a failed check when it was not read.
inline gca::sparse_tsdf_usage read_usage(gca::sparse_tsdf &tsdf, cudaStream_t stream,
                                         const std::string &what)
{
    const gca::usage_result read = tsdf.read_usage(stream);
    check(read.usage.has_value() && read.message.empty(),
          (what + ": usage read, " + read.message).c_str());
    return read.usage.value_or(gca::sparse_tsdf_usage{});
}

/// Prints a usage.
/// @param what [in] the case, printed.
/// @param usage [in] the usage.
inline void print_usage(const std::string &what, const gca::sparse_tsdf_usage &usage)
{
    std::printf("%s: %u of %u blocks in use, %u refused for a used-up pool, %u coordinates "
                "refused, %u slots listed, %u unlisted, %u rays not walked\n",
                what.c_str(), usage.blocks_in_use, usage.capacity_blocks, usage.blocks_refused,
                usage.coordinates_refused, usage.listed_slots, usage.unlisted_slots,
                usage.rays_not_walked);
}

/// Compares two usages count by count.
/// @param a [in] one usage.
/// @param b [in] the other.
/// @return whether every count is equal.
inline bool same_usage(const gca::sparse_tsdf_usage &a, const gca::sparse_tsdf_usage &b)
{
    return a.capacity_blocks == b.capacity_blocks && a.blocks_in_use == b.blocks_in_use &&
           a.blocks_refused == b.blocks_refused && a.coordinates_refused == b.coordinates_refused &&
           a.listed_slots == b.listed_slots && a.unlisted_slots == b.unlisted_slots &&
           a.rays_not_walked == b.rays_not_walked;
}

/// Prints a usage and checks it against the expected counts.
/// @param what [in] the case, printed and named in the check line.
/// @param usage [in] the usage read.
/// @param expected [in] the expected usage.
inline void check_usage(const std::string &what, const gca::sparse_tsdf_usage &usage,
                        const gca::sparse_tsdf_usage &expected)
{
    print_usage(what, usage);
    check(same_usage(usage, expected), (what + ": the usage expected").c_str());
}
} // namespace gca_test
