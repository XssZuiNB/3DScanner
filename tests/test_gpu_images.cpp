// SPDX-License-Identifier: 0BSD
/// @file test_gpu_images.cpp
/// The gpu module's public API, the GPU image, the pinned host buffer, the pool that closes and
/// the GPU frame with its upload, on frames of the tests' synthetic source, a box on the plane:
/// - a frame uploaded and downloaded again is the frame bit for bit, its description, index,
///   timestamp and every byte of both pictures, downloaded into a frame that held other bytes; a
///   frame of another view uploaded into the same GPU frame then replaces the first;
/// - the upload stages the frame: its buffers may be overwritten as soon as it returns, and a
///   second upload waits until the first one's copies have run before it stages again, so that
///   work queued between the two sees the first frame;
/// - an image, a pinned buffer, a GPU frame cannot be copied (checked at compile time); one moved
///   from, by construction or by assignment, holds nothing, and the one moved to holds what it
///   held;
/// - the pool grants while it is open and refuses the same request as closed after its closing,
///   counting each such request, also one of a size cudaMalloc would refuse; it refuses an image
///   of no pixel, one of more bytes than std::size_t holds, one cudaMalloc cannot allocate, and a
///   pinned buffer of no byte or of more than cudaMallocHost can allocate, and CUDA's last error
///   is clear after each refusal;
/// - an upload is refused into an empty GPU frame, into one of another size and from a frame
///   moved from, and a refused upload leaves the GPU frame as it was; a download of a GPU frame
///   that holds no frame yet is refused.

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "check.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_image.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/pinned_buffer.hpp"
#include "gpu/step_result.hpp"
#include "gpu_test_helpers.hpp"
#include "synthetic_source.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_no_cuda_error;
using gca_test::check_refusal;
using gca_test::create_stream;
using gca_test::granted;

// An image, a pinned buffer and a GPU frame are moved, never copied; a pool is neither.
static_assert(!std::is_copy_constructible_v<gca::gpu_image<uchar3>> &&
                  !std::is_copy_assignable_v<gca::gpu_image<uchar3>>,
              "an image is not copied");
static_assert(std::is_nothrow_move_constructible_v<gca::gpu_image<uchar3>> &&
                  std::is_nothrow_move_assignable_v<gca::gpu_image<uchar3>>,
              "an image is moved without an exception");
static_assert(!std::is_copy_constructible_v<gca::pinned_buffer> &&
                  !std::is_copy_assignable_v<gca::pinned_buffer> &&
                  std::is_nothrow_move_constructible_v<gca::pinned_buffer> &&
                  std::is_nothrow_move_assignable_v<gca::pinned_buffer>,
              "a pinned buffer is moved, not copied");
static_assert(!std::is_copy_constructible_v<gca::gpu_frame> &&
                  !std::is_copy_assignable_v<gca::gpu_frame> &&
                  std::is_nothrow_move_constructible_v<gca::gpu_frame> &&
                  std::is_nothrow_move_assignable_v<gca::gpu_frame>,
              "a GPU frame is moved, not copied");
static_assert(!std::is_copy_constructible_v<gca::gpu_pool> &&
                  !std::is_move_constructible_v<gca::gpu_pool>,
              "a pool is neither copied nor moved");

/// The frames' size: small, so that every pixel is compared at once [?: a choice].
constexpr gca::frame_size test_size = {160, 120};

/// A size other than the frames': twice theirs [?: a choice].
constexpr gca::frame_size other_size = {320, 240};

/// A small size for the images that are only moved [?: a choice].
constexpr gca::frame_size small_size = {8, 4};

/// The larger stream size [S: docs/HARDWARE.md, the camera's sizes at 30 fps].
constexpr gca::frame_size large_stream_size = {1280, 800};

/// A size cudaMalloc cannot grant: 2^31 x 2^20 pixels of 2 bytes are 2^52 bytes, 4 PiB, where the
/// board has 7.4 GiB [S: computed; docs/HARDWARE.md], and below the largest std::size_t, so that
/// the pool's own checks pass it on to cudaMalloc.
constexpr gca::frame_size impossible_size = {1u << 31, 1u << 20};

/// A size whose bytes std::size_t cannot hold: 2^31 x 2^31 pixels of 16 bytes are 2^66 bytes,
/// which wrap to 0 in 64 bits [S: computed].
constexpr gca::frame_size overflowing_size = {1u << 31, 1u << 31};

/// The intrinsics: fx and fy differ, and the principal point lies at no whole or half pixel
/// [?: a choice].
constexpr gca::intrinsics test_intrinsics = {150.0f, 140.0f, 83.3f, 56.6f};

/// The depth unit: 0.1 mm, one of the Gemini 2 L's depth precisions [?: a choice], so that the
/// depth values change from pixel to pixel.
constexpr float test_depth_unit_m = 0.0001f;

/// The support plane: 0.6 m x 0.5 m [?: a choice].
constexpr gca_test::synthetic_plane test_plane = {0.30, 0.25};

/// A tabletop box, 0.10 m x 0.08 m x 0.06 m, turned by 25 degrees, standing where both views look,
/// so that each picture holds the box's faces and edges besides the plane [?: a choice].
constexpr gca_test::synthetic_box test_box = {0.02, 0.03, 25.0, 0.10, 0.08, 0.06};

/// The index of the first view's frame: not 0, so that an index lost on the way shows
/// [?: a choice].
constexpr std::uint64_t first_view_index = 2;

/// How long the staging check holds its stream back: far longer than the host needs from the
/// first upload's return to the second's copy into the staging buffer, well under 1 ms at the
/// test's size [?: a choice].
constexpr std::chrono::milliseconds stream_hold{200};

/// Returns the first view: 0.40 m above the plane, looking down at about 40 degrees, rolled by 12
/// degrees [?: a choice].
/// @return the pose, camera frame to world frame; nothing when look_at() refuses it.
std::optional<Eigen::Isometry3d> first_view()
{
    return gca_test::look_at({0.04, -0.45, 0.40}, {0.02, 0.03, 0.0}, 12.0);
}

/// Returns the second view: 0.30 m above the plane from elsewhere, rolled the other way, so that
/// its pictures differ from the first view's [?: a choice].
/// @return the pose, camera frame to world frame; nothing when look_at() refuses it.
std::optional<Eigen::Isometry3d> second_view()
{
    return gca_test::look_at({-0.10, -0.35, 0.30}, {0.0, 0.05, 0.0}, -8.0);
}

/// Takes a frame of the plane and the box from the synthetic source.
/// @param pose [in] the camera's pose, camera frame to world frame; nothing gives no frame.
/// @param index [in] the index of the frame taken: the source's trajectory holds the pose index + 1
///        times, its frames 0 to index are rendered and the last is kept.
/// @param out [out] the frame.
/// @return whether the source gave it.
bool take_frame(const std::optional<Eigen::Isometry3d> &pose, std::uint64_t index, gca::frame &out)
{
    if (!pose)
        return false;
    gca_test::synthetic_source_options options;
    options.description = {test_size, test_intrinsics, test_depth_unit_m};
    options.scene = {.plane = test_plane,
                     .box = test_box,
                     .trajectory = std::vector<Eigen::Isometry3d>(index + 1, *pose)};
    gca_test::synthetic_source source(std::move(options));
    if (!source.open(test_size).opened)
        return false;
    for (std::uint64_t i = 0; i <= index; ++i)
    {
        if (source.next(out).status != gca::next_status::frame)
            return false;
    }
    return true;
}

/// Takes the first view's and the second view's frame, and checks that their pictures are not
/// blank and differ, so that a copy that is missing or stale shows.
/// @param first [out] the first view's frame, index first_view_index.
/// @param second [out] the second view's frame, index 0.
/// @return whether the source gave both.
bool take_two_views(gca::frame &first, gca::frame &second)
{
    const bool taken =
        take_frame(first_view(), first_view_index, first) && take_frame(second_view(), 0, second);
    check(taken, "the synthetic source gave both views' frames");
    if (!taken)
        return false;
    const auto with_depth = [](const gca::frame &frame) {
        return std::ranges::count_if(frame.depth(), [](std::uint16_t value) { return value != 0; });
    };
    const auto pixels = static_cast<std::ptrdiff_t>(first.depth().size());
    check(2 * with_depth(first) > pixels && 2 * with_depth(second) > pixels,
          "more than half of each view's picture has depth");
    check(!std::ranges::equal(first.depth(), second.depth()) &&
              !std::ranges::equal(first.color(), second.color()),
          "the two views differ");
    return true;
}

/// Fills a frame of the test size with bytes no picture of the scene holds: color 0xAB in every
/// byte, depth 0xABAB, index and timestamp 999 [?: a choice].
/// @param out [out] the frame.
void fill_with_other_bytes(gca::frame &out)
{
    out.prepare({test_size, {1.0f, 1.0f, 0.0f, 0.0f}, 1.0f}, 999, 999);
    std::ranges::fill(out.color(), std::uint8_t{0xAB});
    std::ranges::fill(out.depth(), std::uint16_t{0xABAB});
}

/// Counts the elements in which two pictures differ.
/// @param got [in] the picture that came back.
/// @param expected [in] the picture it should be, of the same size.
/// @return the elements that differ.
template <typename Value>
std::size_t differing(std::span<const Value> got, std::span<const Value> expected)
{
    std::size_t count = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        count += got[i] != expected[i] ? 1 : 0;
    return count;
}

/// Checks that two pictures are equal bit for bit and prints how many elements differ.
/// @param what [in] the case, printed and named in the check line.
/// @param got_color [in] the color picture that came back.
/// @param expected_color [in] the color picture it should be.
/// @param got_depth [in] the depth picture that came back.
/// @param expected_depth [in] the depth picture it should be.
void check_same_pictures(const std::string &what, std::span<const std::uint8_t> got_color,
                         std::span<const std::uint8_t> expected_color,
                         std::span<const std::uint16_t> got_depth,
                         std::span<const std::uint16_t> expected_depth)
{
    const bool same_sizes =
        got_color.size() == expected_color.size() && got_depth.size() == expected_depth.size();
    check(same_sizes, (what + ": the pictures' sizes").c_str());
    if (!same_sizes)
        return;
    const std::size_t color_bytes = differing(got_color, expected_color);
    const std::size_t depth_pixels = differing(got_depth, expected_depth);
    std::printf("%s: %zu of %zu color bytes and %zu of %zu depth pixels differ\n", what.c_str(),
                color_bytes, got_color.size(), depth_pixels, got_depth.size());
    check(color_bytes == 0 && depth_pixels == 0, (what + ": every byte of both pictures").c_str());
}

/// Checks that a frame is another bit for bit: description, index, timestamp, both pictures.
/// @param what [in] the case, printed and named in the check line.
/// @param got [in] the frame that came back.
/// @param expected [in] the frame it should be.
void check_same_frame(const std::string &what, const gca::frame &got, const gca::frame &expected)
{
    const gca::frame_description &a = got.description();
    const gca::frame_description &b = expected.description();
    const bool same_description =
        a.size == b.size && a.intrinsics.fx == b.intrinsics.fx &&
        a.intrinsics.fy == b.intrinsics.fy && a.intrinsics.cx == b.intrinsics.cx &&
        a.intrinsics.cy == b.intrinsics.cy && a.depth_unit_m == b.depth_unit_m;
    check(same_description, (what + ": the description").c_str());
    check(got.index() == expected.index() && got.timestamp_us() == expected.timestamp_us(),
          (what + ": the index and the timestamp").c_str());
    check_same_pictures(what, got.color(), expected.color(), got.depth(), expected.depth());
}

/// A frame uploaded and downloaded again is the frame bit for bit; a frame of another view
/// uploaded into the same GPU frame replaces it; a refused upload leaves the GPU frame as it was.
/// @param stream [in] the CUDA stream the copies run on.
void check_round_trip(cudaStream_t stream)
{
    gca::frame first;
    gca::frame second;
    if (!take_two_views(first, second))
        return;
    gca::gpu_pool pool;
    gca::gpu_frame on_gpu = granted("a GPU frame", gca::gpu_frame::request(pool, test_size));
    pool.close();
    gca::frame back;

    const gca::step_result up = on_gpu.upload(first, stream);
    check(up.done, ("round trip: upload, " + up.message).c_str());
    fill_with_other_bytes(back);
    const gca::step_result down = on_gpu.download(back, stream);
    check(down.done, ("round trip: download, " + down.message).c_str());
    check_same_frame("round trip, frame 2 of the first view", back, first);

    const gca::step_result up_again = on_gpu.upload(second, stream);
    check(up_again.done, ("round trip: second upload, " + up_again.message).c_str());
    fill_with_other_bytes(back);
    const gca::step_result down_again = on_gpu.download(back, stream);
    check(down_again.done, ("round trip: second download, " + down_again.message).c_str());
    check_same_frame("round trip, frame 0 of the second view", back, second);

    // A frame moved from keeps its description, not its buffers.
    gca::frame taker(std::move(first));
    const gca::step_result moved = on_gpu.upload(first, stream);
    check_refusal("upload of a frame moved from", !moved.done, moved.message,
                  moved.message.find("a frame moved from") != std::string::npos);
    fill_with_other_bytes(back);
    const gca::step_result kept = on_gpu.download(back, stream);
    check(kept.done, ("refused upload: download, " + kept.message).c_str());
    check_same_frame("refused upload: the GPU frame as it was", back, second);
    check(pool.refused_while_closed() == 0, "round trip: the closed pool asked for nothing");
}

/// Holds a stream back: queued with cudaLaunchHostFunc, it sleeps for stream_hold, and the work
/// queued after it waits. It calls no CUDA function, as a host function must not
/// [S: cuda_runtime_api.h, cudaLaunchHostFunc, CUDA 12.6].
/// @param unused [in] cudaLaunchHostFunc's user data, nullptr.
void CUDART_CB hold_stream(void *unused)
{
    static_cast<void>(unused);
    std::this_thread::sleep_for(stream_hold);
}

/// The upload stages the frame: the frame's buffers may be overwritten once the upload returns,
/// and a second upload stages only after the first one's copies have run, so that work queued
/// between the two sees the first frame. The stream is held back, so that the first upload's
/// copies are still queued when the second upload starts.
/// @param stream [in] the CUDA stream the copies run on.
void check_staging(cudaStream_t stream)
{
    gca::frame first;
    gca::frame second;
    if (!take_two_views(first, second))
        return;
    const std::vector<std::uint8_t> first_color(first.color().begin(), first.color().end());
    const std::vector<std::uint16_t> first_depth(first.depth().begin(), first.depth().end());
    gca::gpu_pool pool;
    gca::gpu_frame on_gpu = granted("a GPU frame", gca::gpu_frame::request(pool, test_size));
    gca::request_result<gca::gpu_image<uchar3>> seen_color = pool.request<uchar3>(test_size);
    gca::request_result<gca::gpu_image<std::uint16_t>> seen_depth =
        pool.request<std::uint16_t>(test_size);
    pool.close();
    check(seen_color.status == gca::request_status::granted &&
              seen_depth.status == gca::request_status::granted,
          "staging: the images of what the work between the uploads sees, granted");

    check(cudaLaunchHostFunc(stream, hold_stream, nullptr) == cudaSuccess,
          "staging: the stream held back");
    const gca::step_result up = on_gpu.upload(first, stream);
    check(up.done, ("staging: first upload, " + up.message).c_str());
    std::ranges::fill(first.color(), std::uint8_t{0});
    std::ranges::fill(first.depth(), std::uint16_t{0});
    // The work between the uploads: a copy of the images as the first upload leaves them.
    check(cudaMemcpyAsync(seen_color.granted.data(), on_gpu.color().data(),
                          on_gpu.color().size_bytes(), cudaMemcpyDeviceToDevice,
                          stream) == cudaSuccess &&
              cudaMemcpyAsync(seen_depth.granted.data(), on_gpu.depth().data(),
                              on_gpu.depth().size_bytes(), cudaMemcpyDeviceToDevice,
                              stream) == cudaSuccess,
          "staging: the work between the uploads queued");
    const auto before = std::chrono::steady_clock::now();
    const gca::step_result up_again = on_gpu.upload(second, stream);
    const std::chrono::duration<double, std::milli> waited =
        std::chrono::steady_clock::now() - before;
    check(up_again.done, ("staging: second upload, " + up_again.message).c_str());
    std::printf("staging: the second upload returned after %.1f ms, the stream held for %lld ms\n",
                waited.count(), static_cast<long long>(stream_hold.count()));

    gca::frame back;
    const gca::step_result down = on_gpu.download(back, stream);
    check(down.done, ("staging: download, " + down.message).c_str());
    check_same_frame("staging: the second upload", back, second);
    std::vector<std::uint8_t> seen_color_host(first_color.size());
    std::vector<std::uint16_t> seen_depth_host(first_depth.size());
    check(cudaMemcpy(seen_color_host.data(), seen_color.granted.data(),
                     seen_color.granted.size_bytes(), cudaMemcpyDeviceToHost) == cudaSuccess &&
              cudaMemcpy(seen_depth_host.data(), seen_depth.granted.data(),
                         seen_depth.granted.size_bytes(), cudaMemcpyDeviceToHost) == cudaSuccess,
          "staging: what the work between the uploads saw, read back");
    check_same_pictures("staging: the work between the uploads sees the first frame as uploaded",
                        seen_color_host, first_color, seen_depth_host, first_depth);
}

/// An image, a pinned buffer and a GPU frame moved from hold nothing, and the one moved to holds
/// what it held.
/// @param stream [in] the CUDA stream the copies run on.
void check_moves(cudaStream_t stream)
{
    gca::gpu_pool pool;
    gca::request_result<gca::gpu_image<std::uint16_t>> granted =
        pool.request<std::uint16_t>(test_size);
    gca::request_result<gca::gpu_image<std::uint16_t>> other =
        pool.request<std::uint16_t>(small_size);
    check(granted.status == gca::request_status::granted &&
              other.status == gca::request_status::granted,
          "moves: both images granted");
    const std::uint16_t *memory = granted.granted.data();

    gca::gpu_image<std::uint16_t> constructed(std::move(granted.granted));
    check(granted.granted.empty() && granted.granted.data() == nullptr &&
              granted.granted.size() == gca::frame_size{0, 0} && granted.granted.size_bytes() == 0,
          "moves: an image moved from by construction holds nothing");
    check(!constructed.empty() && constructed.data() == memory && constructed.size() == test_size &&
              constructed.size_bytes() == 160 * 120 * 2,
          "moves: the image constructed holds its memory and size");

    gca::gpu_image<std::uint16_t> assigned(std::move(other.granted));
    assigned = std::move(constructed);
    check(constructed.empty() && constructed.data() == nullptr &&
              constructed.size() == gca::frame_size{0, 0} && constructed.size_bytes() == 0,
          "moves: an image moved from by assignment holds nothing");
    check(assigned.data() == memory && assigned.size() == test_size,
          "moves: the image assigned to holds its memory and size");
    const gca::gpu_image<uchar3> made_empty;
    check(made_empty.empty() && made_empty.data() == nullptr && made_empty.size_bytes() == 0,
          "moves: an image made empty holds nothing");

    gca::request_result<gca::pinned_buffer> pinned = pool.request_pinned(64);
    gca::request_result<gca::pinned_buffer> other_pinned = pool.request_pinned(16);
    check(pinned.status == gca::request_status::granted &&
              other_pinned.status == gca::request_status::granted,
          "moves: both pinned buffers granted");
    const std::byte *pinned_memory = pinned.granted.bytes().data();
    gca::pinned_buffer pinned_constructed(std::move(pinned.granted));
    check(pinned.granted.empty() && pinned.granted.bytes().empty(),
          "moves: a pinned buffer moved from by construction holds nothing");
    gca::pinned_buffer pinned_assigned(std::move(other_pinned.granted));
    pinned_assigned = std::move(pinned_constructed);
    check(pinned_constructed.empty() && pinned_constructed.bytes().empty(),
          "moves: a pinned buffer moved from by assignment holds nothing");
    check(pinned_assigned.bytes().data() == pinned_memory && pinned_assigned.bytes().size() == 64,
          "moves: the pinned buffer assigned to holds its memory and size");

    gca::frame first;
    gca::frame second;
    if (!take_two_views(first, second))
        return;
    gca::gpu_frame uploaded =
        gca_test::granted("a GPU frame", gca::gpu_frame::request(pool, test_size));
    gca::gpu_frame target =
        gca_test::granted("a GPU frame", gca::gpu_frame::request(pool, test_size));
    check(uploaded.upload(first, stream).done, "moves: a frame uploaded");
    gca::gpu_frame frame_constructed(std::move(uploaded));
    check(uploaded.color().empty() && uploaded.depth().empty() &&
              uploaded.description().size == gca::frame_size{0, 0} && uploaded.index() == 0 &&
              uploaded.timestamp_us() == 0,
          "moves: a GPU frame moved from by construction holds nothing");
    const gca::step_result into_moved = uploaded.upload(first, stream);
    check_refusal("upload into a GPU frame moved from", !into_moved.done, into_moved.message,
                  into_moved.message.find("the GPU frame is empty") != std::string::npos);
    target = std::move(frame_constructed);
    check(frame_constructed.color().empty() && frame_constructed.depth().empty() &&
              frame_constructed.index() == 0,
          "moves: a GPU frame moved from by assignment holds nothing");
    gca::frame back;
    const gca::step_result down = target.download(back, stream);
    check(down.done, ("moves: download, " + down.message).c_str());
    check_same_frame("moves: the GPU frame assigned to holds the uploaded frame", back, first);
}

/// The pool grants while open and refuses as closed after its closing, counting each such
/// request, whatever its size; it refuses sizes it cannot grant, and CUDA's last error is clear
/// after each refusal.
void check_pool()
{
    gca::gpu_pool pool;
    check(!pool.closed() && pool.refused_while_closed() == 0, "pool: open when built");
    const gca::request_result<gca::gpu_image<uchar3>> before =
        pool.request<uchar3>(large_stream_size);
    std::printf("pool, before the closing: %s\n",
                before.status == gca::request_status::granted ? "granted" : before.message.c_str());
    check(before.status == gca::request_status::granted && before.message.empty() &&
              !before.granted.empty() && before.granted.size() == large_stream_size &&
              before.granted.size_bytes() == 1280 * 800 * 3,
          "pool: the request before the closing granted, 1280x800 color");
    check(pool.refused_while_closed() == 0, "pool: nothing refused while open");

    pool.close();
    check(pool.closed(), "pool: closed after close()");
    const gca::request_result<gca::gpu_image<uchar3>> after =
        pool.request<uchar3>(large_stream_size);
    check_refusal("pool, the same request after the closing",
                  after.status == gca::request_status::closed, after.message,
                  after.message.find("the GPU pool is closed") != std::string::npos);
    check(after.granted.empty(), "pool: the refused request's image holds nothing");
    pool.close();
    const gca::request_result<gca::gpu_image<std::uint16_t>> impossible_closed =
        pool.request<std::uint16_t>(impossible_size);
    check_refusal("pool, closed, an image cudaMalloc cannot allocate",
                  impossible_closed.status == gca::request_status::closed,
                  impossible_closed.message,
                  impossible_closed.message.find("the GPU pool is closed") != std::string::npos);
    const gca::request_result<gca::pinned_buffer> pinned_closed = pool.request_pinned(16);
    check_refusal("pool, closed, a pinned buffer",
                  pinned_closed.status == gca::request_status::closed, pinned_closed.message,
                  pinned_closed.message.find("the GPU pool is closed") != std::string::npos);
    const gca::request_result<gca::gpu_frame> frame_closed =
        gca::gpu_frame::request(pool, test_size);
    check_refusal("pool, closed, a GPU frame", frame_closed.status == gca::request_status::closed,
                  frame_closed.message,
                  frame_closed.message.find("the GPU pool is closed") != std::string::npos);
    // The four requests after the closing; the GPU frame's stops at its first image.
    std::printf("pool: %llu requests refused while closed\n",
                static_cast<unsigned long long>(pool.refused_while_closed()));
    check(pool.closed() && pool.refused_while_closed() == 4,
          "pool: closed again, still closed, and the four requests after the closing counted");
    check_no_cuda_error("pool, closed");

    gca::gpu_pool open_pool;
    const gca::request_result<gca::gpu_image<uchar3>> no_width =
        open_pool.request<uchar3>({0, 120});
    check_refusal("pool, an image 0 px wide", no_width.status == gca::request_status::failure,
                  no_width.message, no_width.message.find("holds no pixel") != std::string::npos);
    const gca::request_result<gca::gpu_image<uchar3>> no_height =
        open_pool.request<uchar3>({160, 0});
    check_refusal("pool, an image 0 px high", no_height.status == gca::request_status::failure,
                  no_height.message, no_height.message.find("holds no pixel") != std::string::npos);
    const gca::request_result<gca::gpu_image<float4>> too_large =
        open_pool.request<float4>(overflowing_size);
    check_refusal("pool, an image of 2^66 bytes", too_large.status == gca::request_status::failure,
                  too_large.message,
                  too_large.message.find("more bytes than std::size_t holds") != std::string::npos);
    const gca::request_result<gca::gpu_image<std::uint16_t>> impossible =
        open_pool.request<std::uint16_t>(impossible_size);
    check_refusal("pool, open, an image cudaMalloc cannot allocate",
                  impossible.status == gca::request_status::failure, impossible.message,
                  impossible.message.find("CUDA could not allocate") != std::string::npos);
    check_no_cuda_error("pool, after cudaMalloc's refusal");
    const gca::request_result<gca::pinned_buffer> no_bytes = open_pool.request_pinned(0);
    check_refusal("pool, a pinned buffer of 0 bytes",
                  no_bytes.status == gca::request_status::failure, no_bytes.message,
                  no_bytes.message.find("holds nothing") != std::string::npos);
    // As many bytes as the image cudaMalloc cannot allocate [S: computed].
    const gca::request_result<gca::pinned_buffer> impossible_pinned =
        open_pool.request_pinned(std::size_t{1} << 52);
    check_refusal("pool, open, a pinned buffer cudaMallocHost cannot allocate",
                  impossible_pinned.status == gca::request_status::failure,
                  impossible_pinned.message,
                  impossible_pinned.message.find("CUDA could not allocate") != std::string::npos);
    check_no_cuda_error("pool, after cudaMallocHost's refusal");
    check(no_width.granted.empty() && no_height.granted.empty() && too_large.granted.empty() &&
              impossible.granted.empty() && no_bytes.granted.empty() &&
              impossible_pinned.granted.empty(),
          "pool: the refused requests' buffers hold nothing");
    check(open_pool.refused_while_closed() == 0, "pool: an open pool counts no refusal as closed");
}

/// An upload into an empty GPU frame or one of another size is refused and leaves the GPU frame
/// as it was; a download of a GPU frame that holds no frame yet is refused.
/// @param stream [in] the CUDA stream the copies would run on.
void check_copy_refusals(cudaStream_t stream)
{
    gca::frame frame;
    check(take_frame(first_view(), first_view_index, frame),
          "refusals: the synthetic source gave a frame");

    gca::gpu_frame empty;
    const gca::step_result none = empty.upload(frame, stream);
    check_refusal("upload into an empty GPU frame", !none.done, none.message,
                  none.message.find("the GPU frame is empty") != std::string::npos);

    gca::gpu_pool pool;
    gca::gpu_frame larger = granted("a GPU frame", gca::gpu_frame::request(pool, other_size));
    const gca::step_result other = larger.upload(frame, stream);
    check_refusal("upload into a GPU frame of 320x240 px", !other.done, other.message,
                  other.message.find("the frame is 160x120 px, the GPU frame 320x240 px") !=
                      std::string::npos);
    check(larger.description().size == gca::frame_size{0, 0} && larger.index() == 0 &&
              larger.timestamp_us() == 0,
          "refusals: the refused upload left the GPU frame as it was");

    gca::gpu_frame never_uploaded =
        granted("a GPU frame", gca::gpu_frame::request(pool, test_size));
    gca::frame back;
    const gca::step_result no_frame = never_uploaded.download(back, stream);
    check_refusal("download of a GPU frame that holds no frame yet", !no_frame.done,
                  no_frame.message,
                  no_frame.message.find("holds no frame yet") != std::string::npos);
    const gca::step_result from_empty = empty.download(back, stream);
    check_refusal("download of an empty GPU frame", !from_empty.done, from_empty.message,
                  from_empty.message.find("the GPU frame is empty") != std::string::npos);
    check_no_cuda_error("refusals");
}
} // namespace

int main()
{
    cudaStream_t stream = nullptr;
    if (!create_stream(stream))
        return gca_test::check_summary("test_gpu_images");
    check_round_trip(stream);
    check_staging(stream);
    check_moves(stream);
    check_pool();
    check_copy_refusals(stream);
    check(cudaStreamDestroy(stream) == cudaSuccess, "the CUDA stream destroyed");
    return gca_test::check_summary("test_gpu_images");
}
