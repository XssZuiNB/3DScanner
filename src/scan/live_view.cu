// SPDX-License-Identifier: 0BSD
/// @file live_view.cu
/// The live view: the newest view under its lock, the shading kernel and the encoder with its
/// thread (see live_view.hpp; docs/methods/tracking.md, "The live view").

#include "scan/live_view.hpp"

#include "geometry/color_channel.cuh"
#include "geometry/float3_ops.cuh"
#include "gpu/gpu_image.hpp"
#include "gpu/gpu_messages.hpp"
#include "gpu/kernel_launch.cuh"
#include "gpu/pinned_buffer.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <thread>
#include <utility>

namespace gca
{
namespace
{
static_assert(sizeof(uchar3) == 3, "a picture's pixel is 3 bytes, as OpenCV's CV_8UC3");

/// Shades the surface's level 0 into the live view's picture (docs/methods/tracking.md, "The live
/// view"), one thread for each pixel: a valid pixel with a normal n shows its color c times
/// a + (1 - a) max(0, n . (-p / |p|)), p its point in the camera frame, a = live_view_ambient; a
/// valid pixel without a normal its color unlit; an invalid pixel live_view_background. Written in
/// OpenCV's order, blue, green, red.
/// @param points [in] level 0's points, rows dense, in the camera frame, in m; device memory of
///        the caller's surface, which holds it until the kernel has run. The same holds for the
///        marks, the normals and the colors.
/// @param valid [in] level 0's marks, 1 for a valid pixel.
/// @param normals [in] level 0's normals, unit vectors toward the camera, or (0, 0, 0) for none.
/// @param colors [in] level 0's colors, red, green and blue.
/// @param size [in] level 0's width and height, in px.
/// @param picture [out] the picture, rows dense, blue, green, red; device memory of the encoder,
///        which holds it until the kernel has run.
/// @note Runs on the stream of its launch, after the ray cast it reads. Allocates nothing.
__global__ void __launch_bounds__(picture_block_width *picture_block_height)
    shade_kernel(const float3 *__restrict__ points, const std::uint8_t *__restrict__ valid,
                 const float3 *__restrict__ normals, const uchar3 *__restrict__ colors,
                 frame_size size, uchar3 *__restrict__ picture)
{
    const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= size.width || v >= size.height)
        return;
    const std::size_t pixel = static_cast<std::size_t>(v) * size.width + u;
    uchar3 shown = make_uchar3(live_view_background, live_view_background, live_view_background);
    if (valid[pixel] != 0)
    {
        const uchar3 c = colors[pixel];
        const float3 n = normals[pixel];
        float light = 1.0f;
        if (n.x != 0.0f || n.y != 0.0f || n.z != 0.0f)
        {
            // A valid point lies at the depth range's near limit or deeper, so |p| > 0.
            const float3 p = points[pixel];
            const float facing = -dot(n, p) / norm(p);
            light = live_view_ambient + (1.0f - live_view_ambient) * fmaxf(facing, 0.0f);
        }
        // The light is at most 1 but for float's rounding (rounded_channel()).
        shown = make_uchar3(rounded_channel(static_cast<float>(c.z) * light),
                            rounded_channel(static_cast<float>(c.y) * light),
                            rounded_channel(static_cast<float>(c.x) * light));
    }
    picture[pixel] = shown;
}
} // namespace

newest_live_view::newest_live_view(const frame_size &size,
                                   std::function<void(const live_view &)> on_view)
    : m_on_view(std::move(on_view))
{
    m_published.jpeg.reserve(std::size_t{3} * size.width * size.height);
}

live_view newest_live_view::latest() const
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    return m_published;
}

void newest_live_view::clear() noexcept
{
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        m_published.version++;
        m_published.frame.reset();
        m_published.timestamp_us = 0;
        m_published.size = {0, 0};
        m_published.jpeg.clear();
    }
    notify();
}

void newest_live_view::publish(std::uint64_t frame, std::uint64_t timestamp_us,
                               const frame_size &size, std::vector<std::uint8_t> &jpeg) noexcept
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    m_published.version++;
    m_published.frame = frame;
    m_published.timestamp_us = timestamp_us;
    m_published.size = size;
    m_published.jpeg.swap(jpeg);
}

void newest_live_view::notify() noexcept
{
    // Only the publishing thread changes m_published, so it stays as it is during the call.
    if (m_on_view)
        m_on_view(m_published);
}

/// What the encoder's thread shares with the offers, and what it owns.
struct live_view_encoder::state
{
    /// Builds the state of an encoder whose memory and event were granted.
    /// @param into [in,out] the newest live view the views go into; outlives the state.
    /// @param size_px [in] the pictures' size, in px.
    /// @param on_gpu [in,out] the picture on the GPU; taken over, left empty.
    /// @param on_host_bytes [in,out] its page-locked copy; taken over, left empty.
    /// @param event [in] the event recorded after the copy; owned by the state from now on.
    /// @note Allocates the JPEG's bytes, 3 bytes a pixel; a failed allocation ends the program
    ///       (std::terminate): the function is noexcept.
    state(newest_live_view &into, const frame_size &size_px, gpu_image<uchar3> &&on_gpu,
          pinned_buffer &&on_host_bytes, cudaEvent_t event) noexcept
        : views(into)
        , size(size_px)
        , picture(std::move(on_gpu))
        , on_host(std::move(on_host_bytes))
        , copied(event)
    {
        jpeg.reserve(std::size_t{3} * size.width * size.height);
    }

    state(const state &) = delete;
    state &operator=(const state &) = delete;

    /// Ends the thread after the view it was handed, then destroys the event; the picture and
    /// the page-locked copy are freed after this body.
    ~state()
    {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            ending = true;
        }
        wake.notify_one();
        if (thread.joinable())
            thread.join();
        cudaEventDestroy(copied);
    }

    /// Says why a surface cannot be shaded into the picture.
    /// @param surface [in] the surface.
    /// @return empty when it holds colors of the picture's size; else the reason.
    /// @note Allocates for the reason.
    std::string surface_refusal(const model_surface &surface) const
    {
        if (!surface.colors().empty() && surface.size(0) == size)
            return {};
        return "the surface is " + size_text(surface.size(0)) + ", the live view's picture " +
               size_text(size);
    }

    /// Queues a view's making on the GPU: the shading of the surface's level 0 into the picture,
    /// its copy into the page-locked picture, and the event after it.
    /// @param surface [in] the surface, of the picture's size (surface_refusal()).
    /// @param stream [in] the CUDA stream the surface's cast was queued on.
    /// @return done when all three are queued; else CUDA's error, then cleared, and the picture
    ///         unspecified.
    /// @note On the scan thread, while the encoder is free; waits for nothing. Allocates for the
    ///       message only.
    step_result queue_view(const model_surface &surface, cudaStream_t stream) noexcept
    {
        const dim3 block = picture_block();
        const step_result shaded =
            launch_kernel("the live view's shading kernel", &shade_kernel, grid_for(size, block),
                          block, stream, surface.points(0).data(), surface.validity_mask(0).data(),
                          surface.normals(0).data(), surface.colors().data(), size, picture.data());
        if (!shaded.done)
            return shaded;
        const cudaError_t copy =
            cudaMemcpyAsync(on_host.bytes().data(), picture.data(), picture.size_bytes(),
                            cudaMemcpyDeviceToHost, stream);
        if (copy != cudaSuccess)
            return {false, cuda_failure("CUDA refused the live view's copy", copy)};
        const cudaError_t recorded = cudaEventRecord(copied, stream);
        if (recorded != cudaSuccess)
            return {false, cuda_failure("CUDA refused the live view's event", recorded)};
        return {true, {}};
    }

    /// Hands a frame over to the thread, its view queued (queue_view()): the encoder is busy
    /// from here on.
    /// @param handed_frame [in] the frame's index.
    /// @param handed_timestamp_us [in] its timestamp, in µs.
    /// @pre The caller holds the lock, and the encoder is free.
    void hand_over(std::uint64_t handed_frame, std::uint64_t handed_timestamp_us) noexcept
    {
        busy = true;
        handed = true;
        frame = handed_frame;
        timestamp_us = handed_timestamp_us;
        last_handed = handed_frame;
    }

    /// The thread: takes each frame handed over, waits for its picture's copy, encodes it as a
    /// JPEG, publishes it and calls the listener, and only then becomes free; until the end, after
    /// the frame handed over last.
    /// @note Waits for the GPU (cudaEventSynchronize, blocking); allocates for the JPEG (OpenCV's
    ///       imencode and libjpeg) and for a failure's message, and a failed allocation outside
    ///       imencode ends the program (std::terminate).
    void run() noexcept
    {
        // OpenCV's parameters as pairs of a key and its value.
        const std::vector<int> quality = {cv::IMWRITE_JPEG_QUALITY, live_view_jpeg_quality};
        while (true)
        {
            std::uint64_t taken_frame = 0;
            std::uint64_t taken_timestamp_us = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [this] { return handed || ending; });
                if (!handed)
                    return;
                handed = false;
                taken_frame = frame;
                taken_timestamp_us = timestamp_us;
            }
            std::string failed;
            const cudaError_t waited = cudaEventSynchronize(copied);
            if (waited != cudaSuccess)
                failed = cuda_failure("CUDA failed while the live view's picture was made", waited);
            else
            {
                // The pinned picture as OpenCV's 8-bit three-channel image, without a copy.
                const cv::Mat image(static_cast<int>(size.height), static_cast<int>(size.width),
                                    CV_8UC3, on_host.bytes().data());
                bool encoded = false;
                try
                {
                    encoded = cv::imencode(".jpg", image, jpeg, quality);
                }
                catch (const std::exception &error)
                {
                    failed = std::string("OpenCV could not encode the live view: ") + error.what();
                }
                if (failed.empty() && !encoded)
                    failed = "OpenCV's imencode refused the live view's picture";
            }
            if (failed.empty())
            {
                views.publish(taken_frame, taken_timestamp_us, size, jpeg);
                // Busy until the listener returns, so that a frame offered meanwhile has no view
                // rather than one made after it (newest only).
                views.notify();
            }
            {
                const std::lock_guard<std::mutex> lock(mutex);
                busy = false;
                failure = std::move(failed);
            }
            idle.notify_all();
        }
    }

    newest_live_view &views;        ///< where the views go
    frame_size size;                ///< the pictures' size, in px
    gpu_image<uchar3> picture;      ///< the shaded picture on the GPU, blue, green, red
    pinned_buffer on_host;          ///< its page-locked copy, which the thread encodes
    cudaEvent_t copied;             ///< recorded after the copy of the frame handed over last
    std::vector<std::uint8_t> jpeg; ///< the JPEG the thread encodes into; its thread's alone

    // The lock guards the rest; the conditions wake the thread when a frame is handed over or the
    // end comes, and finish() when the encoder becomes free. The thread holds the lock only to take
    // a frame or to become free.
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    bool busy = false;       ///< from the offer that hands a frame over until its view is published
                             ///< and its listener has returned, or the view has failed
    bool handed = false;     ///< a frame handed over and not yet taken by the thread
    bool ending = false;     ///< the encoder's end: the thread ends after the frame handed over
    std::uint64_t frame = 0; ///< the frame handed over: its index
    std::uint64_t timestamp_us = 0;           ///< its timestamp, in µs
    std::optional<std::uint64_t> last_handed; ///< the last frame handed over; none before the first
    std::string failure; ///< why the last view failed, for the next offer; empty when it did not

    std::thread thread; ///< the thread, started last
};

live_view_encoder::live_view_encoder() noexcept = default;

request_result<live_view_encoder> live_view_encoder::request(gpu_pool &pool, const frame_size &size,
                                                             newest_live_view &views) noexcept
{
    request_result<gpu_image<uchar3>> picture = pool.request<uchar3>(size);
    if (picture.status != request_status::granted)
        return refused_as<live_view_encoder>(picture);
    request_result<pinned_buffer> on_host = pool.request_pinned(picture.granted.size_bytes());
    if (on_host.status != request_status::granted)
        return refused_as<live_view_encoder>(on_host);
    request_result<live_view_encoder> result;
    cudaEvent_t event = nullptr;
    const cudaError_t created =
        cudaEventCreateWithFlags(&event, cudaEventBlockingSync | cudaEventDisableTiming);
    if (created != cudaSuccess)
    {
        result.message = cuda_failure("CUDA could not create the live view's event", created);
        return result;
    }
    result.granted.m_state = std::make_unique<state>(views, size, std::move(picture.granted),
                                                     std::move(on_host.granted), event);
    state &made = *result.granted.m_state;
    made.thread = std::thread(&state::run, &made);
    result.status = request_status::granted;
    return result;
}

live_view_encoder::live_view_encoder(live_view_encoder &&other) noexcept = default;

live_view_encoder &live_view_encoder::operator=(live_view_encoder &&other) noexcept
{
    if (this != &other)
    {
        release();
        m_state = std::move(other.m_state);
    }
    return *this;
}

live_view_encoder::~live_view_encoder()
{
    release();
}

void live_view_encoder::release() noexcept
{
    m_state.reset();
}

live_view_offer live_view_encoder::offer(const model_surface &surface, std::uint64_t frame,
                                         std::uint64_t timestamp_us, cudaStream_t stream) noexcept
{
    if (!m_state)
        return {live_view_offer_status::failed,
                "the live view's encoder is empty: it was not granted by the pool, or moved from"};
    state &s = *m_state;
    std::string refusal = s.surface_refusal(surface);
    if (!refusal.empty())
        return {live_view_offer_status::failed, std::move(refusal)};
    {
        const std::lock_guard<std::mutex> lock(s.mutex);
        if (s.ending)
            return {live_view_offer_status::failed, "the live view's encoder has ended"};
        if (!s.failure.empty())
            return {live_view_offer_status::failed,
                    "the live view's last picture: " + std::exchange(s.failure, std::string())};
        // Only this thread makes the encoder busy, so it stays free until the hand-over below.
        if (s.busy)
            return {live_view_offer_status::busy, {}};
    }
    const step_result queued = s.queue_view(surface, stream);
    if (!queued.done)
        return {live_view_offer_status::failed, queued.message};
    {
        const std::lock_guard<std::mutex> lock(s.mutex);
        s.hand_over(frame, timestamp_us);
    }
    s.wake.notify_one();
    return {live_view_offer_status::queued, {}};
}

step_result live_view_encoder::finish(const model_surface &surface, std::uint64_t frame,
                                      std::uint64_t timestamp_us, cudaStream_t stream) noexcept
{
    if (!m_state)
        return {false,
                "the live view's encoder is empty: it was not granted by the pool, or moved from"};
    state &s = *m_state;
    std::string reasons;
    const auto note = [&reasons](const std::string &reason) {
        if (!reason.empty())
            reasons += (reasons.empty() ? "" : "; ") + reason;
    };
    bool hand = false;
    {
        std::unique_lock<std::mutex> lock(s.mutex);
        if (s.ending)
            return {false, "the live view's encoder has ended"};
        // The end may wait: the view in flight and its listener.
        s.idle.wait(lock, [&s] { return !s.busy; });
        if (!s.failure.empty())
            note("the live view's last picture: " + std::exchange(s.failure, std::string()));
        hand = s.last_handed != frame;
        if (!hand)
            s.ending = true;
    }
    if (hand)
    {
        std::string refusal = s.surface_refusal(surface);
        step_result queued{false, std::move(refusal)};
        if (queued.message.empty())
            queued = s.queue_view(surface, stream);
        note(queued.message);
        // The frame and the end in one step, so that the thread makes this view before it ends.
        const std::lock_guard<std::mutex> lock(s.mutex);
        if (queued.done)
            s.hand_over(frame, timestamp_us);
        s.ending = true;
    }
    s.wake.notify_one();
    if (s.thread.joinable())
        s.thread.join();
    // The thread has ended: the failure it left is read without the lock.
    if (!s.failure.empty())
        note("the live view's last picture: " + std::exchange(s.failure, std::string()));
    return {reasons.empty(), reasons};
}
} // namespace gca
