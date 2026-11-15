// SPDX-License-Identifier: 0BSD
/// @file viewer_loop.cpp
/// orbbec_viewer's loop on a frame source: the first frame, the GPU's organized points and
/// normals, the report line, the window, the snapshots and the sequence (see viewer_loop.hpp).

#include "viewer_loop.hpp"

#include "camera/frame.hpp"
#include "camera/orbbec_rules.hpp"
#include "camera/sequence_format.hpp"
#include "geometry/organized_normals.hpp"
#include "geometry/organized_points.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_image.hpp"
#include "gpu/gpu_messages.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/pinned_buffer.hpp"
#include "gpu/step_result.hpp"
#include "scan/model_file.hpp"
#include "types/camera_param.hpp"
#include "types/console_color.hpp"
#include "types/frame_description.hpp"
#include "types/model_point.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace gca_viewer
{
static_assert(gca::orbbec_frame_time_limit_ms == 5000,
              "the timeout line \"No frames received from Orbbec camera for 5s!\" names the "
              "Orbbec camera's time limit; AGENTS.md §3 reads the line");

namespace
{
/// The depth range of the organized points: a pixel nearer than 0.15 m or farther than 3.0 m along
/// the camera's z is invalid [?: a choice, the viewer's range since CUDA_3D, AGENTS.md §5; the
/// camera measures from 0.25 m to 10 m, S: docs/HARDWARE.md, "What the camera is and reports"].
/// The window's depth picture spans 0 to its far limit.
constexpr gca::organized_points_parameters depth_range = {0.15f, 3.0f};

/// The normals of a snapshot: a 7x7 window, 25 points, 75 degrees, the method's starting values
/// [?: initial; docs/methods/normals.md, "Input"].
constexpr gca::organized_normals_parameters snapshot_normals = {3, 25, 75.0f};

/// Timings and counts accumulated between two reports.
struct frame_stats
{
    /// When these timings started: the loop's start or the last report.
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    int frames = 0;       ///< the frames delivered since
    double grab_ms = 0.0; ///< their grab times summed: next() and the conversion to BGR8, in ms
    /// Their GPU times summed: the upload, the organized points, the mask's copy, the wait and the
    /// count, in ms.
    double gpu_ms = 0.0;
    int queue_full = 0; ///< the pushes since that found the sequence writer's queue full
};

/// The current frame, and its color as OpenCV's window and its PNG file take it, and the frame the
/// next one is taken into.
struct pictures
{
    gca::frame frame;  ///< the current frame, RGB8: the last delivered, shown and saved
    gca::frame spare;  ///< the frame the source fills; swapped with frame when one is delivered
    cv::Mat color_bgr; ///< the current frame's color picture as BGR8, converted once per frame
};

/// The sequence the loop writes, and what its pushes gave.
struct sequence_state
{
    sequence_sink *sink = nullptr; ///< where the frames go, open; nullptr for no sequence
    std::uint64_t taken = 0;       ///< the frames taken from the source, the first included
    std::uint64_t accepted = 0;    ///< the frames the sink took
    std::uint64_t full = 0;        ///< the frames it did not take, its queue being full
    std::string failure;           ///< the reason of the push that failed; empty while none has
};

/// The GPU side of the loop, set up once from the first frame (set_up_gpu()): a CUDA stream of its
/// own, and the buffers each frame goes through, requested before the loop from a pool that the
/// stages never see, so that the loop allocates nothing on the GPU. Not copied or moved; one
/// thread uses it.
struct gpu_path
{
    /// The stream every step of the loop runs on, not blocking with the default stream; nullptr
    /// until set_up_gpu() creates it.
    cudaStream_t stream = nullptr;
    gca::gpu_frame frame;           ///< the current frame, uploaded
    gca::organized_points points;   ///< its organized points
    gca::organized_normals normals; ///< their normals, estimated for a snapshot only
    gca::pinned_buffer mask;        ///< the current frame's validity mask, read back

    /// Builds a path with no stream and empty buffers.
    gpu_path() = default;

    /// A path is not copied or moved: it owns its stream.
    gpu_path(const gpu_path &) = delete;
    gpu_path &operator=(const gpu_path &) = delete;

    /// Waits for the stream's work, so that no buffer is freed under it, and destroys the stream;
    /// the buffers are freed after.
    /// @note Waits for the stream (cudaStreamSynchronize); a failure of either call is not
    ///       reported, its error left as CUDA's last error.
    ~gpu_path()
    {
        if (stream == nullptr)
            return;
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
};

/// A frame's organized points on the host, the snapshot's copy of the GPU's, each image in the
/// frame's pixel order.
struct host_points
{
    std::vector<float3> points;  ///< the points, in m, in the frame's camera frame
    std::vector<float3> normals; ///< their normals, (0, 0, 0) for none
};

/// What one call of next() gave the loop.
enum class taken
{
    frame, ///< a frame, in the spare frame
    none,  ///< a timeout or a failure, its line printed; the loop goes on
    end,   ///< the source's end, its message printed; the loop ends
};

/// Wraps a frame's depth picture as an OpenCV image without a copy.
/// @param frame [in] the frame; it must outlive the image and keep its size.
/// @return the depth picture, CV_16UC1, sharing the frame's buffer.
cv::Mat depth_picture(gca::frame &frame)
{
    const gca::frame_size size = frame.description().size;
    return cv::Mat(static_cast<int>(size.height), static_cast<int>(size.width), CV_16UC1,
                   frame.depth().data());
}

/// Converts the current frame's color picture from RGB8 to BGR8, into the picture that stays,
/// timed as part of the grab.
/// @param current [in,out] the frame, read; color_bgr gains its color, allocated only when its
///        size changes.
/// @param stats [in,out] gains the conversion's time as grab time.
void convert_color(pictures &current, frame_stats &stats)
{
    const auto t0 = std::chrono::steady_clock::now();
    const gca::frame_size size = current.frame.description().size;
    const cv::Mat rgb(static_cast<int>(size.height), static_cast<int>(size.width), CV_8UC3,
                      current.frame.color().data());
    cv::cvtColor(rgb, current.color_bgr, cv::COLOR_RGB2BGR);
    stats.grab_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

/// Estimates the current frame's normals and copies its points and normals to the host.
/// @param gpu [in,out] the GPU path, its organized points holding the current frame's; its normals
///        gain theirs.
/// @param copied [out] the points and the normals, each resized to the frame's pixels.
/// @return done; or not done with the reason: the normals' refusal or CUDA's error, which is then
///         cleared.
/// @note Allocates the host copies, 24 bytes a pixel. Queues the normals' kernel and two copies
///       into pageable memory on the stream and waits for it (cudaStreamSynchronize).
gca::step_result copy_snapshot(gpu_path &gpu, host_points &copied)
{
    const gca::step_result estimated =
        gpu.normals.estimate(gpu.points, snapshot_normals, gpu.stream);
    if (!estimated.done)
        return estimated;
    const gca::gpu_image<float3> &points = gpu.points.points();
    const gca::gpu_image<float3> &normals = gpu.normals.normals();
    copied.points.resize(points.size_bytes() / sizeof(float3));
    copied.normals.resize(normals.size_bytes() / sizeof(float3));
    const cudaError_t points_error =
        cudaMemcpyAsync(copied.points.data(), points.data(), points.size_bytes(),
                        cudaMemcpyDeviceToHost, gpu.stream);
    if (points_error != cudaSuccess)
        return {false, gca::cuda_failure("CUDA refused the snapshot's points' copy", points_error)};
    const cudaError_t normals_error =
        cudaMemcpyAsync(copied.normals.data(), normals.data(), normals.size_bytes(),
                        cudaMemcpyDeviceToHost, gpu.stream);
    if (normals_error != cudaSuccess)
        return {false,
                gca::cuda_failure("CUDA refused the snapshot's normals' copy", normals_error)};
    const cudaError_t wait_error = cudaStreamSynchronize(gpu.stream);
    if (wait_error != cudaSuccess)
        return {false, gca::cuda_failure("CUDA failed while the stream ran to the snapshot's copy",
                                         wait_error)};
    return {true, {}};
}

/// Makes the snapshot's model: each valid pixel's point that has a normal, with its normal and its
/// pixel's color, in the frame's pixel order.
/// @param current [in] the current frame, the points' frame.
/// @param mask [in] its validity mask, 1 for a valid pixel, one byte for each pixel.
/// @param copied [in] its points and normals.
/// @return the model's points, in the frame's camera frame, in m.
/// @note Allocates the points.
std::vector<gca::model_point> snapshot_model(const pictures &current,
                                             std::span<const std::byte> mask,
                                             const host_points &copied)
{
    std::vector<gca::model_point> model;
    const auto color = current.frame.color();
    for (std::size_t k = 0; k < copied.points.size(); k++)
    {
        const float3 p = copied.points[k];
        const float3 n = copied.normals[k];
        // (0, 0, 0) marks a pixel without a normal (docs/methods/normals.md, "Output").
        if (mask[k] == std::byte{0} || (n.x == 0.0f && n.y == 0.0f && n.z == 0.0f))
            continue;
        model.push_back({Eigen::Vector3f(p.x, p.y, p.z),
                         Eigen::Vector3f(n.x, n.y, n.z),
                         {color[3 * k], color[3 * k + 1], color[3 * k + 2]}});
    }
    return model;
}

/// Saves the current frame's snapshot in the current directory: its normals estimated, then its
/// valid points that have a normal as a model, cloud_NNN.ply (snapshot_model(),
/// gca::write_model()), and its pictures as cloud_NNN.ply.color.png and cloud_NNN.ply.depth.png. A
/// model the writer refuses or cannot write prints its reason and the pictures are written all the
/// same.
/// @param current [in] the current frame and its BGR8 color.
/// @param gpu [in,out] the GPU path, holding the current frame's organized points and validity
///        mask; its normals gain the frame's.
/// @param index [in] NNN.
/// @param out [in,out] where "Saved <count> points to cloud_NNN.ply" or the writer's reason is
///        printed.
/// @return done when the GPU's steps ran, whether the files were written or not; else the GPU's
///         reason, with nothing written.
gca::step_result save_snapshot(pictures &current, gpu_path &gpu, int index, std::ostream &out)
{
    host_points copied;
    const gca::step_result copy = copy_snapshot(gpu, copied);
    if (!copy.done)
        return copy;
    char name[64];
    std::snprintf(name, sizeof(name), "cloud_%03d.ply", index);
    const std::vector<gca::model_point> model = snapshot_model(current, gpu.mask.bytes(), copied);
    const gca::write_result written = gca::write_model(name, model);
    if (written.written)
        out << GREEN << "Saved " << model.size() << " points to " << name << RESET << std::endl;
    else
        out << RED << written.reason << RESET << std::endl;
    cv::imwrite(std::string(name) + ".color.png", current.color_bgr);
    cv::imwrite(std::string(name) + ".depth.png", depth_picture(current.frame));
    return {true, {}};
}

/// Shows color | depth (JET, 0 to the depth range's far limit, black where invalid) | 50 % overlay.
/// @param current [in] the current frame and its BGR8 color.
/// @return the key pressed, -1 for none.
int show_frames(pictures &current)
{
    const cv::Mat &color = current.color_bgr;
    const cv::Mat depth = depth_picture(current.frame);
    cv::Mat depth_vis;
    depth.convertTo(depth_vis, CV_8UC1,
                    255.0 / (depth_range.far_m / current.frame.description().depth_unit_m));
    cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);
    depth_vis.setTo(cv::Scalar(0, 0, 0), depth == 0);
    cv::Mat overlay;
    cv::addWeighted(color, 0.5, depth_vis, 0.5, 0, overlay);
    cv::Mat row;
    cv::hconcat(std::vector<cv::Mat>{color, depth_vis, overlay}, row);
    cv::imshow("Orbbec: color | depth | overlay   (s=save PLY, q=quit)", row);
    return cv::waitKey(1);
}

/// Prints frame rate, mean grab and GPU times and the point count once the report period has
/// passed since the last report, with a sequence also the pushes that found its queue full, then
/// resets.
/// @param stats [in,out] the accumulated timings and counts; reset after a report.
/// @param points [in] the current point count.
/// @param writing [in] whether a sequence is written: only then the line ends in " | queue full
///        <count>", so that without one it is the line docs/HARDWARE.md reads, unchanged.
/// @param period [in] the least time between two reports.
/// @param out [in,out] where the report line is printed.
void report_if_due(frame_stats &stats, std::size_t points, bool writing,
                   std::chrono::steady_clock::duration period, std::ostream &out)
{
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.since < period)
        return;
    const double seconds = std::chrono::duration<double>(now - stats.since).count();
    out << "fps " << stats.frames / seconds << " | grab+decode " << stats.grab_ms / stats.frames
        << " ms | gpu point cloud " << stats.gpu_ms / stats.frames << " ms | points " << points;
    if (writing)
        out << " | queue full " << stats.queue_full;
    out << std::endl;
    stats = frame_stats{};
}

/// Takes an opening's first frame. Timeouts are waited through for first_frame_time_limit, as
/// the prototype did.
/// @param source [in,out] the open source.
/// @param first [out] the first frame, when it came.
/// @param out [in,out] where the reason is printed when it did not come.
/// @return true when the first frame came; otherwise false, after "Timeout while waiting for
///         first frames." or the source's message of a failure or its end.
bool take_first_frame(gca::frame_source &source, gca::frame &first, std::ostream &out)
{
    const auto deadline = std::chrono::steady_clock::now() + first_frame_time_limit;
    for (;;)
    {
        const gca::next_result result = source.next(first);
        if (result.status == gca::next_status::frame)
            return true;
        if (result.status != gca::next_status::timeout)
        {
            out << RED << result.message << RESET << std::endl;
            return false;
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            out << RED << "Timeout while waiting for first frames." << RESET << std::endl;
            return false;
        }
    }
}

/// Prints the first frame's intrinsics and depth unit, the line distortion_check's values come
/// from (README.md, "Run").
/// @param description [in] the first frame's description.
/// @param out [in,out] where the line is printed.
void print_intrinsics(const gca::frame_description &description, std::ostream &out)
{
    const gca::intrinsics &in = description.intrinsics;
    out << "Intrinsics fx=" << in.fx << " fy=" << in.fy << " cx=" << in.cx << " cy=" << in.cy
        << "  depth scale=" << description.depth_unit_m << " m/unit" << std::endl;
}

/// Takes a granted buffer from the pool's answer, or prints why it was refused.
/// @param requested [in] the request's result; taken over.
/// @param into [out] the granted buffer; left as it was when refused.
/// @param out [in,out] where the pool's message is printed when refused.
/// @return whether the request was granted.
template <typename Buffer>
bool take_granted(gca::request_result<Buffer> requested, Buffer &into, std::ostream &out)
{
    if (requested.status != gca::request_status::granted)
    {
        out << RED << requested.message << RESET << std::endl;
        return false;
    }
    into = std::move(requested.granted);
    return true;
}

/// Sets up the GPU side of the loop for the frames of one opening: creates its stream and requests
/// the GPU frame, the organized points, the normals and the mask's pinned buffer of the frames'
/// size from a pool that lives only here, so that no step of the loop can ask for memory.
/// @param size [in] the frames' size, in px.
/// @param gpu [in,out] an empty path; gains the stream and the buffers, those set up before a
///        failure included, which it frees.
/// @param out [in,out] where CUDA's error or the pool's message is printed on a failure.
/// @return whether everything was set up; false after the reason: CUDA could not create the
///         stream, or the pool refused a request.
/// @note Allocates device memory (cudaMalloc) and page-locked host memory (cudaMallocHost), 36
///       bytes a pixel, and creates a stream and an event; whether these wait for the GPU is not
///       stated by CUDA [?].
bool set_up_gpu(const gca::frame_size &size, gpu_path &gpu, std::ostream &out)
{
    const cudaError_t created = cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking);
    if (created != cudaSuccess)
    {
        gpu.stream = nullptr;
        out << RED << gca::cuda_failure("CUDA could not create the viewer's stream", created)
            << RESET << std::endl;
        return false;
    }
    gca::gpu_pool pool;
    const std::size_t pixels = static_cast<std::size_t>(size.width) * size.height;
    return take_granted(gca::gpu_frame::request(pool, size), gpu.frame, out) &&
           take_granted(gca::organized_points::request(pool, size), gpu.points, out) &&
           take_granted(gca::organized_normals::request(pool, size), gpu.normals, out) &&
           take_granted(pool.request_pinned(pixels), gpu.mask, out);
}

/// Takes the next frame into the spare frame, timed as part of the grab; prints why when there is
/// no frame.
/// @param source [in,out] the open source.
/// @param into [out] the spare frame; filled when a frame came, else left as it was.
/// @param stats [in,out] gains the time of next() for a frame that came.
/// @param out [in,out] where a timeout, a failure or the end is printed.
/// @return what next() gave.
taken take_frame(gca::frame_source &source, gca::frame &into, frame_stats &stats, std::ostream &out)
{
    const auto t0 = std::chrono::steady_clock::now();
    const gca::next_result result = source.next(into);
    switch (result.status)
    {
    case gca::next_status::frame:
        break;
    case gca::next_status::timeout:
        out << RED << "No frames received from Orbbec camera for 5s!" << RESET << std::endl;
        return taken::none;
    case gca::next_status::failure:
        out << RED << result.message << RESET << std::endl;
        return taken::none;
    case gca::next_status::end:
        out << result.message << std::endl;
        return taken::end;
    }
    stats.grab_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return taken::frame;
}

/// Pushes a frame whose last use is over to the sequence, when there is one and no push has
/// failed, and counts what the push gave.
/// @param sequence [in,out] the sequence; gains the frame's count, or the push's reason.
/// @param done [in,out] the frame, shown and saved for the last time; swapped for a frame of the
///        same description, its pixels unspecified, when the sink took it; else as it was.
/// @param stats [in,out] gains a push that found the queue full.
/// @note Allocates nothing unless the push fails. Takes the writer's lock; waits for nothing
///       else.
void push_frame(sequence_state &sequence, gca::frame &done, frame_stats &stats)
{
    if (sequence.sink == nullptr || !sequence.failure.empty())
        return;
    gca::push_result pushed = sequence.sink->push(done);
    switch (pushed.status)
    {
    case gca::push_status::accepted:
        sequence.accepted++;
        break;
    case gca::push_status::full:
        sequence.full++;
        stats.queue_full++;
        break;
    case gca::push_status::failure:
        sequence.failure = std::move(pushed.reason);
        break;
    }
}

/// Closes the sequence and prints how it went, in one line: "Wrote <accepted> frames to <file>,
/// <full> not written because the writer's queue was full"; or, when the close or a push failed,
/// the reason of the close, else of the push (close() repeats a failed write's), then "; <written>
/// of <taken> frames taken written, <lost> lost from the writer's queue, <full> not written
/// because the writer's queue was full", so that no frame is lost unseen.
/// @param sequence [in,out] the open sequence; closed.
/// @param out [in,out] where the line is printed.
/// @return whether the sequence holds every frame the sink took and its end, synced to disk, and
///         no push failed.
bool close_sequence(sequence_state &sequence, std::ostream &out)
{
    const gca::write_result closed = sequence.sink->close();
    const std::string &reason = closed.written ? sequence.failure : closed.reason;
    if (reason.empty())
    {
        out << "Wrote " << sequence.accepted << " frames to " << sequence.sink->path().string()
            << ", " << sequence.full << " not written because the writer's queue was full"
            << std::endl;
        return true;
    }
    const std::uint64_t written = sequence.sink->frames_written();
    out << RED << reason << "; " << written << " of " << sequence.taken << " frames taken written, "
        << sequence.accepted - written << " lost from the writer's queue, " << sequence.full
        << " not written because the writer's queue was full" << RESET << std::endl;
    return false;
}

/// Copies the organized points' validity mask into the path's pinned buffer and waits for it.
/// @param gpu [in,out] the GPU path, its organized points made on its stream; its mask gains
///        theirs.
/// @return done; or not done with CUDA's error, which is then cleared.
/// @note Queues one copy of a byte a pixel on the stream and waits for the stream
///       (cudaStreamSynchronize), the loop's one wait of a frame. Allocates nothing when done.
gca::step_result read_back_mask(gpu_path &gpu)
{
    const gca::gpu_image<std::uint8_t> &mask = gpu.points.validity_mask();
    const cudaError_t copy_error =
        cudaMemcpyAsync(gpu.mask.bytes().data(), mask.data(), mask.size_bytes(),
                        cudaMemcpyDeviceToHost, gpu.stream);
    if (copy_error != cudaSuccess)
        return {false, gca::cuda_failure("CUDA refused the validity mask's copy", copy_error)};
    const cudaError_t wait_error = cudaStreamSynchronize(gpu.stream);
    if (wait_error != cudaSuccess)
        return {false, gca::cuda_failure("CUDA failed while the stream ran to the validity "
                                         "mask's copy",
                                         wait_error)};
    return {true, {}};
}

/// Uploads the current frame, makes its organized points within depth_range, reads their
/// validity mask back and counts the valid points, and adds the GPU time and the frame to the
/// statistics when every step was done.
/// @param current [in] the current frame.
/// @param gpu [in,out] the GPU path, which gains the frame, its points and its mask.
/// @param stats [in,out] gains the GPU time and the frame when done.
/// @param points [out] the frame's valid points when done; left as it was otherwise.
/// @return done; or not done with the reason of the first step that was not (gpu_frame::upload(),
///         organized_points::deproject(), read_back_mask()).
/// @note Allocates nothing on the GPU or the host when done. Waits for the stream once
///       (read_back_mask()), besides the upload's wait for its last copies (gpu/gpu_frame.hpp).
gca::step_result build_points(const pictures &current, gpu_path &gpu, frame_stats &stats,
                              std::size_t &points)
{
    const auto t0 = std::chrono::steady_clock::now();
    gca::step_result made = gpu.frame.upload(current.frame, gpu.stream);
    if (made.done)
        made = gpu.points.deproject(gpu.frame, depth_range, gpu.stream);
    if (made.done)
        made = read_back_mask(gpu);
    if (!made.done)
        return made;
    const std::span<const std::byte> mask = gpu.mask.bytes();
    points = static_cast<std::size_t>(
        std::count_if(mask.begin(), mask.end(), [](std::byte b) { return b != std::byte{0}; }));
    stats.gpu_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    stats.frames++;
    return made;
}

/// Reads the window's key after a turn of the loop: after a delivered frame the window shows it
/// first; after a timeout or a failure it keeps the last delivered frame and only its keys are
/// read, so that 'q', ESC and 's' keep working through a stall, as before the frame source seam.
/// @param current [in] the current frame and its BGR8 color: the last delivered frame.
/// @param what [in] what the turn's next() gave: a frame, or none.
/// @return the key pressed, -1 for none.
int read_key(pictures &current, taken what)
{
    return what == taken::frame ? show_frames(current) : cv::waitKey(1);
}

/// Grabs, builds and shows frames until quit, the exit flag, the frame limit, the source's end, a
/// failed push to the sequence or a failed GPU step, whose reason it prints; 's', or the last frame
/// of a headless run with a frame limit, saves a snapshot. Each frame taken goes to the sequence
/// once its successor has come, the last one after the loop.
/// @param source [in,out] the open source.
/// @param current [in,out] the first frame, set up from, then each delivered frame in turn; the
///        spare frame of the first frame's description.
/// @param gpu [in,out] the GPU path.
/// @param sequence [in,out] the sequence, open, or none; gains every frame taken.
/// @param opt [in] the options.
/// @param context [in] the report period, the exit flag and where the lines go.
/// @return true when no GPU step failed; false after the failed step's reason.
bool run_loop(gca::frame_source &source, pictures &current, gpu_path &gpu, sequence_state &sequence,
              const options &opt, const run_context &context)
{
    frame_stats stats;
    int frame = 0;
    int saved = 0;
    std::size_t points = 0; // the last delivered frame's valid points
    gca::step_result gpu_step{true, {}};

    while (!context.exit_requested.load() && (opt.max_frames < 0 || frame < opt.max_frames))
    {
        const taken what = take_frame(source, current.spare, stats, context.out);
        if (what == taken::end)
            break;
        if (what == taken::frame)
        {
            sequence.taken++;
            // The frame before has been shown and saved for the last time: it goes to the
            // sequence, and the new one takes its place.
            push_frame(sequence, current.frame, stats);
            if (!sequence.failure.empty())
                break;
            std::swap(current.frame, current.spare);
            convert_color(current, stats);
            gpu_step = build_points(current, gpu, stats, points);
            if (!gpu_step.done)
                break;
            frame++;
            report_if_due(stats, points, sequence.sink != nullptr, context.report_period,
                          context.out);
        }

        const int key = opt.headless ? -1 : read_key(current, what);
        if (key == 'q' || key == 27)
            break;
        // The loop ends at the frame limit, so only a delivered frame can be the last one.
        const bool last_headless_frame = opt.headless && frame == opt.max_frames;
        if ((key == 's' && frame > 0) || last_headless_frame)
        {
            gpu_step = save_snapshot(current, gpu, saved++, context.out);
            if (!gpu_step.done)
                break;
        }
    }
    push_frame(sequence, current.frame, stats);
    if (gpu_step.done)
        return true;
    context.out << RED << gpu_step.message << RESET << std::endl;
    return false;
}
} // namespace

sequence_file::sequence_file(const std::filesystem::path &directory)
    : m_path(directory / sequence_file_name)
{
}

gca::open_result sequence_file::open(const gca::frame_description &description) noexcept
{
    return m_writer.open(m_path, description, gca::sequence_queue_frames);
}

gca::push_result sequence_file::push(gca::frame &in_out) noexcept
{
    return m_writer.push(in_out, gca::not_tracked{});
}

gca::write_result sequence_file::close() noexcept
{
    return m_writer.close();
}

std::uint64_t sequence_file::frames_written() noexcept
{
    return m_writer.frames_written();
}

const std::filesystem::path &sequence_file::path() const noexcept
{
    return m_path;
}

bool run_viewer(gca::frame_source &source, const options &opt, const run_context &context)
{
    pictures current;
    if (!take_first_frame(source, current.frame, context.out))
    {
        context.out << RED << "Couldn't start stream!" << RESET << std::endl;
        return false;
    }
    const gca::frame_description description = current.frame.description();
    print_intrinsics(description, context.out);
    gpu_path gpu;
    if (!set_up_gpu(description.size, gpu, context.out))
    {
        context.out << RED << "Couldn't set up the GPU!" << RESET << std::endl;
        return false;
    }
    sequence_state sequence;
    if (context.sequence != nullptr)
    {
        const gca::open_result opened = context.sequence->open(description);
        if (!opened.opened)
        {
            context.out << RED << opened.refusal << RESET << std::endl;
            return false;
        }
        sequence.sink = context.sequence;
        sequence.taken = 1;
    }
    // Sized before the loop, so that its first frame allocates no frame.
    current.spare.prepare(description, 0, 0);
    const bool gpu_ran = run_loop(source, current, gpu, sequence, opt, context);
    const bool sequence_written = sequence.sink == nullptr || close_sequence(sequence, context.out);
    return gpu_ran && sequence_written;
}
} // namespace gca_viewer
