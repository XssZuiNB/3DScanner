// SPDX-License-Identifier: 0BSD
/// @file viewer_loop.cpp
/// orbbec_viewer's loop on a frame source: the first frame, the prototype's GPU code, the report
/// line, the window, the snapshots and the sequence (see viewer_loop.hpp).

#include "viewer_loop.hpp"

#include "camera/frame.hpp"
#include "camera/orbbec_rules.hpp"
#include "camera/sequence_format.hpp"
#include "geometry/point_cloud.hpp"
#include "gpu/cuda_container.hpp"
#include "types/camera_param.hpp"
#include "types/console_color.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
constexpr float min_depth_m = 0.15f; ///< [?: a choice] points nearer than this are dropped
constexpr float max_depth_m = 3.0f;  ///< [?: a choice] points farther than this are dropped

/// Timings and counts accumulated between two reports.
struct frame_stats
{
    /// When these timings started: the loop's start or the last report.
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    int frames = 0;       ///< the frames delivered since
    double grab_ms = 0.0; ///< their grab times summed: next() and the conversion to BGR8, in ms
    double gpu_ms = 0.0;  ///< their GPU times summed: upload, point cloud and the wait, in ms
    int queue_full = 0;   ///< the pushes since that found the sequence writer's queue full
};

/// The current frame, and its color as the prototype's GPU code, OpenCV's window and its PNG file
/// take it, and the frame the next one is taken into.
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

/// The GPU side of the loop, set up once from the first frame: its parameters and the buffers
/// each frame is uploaded into.
struct gpu_buffers
{
    gca::cuda_camera_param param; ///< the frames' parameters (gpu_camera_param())
    gca::cuda_color_frame color;  ///< the current color picture, BGR8
    gca::cuda_depth_frame depth;  ///< the current aligned depth picture
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

/// Writes points as a binary little-endian PLY with float xyz and uchar rgb.
/// @param path [in] the file to write; an existing file is overwritten.
/// @param pts [in] the points: coordinates in m in the camera frame of the frame they were built
///        from (x right, y down, z forward), colors in [0, 1], each written as uint8(255 c),
///        truncated.
/// @return true when the file was opened; a write that fails after that is not reported.
bool save_ply(const std::string &path, const std::vector<gca::point_t> &pts)
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        return false;
    f << "ply\nformat binary_little_endian 1.0\nelement vertex " << pts.size()
      << "\nproperty float x\nproperty float y\nproperty float z\n"
         "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (const auto &p : pts)
    {
        f.write(reinterpret_cast<const char *>(&p.coordinates), sizeof(float) * 3);
        const uint8_t rgb[3] = {uint8_t(p.color.r * 255.f), uint8_t(p.color.g * 255.f),
                                uint8_t(p.color.b * 255.f)};
        f.write(reinterpret_cast<const char *>(rgb), 3);
    }
    return true;
}

/// Saves the point cloud as cloud_NNN.ply and the frame's pictures as cloud_NNN.ply.color.png and
/// cloud_NNN.ply.depth.png in the current directory.
/// @param current [in] the current frame and its BGR8 color.
/// @param pc [in] the point cloud built from it.
/// @param index [in] NNN.
/// @param out [in,out] where "Saved ..." is printed.
void save_snapshot(pictures &current, const gca::point_cloud &pc, int index, std::ostream &out)
{
    char name[64];
    std::snprintf(name, sizeof(name), "cloud_%03d.ply", index);
    const auto pts = pc.download();
    if (save_ply(name, pts))
        out << GREEN << "Saved " << pts.size() << " points to " << name << RESET << std::endl;
    cv::imwrite(std::string(name) + ".color.png", current.color_bgr);
    cv::imwrite(std::string(name) + ".depth.png", depth_picture(current.frame));
}

/// Shows color | depth (JET, 0 to max_depth_m, black where invalid) | 50 % overlay.
/// @param current [in] the current frame and its BGR8 color.
/// @return the key pressed, -1 for none.
int show_frames(pictures &current)
{
    const cv::Mat &color = current.color_bgr;
    const cv::Mat depth = depth_picture(current.frame);
    cv::Mat depth_vis;
    depth.convertTo(depth_vis, CV_8UC1,
                    255.0 / (max_depth_m / current.frame.description().depth_unit_m));
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

/// Copies the frames' parameters to the GPU. Each argument carries the name of the constructor
/// parameter it fills: three pairs have one type each (the two intrinsics, width and height, the
/// two formats), so the compiler does not catch a swap, and as a frame's depth is aligned to its
/// color the two intrinsics are equal and the extrinsics the identity, so no frame does either.
/// @param description [in] the frames' description.
/// @return the parameters on the GPU.
/// @note The prototype's path, until retire-the-prototype-library: allocates device memory
///       (cudaMalloc, three times) and copies into it with blocking cudaMemcpy() on the default
///       stream [S: cuda_camera_param_container.cu].
gca::cuda_camera_param gpu_camera_param(const gca::frame_description &description)
{
    gca::extrinsics identity{};
    identity.rotation[0] = identity.rotation[4] = identity.rotation[8] = 1.0f;
    return gca::cuda_camera_param(/*depth_intrin=*/description.intrinsics,
                                  /*color_intrin=*/description.intrinsics,
                                  /*depth2color_extrin=*/identity,
                                  /*width=*/description.size.width,
                                  /*height=*/description.size.height,
                                  /*depth_scale=*/description.depth_unit_m,
                                  /*depth_format=*/gca::Z16,
                                  /*color_format=*/gca::BGR8);
}

/// Sets up the GPU side of the loop for the frames of one opening: their parameters and frame
/// buffers of their size.
/// @param description [in] the frames' description.
/// @return the buffers.
/// @note The prototype's path, until retire-the-prototype-library: allocates device memory for
///       the parameters (gpu_camera_param()) and for both pictures (thrust::device_vector), on the
///       default stream [S: cuda_frame.cuh].
gpu_buffers make_gpu_buffers(const gca::frame_description &description)
{
    const gca::frame_size size = description.size;
    return {gpu_camera_param(description), {size.width, size.height}, {size.width, size.height}};
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

/// Uploads the current frame and builds its point cloud on the GPU, and adds the GPU time and the
/// frame to the statistics.
/// @param current [in] the current frame and its BGR8 color.
/// @param gpu [in,out] the GPU buffers, which gain the frame.
/// @param stats [in,out] gains the GPU time and the frame.
/// @return the frame's point cloud.
/// @note The prototype's path, until retire-the-prototype-library: both uploads are blocking
///       copies from pageable host memory (thrust::copy), create_from_rgbd() allocates the point
///       cloud on the GPU for every frame and waits for the device inside, and
///       cudaDeviceSynchronize() then waits for the whole device, all on the default stream
///       [S: cuda_frame.cuh, cuda_point_cloud_factory.cu, point_cloud.cu].
std::shared_ptr<gca::point_cloud> build_point_cloud(pictures &current, gpu_buffers &gpu,
                                                    frame_stats &stats)
{
    const auto t0 = std::chrono::steady_clock::now();
    const gca::frame_size size = current.frame.description().size;
    gpu.color.upload(current.color_bgr.data, size.width, size.height);
    gpu.depth.upload(current.frame.depth().data(), size.width, size.height);
    auto pc = gca::point_cloud::create_from_rgbd(gpu.depth, gpu.color, gpu.param, min_depth_m,
                                                 max_depth_m);
    cudaDeviceSynchronize();
    stats.gpu_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    stats.frames++;
    return pc;
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

/// Grabs, builds and shows frames until quit, the exit flag, the frame limit, the source's end or
/// a failed push to the sequence; 's', or the last frame of a headless run with a frame limit,
/// saves a snapshot. Each frame taken goes to the sequence once its successor has come, the last
/// one after the loop.
/// @param source [in,out] the open source.
/// @param current [in,out] the first frame, set up from, then each delivered frame in turn; the
///        spare frame of the first frame's description.
/// @param gpu [in,out] the GPU buffers.
/// @param sequence [in,out] the sequence, open, or none; gains every frame taken.
/// @param opt [in] the options.
/// @param context [in] the report period, the exit flag and where the lines go.
void run_loop(gca::frame_source &source, pictures &current, gpu_buffers &gpu,
              sequence_state &sequence, const options &opt, const run_context &context)
{
    frame_stats stats;
    int frame = 0;
    int saved = 0;
    std::shared_ptr<gca::point_cloud> cloud; // the last delivered frame's; none before the first

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
            cloud = build_point_cloud(current, gpu, stats);
            frame++;
            report_if_due(stats, cloud->points_number(), sequence.sink != nullptr,
                          context.report_period, context.out);
        }

        const int key = opt.headless ? -1 : read_key(current, what);
        if (key == 'q' || key == 27)
            break;
        // The loop ends at the frame limit, so only a delivered frame can be the last one.
        const bool last_headless_frame = opt.headless && frame == opt.max_frames;
        if ((key == 's' && cloud) || last_headless_frame)
            save_snapshot(current, *cloud, saved++, context.out);
    }
    push_frame(sequence, current.frame, stats);
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
    gpu_buffers gpu = make_gpu_buffers(description);
    run_loop(source, current, gpu, sequence, opt, context);
    return sequence.sink == nullptr || close_sequence(sequence, context.out);
}
} // namespace gca_viewer
