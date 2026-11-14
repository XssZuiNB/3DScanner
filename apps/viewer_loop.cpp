// SPDX-License-Identifier: 0BSD
/// @file viewer_loop.cpp
/// orbbec_viewer's loop on a frame source: the first frame, the prototype's GPU code, the report
/// line, the window and the snapshots (see viewer_loop.hpp).

#include "viewer_loop.hpp"

#include "camera/frame.hpp"
#include "camera/orbbec_rules.hpp"
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

/// Timings accumulated between two reports.
struct frame_stats
{
    /// When these timings started: the loop's start or the last report.
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    int frames = 0;       ///< the frames delivered since
    double grab_ms = 0.0; ///< their grab times summed: next() and the conversion to BGR8, in ms
    double gpu_ms = 0.0;  ///< their GPU times summed: upload, point cloud and the wait, in ms
};

/// The current frame, and its color as the prototype's GPU code, OpenCV's window and its PNG file
/// take it.
struct pictures
{
    gca::frame frame;  ///< the current frame, RGB8; the source fills it
    cv::Mat color_bgr; ///< its color picture as BGR8, converted once per frame
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
    frame, ///< a frame, converted
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

/// Converts the current frame's color picture from RGB8 to BGR8, into the picture that stays.
/// @param current [in,out] the frame, read; color_bgr gains its color, allocated only when its
///        size changes.
void convert_color(pictures &current)
{
    const gca::frame_size size = current.frame.description().size;
    const cv::Mat rgb(static_cast<int>(size.height), static_cast<int>(size.width), CV_8UC3,
                      current.frame.color().data());
    cv::cvtColor(rgb, current.color_bgr, cv::COLOR_RGB2BGR);
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
/// passed since the last report, then resets.
/// @param stats [in,out] the accumulated timings; reset after a report.
/// @param points [in] the current point count.
/// @param period [in] the least time between two reports.
/// @param out [in,out] where the report line is printed.
void report_if_due(frame_stats &stats, std::size_t points,
                   std::chrono::steady_clock::duration period, std::ostream &out)
{
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.since < period)
        return;
    const double seconds = std::chrono::duration<double>(now - stats.since).count();
    out << "fps " << stats.frames / seconds << " | grab+decode " << stats.grab_ms / stats.frames
        << " ms | gpu point cloud " << stats.gpu_ms / stats.frames << " ms | points " << points
        << std::endl;
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

/// Takes the next frame and converts its color to BGR8, timed as the grab; prints why when there
/// is no frame.
/// @param source [in,out] the open source.
/// @param current [in,out] gains the frame and its BGR8 color when one came; else left as it was.
/// @param stats [in,out] gains the grab time of a frame that came.
/// @param out [in,out] where a timeout, a failure or the end is printed.
/// @return what next() gave.
taken take_frame(gca::frame_source &source, pictures &current, frame_stats &stats,
                 std::ostream &out)
{
    const auto t0 = std::chrono::steady_clock::now();
    const gca::next_result result = source.next(current.frame);
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
    convert_color(current);
    stats.grab_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return taken::frame;
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

/// Grabs, builds and shows frames until quit, the exit flag, the frame limit or the source's end;
/// 's', or the last frame of a headless run with a frame limit, saves a snapshot.
/// @param source [in,out] the open source.
/// @param current [in,out] the current frame, reused for every frame.
/// @param gpu [in,out] the GPU buffers.
/// @param opt [in] the options.
/// @param context [in] the report period, the exit flag and where the lines go.
void run_loop(gca::frame_source &source, pictures &current, gpu_buffers &gpu, const options &opt,
              const run_context &context)
{
    frame_stats stats;
    int frame = 0;
    int saved = 0;
    std::shared_ptr<gca::point_cloud> cloud; // the last delivered frame's; none before the first

    while (!context.exit_requested.load() && (opt.max_frames < 0 || frame < opt.max_frames))
    {
        const taken what = take_frame(source, current, stats, context.out);
        if (what == taken::end)
            break;
        if (what == taken::frame)
        {
            cloud = build_point_cloud(current, gpu, stats);
            frame++;
            report_if_due(stats, cloud->points_number(), context.report_period, context.out);
        }

        const int key = opt.headless ? -1 : read_key(current, what);
        if (key == 'q' || key == 27)
            break;
        // The loop ends at the frame limit, so only a delivered frame can be the last one.
        const bool last_headless_frame = opt.headless && frame == opt.max_frames;
        if ((key == 's' && cloud) || last_headless_frame)
            save_snapshot(current, *cloud, saved++, context.out);
    }
}
} // namespace

bool run_viewer(gca::frame_source &source, const options &opt, const run_context &context)
{
    pictures current;
    if (!take_first_frame(source, current.frame, context.out))
    {
        context.out << RED << "Couldn't start stream!" << RESET << std::endl;
        return false;
    }
    print_intrinsics(current.frame.description(), context.out);
    gpu_buffers gpu = make_gpu_buffers(current.frame.description());
    run_loop(source, current, gpu, opt, context);
    return true;
}
} // namespace gca_viewer
