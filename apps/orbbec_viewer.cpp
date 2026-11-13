// SPDX-License-Identifier: 0BSD
/// @file orbbec_viewer.cpp
/// Live test of the Orbbec camera: grabs aligned RGB-D frames, builds the point cloud on the GPU,
/// reports frame rate and timings once a second and shows color | depth | overlay. 's' saves
/// the current point cloud as PLY plus the color and depth images, 'q' or ESC quits.
///
/// The options are viewer_options.hpp's, whose usage line lists them: --list prints the device's
/// stream profiles and exits. --frames N stops after N frames and, headless, saves the last one.
/// A command line with an argument the viewer does not know is refused with the usage line
/// before anything touches the camera, the GPU or a signal handler.

#include "camera/orbbec_device.hpp"
#include "geometry/point_cloud.hpp"
#include "gpu/cuda_container.hpp"
#include "gpu/gpu_check.hpp"
#include "types/console_color.hpp"
#include "viewer_options.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace
{
constexpr float min_depth_m = 0.15f; ///< [?: a choice] points nearer than this are dropped
constexpr float max_depth_m = 3.0f;  ///< [?: a choice] points farther than this are dropped

volatile std::sig_atomic_t g_exit_requested = 0;

/// Timings accumulated between two once-a-second reports.
struct frame_stats
{
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    int frames = 0;
    double grab_ms = 0.0;
    double gpu_ms = 0.0;
};

/// The GPU side of the loop, set up once from the started camera: its parameters and the
/// buffers each frame is uploaded into.
struct gpu_buffers
{
    gca::cuda_camera_param param; ///< the camera's parameters (gpu_camera_param())
    gca::cuda_color_frame color;  ///< the current color frame
    gca::cuda_depth_frame depth;  ///< the current aligned depth frame
};

/// Sets the flag the main loop polls; installed for SIGINT.
/// @param signal_number the signal, unused.
void on_sigint(int /*signal_number*/)
{
    g_exit_requested = 1;
}

/// Writes points as a binary little-endian PLY with float xyz and uchar rgb.
/// @param path the file to write.
/// @param pts the points; colors in [0, 1].
/// @return true when the file was opened.
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

/// Saves the point cloud as cloud_NNN.ply and the frames as cloud_NNN.ply.color.png and
/// cloud_NNN.ply.depth.png in the current directory.
/// @param cam the camera holding the current frames.
/// @param pc the point cloud built from them.
/// @param index NNN.
void save_snapshot(gca::orbbec_device &cam, const gca::point_cloud &pc, int index)
{
    char name[64];
    std::snprintf(name, sizeof(name), "cloud_%03d.ply", index);
    const auto pts = pc.download();
    if (save_ply(name, pts))
        std::cout << GREEN << "Saved " << pts.size() << " points to " << name << RESET << std::endl;
    cv::imwrite(std::string(name) + ".color.png", cam.get_color_cv_mat(false));
    cv::imwrite(std::string(name) + ".depth.png", cam.get_depth_cv_mat(false));
}

/// Shows color | depth (JET, 0 to max_depth_m, black where invalid) | 50 % overlay.
/// @param cam the camera holding the current frames.
/// @return the key pressed, -1 for none.
int show_frames(gca::orbbec_device &cam)
{
    const cv::Mat color = cam.get_color_cv_mat(false);
    const cv::Mat depth = cam.get_depth_cv_mat(false);
    cv::Mat depth_vis;
    depth.convertTo(depth_vis, CV_8UC1, 255.0 / (max_depth_m / cam.get_depth_scale()));
    cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);
    depth_vis.setTo(cv::Scalar(0, 0, 0), depth == 0);
    cv::Mat overlay;
    cv::addWeighted(color, 0.5, depth_vis, 0.5, 0, overlay);
    cv::Mat row;
    cv::hconcat(std::vector<cv::Mat>{color, depth_vis, overlay}, row);
    cv::imshow("Orbbec: color | depth | overlay   (s=save PLY, q=quit)", row);
    return cv::waitKey(1);
}

/// Prints frame rate, mean grab and GPU times and the point count once a second, then resets.
/// @param stats the accumulated timings.
/// @param points the current point count.
void report_if_due(frame_stats &stats, size_t points)
{
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - stats.since).count();
    if (seconds < 1.0)
        return;
    std::cout << "fps " << stats.frames / seconds << " | grab+decode "
              << stats.grab_ms / stats.frames << " ms | gpu point cloud "
              << stats.gpu_ms / stats.frames << " ms | points " << points << std::endl;
    stats = frame_stats{};
}

/// Prints the GPUs and warms up the first one.
void warm_up_gpu()
{
    cuda_print_devices();
    cuda_warm_up_gpu(0);
}

/// Starts the camera and prints its color intrinsics and depth scale.
/// @param cam the camera.
/// @return true when it started.
bool start_camera(gca::orbbec_device &cam)
{
    if (!cam.device_start())
        return false;
    const auto &in = cam.get_color_intrinsics();
    std::cout << "Intrinsics fx=" << in.fx << " fy=" << in.fy << " cx=" << in.cx << " cy=" << in.cy
              << "  depth scale=" << cam.get_depth_scale() << " m/unit" << std::endl;
    return true;
}

/// Copies the started camera's parameters to the GPU. Each argument carries the name of the
/// constructor parameter it fills: three pairs have one type each (the two intrinsics, width and
/// height, the two formats), so the compiler does not catch a swap, and on this camera the two
/// intrinsics are equal and the extrinsics the identity, so the hardware does not either.
/// @param cam the started camera.
/// @return the parameters on the GPU.
gca::cuda_camera_param gpu_camera_param(const gca::orbbec_device &cam)
{
    return gca::cuda_camera_param(/*depth_intrin=*/cam.get_depth_intrinsics(),
                                  /*color_intrin=*/cam.get_color_intrinsics(),
                                  /*depth2color_extrin=*/cam.get_depth_to_color_extrinsics(),
                                  /*width=*/cam.get_width(),
                                  /*height=*/cam.get_height(),
                                  /*depth_scale=*/cam.get_depth_scale(),
                                  /*depth_format=*/cam.get_depth_frame_format(),
                                  /*color_format=*/cam.get_color_frame_format());
}

/// Sets up the GPU side of the loop for the started camera: its parameters and frame buffers of
/// its size.
/// @param cam the started camera.
/// @return the buffers.
gpu_buffers make_gpu_buffers(const gca::orbbec_device &cam)
{
    return {gpu_camera_param(cam),
            {cam.get_width(), cam.get_height()},
            {cam.get_width(), cam.get_height()}};
}

/// Grabs the next frame, uploads it and builds its point cloud on the GPU, and adds the grab and
/// the GPU times to the statistics.
/// @param cam the started camera.
/// @param gpu the GPU buffers.
/// @param stats the timings, which gain this frame.
/// @return the frame's point cloud.
std::shared_ptr<gca::point_cloud> process_frame(gca::orbbec_device &cam, gpu_buffers &gpu,
                                                frame_stats &stats)
{
    const auto t0 = std::chrono::steady_clock::now();
    cam.receive_data();
    const auto t1 = std::chrono::steady_clock::now();
    gpu.color.upload(static_cast<const uint8_t *>(cam.get_color_raw_data()), cam.get_width(),
                     cam.get_height());
    gpu.depth.upload(static_cast<const uint16_t *>(cam.get_depth_raw_data()), cam.get_width(),
                     cam.get_height());
    auto pc = gca::point_cloud::create_from_rgbd(gpu.depth, gpu.color, gpu.param, min_depth_m,
                                                 max_depth_m);
    cudaDeviceSynchronize();
    const auto t2 = std::chrono::steady_clock::now();

    stats.grab_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.gpu_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    stats.frames++;
    return pc;
}

/// Grabs, builds and shows frames until quit, SIGINT or the frame limit; 's', or the last frame
/// of a headless run with a frame limit, saves a snapshot.
/// @param cam the started camera.
/// @param gpu the GPU buffers.
/// @param opt the options.
void run_loop(gca::orbbec_device &cam, gpu_buffers &gpu, const gca_viewer::options &opt)
{
    frame_stats stats;
    int frame = 0;
    int saved = 0;

    while (!g_exit_requested && (opt.max_frames < 0 || frame < opt.max_frames))
    {
        const auto pc = process_frame(cam, gpu, stats);
        frame++;
        report_if_due(stats, pc->points_number());

        const int key = opt.headless ? -1 : show_frames(cam);
        if (key == 'q' || key == 27)
            break;
        const bool last_headless_frame = opt.headless && frame == opt.max_frames;
        if (key == 's' || last_headless_frame)
            save_snapshot(cam, *pc, saved++);
    }
}
} // namespace

/// Runs the camera test. Reading the options comes first, before anything touches the camera,
/// the GPU or a signal handler; then the profiles are listed, or the GPU and the camera are set
/// up and the loop runs until quit, SIGINT or the frame limit.
/// @param argc the argument count.
/// @param argv the arguments (viewer_options.hpp).
/// @return 0 after the list or the loop, 1 when the camera does not start, 2 when the command
///     line is refused.
int main(int argc, char *argv[])
{
    const std::optional<gca_viewer::options> opt =
        gca_viewer::read_options(argc, argv, std::getenv("DISPLAY") == nullptr, std::cerr);
    if (!opt)
        return 2;
    if (opt->list_only)
    {
        gca::orbbec_device::print_stream_profiles();
        return 0;
    }
    std::signal(SIGINT, on_sigint);
    warm_up_gpu();
    gca::orbbec_device cam(opt->width, opt->height, opt->fps);
    if (!start_camera(cam))
        return 1;
    gpu_buffers gpu = make_gpu_buffers(cam);
    run_loop(cam, gpu, *opt);
    cam.device_stop();
    return 0;
}
