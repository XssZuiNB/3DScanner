// SPDX-License-Identifier: 0BSD
/// @file orbbec_viewer.cpp
/// Live test of the Orbbec camera: grabs aligned RGB-D frames, builds the point cloud on the GPU,
/// reports frame rate and timings once a second and shows color | depth | overlay. 's' saves
/// the current point cloud as PLY plus the color and depth images, 'q' or ESC quits.
///
/// Usage: orbbec_viewer [--list] [--headless] [--frames N] [--width W --height H --fps F]
/// --list prints the device's stream profiles and exits. --frames N stops after N frames and,
/// headless, saves the last one.

#include "camera/orbbec_device.hpp"
#include "geometry/point_cloud.hpp"
#include "gpu/cuda_container.hpp"
#include "gpu/gpu_check.hpp"
#include "types/console_color.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace
{
constexpr float min_depth_m = 0.15f; ///< [?: a choice] points nearer than this are dropped
constexpr float max_depth_m = 3.0f;  ///< [?: a choice] points farther than this are dropped

volatile std::sig_atomic_t g_exit_requested = 0;

struct options
{
    uint32_t width = 640;
    uint32_t height = 400; ///< [?: a choice] 640x480 color does not exist on USB 2 [M 2026-11-13]
    uint32_t fps = 30;
    int max_frames = -1; ///< -1: run until quit
    bool headless = false;
    bool list_only = false;
};

/// Timings accumulated between two once-a-second reports.
struct frame_stats
{
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    int frames = 0;
    double grab_ms = 0.0;
    double gpu_ms = 0.0;
};

/// Sets the flag the main loop polls; installed for SIGINT.
/// @param signal_number the signal, unused.
void on_sigint(int /*signal_number*/)
{
    g_exit_requested = 1;
}

/// Reads the command line.
/// @param argc argument count.
/// @param argv arguments.
/// @return the options; headless defaults to true when DISPLAY is unset.
options parse_options(int argc, char *argv[])
{
    options opt;
    opt.headless = (std::getenv("DISPLAY") == nullptr);
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        const bool has_value = (i + 1 < argc);
        if (a == "--list")
            opt.list_only = true;
        else if (a == "--headless")
            opt.headless = true;
        else if (a == "--frames" && has_value)
            opt.max_frames = std::atoi(argv[++i]);
        else if (a == "--width" && has_value)
            opt.width = std::atoi(argv[++i]);
        else if (a == "--height" && has_value)
            opt.height = std::atoi(argv[++i]);
        else if (a == "--fps" && has_value)
            opt.fps = std::atoi(argv[++i]);
    }
    return opt;
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

/// Grabs, builds and shows frames until quit, SIGINT or the frame limit.
/// @param cam the started camera.
/// @param opt the options.
void run_loop(gca::orbbec_device &cam, const options &opt)
{
    gca::cuda_camera_param cu_param(cam.get_depth_intrinsics(), cam.get_color_intrinsics(),
                                    cam.get_depth_to_color_extrinsics(), cam.get_width(),
                                    cam.get_height(), cam.get_depth_scale(),
                                    cam.get_depth_frame_format(), cam.get_color_frame_format());
    gca::cuda_color_frame gpu_color(cam.get_width(), cam.get_height());
    gca::cuda_depth_frame gpu_depth(cam.get_width(), cam.get_height());
    frame_stats stats;
    int frame = 0;
    int saved = 0;

    while (!g_exit_requested && (opt.max_frames < 0 || frame < opt.max_frames))
    {
        const auto t0 = std::chrono::steady_clock::now();
        cam.receive_data();
        const auto t1 = std::chrono::steady_clock::now();
        gpu_color.upload(static_cast<const uint8_t *>(cam.get_color_raw_data()), cam.get_width(),
                         cam.get_height());
        gpu_depth.upload(static_cast<const uint16_t *>(cam.get_depth_raw_data()), cam.get_width(),
                         cam.get_height());
        auto pc = gca::point_cloud::create_from_rgbd(gpu_depth, gpu_color, cu_param, min_depth_m,
                                                     max_depth_m);
        cudaDeviceSynchronize();
        const auto t2 = std::chrono::steady_clock::now();

        stats.grab_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats.gpu_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        stats.frames++;
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

int main(int argc, char *argv[])
{
    const options opt = parse_options(argc, argv);
    if (opt.list_only)
    {
        gca::orbbec_device::print_stream_profiles();
        return 0;
    }
    std::signal(SIGINT, on_sigint);

    cuda_print_devices();
    cuda_warm_up_gpu(0);

    gca::orbbec_device cam(opt.width, opt.height, opt.fps);
    if (!cam.device_start())
        return 1;
    const auto &in = cam.get_color_intrinsics();
    std::cout << "Intrinsics fx=" << in.fx << " fy=" << in.fy << " cx=" << in.cx << " cy=" << in.cy
              << "  depth scale=" << cam.get_depth_scale() << " m/unit" << std::endl;

    run_loop(cam, opt);
    cam.device_stop();
    return 0;
}
