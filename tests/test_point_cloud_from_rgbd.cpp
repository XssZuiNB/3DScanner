// SPDX-License-Identifier: 0BSD
/// @file test_point_cloud_from_rgbd.cpp
/// Offline test of gca::point_cloud::create_from_rgbd() with depth aligned to color (identity
/// extrinsics, shared intrinsics), the setup orbbec_device delivers: a synthetic flat plane
/// must give one point per pixel at the plane's depth, with the color converted from BGR8.
/// Needs the GPU, no camera.

#include "check.hpp"
#include "geometry/point_cloud.hpp"
#include "gpu/cuda_container.hpp"

#include <cmath>
#include <vector>

using gca_test::check;
using gca_test::check_summary;

namespace
{
constexpr uint32_t frame_width = 640;
constexpr uint32_t frame_height = 480;
constexpr float depth_scale_m = 0.001f; ///< meters per raw unit, the Orbbec default [S]
constexpr gca::intrinsics intrinsics = {500.0f, 500.0f, 319.5f, 239.5f};

/// Builds the point cloud of a uniform plane.
/// @param depth_raw the raw depth value of every pixel.
/// @param bgr the color of every pixel.
/// @return the downloaded points.
std::vector<gca::point_t> plane_cloud(uint16_t depth_raw, const uint8_t bgr[3])
{
    gca::extrinsics identity{};
    identity.rotation[0] = identity.rotation[4] = identity.rotation[8] = 1.0f;
    const gca::cuda_camera_param param(intrinsics, intrinsics, identity, frame_width, frame_height,
                                       depth_scale_m);

    const std::vector<uint16_t> depth(frame_width * frame_height, depth_raw);
    std::vector<uint8_t> color(frame_width * frame_height * 3);
    for (size_t i = 0; i < color.size(); i++)
        color[i] = bgr[i % 3];

    const gca::cuda_depth_frame gpu_depth(depth.data(), frame_width, frame_height);
    const gca::cuda_color_frame gpu_color(color.data(), frame_width, frame_height);
    return gca::point_cloud::create_from_rgbd(gpu_depth, gpu_color, param, 0.15f, 3.0f)->download();
}

/// A plane at 1 m inside the depth range: every pixel becomes a point at z = 1 m, x and y
/// spanning the pinhole projection of the image, colors converted from BGR.
void test_plane_in_range()
{
    const uint8_t bgr[3] = {10, 20, 30};
    const auto pts = plane_cloud(1000, bgr);
    check(pts.size() == frame_width * frame_height, "one point per pixel");

    float min_x = 1e9f, max_x = -1e9f, max_dz = 0.0f, max_dcolor = 0.0f;
    for (const auto &p : pts)
    {
        min_x = std::fmin(min_x, p.coordinates.x);
        max_x = std::fmax(max_x, p.coordinates.x);
        max_dz = std::fmax(max_dz, std::fabs(p.coordinates.z - 1.0f));
        max_dcolor = std::fmax(max_dcolor, std::fabs(p.color.r - 30.0f / 255.0f));
        max_dcolor = std::fmax(max_dcolor, std::fabs(p.color.b - 10.0f / 255.0f));
    }
    check(max_dz < 1e-5f, "z = 1 m everywhere");
    // Pixel u maps to x = (u - 0.5 - cx) / fx at z = 1 (the kernel's half-pixel convention).
    check(std::fabs(min_x - (-0.5f - intrinsics.cx) / intrinsics.fx) < 1e-5f, "min x");
    check(std::fabs(max_x - (frame_width - 1.5f - intrinsics.cx) / intrinsics.fx) < 1e-5f, "max x");
    check(max_dcolor < 1e-6f, "BGR8 converted to r, g, b");
}

/// A plane at 5 m, beyond the 3 m limit: no points.
void test_plane_out_of_range()
{
    const uint8_t bgr[3] = {0, 0, 0};
    check(plane_cloud(5000, bgr).empty(), "points beyond the depth limit are dropped");
}
} // namespace

int main()
{
    test_plane_in_range();
    test_plane_out_of_range();
    return check_summary("test_point_cloud_from_rgbd");
}
