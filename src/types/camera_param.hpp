#pragma once

#include <cmath>

namespace gca
{
/// A pinhole camera's intrinsics, the one plain camera struct. The one pinhole convention: the
/// camera frame has x right, y down, z forward, in m; pixel centres lie at whole numbers, so the
/// ray of pixel (u, v) is ((u - cx) / fx, (v - cy) / fy, 1). The prototype's point-cloud kernel
/// deviates from it: it deprojects pixel (u, v) at (u - 0.5, v - 0.5), half a pixel off
/// (AGENTS.md §6; geometry/cuda_point_cloud_factory.cu, depth_uv) [S].
struct intrinsics
{
    float fx; ///< focal length along x, in px, > 0
    float fy; ///< focal length along y, in px, > 0
    float cx; ///< principal point's column, in px, pixel centres at whole numbers
    float cy; ///< principal point's row, in px, pixel centres at whole numbers
};

/// Says whether intrinsics describe a pinhole camera by the one convention: the one check of
/// every stage that takes them (geometry/organized_points.hpp, registration/registration.hpp).
/// @param k [in] the intrinsics, in px.
/// @return whether fx and fy are finite and > 0 and cx and cy finite; false for a NaN.
/// @note Host only; allocates nothing.
inline bool usable_intrinsics(const intrinsics &k)
{
    return std::isfinite(k.fx) && k.fx > 0.0f && std::isfinite(k.fy) && k.fy > 0.0f &&
           std::isfinite(k.cx) && std::isfinite(k.cy);
}

struct extrinsics
{

    float rotation[9];
    float translation[3];
};

enum frame_format
{
    BGR8,
    RGB8,
    Z16 // For depth
};
} // namespace gca
