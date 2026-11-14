// SPDX-License-Identifier: 0BSD
/// @file checkerboard_measure.hpp
/// distortion_check's measurement: whether an aligned color and depth image of a flat
/// checkerboard lies on a pinhole pixel grid. It finds the board's inner corners in the color
/// image and measures, in px, how far they lie from the homography fitted to them (a pinhole
/// picture of a plane is one), and, in mm, how far the corners deprojected with the depth and
/// the intrinsics lie from one plane and from the board's square size. Each is summed up for
/// the middle and the four corners of the image (a 3x3 grid of equal cells) and for all corners.
/// Host only, without the camera and the GPU, so that the offline tests drive it.
///
/// Pixel convention: pixel (u, v) has its centre at the whole numbers u, v, and its ray is
/// ((u - cx) / fx, (v - cy) / fy, 1); the camera frame is x right, y down, z forward. OrbbecSDK's
/// own code uses its intrinsics so (docs/HARDWARE.md, "What the camera is and reports").

#pragma once

#include "types/camera_param.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace gca
{
/// The depth window around a corner reaches this far from the pixel nearest the corner, so it
/// is 9x9 px [?: a choice].
constexpr int depth_window_radius_px = 4;

/// The valid pixels a depth window needs for its corner to get a depth: three quarters of 81,
/// rounded up [?: a choice].
constexpr int depth_window_min_valid = 61;

/// The corners with depth a plane is fitted to: at least this many [S: 3 points always lie on
/// a plane, so the fourth is the first that can deviate from it] ...
constexpr int plane_min_corners = 4;

/// ... and not on one line: the second largest eigenvalue of their centred covariance at least
/// this share of the largest, a spread across their main line of at least a tenth of the spread
/// along it [?: a choice].
constexpr double plane_min_spread_ratio = 0.01;

/// The printed checkerboard.
struct checkerboard
{
    int columns;      ///< inner corners along a row of squares, at least 3
    int rows;         ///< inner corners along a column of squares, at least 3
    double square_mm; ///< edge of one square as measured on the print, in mm, finite and > 0
};

/// The spread of one kind of deviation over the corners or edges of a region.
struct deviation_stats
{
    int count = 0;    ///< how many values; rms and max are 0 when it is 0
    double rms = 0.0; ///< root mean square
    double max = 0.0; ///< largest absolute value
};

/// What was measured in one region of the image.
struct region_measurement
{
    std::string name; ///< "middle", "top left", "top right", "bottom left", "bottom right", "all"
    /// Per corner: distance from the homography's picture of it, in px.
    deviation_stats homography_px;
    /// Per corner with depth: distance from the plane fitted to all corners with depth, in mm.
    deviation_stats plane_mm;
    /// Per edge (two neighbouring corners on the board's grid, both with depth): its length minus
    /// the square size, in mm.
    deviation_stats square_mm;
    /// The edges' signed mean of length minus square size, in mm; 0 without edges.
    double square_mean_mm = 0.0;
};

/// One inner corner of the board, found and measured.
struct board_corner
{
    int column;           ///< its column on the board's grid, 0 to columns - 1
    int row;              ///< its row on the board's grid, 0 to rows - 1
    double u_px;          ///< where it lies in the image, px, pixel centres at whole numbers
    double v_px;          ///< likewise, the row coordinate, px
    double homography_px; ///< distance from the homography's picture of it, px (>= 0)
    /// The corner deprojected with its depth, in mm, camera frame (x right, y down, z forward);
    /// nothing when fewer than depth_window_min_valid pixels of its depth window are valid.
    std::optional<std::array<double, 3>> point_mm;
};

/// What measure_board() found.
struct board_measurement
{
    /// Every inner corner, row by row in the order of the corner finder, which may start at
    /// either end of the board; each grid place once.
    std::vector<board_corner> corners;
    int corners_with_depth = 0; ///< corners with a point_mm
    /// Why the mm part (plane_mm, square_mm, square_mean_mm) is empty: the corners with depth
    /// are fewer than plane_min_corners or lie on one line; empty when it is measured.
    std::string depth_refusal;
    /// middle, top left, top right, bottom left, bottom right, all, in this order. A corner
    /// belongs to the cell of the image's 3x3 grid of equal cells its position lies in, an edge
    /// to the cell of its midpoint.
    std::vector<region_measurement> regions;
};

/// A measurement, or the reason there is none.
struct board_result
{
    std::optional<board_measurement> measurement; ///< nothing when refused
    std::string refusal;                          ///< why it was refused; empty when measured
};

/// Measures a flat checkerboard in an aligned color and depth image: finds its inner corners
/// (OpenCV's findChessboardCornersSB), fits a homography from the board's grid to them
/// (homography_deviations()), takes each corner's depth from a plane fitted to the valid depth
/// pixels of the window around it (depth_window_radius_px, depth_window_min_valid), deprojects
/// it with the intrinsics, fits one plane to the corners with depth (centred covariance in
/// double, its smallest eigenvector; plane_min_corners, plane_min_spread_ratio), and sums the
/// deviations up by region.
/// @param color [in] the color image, 8 bits in each of 3 channels (BGR, as orbbec_viewer
///     saves it), at least 3x3 px; the caller owns it.
/// @param depth [in] the depth image aligned to it, 16 bits in one channel, raw values in units
///     of depth_scale_m, 0 where invalid; the same size as color; the caller owns it.
/// @param board [in] the board's inner corners and square size.
/// @param camera [in] the intrinsics of the image, px; fx and fy > 0, and the principal point
///     within the middle third of the image in each direction.
/// @param depth_scale_m [in] meters per raw depth value, finite and > 0 (orbbec_viewer prints it
///     as "depth scale=... m/unit"; 0.001 on this camera [M 2026-11-13: docs/HARDWARE.md]).
/// @return the measurement, or the refusal with its reason: a color image not of 8 bits in 3
///     channels or under 3x3 px, a depth image not of 16 bits in one channel, images of two
///     sizes, a board under 3x3 inner corners or without a positive square size, intrinsics that
///     are not finite or a focal length that is not positive, a principal point outside the
///     image's middle third (intrinsics of another stream size), a depth scale that is not
///     positive, no board of that size found, a homography that cannot be fitted.
board_result measure_board(const cv::Mat &color, const cv::Mat &depth, const checkerboard &board,
                           const intrinsics &camera, double depth_scale_m);

/// Fits a homography from the board's grid to the corners' image positions by least squares
/// over all of them (OpenCV's findHomography, method 0) and measures each corner's distance from
/// the homography's picture of its grid place.
/// @param grid [in] each corner's place on the board's grid (column, row), in squares; at least
///     4, not all on one line.
/// @param image_px [in] each corner's image position, px, in the order of grid.
/// @return the distance of each corner, px, in the order of grid; nothing when the sizes differ,
///     there are fewer than 4 corners or no homography can be fitted.
std::optional<std::vector<double>> homography_deviations(const std::vector<cv::Point2d> &grid,
                                                         const std::vector<cv::Point2d> &image_px);
} // namespace gca
