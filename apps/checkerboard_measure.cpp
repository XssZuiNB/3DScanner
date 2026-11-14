// SPDX-License-Identifier: 0BSD
/// @file checkerboard_measure.cpp
/// distortion_check's measurement of a flat checkerboard in an aligned color and depth image:
/// the corner finder, the homography deviations, the depth at a corner, the deprojection, the
/// plane and the summary by region (checkerboard_measure.hpp).

#include "checkerboard_measure.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace gca
{
namespace
{
/// The regions the report gives, each with the cell of the 3x3 grid it covers (row-major, 0 top
/// left, 8 bottom right); -1 stands for every corner.
struct region_cell
{
    const char *name; ///< the region's name in the report
    int cell;         ///< its cell, or -1 for all
};

/// middle, the four corner cells, all, in the order of the report.
constexpr region_cell report_regions[] = {{"middle", 4},      {"top left", 0},     {"top right", 2},
                                          {"bottom left", 6}, {"bottom right", 8}, {"all", -1}};

/// Says what is wrong with measure_board()'s inputs, before any of them is used.
/// @param color [in] the color image.
/// @param depth [in] the depth image.
/// @param board [in] the board.
/// @param camera [in] the intrinsics, px.
/// @param depth_scale_m [in] meters per raw depth value.
/// @return the reason for the first input that is refused, or an empty string when every input
///     is accepted.
std::string refusal_of_inputs(const cv::Mat &color, const cv::Mat &depth, const checkerboard &board,
                              const intrinsics &camera, double depth_scale_m)
{
    std::ostringstream reason;
    if (color.type() != CV_8UC3)
        return "the color image is not an image of 8 bits in each of 3 channels";
    if (color.cols < 3 || color.rows < 3)
    {
        reason << "the color image is " << color.cols << "x" << color.rows << ", under 3x3";
        return reason.str();
    }
    if (depth.type() != CV_16UC1)
        return "the depth image is not an image of 16 bits in one channel";
    if (color.size() != depth.size())
    {
        reason << "the color image is " << color.cols << "x" << color.rows
               << " and the depth image " << depth.cols << "x" << depth.rows
               << "; aligned frames have one size";
        return reason.str();
    }
    if (board.columns < 3 || board.rows < 3)
        return "the board needs at least 3x3 inner corners";
    if (!std::isfinite(board.square_mm) || board.square_mm <= 0.0)
        return "the square size is not a positive number of mm";
    if (!std::isfinite(camera.fx) || !std::isfinite(camera.fy) || !std::isfinite(camera.cx) ||
        !std::isfinite(camera.cy) || camera.fx <= 0.0f || camera.fy <= 0.0f)
        return "the intrinsics are not finite with positive fx and fy";
    // A principal point lies near the image's centre; one outside the middle third belongs to
    // another stream size (cx of 1280x800 is 635.81, of 640x400 317.41 [M 2026-11-13,
    // docs/HARDWARE.md]).
    const double centre_u = (color.cols - 1) / 2.0;
    const double centre_v = (color.rows - 1) / 2.0;
    if (std::abs(camera.cx - centre_u) > color.cols / 6.0 ||
        std::abs(camera.cy - centre_v) > color.rows / 6.0)
    {
        reason << "the principal point (" << camera.cx << ", " << camera.cy
               << ") lies outside the middle third of the " << color.cols << "x" << color.rows
               << " image: intrinsics of another stream size";
        return reason.str();
    }
    if (!std::isfinite(depth_scale_m) || depth_scale_m <= 0.0)
        return "the depth scale is not a positive number of m per unit";
    return "";
}

/// Finds the board's inner corners in the color image with OpenCV's findChessboardCornersSB,
/// exhaustive and with its accuracy refinement. On synthetic pictures of a tilted board it
/// placed corners 0.06 px rms and 0.17 px at most from the true ones, findChessboardCorners
/// with cornerSubPix 0.09 px and 0.21 px [M 2026-11-14: a Python prototype with OpenCV 4.8.0,
/// 13x8 corners at 1280x800, 4x4 samples per pixel].
/// @param color [in] the color image, BGR with 8 bits in each channel.
/// @param board [in] the board, for its inner corners.
/// @return the corners, px, row by row as the finder orders them; nothing when no board of
///     that size is found.
std::optional<std::vector<cv::Point2f>> find_corners(const cv::Mat &color,
                                                     const checkerboard &board)
{
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> corners;
    const bool found =
        cv::findChessboardCornersSB(gray, cv::Size(board.columns, board.rows), corners,
                                    cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
    if (!found || corners.size() != static_cast<size_t>(board.columns) * board.rows)
        return std::nullopt;
    return corners;
}

/// Reads the depth at a corner: fits the plane z = a + b (x - u) + c (y - v) by least squares to
/// the valid pixels (x, y) of the depth window around the pixel nearest the corner and takes a,
/// its value at the corner itself.
/// @param depth [in] the depth image, 16 bits, 0 where invalid.
/// @param u_px [in] the corner's column coordinate, px, pixel centres at whole numbers.
/// @param v_px [in] the corner's row coordinate, px.
/// @param depth_scale_m [in] meters per raw depth value, > 0.
/// @return the depth at the corner, mm; nothing when the window leaves the image or has fewer
///     than depth_window_min_valid valid pixels.
std::optional<double> depth_at_corner(const cv::Mat &depth, double u_px, double v_px,
                                      double depth_scale_m)
{
    const int centre_x = static_cast<int>(std::lround(u_px));
    const int centre_y = static_cast<int>(std::lround(v_px));
    const int r = depth_window_radius_px;
    if (centre_x - r < 0 || centre_y - r < 0 || centre_x + r >= depth.cols ||
        centre_y + r >= depth.rows)
        return std::nullopt;
    const double mm_per_unit = depth_scale_m * 1000.0;
    Eigen::Matrix3d normal_matrix = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    int valid = 0;
    for (int y = centre_y - r; y <= centre_y + r; y++)
    {
        for (int x = centre_x - r; x <= centre_x + r; x++)
        {
            const uint16_t raw = depth.at<uint16_t>(y, x);
            if (raw == 0)
                continue;
            const Eigen::Vector3d row(1.0, x - u_px, y - v_px);
            normal_matrix += row * row.transpose();
            rhs += row * (raw * mm_per_unit);
            valid++;
        }
    }
    if (valid < depth_window_min_valid)
        return std::nullopt;
    return normal_matrix.ldlt().solve(rhs)(0);
}

/// Deprojects a pixel with its depth by the pinhole model.
/// @param u_px [in] the column coordinate, px, pixel centres at whole numbers.
/// @param v_px [in] the row coordinate, px.
/// @param z_mm [in] the depth along the optical axis, mm.
/// @param camera [in] the intrinsics, px.
/// @return the point z ((u - cx) / fx, (v - cy) / fy, 1), mm, camera frame (x right, y down,
///     z forward).
std::array<double, 3> deproject(double u_px, double v_px, double z_mm, const intrinsics &camera)
{
    return {z_mm * (u_px - camera.cx) / camera.fx, z_mm * (v_px - camera.cy) / camera.fy, z_mm};
}

/// Converts a corner's point to Eigen.
/// @param point [in] the point, mm.
/// @return the same point.
Eigen::Vector3d to_eigen(const std::array<double, 3> &point)
{
    return {point[0], point[1], point[2]};
}

/// Gives the cell of the 3x3 grid of equal cells a position lies in; the image covers
/// -0.5 to size - 0.5 px, and a position outside counts in the nearest cell.
/// @param u_px [in] the column coordinate, px; any finite value.
/// @param v_px [in] the row coordinate, px; any finite value.
/// @param image_size [in] the image's size, px.
/// @pre image_size is at least 1x1, as measure_board() makes sure (3x3 or more).
/// @return the cell, row-major, 0 top left to 8 bottom right.
int cell_of(double u_px, double v_px, cv::Size image_size)
{
    // Clamped before the cast, so that no value outside int reaches it.
    const double column = std::clamp(std::floor((u_px + 0.5) * 3.0 / image_size.width), 0.0, 2.0);
    const double row = std::clamp(std::floor((v_px + 0.5) * 3.0 / image_size.height), 0.0, 2.0);
    return static_cast<int>(row) * 3 + static_cast<int>(column);
}

/// Sums up a set of deviations.
/// @param values [in] the deviations.
/// @return their count, root mean square and largest absolute value; all 0 for none.
deviation_stats stats_of(const std::vector<double> &values)
{
    deviation_stats stats;
    stats.count = static_cast<int>(values.size());
    if (values.empty())
        return stats;
    double sum_sq = 0.0;
    for (const double value : values)
    {
        sum_sq += value * value;
        stats.max = std::max(stats.max, std::abs(value));
    }
    stats.rms = std::sqrt(sum_sq / stats.count);
    return stats;
}

/// A plane through a point with a unit normal.
struct plane
{
    Eigen::Vector3d centroid; ///< a point of the plane, mm
    Eigen::Vector3d normal;   ///< unit normal, of either sign
};

/// Fits a plane to the corners with a point: the centroid and the eigenvector of the smallest
/// eigenvalue of the centred covariance, in double.
/// @param corners [in] the corners.
/// @return the plane; nothing when fewer than plane_min_corners corners have a point or their
///     spread across their main line is under plane_min_spread_ratio of the spread along it.
std::optional<plane> fit_plane(const std::vector<board_corner> &corners)
{
    std::vector<Eigen::Vector3d> points;
    for (const board_corner &corner : corners)
        if (corner.point_mm)
            points.push_back(to_eigen(*corner.point_mm));
    if (points.size() < static_cast<size_t>(plane_min_corners))
        return std::nullopt;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d &p : points)
        centroid += p;
    centroid /= static_cast<double>(points.size());
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d &p : points)
        covariance += (p - centroid) * (p - centroid).transpose();
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    // Eigenvalues in increasing order: 0 the plane's normal, 2 the points' main line.
    if (solver.eigenvalues()(1) < plane_min_spread_ratio * solver.eigenvalues()(2))
        return std::nullopt;
    return plane{centroid, solver.eigenvectors().col(0)};
}

/// Sums the corners' deviations up by region: each corner's homography deviation and its
/// distance from the plane, each edge's length minus the square size (checkerboard_measure.hpp,
/// board_measurement::regions).
/// @param corners [in] the corners.
/// @param fitted [in] the plane fitted to the corners with a point; nothing leaves the mm
///     deviations empty.
/// @param board [in] the board, for its square size and grid.
/// @param image_size [in] the image's size, px.
/// @pre each grid place at most once among the corners, every place within the board's grid,
///     and image_size at least 1x1, as measure_board() makes sure.
/// @return middle, top left, top right, bottom left, bottom right, all, in this order.
std::vector<region_measurement> summarize_regions(const std::vector<board_corner> &corners,
                                                  const std::optional<plane> &fitted,
                                                  const checkerboard &board, cv::Size image_size)
{
    constexpr int cells = 9;
    std::vector<double> homography[cells];
    std::vector<double> plane_distance[cells];
    std::vector<double> square[cells];

    std::vector<int> at_place(static_cast<size_t>(board.columns) * board.rows, -1);
    for (size_t i = 0; i < corners.size(); i++)
    {
        const board_corner &corner = corners[i];
        at_place[static_cast<size_t>(corner.row) * board.columns + corner.column] =
            static_cast<int>(i);
        const int cell = cell_of(corner.u_px, corner.v_px, image_size);
        homography[cell].push_back(corner.homography_px);
        if (fitted && corner.point_mm)
            plane_distance[cell].push_back(
                fitted->normal.dot(to_eigen(*corner.point_mm) - fitted->centroid));
    }

    // Each edge once: from a corner to its neighbour in the next column and in the next row.
    for (const board_corner &a : corners)
    {
        if (!fitted || !a.point_mm)
            continue;
        const int neighbours[2][2] = {{a.column + 1, a.row}, {a.column, a.row + 1}};
        for (const auto &place : neighbours)
        {
            if (place[0] >= board.columns || place[1] >= board.rows)
                continue;
            const int index = at_place[static_cast<size_t>(place[1]) * board.columns + place[0]];
            if (index < 0 || !corners[index].point_mm)
                continue;
            const board_corner &b = corners[index];
            const double length = (to_eigen(*a.point_mm) - to_eigen(*b.point_mm)).norm();
            const int cell = cell_of((a.u_px + b.u_px) / 2.0, (a.v_px + b.v_px) / 2.0, image_size);
            square[cell].push_back(length - board.square_mm);
        }
    }

    std::vector<region_measurement> regions;
    for (const region_cell &region : report_regions)
    {
        std::vector<double> region_homography;
        std::vector<double> region_plane;
        std::vector<double> region_square;
        for (int cell = 0; cell < cells; cell++)
        {
            if (region.cell >= 0 && region.cell != cell)
                continue;
            region_homography.insert(region_homography.end(), homography[cell].begin(),
                                     homography[cell].end());
            region_plane.insert(region_plane.end(), plane_distance[cell].begin(),
                                plane_distance[cell].end());
            region_square.insert(region_square.end(), square[cell].begin(), square[cell].end());
        }
        double square_sum = 0.0;
        for (const double value : region_square)
            square_sum += value;
        regions.push_back({region.name, stats_of(region_homography), stats_of(region_plane),
                           stats_of(region_square),
                           region_square.empty() ? 0.0 : square_sum / region_square.size()});
    }
    return regions;
}
} // namespace

std::optional<std::vector<double>> homography_deviations(const std::vector<cv::Point2d> &grid,
                                                         const std::vector<cv::Point2d> &image_px)
{
    if (grid.size() != image_px.size() || grid.size() < 4)
        return std::nullopt;
    // On a homography's exact picture of 13x8 corners the fit leaves 4e-6 px [M 2026-11-14:
    // test_checkerboard_measure].
    const cv::Mat homography = cv::findHomography(grid, image_px, 0);
    if (homography.empty())
        return std::nullopt;
    std::vector<cv::Point2d> pictured;
    cv::perspectiveTransform(grid, pictured, homography);
    std::vector<double> deviations;
    for (size_t i = 0; i < grid.size(); i++)
        deviations.push_back(cv::norm(pictured[i] - image_px[i]));
    return deviations;
}

board_result measure_board(const cv::Mat &color, const cv::Mat &depth, const checkerboard &board,
                           const intrinsics &camera, double depth_scale_m)
{
    const std::string refusal = refusal_of_inputs(color, depth, board, camera, depth_scale_m);
    if (!refusal.empty())
        return {std::nullopt, refusal};

    const std::optional<std::vector<cv::Point2f>> found = find_corners(color, board);
    if (!found)
    {
        std::ostringstream reason;
        reason << "no checkerboard of " << board.columns << "x" << board.rows
               << " inner corners found in the color image";
        return {std::nullopt, reason.str()};
    }

    std::vector<cv::Point2d> grid;
    std::vector<cv::Point2d> image_px;
    for (size_t i = 0; i < found->size(); i++)
    {
        grid.emplace_back(static_cast<double>(i % board.columns),
                          static_cast<double>(i / board.columns));
        image_px.emplace_back((*found)[i].x, (*found)[i].y);
    }
    const std::optional<std::vector<double>> deviations = homography_deviations(grid, image_px);
    if (!deviations)
        return {std::nullopt, "no homography fits the corners found"};

    board_measurement measurement;
    for (size_t i = 0; i < image_px.size(); i++)
    {
        board_corner corner{static_cast<int>(grid[i].x),
                            static_cast<int>(grid[i].y),
                            image_px[i].x,
                            image_px[i].y,
                            (*deviations)[i],
                            std::nullopt};
        const std::optional<double> z_mm =
            depth_at_corner(depth, corner.u_px, corner.v_px, depth_scale_m);
        if (z_mm)
        {
            corner.point_mm = deproject(corner.u_px, corner.v_px, *z_mm, camera);
            measurement.corners_with_depth++;
        }
        measurement.corners.push_back(corner);
    }
    const std::optional<plane> fitted = fit_plane(measurement.corners);
    if (!fitted)
    {
        std::ostringstream reason;
        reason << measurement.corners_with_depth << " of " << measurement.corners.size()
               << " corners have depth around them; a plane needs " << plane_min_corners
               << " not on one line";
        measurement.depth_refusal = reason.str();
    }
    measurement.regions = summarize_regions(measurement.corners, fitted, board, color.size());
    return {measurement, ""};
}
} // namespace gca
