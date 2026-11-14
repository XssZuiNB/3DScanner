// SPDX-License-Identifier: 0BSD
/// @file test_checkerboard_measure.cpp
/// Offline test of distortion_check's measurement (checkerboard_measure.hpp) on synthetic
/// pictures of a tilted checkerboard, which the test renders itself, with its own pinhole maths
/// and pixel centres at whole numbers: at 1280x800 and at 640x400 on a pinhole grid, and at
/// 1280x800 through the color camera's Brown-Conrady K6 coefficients (docs/HARDWARE.md). Every
/// expected number comes from the test's own code (its homography fit, cells, plane and
/// sums), never from the measurement's. On the pinhole pictures the deviations stay small and
/// the deprojected corners land on the true ones; on the distorted one the deviations are the
/// ones the coefficients give. Each refused input, a depth window one pixel short and corners
/// with depth that span no plane are refused with their reasons. Needs no GPU and no camera.

#include "check.hpp"
#include "checkerboard_measure.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <opencv2/core.hpp>

using gca_test::check;
using gca_test::check_max_error;
using gca_test::check_refusal;
using gca_test::check_summary;
using gca_test::check_within;
using gca_test::max_error;

namespace
{
/// A camera the test renders for: its intrinsics and its picture's size.
struct camera_model
{
    const char *name;   ///< for the check lines
    gca::intrinsics in; ///< px
    cv::Size size;      ///< px
};

/// The color camera at 1280x800 [M 2026-11-13: docs/HARDWARE.md, "What the camera is and
/// reports"].
const camera_model camera_1280 = {
    "1280x800", {612.369f, 612.286f, 635.814f, 398.267f}, {1280, 800}};

/// The color camera at 640x400 [M 2026-11-13: docs/HARDWARE.md, "The camera on USB 3"].
const camera_model camera_640 = {"640x400", {306.18f, 306.14f, 317.41f, 199.13f}, {640, 400}};

/// Meters per raw depth value, the camera's 1 mm [M 2026-11-13: docs/HARDWARE.md].
constexpr double depth_scale_m = 0.001;

/// What a pinhole picture may deviate by, each about twice what this test measured on it
/// [M 2026-11-14: this test, OpenCV 4.8.0, Release build].
struct pinhole_tolerances
{
    double found_corner_px;   ///< a found corner from the true one
    double homography_rms_px; ///< the homography deviation's rms, every region
    double homography_max_px; ///< its largest, every region
    double corner_mm;         ///< a deprojected corner from the true one
    double offset_mm;         ///< the deprojected corners' mean offset from the true ones
    double plane_rms_mm;      ///< the plane deviation's rms, all corners
    double plane_max_mm;      ///< its largest
    double square_mean_mm;    ///< the square deviation's mean, all edges
    double square_rms_mm;     ///< its rms
    double square_max_mm;     ///< its largest
};

/// At 1280x800, measured: found corners 0.129 px from the true ones; homography 0.036 px rms
/// (the largest of the regions), 0.117 px at most; deprojected corners 0.123 mm, mean offset
/// 0.014 mm; plane 0.017 mm rms, 0.056 mm at most; squares 0.0005 mm mean, 0.034 mm rms,
/// 0.129 mm at most. A deprojection half a pixel off moves every corner by 0.47 mm [S: the
/// pinhole model, 0.5 px x 580 mm / 612 px].
constexpr pinhole_tolerances tolerances_1280 = {0.25, 0.08, 0.25,  0.25, 0.03,
                                                0.04, 0.12, 0.001, 0.07, 0.25};

/// At 640x400, measured: found corners 0.139 px from the true ones; homography 0.046 px rms
/// (the largest of the regions), 0.130 px at most; deprojected corners 0.252 mm, mean offset
/// 0.014 mm; plane 0.021 mm rms, 0.050 mm at most; squares 0.008 mm mean, 0.078 mm rms, 0.274
/// mm at most. Half a pixel is 0.95 mm here [S: the pinhole model, 0.5 px x 580 mm / 306 px].
constexpr pinhole_tolerances tolerances_640 = {0.3,   0.1, 0.25,  0.5,  0.03,
                                               0.045, 0.1, 0.016, 0.16, 0.55};

/// A region's homography rms or max on the lens picture from the one the lens gives: measured
/// 0.027 px [M 2026-11-14: this test].
constexpr double lens_region_px = 0.06;
/// A region's plane rms or max on the lens picture from the lens's: measured 0.036 mm
/// [M 2026-11-14: this test].
constexpr double lens_region_plane_mm = 0.08;
/// A region's square mean, rms or max on the lens picture from the lens's: measured 0.029 mm
/// [M 2026-11-14: this test].
constexpr double lens_region_square_mm = 0.06;

/// A corner's homography deviation on the lens picture from the one the lens gives: measured
/// 0.082 px [M 2026-11-14: this test].
constexpr double lens_corner_px = 0.16;

/// The tool's sums against the test's own sums of the tool's per-corner numbers: the same
/// numbers summed in another order, so equal to rounding [S: double rounding].
constexpr double same_sums = 1e-9;

/// The fitted homography's deviation on a homography's exact picture: measured 3.9e-6 px for
/// the tool's fit, 1e-12 px for the test's own [M 2026-11-14: this test].
constexpr double exact_homography_px = 1e-5;

/// The error of the lens's inverse, in normalized coordinates: measured 1.8e-15; 1e-12 is
/// 6e-10 px [S: 1e-12 x fx = 612 px].
constexpr double lens_inverse_normalized = 1e-12;

/// The color camera's Brown-Conrady K6 coefficients at 1280x800 [M 2026-11-13: docs/HARDWARE.md],
/// in OrbbecSDK's model: x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6)
/// + 2 p1 x y + p2 (r^2 + 2 x^2), y_d likewise with p1 (r^2 + 2 y^2) + 2 p2 x y [S: OrbbecSDK_v2
/// v2.10.6, src/shared/utils/CoordinateUtil.cpp].
struct lens
{
    double k1, k2, k3, k4, k5, k6, p1, p2;
};

/// The color camera's coefficients.
constexpr lens color_lens = {-1.01331, 0.497682,   -0.106898,  -0.994527,
                             0.471835, -0.0972926, 0.00022544, 0.000242177};

/// What the lens gives on the 1280x800 picture, summed over all corners, from a computation
/// of its own (Python with SciPy, without OpenCV): homography 0.442 px rms, 1.153 px at most;
/// plane 0.067 mm rms, 0.248 mm at most; squares -0.304 mm mean, 0.779 mm at most [S: the spec
/// review of this ticket, 2026-11-14]. They pin the lens's direction, which the test's own
/// numbers, rendered and expected with one lens model, cannot.
constexpr double independent_lens[] = {0.442, 1.153, 0.067, 0.248, -0.304, 0.779};
/// independent_lens is given to 3 decimals, so it holds to 0.0005; the test's numbers came within
/// 0.0004 [M 2026-11-14: this test].
constexpr double independent_rounding = 0.0006;

/// The board of the pictures: 13x8 inner corners of 60 mm (make_pose() places it).
constexpr gca::checkerboard board = {13, 8, 60.0};

/// The white margin around the squares, mm; the background beyond it is grey.
constexpr double margin_mm = 36.0;

/// The board's pose in the camera frame, mm.
struct board_pose
{
    Eigen::Vector3d origin; ///< inner corner (0, 0)
    Eigen::Vector3d along;  ///< unit vector from corner (i, j) to (i + 1, j)
    Eigen::Vector3d down;   ///< unit vector from corner (i, j) to (i, j + 1)
    Eigen::Vector3d normal; ///< along x down
};

/// Places the board: its centre at (0, 20, 580) mm, turned by 15 degrees about (1, 0.5, 0), so
/// that at 1280x800 its margin stays inside the picture, from 138 px to 1238 px and from 34 px
/// to 730 px, and 6 corners fall into each corner cell of the 3x3 grid, with the lens and
/// without [M 2026-11-14: the Python prototype of this test]; test_pinhole_picture() checks 4
/// or more.
/// @return the pose.
board_pose make_pose()
{
    const Eigen::Matrix3d r =
        Eigen::AngleAxisd(15.0 * M_PI / 180.0, Eigen::Vector3d(1.0, 0.5, 0.0).normalized())
            .toRotationMatrix();
    const Eigen::Vector3d centre(0.0, 20.0, 580.0);
    board_pose pose;
    pose.along = r.col(0);
    pose.down = r.col(1);
    pose.normal = r.col(2);
    pose.origin = centre - pose.along * board.square_mm * (board.columns - 1) / 2.0 -
                  pose.down * board.square_mm * (board.rows - 1) / 2.0;
    return pose;
}

/// Applies the lens to a point of the normalized image plane.
/// @param x [in] the pinhole coordinate x / z.
/// @param y [in] the pinhole coordinate y / z.
/// @param l [in] the coefficients.
/// @return the distorted normalized coordinates.
Eigen::Vector2d distort(double x, double y, const lens &l)
{
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double radial =
        (1.0 + l.k1 * r2 + l.k2 * r4 + l.k3 * r6) / (1.0 + l.k4 * r2 + l.k5 * r4 + l.k6 * r6);
    return {x * radial + 2.0 * l.p1 * x * y + l.p2 * (r2 + 2.0 * x * x),
            y * radial + l.p1 * (r2 + 2.0 * y * y) + 2.0 * l.p2 * x * y};
}

/// Undoes the lens by fixed-point iteration: x = (x_d - tangential(x)) / radial(x).
/// @param xd [in] the distorted normalized coordinate x.
/// @param yd [in] the distorted normalized coordinate y.
/// @param l [in] the coefficients.
/// @return the pinhole coordinates whose distortion is (xd, yd), to the last steps of double
///     (render() checks it by distort()).
Eigen::Vector2d undistort(double xd, double yd, const lens &l)
{
    double x = xd;
    double y = yd;
    for (int i = 0; i < 50; i++)
    {
        const double r2 = x * x + y * y;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double radial =
            (1.0 + l.k1 * r2 + l.k2 * r4 + l.k3 * r6) / (1.0 + l.k4 * r2 + l.k5 * r4 + l.k6 * r6);
        const double nx = (xd - 2.0 * l.p1 * x * y - l.p2 * (r2 + 2.0 * x * x)) / radial;
        const double ny = (yd - l.p1 * (r2 + 2.0 * y * y) - 2.0 * l.p2 * x * y) / radial;
        const bool settled = std::abs(nx - x) < 1e-15 && std::abs(ny - y) < 1e-15;
        x = nx;
        y = ny;
        if (settled)
            break;
    }
    return {x, y};
}

/// A synthetic aligned frame.
struct picture
{
    cv::Mat color; ///< BGR, 8 bits in each channel
    cv::Mat depth; ///< 16 bits, mm, 0 off the board
    /// The largest error of the lens's inverse over the rays followed through the lens, in
    /// normalized coordinates; 0 for a pinhole picture.
    double lens_inverse_error = 0.0;
};

/// The ray of a position in the image: its pinhole coordinates, through the lens if one is given.
/// @param camera [in] the camera.
/// @param u [in] the column coordinate, px, pixel centres at whole numbers.
/// @param v [in] the row coordinate, px.
/// @param through_lens [in] whether the picture is distorted.
/// @return the ray (x, y, 1), camera frame.
Eigen::Vector3d ray_of(const camera_model &camera, double u, double v, bool through_lens)
{
    const double xd = (u - camera.in.cx) / camera.in.fx;
    const double yd = (v - camera.in.cy) / camera.in.fy;
    if (!through_lens)
        return {xd, yd, 1.0};
    const Eigen::Vector2d p = undistort(xd, yd, color_lens);
    return {p.x(), p.y(), 1.0};
}

/// What the board shows where a ray meets its plane.
struct board_hit
{
    double z_mm;   ///< the depth of the point the ray meets
    bool on_board; ///< within the squares and the margin
    double shade;  ///< 0.1 on a black square, 0.9 on white and the margin, 0.5 off the board
};

/// Follows a ray to the board's plane.
/// @param ray [in] the ray (x, y, 1), camera frame.
/// @param pose [in] the board's pose.
/// @return the point's depth and what the board shows there.
board_hit hit_board(const Eigen::Vector3d &ray, const board_pose &pose)
{
    const double s = pose.normal.dot(pose.origin) / pose.normal.dot(ray);
    const Eigen::Vector3d offset = ray * s - pose.origin;
    const double a = offset.dot(pose.along);
    const double b = offset.dot(pose.down);
    const double sq = board.square_mm;
    board_hit hit{s, false, 0.5};
    if (a < -sq - margin_mm || a >= board.columns * sq + margin_mm || b < -sq - margin_mm ||
        b >= board.rows * sq + margin_mm)
        return hit;
    hit.on_board = true;
    const long column = static_cast<long>(std::floor(a / sq)) + 1;
    const long row = static_cast<long>(std::floor(b / sq)) + 1;
    const bool on_squares = column >= 0 && column <= board.columns && row >= 0 && row <= board.rows;
    hit.shade = on_squares && (column + row) % 2 == 0 ? 0.1 : 0.9;
    return hit;
}

/// Renders the board: each pixel's color is the mean of 4x4 samples spread evenly over the pixel
/// (pixel u covers u - 0.5 to u + 0.5), its depth that of the ray through its centre, rounded to
/// 1 mm, 0 off the board.
/// @param camera [in] the camera.
/// @param pose [in] the board's pose.
/// @param through_lens [in] whether the picture is distorted by the color camera's coefficients.
/// @return the picture.
picture render(const camera_model &camera, const board_pose &pose, bool through_lens)
{
    constexpr int samples = 4;
    picture pic{cv::Mat(camera.size, CV_8UC3), cv::Mat(camera.size, CV_16UC1), 0.0};
    for (int v = 0; v < camera.size.height; v++)
    {
        for (int u = 0; u < camera.size.width; u++)
        {
            double shade = 0.0;
            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < samples; j++)
                {
                    const double su = u - 0.5 + (j + 0.5) / samples;
                    const double sv = v - 0.5 + (i + 0.5) / samples;
                    const Eigen::Vector3d ray = ray_of(camera, su, sv, through_lens);
                    if (through_lens)
                    {
                        const Eigen::Vector2d back = distort(ray.x(), ray.y(), color_lens);
                        pic.lens_inverse_error =
                            std::max({pic.lens_inverse_error,
                                      std::abs(back.x() - (su - camera.in.cx) / camera.in.fx),
                                      std::abs(back.y() - (sv - camera.in.cy) / camera.in.fy)});
                    }
                    shade += hit_board(ray, pose).shade;
                }
            }
            const auto level = static_cast<uint8_t>(std::lround(shade / (samples * samples) * 255));
            pic.color.at<cv::Vec3b>(v, u) = cv::Vec3b(level, level, level);
            const board_hit centre = hit_board(ray_of(camera, u, v, through_lens), pose);
            pic.depth.at<uint16_t>(v, u) =
                centre.on_board ? static_cast<uint16_t>(std::lround(centre.z_mm)) : 0;
        }
    }
    return pic;
}

/// One inner corner as the scene has it.
struct true_corner
{
    int column;            ///< its column on the board's grid
    int row;               ///< its row
    Eigen::Vector3d point; ///< its position, mm, camera frame
    Eigen::Vector2d px;    ///< where the picture shows it, px
};

/// Projects every inner corner of the board, through the lens if one is given.
/// @param camera [in] the camera.
/// @param pose [in] the board's pose.
/// @param through_lens [in] whether the picture is distorted.
/// @return the corners, row by row.
std::vector<true_corner> true_corners(const camera_model &camera, const board_pose &pose,
                                      bool through_lens)
{
    std::vector<true_corner> corners;
    for (int j = 0; j < board.rows; j++)
    {
        for (int i = 0; i < board.columns; i++)
        {
            const Eigen::Vector3d p = pose.origin + pose.along * (i * board.square_mm) +
                                      pose.down * (j * board.square_mm);
            Eigen::Vector2d n(p.x() / p.z(), p.y() / p.z());
            if (through_lens)
                n = distort(n.x(), n.y(), color_lens);
            corners.push_back(
                {i,
                 j,
                 p,
                 {camera.in.fx * n.x() + camera.in.cx, camera.in.fy * n.y() + camera.in.cy}});
        }
    }
    return corners;
}

/// The test's own homography fit: the normalized direct linear transform, then Gauss-Newton on
/// the corners' image distances (both images normalized to their centroid and a mean distance
/// of sqrt(2), which scales every image distance by one factor), in double.
/// @param grid [in] each corner's grid place, in squares.
/// @param image_px [in] each corner's image position, px.
/// @return each corner's distance from the fitted homography's picture of its grid place, px.
std::vector<double> own_homography_deviations(const std::vector<Eigen::Vector2d> &grid,
                                              const std::vector<Eigen::Vector2d> &image_px)
{
    const auto normalizing = [](const std::vector<Eigen::Vector2d> &points) {
        Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
        for (const Eigen::Vector2d &p : points)
            centroid += p;
        centroid /= static_cast<double>(points.size());
        double mean_distance = 0.0;
        for (const Eigen::Vector2d &p : points)
            mean_distance += (p - centroid).norm() / points.size();
        const double s = std::sqrt(2.0) / mean_distance;
        Eigen::Matrix3d t;
        t << s, 0.0, -s * centroid.x(), 0.0, s, -s * centroid.y(), 0.0, 0.0, 1.0;
        return t;
    };
    const Eigen::Matrix3d tg = normalizing(grid);
    const Eigen::Matrix3d ti = normalizing(image_px);
    const size_t n = grid.size();
    std::vector<Eigen::Vector2d> g(n);
    std::vector<Eigen::Vector2d> m(n);
    for (size_t k = 0; k < n; k++)
    {
        g[k] = (tg * grid[k].homogeneous()).hnormalized();
        m[k] = (ti * image_px[k].homogeneous()).hnormalized();
    }
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(2 * n, 9);
    for (size_t k = 0; k < n; k++)
    {
        const double x = g[k].x(), y = g[k].y(), u = m[k].x(), v = m[k].y();
        a.row(2 * k) << -x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u;
        a.row(2 * k + 1) << 0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v;
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
    Eigen::VectorXd h = svd.matrixV().col(8) / svd.matrixV()(8, 8);
    for (int iteration = 0; iteration < 50; iteration++)
    {
        Eigen::MatrixXd jacobian(2 * n, 8);
        Eigen::VectorXd residual(2 * n);
        for (size_t k = 0; k < n; k++)
        {
            const double x = g[k].x(), y = g[k].y();
            const double w = h(6) * x + h(7) * y + 1.0;
            const double pu = (h(0) * x + h(1) * y + h(2)) / w;
            const double pv = (h(3) * x + h(4) * y + h(5)) / w;
            residual(2 * k) = pu - m[k].x();
            residual(2 * k + 1) = pv - m[k].y();
            jacobian.row(2 * k) << x / w, y / w, 1.0 / w, 0.0, 0.0, 0.0, -pu * x / w, -pu * y / w;
            jacobian.row(2 * k + 1) << 0.0, 0.0, 0.0, x / w, y / w, 1.0 / w, -pv * x / w,
                -pv * y / w;
        }
        const Eigen::VectorXd step =
            (jacobian.transpose() * jacobian).ldlt().solve(-jacobian.transpose() * residual);
        h.head(8) += step;
        if (step.norm() < 1e-15)
            break;
    }
    Eigen::Matrix3d hn;
    hn << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), 1.0;
    const Eigen::Matrix3d homography = ti.inverse() * hn * tg;
    std::vector<double> deviations;
    for (size_t k = 0; k < n; k++)
        deviations.push_back(
            ((homography * grid[k].homogeneous()).hnormalized() - image_px[k]).norm());
    return deviations;
}

/// A corner as the test sums it up.
struct corner_record
{
    int column;                           ///< its column on the board's grid
    int row;                              ///< its row
    Eigen::Vector2d px;                   ///< its image position, px
    double homography_px;                 ///< its homography deviation, px
    std::optional<Eigen::Vector3d> point; ///< its deprojected position, mm
};

/// The test's own cell of the 3x3 grid of equal cells: the image covers -0.5 to size - 0.5 px
/// (pixel centres at whole numbers), a position outside counts in the nearest cell.
/// @param px [in] the position, px.
/// @param size [in] the image's size, px.
/// @return the cell, row-major, 0 top left to 8 bottom right.
int own_cell(const Eigen::Vector2d &px, cv::Size size)
{
    const int column =
        std::clamp(static_cast<int>(std::floor((px.x() + 0.5) / size.width * 3)), 0, 2);
    const int row =
        std::clamp(static_cast<int>(std::floor((px.y() + 0.5) / size.height * 3)), 0, 2);
    return row * 3 + column;
}

/// The test's own sums of a set of deviations.
/// @param values [in] the deviations.
/// @return their count, root mean square and largest absolute value.
gca::deviation_stats own_stats(const std::vector<double> &values)
{
    gca::deviation_stats stats;
    stats.count = static_cast<int>(values.size());
    for (const double value : values)
    {
        stats.rms += value * value;
        stats.max = std::max(stats.max, std::abs(value));
    }
    stats.rms = values.empty() ? 0.0 : std::sqrt(stats.rms / values.size());
    return stats;
}

/// The test's own summary by region, the measurement's definitions written out again: one
/// plane through every corner with a point (centred covariance, smallest eigenvector), each
/// corner's distance from it, each edge (a corner and its neighbour in the next column or row,
/// both with a point) minus the square size, a corner in its cell, an edge in its midpoint's.
/// @param corners [in] the corners; with fewer than 4 points the mm sums stay empty.
/// @param size [in] the image's size, px.
/// @return middle, top left, top right, bottom left, bottom right, all.
std::vector<gca::region_measurement> own_regions(const std::vector<corner_record> &corners,
                                                 cv::Size size)
{
    std::vector<Eigen::Vector3d> points;
    for (const corner_record &c : corners)
        if (c.point)
            points.push_back(*c.point);
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    if (points.size() >= 4)
    {
        for (const Eigen::Vector3d &p : points)
            centroid += p / static_cast<double>(points.size());
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (const Eigen::Vector3d &p : points)
            covariance += (p - centroid) * (p - centroid).transpose();
        normal = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(covariance).eigenvectors().col(0);
    }
    const struct
    {
        const char *name;
        int cell;
    } regions[] = {{"middle", 4},      {"top left", 0},     {"top right", 2},
                   {"bottom left", 6}, {"bottom right", 8}, {"all", -1}};
    std::vector<gca::region_measurement> result;
    for (const auto &region : regions)
    {
        const auto in_region = [&](const Eigen::Vector2d &px) {
            return region.cell < 0 || own_cell(px, size) == region.cell;
        };
        std::vector<double> homography;
        std::vector<double> plane;
        std::vector<double> square;
        for (const corner_record &a : corners)
        {
            if (in_region(a.px))
            {
                homography.push_back(a.homography_px);
                if (a.point && points.size() >= 4)
                    plane.push_back(normal.dot(*a.point - centroid));
            }
            for (const corner_record &b : corners)
            {
                const bool neighbour = (b.column == a.column + 1 && b.row == a.row) ||
                                       (b.column == a.column && b.row == a.row + 1);
                if (neighbour && a.point && b.point && points.size() >= 4 &&
                    in_region((a.px + b.px) / 2.0))
                    square.push_back((*a.point - *b.point).norm() - board.square_mm);
            }
        }
        double square_sum = 0.0;
        for (const double value : square)
            square_sum += value;
        result.push_back({region.name, own_stats(homography), own_stats(plane), own_stats(square),
                          square.empty() ? 0.0 : square_sum / square.size()});
    }
    return result;
}

/// The measurement's corners as the test sums them up.
/// @param m [in] the measurement.
/// @return its corners.
std::vector<corner_record> records_of(const gca::board_measurement &m)
{
    std::vector<corner_record> records;
    for (const gca::board_corner &c : m.corners)
    {
        std::optional<Eigen::Vector3d> point;
        if (c.point_mm)
            point = Eigen::Vector3d((*c.point_mm)[0], (*c.point_mm)[1], (*c.point_mm)[2]);
        records.push_back({c.column, c.row, {c.u_px, c.v_px}, c.homography_px, point});
    }
    return records;
}

/// Checks that two summaries agree: the same counts, and each sum within its tolerance.
/// @param what [in] whose summaries, for the check lines.
/// @param got [in] the measurement's.
/// @param want [in] the expected one.
/// @param px [in] the tolerance of the homography sums, px.
/// @param plane_mm [in] the tolerance of the plane sums, mm.
/// @param square_mm [in] the tolerance of the square sums, mm.
void check_same_regions(const std::string &what, const std::vector<gca::region_measurement> &got,
                        const std::vector<gca::region_measurement> &want, double px,
                        double plane_mm, double square_mm)
{
    check(got.size() == want.size(), (what + ": six regions").c_str());
    max_error homography;
    max_error plane;
    max_error square;
    for (size_t r = 0; r < got.size() && r < want.size(); r++)
    {
        const gca::region_measurement &g = got[r];
        const gca::region_measurement &w = want[r];
        check(g.name == w.name && g.homography_px.count == w.homography_px.count &&
                  g.plane_mm.count == w.plane_mm.count && g.square_mm.count == w.square_mm.count,
              (what + ", " + w.name + ": the same corners and edges").c_str());
        homography.add(std::abs(g.homography_px.rms - w.homography_px.rms));
        homography.add(std::abs(g.homography_px.max - w.homography_px.max));
        plane.add(std::abs(g.plane_mm.rms - w.plane_mm.rms));
        plane.add(std::abs(g.plane_mm.max - w.plane_mm.max));
        square.add(std::abs(g.square_mean_mm - w.square_mean_mm));
        square.add(std::abs(g.square_mm.rms - w.square_mm.rms));
        square.add(std::abs(g.square_mm.max - w.square_mm.max));
    }
    check_max_error(what + ": homography rms and max, px", homography, px);
    check_max_error(what + ": plane rms and max, mm", plane, plane_mm);
    check_max_error(what + ": square mean, rms and max, mm", square, square_mm);
}

/// Finds the true corner nearest a found one.
/// @param corners [in] the true corners.
/// @param found [in] a corner the measurement found.
/// @return the index of the nearest true corner.
size_t nearest(const std::vector<true_corner> &corners, const gca::board_corner &found)
{
    size_t best = 0;
    for (size_t i = 1; i < corners.size(); i++)
        if ((corners[i].px - Eigen::Vector2d(found.u_px, found.v_px)).norm() <
            (corners[best].px - Eigen::Vector2d(found.u_px, found.v_px)).norm())
            best = i;
    return best;
}

/// Prints a region's numbers.
/// @param what [in] whose numbers.
/// @param region [in] the region.
void print_region(const std::string &what, const gca::region_measurement &region)
{
    std::printf("%s, %s: %d corners, homography px rms %.3f max %.3f; plane mm rms %.3f max %.3f; "
                "%d edges, square mm mean %.3f rms %.3f max %.3f\n",
                what.c_str(), region.name.c_str(), region.homography_px.count,
                region.homography_px.rms, region.homography_px.max, region.plane_mm.rms,
                region.plane_mm.max, region.square_mm.count, region.square_mean_mm,
                region.square_mm.rms, region.square_mm.max);
}

/// Measures a picture and checks that it was measured with every corner, and that its sums by
/// region are the test's own sums of its corners.
/// @param what [in] the picture, for the check lines.
/// @param camera [in] the camera it was rendered for.
/// @param pic [in] the picture.
/// @return the measurement; empty when it was refused (a failed check).
gca::board_measurement measure(const std::string &what, const camera_model &camera,
                               const picture &pic)
{
    const gca::board_result result =
        gca::measure_board(pic.color, pic.depth, board, camera.in, depth_scale_m);
    if (!result.refusal.empty())
        std::printf("%s: refused: %s\n", what.c_str(), result.refusal.c_str());
    check(result.measurement.has_value(), (what + ": measured").c_str());
    if (!result.measurement)
        return {};
    const gca::board_measurement &m = *result.measurement;
    check(m.corners.size() == static_cast<size_t>(board.columns) * board.rows,
          (what + ": every corner found").c_str());
    check_same_regions(what + ", its sums against the test's sums of its corners", m.regions,
                       own_regions(records_of(m), camera.size), same_sums, same_sums, same_sums);
    return m;
}

/// On a pinhole picture: every corner is found where the scene has it, in the board's grid
/// order or the turned one; every corner cell holds 4 corners or more; the homography
/// deviations, rms and largest, stay small in every region; the deprojected corners land on the
/// true ones, one by one and on average, so a deprojection half a pixel off fails; the plane
/// and square deviations, mean, rms and largest, stay small.
/// @param camera [in] the camera the picture was rendered for.
/// @param pic [in] the picture.
/// @param pose [in] the board's pose.
/// @param tol [in] the tolerances at this size.
void test_pinhole_picture(const camera_model &camera, const picture &pic, const board_pose &pose,
                          const pinhole_tolerances &tol)
{
    const std::string what = std::string("pinhole picture ") + camera.name;
    const gca::board_measurement m = measure(what, camera, pic);
    if (m.corners.empty())
        return;
    check(m.corners_with_depth == static_cast<int>(m.corners.size()) && m.depth_refusal.empty(),
          (what + ": every corner has depth").c_str());
    const std::vector<true_corner> truth = true_corners(camera, pose, false);
    max_error found_px;
    max_error point_mm;
    Eigen::Vector3d offset_sum = Eigen::Vector3d::Zero();
    bool grid_order = true;
    bool turned_order = true;
    for (const gca::board_corner &corner : m.corners)
    {
        const true_corner &t = truth[nearest(truth, corner)];
        found_px.add((t.px - Eigen::Vector2d(corner.u_px, corner.v_px)).norm());
        grid_order &= corner.column == t.column && corner.row == t.row;
        turned_order &=
            corner.column == board.columns - 1 - t.column && corner.row == board.rows - 1 - t.row;
        if (!corner.point_mm)
            continue;
        const Eigen::Vector3d p((*corner.point_mm)[0], (*corner.point_mm)[1],
                                (*corner.point_mm)[2]);
        point_mm.add((p - t.point).norm());
        offset_sum += p - t.point;
    }
    check(grid_order || turned_order,
          (what + ": every corner at its grid place, or every one at the turned place").c_str());
    check_max_error(what + ": found corner from the true one, px", found_px, tol.found_corner_px);
    max_error homography_rms;
    max_error homography_max;
    for (const gca::region_measurement &region : m.regions)
    {
        print_region(what, region);
        check(region.homography_px.count >= 4,
              (what + ": 4 corners or more in " + region.name).c_str());
        homography_rms.add(region.homography_px.rms);
        homography_max.add(region.homography_px.max);
    }
    check_max_error(what + ": homography rms, every region, px", homography_rms,
                    tol.homography_rms_px);
    check_max_error(what + ": homography max, every region, px", homography_max,
                    tol.homography_max_px);
    check_max_error(what + ": deprojected corner from the true one, mm", point_mm, tol.corner_mm);
    check_within((what + ": mean offset of the deprojected corners, mm").c_str(),
                 (offset_sum / m.corners_with_depth).norm(), tol.offset_mm);
    const gca::region_measurement &all = m.regions.back();
    check_within((what + ": plane rms, mm").c_str(), all.plane_mm.rms, tol.plane_rms_mm);
    check_within((what + ": plane max, mm").c_str(), all.plane_mm.max, tol.plane_max_mm);
    check_within((what + ": square mean, mm").c_str(), std::abs(all.square_mean_mm),
                 tol.square_mean_mm);
    check_within((what + ": square rms, mm").c_str(), all.square_mm.rms, tol.square_rms_mm);
    check_within((what + ": square max, mm").c_str(), all.square_mm.max, tol.square_max_mm);
}

/// On the picture through the lens: each corner's homography deviation is the one the lens
/// gives, and every region's homography, plane and square sums are the ones the lens gives, all
/// worked out by the test's own code from the true distorted corners; those agree with a
/// computation of their own; the lens shows above the pinhole picture's tolerance.
/// @param pic [in] the 1280x800 picture through the lens.
/// @param pose [in] the board's pose.
void test_lens_picture(const picture &pic, const board_pose &pose)
{
    const gca::board_measurement m = measure("lens picture", camera_1280, pic);
    if (m.corners.empty())
        return;
    const std::vector<true_corner> truth = true_corners(camera_1280, pose, true);
    std::vector<Eigen::Vector2d> grid;
    std::vector<Eigen::Vector2d> image_px;
    for (const true_corner &t : truth)
    {
        grid.emplace_back(t.column, t.row);
        image_px.push_back(t.px);
    }
    const std::vector<double> deviations = own_homography_deviations(grid, image_px);
    std::vector<corner_record> expected;
    for (size_t i = 0; i < truth.size(); i++)
    {
        // The pinhole deprojection of the distorted position, with the true depth: what the
        // picture's depth gives there.
        const true_corner &t = truth[i];
        const double z = t.point.z();
        const gca::intrinsics &in = camera_1280.in;
        expected.push_back(
            {t.column, t.row, t.px, deviations[i],
             Eigen::Vector3d(z * (t.px.x() - in.cx) / in.fx, z * (t.px.y() - in.cy) / in.fy, z)});
    }

    max_error corner_px;
    for (const gca::board_corner &corner : m.corners)
        corner_px.add(std::abs(corner.homography_px - deviations[nearest(truth, corner)]));
    check_max_error("lens picture: a corner's homography deviation from the lens's, px", corner_px,
                    lens_corner_px);

    const std::vector<gca::region_measurement> want = own_regions(expected, camera_1280.size);
    for (size_t r = 0; r < m.regions.size() && r < want.size(); r++)
    {
        print_region("lens picture", m.regions[r]);
        print_region("the lens gives", want[r]);
    }
    check_same_regions("lens picture against the lens", m.regions, want, lens_region_px,
                       lens_region_plane_mm, lens_region_square_mm);

    const gca::region_measurement &all = want.back();
    const double own[] = {all.homography_px.rms, all.homography_px.max, all.plane_mm.rms,
                          all.plane_mm.max,      all.square_mean_mm,    all.square_mm.max};
    max_error independent;
    for (size_t i = 0; i < std::size(own); i++)
        independent.add(std::abs(own[i] - independent_lens[i]));
    check_max_error("the lens, all corners: the test's numbers from the independent ones",
                    independent, independent_rounding);

    std::printf("lens picture: largest homography deviation %.3f px, the pinhole picture's "
                "tolerance %.3g px\n",
                m.regions.back().homography_px.max, tolerances_1280.homography_max_px);
    check(m.regions.back().homography_px.max > tolerances_1280.homography_max_px,
          "lens picture: the homography deviation shows the lens");
}

/// Checks that measure_board() refuses a case with the expected reason.
/// @param what [in] the case.
/// @param color [in] the color image.
/// @param depth [in] the depth image.
/// @param b [in] the board.
/// @param camera [in] the intrinsics.
/// @param scale_m [in] the depth scale, m per unit.
/// @param expected [in] the expected reason.
void check_refused(const std::string &what, const cv::Mat &color, const cv::Mat &depth,
                   const gca::checkerboard &b, const gca::intrinsics &camera, double scale_m,
                   const std::string &expected)
{
    const gca::board_result result = gca::measure_board(color, depth, b, camera, scale_m);
    check_refusal(what, !result.measurement.has_value(), result.refusal,
                  result.refusal == expected);
}

/// Every input measure_board() refuses is refused with its reason, before the corner finder:
/// the images' types and sizes, the board, the intrinsics, the principal point, the depth scale;
/// and a picture without a board.
/// @param pinhole [in] the 1280x800 pinhole picture.
void test_input_refusals(const picture &pinhole)
{
    const gca::intrinsics in = camera_1280.in;
    const cv::Mat &color = pinhole.color;
    const cv::Mat &depth = pinhole.depth;
    const double scale = depth_scale_m;
    const cv::Mat gray(camera_1280.size, CV_8UC1, cv::Scalar(128));
    check_refused("a gray color image", gray, depth, board, in, scale,
                  "the color image is not an image of 8 bits in each of 3 channels");
    const cv::Mat tiny_color(cv::Size(2, 2), CV_8UC3, cv::Scalar(128, 128, 128));
    const cv::Mat tiny_depth(cv::Size(2, 2), CV_16UC1, cv::Scalar(0));
    check_refused("a 2x2 picture", tiny_color, tiny_depth, board, in, scale,
                  "the color image is 2x2, under 3x3");
    const cv::Mat float_depth(camera_1280.size, CV_32FC1, cv::Scalar(0));
    check_refused("a float depth image", color, float_depth, board, in, scale,
                  "the depth image is not an image of 16 bits in one channel");
    const cv::Mat small_depth(camera_640.size, CV_16UC1, cv::Scalar(0));
    check_refused("a depth image of another size", color, small_depth, board, in, scale,
                  "the color image is 1280x800 and the depth image 640x400; aligned frames have "
                  "one size");
    check_refused("a board of 2x8 corners", color, depth, {2, 8, 60.0}, in, scale,
                  "the board needs at least 3x3 inner corners");
    check_refused("a square of 0 mm", color, depth, {13, 8, 0.0}, in, scale,
                  "the square size is not a positive number of mm");
    check_refused("a square of NaN mm", color, depth,
                  {13, 8, std::numeric_limits<double>::quiet_NaN()}, in, scale,
                  "the square size is not a positive number of mm");
    check_refused("fx of 0", color, depth, board, {0.0f, in.fy, in.cx, in.cy}, scale,
                  "the intrinsics are not finite with positive fx and fy");
    check_refused("cy of infinity", color, depth, board,
                  {in.fx, in.fy, in.cx, std::numeric_limits<float>::infinity()}, scale,
                  "the intrinsics are not finite with positive fx and fy");
    const cv::Mat small_color(camera_640.size, CV_8UC3, cv::Scalar(128, 128, 128));
    check_refused("1280x800 intrinsics on a 640x400 picture", small_color, small_depth, board, in,
                  scale,
                  "the principal point (635.814, 398.267) lies outside the middle third of the "
                  "640x400 image: intrinsics of another stream size");
    check_refused("a depth scale of 0", color, depth, board, in, 0.0,
                  "the depth scale is not a positive number of m per unit");
    const cv::Mat grey(camera_1280.size, CV_8UC3, cv::Scalar(128, 128, 128));
    check_refused("a picture without a board", grey, depth, board, in, scale,
                  "no checkerboard of 13x8 inner corners found in the color image");
}

/// The pinhole picture with depth only in the windows of some corners, each window with all its
/// pixels or all but some.
/// @param pinhole [in] the 1280x800 pinhole picture.
/// @param m [in] its measurement, for the corners' positions.
/// @param kept [in] for each corner to keep (its index in m.corners), how many pixels of its
///     window to zero, the first ones row by row.
/// @return the depth image.
cv::Mat depth_at(const picture &pinhole, const gca::board_measurement &m,
                 const std::vector<std::pair<size_t, int>> &kept)
{
    // The window as the measurement places it: 9x9 px around the pixel nearest the corner.
    constexpr int radius = 4;
    cv::Mat depth(pinhole.depth.size(), CV_16UC1, cv::Scalar(0));
    for (const auto &[index, zeroed] : kept)
    {
        const int cx = static_cast<int>(std::lround(m.corners[index].u_px));
        const int cy = static_cast<int>(std::lround(m.corners[index].v_px));
        int k = 0;
        for (int y = cy - radius; y <= cy + radius; y++)
            for (int x = cx - radius; x <= cx + radius; x++, k++)
                depth.at<uint16_t>(y, x) = k < zeroed ? 0 : pinhole.depth.at<uint16_t>(y, x);
    }
    return depth;
}

/// Depth windows and the plane: a window with 61 valid pixels of 81 gives its corner depth, one
/// with 60 does not; corners with depth that are 4 in a square span a plane, 3 or a row of 13
/// do not and are refused with the reason; a picture without depth is refused so too.
/// @param pinhole [in] the 1280x800 pinhole picture.
void test_depth_refusals(const picture &pinhole)
{
    const gca::board_measurement full =
        gca::measure_board(pinhole.color, pinhole.depth, board, camera_1280.in, depth_scale_m)
            .measurement.value_or(gca::board_measurement{});
    check(full.corners.size() == static_cast<size_t>(board.columns) * board.rows,
          "depth refusals: the pinhole picture measured");
    if (full.corners.size() != static_cast<size_t>(board.columns) * board.rows)
        return;
    const auto measured = [&](const cv::Mat &depth) {
        return gca::measure_board(pinhole.color, depth, board, camera_1280.in, depth_scale_m)
            .measurement.value_or(gca::board_measurement{});
    };
    // Corners 0, 1, 13 and 14: a square of four on the board's grid; 0 to 12: its first row.
    const gca::board_measurement windows = measured(depth_at(pinhole, full, {{0, 21}, {1, 20}}));
    check(windows.corners.size() == full.corners.size() && !windows.corners[0].point_mm &&
              windows.corners[1].point_mm,
          "a window with 60 valid pixels gives no depth, one with 61 does");
    const gca::board_measurement square =
        measured(depth_at(pinhole, full, {{0, 0}, {1, 0}, {13, 0}, {14, 0}}));
    check(square.depth_refusal.empty() && square.regions.back().plane_mm.count == 4,
          "4 corners in a square with depth: a plane through them");
    const gca::board_measurement three =
        measured(depth_at(pinhole, full, {{0, 0}, {1, 0}, {13, 0}}));
    check_refusal("3 corners with depth", !three.depth_refusal.empty(), three.depth_refusal,
                  three.depth_refusal ==
                      "3 of 104 corners have depth around them; a plane needs 4 not on one line");
    std::vector<std::pair<size_t, int>> first_row;
    for (size_t i = 0; i < 13; i++)
        first_row.push_back({i, 0});
    const gca::board_measurement row = measured(depth_at(pinhole, full, first_row));
    check_refusal("a row of 13 corners with depth", !row.depth_refusal.empty(), row.depth_refusal,
                  row.depth_refusal == "13 of 104 corners have depth around them; a plane needs "
                                       "4 not on one line");
    check(row.regions.back().plane_mm.count == 0 && row.regions.back().square_mm.count == 0,
          "a row of 13 corners with depth: no plane and no square deviations");
    const gca::board_measurement none =
        measured(cv::Mat(pinhole.depth.size(), CV_16UC1, cv::Scalar(0)));
    check_refusal("a picture without depth", !none.depth_refusal.empty(), none.depth_refusal,
                  none.depth_refusal ==
                      "0 of 104 corners have depth around them; a plane needs 4 not on one line");
}

/// The homography fits themselves: on a homography's exact picture the tool's fit and the
/// test's own leave no more than rounding, and the tool's refuses fewer than 4 corners or lists
/// of two sizes.
void test_homography_fit()
{
    const cv::Matx33d h(1.1, 0.2, 300.0, -0.1, 0.9, 150.0, 0.0002, -0.0001, 1.0);
    std::vector<cv::Point2d> grid;
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 13; i++)
            grid.emplace_back(i * 60.0, j * 60.0);
    std::vector<cv::Point2d> image_px;
    cv::perspectiveTransform(grid, image_px, cv::Mat(h));
    const std::optional<std::vector<double>> deviations =
        gca::homography_deviations(grid, image_px);
    check(deviations.has_value(), "a homography's picture: fitted");
    if (deviations)
        check_within("a homography's picture: the tool's largest deviation, px",
                     *std::max_element(deviations->begin(), deviations->end()),
                     exact_homography_px);
    std::vector<Eigen::Vector2d> own_grid;
    std::vector<Eigen::Vector2d> own_image;
    for (size_t i = 0; i < grid.size(); i++)
    {
        own_grid.emplace_back(grid[i].x, grid[i].y);
        own_image.emplace_back(image_px[i].x, image_px[i].y);
    }
    const std::vector<double> own = own_homography_deviations(own_grid, own_image);
    check_within("a homography's picture: the test's own largest deviation, px",
                 *std::max_element(own.begin(), own.end()), exact_homography_px);
    check(!gca::homography_deviations({grid.begin(), grid.begin() + 3},
                                      {image_px.begin(), image_px.begin() + 3}),
          "3 corners: refused");
    check(!gca::homography_deviations(grid, {image_px.begin(), image_px.end() - 1}),
          "lists of two sizes: refused");
}
} // namespace

/// Runs the checks.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    test_homography_fit();
    const board_pose pose = make_pose();
    const auto start = std::chrono::steady_clock::now();
    const picture pinhole = render(camera_1280, pose, false);
    const picture through_lens = render(camera_1280, pose, true);
    const picture small = render(camera_640, pose, false);
    std::printf("rendered the three pictures in %.2f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    check_within("the lens's inverse, normalized", through_lens.lens_inverse_error,
                 lens_inverse_normalized);
    test_pinhole_picture(camera_1280, pinhole, pose, tolerances_1280);
    test_pinhole_picture(camera_640, small, pose, tolerances_640);
    test_lens_picture(through_lens, pose);
    test_input_refusals(pinhole);
    test_depth_refusals(pinhole);
    return check_summary("test_checkerboard_measure");
}
