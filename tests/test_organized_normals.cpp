// SPDX-License-Identifier: 0BSD
/// @file test_organized_normals.cpp
/// The geometry module's organized normals (geometry/organized_normals.hpp) on organized points of
/// frames of the tests' synthetic source and of frames built here, with the camera's intrinsics;
/// the method is docs/methods/normals.md. Before each estimate the test fills the normals with
/// 0xFF bytes, so that a pixel the stage leaves unwritten shows.
/// - Every picture: an invalid pixel's normal is (0, 0, 0); every other normal is a unit vector
///   with n . p <= 0, toward the camera; and the stage agrees with the method computed here in
///   double from the same float points (the reference): the same pixels have a normal, and each
///   normal is the reference's within a tolerance. A pixel whose window holds a pair within
///   premise_margin_deg of the depth edge rule's threshold is left out of the reference, where
///   float and double may decide the pair differently.
/// - At both stream sizes, a tilted analytic plane, a plane 2 m away, an analytic sphere in the
///   picture's middle and one near its side: the normals agree with the analytic ones within an
///   angle of each size. Near the picture's side the sphere has normals with n.z > 0 that point
///   toward the camera. The tilted plane also at a size that is no multiple of the kernel's block.
/// - Against the prototype's estimate_normals(), the oracle, run on the stage's own valid points
///   of three frames of the tabletop walk: away from depth edges and creases the two agree as
///   lines within an angle. The same frames with windows of radius 1 and 2 agree with the
///   reference.
/// - Depth edges, too few points and the eigensolver's diagonal case, on frames built here: a step
///   in depth beyond the rule marks its two columns, one within it marks nothing, and beside a
///   large step the normals are the flat ones; a neighbourhood of exactly the fewest points has a
///   normal, one of one point less has none; a window whose covariance is exactly diagonal gives
///   the flat normal.
/// - Refusals as values, each leaving the normals as they were: empty organized normals, each
///   kind of parameters that cannot be used, empty organized points, organized points of another
///   size, and CUDA's last error left by an earlier call; and CUDA's refusal of the launch for a
///   picture whose rows need more blocks than a grid holds. CUDA's last error is clear after each.
///   The largest steepest incidence, 89 degrees, is accepted, one float step above it refused.
/// - Every buffer is requested before the pool closes, and the stage asks the closed pool for
///   nothing.

#include "camera/frame.hpp"
#include "check.hpp"
#include "geometry/organized_normals.hpp"
#include "geometry/organized_points.hpp"
#include "geometry/point_cloud.hpp"
#include "geometry/type.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "gpu_test_helpers.hpp"
#include "stage_checks.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace
{
using gca_test::angle_deg;
using gca_test::built_frame;
using gca_test::check;
using gca_test::check_max_error;
using gca_test::check_no_cuda_error;
using gca_test::check_refused;
using gca_test::check_within;
using gca_test::create_stream;
using gca_test::distance_from;
using gca_test::granted;
using gca_test::grey;
using gca_test::index_of;
using gca_test::is_zero;
using gca_test::large_description;
using gca_test::max_error;
using gca_test::median;
using gca_test::odd_description;
using gca_test::pair_side;
using gca_test::premise_margin_deg;
using gca_test::render;
using gca_test::rendered;
using gca_test::side_of;
using gca_test::small_description;
using gca_test::tabletop_box;
using gca_test::tabletop_plane;
using gca_test::tabletop_sphere;
using gca_test::tall_size;
using gca_test::to_double;

/// The stage's parameters: the method's starting values, a 7x7 window, 25 points, 75 degrees
/// [S: docs/methods/normals.md, "Input"].
constexpr gca::organized_normals_parameters test_parameters = {3, 25, 75.0f};

/// The other windows, each run once on the tabletop walk against the reference: radius 1 with the
/// fewest points the radius allows, 2R + 2, and radius 2 with the whole window and a steepest
/// incidence of 60 degrees [?: choices, the ends of min_points' range].
constexpr gca::organized_normals_parameters radius_1_parameters = {1, 4, 75.0f};
constexpr gca::organized_normals_parameters radius_2_parameters = {2, 25, 60.0f};

/// The tilted plane's depth range: 0.4005 m to 0.8005 m, each limit half a depth unit off a depth
/// value [?: a choice, the organized points' test's]. Inside it the plane is seen at less than the
/// rule's steepest incidence: the reference finds no pair near the rule and no depth edge on it
/// (checked).
constexpr gca::organized_points_parameters tilted_range = {0.4005f, 0.8005f};

/// The plane at 2 m's depth range: 1.0005 m to 3.0005 m [?: a choice, up to the large object
/// class's 3.0 m, src/scan/object_class.cpp].
constexpr gca::organized_points_parameters far_range = {1.0005f, 3.0005f};

/// The spheres' pictures' depth range [?: a choice, the organized points' test's].
constexpr gca::organized_points_parameters sphere_range = {0.3005f, 1.2005f};

/// The tabletop walk's depth range: the tabletop object class's 0.30 m to 1.0 m, each limit half a
/// unit inside [S: src/scan/object_class.cpp].
constexpr gca::organized_points_parameters box_range = {0.3005f, 0.9995f};

/// The range of the frames built here, which lie at 0.5 m to 0.6 m [?: a choice].
constexpr gca::organized_points_parameters built_range = {0.3005f, 0.9995f};

/// The oracle's search radius: 1 cm [?: a choice, a neighbourhood of a few dozen points at the
/// walk's 0.4 m to 0.6 m, as the stage's 7x7 window].
constexpr float oracle_radius_m = 0.01f;

/// | |n| - 1 |: 8 float steps, the eigensolver's normalized cross product rounded a few times
/// [S: float32 ulp]. The largest is 4.24e-7, 3.6 steps [M 2026-11-14].
constexpr double length_tolerance = 8 * gca_test::float_step;

/// The stage against the reference, the method in double over the same neighbourhood, in degrees:
/// float's rounding of the centred covariance and of the eigensolver. The largest over every
/// picture is 7.06e-4 degrees, on the sphere near the picture's side at 1280x800 [M 2026-11-14];
/// with the covariance left uncentred it is 1.40 to 1.94 degrees on the planes and the sphere at
/// 640x400 [M 2026-11-14: docs/methods/normals.md].
constexpr double reference_tolerance_deg = 2e-3;

/// The largest angle and the largest median angle of one surface's normals against the analytic
/// ones that pass, in degrees.
struct analytic_tolerance
{
    double largest_deg; ///< the largest angle that passes
    double median_deg;  ///< the largest median that passes
};

/// The tolerances against the analytic normals at 640x400, each a little above its measured
/// value, largest / median in degrees [M 2026-11-14, this test]: the depth's rounding to whole
/// units tilts a window's fit, most where the window is cut, and on the sphere the curvature
/// across the window adds to it. Tilted plane: 4.90 / 0.321, and 3.28 / 0.321 at 641x401; plane
/// at 2 m: 1.49 / 0.118; sphere in the middle: 5.72 / 0.829; sphere near the side: 8.29 / 1.01.
constexpr analytic_tolerance small_tilted = {6.0, 0.4};
constexpr analytic_tolerance small_far = {2.0, 0.15};
constexpr analytic_tolerance small_sphere = {7.0, 1.0};
constexpr analytic_tolerance small_side_sphere = {10.0, 1.3};

/// The tolerances at 1280x800, as at 640x400 [M 2026-11-14, this test]: a window of radius 3
/// spans half the surface there, over steps of the same height. Tilted plane: 10.6 / 0.775; plane
/// at 2 m: 4.53 / 0.250; sphere in the middle: 15.2 / 1.20; sphere near the side: 25.0 / 2.12.
constexpr analytic_tolerance large_tilted = {13.0, 1.0};
constexpr analytic_tolerance large_far = {5.5, 0.3};
constexpr analytic_tolerance large_sphere = {18.0, 1.5};
constexpr analytic_tolerance large_side_sphere = {30.0, 2.6};

/// The stage against the oracle as lines, in degrees. The largest is 11.9, at a pixel of the
/// plane at the picture's left border, whose window is cut to 4 columns (the test prints both
/// normals' angles to the plane's there), and the median 0.509 [M 2026-11-14].
constexpr analytic_tolerance oracle_tolerance = {15.0, 0.6};

/// A flat picture's normals against (0, 0, -1), in degrees: 8 float steps in radians [S: float32
/// ulp]. The largest is 0: a window of one depth has no spread along z, and the eigensolver gives
/// the axis exactly [M 2026-11-14].
constexpr double flat_tolerance_deg = 8 * gca_test::float_step * 180.0 / std::numbers::pi;

/// The plane seen from 2 m: 6 m x 6 m, larger than the view [?: a choice].
constexpr gca_test::synthetic_plane far_plane = {3.0, 3.0};

/// What the stage works on at one size: a GPU frame, organized points and organized normals of
/// that size, all from the pool, and the stream the work runs on.
struct stage_buffers
{
    gca::gpu_frame frame;           ///< the GPU frame the frames are uploaded into
    gca::organized_points points;   ///< the organized points they are deprojected into
    gca::organized_normals normals; ///< the organized normals estimated from those
    cudaStream_t stream = nullptr;  ///< the CUDA stream of the work
};

/// The points, marks and normals of a picture, downloaded.
struct downloaded
{
    gca::frame_size size{0, 0};     ///< the picture's size, in px
    std::vector<float3> points;     ///< in the camera frame, in m
    std::vector<std::uint8_t> mask; ///< 1 valid, 0 invalid
    std::vector<float3> normals;    ///< unit normals in the camera frame, or (0, 0, 0)
};

/// Fills organized normals with 0xFF bytes, a NaN in every coordinate, which the stage writes over
/// at every pixel it writes.
/// @param name [in] the case, named in the check line.
/// @param normals [in,out] the organized normals.
/// @param stream [in] the CUDA stream the fill is queued on.
void fill_with_0xff(const std::string &name, gca::organized_normals &normals, cudaStream_t stream)
{
    // The test writes into the stage's image on purpose; the stage gives it only to read.
    void *data = const_cast<float3 *>(normals.normals().data());
    check(cudaMemsetAsync(data, 0xFF, normals.normals().size_bytes(), stream) == cudaSuccess,
          (name + ": the normals filled with 0xFF").c_str());
}

/// Uploads a frame, deprojects it, fills the normals with 0xFF bytes and estimates them.
/// @param name [in] the case, named in the check lines.
/// @param host [in] the frame, of the stage's size.
/// @param stage [in,out] the GPU frame, points, normals and stream.
/// @param range [in] the depth range of the organized points.
/// @param parameters [in] the stage's parameters.
/// @return whether the upload, the deprojection and the estimate were done.
bool run_stage(const std::string &name, const gca::frame &host, stage_buffers &stage,
               const gca::organized_points_parameters &range,
               const gca::organized_normals_parameters &parameters)
{
    const gca::step_result up = stage.frame.upload(host, stage.stream);
    check(up.done, (name + ": upload, " + up.message).c_str());
    const gca::step_result made = stage.points.deproject(stage.frame, range, stage.stream);
    check(made.done, (name + ": deproject, " + made.message).c_str());
    fill_with_0xff(name, stage.normals, stage.stream);
    const gca::step_result estimated =
        stage.normals.estimate(stage.points, parameters, stage.stream);
    check(estimated.done, (name + ": estimate, " + estimated.message).c_str());
    return up.done && made.done && estimated.done;
}

/// Downloads the points, marks and normals: waits for the stream's work, then copies them.
/// @param name [in] the case, named in the check line.
/// @param stage [in] the points, normals and stream.
/// @return them; empty, with a failed check, when CUDA failed.
downloaded download(const std::string &name, const stage_buffers &stage)
{
    const gca::frame_size size = stage.normals.size();
    const std::size_t pixels = static_cast<std::size_t>(size.width) * size.height;
    downloaded result{size, std::vector<float3>(pixels), std::vector<std::uint8_t>(pixels),
                      std::vector<float3>(pixels)};
    const bool copied =
        cudaStreamSynchronize(stage.stream) == cudaSuccess &&
        cudaMemcpy(result.points.data(), stage.points.points().data(),
                   stage.points.points().size_bytes(), cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(result.mask.data(), stage.points.validity_mask().data(),
                   stage.points.validity_mask().size_bytes(),
                   cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(result.normals.data(), stage.normals.normals().data(),
                   stage.normals.normals().size_bytes(), cudaMemcpyDeviceToHost) == cudaSuccess;
    check(copied, (name + ": downloaded").c_str());
    return copied ? std::move(result) : downloaded{};
}

/// Checks the marks and orientation of every pixel: an invalid pixel's normal is (0, 0, 0), every
/// other normal is a unit vector with n . p <= 0.
/// @param name [in] the case, printed in each line.
/// @param got [in] the picture.
void check_every_normal(const std::string &name, const downloaded &got)
{
    int valid = 0;
    int with_normal = 0;
    int invalid_not_zero = 0;
    int facing_away = 0;
    max_error length;
    double most_toward = -1.0;
    for (std::size_t pixel = 0; pixel < got.mask.size(); pixel++)
    {
        const float3 &n = got.normals[pixel];
        if (got.mask[pixel] == 0)
        {
            invalid_not_zero += is_zero(n) ? 0 : 1;
            continue;
        }
        valid++;
        if (is_zero(n))
            continue;
        with_normal++;
        const Eigen::Vector3d normal = to_double(n);
        const Eigen::Vector3d point = to_double(got.points[pixel]);
        length.add(std::abs(normal.norm() - 1.0));
        const double toward = normal.dot(point) / point.norm();
        facing_away += toward > 0.0 || std::isnan(toward) ? 1 : 0;
        most_toward = std::max(most_toward, toward);
    }
    std::printf("%s: %d valid pixels, %d with a normal; the largest n . p / |p| is %.4f\n",
                name.c_str(), valid, with_normal, most_toward);
    check_within((name + ": invalid pixels whose normal is not (0, 0, 0)").c_str(),
                 invalid_not_zero, 0);
    check_within((name + ": normals pointing away from the camera, n . p > 0").c_str(), facing_away,
                 0);
    check_max_error(name + ": | |n| - 1 |", length, length_tolerance);
}

/// What the reference says of one valid pixel.
struct reference_normal
{
    bool decided = false;   ///< false when a pair of its window lies near the rule's threshold
    bool at_edge = false;   ///< an adjacent valid pixel lies across a depth edge
    int points = 0;         ///< its neighbourhood's points, its own included
    Eigen::Vector3d normal; ///< its normal, when it has one
};

/// Computes the method in double from a picture's float points, for one valid pixel
/// (docs/methods/normals.md): the neighbourhood, the depth edge, the mean, the centred covariance,
/// the eigenvector of the smallest eigenvalue (Eigen's solver), turned toward the camera.
/// @param got [in] the picture.
/// @param u [in] the pixel's column, in px.
/// @param v [in] the pixel's row, in px.
/// @param parameters [in] the stage's parameters.
/// @return the reference; not decided when a pair of the window lies within premise_margin_deg
///         of the rule's threshold.
reference_normal reference_at(const downloaded &got, std::uint32_t u, std::uint32_t v,
                              const gca::organized_normals_parameters &parameters)
{
    const int radius = static_cast<int>(parameters.radius_px);
    const Eigen::Vector3d centre = to_double(got.points[index_of(got.size, u, v)]);
    std::vector<Eigen::Vector3d> members;
    reference_normal result;
    for (int dv = -radius; dv <= radius; dv++)
    {
        for (int du = -radius; du <= radius; du++)
        {
            const long long column = static_cast<long long>(u) + du;
            const long long row = static_cast<long long>(v) + dv;
            if (column < 0 || row < 0 || column >= got.size.width || row >= got.size.height)
                continue;
            const std::size_t j = index_of(got.size, static_cast<std::uint32_t>(column),
                                           static_cast<std::uint32_t>(row));
            if (got.mask[j] == 0)
                continue;
            const Eigen::Vector3d point = to_double(got.points[j]);
            const pair_side side = du == 0 && dv == 0
                                       ? pair_side::one_surface
                                       : side_of(centre, point, parameters.max_incidence_deg);
            if (side == pair_side::near)
                return {};
            if (side == pair_side::edge)
            {
                result.at_edge = result.at_edge || (std::abs(du) <= 1 && std::abs(dv) <= 1);
                continue;
            }
            members.push_back(point);
        }
    }
    result.decided = true;
    result.points = static_cast<int>(members.size());
    if (result.at_edge || result.points < static_cast<int>(parameters.min_points))
        return result;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d &p : members)
        mean += p;
    mean /= static_cast<double>(members.size());
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d &p : members)
        covariance += (p - mean) * (p - mean).transpose();
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    result.normal = solver.eigenvectors().col(0);
    if (result.normal.dot(centre) > 0.0)
        result.normal = -result.normal;
    return result;
}

/// The comparison of a picture with the reference.
struct reference_counts
{
    int decided = 0;        ///< valid pixels the reference decides
    int undecided = 0;      ///< valid pixels left out: a pair near the rule's threshold
    int at_edge = 0;        ///< decided pixels at a depth edge
    int too_few = 0;        ///< decided pixels with too few points, not at an edge
    int presence_wrong = 0; ///< decided pixels whose normal is there where it is not, or not
    max_error angle_deg;    ///< the stage's normal against the reference's, in degrees
};

/// Compares every valid pixel of a picture with the reference.
/// @param name [in] the case, printed in each line.
/// @param got [in] the picture.
/// @param parameters [in] the stage's parameters the picture was made with.
/// @return the counts; checked: no decided pixel's presence wrong, every angle within
///         reference_tolerance_deg.
reference_counts check_reference(const std::string &name, const downloaded &got,
                                 const gca::organized_normals_parameters &parameters)
{
    reference_counts counts;
    for (std::uint32_t v = 0; v < got.size.height; v++)
    {
        for (std::uint32_t u = 0; u < got.size.width; u++)
        {
            const std::size_t pixel = index_of(got.size, u, v);
            if (got.mask[pixel] == 0)
                continue;
            const reference_normal reference = reference_at(got, u, v, parameters);
            if (!reference.decided)
            {
                counts.undecided++;
                continue;
            }
            counts.decided++;
            const bool expected =
                !reference.at_edge && reference.points >= static_cast<int>(parameters.min_points);
            counts.at_edge += reference.at_edge ? 1 : 0;
            counts.too_few += !reference.at_edge && !expected ? 1 : 0;
            const bool present = !is_zero(got.normals[pixel]);
            if (present != expected)
            {
                counts.presence_wrong++;
                continue;
            }
            if (expected)
                counts.angle_deg.add(angle_deg(to_double(got.normals[pixel]), reference.normal));
        }
    }
    std::printf("%s, reference: %d pixels decided (%d at a depth edge, %d with too few points), "
                "%d left out near the rule's threshold\n",
                name.c_str(), counts.decided, counts.at_edge, counts.too_few, counts.undecided);
    check_within((name + ", reference: pixels whose normal is there or not against it").c_str(),
                 counts.presence_wrong, 0);
    check_max_error(name + ", reference: angle to the normal in double, in degrees",
                    counts.angle_deg, reference_tolerance_deg);
    return counts;
}

/// Returns whether every valid pixel of a pixel's window, of test_parameters' radius, satisfies a
/// condition.
/// @param got [in] the picture.
/// @param u [in] the pixel's column, in px.
/// @param v [in] the pixel's row, in px.
/// @param holds [in] the condition on a valid pixel's index.
/// @return whether it holds at every valid pixel of the window, inside the picture.
template <typename Condition>
bool whole_window(const downloaded &got, std::uint32_t u, std::uint32_t v, Condition holds)
{
    const long long radius = test_parameters.radius_px;
    for (long long row = static_cast<long long>(v) - radius; row <= v + radius; row++)
    {
        for (long long column = static_cast<long long>(u) - radius; column <= u + radius; column++)
        {
            if (column < 0 || row < 0 || column >= got.size.width || row >= got.size.height)
                continue;
            const std::size_t j = index_of(got.size, static_cast<std::uint32_t>(column),
                                           static_cast<std::uint32_t>(row));
            if (got.mask[j] != 0 && !holds(j))
                return false;
        }
    }
    return true;
}

/// Returns whether a pixel's window, of test_parameters' radius, lies inside the picture and
/// holds valid pixels only: a window that is not cut.
/// @param got [in] the picture.
/// @param u [in] the pixel's column, in px.
/// @param v [in] the pixel's row, in px.
/// @return whether all (2R + 1)^2 pixels of the window are inside the picture and valid.
bool window_not_cut(const downloaded &got, std::uint32_t u, std::uint32_t v)
{
    const std::uint32_t radius = test_parameters.radius_px;
    if (u < radius || v < radius || u + radius >= got.size.width || v + radius >= got.size.height)
        return false;
    for (std::uint32_t row = v - radius; row <= v + radius; row++)
        for (std::uint32_t column = u - radius; column <= u + radius; column++)
            if (got.mask[index_of(got.size, column, row)] == 0)
                return false;
    return true;
}

/// Checks a picture's normals against analytic ones: the largest and median angle over every
/// normal compared, and prints them for windows that are not cut and for cut ones apart.
/// @param name [in] the case, printed in each line.
/// @param got [in] the picture.
/// @param analytic [in] gives pixel (u, v)'s analytic unit normal toward the camera, or nothing
///        for a pixel left out.
/// @param tolerance [in] the largest angle and median that pass, in degrees.
/// @return the normals compared.
template <typename Analytic>
std::size_t check_analytic(const std::string &name, const downloaded &got, Analytic analytic,
                           const analytic_tolerance &tolerance)
{
    max_error angles;
    std::vector<double> all;
    std::vector<double> whole;
    double largest_cut = 0.0;
    for (std::uint32_t v = 0; v < got.size.height; v++)
    {
        for (std::uint32_t u = 0; u < got.size.width; u++)
        {
            const std::size_t pixel = index_of(got.size, u, v);
            if (got.mask[pixel] == 0 || is_zero(got.normals[pixel]))
                continue;
            const std::optional<Eigen::Vector3d> expected = analytic(u, v);
            if (!expected)
                continue;
            const double angle = angle_deg(to_double(got.normals[pixel]), *expected);
            angles.add(angle);
            all.push_back(angle);
            if (window_not_cut(got, u, v))
                whole.push_back(angle);
            else
                largest_cut = std::max(largest_cut, angle);
        }
    }
    const std::size_t compared = all.size();
    const std::size_t whole_count = whole.size();
    const double largest_whole = whole.empty() ? 0.0 : *std::ranges::max_element(whole);
    std::printf("%s: %zu normals against the analytic ones; %zu of whole windows, largest %.3g "
                "and median %.3g degrees; %zu of cut windows, largest %.3g degrees\n",
                name.c_str(), compared, whole_count, largest_whole, median(std::move(whole)),
                compared - whole_count, largest_cut);
    check(compared > 0, (name + ": normals to compare with the analytic ones").c_str());
    check_max_error(name + ": angle to the analytic normal, in degrees", angles,
                    tolerance.largest_deg);
    check_within((name + ": median angle to the analytic normal, in degrees").c_str(),
                 median(std::move(all)), tolerance.median_deg);
    return compared;
}

/// A picture the stage made, and its comparison with the reference.
struct checked_picture
{
    downloaded got;             ///< the picture; empty when the stage did not run
    reference_counts reference; ///< its comparison with the reference
};

/// Runs the stage on one frame, downloads the picture, and checks every normal and the reference.
/// @param name [in] the case, printed in each line.
/// @param frame [in] the frame, of the stage's size.
/// @param range [in] the depth range of the organized points.
/// @param stage [in,out] the buffers and stream of the frame's size.
/// @param parameters [in] the stage's parameters.
/// @return the picture and the reference's counts; an empty picture when the stage did not run.
checked_picture run_and_check(const std::string &name, const gca::frame &frame,
                              const gca::organized_points_parameters &range, stage_buffers &stage,
                              const gca::organized_normals_parameters &parameters = test_parameters)
{
    if (!run_stage(name, frame, stage, range, parameters))
        return {};
    checked_picture result{download(name, stage), {}};
    if (result.got.points.empty())
        return {};
    check_every_normal(name, result.got);
    result.reference = check_reference(name, result.got, parameters);
    return result;
}

/// The analytic cases at one stream size: the stage of that size and what each surface is held
/// to.
struct stream_case
{
    const char *name;                          ///< the size, printed: "640x400"
    const gca::frame_description &description; ///< the size, the intrinsics and the depth unit
    stage_buffers &stage;                      ///< the stage of that size
    analytic_tolerance tilted;                 ///< the tilted plane's
    analytic_tolerance far;                    ///< the plane at 2 m's
    analytic_tolerance sphere;                 ///< the sphere's in the picture's middle
    analytic_tolerance side_sphere;            ///< the sphere's near the picture's side
    /// Whether the planes' pictures lie clear of the depth edge rule at this size: every pair of
    /// a window more than premise_margin_deg off its threshold and no pixel at a depth edge.
    bool planes_clear_of_rule;
};

/// Checks one plane's picture: every normal, the reference, the analytic normal, and, where the
/// size's pictures lie clear of the rule, that no pair lies near the rule and no pixel at a depth
/// edge, so that the reference decides every pixel.
/// @param name [in] the case, printed.
/// @param frame [in] the frame, of the stage's size.
/// @param range [in] the depth range.
/// @param stage [in,out] the stage of the frame's size.
/// @param pose [in] the frame's pose, camera frame to world frame; the plane is z = 0.
/// @param tolerance [in] the analytic tolerance.
/// @param clear_of_rule [in] whether the picture lies clear of the depth edge rule, checked.
void check_plane(const std::string &name, const gca::frame &frame,
                 const gca::organized_points_parameters &range, stage_buffers &stage,
                 const Eigen::Isometry3d &pose, const analytic_tolerance &tolerance,
                 bool clear_of_rule)
{
    const checked_picture made = run_and_check(name, frame, range, stage);
    if (made.got.points.empty())
        return;
    // The plane's normal, up from z = 0 in the world, in the camera frame; the camera is above it.
    const Eigen::Vector3d up = pose.linear().transpose() * Eigen::Vector3d::UnitZ();
    check_analytic(
        name, made.got, [&](std::uint32_t, std::uint32_t) { return std::optional(up); }, tolerance);
    if (clear_of_rule)
        check(made.reference.undecided == 0 && made.reference.at_edge == 0,
              (name + ": no pair near the rule and no depth edge").c_str());
}

/// Checks the planes at one stream size: the tilted plane, seen from 0.65 m looking down at about
/// 32 degrees and rolled by 10 degrees, and a plane 2 m away, seen at 20 degrees off its normal at
/// the picture's centre and rolled by 10 degrees [?: choices]; at the smaller size also the tilted
/// plane at a size that is no multiple of the kernel's block.
/// @param size [in,out] the stream size, its stage and tolerances.
/// @param odd_sized [in,out] the stage of odd_description's size, or nothing.
void check_planes(stream_case &size, stage_buffers *odd_sized)
{
    const std::optional<Eigen::Isometry3d> tilted_pose =
        gca_test::look_at({0.0, -0.55, 0.35}, {0.0, 0.0, 0.0}, 10.0);
    const double elevation_rad = 70.0 * std::numbers::pi / 180.0;
    const std::optional<Eigen::Isometry3d> far_pose =
        gca_test::look_at({0.0, -2.0 * std::cos(elevation_rad), 2.0 * std::sin(elevation_rad)},
                          {0.0, 0.0, 0.0}, 10.0);
    check(tilted_pose && far_pose, "the planes: look_at() gives both poses");
    if (!tilted_pose || !far_pose)
        return;
    const std::string at = std::string(" at ") + size.name;
    const rendered tilted =
        render({.plane = tabletop_plane, .trajectory = {*tilted_pose}}, size.description);
    const rendered far = render({.plane = far_plane, .trajectory = {*far_pose}}, size.description);
    if (!tilted.frames.empty())
        check_plane("tilted plane" + at, tilted.frames[0], tilted_range, size.stage, *tilted_pose,
                    size.tilted, size.planes_clear_of_rule);
    if (!far.frames.empty())
        check_plane("plane at 2 m" + at, far.frames[0], far_range, size.stage, *far_pose, size.far,
                    size.planes_clear_of_rule);
    if (odd_sized == nullptr)
        return;
    const rendered odd =
        render({.plane = tabletop_plane, .trajectory = {*tilted_pose}}, odd_description);
    if (!odd.frames.empty())
        check_plane("tilted plane at 641x401", odd.frames[0], tilted_range, *odd_sized,
                    *tilted_pose, size.tilted, size.planes_clear_of_rule);
}

/// Checks the sphere resting on the plane in one view: at pixels whose window lies on the sphere
/// alone the normals agree with the sphere's, taken where the pixel's ray meets it.
/// @param name [in] the case, printed.
/// @param pose [in] the view, camera frame to world frame.
/// @param size [in,out] the stream size, its stage.
/// @param tolerance [in] the analytic tolerance.
/// @return the sphere's normals with n.z > 0, which point away from the camera's axis.
int check_sphere_view(const std::string &name, const Eigen::Isometry3d &pose, stream_case &size,
                      const analytic_tolerance &tolerance)
{
    const rendered scene =
        render({.plane = tabletop_plane, .sphere = tabletop_sphere, .trajectory = {pose}},
               size.description);
    if (scene.frames.empty())
        return 0;
    const checked_picture made = run_and_check(name, scene.frames[0], sphere_range, size.stage);
    const downloaded &got = made.got;
    if (got.points.empty())
        return 0;
    const Eigen::Vector3d centre = pose.inverse() * tabletop_sphere.centre_m;
    const double radius = tabletop_sphere.radius_m;
    const auto on_sphere = [&](std::size_t j) {
        const Eigen::Vector3d world = pose * to_double(got.points[j]);
        return std::abs((world - tabletop_sphere.centre_m).norm() - radius) < std::abs(world.z());
    };
    const gca::intrinsics &k = size.description.intrinsics;
    const auto analytic = [&](std::uint32_t u, std::uint32_t v) -> std::optional<Eigen::Vector3d> {
        if (!whole_window(got, u, v, on_sphere))
            return std::nullopt;
        // The nearer meeting of the pixel's ray t r with the sphere |x - c| = radius.
        const Eigen::Vector3d ray((u - double{k.cx}) / k.fx, (v - double{k.cy}) / k.fy, 1.0);
        const double a = ray.squaredNorm();
        const double b = -2.0 * ray.dot(centre);
        const double c = centre.squaredNorm() - radius * radius;
        const double discriminant = b * b - 4.0 * a * c;
        if (discriminant < 0.0)
            return std::nullopt;
        const double t = (-b - std::sqrt(discriminant)) / (2.0 * a);
        return (t * ray - centre) / radius;
    };
    check_analytic(name, got, analytic, tolerance);
    int leaning_out = 0;
    for (std::size_t j = 0; j < got.normals.size(); j++)
        if (got.mask[j] != 0 && !is_zero(got.normals[j]) && on_sphere(j))
            leaning_out += got.normals[j].z > 0.0f ? 1 : 0;
    std::printf("%s: %d normals of the sphere with n.z > 0\n", name.c_str(), leaning_out);
    return leaning_out;
}

/// The turn of the view about the world's vertical, away from the sphere, that puts the sphere
/// near the picture's side: 35 degrees, so that its far side lies 40 degrees off the optical axis
/// [?: a choice; the horizontal half angle of view is 46 degrees, S: atan(320 / 306.18)].
constexpr double side_turn_deg = 35.0;

/// Checks the sphere at one stream size in two views from 0.46 m: looking at its centre, and
/// turned by side_turn_deg about the world's vertical through the camera, so that the sphere lies
/// near the picture's side. There, on its side away from the picture's centre, normals that point
/// toward the camera have n.z > 0, which a turn toward -z alone would get wrong; the test checks
/// that such normals occur.
/// @param size [in,out] the stream size, its stage and tolerances.
void check_spheres(stream_case &size)
{
    const Eigen::Vector3d camera(0.15, -0.40, 0.30);
    const std::optional<Eigen::Isometry3d> middle_pose =
        gca_test::look_at(camera, tabletop_sphere.centre_m, -7.0);
    const Eigen::Vector3d turned =
        Eigen::AngleAxisd(side_turn_deg * std::numbers::pi / 180.0, Eigen::Vector3d::UnitZ()) *
        (tabletop_sphere.centre_m - camera);
    const std::optional<Eigen::Isometry3d> side_pose =
        gca_test::look_at(camera, camera + turned, -7.0);
    check(middle_pose && side_pose, "the spheres: look_at() gives both poses");
    if (!middle_pose || !side_pose)
        return;
    const std::string at = std::string(" at ") + size.name;
    check_sphere_view("sphere in the middle" + at, *middle_pose, size, size.sphere);
    const int leaning_out =
        check_sphere_view("sphere near the side" + at, *side_pose, size, size.side_sphere);
    check(leaning_out > 0, ("sphere near the side" + at + ": normals with n.z > 0").c_str());
}

/// Runs the oracle, the prototype's estimate_normals(), on a picture's valid points.
/// @param got [in] the picture.
/// @param pixels [out] the index of the pixel of each oracle normal.
/// @return the oracle's normals, one for each valid pixel in index order; empty, with a failed
///         check, when it failed.
/// @note The oracle runs on the default stream and waits for the whole device.
std::vector<float3> oracle_normals(const downloaded &got, std::vector<std::size_t> &pixels)
{
    std::vector<gca::point_t> cloud;
    pixels.clear();
    for (std::size_t pixel = 0; pixel < got.mask.size(); pixel++)
    {
        if (got.mask[pixel] == 0)
            continue;
        gca::point_t point{};
        point.coordinates = got.points[pixel];
        point.color = gca::color3{0.5f, 0.5f, 0.5f};
        point.property = gca::point_property::inactive;
        cloud.push_back(point);
        pixels.push_back(pixel);
    }
    const std::shared_ptr<gca::point_cloud> oracle = gca::point_cloud::create_from_points(cloud);
    const bool estimated = oracle->estimate_normals(oracle_radius_m);
    check(estimated, "oracle: estimate_normals()");
    std::vector<float3> normals = estimated ? oracle->download_normals() : std::vector<float3>{};
    check(normals.size() == pixels.size(), "oracle: one normal for each point");
    return normals.size() == pixels.size() ? normals : std::vector<float3>{};
}

/// The comparison with the oracle over the frames.
struct oracle_counts
{
    std::size_t compared = 0; ///< pixels away from depth edges and creases with both normals
    max_error angle_deg;      ///< the axis angle between the two, in degrees
    std::vector<double> all;  ///< every axis angle, for the median
    /// At the largest axis angle: the stage's normal's and the oracle's axis angle to the normal
    /// of the pixel's known surface, in degrees, its pixel and whether its window is cut.
    double largest_stage_to_face_deg = 0.0;
    double largest_oracle_to_face_deg = 0.0;
    std::uint32_t largest_u = 0;
    std::uint32_t largest_v = 0;
    bool largest_cut = false;
};

/// Returns the angle between two lines.
/// @param a [in] the first line's direction, not zero.
/// @param b [in] the second line's direction, not zero.
/// @return the angle in degrees, 0 to 90.
double axis_angle_deg(const Eigen::Vector3d &a, const Eigen::Vector3d &b)
{
    const double angle = angle_deg(a, b);
    return std::min(angle, 180.0 - angle);
}

/// Compares a picture's normals with the oracle's as lines, at pixels away from depth edges and
/// creases: the pixel's window lies on the pixel's own known surface, and no other known surface
/// lies within the oracle's radius of its point.
/// @param got [in] the picture.
/// @param surfaces [in] the known surfaces.
/// @param pose [in] the picture's pose, camera frame to world frame.
/// @param counts [in,out] the comparisons are added.
void compare_oracle(const downloaded &got, const std::vector<gca_test::synthetic_surface> &surfaces,
                    const Eigen::Isometry3d &pose, oracle_counts &counts)
{
    std::vector<std::size_t> pixels;
    const std::vector<float3> oracle = oracle_normals(got, pixels);
    if (oracle.empty())
        return;
    // Each valid pixel's surface: the nearest known one.
    std::vector<int> label(got.mask.size(), -1);
    for (std::size_t pixel = 0; pixel < got.mask.size(); pixel++)
    {
        if (got.mask[pixel] == 0)
            continue;
        const Eigen::Vector3d world = pose * to_double(got.points[pixel]);
        double nearest = std::numeric_limits<double>::infinity();
        for (std::size_t s = 0; s < surfaces.size(); s++)
        {
            const double distance = distance_from(surfaces[s], world);
            if (distance < nearest)
            {
                nearest = distance;
                label[pixel] = static_cast<int>(s);
            }
        }
    }
    for (std::size_t i = 0; i < pixels.size(); i++)
    {
        const std::size_t pixel = pixels[i];
        const float3 &mine = got.normals[pixel];
        if (is_zero(mine) || is_zero(oracle[i]))
            continue;
        const auto u = static_cast<std::uint32_t>(pixel % got.size.width);
        const auto v = static_cast<std::uint32_t>(pixel / got.size.width);
        if (!whole_window(got, u, v, [&](std::size_t j) { return label[j] == label[pixel]; }))
            continue;
        const Eigen::Vector3d world = pose * to_double(got.points[pixel]);
        bool alone = true;
        for (std::size_t s = 0; s < surfaces.size(); s++)
            alone = alone && (static_cast<int>(s) == label[pixel] ||
                              distance_from(surfaces[s], world) > oracle_radius_m);
        if (!alone)
            continue;
        const double angle = axis_angle_deg(to_double(mine), to_double(oracle[i]));
        if (counts.compared == 0 || angle > counts.angle_deg.max)
        {
            const gca_test::synthetic_surface &face = surfaces[label[pixel]];
            const Eigen::Vector3d face_normal =
                pose.linear().transpose() * face.axis_u.cross(face.axis_v);
            counts.largest_stage_to_face_deg = axis_angle_deg(to_double(mine), face_normal);
            counts.largest_oracle_to_face_deg = axis_angle_deg(to_double(oracle[i]), face_normal);
            counts.largest_u = u;
            counts.largest_v = v;
            counts.largest_cut = !window_not_cut(got, u, v);
        }
        counts.compared++;
        counts.angle_deg.add(angle);
        counts.all.push_back(angle);
    }
}

/// Checks the stage with windows of radius 1 and 2 on one frame against the reference.
/// @param frame [in] the frame, of the stage's size.
/// @param stage [in,out] the stage of the frame's size.
void check_other_windows(const gca::frame &frame, stage_buffers &stage)
{
    struct window_case
    {
        const char *name;                             ///< the case, printed
        gca::organized_normals_parameters parameters; ///< its parameters
    };
    for (const window_case &c :
         {window_case{"tabletop walk, frame 0, radius 1 px, 4 points", radius_1_parameters},
          window_case{"tabletop walk, frame 0, radius 2 px, 25 points, 60 deg",
                      radius_2_parameters}})
    {
        const checked_picture made = run_and_check(c.name, frame, box_range, stage, c.parameters);
        check(made.reference.decided > 0 && made.reference.at_edge > 0,
              (std::string(c.name) + ": pixels decided, and pixels at a depth edge").c_str());
    }
}

/// Checks three frames of the tabletop walk, with depth edges where the box hides the plane: every
/// normal and the reference, at depth edges too, the comparison with the oracle away from them,
/// and the first frame with the other windows.
/// @param stage [in,out] the stage of the smaller stream size.
void check_tabletop_walk(stage_buffers &stage)
{
    // Three views of the tabletop walk, 120 degrees apart [?: a choice].
    gca_test::synthetic_walk walk = gca_test::tabletop_walk();
    walk.pose_count = 3;
    const std::optional<std::vector<Eigen::Isometry3d>> trajectory = gca_test::walk_around(walk);
    check(trajectory.has_value(), "tabletop walk: walk_around() gives the walk");
    if (!trajectory)
        return;
    const rendered scene =
        render({.plane = tabletop_plane, .box = tabletop_box, .trajectory = *trajectory},
               small_description);
    oracle_counts counts;
    int at_edge = 0;
    for (std::size_t i = 0; i < scene.frames.size(); i++)
    {
        const checked_picture made = run_and_check("tabletop walk, frame " + std::to_string(i),
                                                   scene.frames[i], box_range, stage);
        if (made.got.points.empty())
            continue;
        at_edge += made.reference.at_edge;
        compare_oracle(made.got, scene.surfaces, (*trajectory)[i], counts);
    }
    std::printf("tabletop walk: %d pixels at a depth edge, without a normal\n", at_edge);
    check(scene.frames.size() == 3 && at_edge > 0, "tabletop walk: three frames with depth edges");
    std::printf("tabletop walk, oracle: %zu normals away from depth edges and creases\n",
                counts.compared);
    std::printf("tabletop walk, oracle: at the largest axis angle, pixel (%u, %u), its window %s, "
                "the stage's normal is %.3g degrees off its face's and the oracle's %.3g\n",
                counts.largest_u, counts.largest_v, counts.largest_cut ? "cut" : "whole",
                counts.largest_stage_to_face_deg, counts.largest_oracle_to_face_deg);
    check(counts.compared > 0, "tabletop walk, oracle: normals to compare");
    check_max_error("tabletop walk, oracle: axis angle to the prototype's normal, in degrees",
                    counts.angle_deg, oracle_tolerance.largest_deg);
    check_within("tabletop walk, oracle: median axis angle to the prototype's normal, in degrees",
                 median(std::move(counts.all)), oracle_tolerance.median_deg);
    if (!scene.frames.empty())
        check_other_windows(scene.frames[0], stage);
}

/// The column of the built frames' step: near the principal point's column, 317.41 px, so that
/// the step is seen nearly along the line of sight [?: a choice].
constexpr std::uint32_t step_column = 318;

/// The step's rows that are checked: 179 to 219, within 20.2 px of the principal point's row,
/// 199.13 px, where the line of sight lies within 3.8 degrees of the optical axis [S: computed,
/// atan(20.13 / 306.14)]. Farther up and down it tilts, by up to 33 degrees, and the angle of a
/// diagonal pair across the step passes through the rule's threshold [S: computed].
constexpr std::uint32_t step_first_row = 179;
constexpr std::uint32_t step_last_row = 219;

/// The depth of the built frames before the step and of the cells, in depth units: 0.5 m
/// [?: a choice].
constexpr std::uint16_t built_depth = 500;

/// Checks steps in depth on flat built frames at the smaller stream size: depth built_depth left
/// of step_column and built_depth + step from it on. On the checked rows the segment across a step
/// of 4 units at 0.5 m lies 22.2 degrees from the line of sight, inside the depth edge rule's 15 by
/// 7.2, and across one of 7 units 13.2 degrees, beyond it by 1.8 [S: computed from the pixels'
/// points]; every other pair of the step pixels' windows lies more than premise_margin_deg off the
/// threshold (the reference's premise, checked). A step of 100 units lies beyond the rule for
/// every pair of a window across it, so every other normal of that frame is the flat one,
/// (0, 0, -1).
/// @param stage [in,out] the stage of the smaller stream size.
void check_steps(stage_buffers &stage)
{
    struct step_case
    {
        std::uint16_t step; ///< the step, in depth units
        bool marked;        ///< whether the step pixels have no normal
    };
    for (const step_case c : {step_case{4, false}, step_case{7, true}, step_case{100, true}})
    {
        const std::string name = "step of " + std::to_string(c.step) + " mm at 0.5 m";
        const gca::frame frame = built_frame(
            small_description,
            [&](std::uint32_t u, std::uint32_t) {
                return static_cast<std::uint16_t>(u < step_column ? built_depth
                                                                  : built_depth + c.step);
            },
            grey);
        const checked_picture made = run_and_check(name, frame, built_range, stage);
        const downloaded &got = made.got;
        if (got.points.empty())
            continue;
        int step_pixels = 0;
        int undecided = 0;
        int with_normal = 0;
        for (std::uint32_t v = step_first_row; v <= step_last_row; v++)
        {
            for (const std::uint32_t u : {step_column - 1, step_column})
            {
                step_pixels++;
                undecided += reference_at(got, u, v, test_parameters).decided ? 0 : 1;
                with_normal += is_zero(got.normals[index_of(got.size, u, v)]) ? 0 : 1;
            }
        }
        std::printf("%s: %d of %d step pixels of the checked rows have a normal\n", name.c_str(),
                    with_normal, step_pixels);
        check_within((name + ": step pixels near the rule's threshold").c_str(), undecided, 0);
        check(with_normal == (c.marked ? 0 : step_pixels),
              (name +
               (c.marked ? ": every step pixel has no normal" : ": every step pixel has a normal"))
                  .c_str());
        if (c.step != 100)
            continue;
        max_error flat;
        for (std::size_t pixel = 0; pixel < got.normals.size(); pixel++)
            if (!is_zero(got.normals[pixel]))
                flat.add(angle_deg(to_double(got.normals[pixel]), -Eigen::Vector3d::UnitZ()));
        check_max_error(name + ": angle to (0, 0, -1) beside the step, in degrees", flat,
                        flat_tolerance_deg);
    }
}

/// The side of a cell of the few-points frame: the 7x7 window of test_parameters, so that a cell's
/// centre sees its own cell and nothing else.
constexpr std::uint32_t cell_side = 2 * test_parameters.radius_px + 1;

/// Checks neighbourhoods of exactly the fewest points and of one less, on a flat built frame tiled
/// by cells of the window's size. Cell (i, j) holds depth at its first k pixels in raster order,
/// the centre among them, and none at the rest: k = min_points, min_points - 1 or 1, by (i + j)
/// mod 3. The centre of a cell of min_points has a normal, (0, 0, -1); one of fewer has none.
/// @param stage [in,out] the stage of the smaller stream size.
void check_few_points(stage_buffers &stage)
{
    const std::uint32_t fewest = test_parameters.min_points;
    const std::array<std::uint32_t, 3> kinds = {fewest, fewest - 1, 1};
    const auto points_of_cell = [&](std::uint32_t u, std::uint32_t v) {
        return kinds[(u / cell_side + v / cell_side) % kinds.size()];
    };
    const std::uint32_t centre_in_cell = cell_side / 2 * cell_side + cell_side / 2;
    const gca::frame frame = built_frame(
        small_description,
        [&](std::uint32_t u, std::uint32_t v) -> std::uint16_t {
            const std::uint32_t k = points_of_cell(u, v);
            const std::uint32_t in_cell = (v % cell_side) * cell_side + u % cell_side;
            // The centre comes last of the first min_points pixels: a cell keeps the k pixels up
            // to its centre, and a cell of one its centre alone.
            const bool kept = in_cell <= centre_in_cell && in_cell + k > centre_in_cell;
            return kept ? built_depth : 0;
        },
        grey);
    const std::string name = "cells of 25, 24 and 1 points";
    const checked_picture made = run_and_check(name, frame, built_range, stage);
    const downloaded &got = made.got;
    if (got.points.empty())
        return;
    std::array<int, 3> centres{};
    std::array<int, 3> with_normal{};
    max_error flat;
    for (std::uint32_t v = cell_side / 2; v + cell_side / 2 < got.size.height; v += cell_side)
    {
        for (std::uint32_t u = cell_side / 2; u + cell_side / 2 < got.size.width; u += cell_side)
        {
            const std::size_t kind = (u / cell_side + v / cell_side) % kinds.size();
            const float3 &n = got.normals[index_of(got.size, u, v)];
            centres[kind]++;
            with_normal[kind] += is_zero(n) ? 0 : 1;
            if (!is_zero(n))
                flat.add(angle_deg(to_double(n), -Eigen::Vector3d::UnitZ()));
        }
    }
    std::printf("%s: of the cells' centres %d of %d, %d of %d and %d of %d have a normal\n",
                name.c_str(), with_normal[0], centres[0], with_normal[1], centres[1],
                with_normal[2], centres[2]);
    check(centres[0] > 0 && with_normal[0] == centres[0],
          (name + ": every centre of 25 points has a normal").c_str());
    check(centres[1] > 0 && with_normal[1] == 0,
          (name + ": no centre of 24 points has a normal").c_str());
    check(centres[2] > 0 && with_normal[2] == 0,
          (name + ": no centre of 1 point has a normal").c_str());
    check_max_error(name + ": angle to (0, 0, -1), in degrees", flat, flat_tolerance_deg);
}

/// A frame of the smaller stream size whose points are exact: fx = fy = 512 px and a principal
/// point on a whole pixel, (320, 200), with a depth unit of 2^-10 m, so that at a depth of 512
/// units, 0.5 m, every coordinate is a whole multiple of 2^-10 m [S: (u - cx) z / fx with z = 2^-1
/// and fx = 2^9]. At the principal point the window is symmetric, its mean is the pixel's point,
/// and every sum of the centred covariance is exact: x and y spread equally, xy, xz, yz and zz are
/// 0, and the covariance is exactly diagonal [S: computed] [?: a choice of intrinsics, not the
/// camera's].
constexpr gca::frame_description exact_description = {
    {640, 400}, {512.0f, 512.0f, 320.0f, 200.0f}, 1.0f / 1024.0f};

/// Checks the eigensolver's diagonal case: on a flat frame of exact points the pixel at the
/// principal point has an exactly diagonal covariance, whose eigenvalues the solver returns in the
/// diagonal's order, x's and y's spread before z's 0 (symmetric_eigen_3x3.cuh); its normal and
/// every other is (0, 0, -1).
/// @param stage [in,out] the stage of the smaller stream size.
void check_diagonal_covariance(stage_buffers &stage)
{
    const std::string name = "flat frame of exact points";
    const gca::frame frame = built_frame(
        exact_description, [](std::uint32_t, std::uint32_t) -> std::uint16_t { return 512; }, grey);
    const checked_picture made = run_and_check(name, frame, built_range, stage);
    const downloaded &got = made.got;
    if (got.points.empty())
        return;
    const float3 &at_principal_point = got.normals[index_of(got.size, 320, 200)];
    check(!is_zero(at_principal_point), (name + ": a normal at the principal point").c_str());
    check_within((name + ": the principal point's angle to (0, 0, -1), in degrees").c_str(),
                 angle_deg(to_double(at_principal_point), -Eigen::Vector3d::UnitZ()),
                 flat_tolerance_deg);
    max_error flat;
    for (const float3 &n : got.normals)
        if (!is_zero(n))
            flat.add(angle_deg(to_double(n), -Eigen::Vector3d::UnitZ()));
    check_max_error(name + ": angle to (0, 0, -1), in degrees", flat, flat_tolerance_deg);
}

/// The GPU memory the refusals use, all requested before the pool closes.
struct refusal_buffers
{
    gca::organized_points other;         ///< organized points of the larger size, never made
    gca::organized_points tall;          ///< organized points of tall_size, never made
    gca::organized_normals tall_normals; ///< organized normals of tall_size
};

/// Parameters the stage refuses, and the case's name.
struct unusable_parameters
{
    const char *what;                             ///< the case
    gca::organized_normals_parameters parameters; ///< the parameters
};

/// Returns each kind of parameters the stage refuses, each alone [?: a choice of values]: a
/// radius of 0 and of 4; the fewest points one below 2R + 2 and one above the window; an
/// incidence of 0, of 90, one float step above the largest, 89 degrees, not a number and
/// infinite.
/// @return the cases.
std::array<unusable_parameters, 9> unusable()
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinite = std::numeric_limits<float>::infinity();
    const float above_largest = std::nextafter(gca::max_steepest_incidence_deg, infinite);
    return {{{"a radius of 0 px", {0, 4, 75.0f}},
             {"a radius of 4 px", {4, 25, 75.0f}},
             {"7 points at a radius of 3 px", {3, 7, 75.0f}},
             {"50 points at a radius of 3 px", {3, 50, 75.0f}},
             {"an incidence of 0 deg", {3, 25, 0.0f}},
             {"an incidence of 90 deg", {3, 25, 90.0f}},
             {"an incidence one float step above 89 deg", {3, 25, above_largest}},
             {"an incidence that is not a number", {3, 25, nan}},
             {"an incidence that is infinite", {3, 25, infinite}}}};
}

/// Checks the refusals that leave the normals as they were: empty organized normals, each kind of
/// parameters that cannot be used, empty organized points, organized points of another size, and
/// CUDA's last error left by an earlier call; then that the largest steepest incidence, 89
/// degrees, is accepted.
/// @param stage [in,out] the stage of the smaller stream size, its normals estimated from its
///        points.
/// @param buffers [in,out] the other GPU memory the refusals use.
void check_refusals_leave_normals(stage_buffers &stage, refusal_buffers &buffers)
{
    gca::organized_normals &out = stage.normals;
    const cudaStream_t stream = stage.stream;
    const downloaded before = download("refusals, before", stage);
    check_refused("empty organized normals",
                  gca::organized_normals{}.estimate(stage.points, test_parameters, stream),
                  "the organized normals hold no image");
    for (const unusable_parameters &c : unusable())
        check_refused(c.what, out.estimate(stage.points, c.parameters, stream), "is not");
    check_refused("empty organized points",
                  out.estimate(gca::organized_points{}, test_parameters, stream),
                  "the organized points hold no images");
    check_refused("organized points of another size",
                  out.estimate(buffers.other, test_parameters, stream),
                  "the organized points are 1280x800 px, the organized normals 640x400 px");
    // cudaMalloc's refusal of 2^52 bytes, 4 PiB on a board of 7.4 GiB [S: docs/HARDWARE.md],
    // leaves its error as CUDA's last error.
    void *never_granted = nullptr;
    check(cudaMalloc(&never_granted, std::size_t{1} << 52) != cudaSuccess,
          "cudaMalloc refuses 2^52 bytes");
    check_refused("CUDA's last error left by an earlier call",
                  out.estimate(stage.points, test_parameters, stream), "left by an earlier call");
    const downloaded after = download("refusals, after", stage);
    const bool same = !before.normals.empty() && before.normals.size() == after.normals.size() &&
                      std::memcmp(before.normals.data(), after.normals.data(),
                                  before.normals.size() * sizeof(float3)) == 0;
    check(same, "refusals: the organized normals left as they were, bit for bit");
    check_no_cuda_error("refusals");
    const gca::step_result largest =
        out.estimate(stage.points, {3, 25, gca::max_steepest_incidence_deg}, stream);
    check(largest.done, ("an incidence of 89 deg: estimated, " + largest.message).c_str());
}

/// Checks CUDA's refusal of the kernel's launch for a picture of 1 x 600000 px, whose rows need
/// 75000 blocks along y where a grid holds 65535: a failure with CUDA's error, which is cleared.
/// @param buffers [in,out] the organized points and normals of tall_size.
/// @param stream [in] the CUDA stream the work runs on.
void check_refused_launch(refusal_buffers &buffers, cudaStream_t stream)
{
    check_refused("a picture of 1x600000 px, 75000 blocks along y",
                  buffers.tall_normals.estimate(buffers.tall, test_parameters, stream),
                  "cudaErrorInvalidConfiguration");
    check_no_cuda_error("the refused launch");
}

/// Requests the buffers of one size from the pool.
/// @param pool [in,out] the pool, open.
/// @param size [in] the size, in px.
/// @param stream [in] the CUDA stream the work runs on.
/// @return the stage's buffers; empty ones, with a failed check, where a request was refused.
stage_buffers request_stage(gca::gpu_pool &pool, const gca::frame_size &size, cudaStream_t stream)
{
    return {granted("a GPU frame", gca::gpu_frame::request(pool, size)),
            granted("organized points", gca::organized_points::request(pool, size)),
            granted("organized normals", gca::organized_normals::request(pool, size)), stream};
}

/// Requests every buffer, closes the pool, runs the checks and checks that the stage asked the
/// closed pool for nothing.
/// @param stream [in] the CUDA stream the work runs on.
void check_with_closed_pool(cudaStream_t stream)
{
    gca::gpu_pool pool;
    stage_buffers small = request_stage(pool, small_description.size, stream);
    stage_buffers odd_sized = request_stage(pool, odd_description.size, stream);
    stage_buffers large = request_stage(pool, large_description.size, stream);
    refusal_buffers buffers{
        granted("organized points", gca::organized_points::request(pool, large_description.size)),
        granted("organized points", gca::organized_points::request(pool, tall_size)),
        granted("organized normals", gca::organized_normals::request(pool, tall_size))};
    pool.close();
    // At 1280x800 the tilted plane's far corners, seen at up to 72.8 degrees [S: computed from the
    // pose], lie near the rule's threshold once the depth is rounded: 8 of its pixels lie at a
    // depth edge and 788 near the threshold [M 2026-11-14, this test]; at 640x400 none
    // [M 2026-11-14].
    stream_case small_case = {"640x400",    small_description, small, small_tilted, small_far,
                              small_sphere, small_side_sphere, true};
    stream_case large_case = {"1280x800",   large_description, large, large_tilted, large_far,
                              large_sphere, large_side_sphere, false};
    check_planes(small_case, &odd_sized);
    check_planes(large_case, nullptr);
    check_spheres(small_case);
    check_spheres(large_case);
    check_tabletop_walk(small);
    check_steps(small);
    check_few_points(small);
    check_diagonal_covariance(small);
    check_refusals_leave_normals(small, buffers);
    check_refused_launch(buffers, stream);
    check_within("requests to the closed pool", static_cast<double>(pool.refused_while_closed()),
                 0);
}
} // namespace

/// Runs the checks of the organized normals and prints the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    cudaStream_t stream = nullptr;
    if (!create_stream(stream))
        return gca_test::check_summary("test_organized_normals");
    check_with_closed_pool(stream);
    check(cudaStreamDestroy(stream) == cudaSuccess, "the CUDA stream destroyed");
    return gca_test::check_summary("test_organized_normals");
}
