// SPDX-License-Identifier: 0BSD
/// @file test_tracking.cpp
/// Tracking (tracking/tracker.hpp, docs/methods/tracking.md) through its public functions, on
/// frames of the synthetic source at 640x400 with the camera's intrinsics and depth unit, with the
/// tabletop object class's values (its depth range 0.25 m to 0.6 m, the live model at 2 mm,
/// src/scan/object_class.cpp) but for e_track, the offline tests' noise-free value of 1.0 mm, and
/// a sparse TSDF of 273 MB; every tracker's pool is closed before its first frame:
/// - the check on results built by hand: each condition at its threshold and past it, in the
///   check's order, every end of the registration, the step after a long loss and none for the
///   first frame;
/// - the refusals: each kind of parameter out of its range, a size of 0, a closed pool (which
///   checks the live model's parameters without device memory), an empty tracker, a frame of
///   another size and a frame not later than the one before;
/// - a walk once around the tabletop box, 240 poses: every frame tracking, the trajectory within a
///   measured tolerance of the known one, the closed pool asked for nothing; then the walk's last
///   frame again, registered against the cast from its own tracked pose, within a measured
///   tolerance of the identity;
/// - frames without depth during the walk: tracking loss from the first, too few
///   correspondences, the model's points the same before and after, tracking again after;
/// - a view sliding along one grey plane: tracking loss from the first frame, degenerate, never
///   a pose, no point in the model;
/// - the tabletop object's sphere moved by 2 cm between a fused frame and the next: tracking loss
///   by the inlier RMSE at 1.0 mm, the model's points the same; tracking at the tabletop object's
///   own e_track, 5 mm, the cost the user accepted, printed.
/// The test prints its time; a scan-level test ends within 60 s on the board (the stage brief, §6).

#include "check.hpp"
#include "frame_copy.hpp"
#include "fusion/sparse_tsdf.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu_test_helpers.hpp"
#include "registration/registration.hpp"
#include "registration_cases.hpp"
#include "scan/object_class.hpp"
#include "scan/stage_parameters.hpp"
#include "sparse_tsdf_helpers.hpp"
#include "sparse_tsdf_scenes.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "tracking/tracker.hpp"
#include "transform_checks.hpp"
#include "types/camera_pose.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_within;
using gca_test::granted;
using gca_test::small_description;

/// e_track of the offline tests: 1.0 mm, a value of noise-free frames, whose depth carries only
/// its rounding to 1 mm (the user's decision of 2026-11-15): it lies between the walk's largest
/// inlier RMSE, 0.37 mm, and the moved sphere's, 3.0 mm [M 2026-11-15, this test], so that the
/// RMSE's condition is shown to refuse a part of the scene that moved; the object classes' own
/// e_track, 5 mm and 30 mm, is the datasheet's bound on real frames' noise.
constexpr double noise_free_rmse_m = 0.001;

/// The poses of the walk once around the tabletop box: 240, 1.5 degrees of azimuth apart
/// [?: a choice, a hand's turn in a frame time at 30 fps, 45 degrees a second].
constexpr std::size_t walk_poses = 240;

/// The frames of the walk the depth gap's case tracks, the first 12, and its gap: frames 6 to 8
/// without depth [?: a choice].
constexpr std::size_t gap_case_frames = 12;
constexpr gca_test::synthetic_depth_gap depth_gap = {6, 3};

/// The room for the model's points in an extraction: 1000000, above the walk's model
/// [?: a choice].
constexpr std::uint32_t point_room = 1'000'000;

/// The tabletop object class's set (src/scan/object_class.cpp).
const gca::object_class_parameters tabletop_set =
    gca::parameters_of(gca::object_class::tabletop_object);

/// Gives the test's tracking parameters: the tabletop object class's (scan/stage_parameters.hpp),
/// the pyramid's depth range each limit half a depth unit inside, so that no depth value lies on a
/// limit, and e_track the test's value.
/// @param max_inlier_rmse_m [in] e_track, in m.
/// @return the parameters.
gca::tracking_parameters test_parameters(double max_inlier_rmse_m = noise_free_rmse_m)
{
    gca::tracking_parameters p = gca::tracking_parameters_of(tabletop_set);
    p.pyramid.range = gca_test::inside({tabletop_set.min_depth_m, tabletop_set.max_depth_m});
    p.thresholds.max_inlier_rmse_m = max_inlier_rmse_m;
    return p;
}

/// Gives the live model's parameters: the tabletop object class's (scan/stage_parameters.hpp), its
/// live voxel size, truncation and far depth, in a sparse TSDF of 273 MB (gca_test::case_budget).
/// @return the parameters.
gca::sparse_tsdf_parameters live_model_parameters()
{
    gca::sparse_tsdf_parameters p =
        gca::sparse_tsdf_parameters_of(tabletop_set, gca::fused_model::live);
    p.memory_budget_bytes = gca_test::case_budget;
    return p;
}

/// The live model's parameters (live_model_parameters()).
const gca::sparse_tsdf_parameters live_model = live_model_parameters();

/// Returns the name of a check's outcome, for the printed lines.
/// @param outcome [in] the outcome.
/// @return its enumerator's name.
const char *outcome_name(gca::check_outcome outcome)
{
    switch (outcome)
    {
    case gca::check_outcome::passed:
        return "passed";
    case gca::check_outcome::registration_end:
        return "registration end";
    case gca::check_outcome::fitness:
        return "fitness";
    case gca::check_outcome::inlier_rmse:
        return "inlier RMSE";
    case gca::check_outcome::step:
        return "step";
    }
    return "unknown";
}

/// Returns the name of a tracking state, for the printed lines.
/// @param state [in] the state.
/// @return its enumerator's name.
const char *state_name(gca::tracking_state state)
{
    switch (state)
    {
    case gca::tracking_state::tracking:
        return "tracking";
    case gca::tracking_state::lost:
        return "lost";
    case gca::tracking_state::failed:
        return "failed";
    }
    return "unknown";
}

/// Prints a frame's tracking result.
/// @param what [in] the frame.
/// @param result [in] the result.
void print_tracked(const std::string &what, const gca::tracking_result &result)
{
    const gca::registration_result &r = result.registration;
    std::printf("%s: %s (%s), %s after %u systems, fitness %.4f, inlier RMSE %.3g mm, lambda_1 "
                "%.3g, last step %.3g rad and %.3g m, step %.3g deg and %.3g mm%s%s\n",
                what.c_str(), state_name(result.state), outcome_name(result.outcome),
                gca_test::end_name(r.end), r.iterations, r.fitness, 1e3 * r.inlier_rmse_m,
                r.eigenvalues[0], r.last_step_turn_rad, r.last_step_shift_m, result.step.turn_deg,
                1e3 * result.step.shift_m, result.message.empty() ? "" : ", ",
                result.message.c_str());
}

/// Gives a pose as a rigid transform.
/// @param pose [in] the pose.
/// @return the transform p -> orientation p + position.
Eigen::Isometry3d transform_of(const gca::camera_pose &pose)
{
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.linear() = pose.orientation.normalized().toRotationMatrix();
    transform.translation() = pose.position_m;
    return transform;
}

/// A tracker with its own pool, closed, and a buffer for the model's points.
struct tracking_case
{
    gca::gpu_pool pool;         ///< the pool, closed once the tracker and the buffer are granted
    gca::tracker tracker;       ///< the tracker
    gca::surface_points points; ///< the buffer of the model's points
    std::string name;           ///< the case, named in the check lines
};

/// Requests a case's tracker and buffer, and closes the pool.
/// @param c [in,out] the case, its pool open.
/// @param parameters [in] tracking's parameters.
/// @param stream [in] the stream.
void open_case(tracking_case &c, const gca::tracking_parameters &parameters, cudaStream_t stream)
{
    c.tracker =
        granted(c.name + ": the tracker", gca::tracker::request(c.pool, small_description.size,
                                                                parameters, live_model, stream));
    c.points =
        granted(c.name + ": the points' buffer", gca::surface_points::request(c.pool, point_room));
    c.pool.close();
}

/// Gives the model's points as a sorted list of their bits, so that two extractions compare as
/// sets, bit for bit (sparse_tsdf_helpers.hpp, sorted_keys()).
/// @param c [in,out] the case.
/// @param stream [in] the stream.
/// @param what [in] the moment, named in the check lines.
/// @return the points' keys, sorted.
std::vector<gca_test::point_key> model_points(tracking_case &c, cudaStream_t stream,
                                              const std::string &what)
{
    return gca_test::sorted_keys(
        gca_test::extract_points(c.tracker.live_model(), c.points, stream, c.name + ", " + what));
}

/// Renders the frames of a source's options.
/// @param options [in] the options.
/// @return the frames, one for each pose; fewer, with a failed check, when the source fails.
std::vector<gca::frame> frames_of(const gca_test::synthetic_source_options &options)
{
    gca_test::synthetic_source source(options);
    check(source.open(options.description.size).opened, "the synthetic source opened");
    std::vector<gca::frame> frames(options.scene.trajectory.size());
    for (gca::frame &frame : frames)
        check(source.next(frame).status == gca::next_status::frame,
              "the synthetic source gave a frame");
    return frames;
}

/// The walk once around the tabletop box: the tabletop scene's box on its plane, seen along
/// walk_poses poses of tabletop_walk().
/// @return the scene with its trajectory; none, with a failed check, when walk_around() refuses.
std::optional<gca_test::synthetic_scene> walk_scene()
{
    gca_test::synthetic_walk walk = gca_test::tabletop_walk();
    walk.pose_count = walk_poses;
    const std::optional<std::vector<Eigen::Isometry3d>> poses = gca_test::walk_around(walk);
    check(poses.has_value(), "walk: the poses");
    if (!poses)
        return std::nullopt;
    return gca_test::synthetic_scene{
        .plane = gca_test::tabletop_plane, .box = gca_test::tabletop_box, .trajectory = *poses};
}

// ------------------------------------------------------------------------------------ the check

/// A result built by hand for the check.
struct check_case
{
    std::string what;                 ///< the case
    gca::registration_result result;  ///< the registration's result
    std::optional<double> interval_s; ///< Δt, or none
    gca::check_outcome expected;      ///< the outcome it must give
};

/// The check on results built by hand, each against the thresholds f_track 0.5, e_track 1.0 mm,
/// ω_max 90 °/s and v_max 1 m/s, Δt one frame time unless said, so that the step's limits are 3°
/// and 33.3 mm: a result within every threshold passes, converged or with its iterations used up;
/// every other end fails the first condition; each value at its threshold passes, past it fails
/// its condition, and one that is not a number fails it; the conditions are judged in order; after
/// 2 s of loss the limits are 180° and 2.0 m; with no last tracked frame no step is judged. The
/// step of a transform is its angle and its shift's length.
void check_the_check()
{
    const gca::tracking_thresholds thresholds = {0.5, 0.001, 90.0, 1.0};
    const double frame_s = 1.0 / 30.0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const Eigen::Vector3d axis(0.3, -0.8, 0.5);
    gca::registration_result good;
    good.end = gca::registration_end::converged;
    good.fitness = 0.9;
    good.inlier_rmse_m = 0.0005;
    good.transform = gca_test::make_transform(axis, 1.0, Eigen::Vector3d(0.006, -0.004, 0.008));
    const auto with = [&good](auto change) {
        gca::registration_result r = good;
        change(r);
        return r;
    };
    const auto turned = [&axis](double angle_deg, double shift_m) {
        return gca_test::make_transform(axis, angle_deg, Eigen::Vector3d(shift_m, 0.0, 0.0));
    };
    using gca::check_outcome;
    using gca::registration_end;
    const std::vector<check_case> cases = {
        {"within every threshold, converged", good, frame_s, check_outcome::passed},
        {"iterations used up", with([](auto &r) { r.end = registration_end::iterations_used_up; }),
         frame_s, check_outcome::passed},
        {"too few correspondences",
         with([](auto &r) { r.end = registration_end::few_correspondences; }), frame_s,
         check_outcome::registration_end},
        {"degenerate", with([](auto &r) { r.end = registration_end::degenerate; }), frame_s,
         check_outcome::registration_end},
        {"refused", with([](auto &r) { r.end = registration_end::refused; }), frame_s,
         check_outcome::registration_end},
        {"failed", with([](auto &r) { r.end = registration_end::failed; }), frame_s,
         check_outcome::registration_end},
        {"fitness at f_track", with([](auto &r) { r.fitness = 0.5; }), frame_s,
         check_outcome::passed},
        {"fitness below f_track", with([](auto &r) { r.fitness = 0.4999; }), frame_s,
         check_outcome::fitness},
        {"fitness not a number", with([nan](auto &r) { r.fitness = nan; }), frame_s,
         check_outcome::fitness},
        {"RMSE at e_track", with([](auto &r) { r.inlier_rmse_m = 0.001; }), frame_s,
         check_outcome::passed},
        {"RMSE above e_track", with([](auto &r) { r.inlier_rmse_m = 0.0011; }), frame_s,
         check_outcome::inlier_rmse},
        {"RMSE not a number", with([nan](auto &r) { r.inlier_rmse_m = nan; }), frame_s,
         check_outcome::inlier_rmse},
        {"a turn of 2.99 deg in a frame time",
         with([&turned](auto &r) { r.transform = turned(2.99, 0.0); }), frame_s,
         check_outcome::passed},
        {"a turn of 3.01 deg in a frame time",
         with([&turned](auto &r) { r.transform = turned(3.01, 0.0); }), frame_s,
         check_outcome::step},
        {"a shift of 33.2 mm in a frame time",
         with([&turned](auto &r) { r.transform = turned(0.0, 0.0332); }), frame_s,
         check_outcome::passed},
        {"a shift of 33.4 mm in a frame time",
         with([&turned](auto &r) { r.transform = turned(0.0, 0.0334); }), frame_s,
         check_outcome::step},
        {"a turn of 179 deg and a shift of 1.99 m after 2 s",
         with([&turned](auto &r) { r.transform = turned(179.0, 1.99); }), 2.0,
         check_outcome::passed},
        {"a shift of 2.01 m after 2 s",
         with([&turned](auto &r) { r.transform = turned(0.0, 2.01); }), 2.0, check_outcome::step},
        {"a turn of 90 deg with no last tracked frame",
         with([&turned](auto &r) { r.transform = turned(90.0, 0.5); }), std::nullopt,
         check_outcome::passed},
        {"degenerate with a low fitness, a high RMSE and a large step", with([&turned](auto &r) {
             r.end = registration_end::degenerate;
             r.fitness = 0.1;
             r.inlier_rmse_m = 0.01;
             r.transform = turned(10.0, 0.1);
         }),
         frame_s, check_outcome::registration_end},
        {"a low fitness, a high RMSE and a large step", with([&turned](auto &r) {
             r.fitness = 0.1;
             r.inlier_rmse_m = 0.01;
             r.transform = turned(10.0, 0.1);
         }),
         frame_s, check_outcome::fitness},
        {"a high RMSE and a large step", with([&turned](auto &r) {
             r.inlier_rmse_m = 0.01;
             r.transform = turned(10.0, 0.1);
         }),
         frame_s, check_outcome::inlier_rmse},
    };
    for (const check_case &c : cases)
    {
        const gca::check_outcome outcome =
            gca::check_registration(c.result, c.interval_s, thresholds);
        std::printf("the check, %s: %s\n", c.what.c_str(), outcome_name(outcome));
        check(outcome == c.expected,
              ("the check, " + c.what + ": " + outcome_name(c.expected)).c_str());
    }
    const gca::tracking_step step =
        gca::step_of(gca_test::make_transform(axis, 30.0, Eigen::Vector3d(0.3, 0.4, 0.0)));
    check_within("the step of a turn of 30 deg: its angle less 30 (deg)",
                 std::abs(step.turn_deg - 30.0), 1e-9);
    check_within("the step of a shift of 0.5 m: its length less 0.5 (m)",
                 std::abs(step.shift_m - 0.5), 1e-12);
}

// ---------------------------------------------------------------------------------- refusals

/// Checks that a refusal's message holds a text.
/// @param what [in] the case, printed and named in the check line.
/// @param refused [in] whether it was refused.
/// @param message [in] the message.
/// @param expected [in] a text the message holds.
void check_refused_with(const std::string &what, bool refused, const std::string &message,
                        const std::string &expected)
{
    gca_test::check_refusal(what, refused, message, message.find(expected) != std::string::npos);
}

/// The refusals: tracking's parameters out of their ranges, each named; a size of 0; a closed
/// pool, which comes back closed with the live model's first claim refused, and refuses a live
/// model's parameter out of range before any claim; an empty tracker; a frame of another size and
/// a frame not later than the one before, each failed with nothing changed, after which the next
/// frame is tracked.
/// @param stream [in] the stream.
void check_refusals(cudaStream_t stream)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto with = [](auto change) {
        gca::tracking_parameters p = test_parameters();
        change(p);
        return p;
    };
    const struct
    {
        const char *what;
        gca::tracking_parameters parameters;
        const char *reason;
    } refused[] = {
        {"a depth range whose near limit is its far one",
         with([](auto &p) { p.pyramid.range.near_m = p.pyramid.range.far_m; }), "the depth range"},
        {"a normals' window of 0", with([](auto &p) { p.pyramid.normals.radius_px = 0; }),
         "the normals"},
        {"no iteration at level 0",
         with([](auto &p) { p.registration.levels[0].max_iterations = 0; }), "the registration"},
        {"a least fitness of 0", with([](auto &p) { p.thresholds.min_fitness = 0.0; }),
         "the least fitness"},
        {"a least fitness above 1", with([](auto &p) { p.thresholds.min_fitness = 1.5; }),
         "the least fitness"},
        {"an RMSE threshold not a number",
         with([nan](auto &p) { p.thresholds.max_inlier_rmse_m = nan; }), "the largest inlier RMSE"},
        {"a turn rate of 0", with([](auto &p) { p.thresholds.max_turn_rate_deg_s = 0.0; }),
         "the largest turn rate"},
        {"a speed not finite", with([](auto &p) {
             p.thresholds.max_speed_m_s = std::numeric_limits<double>::infinity();
         }),
         "the largest speed"},
    };
    for (const auto &r : refused)
    {
        const std::string reason = gca::tracker::parameters_refusal(r.parameters);
        check_refused_with(std::string("refusals, ") + r.what, !reason.empty(), reason, r.reason);
        gca::gpu_pool pool;
        const gca::request_result<gca::tracker> made =
            gca::tracker::request(pool, small_description.size, r.parameters, live_model, stream);
        check(
            made.status == gca::request_status::failure && made.message == reason,
            (std::string("refusals, ") + r.what + ": the request refused with the reason").c_str());
    }
    check(gca::tracker::parameters_refusal(test_parameters()).empty(),
          "refusals: the test's parameters are usable");
    {
        gca::gpu_pool pool;
        const gca::request_result<gca::tracker> made =
            gca::tracker::request(pool, {0, 400}, test_parameters(), live_model, stream);
        check_refused_with("refusals, a size of 0", made.status == gca::request_status::failure,
                           made.message, "is not > 0");
    }
    {
        gca::gpu_pool pool;
        pool.close();
        const gca::request_result<gca::tracker> made = gca::tracker::request(
            pool, small_description.size, test_parameters(), live_model, stream);
        check(made.status == gca::request_status::closed && pool.refused_while_closed() == 1,
              "refusals, a closed pool: closed, the live model's first claim refused");
        gca::sparse_tsdf_parameters coarse = live_model;
        coarse.voxel_size_m = 0.1f;
        const gca::request_result<gca::tracker> refused_model =
            gca::tracker::request(pool, small_description.size, test_parameters(), coarse, stream);
        check(refused_model.status == gca::request_status::failure &&
                  pool.refused_while_closed() == 1,
              "refusals, a closed pool: a voxel size out of range refused before any claim");
    }
    gca::tracker empty;
    const gca::frame tiny = gca_test::built_frame(
        {{4, 4}, small_description.intrinsics, small_description.depth_unit_m},
        [](std::uint32_t, std::uint32_t) { return std::uint16_t{500}; }, gca_test::grey);
    const gca::tracking_result from_empty = empty.track(tiny, stream);
    check_refused_with("refusals, an empty tracker",
                       from_empty.state == gca::tracking_state::failed, from_empty.message,
                       "holds no buffers");

    const std::optional<gca_test::synthetic_scene> scene = walk_scene();
    if (!scene)
        return;
    gca_test::synthetic_scene first_two = *scene;
    first_two.trajectory.resize(2);
    const std::vector<gca::frame> frames = gca_test::render(first_two, small_description).frames;
    if (frames.size() != 2)
        return;
    tracking_case c;
    c.name = "refusals";
    open_case(c, test_parameters(), stream);
    const gca::tracking_result first = c.tracker.track(frames[0], stream);
    check(first.state == gca::tracking_state::tracking, "refusals: the first frame tracked");
    const gca::tracking_result other_size = c.tracker.track(tiny, stream);
    check_refused_with("refusals, a frame of another size",
                       other_size.state == gca::tracking_state::failed && !other_size.pose,
                       other_size.message, "is not the tracker's 640x400 px");
    const gca::tracking_result same_time = c.tracker.track(
        gca_test::restamped(frames[1], frames[1].index(), frames[0].timestamp_us()), stream);
    check_refused_with("refusals, a frame not later than the one before",
                       same_time.state == gca::tracking_state::failed && !same_time.pose,
                       same_time.message, "is not later than that of the frame before it");
    const gca::tracking_result second = c.tracker.track(frames[1], stream);
    print_tracked("refusals, the next frame", second);
    check(second.state == gca::tracking_state::tracking && second.interval_s &&
              std::abs(*second.interval_s - 1e-6 * gca_test::synthetic_frame_period_us) < 1e-12,
          "refusals: the next frame tracked, one frame time after the first");
    check(c.pool.refused_while_closed() == 0, "refusals: the closed pool asked for nothing");
}

// ---------------------------------------------------------------------------------- the walk

/// The largest error of the walk's tracked trajectory from the known one, each pose's turn and
/// shift: 0.1 degrees and 1.0 mm, about twice the largest measured, 0.054 degrees and 0.52 mm
/// [M 2026-11-15, this test].
constexpr gca_test::transform_tolerance walk_tolerance = {0.1, 0.001};

/// The largest turn and shift of the walk's last frame registered again against the cast from its
/// own tracked pose: 0.005 degrees and 0.05 mm, about three times the measured, 0.0015 degrees
/// and 0.014 mm [M 2026-11-15, this test].
constexpr gca_test::transform_tolerance again_tolerance = {0.005, 0.00005};

/// The walk once around the tabletop box: every frame tracking; each tracked pose within
/// walk_tolerance of the known one in the models' frame, the first frame's camera frame; the
/// closed pool asked for nothing. The least fitness, the largest RMSE, the least λ_1, the ends,
/// the largest last step and the largest step and its rates are printed. Then the last frame
/// again, one frame time later: tracking, its transform within again_tolerance of the identity.
/// @param stream [in] the stream.
void check_walk(cudaStream_t stream)
{
    const std::optional<gca_test::synthetic_scene> scene = walk_scene();
    if (!scene)
        return;
    const std::vector<gca::frame> frames = gca_test::render(*scene, small_description).frames;
    if (frames.size() != walk_poses)
        return;
    tracking_case c;
    c.name = "walk";
    open_case(c, test_parameters(), stream);
    const Eigen::Isometry3d to_model = scene->trajectory.front().inverse();
    std::size_t tracked = 0;
    std::size_t converged = 0;
    double least_fitness = 1.0;
    double largest_rmse_m = 0.0;
    double least_lambda = std::numeric_limits<double>::infinity();
    double largest_last_turn_rad = 0.0;
    double largest_last_shift_m = 0.0;
    double largest_turn_deg = 0.0;
    double largest_shift_m = 0.0;
    double largest_turn_rate = 0.0;
    double largest_speed = 0.0;
    gca_test::transform_error worst{0.0, 0.0};
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        const gca::tracking_result result = c.tracker.track(frames[i], stream);
        if (i % 40 == 0 || result.state != gca::tracking_state::tracking)
            print_tracked("walk, frame " + std::to_string(i), result);
        if (result.state != gca::tracking_state::tracking || !result.pose)
            continue;
        tracked++;
        const gca::registration_result &r = result.registration;
        converged += r.converged() ? 1 : 0;
        least_fitness = std::min(least_fitness, r.fitness);
        largest_rmse_m = std::max(largest_rmse_m, r.inlier_rmse_m);
        least_lambda = std::min(least_lambda, r.eigenvalues[0]);
        largest_last_turn_rad = std::max(largest_last_turn_rad, r.last_step_turn_rad);
        largest_last_shift_m = std::max(largest_last_shift_m, r.last_step_shift_m);
        largest_turn_deg = std::max(largest_turn_deg, result.step.turn_deg);
        largest_shift_m = std::max(largest_shift_m, result.step.shift_m);
        if (result.interval_s)
        {
            largest_turn_rate =
                std::max(largest_turn_rate, result.step.turn_deg / *result.interval_s);
            largest_speed = std::max(largest_speed, result.step.shift_m / *result.interval_s);
        }
        const gca_test::transform_error error =
            gca_test::error_of(transform_of(*result.pose), to_model * scene->trajectory[i]);
        worst.rotation_deg = std::max(worst.rotation_deg, error.rotation_deg);
        worst.translation_m = std::max(worst.translation_m, error.translation_m);
    }
    std::printf("walk: %zu of %zu frames tracking, %zu converged and %zu with their iterations "
                "used up; least fitness %.4f, largest inlier RMSE %.3g mm, least lambda_1 %.3g; "
                "largest last step %.3g rad and %.3g m; largest step %.3g deg and %.3g mm, "
                "%.3g deg/s and %.3g m/s; largest error from the known trajectory %.4g deg and "
                "%.4g mm\n",
                tracked, frames.size(), converged, tracked - converged, least_fitness,
                1e3 * largest_rmse_m, least_lambda, largest_last_turn_rad, largest_last_shift_m,
                largest_turn_deg, 1e3 * largest_shift_m, largest_turn_rate, largest_speed,
                worst.rotation_deg, 1e3 * worst.translation_m);
    check(tracked == frames.size(), "walk: every frame tracking");
    check_within("walk: the largest turn from the known trajectory (deg)", worst.rotation_deg,
                 walk_tolerance.rotation_deg);
    check_within("walk: the largest shift from the known trajectory (m)", worst.translation_m,
                 walk_tolerance.translation_m);

    const gca::tracking_result again = c.tracker.track(
        gca_test::restamped(frames.back(), frames.back().index(),
                            frames.back().timestamp_us() + gca_test::synthetic_frame_period_us),
        stream);
    print_tracked("walk, the last frame again", again);
    check(again.state == gca::tracking_state::tracking, "walk, the last frame again: tracking");
    gca_test::check_near("walk, the last frame again against the cast from its own pose",
                         again.registration.transform, Eigen::Isometry3d::Identity(),
                         again_tolerance);
    check(c.pool.refused_while_closed() == 0, "walk: the closed pool asked for nothing");
}

/// Frames without depth during the walk: the walk's first gap_case_frames frames, frames 6 to 8
/// without depth. Frames 0 to 5 tracking; each frame of the gap tracking loss, the registration's
/// end too few correspondences, no pose; the model's points after the gap those before it, bit
/// for bit as a set; the frames after the gap tracking again.
/// @param stream [in] the stream.
void check_depth_gap(cudaStream_t stream)
{
    const std::optional<gca_test::synthetic_scene> scene = walk_scene();
    if (!scene)
        return;
    gca_test::synthetic_source_options options;
    options.description = small_description;
    options.scene = *scene;
    options.scene.trajectory.resize(gap_case_frames);
    options.depth_gap = depth_gap;
    const std::vector<gca::frame> frames = frames_of(options);
    tracking_case c;
    c.name = "depth gap";
    open_case(c, test_parameters(), stream);
    std::vector<gca_test::point_key> before;
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        const bool in_gap =
            i >= depth_gap.first_frame && i < depth_gap.first_frame + depth_gap.frame_count;
        if (i == depth_gap.first_frame)
            before = model_points(c, stream, "before the gap");
        const gca::tracking_result result = c.tracker.track(frames[i], stream);
        const std::string what = "depth gap, frame " + std::to_string(i);
        print_tracked(what, result);
        if (in_gap)
            check(result.state == gca::tracking_state::lost && !result.pose &&
                      result.outcome == gca::check_outcome::registration_end &&
                      result.registration.end == gca::registration_end::few_correspondences,
                  (what + ": tracking loss, too few correspondences, no pose").c_str());
        else
            check(result.state == gca::tracking_state::tracking && result.pose,
                  (what + ": tracking").c_str());
        if (i + 1 == depth_gap.first_frame + depth_gap.frame_count)
        {
            const std::vector<gca_test::point_key> after = model_points(c, stream, "after the gap");
            std::printf("depth gap: %zu points before the gap, %zu after it\n", before.size(),
                        after.size());
            check(!before.empty() && after == before,
                  "depth gap: the model's points after the gap those before it");
        }
    }
    check(c.pool.refused_while_closed() == 0, "depth gap: the closed pool asked for nothing");
}

/// The frames of the view sliding along the grey plane [?: a choice].
constexpr std::size_t slide_frames = 10;

/// A view sliding along one grey plane, the plane without texture: the camera 0.45 m above the
/// plane, looking at a point 0.2 m ahead on it, moving 1 cm along it each frame [?: a choice],
/// every pixel within the tabletop object's depth range or beyond it. Every frame tracking loss,
/// the registration degenerate, never a pose; the model holds no point.
/// @param stream [in] the stream.
void check_grey_plane(cudaStream_t stream)
{
    gca_test::synthetic_scene scene;
    scene.plane = gca_test::synthetic_plane{2.0, 2.0, gca_test::synthetic_texture::grey};
    for (std::size_t i = 0; i < slide_frames; i++)
    {
        const double x = 0.01 * static_cast<double>(i);
        const std::optional<Eigen::Isometry3d> pose = gca_test::look_at(
            Eigen::Vector3d(x, 0.0, 0.45), Eigen::Vector3d(x + 0.2, 0.0, 0.0), 0.0);
        check(pose.has_value(), "grey plane: a pose");
        if (pose)
            scene.trajectory.push_back(*pose);
    }
    const std::vector<gca::frame> frames = gca_test::render(scene, small_description).frames;
    tracking_case c;
    c.name = "grey plane";
    open_case(c, test_parameters(), stream);
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        const gca::tracking_result result = c.tracker.track(frames[i], stream);
        const std::string what = "grey plane, frame " + std::to_string(i);
        print_tracked(what, result);
        check(result.state == gca::tracking_state::lost && !result.pose &&
                  result.registration.end == gca::registration_end::degenerate,
              (what + ": tracking loss, degenerate, no pose").c_str());
    }
    const std::vector<gca_test::point_key> points = model_points(c, stream, "after the slide");
    check(points.empty(), "grey plane: the model holds no point");
    check(c.pool.refused_while_closed() == 0, "grey plane: the closed pool asked for nothing");
}

/// The sphere's move between the fused frame and the next: 2 cm along the world's x [?: a choice,
/// the move of docs/methods/registration.md, "Huber's weight"].
const Eigen::Vector3d sphere_move_m(0.02, 0.0, 0.0);

/// Tracks the tabletop object seen from its pose, then the same scene with its sphere moved, seen
/// a hand's step on (gca_test::tabletop_delta).
/// @param max_inlier_rmse_m [in] e_track, in m.
/// @param name [in] the case's name.
/// @param stream [in] the stream.
/// @param fused_before [out] the model's points after the first frame.
/// @param fused_after [out] the model's points after the second.
/// @return the second frame's result; failed when the frames could not be rendered.
gca::tracking_result track_moved_sphere(double max_inlier_rmse_m, const std::string &name,
                                        cudaStream_t stream,
                                        std::vector<gca_test::point_key> &fused_before,
                                        std::vector<gca_test::point_key> &fused_after)
{
    const std::optional<Eigen::Isometry3d> pose_a = gca_test::tabletop_object_pose();
    check(pose_a.has_value(), (name + ": the pose").c_str());
    if (!pose_a)
        return {};
    gca_test::synthetic_scene still = gca_test::tabletop_object_scene();
    still.trajectory = {*pose_a};
    gca_test::synthetic_scene moved = gca_test::tabletop_object_scene();
    moved.sphere->centre_m += sphere_move_m;
    moved.trajectory = {*pose_a, *pose_a * gca_test::tabletop_delta};
    const std::vector<gca::frame> first = gca_test::render(still, small_description).frames;
    const std::vector<gca::frame> second = gca_test::render(moved, small_description).frames;
    if (first.size() != 1 || second.size() != 2)
        return {};
    tracking_case c;
    c.name = name;
    open_case(c, test_parameters(max_inlier_rmse_m), stream);
    const gca::tracking_result fused = c.tracker.track(first[0], stream);
    print_tracked(name + ", the first frame", fused);
    check(fused.state == gca::tracking_state::tracking,
          (name + ": the first frame tracked").c_str());
    fused_before = model_points(c, stream, "after the first frame");
    const gca::tracking_result result = c.tracker.track(second[1], stream);
    print_tracked(name + ", the sphere moved", result);
    fused_after = model_points(c, stream, "after the second frame");
    check(c.pool.refused_while_closed() == 0,
          (name + ": the closed pool asked for nothing").c_str());
    return result;
}

/// The tabletop object's sphere moved by 2 cm between the fused frame and the next, the camera
/// moved by a hand's step: at e_track 1.0 mm, tracking loss by the inlier RMSE, no pose, and the
/// model's points the same before and after; at the tabletop object's own e_track, 5 mm, the
/// frame passes and is fused, the cost the user accepted (docs/methods/tracking.md, "Parameters"),
/// printed with the pose's error from the known one.
/// @param stream [in] the stream.
void check_moved_sphere(cudaStream_t stream)
{
    std::vector<gca_test::point_key> before;
    std::vector<gca_test::point_key> after;
    const gca::tracking_result strict =
        track_moved_sphere(noise_free_rmse_m, "moved sphere at 1.0 mm", stream, before, after);
    check(strict.state == gca::tracking_state::lost && !strict.pose &&
              strict.outcome == gca::check_outcome::inlier_rmse,
          "moved sphere at 1.0 mm: tracking loss by the inlier RMSE, no pose");
    check(!before.empty() && after == before,
          "moved sphere at 1.0 mm: the model's points the same before and after");

    const gca::tracking_result own =
        track_moved_sphere(tabletop_set.max_inlier_rmse_m,
                           "moved sphere at the tabletop object's 5 mm", stream, before, after);
    if (own.pose)
    {
        const gca_test::transform_error error =
            gca_test::error_of(transform_of(*own.pose), gca_test::tabletop_delta);
        std::printf("moved sphere at the tabletop object's 5 mm: %s, %.4g deg and %.4g mm from "
                    "the known pose, %zu points before and %zu after\n",
                    state_name(own.state), error.rotation_deg, 1e3 * error.translation_m,
                    before.size(), after.size());
    }
    check(own.state == gca::tracking_state::tracking && own.pose && after != before,
          "moved sphere at the tabletop object's 5 mm: tracking, the frame fused");
}
} // namespace

/// Runs the checks, prints the test's time and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    const auto start = std::chrono::steady_clock::now();
    cudaStream_t stream = nullptr;
    if (!gca_test::create_stream(stream))
        return gca_test::check_summary("test_tracking");
    check_the_check();
    check_refusals(stream);
    check_walk(stream);
    check_depth_gap(stream);
    check_grey_plane(stream);
    check_moved_sphere(stream);
    gca_test::check_no_cuda_error("tracking");
    cudaStreamDestroy(stream);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("test_tracking: %.1f s\n", seconds);
    return gca_test::check_summary("test_tracking");
}
