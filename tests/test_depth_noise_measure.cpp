// SPDX-License-Identifier: 0BSD
/// @file test_depth_noise_measure.cpp
/// Offline test of depth_noise's measurement (depth_noise_measure.hpp, docs/methods/depth-noise.md)
/// on sequences the test writes through the sequence's writer into a temporary directory it
/// removes: a plane seen obliquely by a camera at rest, rendered by the synthetic source with its
/// Gaussian depth noise of a known c, rounded to the camera's depth unit of 1 mm.
/// - The fit recovers c = 0.002 /m, 0.01 /m and 0.0005 /m, each bin's own c_b too, with a small
///   spread; the bins whose noise lies below half the depth unit, at 0.0005 /m, are left out of
///   the fit; a still camera's drift ratio lies near 1; the exactly flat plane's spatial error
///   lies near 0.
/// - A pixel with depth in fewer than 90 % of the frames is left out, one with depth in 90 % kept.
/// - Each refusal, with its reason: a sequence that cannot be read or ends early; a depth unit
///   that is no positive number; intrinsics that are no pinhole camera's; too few pixels (one
///   frame, and pixels too few for a bin); a camera moving along its axis; a noise-free plane,
///   whose noise lies below the rounding.
/// - The report's lines; the program depth_noise's three exits.
/// Needs no GPU and no camera.

#include "check.hpp"
#include "depth_noise_measure.hpp"
#include "program_run.hpp"
#include "synthetic_source.hpp"
#include "test_files.hpp"

#include "camera/sequence_writer.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>

using gca_test::check;
using gca_test::check_refusal;
using gca_test::check_summary;
using gca_test::check_within;
using gca_test::program_run;
using gca_test::run_program;

namespace
{
/// The test's frames: the camera's 640x400 intrinsics halved, by the one pinhole convention
/// (pixel centres at whole numbers, so cx' = (cx - 0.5) / 2), and its depth unit, 1 mm
/// [M 2026-11-13: docs/HARDWARE.md]; a quarter of the pixels, so that the test stays short
/// [?: a choice].
constexpr gca::frame_description test_description = {
    {320, 200}, {153.09f, 153.07f, 158.455f, 99.315f}, 0.001f};

/// The frames of a still sequence [?: a choice]: a pixel's variance from 60 frames has a relative
/// deviation of (2 / 59)^(1/2) = 18 % [S: the chi-square distribution], and a bin averages over at
/// least 1000 pixels.
constexpr std::size_t still_frames = 60;

/// The camera's pose: 0.6 m above the plane z = 0, looking 45 deg down at the point 0.6 m ahead
/// along the world's y axis, so that the picture's rows see the plane from about 0.5 m at the
/// bottom to about 2.4 m at the top [S: computed, the rows' rays 11.8 to 78.2 deg below the
/// horizon].
/// @return the pose, camera frame to world frame.
Eigen::Isometry3d oblique_pose()
{
    return *gca_test::look_at(Eigen::Vector3d(0.0, 0.0, 0.6), Eigen::Vector3d(0.0, 0.6, 0.0), 0.0);
}

/// The scene: a plane of 10 m x 10 m, which fills every pixel of the oblique pose's picture, seen
/// from a trajectory.
/// @param trajectory [in] the poses, one frame for each.
/// @param noise_per_m [in] the depth noise's c, in 1/m; 0 for none.
/// @return the synthetic source's options: the test's frames, the plane, the noise from seed 7.
gca_test::synthetic_source_options plane_options(std::vector<Eigen::Isometry3d> trajectory,
                                                 double noise_per_m)
{
    gca_test::synthetic_source_options options;
    options.description = test_description;
    options.scene = {.plane = gca_test::synthetic_plane{5.0, 5.0},
                     .trajectory = std::move(trajectory)};
    if (noise_per_m > 0.0)
        options.depth_noise = gca_test::synthetic_depth_noise{noise_per_m, 7};
    return options;
}

/// A still camera's trajectory: the oblique pose, once for each frame.
/// @param frames [in] the frames.
/// @return the trajectory.
std::vector<Eigen::Isometry3d> still(std::size_t frames)
{
    return std::vector<Eigen::Isometry3d>(frames, oblique_pose());
}

/// A camera moving 20 mm forward along its own axis over the still sequence's frames, evenly
/// [?: a choice]: each pixel's depth changes by up to about 20 mm, against a noise of 0.5 mm to
/// 12 mm at c = 0.002 /m.
/// @return the trajectory.
std::vector<Eigen::Isometry3d> moving_trajectory()
{
    std::vector<Eigen::Isometry3d> trajectory;
    for (std::size_t k = 0; k < still_frames; k++)
        trajectory.push_back(oblique_pose() *
                             Eigen::Translation3d(0.0, 0.0, 0.02 * k / (still_frames - 1)));
    return trajectory;
}

/// Writes a frame into a sequence, marked as tracked by nothing, waiting while the writer's queue
/// is full.
/// @param writer [in,out] the sequence's writer, open.
/// @param frame [in,out] the frame, swapped with a slot of the queue when written.
void push_frame(gca::sequence_writer &writer, gca::frame &frame)
{
    gca::push_result pushed;
    while ((pushed = writer.push(frame, gca::not_tracked{})).status == gca::push_status::full)
        std::this_thread::yield();
    check(pushed.status == gca::push_status::accepted, "test setup: a frame written");
}

/// Writes a source's frames into a sequence through the sequence's writer, each frame changed
/// first by the test's function when it gives one.
/// @param path [in] the sequence's path, where no file stands.
/// @param options [in] the synthetic source's options.
/// @param change [in] what the test changes in frame k before it is written, or none.
void write_sequence(const std::filesystem::path &path, gca_test::synthetic_source_options options,
                    const std::function<void(gca::frame &, std::size_t)> &change = {})
{
    const gca::frame_description description = options.description;
    gca_test::synthetic_source source(std::move(options));
    check(source.open(description.size).opened, "test setup: the synthetic source opened");
    gca::sequence_writer writer;
    check(writer.open(path, description, 16).opened, "test setup: the writer opened");
    gca::frame frame;
    for (std::size_t k = 0; source.next(frame).status == gca::next_status::frame; k++)
    {
        if (change)
            change(frame, k);
        push_frame(writer, frame);
    }
    check(writer.close().written, "test setup: the sequence closed");
}

/// Prints a measurement's lines.
/// @param what [in] the case.
/// @param result [in] the measurement or the refusal.
void print_result(const std::string &what, const gca::depth_noise_result &result)
{
    std::printf("%s:\n%s\n", what.c_str(),
                result.measurement ? gca::depth_noise_report(what, *result.measurement).c_str()
                                   : result.refusal.c_str());
}

/// A still sequence's case: its known c and what the measurement of it is checked against.
struct known_c_case
{
    double noise_per_m; ///< the known c, in 1/m
    /// The nearest bins whose noise lies below half the depth unit, left out of the fit
    /// [S: computed, sigma = c z^2 below 0.5 mm in the bins nearer than (0.0005 m / c)^(1/2)]: none
    /// at 0.002 /m and 0.01 /m, whose nearest bin, from 0.51 m, lies above it; at 0.0005 /m the
    /// five from 0.5 m to 1.0 m.
    std::size_t bins_below;
    /// The spatial error's tolerance, in mm [?: a choice, from the measurement in test_known_c()].
    double spatial_tolerance_mm;
};

/// A still camera's sequence of the oblique plane with the depth noise of a known c: the fit
/// recovers c, each bin with a c_b recovers it too, the spread stays small, the bins below the
/// rounding are the nearest ones and have no c_b, every pixel is kept, the drift ratio lies near 1
/// and the spatial error near 0, since the plane is exactly flat and its only spatial error is the
/// temporal noise's share of the mean frame, which the method removes. The tolerances
/// [M 2026-11-15: this test, seed 7]: c 0.14 % off at 0.002 /m and 0.01 /m, against 0.5 %; the
/// bins' c_b at most 0.39 %, against 1.5 %; the spread 0.23 % of c, against 1 %; the drift ratio
/// 0.99, against 1 +- 0.05; the spatial error 0.071 mm at c = 0.002 /m and 0.352 mm at
/// c = 0.01 /m, the estimate's statistical floor, which scales with c, against 0.15 mm per
/// 0.002 /m of c. At 0.0005 /m the bins below the rounding keep part of the depth unit's
/// staircase in the mean frame, at most u / 12^(1/2) = 0.29 mm [S: computed], against 0.3 mm.
/// @param path [in] the sequence of still_frames frames of the oblique plane with the case's
///     depth noise.
/// @param known [in] the case.
void test_known_c(const std::filesystem::path &path, const known_c_case &known)
{
    const double noise_per_m = known.noise_per_m;
    const std::string name = "still c=" + std::to_string(noise_per_m);
    const gca::depth_noise_result result = gca::measure_depth_noise(path);
    print_result(name, result);
    check(result.measurement.has_value(), (name + ": measured").c_str());
    if (!result.measurement)
        return;
    const gca::depth_noise_measurement &m = *result.measurement;
    check(m.frames == still_frames, (name + ": every frame read").c_str());
    check(m.pixels_kept == std::size_t{320} * 200, (name + ": every pixel kept").c_str());
    check_within((name + ": c's relative error").c_str(),
                 std::fabs(m.noise_per_m - noise_per_m) / noise_per_m, 0.005);
    gca_test::max_error bin_errors;
    std::size_t below = 0;
    bool below_nearest = true;
    for (std::size_t k = 0; k < m.bins.size(); k++)
    {
        const gca::depth_noise_bin &bin = m.bins[k];
        if (bin.noise_per_m)
            bin_errors.add(std::fabs(*bin.noise_per_m - noise_per_m) / noise_per_m);
        else
        {
            below_nearest = below_nearest && k == below && bin.noise_mm < 0.5;
            below++;
        }
    }
    std::printf("%s: %zu bins below the rounding, %zu fitted\n", name.c_str(), below,
                m.bins_fitted);
    check(below == known.bins_below && below_nearest,
          (name + ": the nearest bins below the rounding, without a c_b").c_str());
    check(m.bins_fitted == m.bins.size() - below && m.bins_fitted >= 3,
          (name + ": the other bins fitted, at least 3").c_str());
    gca_test::check_max_error(name + ": the largest relative error of a bin's c_b", bin_errors,
                              0.015);
    check_within((name + ": the spread over c").c_str(), m.spread_per_m / noise_per_m, 0.01);
    check_within((name + ": the drift ratio's distance from 1").c_str(),
                 std::fabs(m.drift_ratio - 1.0), 0.05);
    check_within((name + ": the spatial error, mm").c_str(), m.spatial_mm,
                 known.spatial_tolerance_mm);
}

/// A pixel with depth in 53 of 60 frames, 88 %, is left out; one with depth in 54, 90 %, is kept:
/// the test removes the depth of the first 10 columns from 7 frames of the first half and of the
/// next 10 from 6 frames of the second half.
/// @param directory [in] the test's directory.
void test_kept_share(const gca_test::temporary_directory &directory)
{
    const std::filesystem::path path = directory / "share.sequence";
    const std::size_t width = test_description.size.width;
    write_sequence(path, plane_options(still(still_frames), 0.002),
                   [&](gca::frame &frame, std::size_t k) {
                       const std::size_t first = k < 7 ? 0 : (k >= still_frames - 6 ? 10 : width);
                       for (std::size_t p = 0; p < frame.depth().size(); p++)
                           if (p % width >= first && p % width < first + 10)
                               frame.depth()[p] = 0;
                   });
    const gca::depth_noise_result result = gca::measure_depth_noise(path);
    std::printf("depth missing in 7 or 6 of the frames: %zu pixels kept\n",
                result.measurement ? result.measurement->pixels_kept : std::size_t{0});
    check(result.measurement && result.measurement->pixels_kept == std::size_t{310} * 200,
          "the 10 columns with depth in 88 % of the frames left out, those with 90 % kept");
}

/// Checks a refusal: the sequence is refused, with the reason given or its start and end.
/// @param what [in] the case.
/// @param result [in] what the measurement gave.
/// @param start [in] the refusal's start.
/// @param end [in] the refusal's end; empty when start is the whole refusal.
void check_refused(const std::string &what, const gca::depth_noise_result &result,
                   const std::string &start, const std::string &end = {})
{
    const std::string &r = result.refusal;
    const bool expected = end.empty() ? r == start
                                      : r.size() >= start.size() + end.size() &&
                                            r.compare(0, start.size(), start) == 0 &&
                                            r.compare(r.size() - end.size(), end.size(), end) == 0;
    check_refusal(what, !result.measurement.has_value(), r, expected);
}

/// A sequence of two frames of a description the test gives, every pixel at 1 m.
/// @param path [in] the sequence's path, where no file stands.
/// @param description [in] the frames' description, written into the header as it is.
void write_flat_frames(const std::filesystem::path &path, const gca::frame_description &description)
{
    gca::sequence_writer writer;
    check(writer.open(path, description, 2).opened, "test setup: the writer opened");
    for (std::uint64_t k = 0; k < 2; k++)
    {
        gca::frame frame;
        frame.prepare(description, k, 1000 * (k + 1));
        for (std::uint16_t &depth : frame.depth())
            depth = 1000;
        push_frame(writer, frame);
    }
    check(writer.close().written, "test setup: the sequence closed");
}

/// The refusals, each with its reason, naming the sequence: one that cannot be opened and one that
/// ends early, with the reader's reason; a depth unit of 0, -0.001 m and NaN; intrinsics with fx 0
/// and with cx NaN; too few pixels: one frame, whose pixels lie in no half's frames, and a
/// sequence with depth in 5 columns only, 1000 pixels over many bins; a camera moving 20 mm along
/// its axis; a noise-free plane, whose noise lies below the rounding everywhere.
/// @param directory [in] the test's directory.
/// @param still_sequence [in] a still sequence of 60 frames of the plane.
/// @param moving_sequence [in] the sequence of moving_trajectory().
void test_refusals(const gca_test::temporary_directory &directory,
                   const std::filesystem::path &still_sequence,
                   const std::filesystem::path &moving_sequence)
{
    const std::filesystem::path missing = directory / "missing.sequence";
    check_refused("a missing file", gca::measure_depth_noise(missing),
                  missing.string() + ": cannot open: No such file or directory");

    const std::filesystem::path cut = directory / "cut.sequence";
    std::filesystem::copy_file(still_sequence, cut);
    std::filesystem::resize_file(cut, std::filesystem::file_size(cut) - 100);
    check_refused("a sequence that ends early", gca::measure_depth_noise(cut),
                  cut.string() + ": ends early: within record 60, before the sequence's end");

    struct header_case
    {
        std::string what;                   ///< the case
        gca::frame_description description; ///< the header's description
        std::string reason;                 ///< the refusal after the path
    };
    gca::frame_description unit_zero = test_description;
    unit_zero.depth_unit_m = 0.0f;
    gca::frame_description unit_negative = test_description;
    unit_negative.depth_unit_m = -0.001f;
    gca::frame_description unit_nan = test_description;
    unit_nan.depth_unit_m = std::numeric_limits<float>::quiet_NaN();
    gca::frame_description fx_zero = test_description;
    fx_zero.intrinsics.fx = 0.0f;
    gca::frame_description cx_nan = test_description;
    cx_nan.intrinsics.cx = std::numeric_limits<float>::quiet_NaN();
    const std::vector<header_case> header_cases = {
        {"a depth unit of 0", unit_zero, "its depth unit, 0 m per value, is no positive number"},
        {"a depth unit of -0.001 m", unit_negative,
         "its depth unit, -0.001 m per value, is no positive number"},
        {"a depth unit of NaN", unit_nan, "its depth unit, nan m per value, is no positive number"},
        {"fx 0", fx_zero,
         "its intrinsics, fx 0 fy 153.07 cx 158.455 cy 99.315 px, are no pinhole camera's"},
        {"cx NaN", cx_nan,
         "its intrinsics, fx 153.09 fy 153.07 cx nan cy 99.315 px, are no pinhole camera's"},
    };
    for (std::size_t k = 0; k < header_cases.size(); k++)
    {
        const std::filesystem::path path = directory / ("header" + std::to_string(k) + ".sequence");
        write_flat_frames(path, header_cases[k].description);
        check_refused(header_cases[k].what, gca::measure_depth_noise(path),
                      path.string() + ": " + header_cases[k].reason);
    }

    const std::filesystem::path one = directory / "one.sequence";
    write_sequence(one, plane_options(still(1), 0.002));
    check_refused("one frame", gca::measure_depth_noise(one),
                  one.string() + ": too few pixels: 0 depth bins of 0.1 m hold at least 1000 of "
                                 "its 0 pixels kept, with depth in at least 90 % of the 1 frames "
                                 "and in both halves; the fit needs 3");

    const std::filesystem::path few = directory / "few.sequence";
    const std::size_t width = test_description.size.width;
    write_sequence(few, plane_options(still(still_frames), 0.002),
                   [&](gca::frame &frame, std::size_t) {
                       for (std::size_t p = 0; p < frame.depth().size(); p++)
                           if (p % width >= 5)
                               frame.depth()[p] = 0;
                   });
    check_refused("depth in 5 columns", gca::measure_depth_noise(few),
                  few.string() + ": too few pixels: 0 depth bins of 0.1 m hold at least 1000 of "
                                 "its 1000 pixels kept, with depth in at least 90 % of the 60 "
                                 "frames and in both halves; the fit needs 3");

    check_refused("a camera moving 20 mm along its axis", gca::measure_depth_noise(moving_sequence),
                  moving_sequence.string() + ": not still: the halves differ by ",
                  " times their standard error; a still sequence stays within 3 times");

    const std::filesystem::path noise_free = directory / "noise_free.sequence";
    write_sequence(noise_free, plane_options(still(still_frames), 0.0));
    check_refused("a noise-free plane", gca::measure_depth_noise(noise_free),
                  noise_free.string() +
                      ": below the rounding: the noise lies below 0.5 depth units, 0.50 mm, in ",
                  " depth bins that hold enough pixels; the fit needs 3 above it");
}

/// The report's lines for a measurement built by hand: the numbers rounded as documented, a bin
/// below the rounding without its c.
void test_report()
{
    gca::depth_noise_measurement m{};
    m.frames = 300;
    m.description = {{1280, 800}, {612.37f, 612.29f, 635.81f, 398.27f}, 0.001f};
    m.pixels_kept = 1000123;
    m.drift_mm = 0.0345;
    m.drift_ratio = 1.0349;
    m.spatial_mm = 1.234;
    m.bins = {{0.4, 0.5, 51234, 0.3149, 0.4551, std::nullopt},
              {0.5, 0.6, 40001, 0.5551, 0.6049, 0.0018994},
              {2.9, 3.0, 1000, 19.0, 3.0, 0.00211149}};
    m.noise_per_m = 0.0021234;
    m.spread_per_m = 0.0001049;
    m.bins_fitted = 2;
    const std::string expected =
        "depth_noise: seq/still: 300 frames of 1280x800, depth unit 1 mm\n"
        "pixels: 1000123 of 1024000 kept, with depth in at least 90 % of the frames and in both "
        "halves\n"
        "still: the halves differ by 0.03 mm rms, 1.03 times their standard error\n"
        "spatial: 1.23 mm rms from the plane fitted to the mean frame\n"
        "bin 0.4 to 0.5 m: 51234 pixels, noise 0.31 mm, spatial 0.46 mm, below the rounding\n"
        "bin 0.5 to 0.6 m: 40001 pixels, noise 0.56 mm, spatial 0.60 mm, c 0.001899 /m\n"
        "bin 2.9 to 3.0 m: 1000 pixels, noise 19.00 mm, spatial 3.00 mm, c 0.002111 /m\n"
        "c: 0.002123 /m, spread 0.000105 /m over 2 bins";
    const std::string report = gca::depth_noise_report("seq/still", m);
    std::printf("the report:\n%s\n", report.c_str());
    check(report == expected, "the report's lines");
}

/// The program on sequences the test wrote, each in a directory of its own as frames.sequence: a
/// still sequence prints the measurement's lines, exit 0; a moving one prints "depth_noise: " and
/// the refusal, exit 1; a directory that does not exist is refused with the usage, exit 2.
/// @param program [in] depth_noise's path.
/// @param directory [in] the test's directory.
/// @param still_sequence [in] a still sequence of the plane.
/// @param moving_sequence [in] a sequence of a moving camera.
void test_program(const std::string &program, const gca_test::temporary_directory &directory,
                  const std::filesystem::path &still_sequence,
                  const std::filesystem::path &moving_sequence)
{
    const std::filesystem::path still_directory = directory / "still";
    const std::filesystem::path moving_directory = directory / "moving";
    std::filesystem::create_directory(still_directory);
    std::filesystem::create_directory(moving_directory);
    std::filesystem::copy_file(still_sequence, still_directory / "frames.sequence");
    std::filesystem::copy_file(moving_sequence, moving_directory / "frames.sequence");

    const program_run measured = run_program(program, {still_directory.string()});
    const gca::depth_noise_result still_result =
        gca::measure_depth_noise(still_directory / "frames.sequence");
    std::printf("depth_noise on a still sequence, exit %d:\n%s", measured.exit_status,
                measured.output.c_str());
    check(measured.exit_status == 0 && still_result.measurement &&
              measured.output ==
                  gca::depth_noise_report(still_directory.string(), *still_result.measurement) +
                      "\n",
          "depth_noise on a still sequence: its lines, exit 0");

    const program_run refused = run_program(program, {moving_directory.string()});
    std::printf("depth_noise on a moving camera, exit %d: %s", refused.exit_status,
                refused.output.c_str());
    check(refused.exit_status == 1 &&
              refused.output ==
                  "depth_noise: " +
                      gca::measure_depth_noise(moving_directory / "frames.sequence").refusal + "\n",
          "depth_noise on a moving camera: the refusal, exit 1");

    const std::string missing = (directory / "missing").string();
    const program_run unread = run_program(program, {missing});
    std::printf("depth_noise on a missing directory, exit %d: %s", unread.exit_status,
                unread.output.c_str());
    check(unread.exit_status == 2 &&
              unread.output.rfind("depth_noise: " + missing + ": no such directory\nusage: ", 0) ==
                  0,
          "depth_noise on a missing directory: the usage, exit 2");
}
} // namespace

/// Runs the measurement's checks and prints each case's lines, the errors, the summary line and
/// the test's time.
/// @param argc [in] the argument count, 2.
/// @param argv [in] the arguments: argv[1] the path of the program depth_noise.
/// @return 0 when every check passed, 1 otherwise; 2 without the program's path.
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: test_depth_noise_measure DEPTH_NOISE\n");
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    const gca_test::temporary_directory directory("test_depth_noise_measure");
    const std::filesystem::path low = directory / "still_0.002.sequence";
    const std::filesystem::path high = directory / "still_0.01.sequence";
    const std::filesystem::path lowest = directory / "still_0.0005.sequence";
    const std::filesystem::path moving = directory / "moving.sequence";
    write_sequence(low, plane_options(still(still_frames), 0.002));
    write_sequence(high, plane_options(still(still_frames), 0.01));
    write_sequence(lowest, plane_options(still(still_frames), 0.0005));
    write_sequence(moving, plane_options(moving_trajectory(), 0.002));
    test_known_c(low, {0.002, 0, 0.15});
    test_known_c(high, {0.01, 0, 0.75});
    test_known_c(lowest, {0.0005, 5, 0.3});
    test_kept_share(directory);
    test_refusals(directory, low, moving);
    test_report();
    test_program(argv[1], directory, low, moving);
    std::printf("test_depth_noise_measure: %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return check_summary("test_depth_noise_measure");
}
