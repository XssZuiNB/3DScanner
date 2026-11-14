// SPDX-License-Identifier: 0BSD
/// @file test_viewer_loop.cpp
/// orbbec_viewer's loop (viewer_loop.hpp) driven by the tests' synthetic source, a box on the
/// plane seen along a walk around it, headless, with a report after every frame unless a case
/// says otherwise; each case runs in a temporary directory that the test removes. Its printed
/// lines, the color codes taken out, are checked line by line, so a line that changes its words
/// or appears where it should not shows:
/// - three frames after the first: the intrinsics line word for word, a report line in today's
///   form after each frame, then the snapshot of the third: "Saved N points to cloud_000.ply",
///   N the last report's points, and its three files, no others. Its color PNG holds the
///   frame's colors and its depth PNG the frame's depth, bit for bit; its PLY has the viewer's
///   header and N vertices, each with the color of a pixel next to where it projects, so colors
///   swapped on their way to the GPU show;
/// - timeouts before the first frame and between frames: the first frame is waited for, the
///   frame limit counts delivered frames only, and the snapshot is the third delivered frame's;
/// - a report period longer than the run: no report line, the snapshot still saved;
/// - a timeout and a failure after two frames: their line each time, no stale frame shown,
///   counted or saved, and the loop going on until the exit flag, which the test sets after the
///   second, as the viewer's SIGINT handler would;
/// - the source's end after two frames: its message, and the loop ends;
/// - a failure before the first frame: its message and "Couldn't start stream!", and the viewer
///   gives up.
/// Needs the GPU, no camera; links no OrbbecSDK.

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "check.hpp"
#include "synthetic_source.hpp"
#include "test_files.hpp"
#include "types/frame_description.hpp"
#include "viewer_loop.hpp"
#include "viewer_options.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace
{
using gca_test::check;

/// The frames' size: small, so that every pixel is compared at once [?: a choice].
constexpr gca::frame_size test_size = {160, 120};

/// The intrinsics: fx and fy differ, and the principal point lies at no whole or half pixel
/// [?: a choice].
constexpr gca::intrinsics test_intrinsics = {150.0f, 140.0f, 83.3f, 56.6f};

/// The depth unit: 1 mm, the camera's [S: docs/HARDWARE.md].
constexpr float test_depth_unit_m = 0.001f;

/// The intrinsics line the viewer prints for them, word for word: distortion_check's usage and
/// README.md, "Run", point the operator at it.
constexpr const char *intrinsics_line =
    "Intrinsics fx=150 fy=140 cx=83.3 cy=56.6  depth scale=0.001 m/unit";

/// The timeout line, which AGENTS.md §3 reads as a stop condition.
constexpr const char *timeout_line = "No frames received from Orbbec camera for 5s!";

/// The message of the timeouts the test puts between the synthetic source's results.
constexpr const char *test_timeout_message = "the test's timeout";

/// The support plane: 1.2 m x 1.2 m, so that it fills most of each view [?: a choice].
constexpr gca_test::synthetic_plane test_plane = {0.60, 0.60};

/// A tabletop box, 0.10 m x 0.08 m x 0.06 m, turned by 25 degrees [?: a choice].
constexpr gca_test::synthetic_box test_box = {0.02, 0.03, 25.0, 0.10, 0.08, 0.06};

/// The bytes of the viewer's PLY for each vertex: float x, y, z and uchar red, green, blue
/// [S: viewer_loop.cpp, save_ply].
constexpr std::size_t ply_vertex_bytes = 3 * sizeof(float) + 3;

static_assert(std::endian::native == std::endian::little,
              "the PLY's floats are read as the host's: the viewer writes them little-endian");

/// A frame source that passes every call through to the synthetic source, except the calls of
/// next() a test names, which give a timeout instead without asking the synthetic source. It
/// counts the calls of next(), and once a number of them has given a result other than a frame,
/// it sets the loop's exit flag, as the viewer's SIGINT handler would.
class scripted_source final : public gca::frame_source
{
public:
    /// Builds the synthetic source, closed.
    /// @param options [in] the synthetic source's options; taken over.
    /// @param timeout_calls [in] the calls of next(), counted from 1, that give a timeout.
    /// @param stop_after [in] after how many results other than a frame the flag is set; 0 for
    ///        never.
    /// @param exit_flag [out] the loop's exit flag; must outlive the source.
    scripted_source(gca_test::synthetic_source_options options, std::vector<int> timeout_calls,
                    int stop_after, std::atomic<bool> &exit_flag)
        : m_source(std::move(options))
        , m_timeout_calls(std::move(timeout_calls))
        , m_stop_after(stop_after)
        , m_exit_flag(exit_flag)
    {
    }

    /// Opens the synthetic source.
    /// @param size [in] the size asked for, in px.
    /// @return the synthetic source's result.
    gca::open_result open(const gca::frame_size &size) noexcept override
    {
        return m_source.open(size);
    }

    /// Counts the call; gives a timeout on a call the test named, else the synthetic source's
    /// next result; and sets the flag after the stop_after-th result other than a frame.
    /// @param out [out] as the synthetic source fills it; left as it was on a timeout.
    /// @return the timeout, or the synthetic source's result.
    gca::next_result next(gca::frame &out) noexcept override
    {
        m_next_calls++;
        const bool timeout =
            std::ranges::find(m_timeout_calls, m_next_calls) != m_timeout_calls.end();
        gca::next_result result =
            timeout ? gca::next_result{gca::next_status::timeout, test_timeout_message}
                    : m_source.next(out);
        if (result.status != gca::next_status::frame && ++m_no_frames == m_stop_after)
            m_exit_flag.store(true);
        return result;
    }

    /// Closes the synthetic source.
    void close() noexcept override
    {
        m_source.close();
    }

    /// Describes the synthetic source.
    /// @return its description.
    gca::source_description describe() const noexcept override
    {
        return m_source.describe();
    }

    /// Gives the calls of next() so far.
    /// @return their count.
    int next_calls() const noexcept
    {
        return m_next_calls;
    }

private:
    gca_test::synthetic_source m_source;
    std::vector<int> m_timeout_calls;
    int m_stop_after;
    std::atomic<bool> &m_exit_flag;
    int m_next_calls = 0;
    int m_no_frames = 0;
};

/// A temporary directory (test_files.hpp) made the current directory while it lives, so that the
/// viewer's snapshots land there; the directory is removed after the current directory is
/// changed back.
class snapshot_directory
{
public:
    /// Makes the directory and changes into it.
    snapshot_directory()
        : m_previous(std::filesystem::current_path())
        , m_directory("test_viewer_loop")
    {
        std::filesystem::current_path(m_directory.path());
    }

    snapshot_directory(const snapshot_directory &) = delete;
    snapshot_directory &operator=(const snapshot_directory &) = delete;

    /// Changes back; the directory is removed after.
    ~snapshot_directory()
    {
        std::error_code ignored;
        std::filesystem::current_path(m_previous, ignored);
    }

    /// Lists the files the directory holds.
    /// @return their names, sorted.
    std::vector<std::string> files() const
    {
        std::vector<std::string> names;
        for (const auto &entry : std::filesystem::directory_iterator(m_directory.path()))
            names.push_back(entry.path().filename().string());
        std::ranges::sort(names);
        return names;
    }

private:
    std::filesystem::path m_previous;
    gca_test::temporary_directory m_directory;
};

/// Builds the synthetic source's options: a box on the plane seen along a walk around it.
/// @param pose_count [in] the frames it delivers, >= 0.
/// @param ending [in] what it reports after them.
/// @param message [in] the ending's message.
/// @return the options; nothing when walk_around() refuses the walk.
std::optional<gca_test::synthetic_source_options> scene_options(std::size_t pose_count,
                                                                gca_test::synthetic_ending ending,
                                                                const std::string &message)
{
    // Views 0.40 m to 0.50 m from the box's centre and 0.30 m to 0.40 m above the plane [?: a
    // choice, a hand-held walk around a tabletop box].
    const gca_test::synthetic_walk walk = {
        Eigen::Vector3d(test_box.centre_x_m, test_box.centre_y_m, 0.5 * test_box.height_m),
        pose_count,
        0.45,
        0.05,
        0.35,
        0.05,
        10.0,
        0.02};
    const std::optional<std::vector<Eigen::Isometry3d>> trajectory = gca_test::walk_around(walk);
    if (!trajectory)
        return std::nullopt;
    gca_test::synthetic_source_options options;
    options.description = {test_size, test_intrinsics, test_depth_unit_m};
    options.scene = {.plane = test_plane, .box = test_box, .trajectory = *trajectory};
    options.ending = ending;
    options.ending_message = message;
    return options;
}

/// What one run of the viewer gave.
struct run_output
{
    bool ran = false;               ///< run_viewer()'s result
    std::vector<std::string> lines; ///< the lines it printed, the color codes taken out
    int next_calls = 0;             ///< the calls of next() it made
};

/// How a case runs the viewer, besides the scene.
struct run_plan
{
    int max_frames;                 ///< --frames: the frames to run, -1 until the end
    std::vector<int> timeout_calls; ///< the calls of next(), from 1, that give a timeout
    int stop_after;                 ///< after how many results other than a frame the exit flag
                                    ///< is set; 0 for never
    /// The least time between two report lines: zero for a report after every frame.
    std::chrono::steady_clock::duration report_period = std::chrono::steady_clock::duration::zero();
};

/// Runs the viewer headless on the synthetic source in the current directory.
/// @param options [in] the synthetic source's options.
/// @param plan [in] the frame limit, the timeouts put in, when the exit flag is set, the report
///        period.
/// @return what the run gave and printed.
run_output run(const gca_test::synthetic_source_options &options, const run_plan &plan)
{
    std::atomic<bool> exit_flag{false};
    scripted_source source(options, plan.timeout_calls, plan.stop_after, exit_flag);
    run_output output;
    if (!source.open(test_size).opened)
    {
        check(false, "the synthetic source opens");
        return output;
    }
    gca_viewer::options opt;
    opt.headless = true;
    opt.max_frames = plan.max_frames;
    std::ostringstream printed;
    output.ran = gca_viewer::run_viewer(
        source, opt,
        {.report_period = plan.report_period, .exit_requested = exit_flag, .out = printed});
    output.next_calls = source.next_calls();

    const std::regex color_code("\x1b\\[[0-9;]*m");
    std::istringstream lines(printed.str());
    for (std::string line; std::getline(lines, line);)
    {
        output.lines.push_back(std::regex_replace(line, color_code, ""));
        std::printf("  | %s\n", output.lines.back().c_str());
    }
    return output;
}

/// Reads a report line's point count.
/// @param line [in] the line.
/// @return the points when the line is a report line in today's form ("fps <number> |
///         grab+decode <number> ms | gpu point cloud <number> ms | points <count>"); nothing
///         otherwise.
std::optional<std::size_t> report_points(const std::string &line)
{
    const std::regex report(R"(fps [0-9.e+]+ \| grab\+decode [0-9.e+-]+ ms \| )"
                            R"(gpu point cloud [0-9.e+-]+ ms \| points ([0-9]+))");
    std::smatch match;
    if (!std::regex_match(line, match, report))
        return std::nullopt;
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

/// Says whether a run of lines are all report lines in today's form.
/// @param lines [in] the lines.
/// @param first [in] the first line looked at.
/// @param count [in] the lines looked at.
/// @return whether all of them are report lines and lie within the lines.
bool reports(const std::vector<std::string> &lines, std::size_t first, std::size_t count)
{
    if (first + count > lines.size())
        return false;
    return std::all_of(lines.begin() + static_cast<std::ptrdiff_t>(first),
                       lines.begin() + static_cast<std::ptrdiff_t>(first + count),
                       [](const std::string &line) { return report_points(line).has_value(); });
}

/// Renders the frame of a given index again, from a source of the same options.
/// @param options [in] the synthetic source's options.
/// @param index [in] the frame's index.
/// @param out [out] the frame.
/// @return whether the source gave it.
bool render_frame(const gca_test::synthetic_source_options &options, std::uint64_t index,
                  gca::frame &out)
{
    gca_test::synthetic_source source(options);
    if (!source.open(test_size).opened)
        return false;
    for (std::uint64_t i = 0; i <= index; ++i)
    {
        if (source.next(out).status != gca::next_status::frame)
            return false;
    }
    return true;
}

/// One vertex of the viewer's PLY.
struct ply_vertex
{
    float x, y, z;                 ///< the point in the camera frame, in m
    std::uint8_t red, green, blue; ///< its color
};

/// Reads the viewer's PLY and checks its header against the header the viewer writes.
/// @param path [in] the file.
/// @return the vertices; nothing when the file is missing, its header is not the viewer's, or
///         its length is not the header's and 15 bytes for each vertex.
std::optional<std::vector<ply_vertex>> read_viewer_ply(const std::string &path)
{
    const std::string bytes = gca_test::bytes_of(path);
    const std::string end = "end_header\n";
    const std::size_t header_end = bytes.find(end);
    const std::string count_key = "element vertex ";
    const std::size_t count_at = bytes.find(count_key);
    if (header_end == std::string::npos || count_at == std::string::npos || count_at > header_end)
        return std::nullopt;
    const std::size_t count =
        std::strtoull(bytes.c_str() + count_at + count_key.size(), nullptr, 10);
    const std::string header = "ply\nformat binary_little_endian 1.0\nelement vertex " +
                               std::to_string(count) +
                               "\nproperty float x\nproperty float y\nproperty float z\n"
                               "property uchar red\nproperty uchar green\nproperty uchar blue\n" +
                               end;
    if (bytes.compare(0, header.size(), header) != 0 ||
        bytes.size() != header.size() + count * ply_vertex_bytes)
        return std::nullopt;
    std::vector<ply_vertex> vertices(count);
    const char *data = bytes.data() + header.size();
    for (ply_vertex &vertex : vertices)
    {
        float xyz[3];
        std::memcpy(xyz, data, sizeof(xyz));
        vertex = {xyz[0],
                  xyz[1],
                  xyz[2],
                  static_cast<std::uint8_t>(data[12]),
                  static_cast<std::uint8_t>(data[13]),
                  static_cast<std::uint8_t>(data[14])};
        data += ply_vertex_bytes;
    }
    return vertices;
}

/// Counts the vertices whose color is that of no pixel next to where they project. The
/// prototype's kernel deprojects pixel (u, v) at (u - 0.5, v - 0.5) (AGENTS.md §6) and takes the
/// color of pixel (int(x' + 0.5), int(y' + 0.5)), where (x', y') is the point projected back, so
/// pixel (u, v) up to the floats' rounding; its color passes through 1/255 and back, which can
/// take one off a channel [S: cuda_point_cloud_factory.cu; viewer_loop.cpp, save_ply]. So a vertex
/// passes when one of the 3 x 3 pixels around that pixel has, in each channel, the vertex's value
/// or one more.
/// @param vertices [in] the PLY's vertices.
/// @param frame [in] the frame they were built from.
/// @return the vertices that pass at no such pixel.
std::size_t vertices_off_their_pixel(const std::vector<ply_vertex> &vertices,
                                     const gca::frame &frame)
{
    const gca::intrinsics &in = frame.description().intrinsics;
    const auto width = static_cast<long>(frame.description().size.width);
    const auto height = static_cast<long>(frame.description().size.height);
    const auto color = frame.color();
    std::size_t off = 0;
    for (const ply_vertex &vertex : vertices)
    {
        // The pixel the kernel takes, give or take the rounding of the half.
        const long u = std::lround(vertex.x * in.fx / vertex.z + in.cx + 0.5);
        const long v = std::lround(vertex.y * in.fy / vertex.z + in.cy + 0.5);
        bool found = false;
        for (long dv = -1; dv <= 1 && !found; ++dv)
        {
            for (long du = -1; du <= 1 && !found; ++du)
            {
                if (u + du < 0 || u + du >= width || v + dv < 0 || v + dv >= height)
                    continue;
                const std::size_t at = 3 * static_cast<std::size_t>((v + dv) * width + u + du);
                const auto near = [](std::uint8_t got, std::uint8_t pixel) {
                    return got == pixel || got + 1 == pixel;
                };
                found = near(vertex.red, color[at]) && near(vertex.green, color[at + 1]) &&
                        near(vertex.blue, color[at + 2]);
            }
        }
        off += found ? 0 : 1;
    }
    return off;
}

/// Checks the snapshot of a frame: its color PNG and depth PNG against the frame, its PLY's
/// header, length and colors, and its vertex count against the report's points.
/// @param snapshot [in] the PLY's name; the PNGs are named after it.
/// @param frame [in] the frame it should hold.
/// @param points [in] the points the last report line and the "Saved" line gave.
void check_snapshot(const std::string &snapshot, const gca::frame &frame, std::size_t points)
{
    const auto w = static_cast<int>(test_size.width);
    const auto h = static_cast<int>(test_size.height);
    const cv::Mat color = cv::imread(snapshot + ".color.png", cv::IMREAD_COLOR);
    const cv::Mat depth = cv::imread(snapshot + ".depth.png", cv::IMREAD_ANYDEPTH);
    check(color.type() == CV_8UC3 && color.cols == w && color.rows == h,
          "snapshot: the color PNG has 8 bits in 3 channels and the frame's size");
    check(depth.type() == CV_16UC1 && depth.cols == w && depth.rows == h,
          "snapshot: the depth PNG has 16 bits and the frame's size");
    if (color.type() == CV_8UC3 && depth.type() == CV_16UC1 && color.cols == w && color.rows == h &&
        depth.cols == w && depth.rows == h)
    {
        std::size_t color_off = 0;
        std::size_t depth_off = 0;
        for (int v = 0; v < h; ++v)
        {
            for (int u = 0; u < w; ++u)
            {
                const std::size_t at = static_cast<std::size_t>(v) * test_size.width + u;
                const cv::Vec3b bgr = color.at<cv::Vec3b>(v, u);
                color_off += bgr[2] != frame.color()[3 * at] ||
                                     bgr[1] != frame.color()[3 * at + 1] ||
                                     bgr[0] != frame.color()[3 * at + 2]
                                 ? 1
                                 : 0;
                depth_off += depth.at<std::uint16_t>(v, u) != frame.depth()[at] ? 1 : 0;
            }
        }
        std::printf("snapshot: pixels of other color %zu, of other depth %zu\n", color_off,
                    depth_off);
        check(color_off == 0, "snapshot: the color PNG holds the frame's colors, red, green, blue");
        check(depth_off == 0, "snapshot: the depth PNG holds the frame's depth");
    }

    const std::optional<std::vector<ply_vertex>> vertices = read_viewer_ply(snapshot);
    check(vertices.has_value(),
          "snapshot: the PLY has the viewer's header and 15 bytes for each vertex");
    if (!vertices)
        return;
    const std::size_t off = vertices_off_their_pixel(*vertices, frame);
    std::printf("snapshot: %zu vertices, %zu of them with no pixel's color near them\n",
                vertices->size(), off);
    check(vertices->size() == points && points > 0,
          "snapshot: the PLY holds the report's points, more than none");
    check(off == 0, "snapshot: each vertex has the color of a pixel next to its projection");
}

/// Three frames headless after the first: the intrinsics line, three report lines, the snapshot
/// of the third frame, and nothing else.
void check_frames_and_snapshot()
{
    std::printf("three frames, headless:\n");
    const auto options = scene_options(5, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "three frames: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    const run_output output =
        run(*options, {.max_frames = 3, .timeout_calls = {}, .stop_after = 0});
    check(output.ran, "three frames: the viewer ran");
    check(output.lines.size() == 5, "three frames: five lines");
    check(output.next_calls == 4, "three frames: the first frame and three more taken");
    if (output.lines.size() != 5)
        return;
    check(output.lines[0] == intrinsics_line, "three frames: the intrinsics line, word for word");
    check(reports(output.lines, 1, 3),
          "three frames: a report line in today's form after each frame");
    const std::optional<std::size_t> points = report_points(output.lines[3]);
    const std::string saved =
        "Saved " + std::to_string(points.value_or(0)) + " points to cloud_000.ply";
    check(output.lines[4] == saved, "three frames: \"Saved <the report's points> points to "
                                    "cloud_000.ply\"");
    const std::vector<std::string> expected_files = {"cloud_000.ply", "cloud_000.ply.color.png",
                                                     "cloud_000.ply.depth.png"};
    check(directory.files() == expected_files, "three frames: the snapshot's three files, no "
                                               "others");
    gca::frame third;
    // The first frame, index 0, sets the viewer up; the third frame after it has index 3.
    check(render_frame(*options, 3, third), "three frames: the third frame rendered again");
    check_snapshot("cloud_000.ply", third, points.value_or(0));
}

/// Timeouts before the first frame and between frames: the first frame is waited for without a
/// line, each later timeout prints its line and counts no frame, and the third delivered frame,
/// not the third turn of the loop, is the last and the snapshot's.
void check_timeouts_between_frames()
{
    std::printf("timeouts before the first frame and between frames:\n");
    const auto options = scene_options(5, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "timeouts between frames: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    // Calls 1 and 2 time out, 3 gives the first frame (index 0), 4 frame 1, 5 a timeout, 6 frame
    // 2, 7 a timeout, 8 frame 3.
    const run_output output =
        run(*options, {.max_frames = 3, .timeout_calls = {1, 2, 5, 7}, .stop_after = 0});
    check(output.ran, "timeouts between frames: the viewer waited for the first frame and ran");
    check(output.next_calls == 8, "timeouts between frames: next() taken eight times");
    const bool lines_as_expected = output.lines.size() == 7 && output.lines[0] == intrinsics_line &&
                                   reports(output.lines, 1, 1) && output.lines[2] == timeout_line &&
                                   reports(output.lines, 3, 1) && output.lines[4] == timeout_line &&
                                   reports(output.lines, 5, 1);
    check(lines_as_expected, "timeouts between frames: the intrinsics line, then a report after "
                             "each frame and the timeout line after each timeout");
    if (!lines_as_expected)
        return;
    const std::optional<std::size_t> points = report_points(output.lines[5]);
    check(output.lines[6] ==
              "Saved " + std::to_string(points.value_or(0)) + " points to cloud_000.ply",
          "timeouts between frames: the snapshot after the third delivered frame");
    gca::frame third;
    check(render_frame(*options, 3, third),
          "timeouts between frames: the third frame after the first rendered again");
    check_snapshot("cloud_000.ply", third, points.value_or(0));
}

/// A report period longer than the run: no report line, the snapshot of the last frame still
/// saved.
void check_long_report_period()
{
    std::printf("a report period longer than the run:\n");
    const auto options = scene_options(3, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "long period: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    // An hour, far longer than the run [?: a choice].
    const run_output output = run(*options, {.max_frames = 2,
                                             .timeout_calls = {},
                                             .stop_after = 0,
                                             .report_period = std::chrono::hours(1)});
    check(output.ran, "long period: the viewer ran");
    const std::regex saved_line("Saved ([0-9]+) points to cloud_000\\.ply");
    std::smatch saved;
    const bool lines_as_expected = output.lines.size() == 2 && output.lines[0] == intrinsics_line &&
                                   std::regex_match(output.lines[1], saved, saved_line);
    check(lines_as_expected,
          "long period: the intrinsics line, then only \"Saved <N> points to cloud_000.ply\"");
    if (!lines_as_expected)
        return;
    gca::frame second;
    check(render_frame(*options, 2, second),
          "long period: the second frame after the first rendered again");
    check_snapshot("cloud_000.ply", second, std::stoull(saved[1].str()));
}

/// A timeout or a failure after two frames: its line each time, no frame shown, counted or saved,
/// the loop going on until the exit flag.
/// @param ending [in] timeout or failure.
/// @param name [in] the case, printed in its check lines.
/// @param line [in] the line expected for each.
void check_no_frame(gca_test::synthetic_ending ending, const std::string &name,
                    const std::string &line)
{
    std::printf("%s after two frames:\n", name.c_str());
    const auto options = scene_options(3, ending, "the test's failure");
    check(options.has_value(), (name + ": walk_around() gives the walk").c_str());
    if (!options)
        return;
    const snapshot_directory directory;
    const run_output output =
        run(*options, {.max_frames = 5, .timeout_calls = {}, .stop_after = 2});
    check(output.ran, (name + ": the viewer ran").c_str());
    check(output.next_calls == 5,
          (name + ": next() taken until the exit flag, then no more").c_str());
    const bool lines_as_expected = output.lines.size() == 5 && output.lines[0] == intrinsics_line &&
                                   reports(output.lines, 1, 2) && output.lines[3] == line &&
                                   output.lines[4] == line;
    check(lines_as_expected,
          (name +
           ": the intrinsics line, a report after each of two frames, then the line for "
           "each " +
           name + " and no report")
              .c_str());
    check(directory.files().empty(), (name + ": no snapshot").c_str());
}

/// The source's end after two frames, without a frame limit: its message, and the loop ends.
void check_end()
{
    std::printf("the end after two frames:\n");
    const auto options = scene_options(3, gca_test::synthetic_ending::end, "the test's end");
    check(options.has_value(), "end: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    const run_output output =
        run(*options, {.max_frames = -1, .timeout_calls = {}, .stop_after = 2});
    check(output.ran, "end: the viewer ran");
    check(output.next_calls == 4, "end: next() not taken after the end");
    check(output.lines.size() == 4 && output.lines[0] == intrinsics_line &&
              reports(output.lines, 1, 2) && output.lines[3] == "the test's end",
          "end: the intrinsics line, a report after each of two frames, the end's message");
    check(directory.files().empty(), "end: no snapshot");
}

/// A failure before the first frame: its message and "Couldn't start stream!", nothing else.
void check_no_first_frame()
{
    std::printf("a failure before the first frame:\n");
    const auto options = scene_options(0, gca_test::synthetic_ending::failure, "no picture");
    check(options.has_value(), "no first frame: walk_around() gives the empty walk");
    if (!options)
        return;
    const snapshot_directory directory;
    const run_output output =
        run(*options, {.max_frames = 3, .timeout_calls = {}, .stop_after = 2});
    check(!output.ran, "no first frame: the viewer gives up");
    check(output.lines == std::vector<std::string>{"no picture", "Couldn't start stream!"},
          "no first frame: the failure's message, then \"Couldn't start stream!\"");
    check(directory.files().empty(), "no first frame: no snapshot");
}
} // namespace

/// Runs the cases.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    check_frames_and_snapshot();
    check_timeouts_between_frames();
    check_long_report_period();
    check_no_frame(gca_test::synthetic_ending::timeout, "timeout", timeout_line);
    check_no_frame(gca_test::synthetic_ending::failure, "failure", "the test's failure");
    check_end();
    check_no_first_frame();
    return gca_test::check_summary("test_viewer_loop");
}
