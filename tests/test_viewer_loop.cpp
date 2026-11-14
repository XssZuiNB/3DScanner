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
/// Without a sequence every report line is in today's form, with nothing after the points. With
/// one, written into a second temporary directory:
/// - three frames and a timeout: the sequence read back holds the first frame and the three
///   after it, bit for bit, each marked as tracked by nothing; the report lines end in
///   " | queue full 0", the snapshot is as without a sequence, and "Wrote 4 frames to <file>, 0
///   not written because the writer's queue was full" closes the run; each frame is pushed once;
/// - the exit flag (Ctrl-C) after a timeout: the last frame taken is still written, and the
///   closing line follows;
/// - pushes answered by a full queue (a scripted sink: a real writer is not full at this pace):
///   each counted in the report after it, the last one's in the closing line, and those frames,
///   and only those, missing from the sequence;
/// - a file in the way: the writer's reason, and the viewer gives up after the first frame;
/// - a push that fails (scripted): its reason with the counts, and the loop ends there; the
///   sequence closes with the frames before;
/// - a disk that refuses (a limit on the file size): the writer's reason with the counts, once,
///   last, the frames lost from the queue counted; the sequence holds the frames before and then
///   ends early.
/// Needs the GPU, no camera; links no OrbbecSDK.

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/posix_file.hpp"
#include "camera/sequence_format.hpp"
#include "camera/sequence_reader.hpp"
#include "camera/sequence_writer.hpp"
#include "check.hpp"
#include "frame_copy.hpp"
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
#include <variant>
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

/// The bytes of one record of a frame without a pose in the sequence: its kind, index and
/// timestamp, then two bytes of depth and three of color for each pixel
/// [S: docs/SEQUENCE_FILE.md, "Layout"].
constexpr std::size_t sequence_record_bytes =
    1 + 8 + 8 + 5 * static_cast<std::size_t>(test_size.width) * test_size.height;

/// The report line in today's form, the points captured: "fps <number> | grab+decode <number> ms
/// | gpu point cloud <number> ms | points <count>".
constexpr const char *report_pattern = R"(fps [0-9.e+]+ \| grab\+decode [0-9.e+-]+ ms \| )"
                                       R"(gpu point cloud [0-9.e+-]+ ms \| points ([0-9]+))";

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

/// The message of the push failure a test puts in.
constexpr const char *test_push_failure = "the test's push failure";

/// The viewer's sequence in a directory of the test's, except the pushes a test names, which are
/// answered with a full queue or a failure without asking the writer, as scripted_source answers
/// chosen calls with a timeout. It counts the calls.
class scripted_sink final : public gca_viewer::sequence_sink
{
public:
    /// Names the sequence; nothing is created until open().
    /// @param directory [in] the directory the sequence is written to.
    /// @param full_pushes [in] the calls of push(), counted from 1, that find the queue full.
    /// @param failing_push [in] the call of push(), counted from 1, that fails with
    ///        test_push_failure; 0 for none.
    scripted_sink(const std::filesystem::path &directory, std::vector<int> full_pushes,
                  int failing_push)
        : m_file(directory)
        , m_full_pushes(std::move(full_pushes))
        , m_failing_push(failing_push)
    {
    }

    /// Opens the viewer's sequence.
    /// @param description [in] the frames' description.
    /// @return the sequence's result.
    gca::open_result open(const gca::frame_description &description) noexcept override
    {
        return m_file.open(description);
    }

    /// Counts the call; answers a full queue or the failure on a call the test named, else the
    /// sequence's result.
    /// @param in_out [in,out] as the sequence leaves it; as it was when full or failed.
    /// @return full, the failure, or the sequence's result.
    gca::push_result push(gca::frame &in_out) noexcept override
    {
        m_pushes++;
        if (std::ranges::find(m_full_pushes, m_pushes) != m_full_pushes.end())
            return {gca::push_status::full, {}};
        if (m_pushes == m_failing_push)
            return {gca::push_status::failure, test_push_failure};
        return m_file.push(in_out);
    }

    /// Counts the call and closes the sequence.
    /// @return the sequence's result.
    gca::write_result close() noexcept override
    {
        m_closes++;
        return m_file.close();
    }

    /// Tells how many frames the sequence wrote whole.
    /// @return the sequence's count.
    std::uint64_t frames_written() noexcept override
    {
        return m_file.frames_written();
    }

    /// Gives the sequence's file.
    /// @return its path.
    const std::filesystem::path &path() const noexcept override
    {
        return m_file.path();
    }

    /// Gives the calls of push() so far.
    /// @return their count.
    int pushes() const noexcept
    {
        return m_pushes;
    }

    /// Gives the calls of close() so far.
    /// @return their count.
    int closes() const noexcept
    {
        return m_closes;
    }

private:
    gca_viewer::sequence_file m_file;
    std::vector<int> m_full_pushes;
    int m_failing_push;
    int m_pushes = 0;
    int m_closes = 0;
};

/// Lists the files a directory holds.
/// @param directory [in] the directory.
/// @return their names, sorted.
std::vector<std::string> files_in(const std::filesystem::path &directory)
{
    std::vector<std::string> names;
    for (const auto &entry : std::filesystem::directory_iterator(directory))
        names.push_back(entry.path().filename().string());
    std::ranges::sort(names);
    return names;
}

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
        return files_in(m_directory.path());
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
    /// Where the frames taken are written; nullptr for no sequence.
    gca_viewer::sequence_sink *sequence = nullptr;
};

/// Runs the viewer headless on the synthetic source in the current directory.
/// @param options [in] the synthetic source's options.
/// @param plan [in] the frame limit, the timeouts put in, when the exit flag is set, the report
///        period, the sequence.
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
    output.ran = gca_viewer::run_viewer(source, opt,
                                        {.report_period = plan.report_period,
                                         .exit_requested = exit_flag,
                                         .out = printed,
                                         .sequence = plan.sequence});
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
    const std::regex report(report_pattern);
    std::smatch match;
    if (!std::regex_match(line, match, report))
        return std::nullopt;
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

/// A report line's numbers in the form with a sequence.
struct sequence_report
{
    std::size_t points; ///< the point count
    int queue_full;     ///< the pushes since the last report that found the queue full
};

/// Reads a report line in the form with a sequence: today's form, then " | queue full <count>".
/// @param line [in] the line.
/// @return its numbers; nothing when the line is not in that form.
std::optional<sequence_report> read_sequence_report(const std::string &line)
{
    const std::regex report(std::string(report_pattern) + R"( \| queue full ([0-9]+))");
    std::smatch match;
    if (!std::regex_match(line, match, report))
        return std::nullopt;
    return sequence_report{static_cast<std::size_t>(std::stoull(match[1].str())),
                           std::stoi(match[2].str())};
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

/// Reads the viewer's sequence back and compares it with the synthetic source's frames, rendered
/// again.
/// @param name [in] the case, for the check lines.
/// @param path [in] the sequence's file.
/// @param options [in] the synthetic source's options.
/// @param indices [in] the indices of the frames the sequence should hold, in order.
/// @param ends_early [in] whether the file should end early after them, its writer cut off; else
///        its end follows them.
void check_read_back(const std::string &name, const std::filesystem::path &path,
                     const gca_test::synthetic_source_options &options,
                     const std::vector<std::uint64_t> &indices, bool ends_early)
{
    std::vector<gca_test::frame_copy> rendered; // the source's frames, by index
    {
        gca_test::synthetic_source source(options);
        check(source.open(test_size).opened, (name + ": the synthetic source opens again").c_str());
        gca::frame one;
        const std::size_t count = indices.empty() ? 0 : indices.back() + 1;
        while (rendered.size() < count && source.next(one).status == gca::next_status::frame)
            rendered.push_back(gca_test::copy_of(one));
    }
    gca::sequence_reader reader(path);
    const gca::open_result opened = reader.open(test_size);
    check(opened.opened, (name + ": the sequence opens; " + opened.refusal).c_str());
    gca::frame read;
    std::vector<std::uint64_t> read_indices;
    std::size_t equal = 0;
    gca::next_result result;
    while ((result = reader.next(read)).status == gca::next_status::frame)
    {
        read_indices.push_back(read.index());
        const bool same = read.index() < rendered.size() &&
                          gca_test::holds(read, rendered[read.index()]) &&
                          gca::same_description(read.description(), options.description) &&
                          std::holds_alternative<gca::not_tracked>(reader.tracking());
        equal += same ? 1 : 0;
    }
    std::string listed;
    for (const std::uint64_t index : read_indices)
        listed += " " + std::to_string(index);
    std::printf("%s: the sequence holds frames%s, %zu of them equal; then %s\n", name.c_str(),
                listed.c_str(), equal, result.message.c_str());
    check(read_indices == indices,
          (name + ": the sequence holds the frames of the expected indices, in order").c_str());
    check(equal == read_indices.size(),
          (name + ": each frame the source's, bit for bit, with its description, marked as "
                  "tracked by nothing")
              .c_str());
    if (ends_early)
        check(result.status == gca::next_status::failure && reader.ended_early(),
              (name + ": then the sequence ends early").c_str());
    else
        check(result.status == gca::next_status::end, (name + ": then the sequence's end").c_str());
}

/// The line that closes a sequence written whole.
/// @param sink [in] the sequence.
/// @param written [in] the frames written.
/// @param full [in] the frames not written, the queue being full.
/// @return the line.
std::string closing_line(const gca_viewer::sequence_sink &sink, int written, int full)
{
    return "Wrote " + std::to_string(written) + " frames to " + sink.path().string() + ", " +
           std::to_string(full) + " not written because the writer's queue was full";
}

/// The counts that follow a failure's reason in the line that closes a sequence: "; <written> of
/// <taken> frames taken written, <lost> lost from the writer's queue, <full> not written because
/// the writer's queue was full".
struct failure_counts
{
    int written; ///< the frames written whole
    int taken;   ///< the frames taken from the source
    int lost;    ///< the frames the writer accepted and lost
    int full;    ///< the frames not written, the queue being full
};

/// Reads the line that closes a sequence after a failure.
/// @param line [in] the line.
/// @param reason [in] the reason it should start with.
/// @return its counts; nothing when it is not the reason followed by the counts.
std::optional<failure_counts> read_failure_line(const std::string &line, const std::string &reason)
{
    if (!line.starts_with(reason))
        return std::nullopt;
    const std::regex counts(R"(; ([0-9]+) of ([0-9]+) frames taken written, ([0-9]+) lost from )"
                            R"(the writer's queue, ([0-9]+) not written because the writer's )"
                            R"(queue was full)");
    std::smatch match;
    const std::string rest = line.substr(reason.size());
    if (!std::regex_match(rest, match, counts))
        return std::nullopt;
    return failure_counts{std::stoi(match[1].str()), std::stoi(match[2].str()),
                          std::stoi(match[3].str()), std::stoi(match[4].str())};
}

/// Three frames and a timeout with a sequence: the first frame and the three after it written bit
/// for bit, the report lines with " | queue full 0", the snapshot as without a sequence, the
/// closing line; each frame pushed once.
void check_sequence_written()
{
    std::printf("three frames and a timeout, with a sequence:\n");
    const auto options = scene_options(5, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "sequence: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    const gca_test::temporary_directory sequence_directory("test_viewer_loop");
    scripted_sink sink(sequence_directory.path(), {}, 0);
    // Call 1 gives the first frame (index 0), 2 frame 1, 3 a timeout, 4 frame 2, 5 frame 3.
    const run_output output =
        run(*options, {.max_frames = 3, .timeout_calls = {3}, .stop_after = 0, .sequence = &sink});
    check(output.ran, "sequence: the viewer ran and wrote the sequence");
    std::vector<std::optional<sequence_report>> reports_read;
    for (const std::size_t at : {std::size_t{1}, std::size_t{3}, std::size_t{4}})
        reports_read.push_back(at < output.lines.size() ? read_sequence_report(output.lines[at])
                                                        : std::nullopt);
    const bool lines_as_expected =
        output.lines.size() == 7 && output.lines[0] == intrinsics_line &&
        std::ranges::all_of(reports_read,
                            [](const auto &report) { return report && report->queue_full == 0; }) &&
        output.lines[2] == timeout_line &&
        output.lines[5] == "Saved " +
                               std::to_string(reports_read[2] ? reports_read[2]->points : 0) +
                               " points to cloud_000.ply" &&
        output.lines[6] == closing_line(sink, 4, 0);
    check(lines_as_expected,
          "sequence: the intrinsics line, reports ending in \" | queue full 0\", the timeout line, "
          "\"Saved ...\", then \"Wrote 4 frames to <file>, 0 not written because ...\"");
    check(sink.pushes() == 4 && sink.closes() == 1,
          "sequence: each of the four frames pushed once, the sequence closed once");
    check(files_in(sequence_directory.path()) ==
              std::vector<std::string>{gca_viewer::sequence_file_name},
          "sequence: the sequence's file alone in its directory");
    if (!lines_as_expected)
        return;
    gca::frame third;
    check(render_frame(*options, 3, third), "sequence: the third frame rendered again");
    check_snapshot("cloud_000.ply", third, reports_read[2]->points);
    check_read_back("sequence", sink.path(), *options, {0, 1, 2, 3}, false);
}

/// Pushes answered by a full queue: each counted in the report line after it, the last frame's in
/// the closing line; those frames, and only those, missing from the sequence.
void check_full_queue()
{
    std::printf("pushes that find the queue full:\n");
    const auto options = scene_options(7, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "full: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    const gca_test::temporary_directory sequence_directory("test_viewer_loop");
    // Push k is frame k - 1's, made when frame k comes, the sixth after the loop: frames 1 and 3
    // are counted in the reports after frames 2 and 4, frame 5 in the closing line.
    scripted_sink sink(sequence_directory.path(), {2, 4, 6}, 0);
    const run_output output =
        run(*options, {.max_frames = 5, .timeout_calls = {}, .stop_after = 0, .sequence = &sink});
    check(output.ran, "full: the viewer ran and wrote the sequence");
    std::vector<int> counts;
    std::size_t points = 0;
    for (std::size_t at = 1; at < output.lines.size() && at <= 5; ++at)
    {
        if (const auto report = read_sequence_report(output.lines[at]))
        {
            counts.push_back(report->queue_full);
            points = report->points;
        }
    }
    const bool lines_as_expected =
        output.lines.size() == 8 && output.lines[0] == intrinsics_line &&
        counts == std::vector<int>{0, 1, 0, 1, 0} &&
        output.lines[6] == "Saved " + std::to_string(points) + " points to cloud_000.ply" &&
        output.lines[7] == closing_line(sink, 3, 3);
    check(lines_as_expected,
          "full: reports counting 0, 1, 0, 1, 0 full pushes, \"Saved ...\", then \"Wrote 3 "
          "frames to <file>, 3 not written because ...\"");
    check(sink.pushes() == 6 && sink.closes() == 1,
          "full: each of the six frames pushed once, the sequence closed once");
    if (!lines_as_expected)
        return;
    gca::frame fifth;
    check(render_frame(*options, 5, fifth), "full: the fifth frame rendered again");
    check_snapshot("cloud_000.ply", fifth, points);
    check_read_back("full", sink.path(), *options, {0, 2, 4}, false);
}

/// The exit flag after a timeout, as Ctrl-C sets it: the loop ends, and the last frame taken is
/// still written before the closing line.
void check_exit_with_sequence()
{
    std::printf("the exit flag with a sequence:\n");
    const auto options = scene_options(6, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "exit: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    const gca_test::temporary_directory sequence_directory("test_viewer_loop");
    scripted_sink sink(sequence_directory.path(), {}, 0);
    // Calls 1 to 3 give frames 0 to 2, call 4 a timeout, after which the flag is set.
    const run_output output =
        run(*options, {.max_frames = -1, .timeout_calls = {4}, .stop_after = 1, .sequence = &sink});
    check(output.ran, "exit: the viewer ran and wrote the sequence");
    check(output.next_calls == 4, "exit: no frame taken after the flag");
    const bool lines_as_expected = output.lines.size() == 5 && output.lines[0] == intrinsics_line &&
                                   read_sequence_report(output.lines[1]).has_value() &&
                                   read_sequence_report(output.lines[2]).has_value() &&
                                   output.lines[3] == timeout_line &&
                                   output.lines[4] == closing_line(sink, 3, 0);
    check(lines_as_expected, "exit: the intrinsics line, two reports, the timeout line, then "
                             "\"Wrote 3 frames to <file>, 0 not written because ...\"");
    check(sink.pushes() == 3 && sink.closes() == 1,
          "exit: each of the three frames pushed once, the last after the loop");
    check_read_back("exit", sink.path(), *options, {0, 1, 2}, false);
}

/// A file where the sequence goes: the writer's reason after the intrinsics line, and the viewer
/// gives up before its loop, the file left as it was.
void check_sequence_in_the_way()
{
    std::printf("a file where the sequence goes:\n");
    const auto options = scene_options(3, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "in the way: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    const gca_test::temporary_directory sequence_directory("test_viewer_loop");
    gca_viewer::sequence_file sequence(sequence_directory.path());
    gca_test::write_bytes(sequence.path(), "a sequence from before");
    const run_output output = run(
        *options, {.max_frames = 2, .timeout_calls = {}, .stop_after = 0, .sequence = &sequence});
    check(!output.ran, "in the way: the viewer gives up");
    check(output.lines ==
              std::vector<std::string>{intrinsics_line,
                                       sequence.path().string() + ": cannot create: File exists"},
          "in the way: the intrinsics line, then the writer's reason");
    check(output.next_calls == 1, "in the way: no frame taken after the first");
    check(gca_test::bytes_of(sequence.path()) == "a sequence from before",
          "in the way: the file left as it was");
    check(directory.files().empty(), "in the way: no snapshot");
}

/// A push that fails, the third: its reason, and the loop ends there, the frame that came with it
/// neither shown nor pushed, and no frame pushed after; the sequence closes whole with the two
/// frames before, and the viewer reports the failure.
void check_push_fails()
{
    std::printf("a push that fails:\n");
    const auto options = scene_options(7, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "push fails: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    const gca_test::temporary_directory sequence_directory("test_viewer_loop");
    scripted_sink sink(sequence_directory.path(), {}, 3);
    // Push 3 is frame 2's, made when frame 3 comes.
    const run_output output =
        run(*options, {.max_frames = 5, .timeout_calls = {}, .stop_after = 0, .sequence = &sink});
    check(!output.ran, "push fails: the viewer reports the failure");
    check(output.next_calls == 4, "push fails: no frame taken after the one the push came with");
    const bool lines_as_expected = output.lines.size() == 4 && output.lines[0] == intrinsics_line &&
                                   read_sequence_report(output.lines[1]).has_value() &&
                                   read_sequence_report(output.lines[2]).has_value() &&
                                   output.lines[3] == std::string(test_push_failure) +
                                                          "; 2 of 4 frames taken written, 0 lost "
                                                          "from the writer's queue, 0 not written "
                                                          "because the writer's queue was full";
    check(lines_as_expected, "push fails: the intrinsics line, a report after each of two frames, "
                             "then the push's reason with the counts");
    check(sink.pushes() == 3 && sink.closes() == 1,
          "push fails: no push after the failed one, the sequence closed once");
    check(directory.files().empty(), "push fails: no snapshot");
    check_read_back("push fails", sink.path(), *options, {0, 1}, false);
}

/// A disk that refuses the third frame's record: the writer's reason, once and last, no closing
/// line, and the viewer reports the failure; the sequence holds the first two frames, then ends
/// early.
void check_disk_refuses()
{
    std::printf("a disk that refuses the third frame:\n");
    const auto options = scene_options(8, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "disk: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    const gca_test::temporary_directory sequence_directory("test_viewer_loop");
    scripted_sink sink(sequence_directory.path(), {}, 0);
    run_output output;
    {
        // Room for the header, two records and part of the third [S: docs/SEQUENCE_FILE.md].
        const gca_test::file_size_limit limit(gca::sequence_header_bytes +
                                              2 * sequence_record_bytes + 100);
        output = run(*options,
                     {.max_frames = -1, .timeout_calls = {}, .stop_after = 0, .sequence = &sink});
    }
    const std::string reason =
        sink.path().string() + ": the write of frame index 2 failed: File too large";
    check(!output.ran, "disk: the viewer reports the failure");
    const std::optional<failure_counts> counts =
        output.lines.empty() ? std::nullopt : read_failure_line(output.lines.back(), reason);
    if (counts)
        std::printf("disk: %d of %d frames taken written, %d lost, %d full\n", counts->written,
                    counts->taken, counts->lost, counts->full);
    check(counts && counts->written == 2 && counts->lost >= 1 && counts->full == 0 &&
              counts->taken >= counts->written + counts->lost,
          "disk: the last line the writer's reason, then frames 0 and 1 written, the frame that "
          "failed and those queued after it lost");
    check(std::ranges::count_if(
              output.lines,
              [&](const std::string &line) { return line.find(reason) != std::string::npos; }) == 1,
          "disk: the reason once");
    check(std::ranges::none_of(output.lines,
                               [](const std::string &line) { return line.starts_with("Wrote "); }),
          "disk: no line of a sequence written whole");
    check(sink.closes() == 1, "disk: the sequence closed once");
    check_read_back("disk", sink.path(), *options, {0, 1}, true);
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
    check_sequence_written();
    check_full_queue();
    check_sequence_in_the_way();
    check_push_fails();
    check_disk_refuses();
    check_exit_with_sequence();
    return gca_test::check_summary("test_viewer_loop");
}
