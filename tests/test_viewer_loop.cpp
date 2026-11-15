// SPDX-License-Identifier: 0BSD
/// @file test_viewer_loop.cpp
/// orbbec_viewer's loop (viewer_loop.hpp) driven by the tests' synthetic source, a box on the
/// plane seen along a walk around it, headless, with a report after every frame unless a case
/// says otherwise; each case runs in a temporary directory that the test removes. Its printed
/// lines, the color codes taken out, are checked line by line, so a line that changes its words
/// or appears where it should not shows:
/// - three frames after the first: the intrinsics line word for word, a report line in today's
///   form after each frame, its points the frame's valid pixels, those with depth inside the
///   viewer's range, then the snapshot of the third: "Saved N points to cloud_000.ply" and its
///   three files, no others. Its color PNG holds the frame's colors and its depth PNG the frame's
///   depth, bit for bit; the tests' PLY reader (ply_reader.hpp) reads its PLY, a model's, with N
///   vertices, more than none and at most the report's points, each at its own pixel once: it
///   projects onto the pixel, its z is the pixel's depth value times the unit, its color the
///   pixel's, so colors swapped on their way show, and its normal faces the camera;
/// - timeouts before the first frame and between frames: the first frame is waited for, the
///   frame limit counts delivered frames only, and the snapshot is the third delivered frame's;
/// - a report period longer than the run: no report line, the snapshot still saved;
/// - a timeout and a failure after two frames: their line each time, no stale frame shown,
///   counted or saved, and the loop going on until the exit flag, which the test sets after the
///   second, as the viewer's SIGINT handler would;
/// - the source's end after two frames: its message, and the loop ends;
/// - a failure before the first frame: its message and "Couldn't start stream!", and the viewer
///   gives up;
/// - a GPU step that fails, CUDA's last error left set as a failed call leaves it: before the
///   second frame after the first, the frame's step; after the last frame's report, the snapshot's
///   normals. Each prints the step's reason naming CUDA's error, the loop ends there with no
///   snapshot written, the viewer gives up, and CUDA's last error is clear after. A failure of the
///   GPU's set-up (CUDA's stream, the pool's requests) is not provoked offline: nothing here makes
///   them fail short of exhausting the GPU's memory;
/// - a directory in the way of the snapshot's PLY: the writer's reason, the pictures still saved,
///   and the viewer runs on;
/// - the depth range: a large plane seen at a grazing angle from 5 cm above it, so that the frame
///   holds depths below 0.15 m, between, beyond 3.0 m and none; the report counts and the snapshot
///   holds only those within the range.
/// Without a sequence every report line is in today's form, with nothing after the points. With
/// one, written into a second temporary directory:
/// - three frames and a timeout: the sequence read back holds the first frame and the three
///   after it, bit for bit with their IMU samples, each marked as tracked by nothing; the report
///   lines end in " | queue full 0", the snapshot is as without a sequence, and "Wrote 4 frames
///   to <file>, 0 not written because the writer's queue was full" closes the run; each frame is
///   pushed once;
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
#include "ply_reader.hpp"
#include "synthetic_source.hpp"
#include "test_files.hpp"
#include "types/frame_description.hpp"
#include "viewer_loop.hpp"
#include "viewer_options.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <ostream>
#include <regex>
#include <sstream>
#include <streambuf>
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

/// The viewer's depth range: a pixel is valid within 0.15 m to 3.0 m [S: viewer_loop.hpp,
/// run_viewer()].
constexpr float viewer_near_m = 0.15f;
constexpr float viewer_far_m = 3.0f;

/// How far a snapshot's vertex may project from a whole pixel, in px: float's rounding of the
/// deprojection and of the test's projection, a few times 1e-5 px at 160 px [S: float32's relative
/// step 6e-8], far below the whole pixel a vertex of another pixel lies off [?: a choice].
constexpr double vertex_pixel_tolerance_px = 1e-3;

/// Gives the bytes of a frame's record without a pose in the sequence: its kind, index,
/// timestamp and the counts of its samples, 20 bytes for each sample, then two bytes of depth and
/// three of color for each pixel [S: docs/SEQUENCE_FILE.md, "Layout"].
/// @param one [in] the frame, of the test's size.
/// @return the bytes.
std::size_t sequence_record_bytes(const gca::frame &one)
{
    return 1 + 8 + 8 + 4 + 20 * (one.accel().size() + one.gyro().size()) +
           5 * static_cast<std::size_t>(test_size.width) * test_size.height;
}

/// The report line in today's form, the points captured: "fps <number> | grab+decode <number> ms
/// | gpu point cloud <number> ms | points <count>".
constexpr const char *report_pattern = R"(fps [0-9.e+]+ \| grab\+decode [0-9.e+-]+ ms \| )"
                                       R"(gpu point cloud [0-9.e+-]+ ms \| points ([0-9]+))";

/// Leaves CUDA's last error set, as a failed CUDA call does: CUDA refuses 2^52 bytes and keeps
/// cudaErrorMemoryAllocation as the thread's last error until a call reads it [S:
/// cuda_runtime_api.h, cudaGetLastError, CUDA 12.6].
void leave_cuda_error()
{
    void *never_granted = nullptr;
    check(cudaMalloc(&never_granted, std::size_t{1} << 52) != cudaSuccess,
          "cudaMalloc refuses 2^52 bytes");
}

/// A frame source that passes every call through to the synthetic source, except the calls of
/// next() a test names, which give a timeout instead without asking the synthetic source. Before
/// a call a test names it leaves CUDA's last error set, as a failed CUDA call does. It counts the
/// calls of next(), and once a number of them has given a result other than a frame, it sets the
/// loop's exit flag, as the viewer's SIGINT handler would.
class scripted_source final : public gca::frame_source
{
public:
    /// Builds the synthetic source, closed.
    /// @param options [in] the synthetic source's options; taken over.
    /// @param timeout_calls [in] the calls of next(), counted from 1, that give a timeout.
    /// @param stop_after [in] after how many results other than a frame the flag is set; 0 for
    ///        never.
    /// @param exit_flag [out] the loop's exit flag; must outlive the source.
    /// @param cuda_error_call [in] the call of next(), counted from 1, before which CUDA's last
    ///        error is left set; 0 for none.
    scripted_source(gca_test::synthetic_source_options options, std::vector<int> timeout_calls,
                    int stop_after, std::atomic<bool> &exit_flag, int cuda_error_call)
        : m_source(std::move(options))
        , m_timeout_calls(std::move(timeout_calls))
        , m_stop_after(stop_after)
        , m_exit_flag(exit_flag)
        , m_cuda_error_call(cuda_error_call)
    {
    }

    /// Opens the synthetic source.
    /// @param size [in] the size asked for, in px.
    /// @return the synthetic source's result.
    gca::open_result open(const gca::frame_size &size) noexcept override
    {
        return m_source.open(size);
    }

    /// Counts the call; leaves CUDA's last error set on the call the test named for it; gives a
    /// timeout on a call the test named, else the synthetic source's next result; and sets the
    /// flag after the stop_after-th result other than a frame.
    /// @param out [out] as the synthetic source fills it; left as it was on a timeout.
    /// @return the timeout, or the synthetic source's result.
    gca::next_result next(gca::frame &out) noexcept override
    {
        m_next_calls++;
        if (m_next_calls == m_cuda_error_call)
            leave_cuda_error();
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
    int m_cuda_error_call;
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
    // The IMU at the camera's 200 Hz, so that a sequence holds its samples
    // (docs/methods/imu-samples.md).
    options.imu = gca_test::synthetic_imu{5000};
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
    /// The call of next(), from 1, before which CUDA's last error is left set; 0 for none.
    int cuda_error_call = 0;
    /// The line the viewer prints, from 1, after which CUDA's last error is left set; 0 for none.
    int cuda_error_line = 0;
};

/// The stream the viewer prints to: it keeps the text and, once a chosen line has been printed
/// whole, leaves CUDA's last error set (leave_cuda_error()), so that a GPU step the loop takes
/// after that line, and before the next frame, fails as after a failed CUDA call.
class printed_lines final : public std::streambuf
{
public:
    /// Builds an empty stream buffer.
    /// @param error_line [in] the line, counted from 1, after which CUDA's last error is left set;
    ///        0 for none.
    explicit printed_lines(int error_line)
        : m_error_line(error_line)
    {
    }

    /// Gives what was printed.
    /// @return the text, valid as long as the buffer.
    const std::string &text() const noexcept
    {
        return m_text;
    }

protected:
    /// Takes one character: the buffer has no put area, so every character comes here or to
    /// xsputn().
    /// @param c [in] the character, or EOF for none.
    /// @return c, or a value other than EOF for EOF.
    int_type overflow(int_type c) override
    {
        if (traits_type::eq_int_type(c, traits_type::eof()))
            return traits_type::not_eof(c);
        take(traits_type::to_char_type(c));
        return c;
    }

    /// Takes a run of characters.
    /// @param s [in] the characters.
    /// @param n [in] their count.
    /// @return n, every character taken.
    std::streamsize xsputn(const char *s, std::streamsize n) override
    {
        for (std::streamsize i = 0; i < n; i++)
            take(s[i]);
        return n;
    }

private:
    /// Keeps a character, and leaves CUDA's error after the chosen line's end.
    /// @param c [in] the character.
    void take(char c)
    {
        m_text.push_back(c);
        if (c == '\n' && ++m_lines == m_error_line)
            leave_cuda_error();
    }

    std::string m_text;
    int m_error_line;
    int m_lines = 0;
};

/// Runs the viewer headless on the synthetic source in the current directory.
/// @param options [in] the synthetic source's options.
/// @param plan [in] the frame limit, the timeouts put in, when the exit flag is set, the report
///        period, the sequence, where CUDA's error is put in.
/// @return what the run gave and printed.
run_output run(const gca_test::synthetic_source_options &options, const run_plan &plan)
{
    std::atomic<bool> exit_flag{false};
    scripted_source source(options, plan.timeout_calls, plan.stop_after, exit_flag,
                           plan.cuda_error_call);
    run_output output;
    if (!source.open(test_size).opened)
    {
        check(false, "the synthetic source opens");
        return output;
    }
    gca_viewer::options opt;
    opt.headless = true;
    opt.max_frames = plan.max_frames;
    printed_lines buffer(plan.cuda_error_line);
    std::ostream printed(&buffer);
    output.ran = gca_viewer::run_viewer(source, opt,
                                        {.report_period = plan.report_period,
                                         .exit_requested = exit_flag,
                                         .out = printed,
                                         .sequence = plan.sequence});
    output.next_calls = source.next_calls();

    const std::regex color_code("\x1b\\[[0-9;]*m");
    std::istringstream lines(buffer.text());
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

/// Reads a snapshot's line, "Saved <count> points to cloud_000.ply".
/// @param line [in] the line.
/// @return the count; nothing when the line has another form.
std::optional<std::size_t> saved_points(const std::string &line)
{
    const std::regex saved(R"(Saved ([0-9]+) points to cloud_000\.ply)");
    std::smatch match;
    if (!std::regex_match(line, match, saved))
        return std::nullopt;
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

/// Counts a frame's valid pixels as the viewer's organized points take them: depth value d not 0
/// and its z, d times the unit in float, within the viewer's range [S:
/// geometry/organized_points.hpp, deproject()].
/// @param frame [in] the frame.
/// @return the valid pixels.
std::size_t valid_pixels(const gca::frame &frame)
{
    const float unit = frame.description().depth_unit_m;
    return static_cast<std::size_t>(std::ranges::count_if(frame.depth(), [&](std::uint16_t d) {
        const float z = static_cast<float>(d) * unit;
        return d != 0 && z >= viewer_near_m && z <= viewer_far_m;
    }));
}

/// Counts the snapshot's vertices that are not a valid pixel's point with its color and a normal
/// toward the camera: a vertex has to project onto a whole pixel within vertex_pixel_tolerance_px
/// by the one pinhole convention (types/camera_param.hpp), no other vertex onto the same; its z
/// has to be the pixel's depth value times the unit, in float, bit for bit, and within the
/// viewer's range; its color the pixel's, red, green, blue; its normal n has n . p <= 0.
/// @param vertices [in] the PLY's vertices.
/// @param frame [in] the frame they were made from.
/// @return the vertices that fail any of these.
std::size_t vertices_off_their_pixel(const std::vector<gca_test::ply_vertex> &vertices,
                                     const gca::frame &frame)
{
    const gca::intrinsics &in = frame.description().intrinsics;
    const float unit = frame.description().depth_unit_m;
    const auto width = static_cast<long>(frame.description().size.width);
    const auto height = static_cast<long>(frame.description().size.height);
    const auto color = frame.color();
    const auto depth = frame.depth();
    std::vector<bool> taken(depth.size(), false);
    std::size_t off = 0;
    for (const gca_test::ply_vertex &vertex : vertices)
    {
        const double x = vertex.position_m[0];
        const double y = vertex.position_m[1];
        const double z = vertex.position_m[2];
        const double u = z > 0.0 ? x * in.fx / z + in.cx : -1.0;
        const double v = z > 0.0 ? y * in.fy / z + in.cy : -1.0;
        const long pu = std::lround(u);
        const long pv = std::lround(v);
        if (std::abs(u - static_cast<double>(pu)) > vertex_pixel_tolerance_px ||
            std::abs(v - static_cast<double>(pv)) > vertex_pixel_tolerance_px || pu < 0 ||
            pu >= width || pv < 0 || pv >= height)
        {
            off++;
            continue;
        }
        const auto pixel = static_cast<std::size_t>(pv * width + pu);
        const std::uint16_t d = depth[pixel];
        const float pixel_z = static_cast<float>(d) * unit;
        const bool at_depth = d != 0 && vertex.position_m[2] == pixel_z &&
                              pixel_z >= viewer_near_m && pixel_z <= viewer_far_m;
        const bool colored = vertex.color[0] == color[3 * pixel] &&
                             vertex.color[1] == color[3 * pixel + 1] &&
                             vertex.color[2] == color[3 * pixel + 2];
        const double facing = vertex.normal[0] * x + vertex.normal[1] * y + vertex.normal[2] * z;
        off += !taken[pixel] && at_depth && colored && facing <= 0.0 ? 0 : 1;
        taken[pixel] = true;
    }
    return off;
}

/// Checks the snapshot of a frame: its color PNG and depth PNG against the frame; its PLY, read by
/// the tests' PLY reader, against the "Saved" line's count, the report's points and the frame's
/// pixels (vertices_off_their_pixel()); and the report's points against the frame's valid pixels.
/// @param snapshot [in] the PLY's name; the PNGs are named after it.
/// @param frame [in] the frame it should hold.
/// @param saved_line [in] the line the snapshot printed.
/// @param report [in] the points the frame's report line gave; nothing when the run printed none.
void check_snapshot(const std::string &snapshot, const gca::frame &frame,
                    const std::string &saved_line, std::optional<std::size_t> report)
{
    const std::size_t valid = valid_pixels(frame);
    const std::size_t pixels = frame.depth().size();
    std::printf("snapshot: %zu valid pixels of %zu\n", valid, pixels);
    check(valid > 0 && valid < pixels,
          "snapshot: the frame has valid pixels and pixels that are not");
    if (report)
        check(*report == valid, "snapshot: the report's points are the frame's valid pixels");
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

    std::string error;
    const std::optional<std::vector<gca_test::ply_vertex>> vertices =
        gca_test::read_ply_file(snapshot, error);
    check(vertices.has_value(), ("snapshot: the tests' PLY reader reads it; " + error).c_str());
    if (!vertices)
        return;
    const std::optional<std::size_t> saved = saved_points(saved_line);
    const std::size_t off = vertices_off_their_pixel(*vertices, frame);
    std::printf("snapshot: %zu vertices, %zu of them not a valid pixel's point\n", vertices->size(),
                off);
    check(saved && *saved == vertices->size() && !vertices->empty() && vertices->size() <= valid,
          "snapshot: the PLY holds the \"Saved\" line's points, more than none and at most the "
          "valid pixels");
    check(off == 0, "snapshot: each vertex a valid pixel's point, once, with its depth and color, "
                    "its normal toward the camera");
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
    check(saved_points(output.lines[4]).has_value(),
          "three frames: \"Saved <N> points to cloud_000.ply\"");
    const std::vector<std::string> expected_files = {"cloud_000.ply", "cloud_000.ply.color.png",
                                                     "cloud_000.ply.depth.png"};
    check(directory.files() == expected_files, "three frames: the snapshot's three files, no "
                                               "others");
    gca::frame third;
    // The first frame, index 0, sets the viewer up; the third frame after it has index 3.
    check(render_frame(*options, 3, third), "three frames: the third frame rendered again");
    check_snapshot("cloud_000.ply", third, output.lines[4], points);
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
    check(saved_points(output.lines[6]).has_value(),
          "timeouts between frames: the snapshot after the third delivered frame");
    gca::frame third;
    check(render_frame(*options, 3, third),
          "timeouts between frames: the third frame after the first rendered again");
    check_snapshot("cloud_000.ply", third, output.lines[6], points);
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
    const bool lines_as_expected = output.lines.size() == 2 && output.lines[0] == intrinsics_line &&
                                   saved_points(output.lines[1]).has_value();
    check(lines_as_expected,
          "long period: the intrinsics line, then only \"Saved <N> points to cloud_000.ply\"");
    if (!lines_as_expected)
        return;
    gca::frame second;
    check(render_frame(*options, 2, second),
          "long period: the second frame after the first rendered again");
    check_snapshot("cloud_000.ply", second, output.lines[1], std::nullopt);
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

/// A GPU step that fails: CUDA's last error left set before the second frame after the first, as
/// a failed call leaves it, so that its organized points refuse to launch. The step's reason,
/// naming CUDA's error, ends the loop; no frame is taken after it, no snapshot saved, the viewer
/// gives up, and CUDA's last error is clear after.
void check_gpu_failure()
{
    std::printf("a GPU step that fails:\n");
    const auto options = scene_options(5, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "GPU failure: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    // Call 1 gives the first frame (index 0), 2 frame 1, 3 frame 2 after CUDA's error is left.
    const run_output output = run(
        *options, {.max_frames = 3, .timeout_calls = {}, .stop_after = 0, .cuda_error_call = 3});
    check(!output.ran, "GPU failure: the viewer gives up");
    check(output.next_calls == 3, "GPU failure: no frame taken after the failed step");
    const bool lines_as_expected =
        output.lines.size() == 3 && output.lines[0] == intrinsics_line &&
        reports(output.lines, 1, 1) &&
        output.lines[2].find("left by an earlier call") != std::string::npos &&
        output.lines[2].find("cudaErrorMemoryAllocation") != std::string::npos;
    check(lines_as_expected, "GPU failure: the intrinsics line, a report after the first frame, "
                             "then the step's reason naming CUDA's error");
    check(directory.files().empty(), "GPU failure: no snapshot");
    check(cudaPeekAtLastError() == cudaSuccess, "GPU failure: CUDA's last error clear after");
}

/// A GPU step of the snapshot that fails: CUDA's last error left set after the last frame's report
/// line, so that the snapshot's normals refuse to launch. The step's reason, naming CUDA's error,
/// ends the loop; nothing of the snapshot is written, the viewer gives up, and CUDA's last error is
/// clear after.
void check_snapshot_gpu_failure()
{
    std::printf("a GPU step of the snapshot that fails:\n");
    const auto options = scene_options(5, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "snapshot GPU failure: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    // Line 1 is the intrinsics line, lines 2 to 4 the reports of the three frames: CUDA's error is
    // left after the third report, between the last frame's step and its snapshot.
    const run_output output = run(
        *options, {.max_frames = 3, .timeout_calls = {}, .stop_after = 0, .cuda_error_line = 4});
    check(!output.ran, "snapshot GPU failure: the viewer gives up");
    check(output.next_calls == 4, "snapshot GPU failure: the first frame and three more taken");
    const bool lines_as_expected =
        output.lines.size() == 5 && output.lines[0] == intrinsics_line &&
        reports(output.lines, 1, 3) &&
        output.lines[4].find("left by an earlier call, before the organized normals' kernel") !=
            std::string::npos &&
        output.lines[4].find("cudaErrorMemoryAllocation") != std::string::npos;
    check(lines_as_expected,
          "snapshot GPU failure: the intrinsics line, three reports, then the normals' reason "
          "naming CUDA's error");
    check(directory.files().empty(), "snapshot GPU failure: nothing of the snapshot written");
    check(cudaPeekAtLastError() == cudaSuccess,
          "snapshot GPU failure: CUDA's last error clear after");
}

/// The camera's height above the large plane in the depth range's case: 5 cm, so that the
/// frame's lowest rows see the plane nearer than 0.15 m [?: a choice; S: computed, the lowest ray
/// 24 degrees below the horizon meets the plane 0.12 m away].
constexpr double grazing_height_m = 0.05;

/// The large plane's half size: 50 m, so that rows just below the horizon see it beyond 3.0 m and
/// its farthest point, 50 m, still fits a depth value of 1 mm [?: a choice; S: 50000 < 65535].
constexpr gca_test::synthetic_plane grazing_plane = {50.0, 50.0};

/// Counts a frame's pixels by their depth against the viewer's range.
struct depth_bands
{
    std::size_t none = 0;    ///< pixels without depth
    std::size_t nearer = 0;  ///< pixels nearer than the range
    std::size_t inside = 0;  ///< pixels within it
    std::size_t farther = 0; ///< pixels farther than it
};

/// Sorts a frame's pixels into the depth bands, z as the viewer's organized points take it.
/// @param frame [in] the frame.
/// @return the counts.
depth_bands bands_of(const gca::frame &frame)
{
    depth_bands bands;
    const float unit = frame.description().depth_unit_m;
    for (const std::uint16_t d : frame.depth())
    {
        const float z = static_cast<float>(d) * unit;
        if (d == 0)
            bands.none++;
        else if (z < viewer_near_m)
            bands.nearer++;
        else if (z > viewer_far_m)
            bands.farther++;
        else
            bands.inside++;
    }
    return bands;
}

/// The depth range: one frame of a large plane seen at a grazing angle from grazing_height_m, its
/// lowest rows nearer than 0.15 m, the rows near the horizon beyond 3.0 m and the rows above it
/// without depth. The report counts only the pixels within the range, and the snapshot holds only
/// points within it (check_snapshot()).
void check_depth_range()
{
    std::printf("the depth range, a plane at a grazing angle:\n");
    // Looking along +x at a point 2 m ahead on the plane, 1.4 degrees below the horizon.
    const std::optional<Eigen::Isometry3d> pose = gca_test::look_at(
        Eigen::Vector3d(0.0, 0.0, grazing_height_m), Eigen::Vector3d(2.0, 0.0, 0.0), 0.0);
    check(pose.has_value(), "depth range: look_at() gives the pose");
    if (!pose)
        return;
    gca_test::synthetic_source_options options;
    options.description = {test_size, test_intrinsics, test_depth_unit_m};
    options.scene = {.plane = grazing_plane, .trajectory = {*pose, *pose}};
    options.ending = gca_test::synthetic_ending::end;
    options.ending_message = "the poses' end";
    const snapshot_directory directory;
    const run_output output = run(options, {.max_frames = 1, .timeout_calls = {}, .stop_after = 0});
    check(output.ran, "depth range: the viewer ran");
    const bool lines_as_expected = output.lines.size() == 3 && output.lines[0] == intrinsics_line &&
                                   reports(output.lines, 1, 1) &&
                                   saved_points(output.lines[2]).has_value();
    check(lines_as_expected, "depth range: the intrinsics line, a report, \"Saved ...\"");
    gca::frame second;
    check(render_frame(options, 1, second),
          "depth range: the frame after the first rendered again");
    const depth_bands bands = bands_of(second);
    std::printf("depth range: %zu pixels without depth, %zu nearer than 0.15 m, %zu within, %zu "
                "beyond 3.0 m\n",
                bands.none, bands.nearer, bands.inside, bands.farther);
    check(bands.none > 0 && bands.nearer > 0 && bands.inside > 0 && bands.farther > 0,
          "depth range: pixels without depth, nearer than the range, within and beyond it");
    if (!lines_as_expected)
        return;
    check_snapshot("cloud_000.ply", second, output.lines[2], report_points(output.lines[1]));
}

/// A directory in the way of the snapshot's PLY: the writer's reason in place of "Saved ...", the
/// pictures saved all the same, and the viewer runs on to the end of its frames.
void check_model_in_the_way()
{
    std::printf("a directory in the way of the snapshot's PLY:\n");
    const auto options = scene_options(5, gca_test::synthetic_ending::end, "the walk's end");
    check(options.has_value(), "PLY in the way: walk_around() gives the walk");
    if (!options)
        return;
    const snapshot_directory directory;
    std::filesystem::create_directory("cloud_000.ply");
    const run_output output =
        run(*options, {.max_frames = 3, .timeout_calls = {}, .stop_after = 0});
    check(output.ran, "PLY in the way: the viewer ran");
    const std::string reason = "cloud_000.ply: the rename from cloud_000.ply.part failed";
    check(output.lines.size() == 5 && output.lines[0] == intrinsics_line &&
              reports(output.lines, 1, 3) && output.lines[4].rfind(reason, 0) == 0,
          "PLY in the way: the intrinsics line, three reports, then the writer's reason");
    const std::vector<std::string> expected_files = {"cloud_000.ply", "cloud_000.ply.color.png",
                                                     "cloud_000.ply.depth.png"};
    check(directory.files() == expected_files,
          "PLY in the way: the directory and the snapshot's pictures, no part file");
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
        output.lines[2] == timeout_line && saved_points(output.lines[5]).has_value() &&
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
    check_snapshot("cloud_000.ply", third, output.lines[5], reports_read[2]->points);
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
    const bool lines_as_expected = output.lines.size() == 8 && output.lines[0] == intrinsics_line &&
                                   counts == std::vector<int>{0, 1, 0, 1, 0} &&
                                   saved_points(output.lines[6]).has_value() &&
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
    check_snapshot("cloud_000.ply", fifth, output.lines[6], points);
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
    gca::frame first;
    gca::frame second;
    check(render_frame(*options, 0, first) && render_frame(*options, 1, second),
          "disk: the first two frames rendered again");
    run_output output;
    {
        // Room for the header, two records and part of the third [S: docs/SEQUENCE_FILE.md].
        const gca_test::file_size_limit limit(gca::sequence_header_bytes +
                                              sequence_record_bytes(first) +
                                              sequence_record_bytes(second) + 100);
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
    check_gpu_failure();
    check_snapshot_gpu_failure();
    check_model_in_the_way();
    check_depth_range();
    check_sequence_written();
    check_full_queue();
    check_sequence_in_the_way();
    check_push_fails();
    check_disk_refuses();
    check_exit_with_sequence();
    return gca_test::check_summary("test_viewer_loop");
}
