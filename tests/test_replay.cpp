// SPDX-License-Identifier: 0BSD
/// @file test_replay.cpp
/// The replay tool (apps/replay.cpp), run as a program through the shell on a short synthetic
/// sequence the test writes: the first 30 poses of the walk around the tabletop box of
/// test_scan_interface.cpp, 1.5 degrees apart, at 640x400 with the camera's intrinsics, each frame
/// marked as tracked by nothing, as orbbec_viewer writes it. The program's path is the argument.
/// - The same scan made in process, the scan interface on the sequence with a writer that waits
///   for room and the tabletop object class's set in a sparse TSDF of 273 MB, then the program
///   with that scan's parameters and --times: both scans' directories hold the same four files;
///   the sequences and the trajectories are the same bytes; the parameters the same but for the
///   start, the stop and the commit, the program's the build's (apps/build_commit.hpp); the full
///   models the same points, compared as sets bit for bit.
/// - The time lines (docs/methods/tracking.md, "The replay tool's lines"): one for each frame
///   taken, in order, naming each stage of the per-frame path in its order, with what tracking
///   made of the frame; the stages add up to the frame's time within a measured margin; each
///   stage the frame ran holds more than 0 (but room and check, which can print 0.000), the ray
///   cast 0 for the first frame, before the model holds one, and the live model more than 0 for
///   it, which takes the scan's first. The rule that the fusion is 0 for a
///   frame in tracking loss is written but not exercised: no frame of the walk is lost. The last
///   line names the scan's directory, its end and its counts.
/// - A kept scan's parameters with the full voxel size changed from 1 mm to 2 mm: the program's
///   scan holds the changed set in its parameters, and its full model's points lie on the edges
///   of a 2 mm grid, where the 1 mm model has none.
/// - Exit 2 with the reason and the usage: an unknown option, a missing sequence, a scans
///   directory that holds something already; exit 1 with the reason alone: a parameters file its
///   reader refuses, a sequence's file that is no sequence.
/// The test prints its time; a scan-level test ends within 60 s on the board (the stage brief).

#include "build_commit.hpp"
#include "camera/frame.hpp"
#include "camera/sequence_format.hpp"
#include "camera/sequence_reader.hpp"
#include "camera/sequence_writer.hpp"
#include "check.hpp"
#include "program_run.hpp"
#include "replay_options.hpp"
#include "scan/model_file.hpp"
#include "scan/object_class.hpp"
#include "scan/parameters_file.hpp"
#include "scan/scan_interface.hpp"
#include "scan_comparison.hpp"
#include "sparse_tsdf_scenes.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "test_files.hpp"
#include "types/model_point.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_refusal;
using gca_test::names_in;
using gca_test::run_program;
using gca_test::scan_in_process;
using gca_test::temporary_directory;
using gca_test::the_scan;

/// The sequence's frames: the walk's first 30 poses [?: a choice, a short scan].
constexpr std::size_t sequence_frames = 30;

/// The walk's poses the frames are taken from, 240 of tabletop_walk(), 1.5 degrees apart, as
/// test_scan_interface.cpp's walk.
constexpr std::size_t walk_poses = 240;

/// The largest difference between a frame's time and the sum of its stages' times, in ms: what
/// lies outside the stages, the state's publication and the commands, and the
/// rounding of ten printed times to 0.001 ms; about 4 times the largest measured, 0.027 ms in
/// replays of 1440 frames at 640x400 and 720 at 1280x800, and 0.004 ms in 6 runs of this test
/// [M 2026-11-16: docs/HARDWARE.md, "The replay's frame time"; this test, under the build lock].
constexpr double stage_sum_margin_ms = 0.1;

/// The changed full voxel size, in m: 2 mm, twice the tabletop object class's 1 mm [?: a choice].
constexpr double changed_full_voxel_m = 0.002;

/// How far a coordinate may lie from a grid's plane and still count as on it, in m: far above
/// float's rounding of a coordinate below 1 m, 6e-8 m, and far below half the 1 mm voxel
/// [S: float32 ulp].
constexpr double on_grid_m = 1e-6;

/// The set the scans run with: the tabletop object class's, in a sparse TSDF of 273 MB
/// (gca_test::case_budget), as test_scan_interface.cpp's.
/// @return the set.
gca::object_class_parameters test_set()
{
    gca::object_class_parameters set = gca::parameters_of(gca::object_class::tabletop_object);
    set.memory_budget_bytes = gca_test::case_budget;
    return set;
}

/// Writes the sequence: the synthetic source's frames along the walk's first poses, each marked
/// as tracked by nothing.
/// @param path [in] the sequence's path; its directory exists.
void write_sequence(const std::filesystem::path &path)
{
    gca_test::synthetic_walk walk = gca_test::tabletop_walk();
    walk.pose_count = walk_poses;
    std::vector<Eigen::Isometry3d> poses = gca_test::poses_of(walk);
    poses.resize(std::min(sequence_frames, poses.size()));
    gca_test::synthetic_source source({.description = gca_test::small_description,
                                       .scene = {.plane = gca_test::tabletop_plane,
                                                 .box = gca_test::tabletop_box,
                                                 .trajectory = poses},
                                       .ending = gca_test::synthetic_ending::end,
                                       .ending_message = "the walk's end",
                                       .open_refusal = {},
                                       .imu = std::nullopt});
    gca::sequence_writer writer(gca::queue_full::wait);
    check(source.open(gca_test::small_description.size).opened, "test setup: the source opened");
    check(writer.open(path, gca_test::small_description, gca::sequence_queue_frames).opened,
          "test setup: the sequence's writer opened");
    gca::frame frame;
    std::size_t written = 0;
    while (source.next(frame).status == gca::next_status::frame)
        written += writer.has_room() &&
                   writer.push(frame, gca::not_tracked{}).status == gca::push_status::accepted;
    source.close();
    check(writer.close().written && written == sequence_frames,
          "test setup: the sequence written, every frame");
}

/// Splits a text into its lines.
/// @param text [in] the text.
/// @return its lines, without their ends.
std::vector<std::string> lines_of(const std::string &text)
{
    std::vector<std::string> lines;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);)
        lines.push_back(line);
    return lines;
}

/// One time line as read back.
struct time_line
{
    std::uint64_t index = 0;                ///< the frame's index
    std::string state;                      ///< what tracking made of it
    std::vector<std::string> names;         ///< the stages' names, in the line's order
    std::map<std::string, double> stage_ms; ///< each stage's time, in ms
    double total_ms = 0.0;                  ///< the frame's time, in ms
};

/// The stages a time line holds: 10, as docs/methods/tracking.md, "The replay tool's lines",
/// writes it, so that a stage added or left out shows here.
constexpr std::size_t line_stages = 10;

/// Reads a time line: "frame <index> <state>", line_stages pairs of a stage's name and its time,
/// "total" and the frame's time.
/// @param line [in] the line.
/// @return the line read; nothing when it is no time line of that form.
std::optional<time_line> read_time_line(const std::string &line)
{
    std::istringstream in(line);
    std::string word;
    time_line read;
    if (!(in >> word) || word != "frame" || !(in >> read.index >> read.state))
        return std::nullopt;
    for (std::size_t k = 0; k < line_stages; k++)
    {
        double ms = 0.0;
        if (!(in >> word >> ms))
            return std::nullopt;
        read.names.push_back(word);
        read.stage_ms[word] = ms;
    }
    if (!(in >> word) || word != "total" || !(in >> read.total_ms) || (in >> word))
        return std::nullopt;
    return read;
}

/// Says whether a model's point lies on an edge of the grid of a voxel size: two of its
/// coordinates on the planes through the voxels' centres, (k + 1/2) v (docs/methods/sparse-tsdf.md,
/// "The extraction").
/// @param p [in] the point, in the models' frame.
/// @param voxel_m [in] the voxel size v, in m.
/// @return whether at least two coordinates lie within on_grid_m of such a plane.
bool on_grid_edge(const gca::model_point &p, double voxel_m)
{
    int on = 0;
    for (int axis = 0; axis < 3; axis++)
    {
        const double cells = static_cast<double>(p.position_m[axis]) / voxel_m - 0.5;
        on += std::abs(cells - std::round(cells)) * voxel_m <= on_grid_m;
    }
    return on >= 2;
}

/// The in-process scan and the program's on the same sequence with the same set, --times asked:
/// the same files, as the file's overview says, and the time lines and the last line.
/// @param program [in] the replay tool's path.
/// @param directory [in] the test's directory, holding sequence/frames.sequence.
/// @return the in-process scan's directory, whose parameters the next case changes.
std::filesystem::path test_same_scan(const std::string &program,
                                     const temporary_directory &directory)
{
    const std::filesystem::path sequence = directory / "sequence" / gca::sequence_file_name;
    const std::filesystem::path in_process_scans = directory / "in_process";
    const std::filesystem::path program_scans = directory / "program";
    std::filesystem::create_directory(in_process_scans);
    std::filesystem::create_directory(program_scans);

    gca::sequence_reader source(sequence);
    const gca::scan_state in_process = scan_in_process(source, in_process_scans, test_set());
    check(in_process.outcome == gca::scan_outcome::refined,
          ("in process: refined; " + in_process.outcome_message).c_str());
    const std::filesystem::path a = the_scan(in_process_scans);

    const gca_test::program_run run = run_program(
        program, {"--sequence", (directory / "sequence").string(), "--parameters",
                  (a / "parameters.json").string(), "--scans", program_scans.string(), "--times"});
    check(run.exit_status == 0, ("the program: exit 0; it printed:\n" + run.output).c_str());
    const std::filesystem::path b = the_scan(program_scans);
    if (a.empty() || b.empty())
        return a;

    gca_test::check_same_scan(a, b, "same scan");

    const gca::read_result<gca::scan_parameters> pa = gca::read_parameters(a / "parameters.json");
    const gca::read_result<gca::scan_parameters> pb = gca::read_parameters(b / "parameters.json");
    check(pa.value && pb.value, "same scan: both parameters read");
    if (pa.value && pb.value)
    {
        gca::scan_parameters expected = *pa.value;
        expected.commit = std::string(gca::build_commit());
        expected.start = pb.value->start;
        expected.stop = pb.value->stop;
        std::printf("same scan: the program's commit %s, source %s\n", pb.value->commit.c_str(),
                    pb.value->source.camera.c_str());
        check(*pb.value == expected,
              "same scan: the parameters the same but for start, stop and the build's commit");
        check(pb.value->start <= pb.value->stop, "same scan: the program's start before its stop");
        check(pa.value->frames.taken == sequence_frames && pa.value->frames.not_taken == 0 &&
                  pa.value->end == gca::scan_end::source_end,
              "same scan: every frame taken, to the sequence's end");
    }

    // The time lines and the last line.
    std::vector<time_line> times;
    std::optional<std::string> last;
    for (const std::string &line : lines_of(run.output))
    {
        if (const std::optional<time_line> read = read_time_line(line))
            times.push_back(*read);
        else if (line.rfind("replay: ", 0) == 0)
            last = line;
    }
    std::printf("time lines: %zu\n", times.size());
    check(times.size() == sequence_frames, "time lines: one for each frame taken");
    const std::vector<std::string> stages = {"room",     "too_close",    "upload", "pyramid",
                                             "ray_cast", "registration", "check",  "fusion",
                                             "writer",   "live_model"};
    double largest_gap_ms = 0.0;
    std::size_t fused = 0;
    std::size_t lost = 0;
    bool in_order = true;
    bool names = true;
    bool stage_rules = true;
    for (std::size_t k = 0; k < times.size(); k++)
    {
        const time_line &t = times[k];
        in_order = in_order && t.index == k;
        names = names && t.names == stages;
        double sum = 0.0;
        for (const auto &[name, ms] : t.stage_ms)
        {
            sum += ms;
            stage_rules = stage_rules && ms >= 0.0;
        }
        largest_gap_ms = std::max(largest_gap_ms, std::abs(t.total_ms - sum));
        fused += t.state == "tracking";
        lost += t.state == "lost";
        // Each stage the frame ran holds its own time: more than 0, but for room and check,
        // which can print 0.000 (docs/methods/tracking.md, "The stages' times").
        for (const char *ran : {"too_close", "upload", "pyramid", "registration", "writer"})
            stage_rules = stage_rules && t.stage_ms.at(ran) > 0.0;
        if (t.state == "lost")
            stage_rules = stage_rules && t.stage_ms.at("fusion") == 0.0;
        if (t.state == "tracking")
            stage_rules = stage_rules && t.stage_ms.at("fusion") > 0.0 &&
                          (k == 0) == (t.stage_ms.at("ray_cast") == 0.0);
        // The first frame takes the scan's first live model, which extracts.
        if (k == 0)
            stage_rules = stage_rules && t.stage_ms.at("live_model") > 0.0;
    }
    check(in_order, "time lines: the frames in order");
    check(names, "time lines: each stage named, in the path's order");
    check(stage_rules, "time lines: each stage the frame ran more than 0, but room and check; no "
                       "ray cast before the model holds a frame, no fusion when lost");
    gca_test::check_within("time lines: the largest difference between a frame's time and its "
                           "stages' sum, in ms",
                           largest_gap_ms, stage_sum_margin_ms);
    if (pb.value)
        check(fused == pb.value->frames.fused && lost == pb.value->frames.lost,
              "time lines: the states as the parameters count them");

    check(last.has_value(), "the last line printed");
    if (last && pb.value)
    {
        std::printf("the last line: %s\n", last->c_str());
        const std::regex form("replay: (.+): refined, end source_end, frames taken ([0-9]+) fused "
                              "([0-9]+) lost ([0-9]+) not taken ([0-9]+), ([0-9]+\\.[0-9]{3}) ms "
                              "a frame");
        std::smatch m;
        const gca::scan_frame_counts &f = pb.value->frames;
        check(std::regex_match(*last, m, form) && m[1] == b.string() &&
                  m[2] == std::to_string(f.taken) && m[3] == std::to_string(f.fused) &&
                  m[4] == std::to_string(f.lost) && m[5] == std::to_string(f.not_taken) &&
                  std::stod(m[6]) > 0.0,
              "the last line: the scan's directory, its end and its counts");
    }
    return a;
}

/// A kept scan's parameters with the full voxel size changed: the program's scan holds the
/// changed set, and its full model lies on the 2 mm grid's edges, where the 1 mm model has no
/// point.
/// @param program [in] the replay tool's path.
/// @param directory [in] the test's directory.
/// @param kept [in] the in-process scan's directory.
void test_changed_voxel(const std::string &program, const temporary_directory &directory,
                        const std::filesystem::path &kept)
{
    const gca::read_result<gca::scan_parameters> read =
        gca::read_parameters(kept / "parameters.json");
    check(read.value.has_value(), "changed voxel: the kept parameters read");
    if (!read.value)
        return;
    gca::scan_parameters changed = *read.value;
    changed.parameters.full_voxel_size_m = changed_full_voxel_m;
    const std::filesystem::path file = directory / "changed.json";
    check(gca::write_parameters(file, changed).written, "changed voxel: test setup, the file");
    const std::filesystem::path scans = directory / "changed";
    std::filesystem::create_directory(scans);
    const gca_test::program_run run =
        run_program(program, {"--sequence", (directory / "sequence").string(), "--parameters",
                              file.string(), "--scans", scans.string()});
    check(run.exit_status == 0, ("changed voxel: exit 0; it printed:\n" + run.output).c_str());
    const std::filesystem::path scan = the_scan(scans);
    if (scan.empty())
        return;
    const gca::read_result<gca::scan_parameters> made =
        gca::read_parameters(scan / "parameters.json");
    check(made.value && made.value->parameters == changed.parameters &&
              made.value->object_class == changed.object_class,
          "changed voxel: the scan's parameters hold the changed set");
    const gca::read_result<std::vector<gca::model_point>> before =
        gca::read_model(kept / "full.ply");
    const gca::read_result<std::vector<gca::model_point>> after =
        gca::read_model(scan / "full.ply");
    check(before.value && after.value, "changed voxel: both full models read");
    if (!before.value || !after.value)
        return;
    const auto count_on = [](const std::vector<gca::model_point> &points) {
        return static_cast<std::size_t>(
            std::count_if(points.begin(), points.end(), [](const gca::model_point &p) {
                return on_grid_edge(p, changed_full_voxel_m);
            }));
    };
    const std::size_t on_after = count_on(*after.value);
    const std::size_t on_before = count_on(*before.value);
    std::printf("changed voxel: %zu of %zu points on the 2 mm grid's edges; at 1 mm %zu of %zu\n",
                on_after, after.value->size(), on_before, before.value->size());
    check(!after.value->empty() && on_after == after.value->size(),
          "changed voxel: every point of the model on the 2 mm grid's edges");
    check(on_before == 0, "changed voxel: no point of the 1 mm model on them");
}

/// The command lines refused: exit 2 with the reason and the usage, exit 1 with the reason alone.
/// @param program [in] the replay tool's path.
/// @param directory [in] the test's directory, holding the sequence and the in-process scans.
void test_refusals(const std::string &program, const temporary_directory &directory)
{
    const std::string sequence = (directory / "sequence").string();
    const std::filesystem::path empty = directory / "empty";
    const std::filesystem::path no_sequence = directory / "no_sequence";
    std::filesystem::create_directory(empty);
    std::filesystem::create_directory(no_sequence);
    gca_test::write_bytes(no_sequence / gca::sequence_file_name, "no sequence at all");
    const std::filesystem::path bad_parameters = directory / "bad.json";
    gca_test::write_bytes(bad_parameters, "{}");
    const std::string usage = gca::replay_usage() + "\n";
    const auto exits = [&](const std::string &what, const std::vector<std::string> &args,
                           int status, const std::string &output, bool whole) {
        const gca_test::program_run run = run_program(program, args);
        const bool matches = whole ? run.output == output : run.output.rfind(output, 0) == 0;
        check_refusal(what + ", exit " + std::to_string(status), run.exit_status == status,
                      run.output.substr(0, run.output.find('\n')),
                      matches && (whole || run.output.find("usage:") == std::string::npos));
    };
    exits("an unknown option",
          {"--sequence", sequence, "--object-class", "tabletop_object", "--scans", empty.string(),
           "--help"},
          2, "replay: unknown argument --help\n" + usage, true);
    exits("a missing sequence",
          {"--sequence", empty.string(), "--object-class", "tabletop_object", "--scans",
           empty.string()},
          2,
          "replay: --sequence " + empty.string() + ": " +
              (empty / gca::sequence_file_name).string() + ": no such file\n" + usage,
          true);
    exits("a scans directory that holds something",
          {"--sequence", sequence, "--object-class", "tabletop_object", "--scans",
           (directory / "in_process").string()},
          2,
          "replay: --scans " + (directory / "in_process").string() + ": holds something already\n" +
              usage,
          true);
    exits("a parameters file its reader refuses",
          {"--sequence", sequence, "--parameters", bad_parameters.string(), "--scans",
           empty.string()},
          1, "replay: " + bad_parameters.string(), false);
    exits("a sequence's file that is no sequence",
          {"--sequence", no_sequence.string(), "--object-class", "tabletop_object", "--scans",
           empty.string()},
          1, "replay: " + (no_sequence / gca::sequence_file_name).string(), false);
    check(names_in(empty).empty(), "the refusals: nothing written");
}
} // namespace

/// Runs the replay tool's cases and prints their measurements, the refusals and the summary line.
/// @param argc [in] the argument count, 2.
/// @param argv [in] the arguments: argv[1] the replay tool's path.
/// @return 0 when every check passed, 1 otherwise.
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: test_replay REPLAY\n");
        return 1;
    }
    const auto begin = std::chrono::steady_clock::now();
    const temporary_directory directory("test_replay");
    std::filesystem::create_directory(directory / "sequence");
    write_sequence(directory / "sequence" / gca::sequence_file_name);
    const std::filesystem::path kept = test_same_scan(argv[1], directory);
    if (!kept.empty())
        test_changed_voxel(argv[1], directory, kept);
    test_refusals(argv[1], directory);
    std::printf("test_replay took %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
    return gca_test::check_summary("test_replay");
}
