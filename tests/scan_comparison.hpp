// SPDX-License-Identifier: 0BSD
/// @file scan_comparison.hpp
/// The tests' comparison of a scan with the same scan made in process: the scan interface run on a
/// frame source with a writer that waits for room, until it is idle again; the one scan a scans
/// directory holds; the names in a directory; and the check that two scans' directories hold the
/// same four files, the sequences and the trajectories the same bytes and the full models the same
/// points, compared as sets bit for bit (docs/SCAN_FILES.md). The parameters, which hold a scan's
/// times, are each test's to compare. Header only; each test program is one translation unit.

#pragma once

#include "camera/frame_source.hpp"
#include "camera/sequence_writer.hpp"
#include "check.hpp"
#include "scan/model_file.hpp"
#include "scan/object_class.hpp"
#include "scan/scan_interface.hpp"
#include "sparse_tsdf_helpers.hpp"
#include "stage_frames.hpp"
#include "test_files.hpp"
#include "types/model_point.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <future>
#include <string>
#include <system_error>
#include <vector>

namespace gca_test
{
/// The commit the scans made in process hold in their parameters [?: a choice].
inline const std::string in_process_commit = "in-process";

/// Makes a scan in process: the scan interface on a source at the tests' small stream size
/// (stage_frames.hpp), with a writer that waits for room, the set given, the object class the
/// tabletop object's, into a scans directory; waits until it is idle again with its outcome.
/// @param source [in,out] the source, closed; the scan opens and closes it.
/// @param scans [in] the scans directory, which exists.
/// @param set [in] the set.
/// @return the state at the scan's end; a failed check and the state then when the start was
///         refused.
inline gca::scan_state scan_in_process(gca::frame_source &source,
                                       const std::filesystem::path &scans,
                                       const gca::object_class_parameters &set)
{
    gca::sequence_writer writer(gca::queue_full::wait);
    std::promise<gca::scan_state> ended;
    std::future<gca::scan_state> end = ended.get_future();
    bool handed = false;
    const gca::scan_options options = {.scans_directory = scans,
                                       .stream_size = small_description.size,
                                       .commit = in_process_commit,
                                       .on_state =
                                           [&ended, &handed](const gca::scan_state &s) {
                                               if (s.phase == gca::scan_phase::idle &&
                                                   s.outcome != gca::scan_outcome::none && !handed)
                                               {
                                                   handed = true;
                                                   ended.set_value(s);
                                               }
                                           },
                                       .stage_times = nullptr};
    gca::scan_interface scanner(options, source, writer);
    const gca::command_reply started = scanner.start(gca::object_class::tabletop_object, set).get();
    check(started.accepted, ("in process: the start; " + started.refusal).c_str());
    if (!started.accepted)
        return scanner.state();
    return end.get();
}

/// Lists a directory's entries' names.
/// @param directory [in] the directory.
/// @return the names, sorted; none when it does not exist.
inline std::vector<std::string> names_in(const std::filesystem::path &directory)
{
    std::vector<std::string> names;
    std::error_code error;
    for (const auto &entry : std::filesystem::directory_iterator(directory, error))
        names.push_back(entry.path().filename().string());
    std::sort(names.begin(), names.end());
    return names;
}

/// Gives the one directory a scans directory holds, the scan's.
/// @param scans [in] the scans directory.
/// @return its one entry; a failed check and an empty path when it holds another count.
inline std::filesystem::path the_scan(const std::filesystem::path &scans)
{
    const std::vector<std::string> names = names_in(scans);
    check(names.size() == 1, ("one scan in " + scans.string()).c_str());
    return names.size() == 1 ? scans / names[0] : std::filesystem::path();
}

/// Checks that two scans' directories hold the same scan but for the parameters: the same four
/// files, the sequences and the trajectories the same bytes, the full models the same points as
/// sets, bit for bit.
/// @param a [in] the one scan's directory.
/// @param b [in] the other's.
/// @param what [in] the case, named in the check lines.
inline void check_same_scan(const std::filesystem::path &a, const std::filesystem::path &b,
                            const std::string &what)
{
    const std::vector<std::string> four = {"frames.sequence", "full.ply", "parameters.json",
                                           "trajectory.txt"};
    check(names_in(a) == four && names_in(b) == four, (what + ": the same four files").c_str());
    check(bytes_of(a / "frames.sequence") == bytes_of(b / "frames.sequence"),
          (what + ": the sequences are the same bytes").c_str());
    check(bytes_of(a / "trajectory.txt") == bytes_of(b / "trajectory.txt"),
          (what + ": the trajectories are the same bytes").c_str());
    const gca::read_result<std::vector<gca::model_point>> ma = gca::read_model(a / "full.ply");
    const gca::read_result<std::vector<gca::model_point>> mb = gca::read_model(b / "full.ply");
    check(ma.value && mb.value, (what + ": both full models read").c_str());
    if (!ma.value || !mb.value)
        return;
    std::printf("%s: full models of %zu and %zu points\n", what.c_str(), ma.value->size(),
                mb.value->size());
    check(!ma.value->empty() && sorted_keys(*ma.value) == sorted_keys(*mb.value),
          (what + ": the full models are the same points, bit for bit").c_str());
}
} // namespace gca_test
