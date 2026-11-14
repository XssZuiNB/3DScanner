// SPDX-License-Identifier: 0BSD
/// @file viewer_loop.hpp
/// orbbec_viewer's loop on a frame source (camera/frame_source.hpp): it takes the first frame,
/// prints the intrinsics line and sets up the prototype's GPU code from that frame, then grabs
/// frame after frame, builds each one's point cloud on the GPU, reports the frame rate and the
/// timings, shows color | depth | overlay and saves snapshots, until quit, the exit flag, the
/// frame limit or the source's end. The viewer runs it on the Orbbec camera, the offline tests on
/// their synthetic source; it links no OrbbecSDK.
///
/// The frame is RGB8 (camera/frame.hpp); the prototype's point-cloud kernel, OpenCV's window and
/// its PNG files take BGR8, so the loop converts each frame's color once, into one picture all
/// three read, until the prototype's GPU code is replaced.

#pragma once

#include "camera/frame_source.hpp"
#include "viewer_options.hpp"

#include <atomic>
#include <chrono>
#include <ostream>

namespace gca_viewer
{
/// How often the viewer prints its report line [?: a choice, the prototype's once a second].
constexpr std::chrono::seconds report_period{1};

/// How long the viewer waits for an opening's first frame before it gives up [S: the
/// prototype's start waited up to 100 times 200 ms for its first frameset].
constexpr std::chrono::seconds first_frame_time_limit{20};

/// Where the viewer prints, how often it reports and what stops it: the viewer's console and
/// SIGINT, or a test's stream and flag.
struct run_context
{
    /// The least time between two report lines: report_period in the viewer, zero for a report
    /// after every frame.
    std::chrono::steady_clock::duration report_period;
    /// Read before each next() of the loop: true ends the loop. The viewer's SIGINT handler sets
    /// it, on whichever thread the signal reaches; it outlives the run.
    const std::atomic<bool> &exit_requested;
    /// Where the lines are printed, with console_color.hpp's color codes; it outlives the run.
    std::ostream &out;
};

/// Runs the viewer on an open frame source. Steps and lines:
/// - the first frame: timeouts are waited through for first_frame_time_limit, then "Timeout
///   while waiting for first frames."; a failure or the end prints the source's message; either
///   way "Couldn't start stream!" follows and nothing else runs. The first frame sets up the GPU
///   and prints "Intrinsics fx=<fx> fy=<fy> cx=<cx> cy=<cy>  depth scale=<m> m/unit"; it is not
///   shown or counted.
/// - the loop, until 'q' or ESC in the window, the exit flag, opt.max_frames delivered frames, or
///   the source's end (its message printed): each delivered frame is converted, uploaded, made a
///   point cloud on the GPU, shown unless headless, and counted. A timeout prints "No frames
///   received from Orbbec camera for 5s!" (AGENTS.md §3 reads it as a stop condition), a failure
///   the source's message; neither counts a frame, and the loop goes on. In the window the keys
///   keep working through them: the window stays on the last delivered frame, which 's' saves.
/// - the report, once the report period has passed since the last: "fps <frames per second> |
///   grab+decode <ms> ms | gpu point cloud <ms> ms | points <count>", the frames per second
///   counting delivered frames only, the times the means over the frames since the last report
///   (docs/HARDWARE.md's numbers come from it).
/// - a snapshot, on 's' in the window and on the last of opt.max_frames frames when headless:
///   cloud_NNN.ply (binary PLY, float x, y, z in m and uchar red, green, blue),
///   cloud_NNN.ply.color.png (8-bit color) and cloud_NNN.ply.depth.png (16-bit depth values) in the
///   current directory, NNN counting from 000, and "Saved <count> points to cloud_NNN.ply".
/// @param source [in,out] the frame source, open; left open.
/// @param opt [in] the options; the loop reads max_frames and headless, the opening took the
///        rest.
/// @param context [in] the report period, the exit flag and the stream the lines go to.
/// @return true when the loop ran; false when no first frame came.
/// @note The prototype's GPU path, until retire-the-prototype-library: allocates the frame, the
///       BGR picture and the GPU buffers once, and a point cloud for every frame; every GPU call
///       runs on the default stream, and each frame waits for the whole device. Runs on one
///       thread; only the exit flag may be set from another.
bool run_viewer(gca::frame_source &source, const options &opt, const run_context &context);
} // namespace gca_viewer
