// SPDX-License-Identifier: 0BSD
/// @file web_test_helpers.hpp
/// What the web layer's tests (test_http_routes.cpp, test_websocket.cpp) share: the scene their
/// scans see, the tabletop box along test_scan_interface.cpp's walk; a frame source in front of the
/// synthetic one that delivers the frames at the camera's rate; the set the scans run with; and the
/// reader of the state message (docs/ROUTES.md, "The state") with OpenCV's JSON reader, which reads
/// true as 1 and false as 0 and refuses null and \u [M 2026-11-17: cv2 4.8.0 on the board]. Header
/// only; each test program is one translation unit.

#pragma once

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "scan/object_class.hpp"
#include "scan/parameters_file.hpp"
#include "sparse_tsdf_scenes.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace gca_test
{
/// The walk's poses, 240 of tabletop_walk(), 1.5 degrees apart, as test_scan_interface.cpp's.
inline constexpr std::size_t walk_poses = 240;

/// The set the web tests' scans run with: the object class's own, in a sparse TSDF of 273 MB
/// (case_budget), as test_scan_interface.cpp's.
/// @param kind [in] the object class asked for.
/// @return its set with the case's budget.
inline gca::object_class_parameters test_set(gca::object_class kind)
{
    gca::object_class_parameters set = gca::parameters_of(kind);
    set.memory_budget_bytes = case_budget;
    return set;
}

/// Gives the synthetic source's options: the tabletop box along the walk's first poses.
/// @param count [in] the poses, from the walk's first, at most walk_poses.
/// @return the options; the source reports its end after them.
inline synthetic_source_options walk_options(std::size_t count)
{
    synthetic_walk walk = tabletop_walk();
    walk.pose_count = walk_poses;
    std::vector<Eigen::Isometry3d> poses = poses_of(walk);
    poses.resize(std::min(count, poses.size()));
    return {.description = small_description,
            .scene = {.plane = tabletop_plane, .box = tabletop_box, .trajectory = poses},
            .ending = synthetic_ending::end,
            .ending_message = "the walk's end",
            .open_refusal = {},
            .imu = std::nullopt};
}

/// A frame source in front of the synthetic source that delivers frame i of an opening no
/// earlier than i frame periods (synthetic_frame_period_us) after the opening, as the camera does
/// at 30 frames a second; the scan thread alone calls it.
class paced_source final : public gca::frame_source
{
public:
    /// Builds the source in front of a synthetic source.
    /// @param options [in] the synthetic source's options; taken over.
    explicit paced_source(synthetic_source_options options)
        : m_inner(std::move(options))
    {
    }

    gca::open_result open(const gca::frame_size &size) noexcept override
    {
        m_opened = std::chrono::steady_clock::now();
        m_next = 0;
        return m_inner.open(size);
    }

    gca::next_result next(gca::frame &out) noexcept override
    {
        std::this_thread::sleep_until(
            m_opened + std::chrono::microseconds(m_next * synthetic_frame_period_us));
        m_next++;
        return m_inner.next(out);
    }

    void close() noexcept override
    {
        m_inner.close();
    }

    gca::source_description describe() const noexcept override
    {
        return m_inner.describe();
    }

private:
    synthetic_source m_inner;
    std::chrono::steady_clock::time_point m_opened; ///< when the source was last opened
    std::uint64_t m_next = 0;                       ///< the next frame's number in the opening
};

/// The keys of the state message, in the document's order (docs/ROUTES.md, "The state").
inline const std::vector<std::string> state_keys = {"version",
                                                    "phase",
                                                    "scan",
                                                    "tracking",
                                                    "lost_cause",
                                                    "too_close",
                                                    "frames",
                                                    "end",
                                                    "end_message",
                                                    "outcome",
                                                    "outcome_message",
                                                    "refusal",
                                                    "live_model_failure"};

/// The keys of the state's frames, in the document's order.
inline const std::vector<std::string> frames_keys = {"taken", "fused", "lost", "not_taken"};

/// A state message as the tests read it back.
struct state_fields
{
    std::vector<std::string> keys;        ///< the top level's keys, in order
    std::vector<std::string> frames_keys; ///< the frames' keys, in order
    int version = 0;
    std::string phase;
    std::string scan;
    std::string tracking;
    std::string lost_cause;
    int too_close = -1; ///< 1 for true, 0 for false, as OpenCV reads them; -1 when no integer
    gca::scan_frame_counts frames{};
    std::string end;
    std::string end_message;
    std::string outcome;
    std::string outcome_message;
    std::string refusal;
    std::string live_model_failure;
};

/// Gives a map's keys in their order.
/// @param map [in] the map.
/// @return the keys.
inline std::vector<std::string> keys_of(const cv::FileNode &map)
{
    std::vector<std::string> keys;
    for (const cv::FileNode &child : map)
        keys.push_back(child.name());
    return keys;
}

/// Reads a state message with OpenCV's JSON reader, which reads true as 1 and false as 0 and a
/// whole number as a 32-bit int [M 2026-11-14: OpenCV 4.8.0, parameters_file.cpp].
/// @param json [in] the message.
/// @return its fields; nothing when the reader refuses it.
inline std::optional<state_fields> read_state(const std::string &json)
{
    try
    {
        const cv::FileStorage storage(json, cv::FileStorage::READ | cv::FileStorage::MEMORY |
                                                cv::FileStorage::FORMAT_JSON);
        const cv::FileNode root = storage.root();
        const auto number = [](const cv::FileNode &n) {
            return n.isInt() ? static_cast<int>(n) : -1;
        };
        const auto text = [](const cv::FileNode &n) {
            return n.isString() ? static_cast<std::string>(n) : std::string("<no string>");
        };
        state_fields read;
        read.keys = keys_of(root);
        read.version = number(root["version"]);
        read.phase = text(root["phase"]);
        read.scan = text(root["scan"]);
        read.tracking = text(root["tracking"]);
        read.lost_cause = text(root["lost_cause"]);
        read.too_close = number(root["too_close"]);
        const cv::FileNode frames = root["frames"];
        read.frames_keys = keys_of(frames);
        read.frames = {static_cast<std::uint64_t>(number(frames["taken"])),
                       static_cast<std::uint64_t>(number(frames["fused"])),
                       static_cast<std::uint64_t>(number(frames["lost"])),
                       static_cast<std::uint64_t>(number(frames["not_taken"]))};
        read.end = text(root["end"]);
        read.end_message = text(root["end_message"]);
        read.outcome = text(root["outcome"]);
        read.outcome_message = text(root["outcome_message"]);
        read.refusal = text(root["refusal"]);
        read.live_model_failure = text(root["live_model_failure"]);
        return read;
    }
    catch (const cv::Exception &e)
    {
        std::printf("the state message refused by OpenCV: %s\n", e.what());
        return std::nullopt;
    }
}
} // namespace gca_test
