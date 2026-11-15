// SPDX-License-Identifier: 0BSD
/// @file object_class.hpp
/// The object class (GLOSSARY.md), the one choice an operator makes before a scan, and the set
/// of parameters it puts in force: the voxel sizes, the depth range, the memory budget, the "too
/// close" notice's threshold, the registration's values and tracking's thresholds. The
/// sets stand in one place, object_class.cpp, each value with its evidence above it (AGENTS.md
/// §7 rule 8). A scan stores the set it was made with in its parameters (scan/parameters_file.hpp),
/// and a stored set is read back as it stands, so a set is a value that need not be an object
/// class's own.

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace gca
{
/// What is scanned (GLOSSARY.md). The enumerators run from 0 without a gap, and each has its row
/// in object_class.cpp's table, a switch that -Wswitch checks.
enum class object_class
{
    tabletop_object, ///< a single rigid thing of 10 cm to 50 cm on a table
    large_object,    ///< a single rigid thing of 0.5 m to 2 m on the floor
};

/// The set of parameters an object class puts in force. Every length and threshold is finite and
/// > 0, the two shares at most 1, and min_depth_m < too_close_depth_m < max_depth_m; the
/// parameters' writer and reader refuse a set that is not (docs/SCAN_FILES.md). The ranges each
/// stage takes its values in are the stage's own, checked when a scan requests its buffers
/// (tracking/tracker.hpp, fusion/sparse_tsdf.hpp).
struct object_class_parameters
{
    double live_voxel_size_m; ///< the voxel edge of the live model's TSDF, in m
    double full_voxel_size_m; ///< the voxel edge of the full models, refinement's fusion, in m
    double min_depth_m;       ///< the nearest depth a frame contributes, along the camera's z, in m
    double max_depth_m; ///< the farthest depth a frame contributes, along the camera's z, in m
    std::uint64_t memory_budget_bytes; ///< all of the device memory of the scan's sparse TSDF
                                       ///< (docs/methods/sparse-tsdf.md), in bytes, from 1 to
                                       ///< 2^53, so that JSON holds it exactly
    /// The "too close" notice's threshold, along the camera's z, in m: a frame is too close when
    /// at least a quarter of its pixels with depth lie nearer (R36); the scan raises the notice,
    /// tracking does not read it.
    double too_close_depth_m;
    /// The registration's d_max_l at levels 0, 1 and 2, in m (docs/methods/registration.md,
    /// "Parameters").
    std::array<double, 3> max_distance_m;
    /// The registration's n_l, the most iterations at levels 0, 1 and 2.
    std::array<std::uint32_t, 3> max_iterations;
    double huber_threshold_m;     ///< the registration's δ_G, in m
    double point_to_plane_weight; ///< the registration's λ, a share in (0, 1]
    /// Tracking's f_track, the least fitness, a share in (0, 1] (docs/methods/tracking.md,
    /// "Parameters").
    double min_fitness;
    double max_inlier_rmse_m;   ///< tracking's e_track, the largest inlier RMSE, in m
    double max_turn_rate_deg_s; ///< tracking's ω_max, the hand's largest turn rate, in °/s
    double max_speed_m_s;       ///< tracking's v_max, the hand's largest speed, in m/s

    /// Compares two sets value by value.
    /// @param other [in] the other set.
    /// @return whether every value is equal.
    bool operator==(const object_class_parameters &other) const = default;
};

/// Gives the set an object class puts in force.
/// @param kind [in] the object class, one of the enumerators.
/// @return its set, each value a starting value with its evidence in object_class.cpp.
object_class_parameters parameters_of(object_class kind) noexcept;

/// Gives an object class's name, the glossary's term in snake_case, as the parameters' JSON
/// writes it.
/// @param kind [in] the object class, one of the enumerators.
/// @return "tabletop_object" or "large_object"; a static string.
std::string_view name_of(object_class kind) noexcept;

/// Finds the object class of a name.
/// @param name [in] the name, as name_of() gives it.
/// @return the object class, or nothing when the name is no object class's.
std::optional<object_class> object_class_named(std::string_view name) noexcept;
} // namespace gca
