// SPDX-License-Identifier: 0BSD
/// @file object_class.cpp
/// The one table of the object classes, their names and their sets (see object_class.hpp).

#include "scan/object_class.hpp"

namespace gca
{
namespace
{
/// One object class's row of the table.
struct class_row
{
    std::string_view name;              ///< the glossary's term in snake_case
    object_class_parameters parameters; ///< the set the object class puts in force
};

/// The table of the object classes: one case for each, so -Wswitch names an object class added
/// without its row (tools/verify.sh fails on the warning).
/// @param kind [in] the object class.
/// @return its row; nothing for a value that is no enumerator.
constexpr std::optional<class_row> row_of(object_class kind) noexcept
{
    switch (kind)
    {
    case object_class::tabletop_object:
        return class_row{
            "tabletop_object",
            {
                // Coarser than the full model's, so that the live model's TSDF fits the frame
                // time, until the per-frame path's measurement sets it [?: initial, the user's
                // choice, 2026-11-14].
                .live_voxel_size_m = 0.002,
                // The voxel size the object class fixes (ADR 0004) [?: initial, the user's
                // choice, 2026-11-14].
                .full_voxel_size_m = 0.001,
                // The camera's least depth, 0.25 m [S: datasheet, in docs/HARDWARE.md, "What the
                // camera is and reports"], so that what lies nearer than too_close_depth_m is
                // tracked and fused too (R36) [?: initial, the user's choice, 2026-11-15,
                // evening]; the camera's own least depth is measured on the camera later.
                .min_depth_m = 0.25,
                // [?: initial, the user's choice, 2026-11-15, evening]: at 1.0 m, the choice of
                // 2026-11-14, the room behind the object filled a pool of 256 MB (61416 blocks) and
                // tracking lost a frame at an inlier RMSE of 5.3 mm; at 0.6 m a pool of 1 GB held
                // 46668 blocks, 19.5 % [M 2026-11-15, the demo window on the camera].
                .max_depth_m = 0.6,
                // 2.0 GB, 2'000'000'000 bytes as 1.0 GB is 10^9: while a scan runs no other
                // program runs, and the budget may rise later [?: the user's choice, 2026-11-14];
                // memory-budget-and-refinement-time-measured sets its measured value.
                .memory_budget_bytes = 2'000'000'000,
                // The start of the camera's optimal range, 0.30 m [S: datasheet, in
                // docs/HARDWARE.md], the near depth of 2026-11-14, now the notice's threshold
                // [?: initial, the user's choice, 2026-11-15, evening].
                .too_close_depth_m = 0.30,
                // The datasheet's bound, 2 % at 2 m [?: the datasheet's bound, until the camera's
                // depth noise is measured: docs/methods/sparse-tsdf.md, "The truncation distance"];
                // measure-depth-noise-on-the-camera measures it.
                .depth_noise_per_m = 0.01,
                // Three standard deviations [?: initial, docs/methods/sparse-tsdf.md, "The
                // truncation distance"].
                .truncation_noise_multiple = 3.0,
                // A voxel below the truncation on each side of a zero crossing with room to spare,
                // its half the ray cast's step [?: initial, docs/methods/sparse-tsdf.md, "The
                // truncation distance"].
                .truncation_min_voxels = 4.0,
                // A 7 x 7 window, the largest the stage takes [?: initial, docs/methods/normals.md,
                // "Input"].
                .normals_radius_px = 3,
                // Half the 7 x 7 window rounded up [?: initial, docs/methods/normals.md, "When a
                // pixel gets no normal"].
                .normals_min_points = 25,
                // A surface seen at more than 75 degrees gives few pixels and poor depth
                // [?: initial, docs/methods/normals.md, "One surface, or a depth edge"].
                .normals_max_incidence_deg = 75.0,
                // 5 cm at level 0, the prototype's correspondence distance [S: CUDA_3D's
                // apps/app.cpp], doubled at each coarser level, whose benefit is measured
                // (docs/methods/registration.md, "Coarse to fine") [?: initial].
                .max_distance_m = {0.05, 0.10, 0.20},
                // Every case of the registration's tests converged within 3 to 4 iterations at
                // level 0 after the coarser levels [M 2026-11-14: docs/methods/registration.md,
                // "Parameters"] [?: initial].
                .max_iterations = {20, 10, 10},
                // About the datasheet's bound on the depth noise at the middle of the range of
                // 2026-11-14, 0.30 m to 1.0 m: 4.2 mm at 0.65 m [S: computed,
                // docs/methods/registration.md, "Parameters"] [?: initial].
                .huber_threshold_m = 0.005,
                // Park et al.'s value by the prototype [?: initial, the user's decision of
                // 2026-11-14, late evening], tuned by tracking-thresholds-from-real-sequences.
                .point_to_plane_weight = 0.968,
                // Half of a frame's candidates (docs/methods/tracking.md, "Parameters")
                // [?: initial].
                .min_fitness = 0.5,
                // δ_G's value, the datasheet's bound on the depth noise [?: initial, the user's
                // choice, 2026-11-15]; real frames tracked at 0.9 to 2.2 mm [M 2026-11-15, the
                // demo window on the camera] (docs/methods/tracking.md, "Parameters").
                .max_inlier_rmse_m = 0.005,
                // Twice the turn rate the registration's tests recover at 30 fps, 45°/s: 3° in a
                // frame time [?: initial] (docs/methods/tracking.md, "Parameters").
                .max_turn_rate_deg_s = 90.0,
                // Three times the speed the registration's tests recover at 30 fps, 0.32 m/s:
                // 33 mm in a frame time [?: initial] (docs/methods/tracking.md, "Parameters").
                .max_speed_m_s = 1.0,
            }};
    case object_class::large_object:
        return class_row{
            "large_object",
            {
                // Coarser than the full model's, as the tabletop object's [?: initial, the
                // user's choice, 2026-11-14].
                .live_voxel_size_m = 0.008,
                // The voxel size the object class fixes (ADR 0004) [?: initial, the user's
                // choice, 2026-11-14].
                .full_voxel_size_m = 0.004,
                // The camera's least depth, as the tabletop object's [?: initial, the user's
                // choice, 2026-11-15, evening].
                .min_depth_m = 0.25,
                // [?: initial, the user's choice, 2026-11-14].
                .max_depth_m = 3.0,
                // 2.0 GB, as the tabletop object's [?: the user's choice, 2026-11-14]; the ticket
                // memory-budget-and-refinement-time-measured sets its measured value.
                .memory_budget_bytes = 2'000'000'000,
                // The near depth of 2026-11-14, now the notice's threshold [?: initial, the user's
                // choice, 2026-11-15, evening].
                .too_close_depth_m = 0.40,
                // The same camera as the tabletop object's, so its depth noise [?: the datasheet's
                // bound, docs/methods/sparse-tsdf.md, "The truncation distance"].
                .depth_noise_per_m = 0.01,
                // As the tabletop object's [?: initial, docs/methods/sparse-tsdf.md, "The
                // truncation distance"].
                .truncation_noise_multiple = 3.0,
                // As the tabletop object's [?: initial, docs/methods/sparse-tsdf.md, "The
                // truncation distance"].
                .truncation_min_voxels = 4.0,
                // As the tabletop object's, a 7 x 7 window [?: initial, docs/methods/normals.md,
                // "Input"].
                .normals_radius_px = 3,
                // As the tabletop object's, half the 7 x 7 window rounded up [?: initial,
                // docs/methods/normals.md, "When a pixel gets no normal"].
                .normals_min_points = 25,
                // As the tabletop object's [?: initial, docs/methods/normals.md, "One surface, or a
                // depth edge"].
                .normals_max_incidence_deg = 75.0,
                // Twice the tabletop object's: the depth noise by the datasheet's bound is 6.8
                // times as large at the middle of the range [S: computed,
                // docs/methods/registration.md, "Parameters"] [?: initial].
                .max_distance_m = {0.10, 0.20, 0.40},
                // As the tabletop object's [?: initial].
                .max_iterations = {20, 10, 10},
                // About the datasheet's bound on the depth noise at the middle of the range of
                // 2026-11-14, 0.4 m to 3.0 m: 29 mm at 1.7 m [S: computed,
                // docs/methods/registration.md, "Parameters"] [?: initial].
                .huber_threshold_m = 0.030,
                // As the tabletop object's [?: initial, the user's decision of 2026-11-14, late
                // evening].
                .point_to_plane_weight = 0.968,
                // As the tabletop object's [?: initial].
                .min_fitness = 0.5,
                // δ_G's value, the datasheet's bound on the depth noise [?: initial, the user's
                // choice, 2026-11-15] (docs/methods/tracking.md, "Parameters").
                .max_inlier_rmse_m = 0.030,
                // As the tabletop object's [?: initial].
                .max_turn_rate_deg_s = 90.0,
                // As the tabletop object's [?: initial].
                .max_speed_m_s = 1.0,
            }};
    }
    return std::nullopt;
}
} // namespace

object_class_parameters parameters_of(object_class kind) noexcept
{
    return row_of(kind)->parameters;
}

std::string_view name_of(object_class kind) noexcept
{
    return row_of(kind)->name;
}

std::optional<object_class> object_class_named(std::string_view name) noexcept
{
    // The enumerators run from 0 without a gap (object_class.hpp): the first value without a row
    // ends them.
    for (int k = 0;; k++)
    {
        const object_class kind = static_cast<object_class>(k);
        const std::optional<class_row> row = row_of(kind);
        if (!row)
            return std::nullopt;
        if (row->name == name)
            return kind;
    }
}
} // namespace gca
