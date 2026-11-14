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
                // Coarser than the final model's, so that the live model's TSDF fits the frame
                // time, until the per-frame path's measurement sets it [?: initial, the user's
                // choice, 2026-11-14].
                .live_voxel_size_m = 0.002,
                // The voxel size the object class fixes (ADR 0004) [?: initial, the user's
                // choice, 2026-11-14].
                .full_voxel_size_m = 0.001,
                // The start of the camera's optimal range, 0.30 m [S: datasheet, in
                // docs/HARDWARE.md, "What the camera is and reports"] [?: initial, the user's
                // choice, 2026-11-14].
                .min_depth_m = 0.30,
                // [?: initial, the user's choice, 2026-11-14].
                .max_depth_m = 1.0,
                // 2.0 GB, 2'000'000'000 bytes as 1.0 GB is 10^9: while a scan runs no other
                // program runs, and the budget may rise later [?: the user's choice, 2026-11-14];
                // memory-budget-and-refinement-time-measured sets its measured value.
                .memory_budget_bytes = 2'000'000'000,
            }};
    case object_class::large_object:
        return class_row{
            "large_object",
            {
                // Coarser than the final model's, as the tabletop object's [?: initial, the
                // user's choice, 2026-11-14].
                .live_voxel_size_m = 0.008,
                // The voxel size the object class fixes (ADR 0004) [?: initial, the user's
                // choice, 2026-11-14].
                .full_voxel_size_m = 0.004,
                // [?: initial, the user's choice, 2026-11-14].
                .min_depth_m = 0.4,
                // [?: initial, the user's choice, 2026-11-14].
                .max_depth_m = 3.0,
                // 2.0 GB, as the tabletop object's [?: the user's choice, 2026-11-14]; the ticket
                // memory-budget-and-refinement-time-measured sets its measured value.
                .memory_budget_bytes = 2'000'000'000,
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
