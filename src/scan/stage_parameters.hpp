// SPDX-License-Identifier: 0BSD
/// @file stage_parameters.hpp
/// The stages' parameters of an object class's set (scan/object_class.hpp): the one copy of each
/// conversion, which the scan interface and the tests that run with an object class's
/// values take. Tracking's: the pyramid's depth range and normals, the registration's values and
/// the check's thresholds (tracking/tracker.hpp). The registration's alone
/// (registration/registration.hpp). The sparse TSDF's, of the live model or the full model
/// (fusion/sparse_tsdf.hpp). Each double of the set that a stage holds as a float is rounded to
/// the nearest float; nothing is checked here, as each stage checks its own ranges when its
/// buffers are requested.

#pragma once

#include "fusion/sparse_tsdf.hpp"
#include "registration/registration.hpp"
#include "scan/object_class.hpp"
#include "tracking/tracker.hpp"

namespace gca
{
/// Which model a sparse TSDF holds (GLOSSARY.md): its voxel size is the set's for that model.
enum class fused_model
{
    live, ///< the live model, which tracking fuses each frame into, at live_voxel_size_m
    full, ///< the full model, which refinement fuses at full_voxel_size_m
};

/// Gives the registration's parameters of a set: each level's d_max and most iterations, δ_G and
/// λ.
/// @param set [in] the set.
/// @return the registration's parameters, the lengths and λ rounded to float.
/// @note Host only; allocates nothing.
registration_parameters registration_parameters_of(const object_class_parameters &set) noexcept;

/// Gives tracking's parameters of a set: the pyramid's depth range from min_depth_m to
/// max_depth_m and the normals' window, least points and steepest incidence; the registration's
/// (registration_parameters_of()); the check's f_track, e_track, ω_max and v_max.
/// @param set [in] the set.
/// @return tracking's parameters, the pyramid's and the registration's floats rounded.
/// @note Host only; allocates nothing.
tracking_parameters tracking_parameters_of(const object_class_parameters &set) noexcept;

/// Gives a sparse TSDF's parameters of a set: the memory budget, the voxel size of the model it
/// holds, the truncation's c, k and μ_min, and max_depth_m as its far depth.
/// @param set [in] the set.
/// @param model [in] the model the sparse TSDF holds, live or full.
/// @return the sparse TSDF's parameters, its floats rounded.
/// @note Host only; allocates nothing.
sparse_tsdf_parameters sparse_tsdf_parameters_of(const object_class_parameters &set,
                                                 fused_model model) noexcept;
} // namespace gca
