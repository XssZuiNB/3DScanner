// SPDX-License-Identifier: 0BSD
/// @file test_stage_parameters.cpp
/// The stages' parameters of an object class's set (scan/stage_parameters.hpp), through its
/// public functions:
/// - a set of neither object class, every value another, gives each value at its place in
///   tracking's, the registration's and the sparse TSDF's parameters, rounded to float where the
///   stage holds a float, the live model's and the full model's voxel size each its own;
/// - each object class's set gives the values its method documents name, and each stage takes
///   them: tracking's check of its parameters passes, and the requests of the tracker and of the
///   live and the full model's sparse TSDF from a closed pool come back closed, which they do only
///   for parameters that pass every check, without device memory.
/// Needs no GPU and no camera: a closed pool grants nothing and queues nothing.

#include "check.hpp"
#include "fusion/sparse_tsdf.hpp"
#include "gpu/gpu_pool.hpp"
#include "registration/registration.hpp"
#include "scan/object_class.hpp"
#include "scan/stage_parameters.hpp"
#include "stage_frames.hpp"
#include "tracking/tracker.hpp"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <string>

namespace
{
using gca_test::check;

/// A set of neither object class: every value another and none a float, so that a value at
/// another's place or not rounded shows [?: a choice].
const gca::object_class_parameters odd_set = {.live_voxel_size_m = 0.0031,
                                              .full_voxel_size_m = 0.0017,
                                              .min_depth_m = 0.27,
                                              .max_depth_m = 1.9,
                                              .memory_budget_bytes = 123456789u,
                                              .too_close_depth_m = 0.33,
                                              .depth_noise_per_m = 0.0073,
                                              .truncation_noise_multiple = 2.6,
                                              .truncation_min_voxels = 3.3,
                                              .normals_radius_px = 2,
                                              .normals_min_points = 11,
                                              .normals_max_incidence_deg = 61.7,
                                              .max_distance_m = {0.07, 0.13, 0.29},
                                              .max_iterations = {17, 8, 3},
                                              .huber_threshold_m = 0.0043,
                                              .point_to_plane_weight = 0.91,
                                              .min_fitness = 0.47,
                                              .max_inlier_rmse_m = 0.0021,
                                              .max_turn_rate_deg_s = 71.5,
                                              .max_speed_m_s = 0.83};

/// Each value of the odd set stands at its place in each stage's parameters, rounded to float
/// where the stage holds a float; the live and the full model take their own voxel sizes.
void test_values_at_their_places()
{
    const gca::tracking_parameters t = gca::tracking_parameters_of(odd_set);
    check(t.pyramid.range.near_m == 0.27f && t.pyramid.range.far_m == 1.9f,
          "tracking: the depth range is min_depth_m to max_depth_m, as floats");
    check(t.pyramid.normals.radius_px == 2 && t.pyramid.normals.min_points == 11 &&
              t.pyramid.normals.max_incidence_deg == 61.7f,
          "tracking: the normals' window, least points and steepest incidence");
    check(t.thresholds.min_fitness == 0.47 && t.thresholds.max_inlier_rmse_m == 0.0021 &&
              t.thresholds.max_turn_rate_deg_s == 71.5 && t.thresholds.max_speed_m_s == 0.83,
          "tracking: the check's f_track, e_track, omega_max and v_max, as doubles");

    const gca::registration_parameters r = gca::registration_parameters_of(odd_set);
    check(r.levels[0].max_distance_m == 0.07f && r.levels[1].max_distance_m == 0.13f &&
              r.levels[2].max_distance_m == 0.29f,
          "registration: each level's d_max, as floats");
    check(r.levels[0].max_iterations == 17 && r.levels[1].max_iterations == 8 &&
              r.levels[2].max_iterations == 3,
          "registration: each level's most iterations");
    check(r.huber_threshold_m == 0.0043f && r.point_to_plane_weight == 0.91f,
          "registration: delta_G and lambda, as floats");
    const gca::registration_parameters in_tracking = t.registration;
    check(in_tracking.levels[0].max_distance_m == r.levels[0].max_distance_m &&
              in_tracking.levels[2].max_iterations == r.levels[2].max_iterations &&
              in_tracking.huber_threshold_m == r.huber_threshold_m &&
              in_tracking.point_to_plane_weight == r.point_to_plane_weight,
          "tracking: the registration's parameters are registration_parameters_of()'s");

    const gca::sparse_tsdf_parameters live =
        gca::sparse_tsdf_parameters_of(odd_set, gca::fused_model::live);
    const gca::sparse_tsdf_parameters full =
        gca::sparse_tsdf_parameters_of(odd_set, gca::fused_model::full);
    check(live.voxel_size_m == 0.0031f && full.voxel_size_m == 0.0017f,
          "sparse TSDF: the live model's and the full model's voxel sizes, as floats");
    for (const gca::sparse_tsdf_parameters &p : {live, full})
    {
        check(p.memory_budget_bytes == 123456789u, "sparse TSDF: the memory budget");
        check(p.truncation.noise_per_m == 0.0073f && p.truncation.noise_multiple == 2.6f &&
                  p.truncation.min_voxels == 3.3f,
              "sparse TSDF: the truncation's c, k and mu_min, as floats");
        check(p.max_depth_m == 1.9f, "sparse TSDF: the far depth is max_depth_m, as a float");
    }
}

/// Each object class's set gives the values its method documents name
/// (docs/methods/normals.md, "Input"; docs/methods/sparse-tsdf.md, "The truncation distance";
/// docs/methods/registration.md and docs/methods/tracking.md, "Parameters"), and each stage takes
/// them without device memory: tracking's check passes, and the tracker and both models' sparse
/// TSDFs come back closed from a closed pool, past every check of their parameters.
void test_object_classes()
{
    const gca::object_class_parameters tabletop =
        gca::parameters_of(gca::object_class::tabletop_object);
    const gca::object_class_parameters large = gca::parameters_of(gca::object_class::large_object);
    const gca::tracking_parameters t = gca::tracking_parameters_of(tabletop);
    check(t.pyramid.range.near_m == 0.25f && t.pyramid.range.far_m == 0.6f,
          "tabletop object: the depth range 0.25 m to 0.6 m");
    const gca::tracking_parameters l = gca::tracking_parameters_of(large);
    check(l.pyramid.range.near_m == 0.25f && l.pyramid.range.far_m == 3.0f,
          "large object: the depth range 0.25 m to 3.0 m");
    for (const gca::tracking_parameters &p : {t, l})
        check(p.pyramid.normals.radius_px == 3 && p.pyramid.normals.min_points == 25 &&
                  p.pyramid.normals.max_incidence_deg == 75.0f,
              "each object class: the normals' 7 x 7 window, 25 points, 75 degrees");
    const gca::sparse_tsdf_parameters tabletop_live =
        gca::sparse_tsdf_parameters_of(tabletop, gca::fused_model::live);
    const gca::sparse_tsdf_parameters large_full =
        gca::sparse_tsdf_parameters_of(large, gca::fused_model::full);
    check(tabletop_live.voxel_size_m == 0.002f && large_full.voxel_size_m == 0.004f,
          "the tabletop object's live voxel 2 mm, the large object's full voxel 4 mm");
    for (const gca::sparse_tsdf_parameters &p : {tabletop_live, large_full})
        check(p.truncation.noise_per_m == 0.01f && p.truncation.noise_multiple == 3.0f &&
                  p.truncation.min_voxels == 4.0f,
              "each object class: the truncation's c 0.01 /m, k 3, mu_min 4 voxels");

    for (const gca::object_class kind :
         {gca::object_class::tabletop_object, gca::object_class::large_object})
    {
        const std::string name(gca::name_of(kind));
        const gca::object_class_parameters set = gca::parameters_of(kind);
        const gca::tracking_parameters tracking = gca::tracking_parameters_of(set);
        const std::string refusal = gca::tracker::parameters_refusal(tracking);
        check(refusal.empty(), (name + ": tracking takes its parameters; " + refusal).c_str());
        for (const gca::fused_model model : {gca::fused_model::live, gca::fused_model::full})
        {
            const std::string what =
                name + (model == gca::fused_model::live ? ", the live model" : ", the full model");
            gca::gpu_pool pool;
            pool.close();
            const gca::request_result<gca::sparse_tsdf> tsdf = gca::sparse_tsdf::request(
                pool, gca::sparse_tsdf_parameters_of(set, model), cudaStream_t{});
            std::printf("%s: the sparse TSDF's request from a closed pool: %s\n", what.c_str(),
                        tsdf.status == gca::request_status::closed ? "closed"
                                                                   : tsdf.message.c_str());
            check(tsdf.status == gca::request_status::closed,
                  (what + ": the sparse TSDF takes its parameters").c_str());
        }
        gca::gpu_pool pool;
        pool.close();
        const gca::request_result<gca::tracker> tracker = gca::tracker::request(
            pool, gca_test::small_description.size, tracking,
            gca::sparse_tsdf_parameters_of(set, gca::fused_model::live), cudaStream_t{});
        check(
            tracker.status == gca::request_status::closed,
            (name + ": the tracker takes its parameters at the tests' 640x400; " + tracker.message)
                .c_str());
    }
}
} // namespace

/// Runs the checks and prints the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    test_values_at_their_places();
    test_object_classes();
    return gca_test::check_summary("test_stage_parameters");
}
