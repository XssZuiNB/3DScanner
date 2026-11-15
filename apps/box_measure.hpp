// SPDX-License-Identifier: 0BSD
/// @file box_measure.hpp
/// The acceptance tool's measurement (R21): fits a box to an object model and gives its three edge
/// lengths and, for each face, the spread of the face's points about its plane. The side the
/// object stands on is not scanned (ADR 0003), so the box shows five faces: its top and four
/// sides. Host only, in double, without the camera and the GPU, so that the offline tests and
/// the hardware acceptance both use it.
///
/// The method, this tool's own (a tool, not a stage, so it has no file in docs/methods/); its
/// numbers carry their evidence in box_measure.cpp:
///
/// 1. The box's axes from the points' normals. A face's normals point one way, so the axis
///    whose cone of half-angle 20 deg, of either sign, holds the most normals is the first: of
///    64 normals spread evenly over the points' order, the one whose cone holds the most, then
///    moved eight times to the mean of the normals in its cone, each turned to its side (a mean
///    shift). The second axis is found the same way from the same 64 seeds, each turned square to
///    the first, and is held square to it; the third is their cross product. The axes only sort
///    the normals into cones: the faces' planes give the edges.
/// 2. The faces: a point belongs to the face along +axis or -axis when its normal lies within
///    20 deg of it. A face keeps the points whose offset along its axis lies within 4 robust
///    standard deviations (1.4826 times the median absolute deviation) of the median offset, so
///    that a remnant of the support plane around the box's foot, whose normals are the top's,
///    does not drag the top down. A face counts when it keeps at least 1 % of the model's points,
///    and at least three.
/// 3. Each face's plane: the orthogonal least-squares fit of its points
///    (separation/plane_fit.hpp), its normal turned out of the box; its spread is the root mean
///    square of its points' distances from that plane.
/// 4. The edges. Across an axis with both faces, the distance between their planes along their
///    mean outward normal, measured between their centroids, so that a face seen only in part
///    does not lengthen it; the longer of the two is the length, which runs between the two ends,
///    the shorter the width, between the two sides. The axis with one face is the height's: from
///    the top's plane down to the foot, the median of the four other faces' lowest points along
///    the top's normal, so that one side seen only in part does not move it. The separation cuts
///    the object model one full voxel above the support plane (docs/methods/separation.md), so a
///    scanned box's sides end that far above its foot and the height comes out short by up to
///    one full voxel: 1 mm for a tabletop object and 4 mm for a large object
///    [S: scan/object_class.cpp], against R21's 2 mm and 10 mm. The rounding the fusion gives a
///    box's lower edges, whose normals leave the sides' cones, can add to it [?].
///
/// A model is refused, with the reason, when fewer than half of its points lie on the faces
/// that count (a sphere puts 18.1 % of its normals within 20 deg of six axes [S: computed in
/// box_measure.cpp]); when it does not show exactly five faces: four, one plane and an empty
/// model are refused, and so are six, since a model the scanner made never holds the side the
/// object stands on (ADR 0003); and when the top's plane lies in the lower half of the sides,
/// where a remnant of the support plane with more points than the top has taken its place.

#pragma once

#include "types/model_point.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>

#include <Eigen/Core>

namespace gca
{
/// One face of the fitted box, in the model's frame (types/model_point.hpp).
struct box_face
{
    std::size_t points;         ///< the points it keeps, the band's outliers left out, >= 3
    Eigen::Vector3d centroid_m; ///< the mean of its points, in m
    Eigen::Vector3d normal;     ///< its plane's unit normal, pointing out of the box
    double spread_mm;           ///< the root mean square of its points' distances from its plane,
                                ///< in mm, >= 0
};

/// The box fitted to an object model.
struct box_measurement
{
    double length_mm;              ///< the longer edge across the four sides, in mm
    double width_mm;               ///< the shorter edge across the four sides, in mm
    double height_mm;              ///< from the top down to the sides' foot, in mm
    box_face top;                  ///< the face across from the one the object stands on
    std::array<box_face, 2> ends;  ///< the two faces the length runs between, of either order
    std::array<box_face, 2> sides; ///< the two faces the width runs between, of either order
    std::size_t points;            ///< the model's points
    std::size_t points_on_faces;   ///< the points the five faces keep
};

/// What measure_box() gave.
struct box_result
{
    std::optional<box_measurement> measurement; ///< the box; nothing when the model is refused
    std::string refusal; ///< why the model is no box, in plain English; empty when it is one
};

/// Fits a box to an object model (the method: this file's overview).
/// @param model [in] the object model's points, in the models' frame, in m, each normal a unit
///     vector (types/model_point.hpp); host memory, read during the call.
/// @return the box; or the refusal: "<P> % of the <N> points lie on the faces found; a box needs
///     at least half", "a box shows five faces, its top and four sides, as the side it stands on
///     is not scanned (ADR 0003); this model shows <F>", or "the points facing up lie in the
///     lower half of the sides: a remnant of the support plane outweighs the top".
/// @note Host only. Allocates at most 96 bytes for each point [S: computed from the vectors it
///       holds at once]: its position and normal in double (48), its index in a face's cone (8),
///       its position kept by a face (24), and one face's offsets and their deviations (16). A
///       failed allocation ends the program (std::terminate): the function is noexcept. Holds no
///       state: two threads may measure at once.
box_result measure_box(std::span<const model_point> model) noexcept;

/// Writes the measurement's lines, which the hardware acceptance reads (AGENTS.md §7 rule 5):
///     box_fit: <NAME>: <N> points, <M> on five faces
///     length: <L> mm
///     width: <W> mm
///     height: <H> mm
///     top: <n> points, spread <S> mm
/// then "side 1", "side 2", "end 1" and "end 2" as the top's line; each length to 0.01 mm.
/// @param name [in] the model's name, its path as box_fit was given it.
/// @param measurement [in] the box.
/// @return the lines, joined by "\n", without a line end at the end.
std::string box_report(const std::string &name, const box_measurement &measurement);
} // namespace gca
