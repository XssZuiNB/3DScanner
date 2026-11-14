// SPDX-License-Identifier: 0BSD
/// @file distortion_check.cpp
/// Says whether the aligned frames lie on a distorted pixel grid, from an orbbec_viewer snapshot
/// of a flat checkerboard: reads its color and depth PNG (cloud_NNN.ply.color.png and
/// cloud_NNN.ply.depth.png), measures the board (checkerboard_measure.hpp) and prints, for the
/// middle and the four corners of the image and for all corners, how far the corners lie from a
/// homography in px, and how far the deprojected corners lie from one plane and from the square
/// size in mm. The intrinsics and the depth scale are the ones orbbec_viewer prints for that
/// stream size ("Intrinsics fx=... fy=... cx=... cy=...  depth scale=... m/unit"). Opens no
/// camera and no GPU.

#include "checkerboard_measure.hpp"
#include "distortion_options.hpp"

#include <cstdio>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace
{
/// Reads the snapshot's two images as they are stored.
/// @param opt [in] the options with the two paths.
/// @param color [out] the color image (BGR, 8 bits in each of 3 channels).
/// @param depth [out] the depth image, at the depth the file holds (16 bits for the viewer's).
/// @return whether both were read; each file that is missing or not an image is printed to
///     stderr.
bool read_images(const gca::distortion_options &opt, cv::Mat &color, cv::Mat &depth)
{
    color = cv::imread(opt.color_path, cv::IMREAD_COLOR);
    depth = cv::imread(opt.depth_path, cv::IMREAD_ANYDEPTH);
    bool read = true;
    for (const auto &[image, path] :
         {std::pair{&color, &opt.color_path}, std::pair{&depth, &opt.depth_path}})
    {
        if (image->empty())
        {
            std::cerr << "distortion_check: cannot read " << *path << " as an image" << std::endl;
            read = false;
        }
    }
    return read;
}

/// Prints one region's line: the corners and their homography deviation, the plane deviation,
/// the edges and their square-size deviation.
/// @param region [in] the region's measurement.
void print_region(const gca::region_measurement &region)
{
    if (region.homography_px.count == 0)
    {
        std::printf("%s: no corners\n", region.name.c_str());
        return;
    }
    std::printf("%s: %d corners: homography px rms %.3f max %.3f", region.name.c_str(),
                region.homography_px.count, region.homography_px.rms, region.homography_px.max);
    if (region.plane_mm.count > 0)
        std::printf(" | %d with depth: plane mm rms %.2f max %.2f", region.plane_mm.count,
                    region.plane_mm.rms, region.plane_mm.max);
    else
        std::printf(" | none with depth");
    if (region.square_mm.count > 0)
        std::printf(" | %d edges: square mm mean %.2f rms %.2f max %.2f", region.square_mm.count,
                    region.square_mean_mm, region.square_mm.rms, region.square_mm.max);
    else
        std::printf(" | no edges with depth");
    std::printf("\n");
}

/// Prints the measurement: a header line with the board and the corners with depth, the reason
/// the mm part is empty if it is, and one line per region.
/// @param opt [in] the options, for the board and the image's name.
/// @param size [in] the image's size, px.
/// @param measurement [in] the measurement.
void print_report(const gca::distortion_options &opt, cv::Size size,
                  const gca::board_measurement &measurement)
{
    std::printf("distortion_check: %s, %dx%d, board %dx%d of %.2f mm: %zu corners, %d with depth\n",
                opt.color_path.c_str(), size.width, size.height, opt.board.columns, opt.board.rows,
                opt.board.square_mm, measurement.corners.size(), measurement.corners_with_depth);
    if (!measurement.depth_refusal.empty())
        std::printf("depth: %s\n", measurement.depth_refusal.c_str());
    for (const gca::region_measurement &region : measurement.regions)
        print_region(region);
}
} // namespace

/// Runs the check: reads the command line, then the two images, measures the board and prints
/// the report.
/// @param argc [in] the argument count.
/// @param argv [in] the arguments (distortion_options.hpp).
/// @return 0 after the report, 1 when an image cannot be read or the measurement is refused
///     (its reason on stderr), 2 when the command line is refused.
int main(int argc, char *argv[])
{
    const std::optional<gca::distortion_options> opt =
        gca::read_distortion_options(argc, argv, std::cerr);
    if (!opt)
        return 2;
    cv::Mat color;
    cv::Mat depth;
    if (!read_images(*opt, color, depth))
        return 1;
    const gca::board_result result =
        gca::measure_board(color, depth, opt->board, opt->camera, opt->depth_scale_m);
    if (!result.measurement)
    {
        std::cerr << "distortion_check: " << result.refusal << std::endl;
        return 1;
    }
    print_report(*opt, color.size(), *result.measurement);
    return 0;
}
