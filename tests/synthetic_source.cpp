// SPDX-License-Identifier: 0BSD
/// @file synthetic_source.cpp
/// The tests' synthetic source: the scene's textures and known surfaces, the trajectories, the
/// renderer of the surfaces, the sphere and the mesh with its depth noise, and the source's states
/// (see synthetic_source.hpp).

#include "synthetic_source.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace gca_test
{
namespace
{
/// The time between two frames, in µs: 30 fps [?: a choice, the camera's rate].
constexpr std::uint64_t frame_period_us = 33333;

/// The least distance from a camera to the point it looks at, in m [?: a choice, far below a
/// scene's size] (look_at()).
constexpr double least_view_distance_m = 1e-9;

/// The least sine of a view's angle from the vertical (look_at()): below it the level x axis,
/// z x up of that length, is not defined [?: a choice]; at it the inputs' rounding of 2.2e-16
/// turns the x axis by up to 2.2e-12 rad [S: computed, 2.2e-16 / 1e-4].
constexpr double least_view_sine = 1e-4;

/// One sine wave of the plane's texture.
struct plane_wave
{
    double direction_deg; ///< its direction in the plane, from the world's x toward its y
    double wavelength_m;  ///< its wavelength, in m
    double phase_rad;     ///< its phase at the world's origin
};

/// The plane's waves, red, green and blue: along x, along y and diagonally, each with its own
/// wavelength [?: a choice].
constexpr plane_wave plane_waves[3] = {{0.0, 0.10, 0.0}, {90.0, 0.07, 0.5}, {60.0, 0.05, 1.0}};

/// One sine wave of the box's texture, in space.
struct solid_wave
{
    double azimuth_deg;   ///< its direction's azimuth, from the world's x toward its y
    double elevation_deg; ///< its direction's rise above the plane, in deg, within +-35 to 50
    double wavelength_m;  ///< its wavelength, in m
    double phase_rad;     ///< its phase at the world's origin
};

/// The box's waves, red, green and blue: three directions apart in azimuth, each rising or
/// falling by 35 to 50 degrees, with wavelengths and phases other than the plane's
/// [?: a choice].
constexpr solid_wave box_waves[3] = {
    {20.0, 35.0, 0.045, 0.3}, {140.0, 50.0, 0.060, 1.7}, {260.0, -40.0, 0.080, 2.9}};

/// Each channel's mean and amplitude: the channel stays within 28 to 228 [S: 128 +- 100].
constexpr double texture_mean = 128.0;
constexpr double texture_amplitude = 100.0;

/// Returns one channel of a texture: a sine wave around the mean.
/// @param along_m [in] the point's distance along the wave's direction from the world's origin,
///        in m.
/// @param wavelength_m [in] the wave's wavelength, in m, > 0.
/// @param phase_rad [in] its phase at the world's origin, in rad.
/// @return the channel, unrounded, within 28 to 228 [S: texture_mean +- texture_amplitude].
double wave_level(double along_m, double wavelength_m, double phase_rad)
{
    return texture_mean +
           texture_amplitude * std::sin(2.0 * M_PI * along_m / wavelength_m + phase_rad);
}

/// Returns a unit vector along the plane, at an angle from the world's x axis.
/// @param angle_deg [in] the angle, from x toward y, in degrees.
/// @return the vector, in the world frame.
Eigen::Vector3d along_plane(double angle_deg)
{
    const double a = angle_deg * M_PI / 180.0;
    return {std::cos(a), std::sin(a), 0.0};
}

/// Builds the scene's known surfaces: the plane when there is one, then the box's top and its sides
/// on its own +x, +y, -x and -y, each side's first axis level and its second up. The bottom face is
/// left out: it lies in the plane, which hides it from either side.
/// @param scene [in] the plane and the box.
/// @return the surfaces, in the world frame.
std::vector<synthetic_surface> known_surfaces(const synthetic_scene &scene)
{
    const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
    std::vector<synthetic_surface> surfaces;
    if (scene.plane)
        surfaces.push_back({Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
                            Eigen::Vector3d::UnitY(), scene.plane->half_size_x_m,
                            scene.plane->half_size_y_m, synthetic_texture::plane});
    if (!scene.box)
        return surfaces;
    const synthetic_box &box = *scene.box;
    const Eigen::Vector3d foot(box.centre_x_m, box.centre_y_m, 0.0);
    const double half_x_m = 0.5 * box.size_x_m;
    const double half_y_m = 0.5 * box.size_y_m;
    const double half_height_m = 0.5 * box.height_m;
    surfaces.push_back({foot + box.height_m * up, along_plane(box.yaw_deg),
                        along_plane(box.yaw_deg + 90.0), half_x_m, half_y_m,
                        synthetic_texture::box});
    // A side whose outward normal is n has the level first axis up x n, and (up x n) x up = n.
    struct side
    {
        double turn_deg;     ///< its outward normal's turn from the box's +x, in deg
        double half_depth_m; ///< half the box's extent along that normal, in m
        double half_width_m; ///< half the side's extent across it, in m
    };
    const std::array<side, 4> sides = {{{0.0, half_x_m, half_y_m},
                                        {90.0, half_y_m, half_x_m},
                                        {180.0, half_x_m, half_y_m},
                                        {270.0, half_y_m, half_x_m}}};
    for (const side &s : sides)
    {
        const Eigen::Vector3d normal = along_plane(box.yaw_deg + s.turn_deg);
        surfaces.push_back({foot + s.half_depth_m * normal + half_height_m * up, up.cross(normal),
                            up, s.half_width_m, half_height_m, synthetic_texture::box});
    }
    return surfaces;
}

/// Maps what the options' ending asks for to the status next() reports.
/// @param ending [in] the ending.
/// @return the status of the same name.
gca::next_status ending_status(synthetic_ending ending)
{
    switch (ending)
    {
    case synthetic_ending::end:
        return gca::next_status::end;
    case synthetic_ending::timeout:
        return gca::next_status::timeout;
    case synthetic_ending::failure:
        return gca::next_status::failure;
    }
    // Not reached: the switch names every ending, and -Wswitch says when one is added.
    return gca::next_status::failure;
}

/// Finds where a ray first meets a sphere in front of its origin: the smaller root s of
/// |origin + s ray - centre|^2 = radius^2 that is > 0, else the larger one when it is > 0, so that
/// a camera inside the sphere sees its inner surface.
/// @param sphere [in] the sphere.
/// @param origin [in] the ray's origin, in the world frame, in m.
/// @param ray [in] the ray's direction, in the world frame, not zero; not unit: s is in the
///        units that make origin + s ray a point in m.
/// @return s; nothing when the ray misses the sphere or meets it only behind its origin.
std::optional<double> hit_sphere(const synthetic_sphere &sphere, const Eigen::Vector3d &origin,
                                 const Eigen::Vector3d &ray)
{
    const Eigen::Vector3d offset = origin - sphere.centre_m;
    const double a = ray.dot(ray);
    const double b = ray.dot(offset);
    const double c = offset.dot(offset) - sphere.radius_m * sphere.radius_m;
    const double discriminant = b * b - a * c;
    if (!(discriminant >= 0.0))
        return std::nullopt;
    const double root = std::sqrt(discriminant);
    const double nearer = (-b - root) / a;
    const double farther = (-b + root) / a;
    if (nearer > 0.0)
        return nearer;
    if (farther > 0.0)
        return farther;
    return std::nullopt;
}

/// The nearest hit of a mesh along each pixel's ray, in a picture's pixel order.
struct mesh_picture
{
    std::vector<double> depth_m;              ///< each pixel's nearest hit's depth; +inf for none
    std::vector<std::array<double, 3>> color; ///< the hit's color, its vertices' weighted, 0 to 255
};

/// Finds where the ray of each pixel first meets a mesh in front of the camera, by the one pinhole
/// convention: for each triangle whose vertices all lie in front of the camera's plane, over the
/// pixels whose centres lie within its projected bounds, the exact hit of the pixel's ray
/// ((u - cx) / fx, (v - cy) / fy, 1) with the triangle [S: T. Möller, B. Trumbore, "Fast,
/// Minimum Storage Ray/Triangle Intersection", Journal of Graphics Tools 2(1), 1997], its edges
/// included, so that two triangles sharing an edge leave no pixel between them unseen. The ray's z
/// is 1, so the hit's parameter is its depth; the nearer of two hits is kept, the earlier triangle
/// on a tie.
/// @param mesh [in] the mesh, in the world frame.
/// @param camera_to_world [in] the camera's pose: camera frame to world frame.
/// @param description [in] the picture's size and intrinsics.
/// @return each pixel's nearest hit's depth, in m, and color.
mesh_picture render_mesh(const synthetic_mesh &mesh, const Eigen::Isometry3d &camera_to_world,
                         const gca::frame_description &description)
{
    const std::uint32_t width = description.size.width;
    const std::uint32_t height = description.size.height;
    const gca::intrinsics &k = description.intrinsics;
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    mesh_picture picture{std::vector<double>(pixels, std::numeric_limits<double>::infinity()),
                         std::vector<std::array<double, 3>>(pixels)};
    const Eigen::Isometry3d world_to_camera = camera_to_world.inverse();
    for (const std::array<std::uint32_t, 3> &triangle : mesh.triangles)
    {
        std::array<Eigen::Vector3d, 3> corner;
        double u_low = std::numeric_limits<double>::infinity();
        double u_high = -u_low;
        double v_low = u_low;
        double v_high = -u_low;
        bool in_front = true;
        for (std::size_t i = 0; i < 3; i++)
        {
            corner[i] = world_to_camera * mesh.vertices_m.at(triangle[i]);
            in_front = in_front && corner[i].z() > 0.0;
            const double u = k.fx * corner[i].x() / corner[i].z() + k.cx;
            const double v = k.fy * corner[i].y() / corner[i].z() + k.cy;
            u_low = std::min(u_low, u);
            u_high = std::max(u_high, u);
            v_low = std::min(v_low, v);
            v_high = std::max(v_high, v);
        }
        if (!in_front)
            continue;
        const double first_u = std::max(std::ceil(u_low), 0.0);
        const double last_u = std::min(std::floor(u_high), static_cast<double>(width) - 1.0);
        const double first_v = std::max(std::ceil(v_low), 0.0);
        const double last_v = std::min(std::floor(v_high), static_cast<double>(height) - 1.0);
        const Eigen::Vector3d edge_1 = corner[1] - corner[0];
        const Eigen::Vector3d edge_2 = corner[2] - corner[0];
        const Eigen::Vector3d from_corner = -corner[0];
        const Eigen::Vector3d q = from_corner.cross(edge_1);
        const double s_numerator = edge_2.dot(q);
        for (double v = first_v; v <= last_v; v++)
        {
            for (double u = first_u; u <= last_u; u++)
            {
                const Eigen::Vector3d ray = pixel_ray(k, u, v);
                const Eigen::Vector3d p = ray.cross(edge_2);
                const double determinant = edge_1.dot(p);
                // A ray in the triangle's plane meets it nowhere or along a line: no hit.
                if (determinant == 0.0)
                    continue;
                const double b_1 = from_corner.dot(p) / determinant;
                const double b_2 = ray.dot(q) / determinant;
                const double s = s_numerator / determinant;
                if (!(b_1 >= 0.0 && b_2 >= 0.0 && b_1 + b_2 <= 1.0 && s > 0.0))
                    continue;
                const std::size_t pixel =
                    static_cast<std::size_t>(v) * width + static_cast<std::size_t>(u);
                if (!(s < picture.depth_m[pixel]))
                    continue;
                picture.depth_m[pixel] = s;
                const double b_0 = 1.0 - b_1 - b_2;
                for (std::size_t c = 0; c < 3; c++)
                    picture.color[pixel][c] = b_0 * mesh.colors.at(triangle[0])[c] +
                                              b_1 * mesh.colors.at(triangle[1])[c] +
                                              b_2 * mesh.colors.at(triangle[2])[c];
            }
        }
    }
    return picture;
}

/// Draws the depth noise of one frame, one standard normal value for each pixel in the frame's
/// order, from std::mt19937_64 seeded with the noise's seed and the frame's index
/// (synthetic_depth_noise).
/// @param noise [in] the noise.
/// @param index [in] the frame's index in its opening.
/// @param pixels [in] the frame's pixels.
/// @return the values, each of mean 0 and standard deviation 1.
std::vector<double> standard_normals(const synthetic_depth_noise &noise, std::uint64_t index,
                                     std::size_t pixels)
{
    std::seed_seq seeds{noise.seed, static_cast<std::uint32_t>(index & 0xFFFFFFFFu),
                        static_cast<std::uint32_t>(index >> 32)};
    std::mt19937_64 generator(seeds);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> values(pixels);
    for (double &value : values)
        value = normal(generator);
    return values;
}

/// Renders the known surfaces, the sphere and the mesh as the camera sees them into a prepared
/// frame, by the one pinhole convention: the ray of pixel (u, v) is ((u - cx) / fx,
/// (v - cy) / fy, 1) in the camera frame. Its point at parameter s has the camera-frame depth s,
/// so the ray's parameter where it meets a surface is the hit's depth, and the nearest surface is
/// the one of the smallest s > 0; on a tie a known surface or the sphere before the mesh. With
/// depth noise, the depth s becomes s + c s^2 n, n the pixel's standard normal value, before it is
/// rounded; the color stays the hit's.
/// @param surfaces [in] the surfaces, in the world frame.
/// @param sphere [in] the sphere, in the world frame, or none.
/// @param mesh [in] the mesh, in the world frame, or none.
/// @param camera_to_world [in] the camera's pose: camera frame to world frame.
/// @param noise [in] the depth noise, or none.
/// @param out [in,out] prepared for its description and index (frame::prepare()); every pixel
///            of its color and depth is written.
void render(const std::vector<synthetic_surface> &surfaces,
            const std::optional<synthetic_sphere> &sphere,
            const std::optional<synthetic_mesh> &mesh, const Eigen::Isometry3d &camera_to_world,
            const std::optional<synthetic_depth_noise> &noise, gca::frame &out)
{
    const gca::frame_description &description = out.description();
    const gca::intrinsics &k = description.intrinsics;
    const Eigen::Matrix3d rotation = camera_to_world.linear();
    const Eigen::Vector3d origin = camera_to_world.translation();
    const std::span<std::uint8_t> color = out.color();
    const std::span<std::uint16_t> depth = out.depth();
    const std::optional<mesh_picture> meshed =
        mesh ? std::optional<mesh_picture>(render_mesh(*mesh, camera_to_world, description))
             : std::nullopt;
    const std::vector<double> normals =
        noise ? standard_normals(*noise, out.index(), depth.size()) : std::vector<double>{};
    for (std::uint32_t v = 0; v < description.size.height; v++)
    {
        for (std::uint32_t u = 0; u < description.size.width; u++)
        {
            const std::size_t pixel = static_cast<std::size_t>(v) * description.size.width + u;
            const Eigen::Vector3d ray_world = rotation * pixel_ray(k, u, v);
            const std::optional<synthetic_hit> nearest =
                first_hit(surfaces, sphere, origin, ray_world);
            const bool on_mesh = meshed && (nearest ? meshed->depth_m[pixel] < nearest->s
                                                    : meshed->depth_m[pixel] <
                                                          std::numeric_limits<double>::infinity());
            const double nearest_s = on_mesh ? meshed->depth_m[pixel] : nearest ? nearest->s : 0.0;
            const double noisy_s =
                noise ? nearest_s + noise->noise_per_m * nearest_s * nearest_s * normals[pixel]
                      : nearest_s;
            const double value = std::round(noisy_s / description.depth_unit_m);
            const bool seen = (on_mesh || nearest.has_value()) && value >= 1.0 &&
                              value <= std::numeric_limits<std::uint16_t>::max();
            depth[pixel] = seen ? static_cast<std::uint16_t>(value) : 0;
            const std::array<double, 3> rgb =
                !seen     ? std::array<double, 3>{}
                : on_mesh ? meshed->color[pixel]
                          : texture_color(nearest->texture, origin + nearest_s * ray_world);
            for (std::size_t c = 0; c < 3; c++)
                color[3 * pixel + c] = static_cast<std::uint8_t>(std::lround(rgb[c]));
        }
    }
}
} // namespace

std::array<double, 3> texture_color(synthetic_texture texture, const Eigen::Vector3d &point_m)
{
    std::array<double, 3> rgb{};
    for (std::size_t c = 0; c < 3; c++)
    {
        switch (texture)
        {
        case synthetic_texture::plane: {
            const plane_wave &wave = plane_waves[c];
            rgb[c] = wave_level(along_plane(wave.direction_deg).dot(point_m), wave.wavelength_m,
                                wave.phase_rad);
            break;
        }
        case synthetic_texture::box: {
            const solid_wave &wave = box_waves[c];
            const double e = wave.elevation_deg * M_PI / 180.0;
            const Eigen::Vector3d direction = std::cos(e) * along_plane(wave.azimuth_deg) +
                                              std::sin(e) * Eigen::Vector3d::UnitZ();
            rgb[c] = wave_level(direction.dot(point_m), wave.wavelength_m, wave.phase_rad);
            break;
        }
        }
    }
    return rgb;
}

Eigen::Vector3d pixel_ray(const gca::intrinsics &k, double u, double v)
{
    return {(u - double{k.cx}) / k.fx, (v - double{k.cy}) / k.fy, 1.0};
}

double distance_from(const synthetic_surface &surface, const Eigen::Vector3d &world)
{
    const Eigen::Vector3d local = world - surface.centre_m;
    const double s = local.dot(surface.axis_u);
    const double t = local.dot(surface.axis_v);
    const double h = local.dot(surface.axis_u.cross(surface.axis_v));
    const double ds = std::abs(s) - std::min(std::abs(s), surface.half_u_m);
    const double dt = std::abs(t) - std::min(std::abs(t), surface.half_v_m);
    return std::sqrt(ds * ds + dt * dt + h * h);
}

std::optional<synthetic_hit> first_hit(const std::vector<synthetic_surface> &surfaces,
                                       const std::optional<synthetic_sphere> &sphere,
                                       const Eigen::Vector3d &origin, const Eigen::Vector3d &ray)
{
    std::optional<synthetic_hit> nearest;
    for (std::size_t i = 0; i < surfaces.size(); i++)
    {
        const synthetic_surface &surface = surfaces[i];
        // Where origin + s * ray meets the surface's plane. A plane behind the camera gives
        // s <= 0, a ray parallel to it an infinite or NaN s, and the comparisons below refuse
        // each.
        const Eigen::Vector3d normal = surface.axis_u.cross(surface.axis_v);
        const double s = normal.dot(surface.centre_m - origin) / normal.dot(ray);
        const Eigen::Vector3d offset = origin + s * ray - surface.centre_m;
        const bool met = s > 0.0 && std::abs(offset.dot(surface.axis_u)) <= surface.half_u_m &&
                         std::abs(offset.dot(surface.axis_v)) <= surface.half_v_m;
        if (met && (!nearest || s < nearest->s))
            nearest = synthetic_hit{s, i, surface.texture};
    }
    const std::optional<double> sphere_s = sphere ? hit_sphere(*sphere, origin, ray) : std::nullopt;
    if (sphere_s && (!nearest || *sphere_s < nearest->s))
        nearest = synthetic_hit{*sphere_s, surfaces.size(), synthetic_texture::box};
    return nearest;
}

std::optional<Eigen::Isometry3d> look_at(const Eigen::Vector3d &position,
                                         const Eigen::Vector3d &target, double roll_deg)
{
    if (!position.allFinite() || !target.allFinite() || !std::isfinite(roll_deg))
        return std::nullopt;
    const Eigen::Vector3d view = target - position;
    const double distance_m = view.norm();
    if (!(distance_m > least_view_distance_m))
        return std::nullopt;
    // The camera's axes in the world: z toward the target; x level and to the right, z x up,
    // whose length is the sine of the view's angle from the vertical; y = z x x, which points
    // down for every view that is not vertical: y . up = -cos of the view's angle from level.
    const Eigen::Vector3d z = view / distance_m;
    const Eigen::Vector3d level = z.cross(Eigen::Vector3d::UnitZ());
    const double level_length = level.norm();
    if (!(level_length > least_view_sine))
        return std::nullopt;
    const Eigen::Vector3d x = level / level_length;
    const Eigen::Vector3d y = z.cross(x);
    Eigen::Matrix3d upright;
    upright.col(0) = x;
    upright.col(1) = y;
    upright.col(2) = z;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = upright * Eigen::AngleAxisd(roll_deg * M_PI / 180.0, Eigen::Vector3d::UnitZ());
    pose.translation() = position;
    return pose;
}

std::optional<std::vector<Eigen::Isometry3d>> walk_around(const synthetic_walk &walk)
{
    std::vector<Eigen::Isometry3d> trajectory;
    trajectory.reserve(walk.pose_count);
    for (std::size_t i = 0; i < walk.pose_count; i++)
    {
        const double a = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(walk.pose_count);
        const double radius_m = walk.radius_m + walk.radius_swing_m * std::sin(2.0 * a);
        const Eigen::Vector3d position(walk.centre_m.x() + radius_m * std::cos(a),
                                       walk.centre_m.y() + radius_m * std::sin(a),
                                       walk.height_m + walk.height_swing_m * std::cos(3.0 * a));
        const Eigen::Vector3d target =
            walk.centre_m +
            walk.target_jitter_m * Eigen::Vector3d(std::sin(3.0 * a), std::cos(5.0 * a), 0.0);
        const std::optional<Eigen::Isometry3d> pose =
            look_at(position, target, walk.roll_swing_deg * std::sin(5.0 * a));
        if (!pose)
            return std::nullopt;
        trajectory.push_back(*pose);
    }
    return trajectory;
}

synthetic_walk tabletop_walk()
{
    return {Eigen::Vector3d(tabletop_box.centre_x_m, tabletop_box.centre_y_m,
                            0.5 * tabletop_box.height_m),
            12,
            0.45,
            0.05,
            0.35,
            0.05,
            10.0,
            0.02};
}

synthetic_source::synthetic_source(synthetic_source_options options)
    : m_options(std::move(options))
    , m_surfaces(known_surfaces(m_options.scene))
{
}

gca::open_result synthetic_source::open(const gca::frame_size &size) noexcept
{
    if (m_open)
        return {false, "the synthetic source is open already"};
    if (!m_options.open_refusal.empty())
        return {false, m_options.open_refusal};
    const gca::frame_size &own = m_options.description.size;
    if (size != own)
        return {false, "the synthetic source renders " + std::to_string(own.width) + "x" +
                           std::to_string(own.height) + " px, not " + std::to_string(size.width) +
                           "x" + std::to_string(size.height) + " px"};
    m_open = true;
    m_next_index = 0;
    return {true, {}};
}

gca::next_result synthetic_source::next(gca::frame &out) noexcept
{
    if (!m_open)
        return {gca::next_status::failure, "the synthetic source is not open"};
    if (m_next_index >= m_options.scene.trajectory.size())
        return {ending_status(m_options.ending), m_options.ending_message};
    out.prepare(m_options.description, m_next_index, m_next_index * frame_period_us);
    render(m_surfaces, m_options.scene.sphere, m_options.scene.mesh,
           m_options.scene.trajectory[m_next_index], m_options.depth_noise, out);
    m_next_index++;
    return {gca::next_status::frame, {}};
}

void synthetic_source::close() noexcept
{
    m_open = false;
}

gca::source_description synthetic_source::describe() const noexcept
{
    return {"synthetic source: known textured surfaces along a known trajectory", {}, {}};
}

const synthetic_scene &synthetic_source::scene() const noexcept
{
    return m_options.scene;
}

const std::vector<synthetic_surface> &synthetic_source::surfaces() const noexcept
{
    return m_surfaces;
}
} // namespace gca_test
