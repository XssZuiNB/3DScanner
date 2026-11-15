// SPDX-License-Identifier: 0BSD
/// @file orbbec_viewer.cpp
/// Live test of the Orbbec camera: opens it and runs the viewer's loop on it (viewer_loop.hpp),
/// which grabs aligned RGB-D frames, makes their organized points on the GPU, reports frame rate
/// and timings once a second and shows color | depth | overlay. 's' saves the current frame's
/// points with their normals as a model's PLY plus the color and depth images, 'q' or ESC quits.
///
/// The options are viewer_options.hpp's, whose usage line lists them: --list prints the device's
/// stream profiles and exits. --frames N stops after N frames and, headless, saves the last one.
/// --sequence DIR writes every frame taken into a sequence in DIR (viewer_loop.hpp). --hold-color
/// holds the color camera's exposure and white balance after the settling and switches both
/// automatic controls on again at the end, after an exception out of the viewer's loop too
/// (camera/color_hold.hpp). A command line with an argument the viewer does not know, or a
/// directory for the sequence it cannot use, is refused with the usage line before anything
/// touches the camera, the GPU or a signal handler. The camera's own lines, its name and link, the
/// streams chosen, the IMU's streams or why it opened without them, a refusal, the frames and IMU
/// readings it skipped and the hold's readings and refusals, come from console_camera here; the
/// loop's lines from viewer_loop.cpp.

#include "camera/color_hold.hpp"
#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/orbbec_device.hpp"
#include "camera/orbbec_rules.hpp"
#include "types/console_color.hpp"
#include "types/frame_description.hpp"
#include "viewer_loop.hpp"
#include "viewer_options.hpp"

#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

namespace
{
/// The flag the viewer's loop reads; SIGINT, SIGTERM and SIGHUP set it. The signal may reach any
/// thread of the program, OrbbecSDK's among them, so a volatile std::sig_atomic_t, which is safe
/// only within one thread, would race with the loop's read; a lock-free atomic is safe both in a
/// signal handler and across threads [S: C++20, [support.signal]].
std::atomic<bool> g_exit_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "a signal handler may only touch a lock-free atomic");

/// Sets the flag the viewer's loop reads; installed for SIGINT, SIGTERM and SIGHUP.
/// @param signal_number the signal, unused.
void on_stop_signal(int /*signal_number*/)
{
    g_exit_requested.store(true);
}

/// Makes Ctrl-C (SIGINT), a kill or a timeout's end (SIGTERM) and a closed terminal or SSH session
/// (SIGHUP) end the viewer's loop as 'q' does, so that the camera is closed and a sequence gets
/// its end and its closing line. A second signal only sets the flag again.
void install_stop_handlers()
{
    for (const int signal_number : {SIGINT, SIGTERM, SIGHUP})
        std::signal(signal_number, on_stop_signal);
}

/// The Orbbec camera, with --hold-color behind the hold of its color controls, and the viewer's
/// console lines about it. It only prints: every call passes through to the camera, or the hold,
/// unchanged and returns its result unchanged.
class console_camera final : public gca::frame_source
{
public:
    /// Prepares the camera; nothing is opened until open().
    /// @param fps [in] the frame rate of both streams, in frames per second.
    /// @param hold_color [in] whether the color camera's exposure and white balance are held.
    console_camera(std::uint32_t fps, bool hold_color)
        : m_camera(fps)
    {
        if (hold_color)
            m_hold.emplace(m_camera, m_camera);
    }

    /// Closes the camera, so that the hold's last reading is printed also when the program ends
    /// by an exception; the hold, destroyed before the camera, switches the controls on even so.
    ~console_camera() override
    {
        close();
    }

    /// Opens the camera and prints what the opening found: "Warning: camera is not connected via
    /// USB3 (<link>). ..." on a slower link, opened or not; then, refused, the refusal and
    /// "Didn't find the camera!"; opened, "Found camera! Name: <name> (SN <serial>, <link>,
    /// firmware <version>, OrbbecSDK <version>)", the streams chosen, the IMU's streams or "No
    /// IMU: <reason>; the frames come without its samples", and, with the hold, its first reading
    /// and the refusals so far.
    /// @param size [in] the stream size, in px.
    /// @return the camera's result.
    gca::open_result open(const gca::frame_size &size) noexcept override
    {
        const gca::open_result opened = source().open(size);
        const gca::orbbec_opening &opening = m_camera.opening();
        if (!opening.connection_type.empty() &&
            opening.connection_type.find("USB3") == std::string::npos)
        {
            std::cout << YELLOW << "Warning: camera is not connected via USB3 ("
                      << opening.connection_type
                      << "). Bandwidth and available profiles are limited." << RESET << std::endl;
        }
        if (!opened.opened)
        {
            std::cout << RED << opened.refusal << RESET << std::endl;
            std::cout << RED << "Didn't find the camera!" << RESET << std::endl;
            return opened;
        }
        m_skips_shown = 0;
        m_imu_skips_shown = 0;
        const gca::source_description described = m_camera.describe();
        std::cout << D_GREEN << "Found camera! Name: " << described.camera << " (SN "
                  << opening.serial_number << ", " << opening.connection_type << ", firmware "
                  << described.firmware_version << ", OrbbecSDK " << described.sdk_version << ")"
                  << RESET << std::endl;
        std::cout << D_GREEN << opening.streams << RESET << std::endl;
        if (opening.imu_refusal.empty())
            std::cout << D_GREEN << opening.imu_streams << RESET << std::endl;
        else
            std::cout << YELLOW << "No IMU: " << opening.imu_refusal
                      << "; the frames come without its samples" << RESET << std::endl;
        m_readings_shown = 0;
        m_refusals_shown = 0;
        print_hold();
        return opened;
    }

    /// Takes the camera's next frame and prints the frames it skipped since the last such line,
    /// "Skipped <count> frame(s), the last because <reason>", the IMU's readings it lost,
    /// "Skipped <count> IMU reading(s), the last because <reason>", and, with the hold, its
    /// readings and refusals of the hold after the settling.
    /// @param out [out] as the camera's next() fills it.
    /// @return the camera's result.
    gca::next_result next(gca::frame &out) noexcept override
    {
        const gca::next_result result = source().next(out);
        print_skips("frame(s)", m_camera.skips(), m_skips_shown);
        print_skips("IMU reading(s)", m_camera.imu_skips(), m_imu_skips_shown);
        print_hold();
        return result;
    }

    /// Closes the camera; with the hold, prints its last reading and the refusals at the end.
    void close() noexcept override
    {
        source().close();
        print_hold();
    }

    /// Describes the camera.
    /// @return the camera's description.
    gca::source_description describe() const noexcept override
    {
        return m_camera.describe();
    }

private:
    /// Gives what the viewer's calls go to.
    /// @return the hold with --hold-color, else the camera.
    gca::frame_source &source() noexcept
    {
        if (m_hold)
            return *m_hold;
        return m_camera;
    }

    /// Prints what the camera skipped since the last such line, when there is more: "Skipped
    /// <count> <what>, the last because <reason>".
    /// @param what [in] what was skipped, "frame(s)" or "IMU reading(s)".
    /// @param skipped [in] the opening's count and the last one's reason.
    /// @param shown [in,out] the count printed so far; set to skipped's count.
    static void print_skips(const char *what, const gca::orbbec_skips &skipped,
                            std::uint64_t &shown)
    {
        if (skipped.count == shown)
            return;
        std::cout << YELLOW << "Skipped " << skipped.count - shown << " " << what
                  << ", the last because " << skipped.last_reason << RESET << std::endl;
        shown = skipped.count;
    }

    /// Prints the hold's readings and refusals that are not printed yet, each as its line
    /// (camera/color_hold.hpp), the refusals in yellow; nothing without the hold.
    void print_hold() noexcept
    {
        if (!m_hold)
            return;
        const gca::color_hold_report &report = m_hold->report();
        for (; m_readings_shown < report.readings.size(); m_readings_shown++)
            std::cout << gca::reading_line(report.readings[m_readings_shown]) << std::endl;
        for (; m_refusals_shown < report.refusals.size(); m_refusals_shown++)
            std::cout << YELLOW << gca::refusal_line(report.refusals[m_refusals_shown]) << RESET
                      << std::endl;
    }

    gca::orbbec_device m_camera;
    /// The hold with --hold-color; declared after the camera, so that it is destroyed first.
    std::optional<gca::color_hold> m_hold;
    std::uint64_t m_skips_shown = 0;     ///< the skipped frames printed since the opening
    std::uint64_t m_imu_skips_shown = 0; ///< the IMU's lost readings printed since the opening
    std::size_t m_readings_shown = 0;    ///< the hold's readings printed since the opening
    std::size_t m_refusals_shown = 0;    ///< the hold's refusals printed since the opening
};

/// Names the sequence --sequence asks for; nothing is created until the loop opens it.
/// @param opt [in] the options.
/// @return the sequence in the directory opt names; nothing without --sequence.
std::optional<gca_viewer::sequence_file> name_sequence(const gca_viewer::options &opt)
{
    // Built in place: a sequence_file is neither copied nor moved.
    if (!opt.sequence_directory)
        return std::nullopt;
    return std::optional<gca_viewer::sequence_file>(std::in_place, *opt.sequence_directory);
}

/// Opens the camera and runs the viewer's loop on it, which sets up the GPU from the first frame,
/// until quit, a stop signal, the frame limit or a failure of the GPU, writing the sequence
/// --sequence asks for; then closes the camera. An exception passes through, and on its way the
/// camera's destructor closes it, with --hold-color after switching the controls on again.
/// @param opt [in] the options.
/// @return 0 after the loop, 1 when the camera does not open or gives no first frame, the GPU
///     cannot be set up or a GPU step fails, or the sequence cannot be created or written.
int run_camera(const gca_viewer::options &opt)
{
    install_stop_handlers();
    console_camera camera(opt.fps, opt.hold_color);
    if (!camera.open({opt.width, opt.height}).opened)
        return 1;
    std::optional<gca_viewer::sequence_file> sequence = name_sequence(opt);
    const bool ran = gca_viewer::run_viewer(camera, opt,
                                            {.report_period = gca_viewer::report_period,
                                             .exit_requested = g_exit_requested,
                                             .out = std::cout,
                                             .sequence = sequence ? &*sequence : nullptr});
    camera.close();
    return ran ? 0 : 1;
}
} // namespace

/// Runs the camera test. Reading the options comes first, before anything touches the camera,
/// the GPU or a signal handler; then the profiles are listed, or the camera runs (run_camera()).
/// An exception out of the camera's run is caught here, so that the stack unwinds and the camera
/// is closed, its color controls switched on again; without a handler the program could end
/// without unwinding [S: C++20, [except.handle]].
/// @param argc the argument count.
/// @param argv the arguments (viewer_options.hpp).
/// @return 0 after the list or the loop, 1 when the camera does not open or gives no first frame,
///     the GPU cannot be set up or a GPU step fails, the sequence cannot be created or written, or
///     an exception ended the run, 2 when the command line is refused.
int main(int argc, char *argv[])
{
    const std::optional<gca_viewer::options> opt =
        gca_viewer::read_options(argc, argv, std::getenv("DISPLAY") == nullptr, std::cerr);
    if (!opt)
        return 2;
    if (opt->list_only)
    {
        gca::orbbec_device::print_stream_profiles();
        return 0;
    }
    try
    {
        return run_camera(*opt);
    }
    catch (const std::exception &e)
    {
        std::cout << RED << "orbbec_viewer: " << e.what() << RESET << std::endl;
    }
    catch (...)
    {
        std::cout << RED << "orbbec_viewer: an exception that is no std::exception" << RESET
                  << std::endl;
    }
    return 1;
}
