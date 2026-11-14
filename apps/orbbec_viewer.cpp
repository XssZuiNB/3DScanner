// SPDX-License-Identifier: 0BSD
/// @file orbbec_viewer.cpp
/// Live test of the Orbbec camera: opens it and runs the viewer's loop on it (viewer_loop.hpp),
/// which grabs aligned RGB-D frames, builds the point cloud on the GPU, reports frame rate and
/// timings once a second and shows color | depth | overlay. 's' saves the current point cloud as
/// PLY plus the color and depth images, 'q' or ESC quits.
///
/// The options are viewer_options.hpp's, whose usage line lists them: --list prints the device's
/// stream profiles and exits. --frames N stops after N frames and, headless, saves the last one.
/// A command line with an argument the viewer does not know is refused with the usage line
/// before anything touches the camera, the GPU or a signal handler. The camera's own lines, its
/// name and link, the streams chosen, a refusal and the frames it skipped, come from
/// console_camera here; the loop's lines from viewer_loop.cpp.

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/orbbec_device.hpp"
#include "camera/orbbec_rules.hpp"
#include "gpu/gpu_check.hpp"
#include "types/console_color.hpp"
#include "types/frame_description.hpp"
#include "viewer_loop.hpp"
#include "viewer_options.hpp"

#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

namespace
{
/// The flag the viewer's loop reads; SIGINT sets it. The signal may reach any thread of the
/// program, OrbbecSDK's among them, so a volatile std::sig_atomic_t, which is safe only within
/// one thread, would race with the loop's read; a lock-free atomic is safe both in a signal
/// handler and across threads [S: C++20, [support.signal]].
std::atomic<bool> g_exit_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "a signal handler may only touch a lock-free atomic");

/// Sets the flag the viewer's loop reads; installed for SIGINT.
/// @param signal_number the signal, unused.
void on_sigint(int /*signal_number*/)
{
    g_exit_requested.store(true);
}

/// The Orbbec camera with the viewer's console lines about it. It only prints: every call passes
/// through to the camera unchanged and returns the camera's result unchanged.
class console_camera final : public gca::frame_source
{
public:
    /// Prepares the camera; nothing is opened until open().
    /// @param fps [in] the frame rate of both streams, in frames per second.
    explicit console_camera(std::uint32_t fps)
        : m_camera(fps)
    {
    }

    /// Opens the camera and prints what the opening found: "Warning: camera is not connected via
    /// USB3 (<link>). ..." on a slower link, opened or not; then, refused, the refusal and
    /// "Didn't find the camera!"; opened, "Found camera! Name: <name> (SN <serial>, <link>,
    /// firmware <version>, OrbbecSDK <version>)" and the streams chosen.
    /// @param size [in] the stream size, in px.
    /// @return the camera's result.
    gca::open_result open(const gca::frame_size &size) noexcept override
    {
        const gca::open_result opened = m_camera.open(size);
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
        const gca::source_description described = m_camera.describe();
        std::cout << D_GREEN << "Found camera! Name: " << described.camera << " (SN "
                  << opening.serial_number << ", " << opening.connection_type << ", firmware "
                  << described.firmware_version << ", OrbbecSDK " << described.sdk_version << ")"
                  << RESET << std::endl;
        std::cout << D_GREEN << opening.streams << RESET << std::endl;
        return opened;
    }

    /// Takes the camera's next frame and prints the frames it skipped since the last such line:
    /// "Skipped <count> frame(s), the last because <reason>".
    /// @param out [out] as the camera's next() fills it.
    /// @return the camera's result.
    gca::next_result next(gca::frame &out) noexcept override
    {
        const gca::next_result result = m_camera.next(out);
        const gca::orbbec_skips &skipped = m_camera.skips();
        if (skipped.count != m_skips_shown)
        {
            std::cout << YELLOW << "Skipped " << skipped.count - m_skips_shown
                      << " frame(s), the last because " << skipped.last_reason << RESET
                      << std::endl;
            m_skips_shown = skipped.count;
        }
        return result;
    }

    /// Closes the camera.
    void close() noexcept override
    {
        m_camera.close();
    }

    /// Describes the camera.
    /// @return the camera's description.
    gca::source_description describe() const noexcept override
    {
        return m_camera.describe();
    }

private:
    gca::orbbec_device m_camera;
    std::uint64_t m_skips_shown = 0; ///< the skipped frames printed since the opening
};

/// Prints the GPUs and warms up the first one.
void warm_up_gpu()
{
    cuda_print_devices();
    cuda_warm_up_gpu(0);
}
} // namespace

/// Runs the camera test. Reading the options comes first, before anything touches the camera,
/// the GPU or a signal handler; then the profiles are listed, or the GPU and the camera are set
/// up and the viewer's loop runs until quit, SIGINT or the frame limit.
/// @param argc the argument count.
/// @param argv the arguments (viewer_options.hpp).
/// @return 0 after the list or the loop, 1 when the camera does not open or gives no first frame,
///     2 when the command line is refused.
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
    std::signal(SIGINT, on_sigint);
    warm_up_gpu();
    console_camera camera(opt->fps);
    if (!camera.open({opt->width, opt->height}).opened)
        return 1;
    const bool ran = gca_viewer::run_viewer(camera, *opt,
                                            {.report_period = gca_viewer::report_period,
                                             .exit_requested = g_exit_requested,
                                             .out = std::cout});
    camera.close();
    return ran ? 0 : 1;
}
