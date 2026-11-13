// SPDX-License-Identifier: 0BSD
/// @file orbbec_device.hpp
/// The Orbbec RGB-D camera (Gemini 2 L) behind the gca::device interface, through OrbbecSDK v2.
/// Depth is aligned to the color camera, so both share one resolution and one set of
/// intrinsics and the extrinsics between them are identity, which is what
/// gca::point_cloud::create_from_rgbd() expects. Color is delivered as BGR8, depth as Z16.

#pragma once

#include "camera/camera_device.hpp"

#include <libobsensor/ObSensor.hpp>

#include <memory>

namespace gca
{
class orbbec_device final : public gca::device
{
public:
    /// Prepares a camera with the given color resolution; nothing is opened until
    /// device_start().
    /// @param width color (and aligned depth) width in pixels.
    /// @param height color (and aligned depth) height in pixels.
    /// @param fps frame rate of both streams.
    orbbec_device(uint32_t width, uint32_t height, uint8_t fps);

    orbbec_device(const orbbec_device &) = delete;
    orbbec_device &operator=(const orbbec_device &) = delete;

    /// Stops the stream if it runs.
    ~orbbec_device();

    /// Prints the first connected device's name, serial number, firmware, USB connection and
    /// every color and depth stream profile to stdout. Opens the device, starts no stream.
    static void print_stream_profiles();

private:
    std::shared_ptr<ob::Context> m_context;
    std::shared_ptr<ob::Device> m_device;
    std::shared_ptr<ob::Pipeline> m_pipeline;
    std::shared_ptr<ob::Align> m_sw_align; ///< set when hardware D2C is not available
    std::shared_ptr<ob::VideoStreamProfile> m_color_profile;
    std::shared_ptr<ob::VideoStreamProfile> m_depth_profile;
    bool m_hw_d2c = false;
    bool m_pipeline_active = false;
    float m_depth_value_scale = 1.0f; ///< millimeters per raw depth value

    // The current frames stay referenced so the raw pointers handed out stay valid.
    std::shared_ptr<ob::Frame> m_color_frame;
    std::shared_ptr<ob::Frame> m_depth_frame;
    cv::Mat m_color_buffer; ///< the current color image as BGR8

    /// Opens the first connected Orbbec device and warns when it is not on USB 3.
    /// @return true when a device was opened.
    bool find_device() final;

    /// Selects the stream profiles, starts the pipeline and waits for the first frameset.
    /// @return true when a complete, correctly sized frameset arrived.
    bool start_stream() final;

    /// Converts the depth value scale that start_stream() read from the first depth frame to
    /// meters.
    /// @return meters per raw depth value.
    float read_depth_scale() final;

    /// Reads the color camera's intrinsics from the selected color profile.
    /// @return the color camera's intrinsics at the selected resolution.
    gca::intrinsics read_color_intrinsics() final;

    /// Gives the depth intrinsics, which are the color camera's: depth is aligned to color.
    /// @return the color intrinsics.
    gca::intrinsics read_depth_intrinsics() final;

    /// Gives the color-to-depth extrinsics, the identity: depth is aligned to color.
    /// @return the identity rotation and a zero translation.
    gca::extrinsics read_color_to_depth_extrinsics() final;

    /// Gives the depth-to-color extrinsics, the identity: depth is aligned to color.
    /// @return the identity rotation and a zero translation.
    gca::extrinsics read_depth_to_color_extrinsics() final;

    /// Waits up to 5 s for the next frameset and makes it the current one. On a timeout it
    /// prints "No frames received from Orbbec camera for 5s!" and keeps the previous frames
    /// [S: orbbec_device.cpp].
    void receive_data_from_device() final;

    /// Hands the current color image to gca::device::get_color_cv_mat().
    /// @return the current color image (BGR8), sharing the device's buffer.
    cv::Mat set_color_to_cv_mat() final;

    /// Wraps the current aligned depth frame as an image for gca::device::get_depth_cv_mat().
    /// @return the current aligned depth image (CV_16UC1), sharing the frame's buffer.
    cv::Mat set_depth_to_cv_mat() final;

    /// Stops the pipeline if it runs.
    void stop_stream() final;

    /// Picks the color profile (BGR, RGB, MJPG or YUYV at the requested size and rate) and a
    /// 16-bit depth profile, hardware D2C when the device offers it for that color profile, else
    /// the SDK's software Align filter.
    /// @return true when both profiles were found.
    bool select_profiles();

    /// Waits for one frameset, aligns it in software if needed, decodes the color frame and
    /// makes both frames the current ones.
    /// @param timeout_ms how long to wait for the frameset.
    /// @return true when a frameset with color and depth arrived.
    bool grab(uint32_t timeout_ms);

    /// Converts a color frame of any supported format into m_color_buffer as BGR8.
    /// @param color the frame to convert.
    void decode_color(const std::shared_ptr<ob::VideoFrame> &color);
};
} // namespace gca
