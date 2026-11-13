// SPDX-License-Identifier: 0BSD
/// @file orbbec_device.cpp
/// The Orbbec camera behind gca::device: profile selection, depth-to-color alignment and
/// color decoding to BGR8 (see orbbec_device.hpp).

#include "orbbec_device.hpp"

#include <iostream>
#include <string>

#include <libobsensor/hpp/TypeHelper.hpp>

namespace gca
{
namespace
{
/// The depth formats asked for, in order. On USB 2 the Gemini 2 L offers only Y14 and RLE
/// [M 2026-11-13]; start_stream() checks that what arrives is 16 bit.
constexpr OBFormat depth_formats[] = {OB_FORMAT_Y16, OB_FORMAT_Z16, OB_FORMAT_Y14, OB_FORMAT_RLE};

/// Names a stream format for log lines.
/// @param f the format.
/// @return the SDK's name for it.
std::string format_name(OBFormat f)
{
    return ob::TypeHelper::convertOBFormatTypeToString(f);
}

/// Finds the first profile in a list that matches; 0 or OB_FORMAT_ANY matches anything.
/// @param list the profiles to search.
/// @param w width in pixels.
/// @param h height in pixels.
/// @param fmt stream format.
/// @param fps frame rate.
/// @return the profile, or nullptr when none matches.
std::shared_ptr<ob::VideoStreamProfile> find_profile(
    const std::shared_ptr<ob::StreamProfileList> &list, uint32_t w, uint32_t h, OBFormat fmt,
    uint32_t fps)
{
    for (uint32_t i = 0; i < list->getCount(); i++)
    {
        auto vsp = list->getProfile(i)->as<ob::VideoStreamProfile>();
        if ((w == 0 || vsp->getWidth() == w) && (h == 0 || vsp->getHeight() == h) &&
            (fmt == OB_FORMAT_ANY || vsp->getFormat() == fmt) && (fps == 0 || vsp->getFps() == fps))
        {
            return vsp;
        }
    }
    return nullptr;
}
} // namespace

orbbec_device::orbbec_device(uint32_t width, uint32_t height, uint8_t fps)
    : device(width, height, fps, BGR8, Z16)
{
    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_WARN);
}

void orbbec_device::print_stream_profiles()
{
    try
    {
        ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_WARN);
        ob::Context ctx;
        auto devices = ctx.queryDeviceList();
        if (devices->getCount() == 0)
        {
            std::cout << RED << "No Orbbec device found." << RESET << std::endl;
            return;
        }
        auto dev = devices->getDevice(0);
        auto info = dev->getDeviceInfo();
        std::cout << "Device: " << info->getName() << "  SN: " << info->getSerialNumber()
                  << "  FW: " << info->getFirmwareVersion()
                  << "  Connection: " << info->getConnectionType() << std::endl;

        ob::Pipeline pipe(dev);
        for (auto sensor : {OB_SENSOR_COLOR, OB_SENSOR_DEPTH})
        {
            std::cout << (sensor == OB_SENSOR_COLOR ? "Color" : "Depth")
                      << " profiles:" << std::endl;
            auto list = pipe.getStreamProfileList(sensor);
            for (uint32_t i = 0; i < list->getCount(); i++)
            {
                auto vsp = list->getProfile(i)->as<ob::VideoStreamProfile>();
                std::cout << "  " << vsp->getWidth() << "x" << vsp->getHeight() << " @"
                          << vsp->getFps() << " " << format_name(vsp->getFormat()) << std::endl;
            }
        }
    }
    catch (ob::Error &e)
    {
        std::cout << RED << "Orbbec error: " << e.what() << RESET << std::endl;
    }
}

bool orbbec_device::find_device()
{
    try
    {
        m_context = std::make_shared<ob::Context>();
        auto devices = m_context->queryDeviceList();
        if (devices->getCount() == 0)
        {
            return false;
        }
        m_device = devices->getDevice(0);
        auto info = m_device->getDeviceInfo();
        m_device_name = std::string(info->getName()) + " (SN " + info->getSerialNumber() + ", " +
                        info->getConnectionType() + ")";

        if (std::string(info->getConnectionType()).find("USB3") == std::string::npos)
        {
            std::cout << YELLOW << "Warning: camera is not connected via USB3 ("
                      << info->getConnectionType()
                      << "). Bandwidth and available profiles are limited." << RESET << std::endl;
        }
        return true;
    }
    catch (ob::Error &e)
    {
        std::cout << RED << "Orbbec error: " << e.what() << RESET << std::endl;
        return false;
    }
}

bool orbbec_device::select_profiles()
{
    auto color_list = m_pipeline->getStreamProfileList(OB_SENSOR_COLOR);
    auto depth_list = m_pipeline->getStreamProfileList(OB_SENSOR_DEPTH);

    // Color: prefer formats that need the least conversion.
    for (auto fmt : {OB_FORMAT_BGR, OB_FORMAT_RGB, OB_FORMAT_MJPG, OB_FORMAT_YUYV})
    {
        m_color_profile = find_profile(color_list, m_width, m_height, fmt, m_fps);
        if (m_color_profile)
            break;
    }
    if (!m_color_profile)
    {
        std::cout << RED << "No color profile " << m_width << "x" << m_height << "@" << (int)m_fps
                  << " available." << RESET << std::endl;
        return false;
    }

    // Depth: hardware D2C first.
    auto hw_list = m_pipeline->getD2CDepthProfileList(m_color_profile, ALIGN_D2C_HW_MODE);
    if (hw_list && hw_list->getCount() > 0)
    {
        for (auto fmt : depth_formats)
        {
            m_depth_profile = find_profile(hw_list, 0, 0, fmt, m_fps);
            if (m_depth_profile)
                break;
        }
    }
    m_hw_d2c = (m_depth_profile != nullptr);

    if (!m_hw_d2c)
    {
        // Software D2C: a depth profile with matching fps, 640x400 preferred.
        for (auto fmt : depth_formats)
        {
            m_depth_profile = find_profile(depth_list, 640, 400, fmt, m_fps);
            if (!m_depth_profile)
                m_depth_profile = find_profile(depth_list, 0, 0, fmt, m_fps);
            if (m_depth_profile)
                break;
        }
        if (!m_depth_profile)
        {
            std::cout << RED << "No depth profile @" << (int)m_fps << "fps available." << RESET
                      << std::endl;
            return false;
        }
        m_sw_align = std::make_shared<ob::Align>(OB_STREAM_COLOR);
    }

    std::cout << D_GREEN << "Color: " << m_color_profile->getWidth() << "x"
              << m_color_profile->getHeight() << "@" << m_color_profile->getFps() << " "
              << format_name(m_color_profile->getFormat())
              << " | Depth: " << m_depth_profile->getWidth() << "x" << m_depth_profile->getHeight()
              << "@" << m_depth_profile->getFps() << " "
              << format_name(m_depth_profile->getFormat())
              << " | D2C: " << (m_hw_d2c ? "hardware" : "software") << RESET << std::endl;
    return true;
}

bool orbbec_device::start_stream()
{
    if (m_pipeline_active)
        return true;

    try
    {
        m_pipeline = std::make_shared<ob::Pipeline>(m_device);
        if (!select_profiles())
            return false;

        auto config = std::make_shared<ob::Config>();
        config->enableStream(m_color_profile);
        config->enableStream(m_depth_profile);
        config->setAlignMode(m_hw_d2c ? ALIGN_D2C_HW_MODE : ALIGN_DISABLE);
        config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
        m_pipeline->enableFrameSync();
        m_pipeline->start(config);
        m_pipeline_active = true;

        // Warm up: wait for the first complete frameset (also gives us the depth value scale).
        for (int i = 0; i < 100; i++)
        {
            if (grab(200))
            {
                auto depth_vf = m_depth_frame->as<ob::VideoFrame>();
                if (depth_vf->getWidth() != m_width || depth_vf->getHeight() != m_height)
                {
                    std::cout << RED << "Aligned depth has size " << depth_vf->getWidth() << "x"
                              << depth_vf->getHeight() << ", expected " << m_width << "x"
                              << m_height << RESET << std::endl;
                    return false;
                }
                // The Gemini 2 L's default depth format is RLE [S: OrbbecSDKConfig.xml];
                // the GPU pipeline reads 16-bit depth only.
                const OBFormat depth_format = depth_vf->getFormat();
                if (depth_format != OB_FORMAT_Y16 && depth_format != OB_FORMAT_Z16)
                {
                    std::cout << RED << "Depth arrives as " << format_name(depth_format)
                              << ", the GPU pipeline needs Y16/Z16." << RESET << std::endl;
                    return false;
                }
                m_depth_value_scale = m_depth_frame->as<ob::DepthFrame>()->getValueScale();
                return true;
            }
        }
        std::cout << RED << "Timeout while waiting for first frames." << RESET << std::endl;
        return false;
    }
    catch (ob::Error &e)
    {
        std::cout << RED << "Orbbec error: " << e.what() << RESET << std::endl;
        return false;
    }
}

void orbbec_device::decode_color(const std::shared_ptr<ob::VideoFrame> &color)
{
    const int w = color->getWidth();
    const int h = color->getHeight();
    void *data = color->getData();

    switch (color->getFormat())
    {
    case OB_FORMAT_BGR:
        cv::Mat(h, w, CV_8UC3, data).copyTo(m_color_buffer);
        break;
    case OB_FORMAT_RGB:
        cv::cvtColor(cv::Mat(h, w, CV_8UC3, data), m_color_buffer, cv::COLOR_RGB2BGR);
        break;
    case OB_FORMAT_MJPG:
        cv::imdecode(cv::Mat(1, color->getDataSize(), CV_8UC1, data), cv::IMREAD_COLOR,
                     &m_color_buffer);
        break;
    case OB_FORMAT_YUYV:
        cv::cvtColor(cv::Mat(h, w, CV_8UC2, data), m_color_buffer, cv::COLOR_YUV2BGR_YUYV);
        break;
    default:
        std::cout << RED << "Unsupported color format " << format_name(color->getFormat()) << RESET
                  << std::endl;
        m_color_buffer = cv::Mat::zeros(h, w, CV_8UC3);
        break;
    }
}

bool orbbec_device::grab(uint32_t timeout_ms)
{
    auto fs = m_pipeline->waitForFrameset(timeout_ms);
    if (!fs)
        return false;

    if (m_sw_align)
    {
        auto aligned = m_sw_align->process(fs);
        if (!aligned)
            return false;
        fs = aligned->as<ob::FrameSet>();
    }

    auto color = fs->getColorFrame();
    auto depth = fs->getDepthFrame();
    if (!color || !depth)
        return false;

    decode_color(color);
    m_color_frame = color;
    m_depth_frame = depth;
    m_color_raw_data = m_color_buffer.data;
    m_depth_raw_data = depth->getData();
    return true;
}

void orbbec_device::receive_data_from_device()
{
    for (int i = 0; i < 50; i++)
    {
        if (grab(100))
            return;
    }
    std::cout << RED << "No frames received from Orbbec camera for 5s!" << RESET << std::endl;
}

float orbbec_device::read_depth_scale()
{
    // Orbbec depth values are in units of `valueScale` millimeters; gca wants meters per unit.
    return m_depth_value_scale * 0.001f;
}

gca::intrinsics orbbec_device::read_color_intrinsics()
{
    auto in = m_color_profile->getIntrinsic();
    return {in.fx, in.fy, in.cx, in.cy};
}

gca::intrinsics orbbec_device::read_depth_intrinsics()
{
    // Depth is aligned to color -> same intrinsics.
    return read_color_intrinsics();
}

gca::extrinsics orbbec_device::read_color_to_depth_extrinsics()
{
    gca::extrinsics e{};
    e.rotation[0] = e.rotation[4] = e.rotation[8] = 1.0f;
    return e;
}

gca::extrinsics orbbec_device::read_depth_to_color_extrinsics()
{
    return read_color_to_depth_extrinsics();
}

cv::Mat orbbec_device::set_color_to_cv_mat()
{
    return m_color_buffer;
}

cv::Mat orbbec_device::set_depth_to_cv_mat()
{
    return cv::Mat(cv::Size(m_width, m_height), CV_16UC1, (void *)m_depth_raw_data,
                   cv::Mat::AUTO_STEP);
}

void orbbec_device::stop_stream()
{
    if (m_pipeline_active)
    {
        try
        {
            m_pipeline->stop();
        }
        catch (ob::Error &e)
        {
            std::cout << RED << "Orbbec error: " << e.what() << RESET << std::endl;
        }
        m_pipeline_active = false;
    }
}

orbbec_device::~orbbec_device()
{
    stop_stream();
}
} // namespace gca
