// SPDX-License-Identifier: 0BSD
/// @file orbbec_device.cpp
/// The Orbbec camera as a frame source: profile selection, depth-to-color alignment, color
/// decoding to RGB8, the IMU's sensors feeding their queues, and OrbbecSDK's report turned into
/// plain data for camera/orbbec_rules.hpp and camera/orbbec_imu.hpp (see orbbec_device.hpp).

#include "orbbec_device.hpp"

#include "types/console_color.hpp"

#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <libobsensor/hpp/TypeHelper.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace gca
{
static_assert(static_cast<int>(OB_EXCEPTION_TYPE_IO) == 7 &&
                  static_cast<int>(OB_EXCEPTION_TYPE_RESOURCE_BUSY) == 14 &&
                  static_cast<int>(OB_EXCEPTION_TYPE_LICENSE_VERIFY_FAILED) == 15,
              "exception_kind() was checked against OrbbecSDK v2.10.6's OBExceptionType, whose "
              "last type is 15 [S: ObTypes.h]; check it again for this OrbbecSDK");

namespace
{
/// Names a stream format for messages.
/// @param f [in] the format.
/// @return OrbbecSDK's name for it.
/// @note Allocates the name.
std::string format_name(OBFormat f)
{
    return ob::TypeHelper::convertOBFormatTypeToString(f);
}

/// Maps OrbbecSDK's stream format to the formats the choice of streams tells apart, by name.
/// @param format [in] the format.
/// @return its orbbec_format, other for a format the choice does not name.
orbbec_format format_kind(OBFormat format) noexcept
{
    switch (format)
    {
    case OB_FORMAT_BGR:
        return orbbec_format::bgr;
    case OB_FORMAT_RGB:
        return orbbec_format::rgb;
    case OB_FORMAT_MJPG:
        return orbbec_format::mjpg;
    case OB_FORMAT_YUYV:
        return orbbec_format::yuyv;
    case OB_FORMAT_Y16:
        return orbbec_format::y16;
    case OB_FORMAT_Z16:
        return orbbec_format::z16;
    case OB_FORMAT_Y14:
        return orbbec_format::y14;
    case OB_FORMAT_RLE:
        return orbbec_format::rle;
    default:
        return orbbec_format::other;
    }
}

/// Reads a list of OrbbecSDK's video stream profiles as plain data.
/// @param list [in] the list; nullptr for none.
/// @return its profiles in its order; empty for nullptr. OrbbecSDK's exceptions pass through.
/// @note Allocates the vector.
std::vector<orbbec_profile> plain_profiles(const std::shared_ptr<ob::StreamProfileList> &list)
{
    std::vector<orbbec_profile> profiles;
    if (!list)
        return profiles;
    for (std::uint32_t i = 0; i < list->getCount(); i++)
    {
        const auto vsp = list->getProfile(i)->as<ob::VideoStreamProfile>();
        profiles.push_back(
            {vsp->getWidth(), vsp->getHeight(), vsp->getFps(), format_kind(vsp->getFormat())});
    }
    return profiles;
}

/// Takes a profile out of a list.
/// @param list [in] the list.
/// @param position [in] its position, < the list's count.
/// @return the profile. OrbbecSDK's exceptions pass through.
std::shared_ptr<ob::VideoStreamProfile> profile_at(
    const std::shared_ptr<ob::StreamProfileList> &list, std::size_t position)
{
    return list->getProfile(static_cast<std::uint32_t>(position))->as<ob::VideoStreamProfile>();
}

/// Maps OrbbecSDK's exception type to the kinds the rules tell apart, by name.
/// @param type [in] the exception's type, ob::Error::getExceptionType().
/// @return io, resource_busy, or other for every other type.
orbbec_exception exception_kind(OBExceptionType type) noexcept
{
    switch (type)
    {
    case OB_EXCEPTION_TYPE_IO:
        return orbbec_exception::io;
    case OB_EXCEPTION_TYPE_RESOURCE_BUSY:
        return orbbec_exception::resource_busy;
    default:
        return orbbec_exception::other;
    }
}

/// Reads the host's steady clock.
/// @return the time since the clock's epoch, in µs.
std::uint64_t host_time_us() noexcept
{
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}

/// Builds a wait that failed.
/// @param message [in] what failed.
/// @return the wait, status failure.
orbbec_wait failed_wait(std::string message)
{
    orbbec_wait wait;
    wait.status = orbbec_wait_status::failure;
    wait.message = std::move(message);
    return wait;
}

/// Picks an IMU sensor's stream at orbbec_imu_rate_hz with choose_imu_profile().
/// @param list [in] the sensor's profiles.
/// @param sensor [in] OB_SENSOR_ACCEL or OB_SENSOR_GYRO, whose profile class the list holds.
/// @return the chosen profile; nullptr when none has the rate. OrbbecSDK's exceptions pass
///         through.
/// @note Allocates the rates and OrbbecSDK's profile objects.
std::shared_ptr<ob::StreamProfile> imu_profile(const std::shared_ptr<ob::StreamProfileList> &list,
                                               OBSensorType sensor)
{
    std::vector<double> rates_hz;
    for (std::uint32_t i = 0; i < list->getCount(); i++)
    {
        const std::shared_ptr<ob::StreamProfile> profile = list->getProfile(i);
        const OBIMUSampleRate rate = sensor == OB_SENSOR_ACCEL
                                         ? profile->as<ob::AccelStreamProfile>()->getSampleRate()
                                         : profile->as<ob::GyroStreamProfile>()->getSampleRate();
        rates_hz.push_back(ob::TypeHelper::convertOBIMUSampleRateTypeToValue(rate));
    }
    const std::optional<std::size_t> chosen = choose_imu_profile(rates_hz);
    if (!chosen)
        return nullptr;
    return list->getProfile(static_cast<std::uint32_t>(*chosen));
}

/// Puts an IMU reading OrbbecSDK hands its sensor's callback into the sensor's queue, with the
/// host's time it came at; counts it as unreadable when it is of another kind or OrbbecSDK
/// throws while it is read. Runs on OrbbecSDK's thread; no exception leaves it, as none may leave
/// a callback into OrbbecSDK's C code.
/// @param frame [in] the reading, an AccelFrame or a GyroFrame.
/// @param kind [in] OB_FRAME_ACCEL or OB_FRAME_GYRO, the sensor's.
/// @param queue [in,out] the sensor's queue; gains the reading.
/// @note Allocates OrbbecSDK's wrapper object of the reading (Frame.hpp, as()) and an exception's
///       message. Takes the queue's lock for the copy, after every OrbbecSDK call.
void queue_imu_reading(const std::shared_ptr<ob::Frame> &frame, OBFrameType kind,
                       orbbec_imu_queue &queue) noexcept
{
    const std::uint64_t received_us = host_time_us();
    try
    {
        if (!frame || frame->getType() != kind)
        {
            queue.add_unreadable();
            return;
        }
        const OBFloat3D value = kind == OB_FRAME_ACCEL ? frame->as<ob::AccelFrame>()->getValue()
                                                       : frame->as<ob::GyroFrame>()->getValue();
        queue.add({frame->getTimeStampUs(), received_us, value.x, value.y, value.z});
    }
    catch (const std::exception &)
    {
        // ob::Error and the std::runtime_error of as(), which derive from std::exception.
        queue.add_unreadable();
    }
}
} // namespace

orbbec_device::orbbec_device(std::uint32_t fps)
    : m_fps(fps)
{
}

orbbec_device::~orbbec_device()
{
    close();
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
    catch (const std::exception &e)
    {
        // ob::Error, and the std::runtime_error of OrbbecSDK's as() (Frame.hpp).
        std::cout << RED << "Orbbec error: " << e.what() << RESET << std::endl;
    }
}

open_result orbbec_device::open(const frame_size &size) noexcept
{
    if (m_open)
        return {false, "the camera is open already"};
    orbbec_open_report report;
    intrinsics color_intrinsics{};
    m_opening = {};
    try
    {
        ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_WARN);
        m_context = std::make_shared<ob::Context>();
        const auto devices = m_context->queryDeviceList();
        report.camera_count = devices->getCount();
        if (report.camera_count > 0)
        {
            m_device = devices->getDevice(0);
            const auto info = m_device->getDeviceInfo();
            m_camera_name = info->getName();
            m_firmware_version = info->getFirmwareVersion();
            m_opening.serial_number = info->getSerialNumber();
            m_opening.connection_type = info->getConnectionType();
            report.connection_type = m_opening.connection_type;
            report.firmware_version = m_firmware_version;
        }
        // The firmware's rule decides here whether a stream may start at all, so that a camera
        // whose firmware it refuses never streams; opening_refusal() below gives the reason.
        if (report.camera_count > 0 && !firmware_refusal(report.firmware_version))
        {
            m_pipeline = std::make_shared<ob::Pipeline>(m_device);
            select_profiles(size, report);
            if (report.color_stream_offered && report.depth_stream_offered)
            {
                const OBCameraIntrinsic in = m_color_profile->getIntrinsic();
                color_intrinsics = {in.fx, in.fy, in.cx, in.cy};
                start_pipeline();
                start_imu();
            }
        }
    }
    catch (const ob::Error &e)
    {
        report.error = orbbec_error{exception_kind(e.getExceptionType()), e.what()};
    }
    catch (const std::exception &e)
    {
        // OrbbecSDK's C++ wrapper throws std::runtime_error when a frame or profile is not of
        // the type asked for (Frame.hpp, as()); no exception type of OrbbecSDK's own.
        report.error = orbbec_error{orbbec_exception::other, e.what()};
    }
    if (const std::optional<std::string> refusal = opening_refusal(report, size, m_fps))
    {
        release();
        return {false, *refusal};
    }
    m_gate.start(size, color_intrinsics);
    m_open = true;
    return {true, {}};
}

void orbbec_device::select_profiles(const frame_size &size, orbbec_open_report &report)
{
    m_color_profile = nullptr;
    m_depth_profile = nullptr;
    m_sw_align = nullptr;
    const auto color_list = m_pipeline->getStreamProfileList(OB_SENSOR_COLOR);
    const std::optional<std::size_t> color =
        choose_color_profile(plain_profiles(color_list), size, m_fps);
    report.color_stream_offered = color.has_value();
    if (!color)
        return;
    m_color_profile = profile_at(color_list, *color);

    const auto hw_list = m_pipeline->getD2CDepthProfileList(m_color_profile, ALIGN_D2C_HW_MODE);
    const auto depth_list = m_pipeline->getStreamProfileList(OB_SENSOR_DEPTH);
    const std::optional<orbbec_depth_choice> depth =
        choose_depth_profile(plain_profiles(hw_list), plain_profiles(depth_list), size, m_fps);
    report.depth_stream_offered = depth.has_value();
    if (!depth)
        return;
    m_hw_d2c = depth->hardware;
    m_depth_profile = profile_at(m_hw_d2c ? hw_list : depth_list, depth->position);
    if (!m_hw_d2c)
        m_sw_align = std::make_shared<ob::Align>(OB_STREAM_COLOR);

    m_opening.streams = "Color: " + std::to_string(m_color_profile->getWidth()) + "x" +
                        std::to_string(m_color_profile->getHeight()) + "@" +
                        std::to_string(m_color_profile->getFps()) + " " +
                        format_name(m_color_profile->getFormat()) +
                        " | Depth: " + std::to_string(m_depth_profile->getWidth()) + "x" +
                        std::to_string(m_depth_profile->getHeight()) + "@" +
                        std::to_string(m_depth_profile->getFps()) + " " +
                        format_name(m_depth_profile->getFormat()) +
                        " | D2C: " + (m_hw_d2c ? "hardware" : "software");
}

void orbbec_device::start_pipeline()
{
    auto config = std::make_shared<ob::Config>();
    config->enableStream(m_color_profile);
    config->enableStream(m_depth_profile);
    config->setAlignMode(m_hw_d2c ? ALIGN_D2C_HW_MODE : ALIGN_DISABLE);
    config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
    m_pipeline->enableFrameSync();
    m_pipeline->start(config);
    m_pipeline_started = true;
}

next_result orbbec_device::next(frame &out) noexcept
{
    if (!m_open)
        return {next_status::failure, "the camera is not open"};
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(orbbec_frame_time_limit_ms);
    for (;;)
    {
        const auto left_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 deadline - std::chrono::steady_clock::now())
                                 .count();
        orbbec_wait wait;
        if (left_ms <= 0)
        {
            wait.status = orbbec_wait_status::timeout;
        }
        else
        {
            try
            {
                wait = wait_for_frameset(static_cast<std::uint32_t>(left_ms));
            }
            catch (const ob::Error &e)
            {
                wait = failed_wait(std::string("OrbbecSDK: ") + e.what());
            }
            catch (const std::exception &e)
            {
                // OpenCV's cv::Exception, and the std::runtime_error of OrbbecSDK's as()
                // (Frame.hpp).
                wait = failed_wait(std::string("decoding the frameset: ") + e.what());
            }
        }
        const std::optional<next_result> result = m_gate.take(wait, out);
        release_frames();
        if (result && result->status == next_status::frame)
        {
            // The frame just taken is the gate's last: the queues place their readings against it.
            m_accel_queue.deliver(m_gate, out.timestamp_us(), out.accel(), m_imu_skips);
            m_gyro_queue.deliver(m_gate, out.timestamp_us(), out.gyro(), m_imu_skips);
        }
        if (result)
            return *result;
    }
}

const orbbec_skips &orbbec_device::skips() const noexcept
{
    return m_gate.skips();
}

const orbbec_skips &orbbec_device::imu_skips() const noexcept
{
    return m_imu_skips;
}

const orbbec_opening &orbbec_device::opening() const noexcept
{
    return m_opening;
}

orbbec_wait orbbec_device::wait_for_frameset(std::uint32_t time_limit_ms)
{
    std::shared_ptr<ob::FrameSet> frameset = m_pipeline->waitForFrameset(time_limit_ms);
    orbbec_wait wait;
    if (!frameset)
    {
        // OrbbecSDK's queue gave no frameset within the time limit [S: Pipeline.cpp,
        // waitForFrame]. An error of OrbbecSDK during the wait comes back the same way: the
        // installed Pipeline.hpp returns nullptr before it hands the error to Error::handle()
        // [S: Pipeline.hpp, waitForFrameset, OrbbecSDK v2.10.6].
        wait.status = orbbec_wait_status::timeout;
        return wait;
    }
    const std::uint64_t received_us = host_time_us();
    if (m_sw_align)
    {
        auto aligned = m_sw_align->process(frameset);
        if (!aligned)
            return failed_wait("OrbbecSDK's software alignment gave no frameset");
        frameset = aligned->as<ob::FrameSet>();
    }
    m_color_frame = frameset->getColorFrame();
    m_depth_frame = frameset->getDepthFrame();
    if (!m_color_frame || !m_depth_frame)
        return failed_wait("a frameset without its color or its depth frame");

    orbbec_frameset &facts = wait.frameset;
    if (const std::optional<std::string> reason = decode_color(m_color_frame, facts))
        return failed_wait(*reason);
    facts.depth_size = {m_depth_frame->getWidth(), m_depth_frame->getHeight()};
    const OBFormat depth_format = m_depth_frame->getFormat();
    // The GPU pipeline reads 16-bit depth only; the Gemini 2 L's default depth format is RLE
    // [S: OrbbecSDKConfig.xml], and Y14 arrives as 16 bit [M 2026-11-13, AGENTS.md §6].
    facts.depth_16_bit = depth_format == OB_FORMAT_Y16 || depth_format == OB_FORMAT_Z16;
    if (facts.depth_16_bit)
    {
        // 16-bit depth values in OrbbecSDK's frame buffer.
        facts.depth = {reinterpret_cast<const std::uint16_t *>(m_depth_frame->getData()),
                       m_depth_frame->getDataSize() / sizeof(std::uint16_t)};
    }
    else
    {
        m_depth_format = format_name(depth_format);
        facts.depth_format = m_depth_format;
    }
    facts.depth_scale_mm = m_depth_frame->getValueScale();
    facts.index = m_color_frame->getIndex();
    facts.camera_time_us = m_color_frame->getTimeStampUs();
    facts.host_time_us = received_us;
    wait.status = orbbec_wait_status::frameset;
    return wait;
}

std::optional<std::string> orbbec_device::decode_color(const std::shared_ptr<ob::ColorFrame> &color,
                                                       orbbec_frameset &frameset)
{
    const std::uint32_t w = color->getWidth();
    const std::uint32_t h = color->getHeight();
    const OBFormat format = color->getFormat();
    std::uint8_t *data = color->getData();
    const std::size_t bytes = color->getDataSize();
    frameset.color_size = {w, h};

    // Sizes m_color_rgb for a picture and hands the gate the picture there; the header it returns
    // has the right size and type, so cv::cvtColor() writes into it and allocates nothing
    // [S: OpenCV, Mat::create()].
    const auto rgb_buffer = [&](const frame_size &size) {
        m_color_rgb.resize(3 * std::size_t{size.width} * size.height);
        frameset.color_size = size;
        frameset.color_rgb = m_color_rgb;
        return cv::Mat(static_cast<int>(size.height), static_cast<int>(size.width), CV_8UC3,
                       m_color_rgb.data());
    };

    switch (format)
    {
    case OB_FORMAT_RGB:
        frameset.color_rgb = {data, bytes};
        return std::nullopt;
    case OB_FORMAT_BGR:
    case OB_FORMAT_YUYV: {
        const bool bgr = format == OB_FORMAT_BGR;
        const std::size_t needed = (bgr ? 3 : 2) * std::size_t{w} * h;
        if (bytes != needed)
            return "the " + format_name(format) + " color frame holds " + std::to_string(bytes) +
                   " bytes, " + std::to_string(w) + "x" + std::to_string(h) + " needs " +
                   std::to_string(needed);
        const cv::Mat raw(static_cast<int>(h), static_cast<int>(w), bgr ? CV_8UC3 : CV_8UC2, data);
        cv::Mat rgb = rgb_buffer({w, h});
        cv::cvtColor(raw, rgb, bgr ? cv::COLOR_BGR2RGB : cv::COLOR_YUV2RGB_YUYV);
        return std::nullopt;
    }
    case OB_FORMAT_MJPG: {
        cv::imdecode(cv::Mat(1, static_cast<int>(bytes), CV_8UC1, data), cv::IMREAD_COLOR,
                     &m_decoded_bgr);
        if (m_decoded_bgr.empty() || m_decoded_bgr.type() != CV_8UC3)
            return "an MJPG color frame of " + std::to_string(bytes) + " bytes does not decode";
        cv::Mat rgb = rgb_buffer({static_cast<std::uint32_t>(m_decoded_bgr.cols),
                                  static_cast<std::uint32_t>(m_decoded_bgr.rows)});
        cv::cvtColor(m_decoded_bgr, rgb, cv::COLOR_BGR2RGB);
        return std::nullopt;
    }
    default:
        return "color arrives as " + format_name(format) +
               ", which the camera source does not decode";
    }
}

void orbbec_device::start_imu() noexcept
{
    m_imu_skips = {};
    orbbec_imu_report report;
    std::string streams;
    try
    {
        const std::shared_ptr<ob::SensorList> sensors = m_device->getSensorList();
        bool has_accel = false;
        bool has_gyro = false;
        for (std::uint32_t i = 0; i < sensors->getCount(); i++)
        {
            has_accel = has_accel || sensors->getSensorType(i) == OB_SENSOR_ACCEL;
            has_gyro = has_gyro || sensors->getSensorType(i) == OB_SENSOR_GYRO;
        }
        std::shared_ptr<ob::StreamProfile> accel_profile;
        std::shared_ptr<ob::StreamProfile> gyro_profile;
        if (has_accel)
        {
            m_accel_sensor = m_device->getSensor(OB_SENSOR_ACCEL);
            accel_profile = imu_profile(m_accel_sensor->getStreamProfileList(), OB_SENSOR_ACCEL);
            report.accel_offered = accel_profile != nullptr;
        }
        if (has_gyro)
        {
            m_gyro_sensor = m_device->getSensor(OB_SENSOR_GYRO);
            gyro_profile = imu_profile(m_gyro_sensor->getStreamProfileList(), OB_SENSOR_GYRO);
            report.gyro_offered = gyro_profile != nullptr;
        }
        if (report.accel_offered && report.gyro_offered)
        {
            m_accel_queue.start();
            m_gyro_queue.start();
            // Each callback runs on OrbbecSDK's thread and touches only its own queue.
            orbbec_imu_queue *const accel_queue = &m_accel_queue;
            orbbec_imu_queue *const gyro_queue = &m_gyro_queue;
            m_accel_sensor->start(accel_profile, [accel_queue](std::shared_ptr<ob::Frame> frame) {
                queue_imu_reading(frame, OB_FRAME_ACCEL, *accel_queue);
            });
            m_accel_started = true;
            m_gyro_sensor->start(gyro_profile, [gyro_queue](std::shared_ptr<ob::Frame> frame) {
                queue_imu_reading(frame, OB_FRAME_GYRO, *gyro_queue);
            });
            m_gyro_started = true;
            const auto accel = accel_profile->as<ob::AccelStreamProfile>();
            const auto gyro = gyro_profile->as<ob::GyroStreamProfile>();
            streams =
                "Accel: " +
                ob::TypeHelper::convertOBIMUSampleRateTypeToString(accel->getSampleRate()) + " " +
                ob::TypeHelper::convertOBAccelFullScaleRangeTypeToString(
                    accel->getFullScaleRange()) +
                " | Gyro: " +
                ob::TypeHelper::convertOBIMUSampleRateTypeToString(gyro->getSampleRate()) + " " +
                ob::TypeHelper::convertOBGyroFullScaleRangeTypeToString(gyro->getFullScaleRange());
        }
    }
    catch (const ob::Error &e)
    {
        report.error = orbbec_error{exception_kind(e.getExceptionType()), e.what()};
    }
    catch (const std::exception &e)
    {
        // The std::runtime_error of OrbbecSDK's as() (Frame.hpp), and std::bad_alloc.
        report.error = orbbec_error{orbbec_exception::other, e.what()};
    }
    if (const std::optional<std::string> refusal = imu_refusal(report))
    {
        stop_imu();
        m_opening.imu_refusal = *refusal;
        return;
    }
    m_opening.imu_streams = streams;
}

void orbbec_device::stop_imu() noexcept
{
    for (const auto &[sensor, started] :
         {std::pair{&m_accel_sensor, &m_accel_started}, std::pair{&m_gyro_sensor, &m_gyro_started}})
    {
        if (*started)
        {
            try
            {
                (*sensor)->stop();
            }
            catch (const ob::Error &)
            {
                // stop_imu() has no result: the sensor is released below either way.
            }
            *started = false;
        }
        *sensor = nullptr;
    }
    // After the sensors stopped: the next opening is given none of this opening's readings, and
    // one whose IMU does not start none at all.
    m_accel_queue.stop();
    m_gyro_queue.stop();
}

void orbbec_device::release_frames() noexcept
{
    m_color_frame = nullptr;
    m_depth_frame = nullptr;
}

void orbbec_device::close() noexcept
{
    release();
    m_open = false;
}

void orbbec_device::release() noexcept
{
    release_frames();
    stop_imu();
    if (m_pipeline_started)
    {
        try
        {
            m_pipeline->stop();
        }
        catch (const ob::Error &)
        {
            // close() has no result: the camera is released below either way.
        }
        m_pipeline_started = false;
    }
    m_sw_align = nullptr;
    m_color_profile = nullptr;
    m_depth_profile = nullptr;
    m_pipeline = nullptr;
    m_device = nullptr;
    m_context = nullptr;
}

source_description orbbec_device::describe() const noexcept
{
    return {
        m_camera_name.empty() ? std::string("Orbbec camera") : m_camera_name, m_firmware_version,
        std::to_string(ob::Version::getMajor()) + "." + std::to_string(ob::Version::getMinor()) +
            "." + std::to_string(ob::Version::getPatch())};
}
} // namespace gca
