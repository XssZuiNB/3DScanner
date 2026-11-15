// SPDX-License-Identifier: 0BSD
/// @file orbbec_imu.cpp
/// The Orbbec camera's IMU without the camera: the choice of a stream, the reason an opening goes
/// on without the IMU, and a sensor's queue of readings with its walk that gives a frame its
/// samples (see orbbec_imu.hpp).

#include "camera/orbbec_imu.hpp"

#include <cstdio>
#include <string>

namespace gca
{
namespace
{
/// Writes the IMU's rate for a reason.
/// @return orbbec_imu_rate_hz as "200 Hz".
std::string rate_text()
{
    char text[32];
    std::snprintf(text, sizeof(text), "%g Hz", orbbec_imu_rate_hz);
    return text;
}
} // namespace

std::optional<std::size_t> choose_imu_profile(std::span<const double> rates_hz)
{
    for (std::size_t i = 0; i < rates_hz.size(); i++)
    {
        if (rates_hz[i] == orbbec_imu_rate_hz)
            return i;
    }
    return std::nullopt;
}

std::optional<std::string> imu_refusal(const orbbec_imu_report &report)
{
    if (report.error)
        return "OrbbecSDK failed while starting the IMU: " + report.error->message;
    if (report.accel_offered && report.gyro_offered)
        return std::nullopt;
    const std::string missing =
        !report.accel_offered && !report.gyro_offered
            ? "no accelerometer and no gyroscope stream"
            : (report.accel_offered ? "no gyroscope stream" : "no accelerometer stream");
    return "the camera offers " + missing + " at " + rate_text();
}

orbbec_imu_queue::orbbec_imu_queue(const char *sensor, std::size_t capacity)
    : m_sensor(sensor)
    , m_ring(capacity)
{
}

void orbbec_imu_queue::start() noexcept
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    m_running = true;
    m_head = 0;
    m_count = 0;
    m_dropped = 0;
    m_unreadable = 0;
    m_last_given_us.reset();
}

void orbbec_imu_queue::stop() noexcept
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    m_running = false;
    m_head = 0;
    m_count = 0;
    m_dropped = 0;
    m_unreadable = 0;
    m_last_given_us.reset();
}

void orbbec_imu_queue::add(const orbbec_imu_reading &reading) noexcept
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_running)
        return;
    if (m_count == m_ring.size())
    {
        // The oldest reading goes, so that the newest, nearest the frames to come, stay.
        m_head = (m_head + 1) % m_ring.size();
        m_count--;
        m_dropped++;
    }
    m_ring[(m_head + m_count) % m_ring.size()] = reading;
    m_count++;
}

void orbbec_imu_queue::add_unreadable() noexcept
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    if (m_running)
        m_unreadable++;
}

void orbbec_imu_queue::deliver(const orbbec_frame_gate &gate, std::uint64_t frame_timestamp_us,
                               std::vector<imu_sample> &into, orbbec_skips &skips) noexcept
{
    // The readings lost under the lock are counted there; the reason of the last one is built
    // after it, from what is kept of it here.
    enum class loss
    {
        none,
        full,
        unreadable,
        not_placed,
        before_zero,
        not_after,
    };
    loss last = loss::none;
    std::uint64_t lost = 0;
    orbbec_imu_reading lost_reading{};
    std::int64_t lost_us = 0;
    std::uint64_t given_before_us = 0;
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        if (m_dropped > 0)
        {
            lost += m_dropped;
            last = loss::full;
            m_dropped = 0;
        }
        if (m_unreadable > 0)
        {
            lost += m_unreadable;
            last = loss::unreadable;
            m_unreadable = 0;
        }

        // Each reading held is given, skipped or kept for a later frame; those kept move up to
        // the ring's front in their order, which overwrites only readings already walked.
        const std::int64_t frame_us = static_cast<std::int64_t>(frame_timestamp_us);
        std::size_t kept = 0;
        for (std::size_t i = 0; i < m_count; i++)
        {
            const orbbec_imu_reading &reading = m_ring[(m_head + i) % m_ring.size()];
            std::int64_t t_us = 0;
            const auto lose = [&](loss why) {
                lost++;
                last = why;
                lost_reading = reading;
                lost_us = t_us;
            };
            if (gate.place(reading.camera_time_us, reading.host_time_us, t_us) !=
                orbbec_placement::placed)
            {
                lose(loss::not_placed);
                continue;
            }
            if (t_us > frame_us)
            {
                m_ring[(m_head + kept) % m_ring.size()] = reading;
                kept++;
                continue;
            }
            if (t_us < 0)
            {
                lose(loss::before_zero);
                continue;
            }
            const std::uint64_t sample_us = static_cast<std::uint64_t>(t_us);
            if (m_last_given_us && sample_us <= *m_last_given_us)
            {
                lose(loss::not_after);
                given_before_us = *m_last_given_us;
                continue;
            }
            into.push_back({sample_us, reading.x, reading.y, reading.z});
            m_last_given_us = sample_us;
        }
        m_count = kept;
    }
    if (last == loss::none)
        return;
    skips.count += lost;
    const std::string the_sensors = std::string("the ") + m_sensor + "'s";
    switch (last)
    {
    case loss::none:
        break;
    case loss::full:
        skips.last_reason = the_sensors + " queue of " + std::to_string(m_ring.size()) +
                            " readings was full, so its oldest reading went";
        break;
    case loss::unreadable:
        skips.last_reason =
            "OrbbecSDK's " + std::string(m_sensor) + " reading could not be read in its callback";
        break;
    case loss::not_placed:
        skips.last_reason =
            the_sensors + " reading at " + std::to_string(lost_reading.camera_time_us) +
            " µs on the camera's clock cannot be placed on the frames' clock: " +
            gate.placement_refusal(lost_reading.camera_time_us, lost_reading.host_time_us);
        break;
    case loss::before_zero:
        skips.last_reason = the_sensors + " sample at " + std::to_string(lost_us) +
                            " µs lies before the frames' clock's 0";
        break;
    case loss::not_after:
        skips.last_reason = the_sensors + " sample at " + std::to_string(lost_us) +
                            " µs is not after the last one given, at " +
                            std::to_string(given_before_us) + " µs";
        break;
    }
}
} // namespace gca
