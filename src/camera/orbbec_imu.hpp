// SPDX-License-Identifier: 0BSD
/// @file orbbec_imu.hpp
/// The Orbbec camera's IMU without the camera (docs/methods/imu-samples.md): the rate its
/// accelerometer and gyroscope are started at, the choice of their streams, the reason an opening
/// goes on without them, and the queue of a sensor's readings, which OrbbecSDK's thread fills and
/// the camera source's next() empties into the frame it has just taken, each reading placed on the
/// frames' clock by the frames' own rule (camera/orbbec_rules.hpp, orbbec_frame_gate::place()).
/// camera/orbbec_device.cpp feeds them with what OrbbecSDK reports; they build and link without
/// OrbbecSDK, so the offline tests drive them (tests/test_orbbec_imu.cpp).

#pragma once

#include "camera/orbbec_rules.hpp"
#include "types/imu_sample.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace gca
{
/// The rate the camera's accelerometer and gyroscope are started at, in Hz [?: initial]: 6 or 7
/// samples of each a frame at 30 fps, the rate the visual-inertial datasets EuRoC and TUM VI
/// record theirs at (docs/methods/imu-samples.md, "The IMU and its streams"). The camera offers
/// 100, 200, 500 Hz and 1 kHz [M 2026-11-13, docs/HARDWARE.md].
inline constexpr double orbbec_imu_rate_hz = 200.0;

/// The readings of one sensor its queue holds, 4 s at orbbec_imu_rate_hz [?: a choice: twice the
/// 2 s that OrbbecSDK's frameset queue of 10 holds at 5 fps (S: Pipeline.cpp, OrbbecSDK v2.10.6),
/// so that a reading still waits for the oldest frame OrbbecSDK hands out].
inline constexpr std::size_t orbbec_imu_queue_readings = 800;

/// Chooses an IMU sensor's stream: the first profile of orbbec_imu_rate_hz.
/// @param rates_hz [in] the rates of the profiles OrbbecSDK offers for the sensor, in Hz, in its
///        order.
/// @return the chosen profile's position; nothing when none has the rate.
std::optional<std::size_t> choose_imu_profile(std::span<const double> rates_hz);

/// What starting the camera's IMU found. A step not reached keeps its default.
struct orbbec_imu_report
{
    bool accel_offered = false;        ///< choose_imu_profile() found the accelerometer's stream
    bool gyro_offered = false;         ///< choose_imu_profile() found the gyroscope's stream
    std::optional<orbbec_error> error; ///< the exception that ended the IMU's start, if one did
};

/// Decides whether an opening of the camera goes on without its IMU, and why. An opening is not
/// refused for its IMU: its frames then come without samples (docs/methods/imu-samples.md, "The
/// IMU and its streams").
/// @param report [in] what starting the IMU found.
/// @return nothing when both streams were offered and started without an exception. Otherwise the
///         reason, in this order: an exception of OrbbecSDK, with its message; the streams the
///         camera does not offer at orbbec_imu_rate_hz, the accelerometer's, the gyroscope's or
///         both.
/// @note Allocates for the reason.
std::optional<std::string> imu_refusal(const orbbec_imu_report &report);

/// One reading of an IMU sensor as OrbbecSDK delivered it, as plain data.
struct orbbec_imu_reading
{
    std::uint64_t camera_time_us; ///< ob::Frame::getTimeStampUs(), on the camera's clock, in µs
    std::uint64_t host_time_us;   ///< when the callback got it, on the host's steady clock, in µs
    float x;                      ///< OrbbecSDK's value along the IMU's x axis, as given
    float y;                      ///< along its y axis
    float z;                      ///< along its z axis
};

/// The readings of one IMU sensor of an opening, from OrbbecSDK's thread to the frames
/// (docs/methods/imu-samples.md, "The queue" and "Which frame carries a sample"). A ring of a fixed
/// size, allocated when the queue is built, holds the readings in the order they came; when it is
/// full the oldest goes. Once the camera source has taken a frame, deliver() places each reading
/// held on the frames' clock against that frame and gives the frame those at or before its
/// timestamp that follow the sensor's last sample given; it leaves those after it for a later
/// frame and skips, counted with the reason, those it cannot place, those not after the last
/// sample given and those before the clock's 0. The queue takes readings only between start() and
/// stop(), which both empty it, so that no opening is given another opening's readings, and an
/// opening whose IMU did not start is given none. One lock guards the ring, the counts and
/// whether the queue runs: add() and add_unreadable() may be called on another thread than
/// start(), stop() and deliver(), OrbbecSDK's.
class orbbec_imu_queue
{
public:
    /// Builds an empty queue of a sensor, stopped: it takes no reading until start().
    /// @param sensor [in] the sensor's name for the reasons, "accelerometer" or "gyroscope"; a
    ///        string literal, which outlives the queue.
    /// @param capacity [in] the readings the ring holds, >= 1; the camera's are
    ///        orbbec_imu_queue_readings.
    /// @note Allocates the ring. Throws std::bad_alloc when that fails.
    orbbec_imu_queue(const char *sensor, std::size_t capacity);

    /// A queue is not copied: it holds a lock that another thread uses.
    orbbec_imu_queue(const orbbec_imu_queue &) = delete;
    orbbec_imu_queue &operator=(const orbbec_imu_queue &) = delete;

    /// Starts an opening's IMU: empties the ring, forgets the counts and the last sample given,
    /// and takes readings from now on.
    /// @note Allocates nothing. Takes the lock.
    void start() noexcept;

    /// Stops the IMU of an opening: empties the ring, forgets the counts and the last sample
    /// given, and takes no reading until the next start(), so that deliver() gives nothing.
    /// @note Allocates nothing. Takes the lock.
    void stop() noexcept;

    /// Puts a reading into the ring while the queue runs; when the ring is full the oldest reading
    /// goes, counted. A stopped queue drops the reading uncounted.
    /// @param reading [in] the reading; copied.
    /// @note Allocates nothing. Takes the lock for the copy; waits for nothing else. Called on
    ///       OrbbecSDK's thread.
    void add(const orbbec_imu_reading &reading) noexcept;

    /// Counts a reading OrbbecSDK delivered and the callback could not read, while the queue
    /// runs.
    /// @note Allocates nothing. Takes the lock. Called on OrbbecSDK's thread.
    void add_unreadable() noexcept;

    /// Gives the frame the camera source has just taken the readings it carries, as samples on the
    /// frames' clock, in the order they came, and counts the readings lost since the last call.
    /// @param gate [in] the opening's gate, whose last taken frame is the frame's; its place()
    ///        places each reading.
    /// @param frame_timestamp_us [in] the frame's delivered timestamp, in µs: the last taken
    ///        frame's.
    /// @param into [in,out] the frame's list of the sensor's samples, empty after the frame's
    ///        prepare(); gains the samples.
    /// @param skips [in,out] gains each reading lost: dropped from a full ring, unreadable in the
    ///        callback, not placed on the frames' clock, not after the last sample given, or
    ///        before the clock's 0, with the last one's reason, naming the sensor.
    /// @pre The gate has taken a frame in this opening; otherwise every reading is skipped, as
    ///      place() refuses it.
    /// @note Allocates the one reason of a call that lost readings, after the lock, and into
    ///       when it grows beyond its capacity, which the frame reserved (camera/frame.hpp,
    ///       frame_imu_samples_reserved); a failed allocation ends the program (std::terminate):
    ///       the function is noexcept. Takes the lock for the walk over the ring, which calls no
    ///       OrbbecSDK and builds no string; waits for nothing else.
    void deliver(const orbbec_frame_gate &gate, std::uint64_t frame_timestamp_us,
                 std::vector<imu_sample> &into, orbbec_skips &skips) noexcept;

private:
    const char *m_sensor; ///< the sensor's name, for the reasons

    // The lock guards everything below: whether the queue runs, the ring, a ring of readings in
    // arrival order, and the counts of readings lost on OrbbecSDK's thread.
    std::mutex m_mutex;
    bool m_running = false;                       ///< whether add() takes readings
    std::vector<orbbec_imu_reading> m_ring;       ///< the readings held, a ring of fixed size
    std::size_t m_head = 0;                       ///< the slot of the oldest reading held
    std::size_t m_count = 0;                      ///< the readings held
    std::uint64_t m_dropped = 0;                  ///< the readings dropped from a full ring
    std::uint64_t m_unreadable = 0;               ///< the readings the callback could not read
    std::optional<std::uint64_t> m_last_given_us; ///< the last sample given's timestamp, in µs
};
} // namespace gca
