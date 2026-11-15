// SPDX-License-Identifier: 0BSD
/// @file test_orbbec_imu.cpp
/// The Orbbec camera's IMU without the camera (camera/orbbec_imu.hpp), fed with plain data as
/// OrbbecSDK reports it, through the frame gate (camera/orbbec_rules.hpp) that takes the frames:
/// - the stream of 200 Hz is chosen, the list's first of that rate; none when none has it;
/// - the IMU's start becomes no reason, or the reason the opening goes on without it: an
///   exception first, then the streams not offered, the accelerometer's, the gyroscope's or both;
/// - a frame is given the readings at or before its timestamp, the one at its very time among
///   them, in the order they came, placed on the frames' clock; a reading after it waits for the
///   next frame; a reading that comes late goes with the next frame, after the samples given;
/// - a jump of OrbbecSDK's timestamps by 2^32 µs is taken out of a reading's time, the frame's
///   own jump or the reading's;
/// - lost readings are skipped and counted with their reason, the frame's samples without them:
///   one the clocks do not place, one at 0 on the camera's clock, one before the frames' clock's
///   0, one not after the last sample given, those the full ring dropped (the oldest), those the
///   callback could not read;
/// - start() empties the ring and forgets the last sample given; stop() empties it and takes no
///   reading until the next start(), so that an opening whose IMU does not start is given none of
///   the last opening's readings, nor one a callback adds after the stop; a queue never started
///   gives none;
/// - readings added on another thread while frames are delivered all come, once, in order.
/// The test links no OrbbecSDK.

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/orbbec_imu.hpp"
#include "camera/orbbec_rules.hpp"
#include "check.hpp"
#include "types/frame_description.hpp"
#include "types/imu_sample.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
using gca_test::check;
using gca_test::check_refusal;

/// The frames' size: one pixel, as the IMU does not depend on the pictures [?: a choice].
constexpr gca::frame_size test_size = {1, 1};

/// The jump of OrbbecSDK's timestamps, 2^32 µs [M 2026-11-13, docs/HARDWARE.md].
constexpr std::uint64_t jump_us = std::uint64_t{1} << 32;

/// The first frame's time on the camera's clock and the host's, in µs [?: a choice, a camera up
/// for 20000 s, more than 2^32 µs, so that a reading may jump by -2^32 µs, and a host for
/// 1000 s].
constexpr std::uint64_t camera_0_us = 20'000'000'000;
constexpr std::uint64_t host_0_us = 1'000'000'000;

/// The time between two frames at 30 fps, in µs [S: 1 s / 30].
constexpr std::uint64_t frame_step_us = 33'333;

/// How long after a frame's camera time the host gets it out of OrbbecSDK's queue, in µs
/// [?: a choice, a frame's way through USB and the queue].
constexpr std::uint64_t frame_latency_us = 40'000;

/// How long after a reading's camera time the callback gets it, in µs [?: a choice].
constexpr std::uint64_t reading_latency_us = 2'000;

/// A frame of the test's camera: its pixels and what the gate is handed.
struct test_frame
{
    std::vector<std::uint8_t> color = std::vector<std::uint8_t>(3, 100);   ///< its one RGB pixel
    std::vector<std::uint16_t> depth = std::vector<std::uint16_t>(1, 500); ///< its one depth value
};

/// The test's camera: a gate and the accelerometer's queue, frames handed to the gate and readings
/// to the queue as OrbbecSDK would, at camera times the test chooses.
struct test_camera
{
    gca::orbbec_frame_gate gate;  ///< the opening's checks
    gca::orbbec_imu_queue accel;  ///< the accelerometer's readings
    gca::orbbec_skips skips;      ///< the readings lost
    gca::frame out;               ///< the frame taken last
    std::uint64_t next_index = 1; ///< the next frame's index
    test_frame pixels;            ///< the frames' pixels

    /// Starts an opening with a queue of a capacity.
    /// @param capacity [in] the ring's readings.
    explicit test_camera(std::size_t capacity)
        : accel("accelerometer", capacity)
    {
        gate.start(test_size, {1.0f, 1.0f, 0.0f, 0.0f});
        accel.start();
    }

    /// Adds a reading of a camera time, received reading_latency_us later on the host, its values
    /// made from its time.
    /// @param camera_us [in] its camera time, in µs.
    /// @param host_us [in] the host's time it came at, in µs; 0 for the host time that goes with
    ///        the camera time from the first frame on.
    void add(std::uint64_t camera_us, std::uint64_t host_us = 0)
    {
        const std::uint64_t host =
            host_us != 0 ? host_us : host_0_us + (camera_us - camera_0_us) + reading_latency_us;
        accel.add({camera_us, host, static_cast<float>(camera_us % 1000), 0.5f, -9.75f});
    }

    /// Takes a frame of a camera time, received frame_latency_us later on the host, and delivers
    /// the readings it carries.
    /// @param camera_us [in] its camera time, in µs.
    /// @param host_us [in] the host's time, in µs; 0 for the one that goes with the camera time.
    /// @return whether the gate took it.
    bool take(std::uint64_t camera_us, std::uint64_t host_us = 0)
    {
        gca::orbbec_wait wait;
        wait.status = gca::orbbec_wait_status::frameset;
        gca::orbbec_frameset &f = wait.frameset;
        f.color_size = test_size;
        f.depth_size = test_size;
        f.color_rgb = pixels.color;
        f.depth_16_bit = true;
        f.depth = pixels.depth;
        f.depth_scale_mm = 1.0f;
        f.index = next_index++;
        f.camera_time_us = camera_us;
        f.host_time_us =
            host_us != 0 ? host_us : host_0_us + (camera_us - camera_0_us) + frame_latency_us;
        const std::optional<gca::next_result> result = gate.take(wait, out);
        if (!result || result->status != gca::next_status::frame)
            return false;
        accel.deliver(gate, out.timestamp_us(), out.accel(), skips);
        return true;
    }

    /// Lists the timestamps of the samples the frame taken last carries.
    /// @return them, in the frame's order.
    std::vector<std::uint64_t> times() const
    {
        std::vector<std::uint64_t> listed;
        for (const gca::imu_sample &sample : std::as_const(out).accel())
            listed.push_back(sample.timestamp_us);
        return listed;
    }
};

/// Writes timestamps for a check line.
/// @param times [in] the timestamps, in µs.
/// @return them, separated by spaces.
std::string listed(const std::vector<std::uint64_t> &times)
{
    std::string text;
    for (const std::uint64_t t : times)
        text += " " + std::to_string(t);
    return text;
}

/// Checks the samples of the frame taken last and prints them.
/// @param what [in] the case.
/// @param camera [in] the camera.
/// @param expected [in] the timestamps expected, in µs, in order.
void check_times(const std::string &what, const test_camera &camera,
                 const std::vector<std::uint64_t> &expected)
{
    const std::vector<std::uint64_t> got = camera.times();
    std::printf("%s: samples at%s µs\n", what.c_str(), listed(got).c_str());
    if (got != expected)
        std::printf("%s: expected at%s µs\n", what.c_str(), listed(expected).c_str());
    check(got == expected, what.c_str());
}

/// The stream's choice and the reason an opening goes on without the IMU.
void test_choice_and_refusal()
{
    const std::vector<double> rates = {100.0, 500.0, 200.0, 1000.0, 200.0};
    const std::optional<std::size_t> chosen = gca::choose_imu_profile(rates);
    check(chosen == 2u, "choice: the list's first stream of 200 Hz");
    const std::vector<double> without = {100.0, 500.0, 1000.0};
    check(!gca::choose_imu_profile(without).has_value(), "choice: none without 200 Hz");

    const auto refusal_is = [](const std::string &what, const gca::orbbec_imu_report &report,
                               const std::string &expected) {
        const std::optional<std::string> reason = gca::imu_refusal(report);
        check_refusal(what, reason.has_value(), reason.value_or(""), reason == expected);
    };
    check(!gca::imu_refusal({true, true, std::nullopt}).has_value(),
          "IMU: both streams started, no reason");
    refusal_is("IMU, no accelerometer stream", {false, true, std::nullopt},
               "the camera offers no accelerometer stream at 200 Hz");
    refusal_is("IMU, no gyroscope stream", {true, false, std::nullopt},
               "the camera offers no gyroscope stream at 200 Hz");
    refusal_is("IMU, neither stream", {false, false, std::nullopt},
               "the camera offers no accelerometer and no gyroscope stream at 200 Hz");
    refusal_is("IMU, an exception while starting it",
               {true, false, gca::orbbec_error{gca::orbbec_exception::other, "start failed"}},
               "OrbbecSDK failed while starting the IMU: start failed");
}

/// A frame is given the readings at or before its time, in order; a reading after it waits for
/// the next frame; a late one goes with the next frame.
void test_assignment()
{
    test_camera camera(gca::orbbec_imu_queue_readings);
    const std::uint64_t c0 = camera_0_us;
    for (const std::uint64_t t : {c0 - 15'000, c0 - 10'000, c0, c0 + 5'000})
        camera.add(t);
    check(camera.take(c0), "assignment: frame 0 taken");
    check_times("assignment, frame 0: the readings up to its time, the one at its time among "
                "them, not the one after it",
                camera, {c0 - 15'000, c0 - 10'000, c0});
    check(camera.out.timestamp_us() == c0, "assignment: frame 0 on the camera's clock as it is");

    const std::uint64_t c1 = c0 + frame_step_us;
    for (const std::uint64_t t : {c0 + 10'000, c0 + 15'000})
        camera.add(t);
    check(camera.take(c1), "assignment: frame 1 taken");
    check_times("assignment, frame 1: the reading that waited, then its own", camera,
                {c0 + 5'000, c0 + 10'000, c0 + 15'000});

    // A reading of frame 1's interval that comes only after frame 1 was taken, then two of frame
    // 2's.
    const std::uint64_t c2 = c1 + frame_step_us;
    for (const std::uint64_t t : {c1 - 3'000, c1 + 1, c2 - 1})
        camera.add(t);
    check(camera.take(c2), "assignment: frame 2 taken");
    check_times("assignment, frame 2: the late reading, then its own", camera,
                {c1 - 3'000, c1 + 1, c2 - 1});
    check(camera.skips.count == 0, "assignment: no reading lost");
}

/// A reading that comes late after a later one was given is not after the last sample given:
/// skipped with its reason.
void test_not_after()
{
    test_camera camera(gca::orbbec_imu_queue_readings);
    const std::uint64_t c0 = camera_0_us;
    camera.add(c0 - 1'000);
    check(camera.take(c0), "not after: frame 0 taken");
    camera.add(c0 - 2'000);
    camera.add(c0 + 1'000);
    check(camera.take(c0 + frame_step_us), "not after: frame 1 taken");
    check_times("not after, frame 1: the reading after the last given only", camera, {c0 + 1'000});
    check_refusal("not after: the reading before the last given", camera.skips.count == 1,
                  camera.skips.last_reason,
                  camera.skips.last_reason == "the accelerometer's sample at " +
                                                  std::to_string(c0 - 2'000) +
                                                  " µs is not after the last one given, at " +
                                                  std::to_string(c0 - 1'000) + " µs");
}

/// OrbbecSDK's jumps of 2^32 µs are taken out of a reading's time: the frame's own, which the gate
/// takes out of the frames, and the reading's.
void test_jumps()
{
    test_camera camera(gca::orbbec_imu_queue_readings);
    const std::uint64_t c0 = camera_0_us;
    camera.add(c0 - 1'000);
    check(camera.take(c0), "jumps: frame 0 taken");
    // Frame 1 comes 2^32 µs late on the camera's clock; its readings do not.
    const std::uint64_t c1 = c0 + frame_step_us;
    camera.add(c1 - 2'000);
    check(camera.take(c1 + jump_us, host_0_us + frame_step_us + frame_latency_us),
          "jumps: frame 1 with a jump taken");
    check(camera.out.timestamp_us() == c1, "jumps: frame 1's jump taken out of the frame");
    check_times("jumps, frame 1 jumped: its reading without a jump in its place", camera,
                {c1 - 2'000});
    // Frame 2 lies on the camera's clock again; a reading of its interval jumps by -2^32 µs, one by
    // +2^32 µs.
    const std::uint64_t c2 = c1 + frame_step_us;
    camera.add(c2 - 20'000 - jump_us, host_0_us + (c2 - 20'000 - c0) + reading_latency_us);
    camera.add(c2 - 10'000 + jump_us, host_0_us + (c2 - 10'000 - c0) + reading_latency_us);
    check(camera.take(c2), "jumps: frame 2 taken");
    check_times("jumps, frame 2: readings that jumped either way in their places", camera,
                {c2 - 20'000, c2 - 10'000});
    check(camera.skips.count == 0, "jumps: no reading lost");
}

/// Readings the clocks do not place, one at 0 on the camera's clock and one before the frames'
/// clock's 0 are skipped with their reasons; the frame carries the others.
void test_not_placed()
{
    test_camera camera(gca::orbbec_imu_queue_readings);
    const std::uint64_t c0 = camera_0_us;
    // 20 s on the camera's clock that the host did not see pass: no jump of 2^32 µs.
    camera.add(c0 - 20'000'000, host_0_us - 1'000);
    camera.add(c0 - 500);
    check(camera.take(c0), "not placed: frame 0 taken");
    check_times("not placed, frame 0: the reading the clocks place", camera, {c0 - 500});
    check_refusal(
        "not placed: a reading 20 s off the host's clock", camera.skips.count == 1,
        camera.skips.last_reason,
        camera.skips.last_reason ==
            "the accelerometer's reading at " + std::to_string(c0 - 20'000'000) +
                " µs on the camera's clock cannot be placed on the frames' clock: its time on the "
                "camera's clock stepped by -20000000 µs from the last taken frame's while the "
                "host's stepped by -41000 µs, which is no jump of 2^32 µs");
    camera.add(0, host_0_us);
    check(camera.take(c0 + frame_step_us), "not placed: frame 1 taken");
    check_refusal("not placed: a reading at 0 on the camera's clock", camera.skips.count == 2,
                  camera.skips.last_reason,
                  camera.skips.last_reason ==
                      "the accelerometer's reading at 0 µs on the camera's clock cannot be placed "
                      "on the frames' clock: its time on the camera's clock, 0 µs, is 0 or 2^60 "
                      "µs or more");

    // A camera whose first frame's time came 2^32 µs short, 1000 µs, and a reading of 1500 µs
    // before it that came whole: placed at -500 µs on the frames' clock.
    test_camera short_clock(gca::orbbec_imu_queue_readings);
    short_clock.add(jump_us - 500, host_0_us - 1'500 + reading_latency_us);
    check(short_clock.take(1'000, host_0_us + frame_latency_us), "before 0: frame 0 taken");
    check_times("before 0: no sample", short_clock, {});
    check_refusal("before 0: a sample before the frames' clock's 0", short_clock.skips.count == 1,
                  short_clock.skips.last_reason,
                  short_clock.skips.last_reason ==
                      "the accelerometer's sample at -500 µs lies before the frames' clock's 0");
}

/// A full ring drops its oldest readings, the callback's unreadable ones are counted; both reach
/// the skips at the next frame, with their reasons. start() empties the ring and forgets the last
/// sample given.
void test_lost_and_start()
{
    test_camera camera(4);
    const std::uint64_t c0 = camera_0_us;
    for (std::uint64_t k = 0; k < 6; k++)
        camera.add(c0 - 6'000 + 1'000 * k);
    check(camera.take(c0), "full: frame 0 taken");
    check_times("full: the 4 newest readings", camera,
                {c0 - 4'000, c0 - 3'000, c0 - 2'000, c0 - 1'000});
    check_refusal("full: the 2 oldest readings dropped", camera.skips.count == 2,
                  camera.skips.last_reason,
                  camera.skips.last_reason == "the accelerometer's queue of 4 readings was full, "
                                              "so its oldest reading went");
    camera.accel.add_unreadable();
    camera.accel.add_unreadable();
    camera.accel.add_unreadable();
    check(camera.take(c0 + frame_step_us), "unreadable: frame 1 taken");
    check_refusal("unreadable: 3 readings the callback could not read", camera.skips.count == 5,
                  camera.skips.last_reason,
                  camera.skips.last_reason ==
                      "OrbbecSDK's accelerometer reading could not be read in its callback");

    // A new opening: what the ring held and the last sample given are forgotten.
    camera.add(c0 + frame_step_us + 30'000);
    camera.gate.start(test_size, {1.0f, 1.0f, 0.0f, 0.0f});
    camera.accel.start();
    camera.skips = {};
    camera.add(c0 - 9'000);
    check(camera.take(c0), "start: frame 0 of the new opening taken");
    check_times("start: the new opening's reading, before the last opening's last sample", camera,
                {c0 - 9'000});
    check(camera.skips.count == 0, "start: nothing lost");
}

/// stop() empties the queue and takes no reading until start(): the next opening, its IMU not
/// started, is given neither the reading the last opening held for a later frame nor one a
/// callback adds after the stop, and nothing is counted lost; a queue never started gives none.
void test_stop()
{
    test_camera camera(gca::orbbec_imu_queue_readings);
    const std::uint64_t c0 = camera_0_us;
    camera.add(c0 - 2'000);
    camera.add(c0 + 5'000);
    check(camera.take(c0), "stop: frame 0 taken");
    check_times("stop, frame 0: its reading, the one after it held", camera, {c0 - 2'000});
    camera.accel.add_unreadable();
    camera.accel.stop();
    camera.add(c0 + 6'000);
    camera.accel.add_unreadable();
    // The next opening, whose IMU does not start.
    camera.gate.start(test_size, {1.0f, 1.0f, 0.0f, 0.0f});
    camera.skips = {};
    check(camera.take(c0 + 10'000), "stop: the next opening's frame 0 taken");
    check_times("stop, the next opening without the IMU: none of the last opening's readings, "
                "none added after the stop",
                camera, {});
    check(camera.skips.count == 0, "stop: nothing counted lost");

    gca::orbbec_imu_queue never("gyroscope", 4);
    never.add({c0 + 9'000, host_0_us + 9'000 + reading_latency_us, 1.0f, 2.0f, 3.0f});
    never.add_unreadable();
    std::vector<gca::imu_sample> given;
    gca::orbbec_skips skips;
    never.deliver(camera.gate, camera.out.timestamp_us(), given, skips);
    check(given.empty() && skips.count == 0, "stop: a queue never started gives nothing");
}

/// Readings added on another thread, as OrbbecSDK's callback adds them, while frames are taken:
/// every reading comes once, in order, none lost. Each side runs at most 1 s of camera time ahead
/// of the other, so that the ring holds at most 2 s of readings, 400 of its 800, and readings come
/// both before their frame and late.
void test_threads()
{
    test_camera camera(gca::orbbec_imu_queue_readings);
    constexpr std::uint64_t readings = 20'000;
    constexpr std::uint64_t step_us = 5'000;
    constexpr std::uint64_t ahead_us = 1'000'000;
    const std::uint64_t c0 = camera_0_us;
    const std::uint64_t last_us = c0 + readings * step_us;
    camera.add(c0 - 1);
    check(camera.take(c0), "threads: frame 0 taken");
    std::vector<std::uint64_t> given = camera.times();
    std::atomic<std::uint64_t> added_us{c0 - 1};
    std::atomic<std::uint64_t> framed_us{c0};
    std::thread callback([&] {
        for (std::uint64_t k = 1; k <= readings; k++)
        {
            const std::uint64_t t = c0 + k * step_us;
            while (t > framed_us.load() + ahead_us)
                std::this_thread::yield();
            camera.add(t);
            added_us.store(t);
        }
    });
    // Frames every 33 ms of camera time until the readings' last.
    std::uint64_t c = c0;
    bool all_taken = true;
    while (c < last_us)
    {
        c += frame_step_us;
        while (c > added_us.load() + ahead_us)
            std::this_thread::yield();
        all_taken = all_taken && camera.take(c);
        framed_us.store(c);
        for (const std::uint64_t t : camera.times())
            given.push_back(t);
    }
    callback.join();
    // The readings added after the last frame was taken go with one more.
    all_taken = all_taken && camera.take(c + frame_step_us);
    for (const std::uint64_t t : camera.times())
        given.push_back(t);
    check(all_taken, "threads: every frame taken");
    std::vector<std::uint64_t> expected = {c0 - 1};
    for (std::uint64_t k = 1; k <= readings; k++)
        expected.push_back(c0 + k * step_us);
    std::printf("threads: %zu samples given of %zu readings, %llu lost\n", given.size(),
                expected.size(), static_cast<unsigned long long>(camera.skips.count));
    check(given == expected && camera.skips.count == 0,
          "threads: every reading given once, in order, none lost");
}
} // namespace

/// Runs the IMU's checks and prints the samples, the reasons and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    test_choice_and_refusal();
    test_assignment();
    test_not_after();
    test_jumps();
    test_not_placed();
    test_lost_and_start();
    test_stop();
    test_threads();
    return gca_test::check_summary("test_orbbec_imu");
}
