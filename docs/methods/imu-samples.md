# The IMU's samples with the frames

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The camera's accelerometer and gyroscope come with the frames: each frame a frame source
delivers carries, for each of the two sensors, the samples taken since the samples the frame
before carried, each with its own timestamp on the frames' clock. The sequence keeps them with
their frame (docs/SEQUENCE_FILE.md). No stage reads them yet: separation's up from gravity is the
first reader (`separation-up-from-gravity`), the gyroscope as tracking's prior a candidate built
only on a measured benefit (ROADMAP.md, "Candidates"). The code: the sample
`src/types/imu_sample.hpp`, the frame's lists `src/camera/frame.*`, the camera's streams
`src/camera/orbbec_device.*`, its queue and the placement on the frames' clock
`src/camera/orbbec_imu.*` and `src/camera/orbbec_rules.*`, the synthetic source's samples
`tests/synthetic_source.*`; the tests `tests/test_orbbec_imu.cpp`,
`tests/test_synthetic_source.cpp` and `tests/test_sequence.cpp`.

## The IMU and its streams

The Gemini 2 L's IMU offers an accelerometer of 4 g and a gyroscope of 1000 dps, each at 100,
200, 500 Hz and 1 kHz [M 2026-11-13: docs/HARDWARE.md, "What the camera is and reports"].
`orbbec_device` starts both at **200 Hz** [?: initial], beside depth and color, when it opens the
camera: 6 or 7 samples of each sensor a frame at 30 fps [S: computed], enough for the gyroscope's
turn between two frames and for gravity near any frame, at the rate the visual-inertial
datasets record theirs (EuRoC, ADIS16448 at 200 Hz [S: M. Burri et al., "The EuRoC micro aerial
vehicle datasets", IJRR 35(10), 2016]; TUM VI, 200 Hz [S: D. Schubert et al., "The TUM VI
benchmark for evaluating visual-inertial odometry", IROS 2018]). A faster rate is chosen only on a
measured benefit on real sequences (the gyroscope's prior).

Each sensor streams through OrbbecSDK's sensor of its own (`ob::Sensor::start()` with the
profile of that rate, the list's first of it), which hands every reading to a callback on
OrbbecSDK's thread, one reading at a time. A pipeline's frame aggregation pairs frames into
framesets, one frame of a kind each, which would drop readings at a rate above the frames';
OrbbecSDK's own example starts the IMU apart from the video for that reason
[S: OrbbecSDK v2.10.6, examples/src/1.stream.multi_streams]. `orbbec_device` sets no property and
writes nothing to the flash; what OrbbecSDK sends to the camera to start a stream is unverified
[?], as for depth and color.

Both sensors start or neither. When the camera offers no accelerometer or no gyroscope stream at
200 Hz, or OrbbecSDK throws while starting them, the camera opens without its IMU and its frames
come without samples [?: a choice, the orchestrator's, 2026-11-15]: a refusal of the camera would
make it unusable wherever its IMU does not stream (on a USB 2 link, unverified), whereas the scan
goes on and the source says what it went without. The opening names the reason
(`orbbec_opening::imu_refusal`, `src/camera/orbbec_imu.hpp`, `imu_refusal()`), and
`orbbec_viewer` prints it as `No IMU: <reason>; the frames come without its samples`. A scan
without samples runs and is refined as usual (the scan interface refuses none); separation then
proposes no support plane and box, without gravity (the user's decision of 2026-11-16,
`separation-up-from-gravity`).

## A sample

A sample (`gca::imu_sample`) is a timestamp and three floats:

* **the timestamp**, in µs on the frames' clock ("The clock");
* **x, y, z**, as OrbbecSDK gives them (`OBAccelValue`, `OBGyroValue`), bit for bit: the
  accelerometer's in m/s², the gyroscope's in rad/s [S: OrbbecSDK v2.10.6,
  examples/src/1.stream.imu/README.md]. The accelerometer measures specific force, the
  acceleration less gravity, as every accelerometer does: held still it reads 9.81 m/s²
  pointing up [S: the physics of a proof mass]; that OrbbecSDK reports it with that sign is
  unverified [?]. The gyroscope's rate is taken right-handed about each axis [?].

## The axes

The samples stay in the IMU's axes as OrbbecSDK reports them; no source rotates them, so that a
rotation measured later applies to every sequence taken before it. OrbbecSDK reports the
accelerometer's and the gyroscope's extrinsics to the color camera equal to the depth camera's,
to every digit, and the IMU's intrinsics empty (noise, random walk and bias 0, the scale the
identity) [M 2026-11-13: docs/HARDWARE.md]: by that report the IMU's axes are the depth camera's,
whose rotation to the color camera lies within 0.001 of the identity (docs/HARDWARE.md), x right,
y down, z forward. Whether the IMU's axes lie so is unverified [?]: the report may be a
placeholder. Until the live check measures it, the
rotation from the IMU's axes to the camera frame is the identity [?: initial], one constant of
this document that a reader of the samples applies.

What that predicts, and the live check checks (`hardware-check-orbbec-on-seam`): the camera held
still and level, looking ahead, the accelerometer reads about (0, -9.81, 0) m/s²; turned to the
right about the vertical, the gyroscope's y reads positive, and every rate about 0 while it is
held still.

## The clock

OrbbecSDK gives every frame, an IMU reading's as a picture's, a timestamp it calls the device's,
"the time point when the frame was captured by the device, on device clock domain"
[S: OrbbecSDK v2.10.6, Frame.hpp, `getTimeStampUs`]. The IMU's readings are taken to lie on the
same camera clock as the color frames' [?: the live check].

The frames' clock is the clock of the timestamps the frame source delivers: for the camera, the
color frame's camera time with OrbbecSDK's jumps of 2^32 µs taken out
(`src/camera/orbbec_rules.hpp`, `orbbec_frame_gate`); for a sequence, the timestamps it holds;
for the synthetic source, i · 33333 µs for frame i. A reading is placed on it by the frames' own
rule, measured against the last frame taken, the frame it is placed for:

    d = c - C,   h_step = h - H,   d - h_step = k · 2^32 µs + r,  |r| <= 2^31 µs,
    t = T + d - k · 2^32 µs

c is the reading's camera time and h the host's steady clock when the callback got it; C, H and
T are the frame's camera time, the host's time when the frame came out of OrbbecSDK's queue, and
its delivered timestamp. A rest |r| above 10 s (`orbbec_clock_tolerance_us`, the frames'
tolerance) means the reading is not on the frames' clock: it is skipped, with its reason. So a
jump of OrbbecSDK's timestamps in the IMU's readings is taken out as in the frames', and a
reading needs no jump-free history of its own. The host times only tell k: the frames come out of
OrbbecSDK's queue up to 10 frames late, 0.33 s at 30 fps [S: Pipeline.cpp, the queue of 10], far
inside the tolerance.

## Which frame carries a sample

Frame i carries, for each sensor, the samples not yet given whose timestamp is at most its own,
t_i, in increasing order. Each sensor's samples of an opening are given once, and those of all
frames, in frame order, form the sensor's stream, strictly increasing in time. Normally a frame
carries the samples of (t_{i-1}, t_i]; the first frame carries those the queue holds from before
it ("The queue"). A frame never waits for a sample: next() takes what has arrived when the frame
is taken.

* **Late**: a reading that comes after the frame whose interval holds it was taken goes with the
  next frame, keeping its own timestamp, at or before that earlier frame's; it is given when it
  follows the sensor's last sample given, as a reading that came in its order does.
* **Ahead**: a reading whose timestamp lies after t_i waits in the queue for a later frame.
* **Missing**: a frame may carry no sample of a sensor, or fewer than the rate gives; that is a
  value, not a failure, and a reader sees the gap in the samples' timestamps. A frame the
  sequence's writer does not take (its queue full, docs/SEQUENCE_FILE.md) takes its samples
  with it [?: a choice: the sequence keeps a frame's record whole, and the scanner fuses no frame
  it does not write; keeping them would take a record of samples without pictures]. A replayed
  sequence's samples have a gap there, likelier at 1280x800, whose frames fill the writer's queue
  sooner.
* **Lost**: a reading is skipped and counted with its reason when it cannot be placed on the
  frames' clock, when its timestamp is not after the sensor's last sample given or lies before
  the clock's 0, when OrbbecSDK's reading cannot be read in the callback, and when the queue is
  full (the oldest reading goes). `orbbec_viewer` prints the count and the last reason,
  `Skipped <count> IMU reading(s), the last because <reason>`, as it does for frames skipped.

A frame source that has no IMU, or whose IMU did not open, gives frames without samples.

## The queue

The callback puts each reading, its camera time, its host time and its three values, into a
ring of its sensor that holds 4 s of readings at the rate, 800 [?: a choice: twice the 2 s that
OrbbecSDK's frameset queue of 10 holds at 5 fps (S: Pipeline.cpp, OrbbecSDK v2.10.6), so a
reading still waits for the oldest frame OrbbecSDK hands out], allocated when the camera source
is built. It is emptied when the IMU starts at an opening and when it stops, and takes readings
only in between, so that no opening is given another opening's readings and one whose IMU did not
start is given none. A lock guards the ring and the counts of readings lost; the callback holds
it for the copy of one reading, after its OrbbecSDK calls, next() for the walk over the ring once
a frame is taken, which places each reading against that frame, gives it to the frame, skips it
or leaves it, in its order, for a later frame; the walk counts what it skips and builds the one
reason after it lets the lock go.

## The synthetic source

The tests' synthetic source (`tests/synthetic_source.hpp`) samples an IMU when its options ask
for one, every period from time 0 on, in the camera frame (x right, y down, z forward), so that a
test knows the truth: between frames i - 1 and i the camera turns at the constant rate
ω_i = log(R_{i-1}^T R_i) / (t_i - t_{i-1}), in the camera frame, R_i the rotation of pose i, and
at a time t of (t_{i-1}, t_i] its rotation is R(t) = R_{i-1} exp((t - t_{i-1}) ω_i). The
gyroscope reads ω_i, the accelerometer gravity's specific force R(t)^T (0, 0, g) with g =
9.80665 m/s² [S: the standard acceleration of gravity, CGPM 1901], the world's z up; no
acceleration of the hand. Before frame 0, at its time 0, the camera rests at pose 0. Frame i
carries the samples of (t_{i-1}, t_i], frame 0 the one at 0.

## On the per-frame path

A frame holds the samples in two lists of its own, which prepare() empties and reserves for 64
samples of each sensor [?: a choice: twice the 33 a frame gets at 1 kHz and 30 fps]
(`frame_imu_samples_reserved`, `src/camera/frame.hpp`). At 200 Hz a frame carries 6 or 7; it
carries more only after a gap: the first frame of an opening, which takes what the queue holds
from before it, up to 800, and a frame after a stall of the camera's frames or of the caller
longer than 0.32 s [S: computed, 64 at 200 Hz]. Such a frame's list grows once, as a frame's
pictures grow when it is first prepared, and keeps its capacity; the frames' buffers swap with
the sequence's writer and reader, so a list that grew travels on. The writer takes a frame by a
swap of buffers, samples included, and its reader reads them into its own frame, so neither
allocates for the samples of a frame within its lists' capacity. next() waits for nothing for
the IMU; it takes each ring's lock for its walk, which calls no OrbbecSDK and no GPU.

## Live checks

On the user's word (`hardware-check-orbbec-on-seam`): the IMU streams with depth and color at
both stream sizes, about 200 samples of each sensor a second, none skipped; their timestamps rise
and lie on the frames' clock as "The clock" says (the samples a frame carries lie within
(t_{i-1}, t_i] but for late ones); gravity along -y with the camera held still and level, the
gyroscope's y positive on a turn to the right; whether the IMU streams on a USB 2 link, or the
camera opens there without it.
