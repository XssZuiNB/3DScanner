# Hardware

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The board this tree runs on. Every fact carries its tag (AGENTS.md).

## Platform

| what | value | evidence |
|---|---|---|
| computer | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super, 7.4 GiB RAM | [M 2026-11-13] `/proc/device-tree/model`, `free` |
| power mode | MAXN_SUPER | [M 2026-11-13] `nvpmodel -q` |
| OS | L4T R36, revision 5.2 (JetPack 6), Ubuntu 22.04, kernel 5.15.199-tegra, aarch64 | [M 2026-11-13, since the boot at 20:13] `/etc/nv_tegra_release`, `uname -a` |
| toolchain | CUDA 12.6.68, GCC 11.4.0, OpenCV 4.8.0; CMake 4.4.4 from PyPI in `~/.local/bin`, before apt's 3.22.1 in `/usr/bin` (CODING_STANDARDS.md, "Tools") | [M 2026-11-13]; CMake [M 2026-11-14] |
| GPU | Ampere, compute capability 8.7 | [S] CUDA documentation |
| camera | Orbbec Gemini 2 L, USB `2bc5:0673`, firmware 1.5.02 since 2026-11-15, 1.4.38 before ("What the camera is and reports") | [M 2026-11-13] `lsusb`, `orbbec_viewer --list`; the firmware [M 2026-11-15] |
| freezes | 2026-11-13 under a build of six jobs (AGENTS.md §4); 2026-11-15 at about 13:51 with no job running and the camera closed (at 13:47), cause unknown [?]: syslog's last line before it is a Wi-Fi rekey, followed by NUL bytes, and no OOM, nvgpu, Xid, thermal or hung-task line precedes it. The journal is kept in memory only (no `/var/log/journal`), so after a freeze only syslog remains; `/sys/fs/pstore` is root's | [M 2026-11-15] syslog, `last -x` |
| camera SDK | OrbbecSDK v2.10.6, linux arm64, in `~/.local/share/3dscanner` | tools/fetch_deps.sh |

## Memory, storage, network

What the scanner has to fit into and how it is reached [M 2026-11-13].

| what | value | evidence |
|---|---|---|
| memory | 7607 MB, shared by CPU and GPU; about 4.0 GB available with a remote editor session open and no build running | `free -m` |
| swap | six zram devices of 634 MB each, and `/swapfile`, 8 GB, on the NVMe | `swapon --show` |
| storage | one NVMe, 937 GB, 852 GB free | `df -h /` |
| writing a sequence | a frame of 1280x800, 5.12 MB: `write()` into the page cache 5.9 ms (median of 100), with `fdatasync()` after each 12.9 ms (max 20.1 ms); the sequence's writer back to back 7.8 ms per frame once past the kernel's background limit (at most 351 MB dirty), 3.07 GB with `close()`'s sync 7.9 ms per frame, 646 MB/s, the longest wait for room in its queue of 16 46 ms; at 30 fps (154 MB/s) for 30 s, 4.61 GB, it took 900 of 900 frames | [M 2026-11-14, MAXN_SUPER, no build running, ext4 on the NVMe] a program outside the tree on `src/camera/sequence_writer.hpp` |
| network | Wi-Fi `wlP1p1s0` connected; Ethernet `eno1` without a cable | `nmcli device` |
| the user's services | `Linger=no`: they start at the user's login, not at boot; `sudo loginctl enable-linger <user>` changes it, a root step | `loginctl show-user` |

## The camera on USB 3

What the camera offers and does at 5000M; OrbbecSDK reports `USB3.0` [M 2026-11-13,
`orbbec_viewer --list`, `--frames 100 --headless`].

* **Color:** RGB, MJPG and YUYV at 30, 15, 10, 5 fps in 1280x800, 1280x720, 800x600,
  640x400, 640x360. No 640x480, nothing above 1280x800.
* **Depth:** Y14 and RLE at 30, 15, 10, 5 fps in 1280x800, 640x400, 320x200. The SDK
  delivers Y14 as 16-bit frames.
* **Both runs below:** color RGB, depth Y14, hardware depth-to-color alignment, depth unit
  1 mm, about 47 % of pixels with depth in an indoor scene at 0.5 m median.
* **The times in this section** are the viewer's `gpu point cloud` while it ran the prototype's
  point cloud from a frame with its upload, a path that is deleted (in git at 74dda68); since then
  the line times the upload, the organized points and the read-back of their validity mask
  (`apps/viewer_loop.hpp`), not measured on the camera yet [?]:
  `first-trial-of-held-exposure` measures it and replaces these times.

| size at 30 fps | intrinsics fx, fy, cx, cy | fps | the prototype's point cloud with upload | points |
|---|---|---|---|---|
| 640x400 | 306.18, 306.14, 317.41, 199.13 | 29.1 to 30.2 | 5.7 to 5.9 ms | about 91 000 |
| 1280x800 | 612.37, 612.29, 635.81, 398.27 | 28.6 to 30.3 | 19.4 to 19.9 ms | about 383 000 |

After the cleanup (the tree at b23f168), the same session ran main's viewer (b5c70dc) and the
cleaned tree alternately, 100 frames headless each, on a scene with about 1.5x the points of the
table above [M 2026-11-13]:

| size at 30 fps | the prototype's point cloud with upload, main (two runs) | the cleaned tree (two runs) | points |
|---|---|---|---|
| 640x400 | 6.99 / 7.55 ms | 7.40 / 7.60 ms | 133 000 to 136 000 |
| 1280x800 | 25.64 / 25.04 ms | 25.96 / 24.32 ms | 530 000 to 558 000 |

The fps is about 30 in every run [M 2026-11-13]. Both trees run above the table above on this
scene; whether its points, its content or the GPU clock account for that is open [?]. Main
and the cleaned tree differ by no more than main's own run-to-run spread (8 % at 640x400)
[M 2026-11-13]. The GPU clock runs under the governor `nvhost_podgov`, 306 to
1020 MHz, and idles at 306 MHz; each run lasts about 3 s, so the governor's ramp is part of
every number [M 2026-11-13].

OrbbecSDK's warnings in these runs, the stream going on after each:

* `Not added to property map ... id=108` and `... id=5003`, on every open [M 2026-11-13].
* `Timestamp anomaly detected ... currentDiff: 4294967296`: the frame timestamp jumps by
  exactly 2^32 µs [M]. OrbbecSDK joins the high 32 bits of a camera time it reads now and then
  to each frame's low 32 bits, so a misjudged high part jumps by 2^32 µs either way, and its
  detector drops the frame it flags [S: OrbbecSDK v2.10.6, FrameTimestampCalculator.cpp,
  SensorBase.cpp]; what makes it misjudge the high part on this camera, at start and at stop,
  is open [?]. The first session saw it a few times at the start of every run [M]. The
  after-cleanup session saw it in 8 of 10 runs, in both trees: before the first report line in
  3 runs, at stop in 6 (one run had both), none in 2; at stop it is an `[error]` line for each
  sensor, and the stream goes on until the program closes [M 2026-11-13]. `orbbec_device` takes
  the jump out of the timestamps it delivers (src/camera/orbbec_rules.hpp). On 1.4.38, on
  2026-11-15, it came steadily, once every 4.29 s while the stream ran, and every frame after
  the first carried the same timestamp, its low 32 bits standing at about 60 µs (in one run 61 ×
  2^32 + 60 µs) [M 2026-11-15, 13:46 and 16:40, the camera power-cycled before the second;
  OrbbecSDK's own `ob_color` the same]: `orbbec_device` skipped every frame as not after the last
  taken frame's, so no frame came for 30 s, and it never recovers from that. On 1.5.02
  `ob_color` ran 20 s without the line and `orbbec_device` streamed with the device's
  timestamps, no frame skipped [M 2026-11-15, 16:51 to 17:01].
* A depth frame dropped at the start. At 1280x800: `Update data size(1824744) > data buffer
  size! (1792000)`; 1792000 bytes is 1280 x 800 x 14 bit, one packed Y14 frame [S]. At 640x400
  too, under a second wording: `This frame will be dropped because the data size does not
  match the expectation! ... expected=1792000 @Depth`; at 1280x800 this wording also appears
  [M 2026-11-13].

## The camera on USB 2

What the camera offers and does at 480M [M 2026-11-13, `orbbec_viewer --list`, `--frames 100`],
the fallback when the USB 3 link does not come up.

* **Color:** RGB and MJPG up to 1280x800 (30, 15, 10, 5 fps); YUYV up to 800x600; sizes
  1280x800, 1280x720 (MJPG only at 30 fps), 800x600, 640x400, 640x360. No 640x480.
* **Depth:** Y14 and RLE only: 1280x800 at 10 and 5 fps; 640x400 and 320x200 at 30, 15, 10,
  5 fps. The SDK delivers Y14 as 16-bit frames.
* **640x400 at 30 fps, color RGB, depth Y14:** hardware depth-to-color alignment;
  intrinsics fx 306.18, fy 306.14, cx 317.41, cy 199.13; depth unit 1 mm; 30.0 fps;
  the prototype's point cloud 6.5 to 6.9 ms with upload (as the times of "The camera on USB 3");
  58 % of pixels with depth in an indoor scene, 0.25 m to 21 m.

## What the camera is and reports

From Orbbec's "Gemini 2L Datasheet v1.0" [S: datasheet], and read from the camera over
OrbbecSDK on USB 3 with no stream running and no property set [M 2026-11-13].

* **Optics** [S: datasheet]: active stereo IR at 850 nm, baseline 100 mm, depth computed in
  the camera (MX6600 ASIC); IR and color sensors are both global shutter; depth from 0.25 m
  to 10 m; depth accuracy 2 % or better at 2 m (1280x800, the central 81 %, RMS to a fitted
  plane).
* **Depth work modes** [M]: `Unbinned Dense Default` (in force), `Dimensioning`,
  `Binned Sparse Default`, `Unbinned Sparse Default`, `In-scene Calibration`. Optimal range
  0.30 m to 7.0 m for Unbinned Dense Default, 0.30 m to 4.0 m for Dimensioning
  [S: datasheet]. The camera has no presets: `getAvailablePresetList()` fails with
  `Device Component 'preset manager' not found!` [M].
* **Depth unit** [M]: `OB_PROP_DEPTH_PRECISION_LEVEL_INT` is readable and writable and
  stands at 0 (1 mm); the SDK reports its range as min 6, max 0. Which levels the camera
  accepts is untested [?]. `OB_PROP_DEPTH_UNIT_FLEXIBLE_ADJUSTMENT_FLOAT` is not supported.
* **Exposure, laser, filter** [M], all readable and writable: depth auto exposure (on),
  depth exposure (5000; 200 to 10000), color auto exposure (on), color exposure (10000; 0 to
  30000), color gain (101; 1 to 255), color auto white balance (on), laser power level (5;
  0 to 7), the camera's depth noise removal (on; max diff 256, max speckle size 200).
  `OB_PROP_LASER_BOOL` reads 0 with no stream running; what it reads while streaming is
  unverified [?]. `OB_PROP_LASER_CONTROL_INT` and `OB_PROP_LASER_ON_OFF_PATTERN_INT` are not
  supported.
* **IMU** [M]: accelerometer 4 g, gyroscope 1000 dps, each at 100, 200, 500 Hz and 1 kHz.
  The intrinsics the SDK reports are empty: noise density, random walk and bias 0, the scale
  matrix the identity. The accel-to-color and gyro-to-color extrinsics equal depth-to-color
  to every digit and accel-to-gyro is the identity: where the IMU sits and how its axes lie
  is unverified [?]. `orbbec_device` streams both at 200 Hz with the frames, each frame carrying
  the samples since the frame before; their axes, clock and live checks:
  docs/methods/imu-samples.md.
* **Firmware** [M 2026-11-15]: 1.5.02 since 2026-11-15, 16:50, updated on the user's word from
  1.4.38, below OrbbecSDK v2's least firmware for the Gemini 2 L, 1.4.53 (its recommended:
  1.5.2) [S: OrbbecSDK v2's README and release notes, web]. On 1.4.38 every frame's device
  timestamp stood still ("The camera on USB 3", OrbbecSDK's warnings), and `orbbec_device`
  skipped every frame without recovering; so `orbbec_device` refuses a camera whose firmware is
  older than 1.4.53, or whose version is not three numbers, when it opens it, before any stream
  starts (src/camera/orbbec_rules.hpp, `firmware_refusal()`). The version is the camera's own
  string, which OrbbecSDK reads from the camera's version structure and passes on unchanged
  [S: OrbbecSDK v2.10.6, DeviceBase.cpp, `fetchDeviceInfo`]: "1.4.38" and "1.5.02" here
  [M 2026-11-15: OrbbecSDK's log]. The update:
  `MFG_MX6600_Gemini2_L_1502.bin`, 327680 bytes, from github.com/orbbec/OrbbecFirmware, release
  "Gemini2L-Firmware", kept outside the tree in `~/3DScanner-backup/camera-firmware/`, written by
  OrbbecSDK's example `ob_device_firmware_update -f FILE`, built from the SDK's sources outside
  the tree: "Update successfully" after 8 s, and the camera came back on USB 3 as 1.5.02. That
  tool forces OrbbecSDK's libusb backend, and each of its opens made the camera leave and
  re-enter USB 3 ("Kernel log"); its first device list found no camera, the second found it.
  The measurements of this document dated before 2026-11-15, 16:50 were made on 1.4.38.

| stream at 1280x800 | fx, fy, cx, cy | distortion |
|---|---|---|
| color | 612.369, 612.286, 635.814, 398.267 | Brown-Conrady K6: k1 -1.01331, k2 0.497682, k3 -0.106898, k4 -0.994527, k5 0.471835, k6 -0.0972926, p1 0.00022544, p2 0.000242177 |
| depth, not aligned | 617.492, 617.492, 641.474, 398.699 | all coefficients 0 |

Depth to color [M]: a rotation within 0.001 of the identity, translation -25.064, -0.120,
0.099 mm. By the color coefficients a color pixel is displaced 4.0 px at the middle of the
left and right edges, 2.5 px at the middle of the top and bottom edges and 8.6 px in a
corner [?: computed from the coefficients, not measured on an image]. Whether hardware
depth-to-color alignment delivers depth on the distorted color grid or on an undistorted one
is unverified [?]; `distortion_check` measures it on a snapshot of a flat checkerboard
(README.md, "Run").

**Pixel convention** [S: OrbbecSDK_v2's source at tag v2.10.6, the installed version]: the
SDK's own code uses its intrinsics with pixel centres at whole numbers, the convention of the
rewrite. Its deprojection takes the pixel index as it is, x = d (u - cx) / fx
(`src/shared/utils/CoordinateUtil.cpp`, `transformation2dTo3d`; its xy tables put pixel u at
`(float)u`), and mirroring an image maps cx to w - cx - 1
(`src/shared/utils/CameraParamProcess.cpp`, `cameraIntrinsicParamsMirror`). That is how the
SDK's code uses the intrinsics, not a statement of the calibration that produced them. A
checkerboard cannot check it: an error of the principal point by δ leaves the picture of a
plane a homography and moves every deprojected point by z δ / f, 0.4 mm at 0.5 m for half a
pixel, without bending a plane [S: the pinhole model].

## The per-frame path's stages

Each stage's time on the board (CODING_STANDARDS.md, "Measured stage by stage"), measured on
synthetic frames by a scratch program, under the build lock with no compiler running
(AGENTS.md §4), in the power mode MAXN_SUPER with the GPU clock under `nvhost_podgov`, 306 to
1020 MHz [M 2026-11-14 and 2026-11-15, each stage's date with its table]. The replay of sequences
taken on the hardware measures the whole path later (`per-stage-times-from-replay`).

**The upload** (`src/gpu/gpu_frame.hpp`), staged through pinned memory: medians over 10 s per
case after 2 s [M 2026-11-14, 09:55 to 09:57; the GPU clock was not recorded]. Behind queued
work, 8 memsets of 256 MiB, 22.3 ms of work, stand queued on the stream before the upload.

| size | stream | host blocked in the call | copies on the GPU | call to copies done |
|---|---|---|---|---|
| 640x400 | idle | 0.161 ms | 0.314 ms (1) | 0.342 ms |
| 640x400 | behind queued work | 0.258 ms | 0.063 ms | 22.285 ms |
| 1280x800 | idle | 0.896 ms | 1.488 ms (1) | 1.522 ms |
| 1280x800 | behind queued work | 0.906 ms | 0.201 ms | 22.603 ms |

(1) On an idle stream the GPU's time starts at the call, so it includes the host's staging.

**The organized points** (docs/methods/organized-points.md): the kernel's time by CUDA events
and the host's call, over 20 s per case after 5 s, on one frame of a box on a plane, depth unit
1 mm, range 0.30 m to 1.0 m [M 2026-11-14, 10:29 to 10:31]. Paced at 30 fps the governor keeps
the GPU at 306 MHz throughout; back to back it runs at 1020 MHz throughout.

| size | pace | kernel median / p95 / max | host call median |
|---|---|---|---|
| 640x400 | 30 fps | 0.149 / 0.151 / 0.153 ms | 0.027 ms |
| 640x400 | back to back | 0.055 / 0.057 / 0.096 ms | 0.003 ms |
| 1280x800 | 30 fps | 0.546 / 0.549 / 0.552 ms | 0.025 ms |
| 1280x800 | back to back | 0.196 / 0.199 / 0.349 ms | 0.003 ms |

**The normals** (docs/methods/normals.md): measured as the organized points, on the organized
points of the same frame, with the starting parameters, a 7x7 window, 25 points and 75°
[M 2026-11-14, 12:40 to 12:42]: 126118 valid pixels at 640x400 and 504939 at 1280x800, 99.5 %
and 99.8 % of them with a normal. Paced at 30 fps the governor keeps the GPU at 306 MHz
throughout; back to back it runs at 1020 MHz throughout. Choosing the eigenvector's column by
comparison instead of at run time cut the kernel's stack from 88 to 40 bytes [B 2026-11-14:
cuobjdump -res-usage] and changed the time by about +1 % [M 2026-11-14, 13:21 to 13:23: 0.772 ms
and 2.969 ms back to back], so it was not kept.

| size | pace | kernel median / p95 / max | host call median |
|---|---|---|---|
| 640x400 | 30 fps | 2.478 / 2.486 / 2.494 ms | 0.020 ms |
| 640x400 | back to back | 0.763 / 0.766 / 0.781 ms | 0.004 ms |
| 1280x800 | 30 fps | 9.488 / 9.499 / 9.509 ms | 0.019 ms |
| 1280x800 | back to back | 2.934 / 2.938 / 2.949 ms | 0.005 ms |

**The frame's pyramid** (docs/methods/frame-pyramid.md): `frame_pyramid::build()`, its seven
kernels timed together by CUDA events, and the host's call, over 20 s per case after 5 s, on the
frame of the organized points and the normals, range 0.30 m to 1.0 m, the normals' starting
parameters [M 2026-11-14, 16:18 to 16:23]; valid pixels at level 0, 1 and 2: 126118, 31441 and
7815 at 640x400, 504939, 126058 and 31425 at 1280x800. In the same runs level 0's stages alone,
the organized points and their normals on buffers of their own, and the normals of level 1 and 2
alone on the pyramid's points; the rest of the build, level 0's intensity and the two
combinings, is the difference. Paced at 30 fps the governor keeps the GPU at 306 MHz throughout;
back to back it runs at 1020 MHz throughout.

| size | pace | build median / p95 / max | host call median | level 0's stages | normals of level 1 and 2 | the rest |
|---|---|---|---|---|---|---|
| 640x400 | 30 fps | 3.845 / 3.856 / 3.868 ms | 0.085 ms | 2.622 ms | 0.922 ms | 0.30 ms |
| 640x400 | back to back | 1.198 / 1.203 / 1.211 ms | 0.029 ms | 0.811 ms | 0.286 ms | 0.10 ms |
| 1280x800 | 30 fps | 14.201 / 14.215 / 14.240 ms | 0.079 ms | 10.029 ms | 3.160 ms | 1.01 ms |
| 1280x800 | back to back | 4.439 / 4.446 / 4.489 ms | 0.029 ms | 3.102 ms | 0.972 ms | 0.37 ms |

The normals of level 1 and 2 cost 31 % to 35 % of level 0's stages [M 2026-11-14: the table,
0.972 / 3.102 ms to 0.286 / 0.811 ms], their share of the pixels, 31 % [S: 1/4 + 1/16];
combining the finer level's normals instead is a candidate, built only on a measured need
(docs/methods/frame-pyramid.md, "Normals at every level").

**The allocation near a frame's surface** (docs/methods/sparse-tsdf.md): the reset of the list's
count and the kernel, timed together by CUDA events, and the host's call, over 20 s per case
after 5 s, the same frame allocated again and again into a sparse TSDF of 1.0 GB, 239810 blocks
and 1048576 slots, that holds its blocks already, as a scan's frame finds
most of its blocks held [M 2026-11-14, 18:23 to 18:27, MAXN_SUPER by `nvpmodel -q`, no compiler
running]. The scenes of `tests/test_sparse_tsdf_allocation.cpp`: the tabletop box on its plane
from the first pose of its walk, at 2 mm voxels and the range 0.30 m to 1.0 m; a box of 1.2 m x
0.8 m x 1.0 m on a floor from 2.7 m, at 8 mm voxels and 0.4 m to 3.0 m; the far depth the
range's. Paced at 30 fps the governor keeps the GPU at 306 MHz throughout; back to back it runs
at 1020 MHz throughout (`/sys/class/devfreq/17000000.gpu/cur_freq`, sampled every 16 runs).

| size | voxel | valid pixels, blocks | pace | kernel median / p95 / max | host call median |
|---|---|---|---|---|---|
| 640x400 | 2 mm | 131065, 5180 | 30 fps | 0.563 / 0.569 / 0.576 ms | 0.041 ms |
| 640x400 | 2 mm | | back to back | 0.191 / 0.195 / 0.350 ms | 0.007 ms |
| 1280x800 | 2 mm | 524849, 5215 | 30 fps | 1.521 / 1.527 / 1.533 ms | 0.044 ms |
| 1280x800 | 2 mm | | back to back | 0.523 / 0.527 / 0.604 ms | 0.007 ms |
| 640x400 | 8 mm | 118421, 10877 | 30 fps | 1.561 / 1.569 / 1.576 ms | 0.041 ms |
| 640x400 | 8 mm | | back to back | 0.472 / 0.476 / 0.601 ms | 0.007 ms |
| 1280x800 | 8 mm | 473794, 10950 | 30 fps | 3.237 / 3.247 / 3.255 ms | 0.043 ms |
| 1280x800 | 8 mm | | back to back | 1.012 / 1.016 / 1.070 ms | 0.007 ms |

The first frame into a new, empty sparse TSDF, each right after the request's clearing, the
clock not controlled: medians of 10 of 0.670, 1.664, 1.724 and 3.469 ms, in the table's order of
size and voxel [M 2026-11-14, same runs]. The medians are 0.2 % to 5.6 % above those of the same
measurement with the hash at twice the pool's blocks, 524288 slots, and no far depth, which were
0.546, 1.513, 1.478 and 3.131 ms at 30 fps and 0.186, 0.522, 0.447 and 0.985 ms back to back
[M 2026-11-14, 15:43 to 15:47]; the slots' layout and the far depth's comparison were not timed
apart. Before its atomic swap the insertion reads the slot's key, and before its exchange the
slot's stamp, each by a relaxed atomic load, so that a block a frame finds held costs no atomic
(`src/fusion/block_hash.cuh`, `src/fusion/sparse_tsdf.cu`). On the code before the cap of blocks
per ray, with the hash at 524288 slots, the medians with both loads were 0.3 % to 3.5 % below
those of 15:43 [M 2026-11-14, 14:45 to 14:48], and the medians without both loads were 0.942,
3.758, 2.173 and 8.754 ms at 30 fps and 0.329, 1.428, 0.704 and 3.154 ms back to back [M
2026-11-14, 14:40 to 14:44]; at 1280x800 back to back the keys' load alone gave 0.900 and 1.888
ms, the stamps' alone 0.972 and 1.949 ms [M 2026-11-14, 14:50 to 14:54], so both are kept.

A frame whose every valid point lies past the far depth, which the allocation skips and counts:
the tabletop frame at 1280x800 with a far depth of 0.25 m, 524849 points skipped, takes 0.396 /
0.400 / 0.404 ms paced at 30 fps and 0.161 / 0.163 / 0.208 ms back to back; the same size with
every pixel invalid, the kernel's floor, 0.161 / 0.161 / 0.162 ms and 0.055 / 0.058 / 0.108 ms
(median / p95 / max) [M 2026-11-14, 18:49 to 18:51, as the table, MAXN_SUPER, 306 / 1020 MHz].
The compiler already gathers the skip's count per warp, one reduction of the warp's population
count [B 2026-11-14: `cuobjdump -sass` of `allocate_kernel`, VOTEU.ANY, POPC, RED.E.ADD], so it
is not gathered by hand; the difference between the two frames holds the read of the 6.3 MB of
points besides that count, not timed apart [?].

**The integration of a frame** (docs/methods/sparse-tsdf.md, "Integration, and the weight of an
observation"): `sparse_tsdf::integrate()`, the allocation near the frame's surface and the
integration's kernel together, timed by CUDA events, and the host's call; in the same runs the
allocation alone, so that the integration's kernel is the difference. 20 s per case after 5 s, the
same frame fused again and again into a sparse TSDF of 0.38 GB, 90636 blocks and 524288 slots,
that holds its blocks already, as a scan's frame finds most of its blocks held; the scenes, voxel
sizes and depth ranges of the allocation's table above [M 2026-11-15, 01:52 to 01:59, MAXN_SUPER
by `nvpmodel -q`, no compiler running, a scratch program; the GPU's clock sampled every 16 runs,
the warm-up included]. Paced at 30 fps the governor keeps the GPU at 306 MHz once warm; back to
back it runs at 1020 MHz once warm.

| size | voxel | blocks | pace | integrate() median / p95 / max | allocation alone, median | the integration's kernel | host call median |
|---|---|---|---|---|---|---|---|
| 640x400 | 2 mm | 5180 | 30 fps | 3.159 / 3.190 / 3.215 ms | 0.546 ms | 2.61 ms | 0.058 ms |
| 640x400 | 2 mm | | back to back | 1.038 / 1.049 / 1.095 ms | 0.188 ms | 0.85 ms | 0.012 ms |
| 1280x800 | 2 mm | 5215 | 30 fps | 4.256 / 4.549 / 4.780 ms | 1.518 ms | 2.74 ms | 0.052 ms |
| 1280x800 | 2 mm | | back to back | 1.392 / 1.397 / 1.445 ms | 0.524 ms | 0.87 ms | 0.012 ms |
| 640x400 | 8 mm | 10877 | 30 fps | 7.346 / 7.381 / 7.427 ms | 1.478 ms | 5.87 ms | 0.055 ms |
| 640x400 | 8 mm | | back to back | 2.414 / 2.428 / 2.449 ms | 0.449 ms | 1.97 ms | 0.012 ms |
| 1280x800 | 8 mm | 10950 | 30 fps | 9.288 / 9.676 / 11.550 ms | 3.136 ms | 6.15 ms | 0.044 ms |
| 1280x800 | 8 mm | | back to back | 3.047 / 3.064 / 3.219 ms | 0.988 ms | 2.06 ms | 0.012 ms |

The integration's kernel takes about the same at both sizes: it runs over the listed blocks'
voxels, 512 a block, and reads one pixel for each, so its time follows the blocks, 0.16 to
0.19 µs a block back to back and 0.50 to 0.56 µs paced [S: computed, the table: 0.85 ms / 5180 to
6.15 ms / 10950 blocks]. The allocation alone takes 0.2 % to 5.3 % less than in its table above,
which used a sparse TSDF of 1.0 GB [S: computed, the two tables' medians]. The kernels go through the list on a fixed grid of thread blocks
(`src/fusion/sparse_tsdf.cu`, `voxel_kernel_grid`): 24, the thread blocks of 512 the GPU holds at
once, against 96 and 384, `integrate()` back to back for 10 s after 3 s per case, medians in the
table's order of size and voxel: 1.037, 1.392, 2.416 and 3.048 ms at 24; 1.115, 1.516, 2.425 and
3.037 ms at 96; 1.151, 1.612, 2.614 and 3.384 ms at 384 [M 2026-11-15, 01:47 to 01:51, as the
table]. So 24 is kept: 0.4 % longer to 8.2 % shorter than 96, 7.6 % to 13.7 % shorter than 384.

**The extraction** (docs/methods/sparse-tsdf.md, "The extraction"): `sparse_tsdf::extract()`, the
call with its one wait, by `std::chrono::steady_clock`, on the walk of
`tests/test_scan_interface.cpp`, 240 poses 1.5° apart at 640x400, every frame tracked, fused by
the tracker into the tabletop object's live model (2 mm) or into a sparse TSDF of its full voxel
size (1 mm), at budgets of 272891920 bytes (S = 2^18), 1.0 GB (2^20) and 2.0 GB (2^21), after 1,
30, 120 and 240 frames. The method: a scratch program with the committed kernel, which took one
slot a thread block per step, and the candidates beside it in one copy of `sparse_tsdf.cu`; at
each count each candidate's points compared with the committed kernel's as sets, bit for bit, then
each called 9 rounds back to back in an order that turns each round, into a buffer of 2^21 points;
under the build lock, MAXN_SUPER by `nvpmodel -q`, the GPU at 1020 MHz at every round's start but
each count's first (306 to 816 MHz) [M 2026-11-16, 21:45:39 to 21:49:35; MemAvailable 4.01 GB
before each 2.0 GB run, the monitor's least 1754 MB; the kernel log empty after it]. Every
candidate gave the committed kernel's points and count in all 96 comparisons. Medians in ms at the four counts: the live model
2682, 4472, 7640 and 7932 blocks (96396, 160244, 280422 and 283737 points), the full model 12599,
21078, 37239 and 39253 blocks (426447, 696239, 1172426 and 1162601 points):

| model | S | one slot a step (before) | chunks, 3 blocks | chunks, no bound | list pass | blocks in use |
|---|---|---|---|---|---|---|
| live | 2^18 | 19.0 / 21.5 / 27.7 / 29.9 | 1.9 / 3.3 / 5.6 / 5.6 | 2.8 / 4.9 / 7.7 / 7.7 | 2.6 / 5.0 / 9.6 / 9.9 | 2.4 / 4.7 / 9.5 / 9.4 |
| live | 2^20 | 113 / 140 / 136 / 138 | 2.5 / 3.9 / 5.9 / 6.0 | 4.0 / 6.1 / 8.9 / 9.2 | 2.7 / 5.0 / 9.8 / 10.0 | 2.5 / 4.5 / 9.2 / 9.7 |
| live | 2^21 | 282 / 372 / 374 / 359 | 2.9 / 4.4 / 7.2 / 6.8 | 4.4 / 6.6 / 10.6 / 10.6 | 2.8 / 4.9 / 10.0 / 9.8 | 2.4 / 4.5 / 9.4 / 9.3 |
| full | 2^18 | 36.5 / 59.7 / 104 / 109 | 9.2 / 18.0 / 33.9 / 34.9 | 11.9 / 23.9 / 43.9 / 45.2 | 16.6 / 38.2 / 94.3 / 101 | 16.4 / 35.3 / 75.2 / 79.5 |
| full | 2^20 | 136 / 166 / 212 / 219 | 8.7 / 15.5 / 26.5 / 26.5 | 12.9 / 22.4 / 34.2 / 34.1 | 17.1 / 41.9 / 108 / 115 | 16.8 / 37.0 / 79.2 / 81.2 |
| full | 2^21 | 331 / 385 / 440 / 437 | 8.5 / 14.5 / 24.1 / 24.4 | 13.2 / 21.0 / 33.3 / 33.6 | 17.0 / 45.5 / 121 / 129 | 15.6 / 34.8 / 77.0 / 81.2 |

"Chunks, 3 blocks" had three barriers a chunk, the third after the list's blocks; "chunks, no
bound" is the same without its launch bounds' 3 blocks: 57 registers, 2 thread blocks a
multiprocessor. "List pass": a kernel of one thread per slot lists the held slots, then one thread
block for each entry. "Blocks in use": one thread block for each block number below the claims,
its slot from an array of 4 bytes a block, filled for the measurement by a pass over the slots of
0.14 to 1.0 ms, not counted. Both took their buffers outside the pool.

The landed kernel is "chunks, 3 blocks" with two barriers a chunk, the third not needed
(docs/methods/sparse-tsdf.md, "The sweep over the slots"); its machine code is the scratch
program's "two barriers" byte for byte (`cuobjdump -sass`, the encodings compared, its name aside)
[B 2026-11-16]. Timed as above back to back against the three barriers and the committed kernel
[M 2026-11-16, 23:14:25 to 23:16:39, under the build lock and the memory guard, MemAvailable 3676
to 3704 MB before each run and never below 2000 MB during them by the guard's log, the GPU at 1020
MHz at every round's start but each count's first two (306 to 816 MHz); the kernel log empty after
it], the same points and count in all 32 comparisons; medians in ms at the counts above:

| model | S | one slot a step (before) | three barriers | two barriers (landed) |
|---|---|---|---|---|
| live | 2^18 | 19.0 / 21.8 / 27.6 / 29.9 | 1.93 / 3.29 / 5.57 / 5.56 | 1.97 / 3.36 / 5.67 / 5.64 |
| live | 2^21 | 284 / 374 / 378 / 360 | 3.15 / 4.44 / 6.94 / 6.85 | 3.15 / 4.48 / 7.21 / 6.85 |
| full | 2^18 | 36.4 / 59.8 / 103 / 108 | 9.21 / 18.1 / 33.7 / 34.9 | 9.32 / 18.2 / 34.0 / 35.1 |
| full | 2^21 | 332 / 384 / 438 / 435 | 8.46 / 14.5 / 24.1 / 24.4 | 8.54 / 14.8 / 24.6 / 24.9 |

The landed kernel took 0 % to 3.9 % longer than with the third barrier (its least times 0 % to
1.4 %) [S: computed]: too little to keep a barrier the code does not need. The committed kernel's
time followed S; the landed one's follows the blocks held, 0.6 to 1.2 µs a block [S: computed,
the medians over the blocks], and S adds about 1 ms at 2^21: 1.97 ms at S = 2^18 and 3.15 ms at
2^21 for 2682 blocks. It is 3.1 to 9.6 times shorter than before at S = 2^18 and 17 to 90 times
at 2^21 [S: computed]. Larger models are unmeasured: at the 0.64 to 0.94 µs a block of the
counts from 7640 blocks up, the 46668 blocks the demo window's live model held [M 2026-11-15]
would take about 30 to 44 ms [S: computed].

**The live model** (docs/methods/tracking.md, "The live model"): `live_model_taker::take()`, the
extraction at the live model's spacing into a buffer of 200000 points, the copies and the
conversion to model points or the merge with the last live model on the host, by
`std::chrono::steady_clock`; MAXN_SUPER by `nvpmodel -q`, under the build lock and the memory
guard, the kernel log empty after each run.

* **A scene of the demo window's size** [M 2026-11-16, 23:30 and 23:43, scratch programs]: the
  tabletop object's live model (2 mm, 0.25 to 0.6 m) at its own 2.0 GB, fused at known poses from 21
  frames of 640x400 over a 3 m x 3 m textured plane with the tabletop box, the camera 0.5 m above
  it on a grid 0.5 m apart: 46849 blocks, against the 46668 of the demo window's tabletop live model
  [M 2026-11-15]. A live model of every block, the extraction, the copy and the conversion, the
  medians of 15 back to back with the GPU at 1020 MHz but the first rounds:

  | spacing | points found | extraction | copy | conversion | whole |
  |---|---|---|---|---|---|
  | 1 | 1479534, 200000 written | 30.181 ms | 0.273 ms | 1.633 ms | 32.168 ms |
  | 2 | 369600, 200000 written | 23.190 ms | 0.286 ms | 1.686 ms | 25.231 ms |
  | 4 | 92703 | 22.809 ms | 0.146 ms | 0.584 ms | 23.536 ms |
  | 8 | 23044 | 23.945 ms | 0.064 ms | 0.145 ms | 24.156 ms |

  The spacing's rule takes 4 there; with the path's 11.2 ms at 640x400 ("The replay's frame time") a
  frame that took one would last 34.7 ms, over the frame's 33.3 ms, so the live model extracts only
  the blocks touched since the last [S: computed]. With that, the GPU brought to 1020 MHz before
  each take: the first take, every block at spacing 1, found too many in 31.370 ms; the second,
  every block at 4, 27.780 ms; then, after each 15 frames along a line 1 cm a frame over the scanned
  plane, 5.511 to 6.971 ms for 4734 to 6879 blocks touched and 9340 to 13852 new points, 92703
  published, which an extraction of every block at 4 matched bit for bit as sets.
* **The replay's `live_model` column** [M 2026-11-17, 00:00:19 to 00:00:28]: `build/apps/replay
  --times` on the walk of `tests/test_scan_interface.cpp` forth and back, 479 frames of 640x400
  written as a sequence by a scratch program, the tabletop object class's own set (2.0 GB), every
  frame tracked and fused. 11 frames took a live model: the first, every block at spacing 1,
  8.261 ms; the one after the spacing rose to 2, every block, 14.492 ms; the 9 others, the blocks
  touched since, 5.792 to 6.712 ms. Those frames lasted 15.260 to 26.445 ms against a median
  10.497 ms of the others, and the frame after each a median 11.133 ms; the other frames' column
  printed 0.002 ms at most; a frame's stages added up to its total within 0.032 ms. The replay
  without `--times` took 12.283 ms a frame [M 2026-11-17, 00:00:28 to 00:00:37].

**The live view** (docs/methods/tracking.md, "The live view"): MAXN_SUPER by `nvpmodel -q`, under
the build lock and the memory guard, nothing else running; the kernel log showed only a keyboard
plugged in and out at 11:02, benign ("Kernel log") [M 2026-11-17, 10:25 to 11:06].

* **The colors in the cast**: the timing program of "The ray cast" on its scenes and its sparse
  TSDF of 273 MB, 10 s per case after 3 s, the cast of cf8a02a against the cast with the colors at
  level 0 (`sparse_tsdf::ray_cast()`), back to back with cf8a02a's run before and after, then paced:

  | size | voxel | pace | cf8a02a, median | with the colors | change |
  |---|---|---|---|---|---|
  | 640x400 | 2 mm | back to back, 1020 MHz | 4.533, 4.551 ms | 4.579 ms | +0.8 % |
  | 1280x800 | 2 mm | back to back, 1020 MHz | 13.071, 13.130 ms | 13.191 ms | +0.7 % |
  | 640x400 | 8 mm | back to back, 1020 MHz | 6.116, 6.117 ms | 6.125 ms | +0.1 % |
  | 1280x800 | 8 mm | back to back, 1020 MHz | 17.231, 17.232 ms | 17.180 ms | -0.3 % |
  | 640x400 | 2 mm | 30 fps, 306 MHz | 14.311 ms | 14.389 ms | +0.5 % |
  | 1280x800 | 2 mm | 30 fps, 612 MHz | 21.365 ms | 21.458 ms | +0.4 % |
  | 640x400 | 8 mm | 30 fps, 306 MHz | 19.387 ms | 19.399 ms | +0.1 % |
  | 1280x800 | 8 mm | 30 fps, 816 MHz | 21.484 ms | 21.517 ms | +0.2 % |

  The color carried through the march as one 32-bit word: 64 registers a thread and 24 bytes of
  stack against cf8a02a's 16 [B 2026-11-17: `cuobjdump -res-usage`]. Carried as three bytes it
  took 32 bytes of stack and 4.622, 13.320, 6.173 and 17.398 ms back to back, 0.9 % to 1.8 % above
  cf8a02a, so the word is kept.
* **The shading, the copy and the encoding**: a scratch program on the tabletop sphere's scene of
  "The ray cast" cast from its walk's first pose at 640x400 (2 mm, 273 MB), offering a view to the
  encoder (`live_view_encoder::offer()`) as soon as the last was published, 10 s after 3 s, back to
  back and paced at 30 fps, the GPU at 306 MHz in both:

  | pace | shading and copy on the GPU, median / p95 | the offer's host call | offer to publication, median / p95 | `imencode` alone | bytes |
  |---|---|---|---|---|---|
  | back to back | 0.294 / 0.305 ms | 0.021 ms | 2.278 / 2.301 ms | 1.910 ms | 31222 |
  | 30 fps | 0.908 / 0.939 ms | 0.061 ms | 5.471 / 5.681 ms | 4.594 ms | 31222 |

  The shading and the copy by CUDA events around the offer on the stream; the offer to its
  publication from the offer's return to the listener's call, the encoder's wait for the copy,
  `imencode` at quality 85 and the swap; `imencode` alone on the decoded view on the program's own
  thread, the same seconds after. Paced, the CPU's clock falls between the frames (cpu0 sampled at
  729 to 1420 MHz) and the encoding takes 2.4 times as long [S: computed]: still a sixth of a
  frame's 33.3 ms, so the encoder keeps up with the camera. The test's walk around the tabletop
  box gave 26.2 KB a view [M 2026-11-17: `tests/test_scan_interface.cpp`].
* **The replay's columns**: `build/apps/replay` on the walk of `tests/test_scan_interface.cpp`
  forth and back, 479 frames at 640x400 written as a sequence by a scratch program, the tabletop
  object class's set with a budget of 600 MB from a copy of a kept scan's parameters, every frame
  tracked and fused, at cf8a02a (the main tree's build) and at the change. With `--times`: the
  `live_view` column a median 0.182 ms, 0.153 to 1.456 ms, so every frame's view was taken; the
  `ray_cast` column, the cast after the fusion, 5.769 ms against 5.716 ms at cf8a02a, the cast
  before the registration; a frame's total 10.627 ms against 10.332 ms. Without `--times`, three
  rounds alternating: 10.086, 10.078 and 10.076 ms a frame at cf8a02a, 10.060, 9.956 and 9.934 ms
  with the change. The trajectory was the same file, byte for byte (SHA-256), in all ten runs
  [M 2026-11-17, 11:03 to 11:05].
* **On the user's kept sequence** (`~/3dscanner-scans/20261117T084909.343073Z`, the first scan
  with the service, 1900 frames at 640x400 of the tabletop object class), replayed with `--times`
  at the change and at cf8a02a with a copy of its parameters at a budget of 1.0 GB, as the 2.0 GB
  did not fit above the memory guard's floor [S: computed, 3.2 GB available less 2.0 GB is below
  its 1.2 GB]: both 1874 frames tracked, 26 lost, the same
  trajectory byte for byte; the `live_view` column a median 0.204 ms (0.170 to 3.100 ms), every
  tracked frame's view taken, at most 0.005 ms on a lost frame; the `ray_cast` column 12.765 ms
  against 12.728 ms and 0 on every lost frame; a frame's total 24.733 ms against 24.431 ms; 28.057
  and 28.015 ms a frame by the last line, the 2.4 GB sequence read from the disk [M 2026-11-17,
  11:06 to 11:08]. The ray cast and the registration take 12.7 ms and 9.2 ms there against 5.8 ms
  and 2.0 ms on the synthetic walk, which is `per-stage-times-from-replay`'s to explain.

**The ray cast** (docs/methods/sparse-tsdf.md, "The ray cast"): `sparse_tsdf::ray_cast()`, the
clearing of its tiles and its two kernels timed together by CUDA events, and the host's call, over
20 s per case after 5 s, the same cast again and again from the first pose of the walk, into a
`model_surface` of the cast's size. The scenes of `tests/test_sparse_tsdf_ray_cast.cpp`: the
tabletop sphere on its plane fused from its walk of 12 frames at 640x400 at 2 mm voxels, range
0.30 m to 1.0 m, 6779 blocks; the large sphere on its plane of 3 m x 3 m from its walk at 8 mm,
0.4 m to 3.0 m, 11284 blocks; a sparse TSDF of 273 MB, 65536 blocks and 262144 slots [M 2026-11-15,
22:30 to 22:49, the cast whose normal reads each voxel once (`ray-cast-normal-reads-once`),
MAXN_SUPER by `nvpmodel -q`, the governor `nvhost_podgov` from 306 to 1020 MHz, under the build
lock, a scratch program; `/sys/class/devfreq/17000000.gpu/cur_freq` sampled every 16 runs; the
kernel log empty after it]. Back to back the GPU runs at 1020 MHz; paced at 30 fps the governor
kept it at 306 MHz at 640x400, at 612 and 816 MHz at 1280x800.

| size | voxel | valid pixels | pace | median / p95 / max | GPU clock | host call median |
|---|---|---|---|---|---|---|
| 640x400 | 2 mm | 130678 | 30 fps | 14.310 / 14.355 / 14.443 ms | 306 MHz | 0.041 ms |
| 640x400 | 2 mm | | back to back | 4.550 / 4.563 / 4.581 ms | 1020 MHz | 0.014 ms |
| 1280x800 | 2 mm | 523234 | 30 fps | 21.382 / 21.482 / 21.510 ms | 612 MHz | 0.024 ms |
| 1280x800 | 2 mm | | back to back | 13.126 / 13.143 / 13.166 ms | 1020 MHz | 0.015 ms |
| 640x400 | 8 mm | 117477 | 30 fps | 19.383 / 19.467 / 19.517 ms | 306 MHz | 0.030 ms |
| 640x400 | 8 mm | | back to back | 6.115 / 6.141 / 6.994 ms | 1020 MHz | 0.014 ms |
| 1280x800 | 8 mm | 470646 | 30 fps | 21.494 / 21.523 / 21.547 ms | 816 MHz | 0.020 ms |
| 1280x800 | 8 mm | | back to back | 17.237 / 17.259 / 17.286 ms | 1020 MHz | 0.015 ms |

The baseline, the committed cast of `ray-cast-faster-by-measurement` (a68c352), whose normal
interpolates its six ends, in the same session before and after the cast above: back-to-back
medians 6.314, 17.967, 7.735 and 21.402 ms in the table's order, and 6.341, 17.966, 7.742 and
21.413 ms at the session's end; paced 20.039 ms at 306 MHz, 22.394 ms at 816 MHz, 18.547 ms at
408 MHz and 21.306 ms at 1020 MHz [M 2026-11-15, 22:30 to 22:49]. So the normal's reads once take
27.9 %, 26.9 %, 20.9 % and 19.5 % off the back-to-back medians; paced, 28.6 % at 640x400 and 2 mm,
where the clock stayed at 306 MHz, while the governor lowered the clock by one or two steps in the
other three cases, where the paced medians moved by -4.5 % to +4.5 % [S: computed]. Where that
time comes from and what was measured and not built: docs/methods/sparse-tsdf.md, "The ray cast".

The march through every block, before the depth interval, took 8.809, 28.046, 10.074 and 31.111 ms
back to back in the table's order, and paced 21.065 ms at 408 MHz, 27.931 ms at 1020 MHz, 19.659 ms
at 510 MHz and 31.098 ms at 1020 MHz [M 2026-11-15, 10:11 to 10:15]: the depth interval took 23 % to
36 % off the back-to-back medians of the cast with it then, 6.354, 17.966, 7.747 and 21.402 ms [M
2026-11-15, 10:15 to 10:18], and paced 4.5 % and 5.4 % off at 640x400 and 20 % and 31 % at 1280x800,
the governor lowering the clock by a step in three of the four cases [S: computed]. Its host call,
for the clearing and the second launch, takes 0.009 to 0.010 ms more back to back and 0.013 to 0.024
ms more paced: its medians before were 0.011, 0.007, 0.012 and 0.007 ms paced and 0.005 to 0.006 ms
back to back [M 2026-11-15, 10:11 to 10:18, those runs]. Tiles of 16 x 16 pixels gave 6.365, 17.955,
7.766 and 21.378 ms, within 0.3 % of the cast with tiles of 8 x 8 then [M 2026-11-15, 10:18 to
10:22], so the tiles stay at 8 x 8. The depth interval reads every slot of the hash: the clearing of
the tiles at 1280x800 and its kernel over empty slots took 0.045 ms for 2^18 slots, the table's, and
0.260 ms for 2^21, the object classes' sparse TSDF at 2.0 GB, which therefore adds about 0.22 ms to
the table's casts [M 2026-11-15, 11:08, back to back, a scratch program; S: computed]. Where the
rest of the time goes and what was measured and not built: docs/methods/sparse-tsdf.md, "The ray
cast". Before the depth interval, the march's cache of the last other block a voxel was found in
took 11 % to 16 % off the back-to-back medians, and a jump over a block without voxels to the first
sample past its exit, in place of stepping through its samples' indices, made them 7 % to 10 %
longer, 9.569, 30.090, 11.118 and 33.717 ms back to back, the kernel at 68 registers a thread
against 64 [M 2026-11-15, 07:00 to 07:04; B 2026-11-15: `cuobjdump -res-usage`], so it is not kept.

Back to back, the stages measured so far add up to 9.4 ms and 12.4 ms a frame at 640x400 and 27.0 ms
and 32.7 ms at 1280x800, at 2 mm and 8 mm voxels: the upload on an idle stream, the pyramid, the ray
cast, the registration of the tabletop pair (below) and the integration [S: computed, the medians of
this section; the stages' scenes differ]. At 1280x800 the path built so far fits a frame's 33 ms at
the full clock at both voxel sizes, with 6.3 ms to spare at 2 mm and 0.6 ms at 8 mm, before the slot
scan's 0.22 ms at 2.0 GB [S: computed, 33.3 ms less the sum].

**The registration of one frame** (docs/methods/registration.md): the host's call of
`registration::align()` by the full method, both terms coarse to fine with the tabletop object's
starting values, on the tabletop object's pair of the registration's tests
(`tests/registration_cases.hpp`: a box and a sphere on a plane from 0.5 m, moved by 1.5° and
10.8 mm), every run converged; the surface's gradient kernel alone on its three levels, a copy of
the tree's kernel timed by CUDA events; and the call with one iteration on level 0 alone, its
gradient kernel included. The medians of 20 s per case
after 5 s [M 2026-11-15, 00:01 to 00:06, MAXN_SUPER by `nvpmodel -q`, no compiler running, a
scratch program; `/sys/class/devfreq/17000000.gpu/cur_freq` sampled every 16 runs]. Paced at 30 fps
the governor keeps the GPU at 306 MHz throughout; back to back it runs at 1020 MHz throughout.

| size | pace | one frame: call median / p95 / max | systems | gradient kernel, three levels | one iteration on level 0 |
|---|---|---|---|---|---|
| 640x400 | 30 fps | 6.835 / 6.990 / 8.134 ms | 15 | 0.658 ms | 2.165 ms |
| 640x400 | back to back | 2.311 / 2.362 / 4.401 ms | 15 | 0.219 ms | 0.530 ms |
| 1280x800 | 30 fps | 19.529 / 19.662 / 20.246 ms | 11 | 2.442 ms | 6.356 ms |
| 1280x800 | back to back | 6.484 / 6.558 / 6.891 ms | 11 | 0.803 ms | 1.893 ms |

Paced at 30 fps, at the governor's 306 MHz, the frame's pyramid (above, 14.2 ms) and its
registration take 33.7 ms at 1280x800, more than a frame's 33 ms before the ray cast and the
fusion, and 10.7 ms at 640x400 [S: computed, the medians]. The stream size is settled by the rule
of the replay's measurement of the whole path, not before (the user's decision, 2026-11-14, late
evening; `per-stage-times-from-replay`).

**One iteration of the point-to-plane term** (docs/methods/registration.md), before the color term
joined, kept as the evidence of the decision that live tracking sums in float within a warp (the
user's, 2026-11-14; docs/methods/registration.md, "Precision"): the host's call of
`registration::align()` with one iteration, which queues the system's kernel and the totals' kernel,
copies the 280 bytes of the sums, waits for them and solves the system, on level 0 of the same pair
(fitness 0.92 to 0.93 at the start), the medians of 20 s per case after 5 s [M 2026-11-14, 19:54 to
19:58, MAXN_SUPER by `nvpmodel -q`, no compiler running, a scratch program;
`/sys/class/devfreq/17000000.gpu/cur_freq` sampled every 16 runs]. Paced at 30 fps the governor
keeps the GPU at 306 MHz throughout; back to back it runs at 1020 MHz throughout.

| size | pace | call median / p95 / max |
|---|---|---|
| 640x400 | 30 fps | 1.649 / 1.690 / 1.909 ms |
| 640x400 | back to back | 0.355 / 0.363 / 0.670 ms |
| 1280x800 | 30 fps | 4.348 / 4.401 / 4.604 ms |
| 1280x800 | back to back | 1.209 / 1.232 / 1.759 ms |

With each warp adding in double instead (docs/methods/registration.md, "Precision"), the same
runs took 13.718, 3.988, 22.813 and 15.712 ms, the last paced case at a GPU clock of 714 MHz; with
each warp's 35 sums converted to double in its own lane, 1.49 and 4.34 ms back to back
[M 2026-11-14, as the table, but 1 s per case after 0.2 s, at 19:50].

**The whole per-frame path** (docs/methods/tracking.md, "The path of a frame"): the host's call of
`tracker::track()` to the end of the stream's work, the frame's fusion included, and the call
alone, which returns after the registration's last wait with the fusion queued; 20 s per case
after 5 s, the first 60 poses of the walk around the tabletop box of `tests/test_tracking.cpp`
(1.5° apart) played forth and back, every frame tracking; the tabletop object class's values (the
range 0.25 m to 0.6 m, 2 mm voxels, e_track 5 mm); a sparse TSDF of 273 MB, 65536 blocks and 262144
slots, holding 5659 to 5860 blocks, since frames of 1280x800 and the object classes' 2.0 GB would
not fit the memory free beside the other jobs that evening, about 3.4 GB [?: the orchestrator's
reading], and the cast's scan of the slots adds about 0.22 ms at 2.0 GB ("The ray cast")
[M 2026-11-15, 21:20 to 21:22, MAXN_SUPER by `nvpmodel -q`, under the build lock, a scratch
program on the tree of `track-frames-against-the-model`, whose ray cast is a68c352's, before
`ray-cast-normal-reads-once`; `/sys/class/devfreq/17000000.gpu/cur_freq` sampled every 16 frames;
the kernel log empty after it].

| size | pace | path median / p95 / max | call median | GPU clock |
|---|---|---|---|---|
| 640x400 | 30 fps | 20.897 / 25.355 / 31.104 ms | 19.828 ms | 510 to 918 MHz, median 612 MHz |
| 640x400 | back to back | 12.497 / 13.838 / 20.141 ms | 11.764 ms | 1020 MHz |
| 1280x800 | 30 fps (1) | 34.875 / 37.645 / 55.159 ms | 33.771 ms | 1020 MHz |
| 1280x800 | back to back | 34.933 / 37.541 / 50.919 ms | 33.820 ms | 1020 MHz |

(1) A frame takes longer than 33.3 ms, so the pace never waits: the paced run is a run back to
back.

At 640x400 the path fits a frame's 33.3 ms, with 12 ms to spare paced at the governor's clock and
21 ms back to back; at 1280x800 it takes 1.6 ms more than a frame at the median at the full clock
[S: computed, the medians]. The stages measured apart above add up to 11.2 ms and 31.8 ms back to
back at 2 mm, 10 % and 9 % below the path's medians [S: computed]: the path's frame and scene are
other ones, and its host adds the check and the pose. The fusion runs 0.7 to 1.1 ms after the call
returns [S: computed, the path's median less the call's]. These are a first number for
`per-stage-times-from-replay`, which settles the stream size on real sequences.

**The "too close" notice** (docs/methods/tracking.md, "The scan's loop"; `too_close()`,
`src/scan/scan_interface.hpp`): one pass over a frame's depth on the host, by the scan thread before
tracking, on frames of random depth, a tenth of their pixels without depth; a scratch program
pinned to cpu0 by `taskset -c 0`, each size called back to back for 10 s, the first 100 calls not
recorded [M 2026-11-16, 07:48:24 to 07:48:44, MAXN_SUPER, cpu0 at 1728 MHz by `scaling_cur_freq`,
under the build lock, no compiler running]:

| size | calls | median | p95 | max |
|---|---|---|---|---|
| 640x400 | 117931 | 0.084 ms | 0.084 ms | 0.211 ms |
| 1280x800 | 29601 | 0.335 ms | 0.340 ms | 0.550 ms |

Counted in 64 bits with a 32-bit compare the same pass took 0.176 ms and 0.704 ms (medians); counts
of 32 bits a row and a 16-bit compare let the compiler vectorize it [M 2026-11-16, 07:47:14 to
07:47:34 and a scratch comparison of the three loops at 1280x800: 0.704, 0.408 and 0.334 ms].

**The replay's frame time** (`build/apps/replay`; docs/methods/tracking.md, "The stages' times"),
with and without `--times`: the walk around the tabletop box of `tests/test_scan_interface.cpp`
(240 poses 1.5° apart) played forth and back, 1440 frames at 640x400 and 720 at 1280x800, written
as sequences by a scratch program, every frame tracked and fused; the tabletop object class's set
with a budget of 1.0 GB (10^9 bytes) from a parameters file the scratch program wrote; each size
replayed without `--times`, with it, and without again; MAXN_SUPER by `nvpmodel -q`, under the
build lock, nothing else running; the GPU clock sampled every 0.5 s from `cur_freq` over each
whole run, refinement included [M 2026-11-16, 10:18:55 to 10:21:45; the kernel log empty after
it]. "A frame" is the last line's: the scanning phase's time over the frames taken, the source's
reads included.

| size | `--times` | a frame | GPU clock samples |
|---|---|---|---|
| 640x400 | no (1) | 12.738 ms | 42 of 49 at 1020 MHz |
| 640x400 | yes | 15.071 ms | 29 of 55 at 1020 MHz, 13 at 918, 10 at 816 |
| 640x400 | no | 11.172 ms | 39 of 44 at 1020 MHz |
| 1280x800 | no | 36.961 ms | 56 of 63 at 1020 MHz |
| 1280x800 | yes | 37.813 ms | 57 of 64 at 1020 MHz |
| 1280x800 | no | 35.758 ms | 52 of 61 at 1020 MHz |

(1) Part of its sequence came from the disk: the 3.5 GB of 1280x800 written after it had pushed it
out of the page cache [?: inferred from the second run].

The time lines' medians in the runs with `--times`, in ms:

| size | room | too_close | upload | pyramid | ray_cast | registration | check | fusion | writer | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 640x400 | 0.001 | 0.230 | 0.349 | 1.587 | 5.989 | 2.095 | 0.002 | 0.786 | 0.015 | 11.068 |
| 1280x800 | 0.003 | 0.821 | 1.170 | 4.825 | 15.526 | 5.380 | 0.004 | 1.142 | 0.019 | 29.072 |

The waits cost 3.9 ms a frame at 640x400 (15.071 against 11.172 ms, 35 %) and 0.9 to 2.1 ms at
1280x800 (2 % to 6 %) [S: computed]: they keep the host's "too close" pass and upload from running
beside the frame before's fusion, and with the GPU idle between stages the governor lowered its
clock (29 of 55 samples at 1020 MHz against 39 of 44). So the time lines give each stage's time,
and the frame time of a scan without them is the last line's. At 1280x800 a time line's total,
29.1 ms, lies 8.7 ms below a frame's 37.8 ms [S: computed, the tables]: the source's read of 5.12 MB a frame, outside the
per-frame path, came from the disk, the sequence being larger than the page cache [?: inferred, not
timed by itself]. A stage's time differs between frames by up to several times its median (the
registration's largest 17.7 ms and 33.1 ms against its medians 2.095 ms and 5.380 ms), the largest
difference between a frame's total and its stages' sum was 0.027 ms, and `room` printed 0.000 in 2
of the 1440 frames at 640x400, the smallest `check` 0.001 ms [M 2026-11-16: the time lines of the
runs with `--times` above]. The "too close" pass took 0.230 ms and 0.821 ms on the scan
thread, against the 0.084 ms and 0.335 ms above on cpu0 at 1728 MHz [?: the scan thread is not
pinned, its core's clock not recorded].

## Refinement

What refinement takes on the board after a scan (docs/methods/refinement.md), measured by scratch
programs: the pose graph's solve on the host's CPU, on graphs built as `tests/test_pose_graph.cpp`
builds them, and the fusion of the sequence on the GPU, on the walk `tests/test_sequence_fusion.cpp`
writes.

**The pose graph's solve** (`optimize_pose_graph()`, `src/refinement/pose_graph.hpp`), on the graph
of `check_300_keyframes()` in `tests/test_pose_graph.cpp`:

* **The graph**: the known poses are `walk_around()` of `tabletop_walk()` with 300 poses, one lap
  0.40 m to 0.50 m around the tabletop box (`tests/synthetic_source.hpp`). Each keyframe k is the
  target of three edges, to the sources (k + 1), (k + 10) and (k + 25) modulo 300, 900 edges in
  that order, all offsets of 1 first. Each edge's transform is exp(-ε^) T_k^-1 T_source, ε its error,
  `edge_noise(e, 1)` of its position e: 1 mrad and 1 mm times sines of e on each axis. Its
  information is diag(1e6, ..., 1e6), standard deviations of 1 mrad and 1 mm. The start is node 0
  at its known pose and node k at start_(k-1) T_(k-1)^-1 T_k exp(d^), d a tenth of `ring_drift()`:
  a turn of 0.015° about (0.3, 0.8, -0.5) and a shift of 0.15 mm along (1, -0.5, 0.3), each axis
  normalized. The start lies 1.71° and 29 mm off; max_iterations is 100.
* **The method**: a scratch program, built with g++ 11.4 `-O3 -DNDEBUG` against the Release
  build's `librefinement.a`, builds the graph once and times each call by `std::chrono::steady_clock`.
  Calls run for 5 s unrecorded, then 20 s recorded, one thread pinned to cpu0 by `taskset -c 0`.
  The run holds the build lock, so no compiler runs beside it.

It converged after 4 iterations, 64 runs: median 314.3 ms, p95 316.1 ms, max 316.8 ms
[M 2026-11-15, 01:21:42 to 01:22:07, MAXN_SUPER by `nvpmodel -q`, cpu0 at 1728 MHz after the warm-up
and at the end by `scaling_cur_freq`]. The undamped Gauss-Newton step that confirms the end
(docs/methods/refinement.md, "The end") adds one factorization and solve: without it the same run
took 255.2 / 256.9 / 257.7 ms [M 2026-11-15, 01:06:55 to 01:07:20], so it costs 59 ms, a fifth of
the four iterations' time. Graphs of other sizes and shapes are unmeasured.

**The fusion of the sequence** (`fuse_sequence()`, `src/refinement/sequence_fusion.hpp`), on the
walk of `tests/test_sequence_fusion.cpp`: 30 frames of 640x400 in a sequence of 38.4 MB
[M 2026-11-16: `ls -l`, 38402025 bytes], 24 fused at the tabletop object's full voxel size of 1 mm
and 6 skipped as tracking loss, into a sparse TSDF of 273 MB, 863311 points extracted
(docs/methods/refinement.md, "Accuracy"). Each call does all of it: it opens the sequence, which
stands in the page cache, requests its buffers, fuses, reads the pool's usage, extracts twice,
copies the points to the host and frees it all.

* **The method**: a scratch program, built with g++ 11.4 `-O3 -DNDEBUG` against the Release build's
  libraries, writes the sequence once and times each call by `std::chrono::steady_clock`, back to
  back for 25 s, the first 3 calls not recorded. The run holds the build lock, so no compiler runs
  beside it.

The first call took 0.349 s; the next 68 recorded calls a median of 0.362 s, p95 0.374 s, max
0.376 s, min 0.301 s [M 2026-11-16, 07:28:30 to 07:28:55, MAXN_SUPER by `nvpmodel -q`, the GPU's
clock 306 to 1020 MHz at a call's start (`/sys/class/devfreq/17000000.gpu/cur_freq`); no kernel
log line of usb, uvc, nvgpu or Xid], 15.1 ms a fused frame with everything else [S: computed,
0.362 s / 24]. Its steps are not timed apart, and a scan's longest sequence, its reading from the
drive past the page cache and the large object's full model are unmeasured. These times predate
the extraction's chunks ("The extraction" above), whose two extractions per call now take less;
the call is not timed again [?].

## Separation

What separation's gravity and proposal take on the board after refinement
(docs/methods/separation.md), on the host's CPU.

**The proposal** (`propose_separation()`, `src/separation/proposal.hpp`) and **gravity's estimate**
(`estimate_gravity()`, `src/separation/gravity.hpp`) on the large scene of
`tests/test_separation.cpp`, `large_scene()` built with the noise's seed 2: 2 048 582 points, the large
box among a cabinet, a crate and a ball on a floor of 4.2 m x 4.2 m at 4 mm voxels, its walk of 1200
poses, a lap of 40 s at 30 frames a second, and the synthetic accelerometer's 7994 samples at 200 Hz
along it. A scratch program, built with g++ 11.4 `-O3 -DNDEBUG` against the Release build's
`libseparation.a`, builds the model once and times each call by `std::chrono::steady_clock`: 5 s
unrecorded, then 20 s recorded, one thread pinned to cpu0 by `taskset -c 0`, holding the build lock,
so no compiler runs beside it. Every call estimated gravity and proposed the right box.

44 calls recorded [M 2026-11-17, 07:16:59 to 07:17:25, MAXN_SUPER by `nvpmodel -q`, cpu0 at
1574 MHz before the run and 960 MHz read after it ended, by `scaling_cur_freq`]: gravity's estimate
median 1.667 ms, p95 1.716 ms, max 1.750 ms; the proposal median 455.3 ms, p95 456.7 ms, max
457.5 ms. Before gravity, with up from the cameras' axes and the walk of 24 poses, the proposal took
a median of 502.5 ms [M 2026-11-15, 04:30:22 to 04:30:48, the same method]; the board's state moves
the time by a tenth: the same build of the module took 454.8 ms at 03:22 and 515.0 ms at 04:13
[M 2026-11-15, the same program; ?: the cause, not measured]. A copy of the module of 2026-11-15
with a clock between its steps gave, per call: the checks and the trajectory 6 ms, the window 58 ms,
the 4000 hypotheses 159 ms, the plane's fits and judgement 74 ms, the standing points and the sort of
their cells 159 ms, the pieces 2 ms, the object's hull and box 39 ms [M 2026-11-15, 04:30:48 to
04:30:58, the mean over every call of 10 s, the same conditions]. Other models are unmeasured.

## Connections

The four USB-A ports sit on a Realtek USB 3.2 hub, seen twice by the host controller
`tegra-xusb 3610000.usb` [M 2026-11-13, `lsusb -t`]:

| hub | bus | speed |
|---|---|---|
| `0bda:0489` (SuperSpeed half) | 2, port 1 | 10000M |
| `0bda:5489` (USB 2 half) | 1, port 2 | 480M |

The camera: `2-1.4`, bus 2, 5000M, with the third cable tried [M 2026-11-13]. With the two
cables before, it came up on bus 1 at 480M in every port; AGENTS.md §6.

Display: one connector, `card1-DP-1` (DisplayPort), nothing plugged in
[M 2026-11-13, `/sys/class/drm`].

## Permissions

* `/etc/udev/rules.d/99-obsensor-libusb.rules`, from OrbbecSDK's `shared/`, installed by the
  user 2026-11-13: Orbbec devices get `MODE 0666`, group `video`, and a symlink
  (`/dev/Gemini_2_L` for this one) [M 2026-11-13, `ls -l /dev/bus/usb/001/012`].
* The user is in `video`, `plugdev`, `adm` (the kernel log through `journalctl -k`)
  [M 2026-11-13, `groups`].

## Kernel log

The kernel's lines about this hardware, sorted. A harmful line, or one this table does not
list, is a stop condition (AGENTS.md §3).

| line (a pattern) | harmful or benign | what it means | evidence |
|---|---|---|---|
| `usb 1-2.N: new high-speed USB device ... tegra-xusb` for the camera | harmful for the work | the camera is on USB 2; AGENTS.md §6 | [M 2026-11-13] |
| `usb 2-1.N: Device not responding to setup address.` once, then `new SuperSpeed USB device` | benign [?] | the first address request failed and the kernel's retry succeeded; seen once on plugging in | [M 2026-11-13] |
| `usb 1-2.N: USB disconnect` 2 to 4 s after plugging in, two or three times, then stable | benign [?] | the camera re-enumerates after power-up; seen on every plug-in on USB 2, not on USB 3 | [M 2026-11-13] |
| `Failed to query (GET_INFO) UVC control 18 on unit 3: -32 (exp. 1).` | benign [?] | uvcvideo probes a control the camera does not answer; on every enumeration, before any program ran | [M 2026-11-13] |
| `Found UVC 1.10 device Orbbec(R) Gemini 2 L(TM) (2bc5:0673)`, `hid-generic ... hidraw0` | benign | enumeration; also each time a program closes the camera and uvcvideo takes it back | [M 2026-11-13] |
| `usb 1-N.M: new full-speed USB device number D using tegra-xusb`, then `input: ...` and `hid-generic ... USB HID ... Keyboard [...]`, and later that device's `usb 1-N.M: USB disconnect, device number D` | benign | a USB keyboard or other input device plugged in or out on the board's USB 2 bus, not the camera; the camera's own `USB disconnect` while it streams stays a stop condition (AGENTS.md §3) | [M 2026-11-15: usb 1-2.3, "SONiX QK65 KB", in at 12:00:09, out at 12:05:40, the camera on bus 2 at 5000M throughout (`lsusb -t`); the user's answer: they plugged the keyboard in and out] |
| `usb 2-1.N: USB disconnect, device number D`, then `Device not responding to setup address.` and `new SuperSpeed USB device number D using tegra-xusb`, while OrbbecSDK's libusb backend opens the camera without a stream (its firmware tool, `ob_device_firmware_update`) | benign [?] | the camera leaves USB 3 and comes back on each open by that tool; the camera's `USB disconnect` while it streams stays a stop condition (AGENTS.md §3) | [M 2026-11-15, 16:47 to 16:50, each open by the tool; the user's answer: benign while that tool opens it] |
| `uvcvideo 1-2.N:1.1:` or `2-1.N:1.1: Unknown video format 20343159-0000-0010-8000-00aa00389b71` | benign | uvcvideo does not know the depth format Y14 (the GUID starts with the FourCC `Y14 `); after a program closes the camera | [M 2026-11-13], [S] UVC GUID layout |
| `perf: interrupt took too long (A > B), lowering kernel.perf_event_max_sample_rate to R` | benign while a profiler samples the CPU (`nsys profile`, `perf record`) [?: seen once]; with no profiler running, a stop condition | the kernel's handling of the profiler's samples took longer than `kernel.perf_cpu_time_max_percent` (25) allows, so it lowers the samples per second it grants [?: the sysctl's documented purpose, not read here]; the lowered rate stays: 30000 two hours later | [M 2026-11-14: five lines at 19:50:48 to 19:50:51, the only ones of that boot; an `nsys profile` session of a scratch timing program began at 19:50:48 and recorded 1979 CPU samples (its report's tables `TARGET_INFO_SESSION_START_TIME`, `SAMPLING_CALLCHAINS`); about 130 runs of the hash experiment that day, with no profiler, logged none; `sysctl kernel.perf_event_max_sample_rate` 30000 at 21:47] |
