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
| camera | Orbbec Gemini 2 L, USB `2bc5:0673`, firmware 1.4.38 | [M 2026-11-13] `lsusb`, `orbbec_viewer --list` |
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

| size at 30 fps | intrinsics fx, fy, cx, cy | fps | `create_from_rgbd` with upload | points |
|---|---|---|---|---|
| 640x400 | 306.18, 306.14, 317.41, 199.13 | 29.1 to 30.2 | 5.7 to 5.9 ms | about 91 000 |
| 1280x800 | 612.37, 612.29, 635.81, 398.27 | 28.6 to 30.3 | 19.4 to 19.9 ms | about 383 000 |

After the cleanup (the tree at b23f168), the same session ran main's viewer (b5c70dc) and the
cleaned tree alternately, 100 frames headless each, on a scene with about 1.5x the points of the
table above [M 2026-11-13]:

| size at 30 fps | `create_from_rgbd` with upload, main (two runs) | the cleaned tree (two runs) | points |
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
  the jump out of the timestamps it delivers (src/camera/orbbec_rules.hpp).
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
  `create_from_rgbd` 6.5 to 6.9 ms with upload; 58 % of pixels with depth in an indoor scene,
  0.25 m to 21 m.

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
  is unverified [?].

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
synthetic frames by a scratch program, under the build lock with no compiler running, in the
power mode MAXN_SUPER with the GPU clock under `nvhost_podgov`, 306 to 1020 MHz [M 2026-11-14].
The replay of sequences taken on the hardware measures the whole path later.

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

**The registration of one frame** (docs/methods/registration.md): the host's call of
`registration::align()` by the full method, both terms coarse to fine with the tabletop object's
starting values, on the tabletop object's pair of `tests/test_registration.cpp` (a box and a sphere
on a plane from 0.5 m, moved by 1.5° and 10.8 mm), every run converged; the surface's gradient
kernel alone on its three levels, a copy of the tree's kernel timed by CUDA events; and the call
with one iteration on level 0 alone, its gradient kernel included. The medians of 20 s per case
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
| `uvcvideo 1-2.N:1.1:` or `2-1.N:1.1: Unknown video format 20343159-0000-0010-8000-00aa00389b71` | benign | uvcvideo does not know the depth format Y14 (the GUID starts with the FourCC `Y14 `); after a program closes the camera | [M 2026-11-13], [S] UVC GUID layout |
| `perf: interrupt took too long (A > B), lowering kernel.perf_event_max_sample_rate to R` | benign while a profiler samples the CPU (`nsys profile`, `perf record`) [?: seen once]; with no profiler running, a stop condition | the kernel's handling of the profiler's samples took longer than `kernel.perf_cpu_time_max_percent` (25) allows, so it lowers the samples per second it grants [?: the sysctl's documented purpose, not read here]; the lowered rate stays: 30000 two hours later | [M 2026-11-14: five lines at 19:50:48 to 19:50:51, the only ones of that boot; an `nsys profile` session of a scratch timing program began at 19:50:48 and recorded 1979 CPU samples (its report's tables `TARGET_INFO_SESSION_START_TIME`, `SAMPLING_CALLCHAINS`); about 130 runs of the hash experiment that day, with no profiler, logged none; `sysctl kernel.perf_event_max_sample_rate` 30000 at 21:47] |
