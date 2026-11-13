# Hardware

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The board this tree runs on. Every fact carries its tag (AGENTS.md).

## Platform

| what | value | evidence |
|---|---|---|
| computer | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super, 7.4 GiB RAM | [M 2026-11-13] `/proc/device-tree/model`, `free` |
| power mode | MAXN_SUPER | [M 2026-11-13] `nvpmodel -q` |
| OS | L4T R36.5.0 (JetPack 6), Ubuntu 22.04, kernel 5.15.185-tegra, aarch64 | [M 2026-11-13] `/etc/nv_tegra_release`, `uname -a` |
| toolchain | CUDA 12.6.68, GCC 11.4.0, CMake 3.22.1, OpenCV 4.8.0 | [M 2026-11-13] |
| GPU | Ampere, compute capability 8.7 | [S] CUDA documentation |
| camera | Orbbec Gemini 2 L, USB `2bc5:0673`, firmware 1.4.38 | [M 2026-11-13] `lsusb`, `orbbec_viewer --list` |
| camera SDK | OrbbecSDK v2.10.6, linux arm64, in `~/.local/share/3dscanner` | tools/fetch_deps.sh |

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

OrbbecSDK's warnings in these runs, the stream going on after each:

* `Timestamp anomaly detected ... currentDiff: 4294967296`, a few times at the start of every
  run: the frame timestamp jumps by exactly 2^32 [M]; why is open [?].
* 1280x800 only, at the start: `Update data size(1824744) > data buffer size! (1792000)`,
  the frame dropped. 1792000 bytes is 1280 x 800 x 14 bit, one packed Y14 frame [S].

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

## Connections

The four USB-A ports sit on a Realtek USB 3.2 hub, seen twice by the host controller
`tegra-xusb 3610000.usb` [M 2026-11-13, `lsusb -t`]:

| hub | bus | speed |
|---|---|---|
| `0bda:0489` (SuperSpeed half) | 2, port 1 | 10000M |
| `0bda:5489` (USB 2 half) | 1, port 2 | 480M |

The camera: `2-1.4`, bus 2, 5000M, with the third cable tried [M 2026-11-13]. With the two
cables before, it came up on bus 1 at 480M in every port; AGENTS.md §6.

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
