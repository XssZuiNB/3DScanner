# Working on this repository

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

This is the operating manual for anyone, human or agent, who touches this tree. The hardware
is live and shared. A mistake on it costs a reset at best and a reboot at worst, so read
sections 1 to 3 before running anything.

Every claim carries its evidence: **[M date]** measured on this hardware, **[S]** from
source, a datasheet or a header in the repo, **[B]** read out of a binary, **[?]**
unverified. A number without a tag is a mistake; treat it as [?].

## 1. What this repository is

3DScanner: a 3D scanner built from an Orbbec Gemini 2 L RGB-D camera and a Jetson Orin Nano
(docs/HARDWARE.md). It is a full rewrite of CUDA_3D, the user's GPU point-cloud library
written for Intel RealSense (its history ends at 5e568d9).

What the rewrite builds: an operator scans a tabletop object or a large object with the
hand-held camera and works the scan from a web page the Jetson serves (the terms: GLOSSARY.md;
the decisions: docs/adr/; the next step: ROADMAP.md). Built so far: the libraries below and seven
programs, each with its offline tests, and the scan interface, which runs a whole scan offline
(its test drives it with the synthetic source, the replay tool with a sequence), and the web
layer's HTTP routes and WebSocket over it (docs/ROUTES.md), which their tests drive on the loopback
interface, and the page an operator makes a scan on (`src/web/page/`), whose code runs only in a
browser, and the service, which composes the camera, the scan interface and the web layer
(docs/SERVICE.md); its tests drive it with a stand-in for the camera, and it has not run on the
camera yet (`service-at-boot-on-the-board`). The tree began as CUDA_3D cleaned up (dead code, RealSense and PCL removed), made to
build here and given the Orbbec camera: the rewrite's prototype, whose methods carry over. Its
library was the test oracle of the organized points, the normals, the pyramid's intensity and the
registration until each agreed with it, and is deleted; each method document keeps the agreement
as measured (docs/methods/; CODING_STANDARDS.md, "The rewrite's code"), and its code stands in git
at 74dda68.

The tree today: `src/` holds one library per directory, `src/camera/` two, so that only the
Orbbec camera links OrbbecSDK, and `src/scan/` two, so that the programs that only read and write a
scan's files link no CUDA. Each library links only libraries above it in this table
(`src/CMakeLists.txt`; the file of each thing: §7 rule 4):

| library | holds | links, of the tree's |
|---|---|---|
| `types` | header only: the plain structs (camera parameters, a frame's description, the camera's pose, a trajectory's entry, an IMU sample, a model's point), console colors | none |
| `camera` | the frame with its IMU samples, the frame source seam, the sequence on disk, the POSIX file calls, the Orbbec camera's rules and its IMU's queue, which need no camera, the seam of the camera's run-time properties and the hold of the color exposure and white balance in front of a camera; the offline tests link it without OrbbecSDK | `types` |
| `orbbec_camera` | `orbbec_device`, the camera as a frame source, which delivers depth aligned to color, so the GPU code sees one resolution and one set of intrinsics, and with each frame the IMU's samples, and as the camera's run-time properties | `camera` |
| `gpu` | the GPU array and image, the pool that closes, a frame's upload, a stage kernel's launch | `types`, `camera` for the frame it uploads |
| `geometry` | a frame's organized points, their normals, the frame's pyramid with the intensity | `gpu` |
| `registration` | the registration of a frame against a surface by colored ICP, coarse to fine over the pyramid; the SE(3) exponential | `geometry` |
| `fusion` | the sparse TSDF: its hash of blocks, its pool, the allocation of the blocks near a frame's surface, the integration of a frame, the ray cast of the model's surface at each level of the pyramid, the extraction of its colored points with normals | `geometry` |
| `tracking` | the per-frame path's steps 1 to 6 for each frame: the upload, the pyramid, the ray cast from the last tracked pose, the registration, the check, and the fusion when it passes; the stage clock that times each stage for the replay tool; no kernel of its own | `fusion`, `registration` |
| `refinement` | the pose graph of keyframes and its solve, on the host; the fusion of a scan's sequence at the full voxel size, with the full model's extraction to the host | `types`, `registration` for the SE(3) exponential, `fusion` for the sparse TSDF and through it `camera` for the sequence's reader |
| `separation` | the proposal of the support plane and the box from a full model and its trajectory, the least-squares plane, on the host | `types` only, so that the scan can call it |
| `scan` | the files a scan keeps, a scan's time in UTC, the parameters of each object class; host only, so that `box_fit` and the viewer's loop link no CUDA | `types`, `camera` for the POSIX file calls and the source's description |
| `scan_interface` | the scan interface: its scan thread, commands and state, a scan's phases from its start to its refinement, the "too close" notice, the live model; and the stages' parameters of an object class's set, the one copy of each conversion | `scan`, `tracking`, `refinement` |
| `web` | the web layer: the HTTP routes of the scan interface's commands and state and of the page's static files, on cpp-httplib, and the messages they answer with (docs/ROUTES.md) | `scan_interface` |

The programs in `apps/` keep their command lines and their work (the viewer's loop, the three
measurements) in libraries of their own, which the offline tests drive. `orbbec_viewer` runs its
loop (`apps/viewer_loop.*`) on the camera as a frame source, and the offline tests run the same
loop on their synthetic source; the loop makes each frame's organized points on the GPU, saves a
snapshot's points with their normals as a model's PLY and, with `--sequence`, writes the frames
it takes into a sequence; with `--hold-color` it holds the color camera's exposure and white
balance after the settling and prints their values (§3). `distortion_check` measures the
viewer's snapshots of a checkerboard; `box_fit`, the acceptance tool, takes a box's edges from an
object model's PLY; `depth_noise` takes the depth noise's c, which sets the sparse TSDF's
truncation, from a sequence taken with the camera at rest (docs/methods/depth-noise.md); `replay`,
the replay tool, runs a sequence through the scan interface and writes the scan's directory, and
with `--times` prints each frame's stage times (docs/methods/tracking.md, "The replay tool's
lines"), so that a stage is measured on the board without the camera; none of the four opens the
camera. `scanner_service`, the service (docs/SERVICE.md), serves the page and runs the scans it
starts, with the Orbbec camera opened only from a scan's start to its stop and its color exposure
and white balance held during the scan; its parts (`apps/service.*`) and its command line
(`apps/service_options.*`) are libraries the offline tests drive with a stand-in camera;
`scan_in_progress` asks it whether a scan is in progress. The service is the tree's one daemon:
once `install-the-user-service` has installed it, it runs from boot as a service of the user's
account and holds the camera during every scan an operator starts from the page (§2). Its page has
no login and is for the operator's own network, never the internet (R6); the scanner is for rigid,
opaque objects that are not mirror-like, and people are not scanned (A5).

## 2. Before you run anything

**Check the machine state.** These read only:

    lsusb -t                                   # the camera, its bus and speed (5000M = USB 3)
    pgrep -af 'orbbec|ob_|scanner_service'     # a program that holds or may open the camera
    build/apps/scan_in_progress                # the service: exit 0 a scan in progress, 1 none, 3 cannot tell
    journalctl -k --since -10min | grep -iE 'usb|uvc|nvgpu'

`dmesg` is closed to the user [M 2026-11-13]; `journalctl -k` works (group `adm`).

**The service.** `scanner_service` (§1; docs/SERVICE.md) may be running, from boot once it is
installed, and holds the camera, writes its properties 2000 and 2003 and runs the GPU during every
scan an operator starts from the page, and the GPU during refinement. Before a camera program
runs, before a timing run, and before the service is stopped or replaced, ask it:
`build/apps/scan_in_progress` (with `--address` and `--port` when it serves elsewhere than
127.0.0.1:8080) exits 0 while a scan starts, scans or refines: wait, or ask the user; 1 when it is
idle; 3 when it cannot tell: the service is not running, or does not answer, which `pgrep` above
tells apart. While it is idle a camera program may run; a scan an operator starts meanwhile is
refused, as the camera is held. Stopping the service in the middle of a scan leaves the scan
unfinished (docs/SERVICE.md, "The stop"); tell the user before a step that stops it.

**Root.** Nothing in the daily work: the udev rule gives the user the camera
(docs/HARDWARE.md, "Permissions"), and tools/fetch_deps.sh puts the SDK and Eigen in the user's
home. An `apt install` or a new udev rule goes to the user as a command to run. Ask the user for
the password when one is needed; never write it into a file, a script, a command line or a
commit.

**One writer.** The program that opens the camera through OrbbecSDK claims its USB interfaces;
a second program cannot open it then [?]. Nothing locks it: check with `pgrep` and
`scan_in_progress` above before starting one.

**Privacy.** Data recorded on the hardware (pictures, sound, logs with names) stays in a
scratch directory outside the repository; `.gitignore` keeps its file types out of the tree.

**Build as the user, never as root.** An install builds nothing as root.

## 3. What each command does to the hardware

The left column looks diagnostic and often is not.

| command | what it actually does |
|---|---|
| `tools/build.sh`, `tools/verify.sh`, `tools/run_mutant.sh` | no hardware access |
| `tools/fetch_deps.sh` | no hardware access; downloads into `~/.local/share/3dscanner` |
| `build/apps/distortion_check` | no hardware access: reads a snapshot's two PNG files and prints its numbers; a command line it refuses exits 2 with the reason and the usage [S: distortion_check.cpp; the refusals: ctest `distortion_options`]; links neither OrbbecSDK nor CUDA [M 2026-11-14: `ldd`] |
| `build/apps/box_fit MODEL.ply` | no hardware access: reads an object model's PLY (docs/SCAN_FILES.md, "Models") and prints the length, width and height of the box fitted to it and each face's spread; a command line it refuses (an argument it does not know, `--help` too, a second path, no path, a path that does not exist) exits 2 with the reason and the usage, a model it cannot read or that is no box exits 1 with the reason [S: box_fit.cpp; the refusals: ctests `box_fit_options` and `box_measure`, which also runs the program for each exit]; links neither OrbbecSDK nor CUDA nor OpenCV [M 2026-11-15: `ldd`] |
| `build/apps/depth_noise DIR` | no hardware access: reads a sequence taken with the camera at rest, `DIR/frames.sequence` (docs/SEQUENCE_FILE.md), twice, and prints the depth noise's c with these lines (docs/methods/depth-noise.md, "Output"): `depth_noise: <DIR>: <N> frames of <W>x<H>, depth unit <u> mm`, `pixels: <K> of <P> kept, with depth in at least 90 % of the frames and in both halves`, `still: the halves differ by <D> mm rms, <T> times their standard error`, `spatial: <S> mm rms from the plane fitted to the mean frame`, for each depth bin that counts `bin <a> to <b> m: <n> pixels, noise <sigma> mm, spatial <s> mm, c <c_b> /m` or the same ending in `below the rounding`, and last `c: <c> /m, spread <r> /m over <B> bins`; a command line it refuses (an argument it does not know, `--help` too, a second directory, none, one that does not exist) exits 2 with the reason and the usage, a sequence it cannot read or refuses (its depth unit or intrinsics, too few pixels, not still, its noise below the rounding) exits 1 with `depth_noise: ` and the reason [S: depth_noise.cpp; the refusals: ctests `depth_noise_options` and `depth_noise_measure`, which also runs the program for each exit]; links neither OrbbecSDK nor CUDA nor OpenCV [M 2026-11-15: `ldd`] |
| `build/apps/replay --sequence DIR (--object-class CLASS \| --parameters FILE) --scans OUT [--times]` | no hardware access, no camera: reads the sequence `DIR/frames.sequence` and, with `--parameters`, a kept scan's `parameters.json`, whose object class and set it runs with, and runs one scan through the scan interface on the GPU, as a scan made with the camera runs (the sparse TSDF's memory budget is the set's, 2.0 GB for both object classes' own sets); the writer waits for room, so every frame is taken. It writes the scan's directory, named by its start in UTC, into OUT, an existing empty directory: a copy of the sequence, the parameters with the build's commit (`apps/build_commit.hpp`), the trajectory and the full model, as large as the sequence and more. OUT holds frames of the camera when DIR does, so it lies outside the repository as DIR does (§2, "Privacy"). It prints the lines of docs/methods/tracking.md, "The replay tool's lines": with `--times` a time line for each frame, for which it waits for the GPU after each stage; last `replay: <OUT>/<scan>: refined, end <end>, frames taken ...`, exit 0. A command line it refuses (an argument it does not know, `--help` too, an option without its value or given twice, neither or both of `--object-class` and `--parameters`, an unknown object class, a parameters file, a `frames.sequence` or `--scans` missing, a scans directory that is no directory, holds something already or cannot be written) exits 2 with the reason and the usage, before any GPU work; a parameters file or a sequence's header its reader refuses, a start refused or a scan left unfinished exits 1 with `replay: ` and the reason, a CUDA error among the reasons a stop condition as any (below) [S: replay.cpp, replay_options.cpp; the refusals: ctests `replay_options` and `replay`, which also runs the program for each exit]; links no OrbbecSDK [M 2026-11-16: `ldd`] |
| `build/apps/orbbec_viewer --list` | opens the camera, reads its info and stream profiles, starts no stream [S: orbbec_device.cpp]; prints the firmware's version without the check of the least version that an opening for frames makes (below); what OrbbecSDK sends on open is unverified [?] |
| `build/apps/orbbec_viewer` with an argument it does not know (`--help` too), an option without its value, or a `--sequence` directory that does not exist, holds something or cannot be read or written | prints the reason and the usage line and exits 2; `main` reads the options, and looks at that directory without changing it, before it opens the camera or the GPU or installs a signal handler [S: orbbec_viewer.cpp, `main`; the refusals: ctest `viewer_options`]. The load-time code of OrbbecSDK, OpenCV and CUDA runs before `main`; what it does is unverified [?] |
| `build/apps/orbbec_viewer [--frames N]` | opens the camera and refuses it, before any stream starts, when its firmware is older than 1.4.53, OrbbecSDK v2's least for the Gemini 2 L, or its version is not three numbers: prints the reason and `Didn't find the camera!` and exits 1, as for every refusal of the opening (held by another program, no camera, no such stream) [S: orbbec_device.cpp, `open`; orbbec_rules.cpp, `firmware_refusal`; ctest `orbbec_rules`]; otherwise streams color and depth with depth-to-color alignment, and opens the IMU's streams, the accelerometer and the gyroscope at 200 Hz, each through OrbbecSDK's sensor of its own, and prints them as `Accel: <rate> <range> \| Gyro: <rate> <range>`; when they cannot be started it opens without them, refusing nothing, and prints `No IMU: <reason>; the frames come without its samples`; readings it loses it prints as `Skipped <count> IMU reading(s), the last because <reason>` [S: orbbec_device.cpp, `start_imu`; orbbec_viewer.cpp; ctest `orbbec_imu`; docs/methods/imu-samples.md]; without `--hold-color` sets no option of the camera [S: orbbec_device.cpp, orbbec_viewer.cpp]; makes each frame's organized points on the GPU on a stream of its own, after the first frame has come; `s`, or the last of N frames headless, writes `cloud_NNN.ply`, the frame's points within 0.15 m to 3.0 m that have a normal, in its camera frame, as a model's PLY (docs/SCAN_FILES.md, "Models"), `.color.png`, `.depth.png` into the current directory: run it in a scratch directory. N counts delivered frames: a timeout or a failure of the camera prints its line, counts no frame and the loop goes on; a failure of the GPU prints CUDA's or the pool's reason and ends the run with exit 1, a stop condition as any CUDA error is (below) [S: viewer_loop.cpp; ctest `viewer_loop`] |
| `build/apps/orbbec_viewer --sequence DIR [...]` | as above, and writes every frame it takes, the first included, unchanged, with its IMU samples, into one file, `DIR/frames.sequence` (docs/SEQUENCE_FILE.md), each marked as tracked by nothing. These are frames of the camera, so DIR lies in the user's home, outside the repository and every worktree, and not under `/tmp`, which is emptied at boot (§2, "Privacy"; §6), e.g. `~/3dscanner-sequences/<name>/` [?: a choice]; it is not the current directory, where the snapshots and OrbbecSDK's log go. About 38 MB/s at 640x400 and 154 MB/s at 1280x800, 30 fps, 2.3 GB and 9.2 GB a minute, so a 5-minute walk takes about 12 GB and 46 GB; the IMU's samples add 8 KB/s [S: computed, docs/SEQUENCE_FILE.md, "Layout"]. The report line goes on with ` \| queue full <count>`, the frames since the last report the writer's queue had no room for, which were shown but not written; without `--sequence` the line is unchanged. The run ends on the frame limit, `q`, ESC, Ctrl-C (SIGINT), SIGTERM (a kill, `timeout`) or SIGHUP (a closed terminal or SSH session): all four signals set the loop's exit flag, so the last frame is written and the file gets its end [S: orbbec_viewer.cpp, `install_stop_handlers`; the exit flag: ctest `viewer_loop`]. Then `Wrote <A> frames to DIR/frames.sequence, <F> not written because the writer's queue was full`; a file that cannot be created prints the writer's reason, and a failed write or close its reason followed by `; <W> of <T> frames taken written, <L> lost from the writer's queue, <F> not written because the writer's queue was full`, each with exit 1 [S: viewer_loop.cpp; ctest `viewer_loop`] |
| `build/apps/orbbec_viewer --hold-color [...]` | as above, and WRITES two of the camera's run-time properties, the switches of the list below, never their values; `orbbec_device` refuses a write of any other [S: color_hold.cpp, orbbec_device.cpp; ctest `color_hold`]. Once the streams run it reads the five properties of the list, switches both automatic controls on and prints what it read: `Color at the start: auto exposure <on\|off>, exposure <n>, gain <n>, auto white balance <on\|off>, white balance <n>`. After the 30th frame delivered [?: initial] (a timeout or a failure counts none) it reads them (`Color before the hold: ...`), switches both controls off, so that the camera holds what they reached [?: unverified on the camera], and reads them again (`Color while held: ...`). At the end, the frame limit, `q`, ESC, the three stop signals, a failure of the GPU and an exception out of the viewer's loop included (`main` catches it, so the stack unwinds), it switches both on again and reads them (`Color after: ...`). A read or write the camera refuses prints `Color: the camera refused to <read\|switch> <property> (<number>) [on\|off] <at the start\|at the hold\|at the end>: <reason>` and the frames go on: a control with a refused read or a refused switch-off before its hold stays automatic, one whose switch-on at the start was refused stays as the camera had it, and a read refused while held or at the end changes nothing. An exception inside the camera's or the hold's own calls, which are noexcept, ends the program through std::terminate and, like a SIGKILL or a crash while held, leaves both off until the next `--hold-color` start or a replug [?]. `--list` ignores the option. Whether these writes persist in the camera's flash is unverified [?]: `first-trial-of-held-exposure` checks it first |
| `build/apps/scanner_service` with an argument it does not know (`--help` too), an option without its value or given twice, a port that is no whole number from 1 to 65535, or a scans directory that does not exist, is no directory or cannot be written | prints `scanner_service: <reason>` and the usage and exits 2; `main` reads the options, and looks at the scans directory without changing it, before it opens the GPU, a port or the camera or installs a signal handler [S: scanner_service.cpp, `main`; the refusals: ctest `service_options`]. The load-time code of OrbbecSDK, OpenCV and CUDA runs before `main`; what it does is unverified [?] |
| `build/apps/scanner_service [--scans DIR] [--port N] [--address A] [--page DIR] [--keep-sequence]` | the service (docs/SERVICE.md), until SIGINT, SIGTERM or SIGHUP: serves the page and its routes on port N (8080) of address A, every interface of the board (`0.0.0.0`) without `--address`, with no login, so for the operator's own network only, never the internet (R6), and for scanning rigid, opaque objects that are not mirror-like, never people (A5); opens the GPU's CUDA stream at once and holds no camera while idle. A scan started from the page opens the camera, with the refusals of the viewer's opening (held by another program, no camera, a firmware below 1.4.53), each failing the start with the reason in the state and the log's `the start failed: ...`; it runs the per-frame path and refinement on the GPU in a sparse TSDF of the object class's budget, 2.0 GB, and WRITES the camera's switches 2000 and 2003 as `--hold-color` does (below): on at the opening, off after the 30th frame, on again at every ending, the service's stop included; it closes the camera at the scan's stop, before refinement. Each scan's directory, frames of the camera, goes into DIR (`~/3dscanner-scans`), outside the repository (§2, "Privacy"); without `--keep-sequence` a refined scan's sequence is removed, an unfinished one keeps it. Its lines, a stop's order and worst case (about 15 s [S: computed], which systemd's stop timeout must exceed) and its exits: docs/SERVICE.md [S: service.cpp, scanner_service.cpp; ctest `scanner_service`, with a stand-in camera]. It has not run on the camera: its first run is `service-at-boot-on-the-board`'s, on the user's word |
| `build/apps/scan_in_progress [--address A] [--port N]` | no hardware access: one `GET /api/state` to the service on A:N (127.0.0.1:8080); exit 0 and `scan_in_progress: yes, <phase>` while a scan starts, scans or refines, 1 and `scan_in_progress: no, idle` when idle, 3 and `scan_in_progress: cannot tell: <reason>` when no service answers or the answer holds no phase; a command line it refuses exits 2 with the reason and the usage (docs/SERVICE.md) [S: scan_in_progress.cpp; ctest `scanner_service` runs it for each exit]; links neither OrbbecSDK nor CUDA nor OpenCV [M 2026-11-17: `ldd`] |
| any program using OrbbecSDK | writes `Log/OrbbecSDK.log.txt` into the current directory [M 2026-11-13]; closing it hands the camera back to uvcvideo, which logs the enumeration lines again (docs/HARDWARE.md) |
| OrbbecSDK's examples, OrbbecViewer | not built here; `ob_device_firmware_update`, `ob_device_optional_depth_presets_update`, `ob_multi_devices_firmware_update` and OrbbecViewer's firmware and preset pages WRITE the camera's flash |
| `nsys profile` with CPU sampling (on in the one run seen; `--sample=none` turns it off [?: nsys's documentation, not read here]), `perf record` | samples the CPU through the kernel's perf events; the kernel may log `perf: interrupt took too long ... lowering kernel.perf_event_max_sample_rate` and keep the lowered rate [M 2026-11-14: one `nsys profile` run]. The line is benign only while such a profiler runs (docs/HARDWARE.md, "Kernel log"): say in your report that one ran, with its time |

**The camera's run-time properties the scanner may set** (R13), through
`src/camera/camera_properties.hpp` and only at run time, the camera's flash never; each under
OrbbecSDK's number and name [S: OrbbecSDK v2.10.6, Property.h]:

| property | written | why |
|---|---|---|
| 2000 `OB_PROP_COLOR_AUTO_EXPOSURE_BOOL` | on and off; unverified for the flash [?] | the color exposure held for a scan (`src/camera/color_hold.*`), by `orbbec_viewer --hold-color` and by the service in every scan |
| 2003 `OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL` | on and off; unverified for the flash [?] | the white balance held for a scan |

The hold also reads 2001 `OB_PROP_COLOR_EXPOSURE_INT`, 2002 `OB_PROP_COLOR_GAIN_INT` and 2004
`OB_PROP_COLOR_WHITE_BALANCE_INT`, and writes none of them: `orbbec_device` refuses a write of
any property but the two switches (`is_switch()`, the table in `src/camera/camera_properties.hpp`;
ctest `color_hold`). A property enters this list on the user's word, with its first trial on the
camera (R13).

**Standing authorisations.** The hardware steps the user allows agents in general, each with
its date and its conditions. None yet: until a line is here, every hardware step waits for
the user's words in the conversation.

**Stop conditions.** On any of these, run nothing more on the hardware and tell the user:

* a kernel log line that docs/HARDWARE.md, "Kernel log", lists as harmful, or one it does
  not list at all;
* a process that stops answering (a wedged thread, a call that does not return);
* a failed health check;
* the camera leaves the bus while streaming (`USB disconnect` in the kernel log), or
  `orbbec_viewer` prints `No frames received from Orbbec camera for 5s!`;
* a GPU fault (`nvgpu` or `Xid` in the kernel log, a CUDA error from a program).

## 4. Build and verify

| command | what | when |
|---|---|---|
| `tools/build.sh` | the incremental build into `build/`, warnings included | while working |
| `tools/verify.sh` | the one offline check: the checkers of CODING_STANDARDS.md, "Tools", a fresh configure and build in a new `verify-build.XXXXXX/` beside `src/` (removed after), which fails on any warning in their logs (CODING_STANDARDS.md, "Warnings"), every test; no hardware, no root; it also passes in a copy of the tree without `.git`. A pass records the tree's fingerprint | at the end of every change |
| `tools/verify.sh --passed` | the passed-tree check: whether the working tree as it is now has passed `tools/verify.sh`; exit 0 yes, 1 no, 2 cannot tell (no git working tree); builds nothing | before every commit: a commit needs its yes |
| `tools/run_mutant.sh <patch>...` | whether `tools/verify.sh` catches a deliberate mistake: a scratch copy of the working tree, built once; each patch applied in turn, checked by `tools/verify.sh --incremental` in the copy's `build/` (the checkers; when a file was added, removed or renamed or a CMake file changed, a configure on a fresh cache; the rebuild of what the patch touched and its warnings; every test), within time limits, then taken out | to show a new rule, refusal or branch bites |

**The fingerprint** is the SHA-256 of the tree's files, tracked and untracked, ignored files
left out, a nested repository's files included by the same rule (the set `tools/run_mutant.sh`
copies): their paths, contents and executable bits.
`tools/verify.sh` records it in `~/.local/state/3dscanner/verify-passed` when a fresh run
passes and the tree did not change while it ran. A changed, added or deleted file makes the
passed-tree check say no until a new run passes; returning a tree to a state that passed makes
it say yes again. It covers the tree only: after a toolchain update or a new download of
`tools/fetch_deps.sh`, run `tools/verify.sh` again.

**Jobs.** Builds run four parallel compile jobs, not one per core (six): `BUILD_JOBS` in
`tools/build_common.sh` sets it for `tools/build.sh`, `tools/verify.sh` and, through verify,
`tools/run_mutant.sh`, and holds the measurements: one compiler process peaks at up to 1.5 GB and
a -j6 build froze the board on 2026-11-13 [M 2026-11-13].

**The build lock: one heavy job at a time.** Every `tools/build.sh`, `tools/verify.sh` and
`tools/run_mutant.sh`, and every timing run on the board, runs under one lock, so that no two
builds overlap and no compiler runs beside a timing run. The tools do not take it themselves
[S: tools/]; the caller does, and a second caller waits until the first is done:

    flock ~/.local/state/3dscanner/build.lock tools/verify.sh

After a timing run, read the kernel log (§2) and say in the report what it showed. While a heavy
job runs, `~/.local/state/3dscanner/monitor.sh` (outside the tree) logs available
memory, free swap, the over-current counters, the input power and the hottest zone every 2 s
into `monitor.log` beside it, syncing each line so that a hard reset leaves the last state on
disk [S: the script]; start it in the background and end it with the session.

**ccache.** Every compile of the three tools goes through ccache (CODING_STANDARDS.md,
"Tools"), set in the build directory's compiler launcher, so that the builds of one tree hit
what another tree's builds wrote (`configure_build` in `tools/build_common.sh`): a verify
after `tools/build.sh` and the mutant copy's first build each hit 20 of 20 compiles
[M 2026-11-13: `ccache -s`, ccache's log]. Verify takes 243 s when every compile runs
(`CCACHE_RECACHE=1`) and 89 s with a warm cache, 79 s of it the tests [M 2026-11-15: the tree
of `retire-the-prototype-library`, under the build lock]. On a hit ccache replays the compile's
warnings, g++'s and nvcc's [M 2026-11-13], so verify's warning check sees them on a hit too.

**Mutants.** In a patched run each test may take 3 × the unpatched copy's run, at least 60 s,
and the whole run 600 s more for the configure and the build [S: tools/run_mutant.sh]; a run
past either is "caught by the time limit". Taking a patch out writes its files anew, so the
next patch's build first rebuilds what it broke; a configure starts from a fresh cache, so no
cache variable a patch set outlives it.

A test touches no hardware: what it needs from the hardware comes from a simulation or a
recording. The tests (`tests/`, run by ctest) need the GPU and no camera. No frozen files yet.

## 5. Which numbers may be changed

From the hardware or a source, changed only with new evidence written above them:

* depth unit: read from each depth frame (`DepthFrame::getValueScale()`, mm per value);
  1 mm is the Orbbec default [S: OrbbecSDK].
* `GCA_CUDA_ARCHITECTURES` 87: the Orin's compute capability 8.7 [S: CUDA documentation].

Free choices: the viewer's stream size, 640x400 at 30 fps in `apps/viewer_options.hpp`
[?: a choice; it exists on USB 2], and its depth range, 0.15 m to 3.0 m in
`apps/viewer_loop.cpp` [?: a choice].

## 6. Traps

* **The camera is on USB 2 although every port is USB 3.2.** The board's four ports sit on a
  Realtek USB 3.2 hub that shows up twice: `0bda:0489` on bus 2 (10000M) and its USB 2 half
  `0bda:5489` on bus 1 (480M). A device whose USB 3 link does not come up lands on bus 1, and
  the SuperSpeed side logs nothing at all. Two cables gave this in three ports; a third cable
  gives 5000M on bus 2 [M 2026-11-13]: a USB-C cable that charges and carries USB 2 may still
  lack the USB 3 lines. After plugging in, check `lsusb -t` for 5000M.
* **Gemini 2 L depth defaults to RLE** (compressed) [S: OrbbecSDKConfig.xml, `Gemini2L`],
  and on USB 2 Y14 and RLE are the only depth formats [M 2026-11-13]. `orbbec_device` asks for
  Y16, Z16, Y14, RLE in that order and refuses every depth frame that does not arrive as
  16 bit; Y14 arrives as 16 bit [M 2026-11-13].
* **`/tmp` is emptied at boot** [S: `D /tmp` in `/usr/lib/tmpfiles.d/tmp.conf`]: what must outlive a reset goes elsewhere, and
  a scratch directory there is created anew.
* **Thrust's `make_pair` forwards its arguments since CUDA 12.x's Thrust 2.x**:
  `thrust::make_pair<float, float3>(a, b)` no longer compiles with lvalues; construct
  `thrust::pair<float, float3>(a, b)` [M 2026-11-13, CUDA 12.6]. No code of the tree uses Thrust
  today; this holds for code that takes it up (CODING_STANDARDS.md, "The rewrite's code").

## 7. How the code is written

[CODING_STANDARDS.md](CODING_STANDARDS.md) holds the style guide of each language and the
checkers; these are the rules no checker holds.

1. **A file starts with SPDX, then its name and an overview of what its module does**, and
   nothing else; a module is one source file and its header. C and C++: `/// @file <name>`.
   JavaScript: `// SPDX-License-Identifier: 0BSD`, then `/** @fileoverview <name> - <overview> */`;
   HTML: `<!doctype html>` first (Google's HTML/CSS guide), then the SPDX line and
   `<name> - <overview>` in comments; CSS: both in `/* */` comments.
   A script puts its SPDX line under the shebang, then `# <name> - <overview>`; a CMake file
   starts with its SPDX line, then `# <path> - <overview>`, the path from the tree's root.
   A trap, an ownership rule or a cache note goes into the comment of the function it
   concerns.
2. **Every function's comment states its purpose, every parameter and its return value**:
   a public function in its header above its declaration, a file-local one above its
   definition. C and C++: a `///` comment whose first sentence is the purpose, one `@param`
   for each parameter, even one whose name looks obvious, and `@return` for every function
   that returns a value. Shell: the Google Shell Style Guide's function comment. JavaScript: the
   Google JavaScript Style Guide's JSDoc, `@param {type} name` and `@return {type}`. Python: the
   Google Python Style Guide's docstring. A fact in a comment carries the same tags as the
   documents. A kernel and a device function are functions like any other.

   The rewrite's code, all of the tree's code, states more, each part wherever it applies.

   | part | states |
   |---|---|
   | `@param` | its direction (`[in]`, `[out]`, `[in,out]`); its unit (m, mm, px, rad, s); for a geometric value its coordinate frame; its valid range; for a pointer or a buffer who owns it, how long it must live, and whether it is host or device memory |
   | `@return` | unit and frame as for a parameter; what comes back on each failure |
   | `@pre` | what the caller guarantees, and what the function does when it does not hold |
   | `@note` | for GPU code: the stream it runs on, whether it allocates, whether it waits for the GPU; whether two threads may call it at once |
3. **`main()` is a composition of named steps.** Options, setup, the loop and the report are
   separate functions.
4. **One copy.** Each shared thing lives in one place, the lowest layer that needs it.

   | concern | home |
   |---|---|
   | camera parameters, a frame's description, the camera's pose, a trajectory's entry, an IMU sample with the IMU's rotation into the camera frame, gravity's estimate with its closed set of outcomes and the one check of its up, and a model's point (plain structs), the value comparison of the intrinsics and of a frame's description (`operator==`, which a parameters file's round trip uses; a sequence compares descriptions bit for bit, `src/camera/sequence_format.*`), the one check that intrinsics can be used, the one check of a pose, console colors, a number's little-endian bytes in the tree's binary layouts (the sequence, the live model message) | `src/types/` |
   | the frame with its lists of IMU samples and their reserve; the frame source seam | `src/camera/frame.*`; `src/camera/frame_source.hpp` |
   | the sequence on disk (docs/SEQUENCE_FILE.md): its layout, its file's name in a directory, writer and reader, the header read alone, the comparison of two frames' descriptions bit for bit, the rule of each sensor's samples' order (`src/camera/sequence_format.*`); the POSIX file calls the writers and readers share | `src/camera/sequence_*`; `src/camera/posix_file.*` |
   | the writer's calls a scan makes, a seam (`sequence_sink`) that the sequence's writer implements, so that the scan interface's test can make the writer slow, as the camera's run-time properties have their seam for the test of the hold; what the writer does with a full queue (`queue_full`): it says so for a live source, it waits for room for a sequence as the source, the replay tool's and the tests' | `src/camera/sequence_writer.hpp` |
   | Orbbec: alignment, color decoding, the IMU's sensors and their callbacks, OrbbecSDK's report as plain data, the color properties read and written | `src/camera/orbbec_device.*` |
   | the camera's run-time properties: their numbers, names, the one table of which is a switch and which control each belongs to, and their seam; the hold of the color exposure and white balance in front of a camera, its report and lines, and a reader's place in the report, which the viewer, the service's log and the scan interface take the new entries with | `src/camera/camera_properties.hpp`; `src/camera/color_hold.*` |
   | Orbbec: the choice of streams, the checks of a frame, the rule for timestamps, which also places an IMU reading's time, the refusals of an opening | `src/camera/orbbec_rules.*` |
   | Orbbec's IMU (docs/methods/imu-samples.md): its rate, the choice of its streams, the reason an opening goes on without it, the queue of a sensor's readings and their walk that gives a frame its samples | `src/camera/orbbec_imu.*` |
   | the GPU array, the one owner of device memory; the GPU image, an array of pixels; the pinned host buffer, the pool that closes, a frame's upload, a GPU step's result, their messages, a stage kernel's block of threads and launch | `src/gpu/gpu_array.hpp`, `src/gpu/gpu_image.hpp`, `src/gpu/pinned_buffer.hpp`, `src/gpu/gpu_pool.*`, `src/gpu/gpu_frame.*`, `src/gpu/step_result.hpp`, `src/gpu/gpu_messages.*`, `src/gpu/kernel_launch.cuh` |
   | organized points from a frame's depth, their validity mask | `src/geometry/organized_points.*` |
   | the organized points' normals in image space | `src/geometry/organized_normals.*` |
   | the frame's pyramid: the number of levels, the intensity of a color, the coarser levels, their sizes and their intrinsics; the ray cast takes the number of levels, the intensity of a color (`frame_pyramid::intensity_of()`), a level's size (`level_size()`) and its intrinsics (`intrinsics_at_level()`) from there too; the rule of one surface the stages share | `src/geometry/frame_pyramid.*`; `src/geometry/one_surface.cuh` |
   | the sparse TSDF (docs/methods/sparse-tsdf.md): its voxel, its parameters (the voxel size, the truncation's rule, the far depth) and their check against one frame's bound, its hash of blocks, its pool and the division of the memory budget, the allocation of the blocks near a frame's surface, the integration of a frame, the ray cast with the model's surface it writes at each level, the extraction of the surface's points with its buffer, its choice (the grid's thinning, the blocks touched since a stamp) and result, the one conversion of the GPU's surface point into a model's point; the hash's insertion, lookup and the block a slot holds as device code, which every fusion kernel calls | `src/fusion/sparse_tsdf.*`; `src/fusion/block_hash.cuh` |
   | the 3x3 eigensolver, `float3` arithmetic and the conversion of Eigen's vector into a `float3` | `src/geometry/symmetric_eigen_3x3.cuh`, `src/geometry/float3_ops.cuh`, `src/geometry/float3_eigen.cuh` |
   | the registration of a frame against a surface (docs/methods/registration.md): the surface's view at each level, a frame's pyramid as a surface, the parameters and their check, the result, the surface's intensity gradient, the association and the system of both terms on the GPU, the levels coarse to fine | `src/registration/registration.*` |
   | tracking (docs/methods/tracking.md): its parameters and thresholds, the check of a registration's result, the step of a transform, the ray cast's surface as the registration takes it, the tracker with its buffers, the live model and the last tracked pose, the tracking result | `src/tracking/tracker.*` |
   | the stage clock (docs/methods/tracking.md, "The stages' times"): the per-frame path's stages and their names, the marks with the measurement's one wait for the GPU, a frame's times | `src/tracking/stage_clock.*` |
   | one iteration's normal equations solved on the host: the sums' positions, the centring, the degeneracy test and the damped step, the constants that judge a system | `src/registration/normal_equations.*` |
   | the SE(3) exponential, the registration's and the pose graph's pose update; the check that a transform is rigid; the one copy of each | `src/registration/se3.*` |
   | refinement's pose graph (docs/methods/refinement.md): its nodes and edges, the robust weight, the damped Gauss-Newton on SE(3) on the host in double, the result, the constants that end a solve | `src/refinement/pose_graph.*` |
   | refinement's fusion of a scan's sequence at the full voxel size (docs/methods/refinement.md, "The fusion of the sequence"): the frames read back and fused at their poses, tracking loss skipped, the extraction counted then written, the trajectory and the accelerometer's samples as read | `src/refinement/sequence_fusion.*` |
   | separation's gravity (docs/methods/separation.md, "Gravity's up"): the samples taken between tracked frames, their turn into the models' frame, the mean, the groups' spread, its constants and the words of each outcome; separation's proposal: the trajectory's meeting point, the window, the support plane and its conditions, the footprint's pieces, the box, its parameters, result and constants; the orthogonal least-squares plane, which distortion_check's measurement, the acceptance tool's box fit and depth_noise's measurement call too; the words gravity and the proposal both refuse with, a number's text and a pose's fault | `src/separation/gravity.*`; `src/separation/proposal.*`; `src/separation/plane_fit.*`; `src/separation/separation_messages.*` |
   | the files a scan keeps (docs/SCAN_FILES.md), with the closed sets of why a scan ended and of gravity's outcome; a scan's time in UTC and its one text; the parameters of each object class | `src/scan/file_io.*`, `model_file.*`, `trajectory_file.*`, `parameters_file.*`; `src/scan/utc_time.*`; `src/scan/object_class.*` |
   | the stages' parameters of an object class's set: tracking's, the registration's and the sparse TSDF's of the live or the full model, the one copy of each conversion | `src/scan/stage_parameters.*` |
   | the scan interface (the spec, "The scan interface"; docs/methods/tracking.md, "The scan's loop"): its commands and their answers, its state and its snapshots, the scan thread with its two locks, a scan's phases, its directory and files (docs/SCAN_FILES.md, "A scan's directory") with gravity estimated at refinement and the sequence's removal when it is not kept, the frames counted as taken or not, the camera's properties refused, taken from the hold's report, the "too close" rule, the live model's pacing and publication | `src/scan/scan_interface.*` |
   | the live model (docs/methods/tracking.md, "The live model"): its value, the spacing's rule, the taking of it, every block or the blocks touched since the last, and the merge of their points on the host | `src/scan/live_model.*` |
   | the web layer's routes and WebSocket (docs/ROUTES.md): the server on its address and port with its accepting thread and socket option, the routes of the commands and the state, the static route with its directory check and media types; the counts of the scan interface's news (`scan_news`), the newest messages laid out once for every browser, each browser's loop with its poll and the pings it may miss | `src/web/web_server.*` |
   | the web layer's messages (docs/ROUTES.md, "Messages"): the state and a command's answer as JSON, the escape of a JSON string, the live model's binary layout | `src/web/messages.*` |
   | the page an operator makes a scan on (docs/ROUTES.md, "The page's files"): its parts and their words (`index.html`), its look (`page.css`); the WebSocket with its reconnection, the state shown by phase, the commands with their refusals, the discard's confirmation, the "too close" notice's smoothing (`scan_page.js`); the 3D view of the live model, the message read, following the camera and turning (`live_view.js`); the copy into the build's page directory with three.js's files (`src/CMakeLists.txt`, "The page") | `src/web/page/*` |
   | the acceptance tool's box fit: the box's axes from the normals, its faces, their planes and spreads, its edges, its refusals and the lines the hardware acceptance reads | `apps/box_measure.*` |
   | `depth_noise`'s measurement of c (docs/methods/depth-noise.md): the sequence read twice, each pixel's sums, the pixels kept, the depth bins, the rounding's share, the drift between the halves, the spatial error from the plane, the fit of c, its refusals and the lines the measurement on the camera reads | `apps/depth_noise_measure.*` |
   | `orbbec_viewer`'s loop on a frame source: the first frame, its GPU path (a stream, the organized points, the snapshot's normals), the report line, the window, the snapshots as a model's PLY, the sequence it writes | `apps/viewer_loop.*` |
   | the check of the directory a command line names for what a program writes, `orbbec_viewer`'s `--sequence` and the replay tool's `--scans`: it exists, is a directory, holds nothing and can be written; the service's `--scans`, which may hold scans: it exists, is a directory and can be written | `apps/output_directory.*` |
   | the service's command line (docs/SERVICE.md, "Options"): its options with their defaults, usage and refusals, the start of every line it prints, and the reading of a port that `scan_in_progress` shares | `apps/service_options.*` |
   | the signals that stop the programs that hold the camera, orbbec_viewer and the service, with their names | `apps/stop_signals.hpp` |
   | the service's parts (docs/SERVICE.md): the stop signals, its log with its lines, the hold's lines in the log, the composition of the hold, the writer, the news, the scan interface and the web layer with their order, the serving until a stop signal; the program on the Orbbec camera; the question whether a scan is in progress, with its exits | `apps/service.*`; `apps/scanner_service.cpp`; `apps/scan_in_progress.cpp` |
   | the replay tool's command line, its usage and refusals, and the set its scan runs with, an object class's or a kept scan's; its run of the scan interface on a sequence, the time lines and the last line (docs/methods/tracking.md, "The replay tool's lines") | `apps/replay_options.*`; `apps/replay.cpp` |
   | the tree's commit the programs are built from, for a scan's parameters, written at each build by `apps/build_commit.cmake` | `apps/build_commit.*` |
   | the tests' synthetic source of frames, their one renderer with its pinhole ray and first hit, the depth noise from a fixed seed, a run of frames without depth, the IMU's samples along its trajectory with the hand's acceleration, a point's distance from a known surface, the trajectories it renders along, the Fibonacci lattice the tests sample a sphere on | `tests/synthetic_source.*` |
   | the tests' checks, counted into the program's summary line and exit status; their files (a temporary directory, a file's bytes, a limit that stands for a full disk); their GPU helpers (a stream, the pool's grant, CUDA's last error, a step's refusal, an image's download); their reader of a model's PLY, apart from the scanner's; their run of a program through the shell, for a program's exits | `tests/check.hpp`; `tests/test_files.hpp`; `tests/gpu_test_helpers.hpp`; `tests/ply_reader.hpp`; `tests/program_run.hpp` |
   | the tests' scan made in process with a writer that waits for room, the one scan of a scans directory, a directory's names, and the check that two scans' directories hold the same scan but for the parameters | `tests/scan_comparison.hpp` |
   | the web tests' scene along the walk, the source that paces the synthetic one at the camera's rate, the set their scans run with, the state message's keys and its reader | `tests/web_test_helpers.hpp` |
   | the tests' stand-in for the camera's run-time properties: a table under OrbbecSDK's numbers in front of a frame source, which logs every access | `tests/stand_in_camera.hpp` |
   | the tests' copy of a frame, to check a source left it as it was; a frame given again under another index and timestamp; the comparisons of IMU samples and of poses bit for bit | `tests/frame_copy.hpp` |
   | the sparse TSDF tests' lookups of blocks, their readings of the usage, the method's starting truncation, a pose as the sparse TSDF takes it, a frame's buffers on the GPU, the extraction of the model's points with their download, and the points' bits as a sorted list, to compare two extractions or models as sets | `tests/sparse_tsdf_helpers.hpp` |
   | the sparse TSDF tests that fuse frames: a case's budget, an object class's model, the fusion of a frame, walls facing the camera, the scenes of a sphere on a plane with their walks, a point's nearest known surface | `tests/sparse_tsdf_scenes.hpp` |
   | the tests that register frames: the normals' parameters, a measured range for the organized points, the object classes' starting values of the registration, the tabletop object's and the large object's scenes, poses and known motions, a result's printing and its check against the known motion | `tests/registration_cases.hpp` |
   | the stages' test frames: the camera's stream sizes and intrinsics, the depth ranges the stages' tests were measured at, the tabletop sphere, a scene's frames rendered, with depth noise or without, frames built from functions of the pixel | `tests/stage_frames.hpp` |
   | the stages' tests' checks of an organized picture: a pixel's index, a float3 in double, the zero mark, angles, medians, the rule of one surface in double with its margin | `tests/stage_checks.hpp` |
   | the tests' rigid transforms in double: one built from an axis, an angle and a shift, how far one lies from a known one, its tolerance and check | `tests/transform_checks.hpp` |

5. **Log lines are an interface.** A tool that reads a program's output holds the strings
   it reads: before a log line, a refusal or a dump key changes, grep the tools and tests
   for it and change the reader with it.
6. **No speculative generality.** No stubs, no second implementation, no parameter with one
   value at every call site. Dead code is deleted, not commented out. An experiment without
   a measured benefit is removed and its finding kept in the document that owns it.
7. **Better numerics replace the old path.** A change that keeps the result but improves
   conditioning or exactness becomes the one path, proven within a stated tolerance; it is
   no switch.
8. **Nothing shipped from arithmetic alone.** A value enters a description or tuning file
   with the measurement that produced it written above it.

## 8. Where things are documented

| document | holds |
|---|---|
| `CLAUDE.md` | the one-page entry point: what this is, §3's rules in short, where to go next |
| `README.md` | what the tree is, its layout, how to build, install and run it |
| `ROADMAP.md` | what to do next, in order, with a definition of done and who can do each step |
| `docs/HARDWARE.md` | the board and the camera: platform, memory, what the camera offers and reports, connections, permissions, kernel log lines, and each stage's time on the board |
| `CODING_STANDARDS.md` | the style guides, what they leave to this tree, the rules for the rewrite's code, the checkers |
| `GLOSSARY.md`, `docs/adr/` | the scanner's terms; the decisions in force and why |
| `docs/SCAN_FILES.md` | the formats of the files a scan keeps: the models (PLY), the trajectory, the parameters |
| `docs/methods/` | each stage's method document, written before its code: its method, parameters and sources, and what it does not carry from the prototype; today `organized-points.md`, `normals.md`, `frame-pyramid.md`, `registration.md`, `sparse-tsdf.md`, `tracking.md` (with the stage clock and the replay tool's lines), `refinement.md`, `separation.md` and `imu-samples.md`, the IMU's samples with the frames; besides them the method of a tool's measurement, the depth noise's, `depth-noise.md` |
| `docs/ROUTES.md` | the web layer's HTTP routes, its WebSocket and the layout of every message, which the page, the service and the web tests follow; the page's static files, the threads |
| `docs/SERVICE.md` | the scanner service: its options, the camera during a scan, its stop with its order and worst case, its log lines, and `scan_in_progress` with its exits, which the install tool and the service's live check read |
| `docs/SEQUENCE_FILE.md` | the format of a sequence: a scan's frames with their IMU samples and poses, which the viewer and the scan interface write, refinement and the replay tool read back |
| `docs/agents/` | how the workflow skills in `.claude/skills/` find the tracker and the domain docs |

Rules for these documents: each opens with the notice line above; plain, concise English,
present tense, current truth only, history in git. One fact lives in one document and the
others link to it. Every number carries a tag. When something turns out wrong, state what
is right and drop the retraction. A claim carried over from an older document is checked
against the tree first, above all one that says something is missing.
