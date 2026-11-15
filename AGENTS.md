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
the decisions: docs/adr/; the next step: ROADMAP.md). Built so far: the frame, the frame source
seam and the sequence on disk (`src/camera/`); the GPU image, the pool that closes and a
frame's upload (`src/gpu/`); the organized points from a frame's depth
(`src/geometry/organized_points.*`), their normals (`src/geometry/organized_normals.*`) and the
frame's pyramid of them with the intensity (`src/geometry/frame_pyramid.*`); the registration of
a frame against a surface by colored ICP, coarse to fine over the pyramid
(`src/registration/registration.*`); the sparse TSDF's hash of blocks, its pool and the allocation
of the blocks near a frame's surface (`src/fusion/`); refinement's pose graph of keyframes and its
solve (`src/refinement/pose_graph.*`); the files a scan keeps and the parameters of each object
class (`src/scan/`). The tree is CUDA_3D cleaned up (dead code, RealSense and PCL
removed), made to build here and given the Orbbec camera. It is the rewrite's prototype: its
methods carry over, and its code is the test oracle until new code replaces it
(CODING_STANDARDS.md, "The rewrite's code").

The tree today: `src/` is eight static libraries and the header-only `types`, each linking only
the ones below it: `types`, `camera`, `orbbec_camera`, `gpu`, `geometry`, `registration`,
`fusion`, `refinement`, `scan` (`src/CMakeLists.txt`; `gpu` links `camera` for the frame it
uploads, `scan` for the POSIX file calls; `fusion` links `geometry` for a frame's organized
points; `refinement` links `registration` for the SE(3) exponential). There is one
library per directory, except `src/camera/`, which holds two so that only the Orbbec camera
links OrbbecSDK: `camera`, the frame, the frame source seam, the sequence on disk and the Orbbec
camera's rules that need no camera, which the offline tests link without OrbbecSDK; and
`orbbec_camera`, where `orbbec_device` is the camera as a frame source and delivers depth
aligned to color, so the GPU code sees one resolution and one set of intrinsics.
`orbbec_viewer` in `apps/` runs its loop (`apps/viewer_loop.*`) on the camera as a frame source,
and the offline tests run the same loop on their synthetic source; the loop turns each frame's
color to BGR8 for the prototype's GPU code and, with `--sequence`, writes the frames it takes
into a sequence. `distortion_check` reads the viewer's snapshots without the camera; there is no
daemon.

## 2. Before you run anything

**Check the machine state.** These read only:

    lsusb -t                                   # the camera, its bus and speed (5000M = USB 3)
    pgrep -af 'orbbec|ob_'                     # a program holding the camera
    journalctl -k --since -10min | grep -iE 'usb|uvc|nvgpu'

`dmesg` is closed to the user [M 2026-11-13]; `journalctl -k` works (group `adm`).

**Root.** Nothing in the daily work: the udev rule gives the user the camera
(docs/HARDWARE.md, "Permissions"), and tools/fetch_deps.sh puts the SDK and Eigen in the user's
home. An `apt install` or a new udev rule goes to the user as a command to run. Ask the user for
the password when one is needed; never write it into a file, a script, a command line or a
commit.

**One writer.** The program that opens the camera through OrbbecSDK claims its USB interfaces;
a second program cannot open it then [?]. Nothing locks it: check with `pgrep` above before
starting one.

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
| `build/apps/orbbec_viewer --list` | opens the camera, reads its info and stream profiles, starts no stream [S: orbbec_device.cpp]; what OrbbecSDK sends on open is unverified [?] |
| `build/apps/orbbec_viewer` with an argument it does not know (`--help` too), an option without its value, or a `--sequence` directory that does not exist, holds something or cannot be read or written | prints the reason and the usage line and exits 2; `main` reads the options, and looks at that directory without changing it, before it opens the camera or the GPU or installs a signal handler [S: orbbec_viewer.cpp, `main`; the refusals: ctest `viewer_options`]. The load-time code of OrbbecSDK, OpenCV and CUDA runs before `main`; what it does is unverified [?] |
| `build/apps/orbbec_viewer [--frames N]` | streams color and depth with depth-to-color alignment; sets no option of the camera [S]; `s`, or the last of N frames headless, writes `cloud_NNN.ply`, `.color.png`, `.depth.png` into the current directory: run it in a scratch directory. N counts delivered frames: a timeout or a failure prints its line, counts no frame and the loop goes on [S: viewer_loop.cpp; ctest `viewer_loop`] |
| `build/apps/orbbec_viewer --sequence DIR [...]` | as above, and writes every frame it takes, the first included, unchanged, into one file, `DIR/frames.sequence` (docs/SEQUENCE_FILE.md), each marked as tracked by nothing. These are frames of the camera, so DIR lies in the user's home, outside the repository and every worktree, and not under `/tmp`, which is emptied at boot (§2, "Privacy"; §6), e.g. `~/3dscanner-sequences/<name>/` [?: a choice]; it is not the current directory, where the snapshots and OrbbecSDK's log go. About 38 MB/s at 640x400 and 154 MB/s at 1280x800, 30 fps, so a 5-minute walk takes about 12 GB and 46 GB [S: computed, docs/SEQUENCE_FILE.md, "Layout"]. The report line goes on with ` \| queue full <count>`, the frames since the last report the writer's queue had no room for, which were shown but not written; without `--sequence` the line is unchanged. The run ends on the frame limit, `q`, ESC, Ctrl-C (SIGINT), SIGTERM (a kill, `timeout`) or SIGHUP (a closed terminal or SSH session): all four signals set the loop's exit flag, so the last frame is written and the file gets its end [S: orbbec_viewer.cpp, `install_stop_handlers`; the exit flag: ctest `viewer_loop`]. Then `Wrote <A> frames to DIR/frames.sequence, <F> not written because the writer's queue was full`; a file that cannot be created prints the writer's reason, and a failed write or close its reason followed by `; <W> of <T> frames taken written, <L> lost from the writer's queue, <F> not written because the writer's queue was full`, each with exit 1 [S: viewer_loop.cpp; ctest `viewer_loop`] |
| any program using OrbbecSDK | writes `Log/OrbbecSDK.log.txt` into the current directory [M 2026-11-13]; closing it hands the camera back to uvcvideo, which logs the enumeration lines again (docs/HARDWARE.md) |
| OrbbecSDK's examples, OrbbecViewer | not built here; `ob_device_firmware_update`, `ob_device_optional_depth_presets_update`, `ob_multi_devices_firmware_update` and OrbbecViewer's firmware and preset pages WRITE the camera's flash |
| `nsys profile` with CPU sampling (on in the one run seen; `--sample=none` turns it off [?: nsys's documentation, not read here]), `perf record` | samples the CPU through the kernel's perf events; the kernel may log `perf: interrupt took too long ... lowering kernel.perf_event_max_sample_rate` and keep the lowered rate [M 2026-11-14: one `nsys profile` run]. The line is benign only while such a profiler runs (docs/HARDWARE.md, "Kernel log"): say in your report that one ran, with its time |

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
a -j6 build froze the board on 2026-11-13 [M 2026-11-13]. One heavy build runs at a time.
While it runs, `~/.local/state/3dscanner/monitor.sh` (outside the tree) logs available
memory, free swap, the over-current counters, the input power and the hottest zone every 2 s
into `monitor.log` beside it, syncing each line so that a hard reset leaves the last state on
disk [S: the script]; start it in the background and end it with the session.

**ccache.** Every compile of the three tools goes through ccache (CODING_STANDARDS.md,
"Tools"), set in the build directory's compiler launcher, so that the builds of one tree hit
what another tree's builds wrote (`configure_build` in `tools/build_common.sh`): a verify
after `tools/build.sh` and the mutant copy's first build each hit 20 of 20 compiles
[M 2026-11-13: `ccache -s`, ccache's log]; verify takes 196 s with an empty cache and 9.5 s
with a warm one [M 2026-11-13]. On a hit ccache replays the compile's warnings,
g++'s and nvcc's [M 2026-11-13], so verify's warning check sees them on a hit too.

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

Free choices: the stream size 640x400 at 30 fps [?: a choice; it exists on USB 2], the depth range 0.15 m to
3.0 m in `apps/viewer_loop.cpp` [?: a choice].

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
  `thrust::pair<float, float3>(a, b)` [M 2026-11-13, CUDA 12.6].
* **The point-cloud kernel deprojects pixel u at u - 0.5**
  (`cuda_point_cloud_factory.cu`, `depth_uv`), half a pixel off the convention OrbbecSDK's own
  code uses for its intrinsics, pixel centres at whole numbers (docs/HARDWARE.md, "What the
  camera is and reports"): about 1 mm at 1 m with fx = 500 [S]. Pinned as is by
  `tests/test_point_cloud_from_rgbd.cpp`. The rewrite's organized points deproject pixel u at u
  (docs/methods/organized-points.md); `tests/test_organized_points.cpp` states the half pixel
  between the two.

## 7. How the code is written

[CODING_STANDARDS.md](CODING_STANDARDS.md) holds the style guide of each language and the
checkers; these are the rules no checker holds.

1. **A file starts with SPDX, then its name and an overview of what its module does**, and
   nothing else; a module is one source file and its header. C and C++: `/// @file <name>`.
   A script puts its SPDX line under the shebang, then `# <name> - <overview>`; a CMake file
   starts with its SPDX line, then `# <path> - <overview>`, the path from the tree's root.
   A trap, an ownership rule or a cache note goes into the comment of the function it
   concerns.
2. **Every function's comment states its purpose, every parameter and its return value**:
   a public function in its header above its declaration, a file-local one above its
   definition. C and C++: a `///` comment whose first sentence is the purpose, one `@param`
   for each parameter, even one whose name looks obvious, and `@return` for every function
   that returns a value. Shell: the Google Shell Style Guide's function comment. Python: the
   Google Python Style Guide's docstring. A fact in a comment carries the same tags as the
   documents. A kernel and a device function are functions like any other.

   The rewrite's code states more, each part wherever it applies; the prototype's code keeps
   the three parts above until it is replaced.

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
   | camera parameters, a frame's description and the camera's pose (plain structs), the one check that intrinsics can be used, console colors | `src/types/` |
   | the frame; the frame source seam | `src/camera/frame.*`; `src/camera/frame_source.hpp` |
   | the sequence on disk (docs/SEQUENCE_FILE.md): its layout, writer and reader, the comparison of two frames' descriptions bit for bit (`src/camera/sequence_format.*`); the POSIX file calls the writers and readers share | `src/camera/sequence_*`; `src/camera/posix_file.*` |
   | Orbbec: alignment, color decoding, OrbbecSDK's report as plain data | `src/camera/orbbec_device.*` |
   | Orbbec: the choice of streams, the checks of a frame, the rule for timestamps, the refusals of an opening | `src/camera/orbbec_rules.*` |
   | the GPU array, the one owner of device memory; the GPU image, an array of pixels; the pinned host buffer, the pool that closes, a frame's upload, a GPU step's result, their messages, a stage kernel's block of threads and launch | `src/gpu/gpu_array.hpp`, `src/gpu/gpu_image.hpp`, `src/gpu/pinned_buffer.hpp`, `src/gpu/gpu_pool.*`, `src/gpu/gpu_frame.*`, `src/gpu/step_result.hpp`, `src/gpu/gpu_messages.*`, `src/gpu/kernel_launch.cuh` |
   | the prototype's frames and camera parameters on the GPU, CUDA helpers, GPU checks | `src/gpu/` |
   | organized points from a frame's depth, their validity mask | `src/geometry/organized_points.*` |
   | the organized points' normals in image space | `src/geometry/organized_normals.*` |
   | the frame's pyramid: the intensity, the coarser levels, their sizes and their intrinsics; the rule of one surface the stages share | `src/geometry/frame_pyramid.*`; `src/geometry/one_surface.cuh` |
   | the sparse TSDF (docs/methods/sparse-tsdf.md): its voxel, its parameters (the voxel size, the truncation's rule, the far depth) and their check against one frame's bound, its hash of blocks, its pool and the division of the memory budget, the allocation of the blocks near a frame's surface; the hash's insertion and lookup as device code, which every fusion kernel calls | `src/fusion/sparse_tsdf.*`; `src/fusion/block_hash.cuh` |
   | point clouds: creation, filters, normals, search; the 3x3 eigensolver, `float3` arithmetic and its Eigen conversion | `src/geometry/` |
   | the registration of a frame against a surface (docs/methods/registration.md): the surface's view at each level, the parameters, the result, the surface's intensity gradient, the association and the system of both terms on the GPU, the levels coarse to fine | `src/registration/registration.*` |
   | one iteration's normal equations solved on the host: the sums' positions, the centring, the degeneracy test and the damped step, the constants that judge a system | `src/registration/normal_equations.*` |
   | the SE(3) exponential, the registration's and the pose graph's pose update; the check that a transform is rigid; the one copy of each | `src/registration/se3.*` |
   | the prototype's colored ICP, the oracle until it is retired; the 3x3 LDLT solve | `src/registration/` |
   | refinement's pose graph (docs/methods/refinement.md): its nodes and edges, the robust weight, the damped Gauss-Newton on SE(3) on the host in double, the result, the constants that end a solve | `src/refinement/pose_graph.*` |
   | the files a scan keeps (docs/SCAN_FILES.md); the parameters of each object class | `src/scan/` |
   | `orbbec_viewer`'s loop on a frame source: the first frame, the report line, the window, the snapshots, the sequence it writes | `apps/viewer_loop.*` |
   | the tests' synthetic source of frames, their one renderer with its pinhole ray and first hit, a mesh's frames (the colored-ICP fixture's), a point's distance from a known surface, the trajectories it renders along | `tests/synthetic_source.*` |
   | the tests' copy of a frame, to check a source left it as it was | `tests/frame_copy.hpp` |
   | the sparse TSDF tests' lookups of blocks, their readings of the usage and the method's starting truncation | `tests/sparse_tsdf_helpers.hpp` |
   | the stages' test frames: the camera's stream sizes and intrinsics, the tabletop sphere, a scene's frames rendered, frames built from functions of the pixel | `tests/stage_frames.hpp` |
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
| `docs/HARDWARE.md` | the board: platform, connections, permissions, kernel log lines |
| `CODING_STANDARDS.md` | the style guides, what they leave to this tree, the rules for the rewrite's code, the checkers |
| `GLOSSARY.md`, `docs/adr/` | the scanner's terms; the decisions in force and why |
| `docs/SCAN_FILES.md` | the formats of the files a scan keeps: the models (PLY), the trajectory, the parameters |
| `docs/methods/` | each stage's method document, written before its code: its method, parameters and sources, and what it does not carry from the prototype; today `organized-points.md`, `normals.md`, `frame-pyramid.md`, `registration.md`, `sparse-tsdf.md` and `refinement.md` |
| `docs/SEQUENCE_FILE.md` | the format of a sequence: a scan's frames with their poses, which the viewer writes and the scanner and the replay tool are planned to write or read |
| `docs/agents/` | how the workflow skills in `.claude/skills/` find the tracker and the domain docs |

Rules for these documents: each opens with the notice line above; plain, concise English,
present tense, current truth only, history in git. One fact lives in one document and the
others link to it. Every number carries a tag. When something turns out wrong, state what
is right and drop the retraction. A claim carried over from an older document is checked
against the tree first, above all one that says something is missing.
