# 3DScanner

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

A 3D scanner on a Jetson Orin Nano with an Orbbec Gemini 2 L RGB-D camera, being rewritten
from CUDA_3D, a GPU point-cloud library (point clouds from RGB-D, voxel filter, normals, clustering,
[colored ICP](https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf);
its RealSense SLAM demo: https://youtu.be/5NcDEhguzGU). Agents start at [CLAUDE.md](CLAUDE.md).

What the rewrite builds, of which so far only parts exist (libraries and their tests, no scan
yet): an operator walks the hand-held camera around a tabletop object or a large object and
works the scan from a web page the Jetson serves; the result is a colored model of the object,
later a mesh and measurements ([GLOSSARY.md](GLOSSARY.md), [docs/adr/](docs/adr/), [ROADMAP.md](ROADMAP.md)).

## Layout

| directory | what |
|---|---|
| `src/` | one library per directory: `types/` (camera parameters, a frame's description, the camera's pose, an IMU sample, a model's point and console colors, header only), `camera/`, `gpu/` (the GPU array and image, the pool that closes, a frame's upload), `geometry/` (a frame's organized points, their normals and the frame's pyramid with the intensity), `registration/` (the registration of a frame against a surface by colored ICP coarse to fine, its normal equations and the SE(3) exponential), `fusion/` (the sparse TSDF: today its hash of blocks, its pool, the allocation of the blocks near a frame's surface, the integration of a frame, the ray cast of the model's surface and the extraction of the surface's points), `tracking/` (each frame registered against the live model, checked and fused only when the check passes), `refinement/` (today the pose graph of keyframes and its solve, on the host), `separation/` (today the proposal of the support plane and the box from a full model and its trajectory, and the least-squares plane, on the host), `scan/` (the files a scan keeps, the parameters of each object class); `camera/` holds two, so that only the Orbbec camera links OrbbecSDK: `camera` (the frame with its IMU samples, the frame source seam, the sequence and the Orbbec camera's rules and its IMU's queue, which need no camera) and `orbbec_camera` (the Orbbec camera as a frame source) |
| `apps/` | `orbbec_viewer` (camera test: each frame's organized points live, a snapshot as a model's PLY), its command line, `viewer_options`, and its loop on a frame source, `viewer_loop`; `distortion_check` (whether the aligned frames are distorted, from a snapshot of a checkerboard), its measurement, `checkerboard_measure`, and its command line, `distortion_options`; `box_fit` (the acceptance tool: the edges of the box in an object model), its measurement, `box_measure`, and its command line, `box_fit_options`; `depth_noise` (the depth noise's c from a sequence taken with the camera at rest), its measurement, `depth_noise_measure`, and its command line, `depth_noise_options` |
| `tests/` | offline tests, run by ctest; they need the GPU, no camera; the synthetic source that renders their frames; the Orbbec camera's rules fed with plain data; the viewer's loop on the synthetic source |
| `tools/` | fetch the dependencies, build, verify (AGENTS.md §4) |
| `docs/` | the hardware, the files a scan keeps, the sequence, the decisions (`adr/`), each stage's method (`methods/`), the workflow's tracker and domain conventions |

## Build

Needs CUDA 12 (12.6 tested), CMake 3.25.2+ (4.4.4 tested), OpenCV 4 (4.8.0 tested) with
core, imgproc, imgcodecs, highgui, and calib3d with the features2d and flann it links, which
`distortion_check` adds for the checkerboard, and ccache (CODING_STANDARDS.md, "Tools"). The
tree builds as C++20 for C++ and CUDA, and Eigen's LGPL files are shut out
(CODING_STANDARDS.md, "What the guides leave to this tree").
Then:

    tools/fetch_deps.sh     # OrbbecSDK v2 and Eigen into ~/.local/share/3dscanner
    tools/build.sh          # build/
    tools/verify.sh         # every offline check (AGENTS.md §4)

## Run

Once per machine, as root: the Orbbec udev rule, from the SDK
(`~/.local/share/3dscanner/OrbbecSDK_v2*/shared/install_udev_rules.sh`). Then, after AGENTS.md
§2 and §3, from a scratch directory:

    <repo>/build/apps/orbbec_viewer --list     # device and stream profiles
    <repo>/build/apps/orbbec_viewer            # color | depth | overlay; s saves a PLY, q quits
    <repo>/build/apps/orbbec_viewer --sequence <dir>   # also writes every frame to <dir>/frames.sequence

`<dir>` exists and is empty, in the home directory outside the repository and every worktree,
not under `/tmp`, e.g. `~/3dscanner-sequences/<name>/` (AGENTS.md §3).

A snapshot of a flat checkerboard, measured without the camera; the intrinsics and the depth
scale are the ones the viewer prints at that stream size:

    <repo>/build/apps/distortion_check --fx 612.369 --fy 612.286 --cx 635.814 --cy 398.267 \
        --depth-scale 0.001 --square-mm 30 --corners 11x7 cloud_000.ply.color.png cloud_000.ply.depth.png

The edges of a box scanned and separated, taken from its object model without the camera: its
length, width and height in mm and each face's spread about its plane:

    <repo>/build/apps/box_fit <scan>/<object model>.ply

The depth noise's c, which sets the sparse TSDF's truncation, from a sequence of a plain wall or
floor seen obliquely by the camera at rest (docs/methods/depth-noise.md), without the camera:

    <repo>/build/apps/depth_noise <dir>      # reads <dir>/frames.sequence

## License

[0BSD](LICENSE). Dependencies carry permissive licenses only
([ADR-0002](docs/adr/0002-own-reconstruction-code.md)); the build refuses Eigen's LGPL files
(CODING_STANDARDS.md, "What the guides leave to this tree"). Except: the workflow skills in
`.claude/skills/` are MIT (`.claude/skills/LICENSE-mattpocock-skills`).

| dependency | version | license | where it comes from |
|---|---|---|---|
| OrbbecSDK | v2.10.6 | MIT, with libusb (LGPL-2.1) and live555 (LGPL) statically inside `libOrbbecSDK.so` [B 2026-11-15: `strings`], accepted in ADR-0002 | `tools/fetch_deps.sh`, linked dynamically, not distributed by the tree |
| Eigen | 3.4.0 | MPL-2.0 only (`EIGEN_MPL2_ONLY`) | `tools/fetch_deps.sh` |
| OpenCV | 4.8.0 | Apache-2.0 [S: Ubuntu's copyright file for 4.5.4, which says upstream relicensed to Apache-2.0 from 4.5 on; the board's 4.8.0 package ships no license file [M 2026-11-15: `dpkg -L libopencv`]] | the board's JetPack: `/usr/lib/cmake/opencv4` [M 2026-11-15: the build's CMakeCache] |

The programs load OpenCV 4.8.0 [M 2026-11-15: `ldd build/apps/orbbec_viewer` names
`/lib/libopencv_core.so.408` and the other `libopencv_*.so.408`]; Ubuntu's libopencv-core4.5d
4.5.4 is also installed [M 2026-11-15: `dpkg -l`] and no program of the tree loads it.
