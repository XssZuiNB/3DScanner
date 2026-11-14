# 3DScanner

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

A 3D scanner on a Jetson Orin Nano with an Orbbec Gemini 2 L RGB-D camera, being rewritten
from CUDA_3D, a GPU point-cloud library (point clouds from RGB-D, voxel filter, normals, clustering,
[colored ICP](https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf);
its RealSense SLAM demo: https://youtu.be/5NcDEhguzGU). Agents start at [CLAUDE.md](CLAUDE.md).

What the rewrite builds, none of it yet: an operator walks the hand-held camera around a
tabletop object or a large object and works the scan from a web page the Jetson serves; the
result is a colored model of the object, later a mesh and measurements
([GLOSSARY.md](GLOSSARY.md), [docs/adr/](docs/adr/), [ROADMAP.md](ROADMAP.md)).

## Layout

| directory | what |
|---|---|
| `src/` | one library per directory: `types/` (camera parameters, a frame's description, the camera's pose and console colors, header only), `camera/`, `gpu/`, `geometry/`, `registration/`, `scan/` (the files a scan keeps, the parameters of each object class); `camera/` holds two, so that only the Orbbec camera links OrbbecSDK: `camera` (the frame, the frame source seam, the sequence and the Orbbec camera's rules, which need no camera) and `orbbec_camera` (the Orbbec camera as a frame source) |
| `apps/` | `orbbec_viewer` (camera test, live point cloud), its command line, `viewer_options`, and its loop on a frame source, `viewer_loop`; `distortion_check` (whether the aligned frames are distorted, from a snapshot of a checkerboard), its measurement, `checkerboard_measure`, and its command line, `distortion_options` |
| `tests/` | offline tests, run by ctest; they need the GPU, no camera; the synthetic source that renders their frames; the Orbbec camera's rules fed with plain data; the viewer's loop on the synthetic source |
| `tools/` | fetch the dependencies, build, verify (AGENTS.md §4) |
| `data/` | the two colored-ICP fragments, `frag_115.ply` and `frag_116.ply` |
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

A snapshot of a flat checkerboard, measured without the camera; the intrinsics and the depth
scale are the ones the viewer prints at that stream size:

    <repo>/build/apps/distortion_check --fx 612.369 --fy 612.286 --cx 635.814 --cy 398.267 \
        --depth-scale 0.001 --square-mm 30 --corners 11x7 cloud_000.ply.color.png cloud_000.ply.depth.png

## License

[0BSD](LICENSE). Dependencies carry permissive licenses only
([ADR-0002](docs/adr/0002-own-reconstruction-code.md)); the build refuses Eigen's LGPL files
(CODING_STANDARDS.md, "What the guides leave to this tree"). Except:

* The workflow skills in `.claude/skills/` are MIT (`.claude/skills/LICENSE-mattpocock-skills`).
* `data/`: Open3D test data [?], license unchecked.
