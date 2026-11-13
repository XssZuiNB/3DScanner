# 3DScanner

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

A 3D scanner on a Jetson Orin Nano with an Orbbec Gemini 2 L RGB-D camera, being rewritten
from CUDA_3D, a GPU point-cloud library (point clouds from RGB-D, voxel filter, normals, clustering,
[colored ICP](https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf);
its RealSense SLAM demo: https://youtu.be/5NcDEhguzGU). Agents start at [CLAUDE.md](CLAUDE.md).

## Layout

| directory | what |
|---|---|
| `src/` | the library `gca_lib`: `camera/` (Orbbec, RealSense behind `gca::device`), `cuda_container/`, `geometry/`, `registration/`, `movement_detection/`, `visualizer/` (PCL only), `util/` |
| `apps/` | `orbbec_viewer` (camera test, live point cloud); `app.cpp`, CUDA_3D's RealSense demo, built only with RealSense and PCL |
| `tests/` | offline tests, run by ctest; they need the GPU, no camera |
| `tools/` | fetch the dependencies, build, verify (AGENTS.md §4) |
| `data/` | CUDA_3D's sample point clouds |
| `docs/` | the hardware, the workflow's tracker and domain conventions |

## Build

Needs CUDA 12 (12.6 tested), CMake 3.18+, OpenCV 4. Then:

    tools/fetch_deps.sh     # OrbbecSDK v2 and Eigen into ~/.local/share/3dscanner
    tools/build.sh          # build/
    tools/verify.sh         # every offline check (AGENTS.md §4)

librealsense and PCL are used when CMake finds them.

## Run

Once per machine, as root: the Orbbec udev rule, from the SDK
(`~/.local/share/3dscanner/OrbbecSDK_v2*/shared/install_udev_rules.sh`). Then, after AGENTS.md
§2 and §3, from a scratch directory:

    <repo>/build/apps/orbbec_viewer --list     # device and stream profiles
    <repo>/build/apps/orbbec_viewer            # color | depth | overlay; s saves a PLY, q quits

## License

[0BSD](LICENSE), except:

* `src/util/math.cuh` comes from VoxelHashing and is CC BY-NC-SA 3.0: attribution,
  **non-commercial**, share-alike. The rewrite replaces it (ROADMAP.md).
* `src/util/eigen_disable_bad_warnings.hpp` derives from Eigen and is MPL-2.0.
* The workflow skills in `.claude/skills/` are MIT (`.claude/skills/LICENSE-mattpocock-skills`).
* `data/`: Open3D test data [?], license unchecked.
