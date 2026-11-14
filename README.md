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
| `src/` | one library per directory: `types/` (camera parameters and console colors, header only), `camera/` (Orbbec behind `gca::device`), `gpu/`, `geometry/`, `registration/` |
| `apps/` | `orbbec_viewer` (camera test, live point cloud) and its command line, `viewer_options` |
| `tests/` | offline tests, run by ctest; they need the GPU, no camera |
| `tools/` | fetch the dependencies, build, verify (AGENTS.md §4) |
| `data/` | the two colored-ICP fragments, `frag_115.ply` and `frag_116.ply` |
| `docs/` | the hardware, the decisions (`adr/`), the workflow's tracker and domain conventions |

## Build

Needs CUDA 12 (12.6 tested), CMake 3.25.2+ (4.4.4 tested), OpenCV 4 and ccache
(CODING_STANDARDS.md, "Tools"). The tree builds as C++20 for C++ and CUDA, and Eigen's LGPL
files are shut out (CODING_STANDARDS.md, "What the guides leave to this tree").
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

## License

[0BSD](LICENSE). Dependencies carry permissive licenses only
([ADR-0002](docs/adr/0002-own-reconstruction-code.md)); the build refuses Eigen's LGPL files
(CODING_STANDARDS.md, "What the guides leave to this tree"). Except:

* The workflow skills in `.claude/skills/` are MIT (`.claude/skills/LICENSE-mattpocock-skills`).
* `data/`: Open3D test data [?], license unchecked.
