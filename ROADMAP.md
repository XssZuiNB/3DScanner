# Roadmap

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

What to do next, in order. Each step has a definition of done and says who can do it:
**agent** (offline, from a `ready-for-agent` ticket), **hardware** (the board and the user's
go), **decision** (the user's). A finished step is deleted; its outcome goes into the
document that owns the fact.

## 1. Clean up the CUDA_3D tree (agent)

The inventory is done: `.scratch/cleanup/inventory.md`, read by the user 2026-11-13. Decided
by the user the same day, in this order, each step done when `tools/verify.sh` passes:

1. **Dead code:** delete `apps/test_color_icp.cpp`, `scripts/`, `graph.png`,
   `data/cloud_bin_*.pcd`, `src/visualizer/`, `src/movement_detection/`,
   `src/geometry/cuda_clustering.*` with its `point_cloud` wrappers, the unused functions and
   classes the inventory lists (the unused part of `math.cuh` among them), RealSense
   (`realsense_device.*`, `apps/app.cpp`) and PCL support, and the CMake settings nothing
   needs. `data/frag_115.ply` and `frag_116.ply` stay as the fixture of a future colored-ICP
   test.
2. **Directory layout** (decision, then agent): a layout for the scanner, checked by the user,
   the files moved.
3. **The rules for the code that stays** (agent): AGENTS.md §7 for `orbbec_device`,
   `orbbec_viewer`, the test and the tools, and a compiler warning set. The rest gets the rules
   when the rewrite replaces it.

## 2. The rewrite's requirements (decision)

The rewrite starts with its list of requirements, which the user checks. `/grill-with-docs`
on the scanner: what it scans (objects, rooms), how the camera moves (hand, turntable), what
it delivers (point cloud, mesh, which format), accuracy and speed, where it runs and who
works it, and which parts of CUDA_3D carry over (its algorithms, `gca::device`, the
`movement_detection/` module, `scripts/`, `graph.png`). Fixed already: `src/util/math.cuh` is
CC BY-NC-SA 3.0 (README.md, "License") and is replaced, not carried over. Done when the user
has checked the list and `/to-spec` has turned it into a spec.

## Candidates, once measured

Each needs a measurement that shows the cost first (docs/HARDWARE.md has the first ones).

* `create_from_rgbd` takes 19.4 to 19.9 ms of the 33 ms frame at 1280x800 (docs/HARDWARE.md):
  where the time goes (the bilateral filter, the invalid-point removal, the upload).
* Undistort color, or check that the SDK's aligned output needs none.
* The half-pixel deprojection of the point-cloud kernel (AGENTS.md §6).
* A viewer without PCL.
* A compiler warning set (CODING_STANDARDS.md).
