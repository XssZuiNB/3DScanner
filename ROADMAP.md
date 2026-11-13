# Roadmap

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

What to do next, in order. Each step has a definition of done and says who can do it:
**agent** (offline, from a `ready-for-agent` ticket), **hardware** (the board and the user's
go), **decision** (the user's). A finished step is deleted; its outcome goes into the
document that owns the fact.

## 1. Clean up the CUDA_3D tree (agent)

Spec: `.scratch/cleanup/spec.md`, from the inventory (`.scratch/cleanup/inventory.md`) and
the user's decisions of 2026-11-13. In order, each step done when `tools/verify.sh` passes:
offline tests on the library's public API; the dead code, RealSense and PCL deleted
(`data/frag_115.ply` and `frag_116.ply` stay as fixtures); `src/util/math.cuh` replaced by
Eigen and the tree's own `float3` operators; one static library per module; AGENTS.md §7
and a warning set for the code that stays. Done when, on the user's go, `orbbec_viewer` still
runs at 30 fps with the timings of docs/HARDWARE.md.

## 2. The rewrite's requirements (decision)

The rewrite starts with its list of requirements, which the user checks. `/grill-with-docs`
on the scanner: what it scans (objects, rooms), how the camera moves (hand, turntable), what
it delivers (point cloud, mesh, which format), accuracy and speed, where it runs and who
works it, and which parts of CUDA_3D carry over (its algorithms, `gca::device`, the
`movement_detection/` module). Fixed already: `src/util/math.cuh` is
CC BY-NC-SA 3.0 (README.md, "License") and is replaced, not carried over. Done when the user
has checked the list and `/to-spec` has turned it into a spec.

## Candidates, once measured

Each needs a measurement that shows the cost first (docs/HARDWARE.md has the first ones).

* `create_from_rgbd` takes 19.4 to 19.9 ms of the 33 ms frame at 1280x800 (docs/HARDWARE.md):
  where the time goes (the bilateral filter, the invalid-point removal, the upload).
* Undistort color, or check that the SDK's aligned output needs none.
* The half-pixel deprojection of the point-cloud kernel (AGENTS.md §6).
* A live 3D view of the point cloud: `orbbec_viewer` shows only 2D images.
* A compiler warning set (CODING_STANDARDS.md).
