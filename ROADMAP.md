# Roadmap

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

What to do next, in order. Each step has a definition of done and says who can do it:
**agent** (offline, from a `ready-for-agent` ticket), **hardware** (the board and the user's
go), **decision** (the user's). A finished step is deleted; its outcome goes into the
document that owns the fact.

## 1. Open decisions (decision)

* The license: the SPDX line every file starts with (`TODO(template)` in each new file).

## 2. The rewrite's requirements (decision)

The rewrite starts with its list of requirements, which the user checks. `/grill-with-docs`
on the scanner: what it scans (objects, rooms), how the camera moves (hand, turntable), what
it delivers (point cloud, mesh, which format), accuracy and speed, where it runs and who
works it, and which parts of CUDA_3D carry over (its algorithms, `gca::device`, the
`movement_detection/` module, `scripts/`, `graph.png`). Done when the user has checked the
list and `/to-spec` has turned it into a spec.

## Candidates, once measured

Each needs a measurement that shows the cost first (docs/HARDWARE.md has the first ones).

* `create_from_rgbd` takes 19.4 to 19.9 ms of the 33 ms frame at 1280x800 (docs/HARDWARE.md):
  where the time goes (the bilateral filter, the invalid-point removal, the upload).
* Undistort color, or check that the SDK's aligned output needs none.
* The half-pixel deprojection of the point-cloud kernel (AGENTS.md §6).
* A viewer without PCL.
* A compiler warning set (CODING_STANDARDS.md).
