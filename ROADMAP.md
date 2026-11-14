# Roadmap

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

What to do next, in order. Each step has a definition of done and says who can do it:
**agent** (offline, from a `ready-for-agent` ticket), **hardware** (the board and the user's
go), **decision** (the user's). A finished step is deleted; its outcome goes into the
document that owns the fact.

## 1. The first phase of the rewrite: the colored model (agent, hardware, decision)

The spec is `.scratch/rewrite/spec.md`, from the requirements in
`.scratch/rewrite/requirements.md` (the grilling of 2026-11-13; its section 11 holds the old
reconstruction loop of CUDA_3D) and the user's answers of 2026-11-14 to its assumptions. Its
tickets are `.scratch/rewrite/issues/`, and `.scratch/rewrite/brief.md` is what an implementer
reads first. What outlives them is in GLOSSARY.md, docs/adr/ 0002 to 0004 and
CODING_STANDARDS.md, "The rewrite's code".

The user agreed on 2026-11-14 to the choices the spec marks (spec) and to the breakdown into
tickets. The frontier is `frame-source-with-synthetic-frames` and
`aligned-frame-distortion-tool` (agent). Done when `accept-the-colored-model` is done.
`.scratch/cleanup/` can go on the user's word: the requirements hold what it recorded of the
old loop.

## 2. What the first phase needs from the user (hardware, decision)

* An active USB 3 cable of 3 m to 5 m [?: a choice] for the hand-held camera, checked for
  5000M on the board (AGENTS.md §6).
* The root step that starts the user's services at boot (docs/HARDWARE.md, "Memory, storage,
  network").
* A box of measured size for each object class, for the phase's acceptance.
* A printed checkerboard on a flat, stiff board, its square size measured, for the
  measurement of the aligned frames' distortion. It has to span at least half the picture,
  about A3 (11x7 inner corners of 30 mm) at 0.35 m [?: computed from the color coefficients,
  not measured]: reaching from the middle into a corner it is 0.3 px rms and up to 0.8 to 1.1 px
  off a homography at 1280x800, centred 0.2 px rms and 0.5 px at most, and a small board near a
  corner only 0.02 to 0.05 px, below the 0.04 px rms (0.12 px at most) the corner finder leaves
  on a synthetic pinhole picture [M 2026-11-14: `tests/test_checkerboard_measure.cpp`].
* The license of `data/frag_115.ply` and `frag_116.ply`: unchecked (README.md, "License"),
  while ADR 0002 admits permissive licenses only.
* The go for each `ready-for-human` ticket.

## Candidates, once measured

Each needs a measurement that shows the cost first (docs/HARDWARE.md has the first ones).

* `create_from_rgbd` takes 19.4 to 19.9 ms of the 33 ms frame at 1280x800, 24 to 26 ms on a
  scene with about 1.5x the points [M 2026-11-13] (docs/HARDWARE.md):
  where the time goes (the bilateral filter, the invalid-point removal, the upload).
* Undistort color, or check that the SDK's aligned output needs none.
* The half-pixel deprojection of the point-cloud kernel (AGENTS.md §6).
