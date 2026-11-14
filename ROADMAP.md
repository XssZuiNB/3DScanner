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
tickets. 18 of 53 tickets are done (git log, `Ticket ...`), 29 are `ready-for-agent` and 6
`ready-for-human` [M 2026-11-14: a script over the tickets' Status and Blocked-by lines]. The
frontier, the open tickets with no open blocker: `memory-budget-two-gigabytes`,
`color-term-and-coarse-to-fine`, `pose-graph-solver`, `sparse-tsdf-integrate-and-extract`,
`hold-color-exposure-and-white-balance`, `propose-support-plane-and-box` and
`box-fit-acceptance-tool` (agent), and `measure-aligned-frame-distortion` (hardware, the
checkerboard below). The hash's design is decided by a measurement once the ray cast exists
(`.scratch/rewrite/brief.md`, §8). The other 22 tickets for an agent and 5 for the user wait on a
ticket of the frontier or after it. Done when `accept-the-colored-model` is done.
`.scratch/cleanup/` can go on the user's word: the requirements hold what it recorded of the old
loop.

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
* Optional, root: the user in group `debug`, so that `compute-sanitizer` can check the GPU's
  kernels; today it prints "GPU debugging features are disabled", because the debugger's
  devices (`/dev/nvhost-dbg-gpu`, `/dev/nvgpu/igpu0/dbg`) are `root:debug` 0660
  [M 2026-11-14].

## Candidates, once measured

Each needs a measurement that shows the cost first (docs/HARDWARE.md has the first ones).

* Undistort color, or check that the SDK's aligned output needs none.
