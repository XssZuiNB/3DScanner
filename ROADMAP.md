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
tickets, and asked for `ray-cast-faster-by-measurement` on 2026-11-15. 26 of 54 tickets are
done (git log, `Ticket ...`), 22 are `ready-for-agent` and 6 `ready-for-human` [M 2026-11-15: a
script over the tickets' Status and Blocked-by lines]. The frontier, the open tickets with no
open blocker:

* `track-frames-against-the-model` (agent): its plan and the draft of its method document wait
  in its Comments for a fresh implementer.
* `ray-cast-faster-by-measurement` (agent): `per-stage-times-from-replay` waits for it, so that
  the stream size is decided on the faster cast.
* `hold-color-exposure-and-white-balance` (agent): waits for the user's answer on the camera's
  flash (its Comments; step 2).
* `measure-aligned-frame-distortion` (hardware: the checkerboard of step 2 and the user's go).

The other 19 tickets for an agent and 5 for the user wait on one of these or on a ticket after
it: tracking leads to the scan interface and everything after it; the checkerboard's measurement
leads to the hardware check of the camera on the seam, whose sequences the replay's measurements
need; the held exposure leads to its first trial on the camera and to the service. The hash's
design is decided by a measurement that replays the cast `ray-cast-faster-by-measurement`
leaves; no ticket holds it yet (`.scratch/rewrite/brief.md`, §8). Done when
`accept-the-colored-model` is done.

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
* The go for each `ready-for-human` ticket.
* The answers the tickets wait for, each in the ticket's Comments: whether writing the color
  camera's auto exposure and auto white balance persists anything in its flash
  (`hold-color-exposure-and-white-balance`); tracking's starting threshold of the inlier RMSE
  (`track-frames-against-the-model`); and the orchestrator's choices of 2026-11-15, to confirm or
  overturn (`.scratch/rewrite/brief.md`, §8).
* Optional, root: the user in group `debug`, so that `compute-sanitizer` can check the GPU's
  kernels; today it prints "GPU debugging features are disabled", because the debugger's
  devices (`/dev/nvhost-dbg-gpu`, `/dev/nvgpu/igpu0/dbg`) are `root:debug` 0660
  [M 2026-11-14]. Two mutants of the sparse TSDF's integration wait for it
  (`.scratch/rewrite/brief.md`, §8).
* Optional, root: `sudo apt install earlyoom`, a guard against a frozen board (a build of six
  jobs froze it once, AGENTS.md §4).

## Candidates, once measured

Each is built only on a measurement that shows its benefit (AGENTS.md §7 rule 6); the document
named holds the reason.

* The coarser levels' normals combined from the finer level's, in place of their own
  (docs/methods/frame-pyramid.md, "Normals at every level").
* A strided window for the normals at 1280x800, where the 7 x 7 window spans half the surface
  (docs/methods/normals.md, "Accuracy"; `.scratch/rewrite/brief.md`, §8).
* The color term's gradient from the image's 2D gradient through the projection, DVO-style,
  which the user asked about on 2026-11-14 (`.scratch/rewrite/brief.md`, §8).
* The frames between two keyframes following an interpolation of their corrections
  (docs/methods/refinement.md, "How the other frames follow their keyframes").
