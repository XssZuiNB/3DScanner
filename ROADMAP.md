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
tickets, asked for `ray-cast-faster-by-measurement` on 2026-11-15 and, after an experiment that
measured it, for `ray-cast-normal-reads-once`. 27 of 61 tickets are done (git log,
`Ticket ...`), 27 are `ready-for-agent` and 7 `ready-for-human` [M 2026-11-15: a script over the
tickets' Status and Blocked-by lines]. The frontier, the open tickets with no open blocker:

* `track-frames-against-the-model` (agent): its plan, the draft of its method document and the
  orchestrator's plan check wait in its Comments for a fresh implementer; tracking's starting
  inlier RMSE threshold is the user's, 5 mm and 30 mm (`.scratch/rewrite/brief.md`, §7, 17).
* `ray-cast-normal-reads-once` (agent): the ray cast's normal reads each voxel once, measured in
  scratch at 19 % to 28 % shorter back to back, bit for bit (its Comments).
* `imu-samples-with-the-frames` (agent): the IMU's samples in the frame source and the sequence,
  before `hardware-check-orbbec-on-seam` records sequences; `separation-up-from-gravity` follows.
* `hold-color-exposure-and-white-balance` (agent): written offline; whether writing the
  properties keeps anything in the camera's flash is checked first in
  `first-trial-of-held-exposure` (brief §7, 16).
* `depth-noise-from-a-still-sequence` (agent): the tool that fits the depth noise's c, which
  `measure-depth-noise-on-the-camera` (hardware) runs on still sequences.
* `kernels-under-compute-sanitizer` (agent): the GPU tests under memcheck and initcheck in
  verify; two mutants of the sparse TSDF's integration wait for it.
* `glossary-and-dependency-licenses` (agent, sonnet): Keyframe in the glossary, "final model"
  renamed to "full model", the dependencies' licenses.
* `measure-aligned-frame-distortion` (hardware: the checkerboard of step 2 and the user's go).

The other tickets wait on one of these or on a ticket after it: tracking leads to the scan
interface and everything after it, and the shortest way to a first scan of a recorded sequence
is tracking, `scan-interface-first-whole-scan` and `replay-tool`; the checkerboard's
measurement leads to the hardware check of the camera on the seam, whose sequences the replay's
measurements need; the held exposure leads to its first trial on the camera and to the service.
The stream size (`per-stage-times-from-replay`) is decided on the cast both ray cast tickets
leave. The hash's design is compared later on real scans of objects (brief §7, 22); no ticket
holds it yet. Done when `accept-the-colored-model` is done.

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
* The go for each `ready-for-human` ticket, among them the still sequences of
  `measure-depth-noise-on-the-camera`: 300 frames at each stream size, aimed obliquely at a wall
  or the floor (its ticket).
* Optional, root: `sudo apt install earlyoom`, a guard against a frozen board (a build of six
  jobs froze it once, AGENTS.md §4).
* Optional, root: the journal kept across boots, so that the next freeze leaves the kernel's
  last lines on disk: the freeze of 2026-11-15 left only syslog, and its cause is unknown
  (docs/HARDWARE.md, "Platform"). journald's `Storage` stands at its default, `auto`
  [M 2026-11-15: /etc/systemd/journald.conf], which keeps the journal on disk once
  `/var/log/journal` exists [?: journald.conf(5), not read here]: `sudo mkdir -p
  /var/log/journal`, `sudo systemd-tmpfiles --create --prefix /var/log/journal`, `sudo systemctl
  restart systemd-journald`.

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
* The gyroscope as tracking's prior, once the sequences hold the IMU's samples
  (`.scratch/rewrite/brief.md`, §7, 13).
* The hash's design against the bucketed hash of Nießner et al. 2013, compared on real scans of
  objects (`.scratch/rewrite/brief.md`, §7, 22 and §8).
