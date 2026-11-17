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
tickets; later decisions are in `.scratch/rewrite/brief.md`, §7. 46 of 76 tickets are done (git
log, `Ticket ...`), 23 are `ready-for-agent` and 7 `ready-for-human` [M 2026-11-17: a script over
the tickets' Status and Blocked-by lines]. Built in the run of 2026-11-16 to 17: the extraction's
coalesced sweep, the web chain (HTTP routes, the WebSocket, the page's first half,
`scanner_service`), separation's up from gravity, and the live view: the TSDF's ray cast in color
for every tracked frame. The user's first scans with `scanner_service` (2026-11-17) found the
point-cloud live model unusable and a loss right after a scan's start; they led to the live view,
to the page as an app in Rust and egui (.scratch/rewrite/page-app-requirements.md), to a start
that waits for the camera to be still, the gyroscope's prior and relocalization (brief §7, 46 to
54). The frontier, the open tickets with no open blocker, in the order to take them:

* **The workstation as a second platform** (brief §7, 55): `workstation-as-a-second-platform`
  (agent: the ADR, which the user confirms), then `builds-and-verifies-on-the-workstation`
  (agent, on the workstation); `build-runs-two-jobs` after it measures the Jetson's cold build
  at 2 jobs (agent, on the Jetson).
* `page-shows-the-live-view` (agent): the live view on the WebSocket, one client at a time, the
  state's object class, the point-cloud live model taken out.
* `page-app-first-step` (agent; the user confirms its crates), then `page-app-scan-screen` and
  `page-app-replaces-the-javascript-page`.
* `imu-clock-offset-from-the-scan` (agent), then `scan-starts-when-the-camera-is-still` and
  `gyroscope-prior-for-tracking`.
* `relocalize-from-keyframes` (agent; its plan accepted, the color condition added, its worktree
  rebased onto main first), then `refinement-optimizes-the-trajectory`.
* `kernels-under-compute-sanitizer` (agent, parked in its worktree, rebased first; brief §7, 39).
* `scans-list-delete-and-unfinished`, `scan-ends-on-budget-and-disk`, `install-the-user-service`
  (agent).
* `measure-aligned-frame-distortion` (hardware: the checkerboard of step 2 and the user's go).
* `measure-depth-noise-on-the-camera` (hardware: still sequences for the `depth_noise` tool).

The other tickets wait on one of these. The checkerboard's measurement leads to the hardware
check of the camera on the seam, whose sequences the replay's measurements need
(`per-stage-times-from-replay`, `tracking-thresholds-from-real-sequences`,
`memory-budget-and-refinement-time-measured`, `sequence-compression-by-measurement`). The stream
size and the GPU's clock during a scan are decided on real sequences, each platform its own. The
hash's design is compared later on real scans of objects (brief §7, 22); no ticket holds it yet.
Tickets run one at a time on the Jetson (brief §7, 36). Done when `accept-the-colored-model` is
done.

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
* Each block's neighbours looked up once in the extraction, in place of a hash lookup for each
  edge's neighbour (docs/methods/sparse-tsdf.md, the extraction: "found through the hash").
* A reader thread beside each WebSocket's loop, so that a browser's frame split across the
  loop's 20 ms read keeps the connection (docs/ROUTES.md: the library then closes it).
* The hash's design against the bucketed hash of Nießner et al. 2013, compared on real scans of
  objects (`.scratch/rewrite/brief.md`, §7, 22 and §8).
* The GPU's clock held at its highest during a scan (a root setting, the user's) against the
  governor: on the demo on the camera, tracking, frames 30 % to 40 % shorter [M 2026-11-15:
  `per-stage-times-from-replay`'s Comments, not a timing run].
