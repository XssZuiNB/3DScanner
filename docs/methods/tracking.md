# Tracking: each frame registered against the model and checked before it is fused

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The per-frame path's steps 1 to 6 (the spec, "The per-frame path") as one call for each frame of a
scan: the frame is uploaded, its pyramid built, the model's surface ray cast from the last tracked
pose, the frame registered against that surface, the registration's result checked, and the frame
fused at its pose only when the check passes (R10, R11). Each frame comes back with its tracking
state as a value: tracking with its pose, or tracking loss without one (GLOSSARY.md, "Tracking").
This is the frame-to-model tracking of KinectFusion, every frame registered against the model's
surface predicted from the last pose rather than against the frame before [S: R. A. Newcombe et
al., "KinectFusion: Real-time dense surface mapping and tracking", ISMAR 2011, sections 3.4 and
3.5]; the check of a result before its frame enters the model is this tree's (R11, R32). This
tree's code is written from the sources named here; no reconstruction library's code is read or
copied for it (docs/adr/0002-own-reconstruction-code.md).

Built: the tracker and the check (`src/tracking/tracker.*`; the test `tests/test_tracking.cpp`),
and the scan's loop around them, with the path's steps 7 and 8, the sequence's writer, the
published state, the "too close" notice (R36), the live model and the live view ("The scan's
loop", "The live model", "The live view"; `src/scan/scan_interface.*`, `src/scan/live_model.*`,
`src/scan/live_view.*`; the tests `tests/test_scan_interface.cpp`, `tests/test_live_model.cpp`).
Left to `relocalize-from-keyframes`: keyframes and the candidates they add for a lost frame. Left to
`tracking-thresholds-from-real-sequences`: the thresholds' values and ε, measured on real
sequences.

## Input

* **A frame** (`gca::frame`): its color and its depth, its index, its timestamp and its description,
  of the size the tracker's buffers were requested for. Frames come in their source's order, their
  timestamps increasing strictly, as a frame source delivers them (`src/camera/frame_source.hpp`,
  `next()`). A frame not later than the frame before it, or of another size, is refused: the
  tracker's failure, not tracking loss ("Output") [?: a choice]. A frame source delivers strictly
  increasing timestamps, and the camera's source skips and counts a frame whose timestamp does not
  advance (`src/camera/orbbec_rules.hpp`), so such a frame reaching the tracker is its caller's
  error, which no state of the scan explains; and Δt, which the step's limit grows with ("The step
  a hand can make"), would be 0 or less.
* **The parameters** (`gca::tracking_parameters`), each the object class's
  (`src/scan/object_class.cpp`), converted once (`gca::tracking_parameters_of()`,
  `src/scan/stage_parameters.hpp`): the pyramid's depth range and normals'
  (docs/methods/frame-pyramid.md); the registration's (docs/methods/registration.md,
  "Parameters"); the check's thresholds ("Parameters" below). The live model's sparse TSDF takes
  its own (docs/methods/sparse-tsdf.md, `gca::sparse_tsdf_parameters_of()`): the object class's
  live voxel size, memory budget, truncation and far depth. The depth range is 0.25 m to 0.6 m for the tabletop object and 0.25 m to 3.0 m for the
  large object [?: initial, the user's choices of 2026-11-15, evening]: 0.25 m is the datasheet's
  least depth [S: datasheet, docs/HARDWARE.md], so what the camera sees nearer than the object
  class's near depth is tracked and fused like the rest (R36); the far depth of the tabletop
  object keeps the room behind it out of the live model (a pool of 256 MB filled with it at
  1.0 m, 46668 blocks of 1 GB at 0.6 m [M 2026-11-15, the demo window on the camera]).

## Output: the tracking state of a frame

A value for every frame (`gca::tracking_result`), one of:

* **Tracking**, with the frame's pose: from its camera frame (x right, y down, z forward, in m)
  into the models' frame (`src/types/camera_pose.hpp`). The frame has been fused at that pose.
* **Tracking loss**, with its cause, the first condition of the check that failed ("The check"),
  and no pose. Nothing of the frame enters the model.
* **Failed**, with the reason: a frame or a state the tracker refuses, a registration refused or
  failed, or CUDA's error, which is cleared (`src/gpu/gpu_messages.hpp`). No pose. Nothing was
  fused, unless the failure is the fusion's own, which leaves what `sparse_tsdf::integrate()` says
  it leaves, or the cast's after the fusion, which leaves the frame fused at its pose, the last
  tracked one, and the surface unspecified. A scan ends on it.

With each, the evidence the check judged: the registration's result (its end, transform, fitness,
inlier RMSE, iterations, eigenvalues, and its last step's turn and shift, the values the
registration compared with ε_ω and ε_υ; docs/methods/registration.md, "Output"), the step from the
last tracked pose, its turn and its shift, and Δt ("The step a hand can make"), which
`tracking-thresholds-from-real-sequences` records for every frame (the user's decision of
2026-11-15, evening).

## The path of a frame

For each frame, on the caller's stream, in this order:

1. **The upload** into the tracker's GPU frame (`gca::gpu_frame`, staged through pinned memory).
2. **The pyramid** (`frame_pyramid::build()`): three levels of organized points, normals and
   intensity, within the object class's depth range.
3. **The surface**: the ray cast of the live model from the last tracked pose P_L
   (`sparse_tsdf::ray_cast()`), at the frames' intrinsics and the same depth range, every level
   of the pyramid, with the colors at level 0 the live view shows (docs/methods/sparse-tsdf.md,
   "The ray cast"; "The live view"). It is cast right after the fusion that made P_L the last
   tracked pose (step 6), not when the next frame comes: the model and P_L change only there, so
   it is the cast the next frame would make and the images the registration reads are the same bit
   for bit, only the time of the cast moves [M 2026-11-17: the replay's trajectory of a synthetic
   walk of 479 frames and of the user's first scan, byte for byte cf8a02a's, docs/HARDWARE.md,
   "The live view"]; it is the view of the frame just fused from its own pose, and its GPU work
   runs while the scan waits for the next frame. A lost frame casts nothing: it leaves P_L and the
   model as they were, and the surface with them. Before the first fused frame there is no model:
   see "The first frame".
4. **The registration** of the frame's pyramid against the surface's levels, from the identity
   (`registration::align()`; docs/methods/registration.md, "Input": tracking passes the identity
   when the surface is ray cast from the last pose). Its transform T carries the frame's camera
   frame F into the surface's, the camera frame at P_L.
5. **The check** of the result ("The check"). The frame's pose is P = P_L T
   (docs/methods/registration.md, "Output"), composed in double, its rotation kept as a unit
   quaternion.
6. **The fusion**, only when the check passes: `sparse_tsdf::integrate()` of the GPU frame with
   the pyramid's level 0 points at P, which allocates the blocks near the frame's surface and
   updates their voxels. P becomes the last tracked pose, the frame's timestamp the last tracked
   time, and the surface is cast from it (step 3).

On tracking loss P_L and the last tracked time stay as they were, and the model and its surface are
not touched.

**The buffers** of every step are requested from the pool (`src/gpu/gpu_pool.hpp`) by one request
before the first frame, for one frame size, and the pool is closed then; `track()` allocates
nothing (R29). Besides the live model's sparse TSDF, whose device memory is the memory budget,
they take 25.9 MB of device memory and 1.28 MB of pinned memory at 640x400, 103.5 MB and 5.12 MB at
1280x800 [S: computed: the GPU frame 5 bytes a pixel and its staging buffer as much, the pyramid
and the surface 29 bytes a pixel of each level, the surface's colors 3 bytes a pixel of level 0 and
its tiles 0.125,
the registration 12 bytes a pixel of each level and 280 bytes a block of 32 x 8 pixels; the levels
hold 1 + 1/4 + 1/16 of the pixels]. The live model is requested first, so that a closed pool checks
every parameter without device memory (`sparse_tsdf::request()`).

**The waits**: the upload's wait on its own staging buffer, which returns at once when the frame
before was waited for (`src/gpu/gpu_frame.hpp`), and the registration's one written wait per
iteration, which also waits for the stream's earlier work, the previous frame's fusion included.
Nothing else waits: the fusion and the cast after it are queued and the call returns; the next
frame's registration, or the caller's own wait, finds them done. A replay's measurement alone adds
a wait after each stage ("The stages' times"). What comes back to the host is the registration's
280 bytes per iteration (docs/methods/registration.md, "On the GPU") and nothing more.

## The models' frame and the first frame

The models' frame is the camera frame of the first fused frame (the spec, "Tracking, tracking
loss, relocalization"): that frame's pose is the identity.

Until a frame is fused the model holds nothing to register against. Then each frame is registered
against its own pyramid, from the identity, and checked as any other, without a step; it is fused
at the identity when it passes, and is tracking loss when it does not, and the next frame is tried
the same way [?: a choice]. A frame against itself takes at each level one system whose residuals
are all 0, every candidate its own partner at distance 0, so its step is 0 and it ends at the
identity exactly, with a fitness of 1 and an inlier RMSE of 0 [S: the association by projection,
docs/methods/registration.md]: what decides is that last system's conditioning, converged when its
λ_1 >= τ and degenerate when not, at the cost of three systems. A first view that does not hold
every motion, one plane without texture, would start a model against which no later frame of that
view could be tracked; it does not start one, and the operator sees tracking loss with the cause
"degenerate" until the view holds every motion. The test's view sliding along a plane without
texture is tracking loss from its first frame ("Accuracy").

## The check

A frame enters the model only when its registration's result passes every condition, in this
order; the first that fails is the cause of the tracking loss:

| | condition | what it refuses |
|---|---|---|
| 1 | the registration ended converged or with its iterations used up | too few correspondences, degenerate (R32); a refused or failed registration is the tracker's failure, not tracking loss |
| 2 | fitness >= f_track | a frame of which less than f_track found the model: covered, turned away, or seeing what the model does not hold |
| 3 | inlier RMSE <= e_track | a minimum the surface does not fit, as a part of the scene that moved |
| 4 | the step's turn <= ω_max Δt and its shift <= v_max Δt | a step no hand makes in that time ("The step a hand can make") |

* **Converged or iterations used up** (the user's decision of 2026-11-15, evening): both ends
  leave the last system not degenerate, its λ_1 >= τ (docs/methods/registration.md, "The
  iterations and the end"); "converged" says besides that the last step lay below ε_ω = 1e-5 rad
  and ε_υ = 1e-5 m. On real frames the depth's noise holds the last step above ε: on the camera
  (firmware 1.5.02, 640x400, the tabletop object's values) good alignments ended "iterations used
  up" at a fitness of 0.87 to 0.93 and an inlier RMSE of 0.9 to 2.2 mm, and a check that demanded
  "converged" lost every frame after the first that so ended, each lost frame leaving the pose
  behind, so that the next started farther away; with "iterations used up" accepted at a fitness
  of at least 0.5 and an RMSE of at most 5 mm, 829 of 1443 frames were fused, the rest lost in fast
  moves with a fitness below 0.5 or an RMSE above 5 mm [M 2026-11-15, the demo window, the old loop
  without this module, 0.3 m to 0.6 m, a pool of 1 GB]. Fitness, RMSE and the step judge the
  minimum; ε stays until `tracking-thresholds-from-real-sequences` sets it from the floor of the
  last step measured on real sequences, which every result records ("Output").
* **The conditioning** is the registration's degeneracy test, which condition 1 holds: neither
  accepted end leaves a last system whose smallest eigenvalue, centred, scaled and normalized, lies
  below τ = 0.01 (docs/methods/registration.md, "The degeneracy test and the damped solve"). The
  check adds no threshold of its own on the eigenvalues [?: a choice]: the point-to-plane term
  alone holds well-conditioned scenes at λ_1 = 0.017 to 0.036 (docs/methods/registration.md,
  "Accuracy"), so a threshold above τ would lose frames of objects without texture whose geometry
  holds them, and one equal to τ would refuse nothing condition 1 accepts. Relocalization's
  stricter thresholds may add one (the spec).
* **Fitness and RMSE** are those of the registration's last system at level 0: the share of the
  frame's candidates with a correspondence, and the root mean square of their point-to-plane
  residuals (docs/methods/registration.md, "Output"). The registration's own least fitness, f_min =
  0.1, only decides whether a system is solved; f_track is the check's.

## The step a hand can make

The step from the last tracked pose is T itself, F into the camera frame at P_L: its **turn** θ is
the angle of T's rotation, in degrees, and its **shift** s the distance the camera's centre moved,
|t_T|, in m. The check allows θ <= ω_max Δt and s <= v_max Δt, the hand's largest turn rate and
speed times Δt, the time from the last tracked frame's timestamp to this frame's [?: a choice]:

* **Not one frame time**: frames the scanner did not take (the spec, "The sequence": a frame enters
  the path only when the writer has room) and frames lost in between lengthen the time the hand had,
  and a fixed step would lose tracking on them though the hand moved no faster. At 30 fps and no
  frame missed, Δt is one frame time, 33.3 ms, and the limits are 3° and 33 mm.
* **What it leaves to the other conditions**: the limit grows with the loss. After 2 s of tracking
  loss it is 180° and 2.0 m, more than any step against the last tracked pose that a registration
  of d_max_0 = 5 cm (tabletop) or 10 cm (large) can reach [S: computed, the starting values]: from
  then on the step refuses nothing, and a lost frame is judged by its registration's end, its
  fitness and its RMSE alone, and relocalization's stricter thresholds its candidates
  (`relocalize-from-keyframes`).
* **The timestamps** increase strictly (Input), so Δt > 0.
* **The first frame** has no last tracked pose and no step.

## Tracking loss: what a lost frame is tried against

A lost frame changes nothing of the tracker's state: the last tracked pose, its time and the model
stay, and with them the surface cast from that pose after its fusion. Each following frame goes
through the same path: it is registered against that surface, from the identity, under the same
check, Δt counted from the last tracked frame. The first that passes is fused and tracking goes on
from it.
`relocalize-from-keyframes` adds the keyframes as further candidates, after the last tracked pose
(the spec).

## Parameters

The check's thresholds (`gca::tracking_thresholds`) are each object class's
(`src/scan/object_class.cpp`, which also holds the registration's values of
docs/methods/registration.md, "Parameters", and the "too close" notice's threshold), starting
values [?: initial] until `tracking-thresholds-from-real-sequences` measures them on real walks,
and refused outside their ranges [?: a choice]:

| threshold | meaning | tabletop object | large object | refused outside |
|---|---|---|---|---|
| f_track | the least fitness | 0.5 | 0.5 | (0, 1] |
| e_track | the largest inlier RMSE, m | 0.005 | 0.030 | finite, > 0 |
| ω_max | the hand's largest turn rate, °/s | 90 | 90 | finite, > 0 |
| v_max | the hand's largest speed, m/s | 1.0 | 1.0 | finite, > 0 |

* **f_track**: half of the frame's candidates. A hand's step at 30 fps, 1.5° and 10.8 mm in the
  registration's tests, moves the picture by about 8 of its 640 columns at 640x400 [S: computed,
  306 px/rad], so a frame of a walk sees mostly what the model holds: the test's walk's least
  fitness is 0.944 [M 2026-11-15: "Accuracy"], real frames' 0.87 to 0.93 while tracking ("The
  check"). A frame half covered or turned to what the model lacks is lost.
* **e_track**: δ_G's values, the datasheet's bound on the depth noise at the middle of each class's
  former range, 4.2 mm at 0.65 m and 29 mm at 1.7 m (docs/methods/registration.md, "Parameters"),
  rounded [?: initial, the user's choice of 2026-11-15]: real frames' RMSE, 0.9 to 2.2 mm while
  tracking on the camera ("The check"), lies below it. The cost the user accepted: a part of the
  scene that moved may be fused. The moved sphere of docs/methods/registration.md, "Huber's
  weight", converged 12.3 mm off with an inlier RMSE of 1.79 mm against 0.29 mm of a static scene
  [M 2026-11-14]; registered against the model's surface it converges at 3.0 mm, passes 5 mm and
  is fused with its pose 0.42° and 2.8 mm off ("Accuracy"). The offline tests take e_track =
  1.0 mm, a named value of noise-free frames, which lies between the walk's largest RMSE, 0.37 mm,
  and the moved sphere's, so that the RMSE's condition is shown to refuse it.
* **ω_max, v_max**: twice the turn and three times the shift of the hand's motion the
  registration's tests recover at 30 fps, 45°/s and 0.32 m/s (docs/methods/registration.md,
  "Accuracy"; tests/registration_cases.hpp, `tabletop_delta`): 3° and 33 mm in a frame time. A step
  beyond them is one no hand scanning an object makes, which only a registration that slid to a
  wrong minimum gives [?: initial]. The test's walk turns by up to 2.65° in a frame time, 79°/s,
  its roll's swing added to its azimuth's 1.5°, within 12 % of ω_max [M 2026-11-15: "Accuracy"];
  how fast a hand turns on real walks is `tracking-thresholds-from-real-sequences`' to measure.
* **The "too close" threshold**, 0.30 m (tabletop) and 0.40 m (large), the object classes' former
  near depths: a frame is too close when at least a quarter of its pixels with depth lie nearer
  [?: initial, the user's choice of 2026-11-15, evening]. Tracking does not read it; the scan
  interface raises the notice.

## Accuracy

[M 2026-11-15: `tests/test_tracking.cpp`, frames of the synthetic source without noise, 640x400
with the camera's intrinsics, depth unit 1 mm, the tabletop object class's values (the range
0.25 m to 0.6 m, the live model at 2 mm) with e_track = 1.0 mm, a sparse TSDF of 273 MB; the test
takes 17.0 s on the board]

* **A walk once around the tabletop box**: 240 poses of `tabletop_walk()`, 1.5° of azimuth apart,
  the box of 0.20 m x 0.14 m x 0.10 m on its plane seen from 0.4 m to 0.5 m along the plane and
  0.30 m to 0.40 m above it: every frame tracking, 236 converged and 4 with their iterations used
  up. The tracked trajectory lies within 0.054° and 0.52 mm of the known one at every pose. Over
  the walk the least fitness is 0.944, the largest
  inlier RMSE 0.37 mm, the least λ_1 0.159, the largest last step 1.6e-5 rad and 4.1e-6 m, and the
  largest step 2.65° and 13.6 mm in a frame time, 79°/s and 0.41 m/s. The closed pool was asked
  for nothing.
* **A frame against the cast from its own pose**: the walk's last frame again, one frame time
  later, registered against the cast from its tracked pose: 0.0015° and 0.014 mm from the
  identity, converged in 6 systems.
* **Frames without depth**: frames 6 to 8 of the walk without depth: tracking loss from the
  first, too few correspondences (fitness 0); the model's 106830 extracted points the same before
  and after, bit for bit as a set; frame 9 tracking again, its step 6.05° and 50.6 mm over four
  frame times, within the limits of 12° and 133 mm.
* **A view sliding along one plane without texture**: a grey plane seen from 0.45 m, the camera
  moving 1 cm a frame: tracking loss from the first frame, degenerate (λ_1 = 9.2e-6, fitness 1
  against its own pyramid); never a pose; the model holds no point.
* **The sphere moved by 2 cm** along x between the fused frame of the tabletop object (a box of
  0.30 m x 0.20 m x 0.25 m and a sphere of 0.10 m on a plane, seen from 0.5 m) and the next, the
  camera moved by 1.5° and 10.8 mm: converged at an inlier RMSE of 3.0 mm, fitness 0.905, so
  tracking loss by the RMSE at e_track = 1.0 mm, the model's points the same; at the tabletop
  object's own e_track, 5 mm, the same frame is tracking and is fused, its pose 0.42° and 2.8 mm
  off the known one, the cost the user accepted ("Parameters").

## On the GPU, and the time

Tracking launches no kernel of its own: its steps are the stages' (docs/methods/frame-pyramid.md,
docs/methods/sparse-tsdf.md, docs/methods/registration.md), and the check runs on the host. The
whole path's time for a frame on the board stands in docs/HARDWARE.md, "The per-frame path's
stages": on the synthetic walk at the tabletop object's values, a median of 12.5 ms at 640x400 and
34.9 ms at 1280x800 back to back, and 20.9 ms paced at 30 fps at 640x400, the governor's clock at
510 to 918 MHz [M 2026-11-15]. At 1280x800 the path does not fit a frame's 33.3 ms; the stream
size is `per-stage-times-from-replay`'s to settle on real sequences.

## The scan's loop

The scan interface (`src/scan/scan_interface.hpp`; the spec, "The scan interface") runs the path
on its scan thread, which owns the source, the GPU's work, the writer's calls and the scan's files.
Before the first frame it requests the tracker's buffers and the live model from a pool it then
closes, opens the source and takes its first frame. Then, for each frame the source delivers, in
this order:

1. **A gap** in the frames' indices, the frames the source did not deliver since the frame before,
   counts as not taken. The camera's source gives the oldest frame of OrbbecSDK's queue of up to
   10, which drops its oldest when it is full [S: OrbbecSDK_v2 v2.10.6, Pipeline.cpp, per
   `orbbec-device-on-frame-source`'s Comments], so a loop that falls behind runs up to 10 frames
   late and the frames dropped arrive as such gaps. The newest frame is not taken by a mailbox in
   front of the camera: at 640x400 the path takes 20.9 ms paced, within the frame's 33 ms
   [M 2026-11-15: "On the GPU, and the time"], and a mailbox comes only on a measured backlog that
   hurts (`hardware-check-orbbec-on-seam`).
2. **The writer's room** (`src/camera/sequence_writer.hpp`, `has_room()`): a frame the writer has
   no room for counts as not taken and goes no further, so that no frame is fused that is not
   written. When the source is a sequence, as in the replay tool, the writer waits for room
   instead (`queue_full::wait`): the frames wait on the disk, not on a camera, so a replay takes
   every frame the sequence holds and its scan is the scan made in process on those frames.
3. **The "too close" notice**: at least a quarter of the frame's pixels with depth lie nearer than
   the object class's `too_close_depth_m` ("Parameters"); a pixel with depth is one whose value is
   above 0, beyond the far depth too. One pass over the depth on the host, 0.084 ms at 640x400 and
   0.335 ms at 1280x800 [M 2026-11-16: docs/HARDWARE.md, "The per-frame path's stages"].
4. **Tracking** (`tracker::track()`, steps 1 to 6 above). The frame counts as taken, and as fused
   or lost. A frame tracking fails on ends the scan (`end` `tracking_failed`): it counts as taken,
   neither fused nor lost, and goes neither into the sequence nor into the trajectory. When the
   failure is the ray cast's after the fusion, the frame is in the live model, which only the live
   view and the live model show; the scan's files hold nothing of it.
5. **The writer** takes the frame with its pose or the mark of tracking loss (step 7); a failure
   ends the scan (`writer_failed`).
6. **The live view and the live model**, after a fused frame: the live view is offered to its
   encoder, which takes it when free and skips it when busy ("The live view"); then the live model,
   when its interval has passed ("The live model"). Neither after a lost frame.
7. **The state** is published for the frame (step 8): the counts, the tracking state with the
   condition that failed when lost, and the notice; no lock is held while the listener runs.
8. **The commands** queued meanwhile are answered: a second start is refused, a stop or a discard
   ends the scan before the next frame; a stop or a discard given while the scan was starting is
   answered here, after the first frame. Then the next frame is asked for; a timeout, a failure or
   the end of the source ends the scan with its reason.

After the loop the source is closed, the live view's encoder finished with the last fused frame's
view ("The live view") and the writer closed, so that the sequence holds every frame taken but the
one tracking failed on; the last frame's GPU work is waited for and the tracker,
which holds the live model, is freed as the function that runs the loop returns, so that it
cannot outlive the loop, as one sparse TSDF exists at a time; the parameters are written
(docs/SCAN_FILES.md); refinement fuses the sequence at the full voxel size
(docs/methods/refinement.md, "The fusion of the sequence"), answering the commands before each
frame; gravity's up is estimated from the accelerometer's samples and the trajectory it read
(docs/methods/separation.md, "Gravity's up") and the parameters are written again with it, then
the trajectory and the full model; and the scan answers the commands once more before it is idle,
so that a discard that comes during the extraction or the writes still discards it. A discard removes the
scan's directory; the end of the scan interface during a scan or a refinement leaves the scan
unfinished, without a trajectory or a full model.

## The live model

The scan interface publishes the live model beside the state (`src/scan/live_model.hpp`; R10; the
spec, "The scan interface"): the colored points of the live model's sparse TSDF with the pose
tracking gave the frame they were taken at, at most `live_model_points` of them, 200 000
[?: initial], at most once a `live_model_interval`, 500 ms [?: initial], 2 a second.

* **When.** After a fused frame, once the writer has taken it and before its state is published:
  when no live model was published in the scan yet, or the interval has passed since the scan
  thread decided to take the last one. The pacing runs on the times of the decisions, which each
  live model carries, so two live models lie at least the interval apart by them. Never after a
  lost frame: during tracking loss nothing is fused, and the last live model stays as it is,
  its points and its pose. A scan's start publishes an empty live model, without a frame.
* **Which points.** `sparse_tsdf::extract()` with a spacing m, a power of two
  (docs/methods/sparse-tsdf.md, "The extraction"): a crossing gives a point only when both
  coordinates of its voxel across its edge are multiples of m. So the points of 2m are a subset of
  those of m, a surface keeps the same points from one live model to the next, and a doubling keeps
  about a quarter of them (0.250 and 0.249 [M 2026-11-16: `tests/test_sparse_tsdf_fusion.cpp`]).
  The extraction writes into a buffer of the limit; a live model never holds more.
* **The spacing.** It starts at 1 for each scan and only rises, so the points do not flicker:
  after a live model of more than three quarters of the limit [?: a choice, room to grow by a
  third until the next], the next one is taken at the least power-of-two multiple of the spacing
  at which, a doubling keeping a quarter, it would hold three quarters or less
  (`gca::spacing_after()`). A take that finds more points than the limit publishes nothing; the
  spacing rises, and the next fused frame tries again.
* **Which blocks.** The first live model of a scan, and the first after the spacing rose, extract
  every block. The others extract only the blocks an allocation near a frame's surface touched
  since the last live model (the slots' stamps, `extraction_choice::since`), and replace those
  blocks' points in the last live model on the host; the blocks not touched keep theirs. The
  extraction lists the blocks it went through and gives each point its block's slot; the scan
  thread marks the listed slots, one bit a slot, keeps the last live model's points of the slots
  not marked, adds the new ones, and takes the marks back. A block not touched keeps even those
  points whose crossing or gradient reads a touched neighbour's voxels, so they lag behind it: on
  the tabletop sphere's walk 41 of 12599 points, 0.33 %, differed from an extraction of every
  block at the same spacing, the older ones within 0.49 mm of the newer [M 2026-11-16:
  `tests/test_live_model.cpp`]; on the synthetic scene below, a line of 150 frames over surfaces
  already seen left none [M 2026-11-16]. A take whose points kept and new together exceed the limit
  publishes nothing, as above.
* **Its memory.** The taker's, requested with the tracker's buffers before the pool closes,
  outside the sparse TSDF's budget: on the GPU the limit's points with their slots and a list of
  every block, 8.3 MB, as much pinned, and on the host a bit a slot, 256 KB, for the 2.0 GB TSDF's
  479620 blocks and 2^21 slots; the scan interface's two live models with their slots, 12.8 MB on
  the host, reserved when it is built [S: computed, 28 bytes a point and 4 a slot]. The take
  allocates nothing.
* **Its publication.** Under a lock of its own, not the state's, so that a reader's copy of up to
  5.6 MB never holds up the state of each frame: the scan thread fills the working live model and
  swaps it in under the lock, then calls the live model's listener outside it; a reader copies the
  published one under the lock (`scan_interface::latest_live_model()`).
* **A failure.** CUDA refusing or failing in the extraction, the copies or the wait: nothing is
  published, the reason goes into the state's `live_model_failure`, the scan goes on, and the
  interval passes before the next try. No test reaches it: it takes a CUDA failure in the live
  model's work alone, and a sticky one fails the next frame's tracking anyway.
* **Its cost** [M 2026-11-16: scratch programs under the build lock, MAXN_SUPER, the GPU at 1020
  MHz; docs/HARDWARE.md, "The live model"]. On a synthetic scene of 46849 blocks, the demo
  window's 46668 of a tabletop scan [M 2026-11-15], at 640x400 and the tabletop object's 2.0 GB:
  one live model of every block, extraction, copy and conversion, took 23.5 ms at the spacing the
  rule chose, 4 (92703 points); the path's 11.2 ms at 640x400 (docs/HARDWARE.md, "The replay's frame
  time") with it is 34.7 ms, over the frame's 33.3 ms, which is why only the touched blocks are
  extracted (the user's decision, `live-model-snapshots`' Comments). With them, a live model after
  15 frames moving 1 cm each took 5.5 to 7.0 ms, 4734 to 6879 blocks touched; a live model of every
  block, at a scan's start and after each rise of the spacing, takes 27.8 ms there and goes over
  the frame time once. The spacing barely shortens the extraction past 2 (23.2 to 24.0 ms at 2, 4
  and 8): a block's time is its own work's latency, not its edges' [?: inferred, not measured
  apart].

## The live view

The scan interface publishes the live view beside the state (`src/scan/live_view.hpp`; the user's
decision of 2026-11-17, brief §7, 46): the picture of the live model's surface as the camera sees
it from the pose tracking gave a frame, after that frame's fusion, in the model's colors lit
lightly by the surface's normal, as a JPEG file with the frame's index, its timestamp and the
picture's size; one for each tracked frame while its encoder keeps up. It is the fused surface
from the camera's view, as commercial scanners show while they scan, in place of the live model's
points, which come two times a second and thin as the model grows (the user's first scans with the
service: "画面一跳一跳的没法使用"); `page-shows-the-live-view` puts it on the page and takes the
live model out.

* **Which cast.** The surface the path casts for the next frame's registration ("The path of a
  frame", step 3), right after the frame's fusion: the model with the frame in it, from the frame's
  pose, at its intrinsics and the object class's depth range. No second cast: one from the new pose
  after the fusion, beside the one at the next frame's start, would add a cast to every frame,
  5.989 ms of the 11.068 ms a frame took at 640x400 (the replay's medians) and 14.310 ms paced at
  the governor's 306 MHz [M 2026-11-15, 2026-11-16: docs/HARDWARE.md, "The ray cast", "The replay's
  frame time"]; and the cast at the next frame's start, from the last tracked pose, would show the
  model without that frame from the pose before it, one frame behind. Cast after the fusion, the one
  cast is both the next frame's surface and the frame's view: the replay's frame time 9.93 to
  10.06 ms against cf8a02a's 10.08 ms, its trajectory byte for byte cf8a02a's [M 2026-11-17:
  docs/HARDWARE.md, "The live view"].
* **The colors**: the cast's at level 0 (docs/methods/sparse-tsdf.md, "The ray cast"), the
  trilinear interpolation of the 8 voxels' colors at the crossing, each voxel holding the running
  average of the colors of the frames that observed it.
* **The shading**, the model's color lit lightly by the normal (the user's choice of 2026-11-17,
  "模型颜色加轻度光照", brief §7, 49): a pixel with a normal n, a unit vector toward the camera,
  and its point p in the camera frame shows its color c as c (a + (1 - a) max(0, n · (-p / |p|))),
  each channel rounded to the nearest integer, a = 0.6 and 1 - a = 0.4 [?: initial]: Lambert's
  cosine law [S: J. H. Lambert, Photometria, 1760] with the light at the camera [?: a choice, a
  headlight], so that a surface facing the camera shows its color and one seen at a slant darkens
  toward 0.6 of it, which shows the shape where the color shows none. A valid pixel without a
  normal shows its color unlit; a pixel whose ray met no surface shows the background,
  (48, 48, 48) [?: initial: a dark grey, so that a black object stands out from it]. A kernel of
  one thread a pixel, in the stages' block of 32 x 8, reads the surface's level 0 and writes the
  picture in OpenCV's order, blue, green, red, so that the encoder takes it without a conversion.
* **To the host.** After a fused frame has gone to the writer, the scan thread offers its view:
  when the encoder is free, it queues on the scan's stream the shading into a picture on the GPU,
  the picture's copy into pinned memory and an event after them, then hands the frame's index,
  timestamp and size to the encoder's thread and goes on; when the encoder is busy, the frame has
  no view: skipped, never queued. One view is in flight at a time, the newest a free encoder took,
  and the scan thread never waits for the encoder: a view costs the frame the shading and the copy
  on the GPU and the offer on the host.
* **The encoder**: a thread of the scan's own, started with the scan's buffers and ended with
  them, which finishes the view it was handed first. It waits for the event (cudaEventSynchronize
  on an event made with cudaEventBlockingSync, so that the thread sleeps rather than spins
  [S: cuda_runtime_api.h, CUDA 12.6]), encodes the pinned picture with OpenCV's `imencode` as a
  JPEG of quality 85 [?: a choice; its bytes and time below], publishes it, calls the live view's
  listener, and is free when the listener has returned. So the frames offered while the listener
  runs have no view, and a view published is never older than its frame's offer by more than one
  view's making: newest only, not a queue of one. JPEG on the CPU is the first choice (the user's
  answer, the ticket's Comments); the board's JPEG engine is not tried unless the CPU's time asks
  for it [?].
* **At the scan's end** the newest view shows the last fused frame: the scan thread waits until
  the encoder is free (the end may wait), and when that frame's view was skipped it queues it, the
  surface unchanged since (a lost frame casts nothing), and hands it over together with the end,
  in one step of the encoder's lock, so that the thread makes it before it ends
  (`live_view_encoder::finish()`). Not after a frame tracking failed on, whose failed cast may
  leave the surface unspecified.
* **Its publication**: under a lock of its own, neither the state's nor the live model's: the
  encoder's thread swaps the new view in, a reader copies the newest
  (`scan_interface::latest_live_view()`). A scan's start publishes an empty view, without a frame.
  A lost frame offers nothing: the newest view stays as it is, and the state says loss.
* **A failure**: CUDA refusing the shading's launch, the copy or the event's record on the scan
  thread, or failing in the event's wait on the encoder's, and `imencode` refusing the picture:
  no view is published, the reason goes into the state's `live_view_failure`, which the service
  logs (docs/SERVICE.md), and the scan goes on. A failure on the scan thread is its offer's, and
  the next tracked frame offers its view; the encoder's reaches the state through the next offer,
  which reports it and queues nothing, one value for one outcome [?: a choice], so the offer after
  it is the next to queue a view. No test reaches it: it takes a CUDA failure in the view's work
  alone, as the live model's does.
* **Its memory**, requested with the tracker's buffers before the pool closes: the picture on the
  GPU and its pinned copy, 3 bytes a pixel of level 0 each, 0.77 MB at 640x400 and 3.07 MB at
  1280x800 [S: computed], beside the surface's colors (the buffers of "The path of a frame"); one
  event; on the host the bytes of the JPEG the encoder fills, reserved for 3 bytes a pixel with
  its request, which it swaps with the newest view's, reserved as much when the scan interface is
  built. The encoder's thread allocates for the JPEG (`imencode` and libjpeg), off the scan
  thread.
* **Its cost** at 640x400 [M 2026-11-17: docs/HARDWARE.md, "The live view"]: the colors add 0.8 %
  at most to the cast back to back and 0.5 % paced; the shading and the copy take 0.29 ms on the
  GPU back to back and 0.91 ms paced, the offer 0.02 to 0.06 ms on the scan thread; the encoder
  publishes a view 2.3 ms after the offer back to back and 5.5 ms paced, `imencode` taking 1.9 ms
  and 4.6 ms of it; a view of the synthetic scenes takes 26 to 31 KB. The replay's `live_view`
  column, the shading and the copy with the measurement's wait, is a median 0.182 ms on the
  synthetic walk and 0.204 ms on the user's first scan with the service, every tracked frame of
  both with a view.

## The stages' times

For a measurement, the replay tool hands the scan interface a stage clock
(`src/tracking/stage_clock.hpp`; `scan_options::stage_times`), which `tracker::track()` takes too.
The scan's loop begins a frame's clock once its gap is counted, before the writer's room, and
marks the end of each stage; after a stage that queues work on the GPU the clock waits for the
stream (`cudaStreamSynchronize`), so that the stage's time holds that work, and each stage's time
is the time since the mark before. A frame's time runs from its beginning to the end of step 8
above, the commands answered; the stages leave out only the state's publication and the commands.
The stages, each by the name the time lines print:

| name | step | waits for the GPU |
|---|---|---|
| `room` | the scan's loop, step 2: the writer's room | no |
| `too_close` | the scan's loop, step 3: the "too close" notice | no |
| `upload` | the path, step 1: the upload | yes |
| `pyramid` | the path, step 2: the pyramid | yes |
| `ray_cast` | the path, step 3: the surface, the live model's ray cast after the fusion | yes |
| `registration` | the path, step 4: the registration | yes, already idle |
| `check` | the path, step 5: the check | no |
| `fusion` | the path, step 6: the fusion | yes |
| `writer` | the scan's loop, step 5: the writer takes the frame | no |
| `live_model` | the scan's loop, step 6: the live model, when the frame takes one | its own waits |
| `live_view` | the scan's loop, step 6: the live view's shading and copy, when a free encoder takes the frame's | yes |

The columns stand in the table's order; the stages run in the path's, the ray cast after the fusion
and the live view before the live model. A stage a frame does not run takes 0: the ray cast and the
fusion of a frame in tracking loss, the stages after the one that failed. Every other stage takes
more than 0 but `room` and `check`, host steps that can take less than the lines' 0.0005 ms and print 0.000
(`room` did in 2 of 1440 frames [M 2026-11-16: docs/HARDWARE.md, "The replay's frame time"]), and
`live_model`, which on a frame that takes none is the pacing's decision alone and prints 0.000 or
nearly; the scan's first frame takes one, and its `live_model` holds the extraction's time; and
`live_view`, which on a lost frame and on one whose encoder is busy is the offer's test alone.
The waits serialize the host and the GPU, which overlap without them (the fusion runs while the
next frame's "too close" pass and upload run on the host), so a frame takes longer with the clock
than without it (docs/HARDWARE.md, "The replay's frame time"). A live scan has no clock: each use
is behind a test of a null pointer, and nothing more waits.

## The replay tool's lines

`build/apps/replay` (AGENTS.md §3) prints these lines, an interface (AGENTS.md §7 rule 5): the
measurement tickets read them, and ctest `replay` parses them as written here.

- With `--times`, one time line on stdout for each frame the loop handles, as the frame ends:

      frame <index> <state> room <t> too_close <t> upload <t> pyramid <t> ray_cast <t> registration <t> check <t> fusion <t> writer <t> live_model <t> live_view <t> total <t>

  `<index>` is the frame's index, `<state>` `tracking`, `lost`, `failed` or `not_taken` (a frame
  the writer had no room for, which a replay does not have), each `<t>` a time in ms with three
  decimals, the stages in the table's order and `total` the frame's time. The stages add up to
  `total` within 0.1 ms, the margin ctest `replay` checks: 0.027 ms at most in replays of 1440
  frames at 640x400 and 720 at 1280x800, 0.004 ms in 6 runs of the test's 30 frames
  [M 2026-11-16: docs/HARDWARE.md, "The replay's frame time"], the rounding of eleven times to
  0.001 ms among it.
- Last, when the scan was refined, on stdout:

      replay: <OUT>/<scan>: refined, end <end>, frames taken <T> fused <F> lost <L> not taken <N>, <x> ms a frame

  `<scan>` is the scan's directory's name, `<end>` the key of why the scan ended
  (docs/SCAN_FILES.md, "Parameters"), the counts the parameters', and `<x>` the scanning phase's
  time, from its first state to its last frame's, divided by the frames taken, with three
  decimals: the frame time with the source's reads, and with the time lines' printing when they
  are asked for. When the scan ended other than at the sequence's end, stderr adds `replay: the
  scan ended before the sequence's end: <message>`.
- A failure, on stderr: `replay: ` and the reason (the parameters' reader's, the sequence's
  header's, `the scan's start was refused: <reason>`, or `<OUT>/<scan>: unfinished: <message>`)
  with exit 1; a command line refused, `replay: <reason>` and the usage, with exit 2
  (`apps/replay_options.hpp`).

## What the prototype did, and what is not carried

The prototype's loop (CUDA_3D's `apps/app.cpp`, deleted in eabb407) registered each frame against
the previous frame and on every fifth frame against its model, applied the result unchecked, and
fused by appending the frame's points (`.scratch/rewrite/requirements.md`, section 11). Not
carried:

* **The unchecked pose**: a diverged registration was fused. Here every frame passes the check
  first: a registration that ended with too few correspondences or degenerate is tracking loss,
  and a refused or failed one the tracker's failure, which ends the scan ("Output").
* **The motion gate**: a step under 0.081° and 1 mm was not applied while the reference advanced,
  so slow motion drifted (docs/methods/sparse-tsdf.md, "What the prototype did"). Every frame that
  passes the check is fused at its pose, however small its step.
* **Registration against the previous frame**: here always against the model's surface, which
  holds every frame fused so far.

