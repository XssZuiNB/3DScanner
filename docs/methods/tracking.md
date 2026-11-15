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

Built: the tracker and the check (`src/tracking/tracker.*`; the test `tests/test_tracking.cpp`).
Left to `relocalize-from-keyframes`: keyframes and the candidates they add for a lost frame. Left
to `tracking-thresholds-from-real-sequences`: the thresholds' values and ε, measured on real
sequences. Left to the scan interface (`scan-interface-first-whole-scan`): the scan's steps 7 and
8, the sequence's writer and the published state, and the "too close" notice (R36), whose
threshold the object class holds ("Parameters").

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
  (`src/scan/object_class.cpp`): the pyramid's depth range and normals'
  (docs/methods/frame-pyramid.md); the registration's (docs/methods/registration.md,
  "Parameters"); the check's thresholds ("Parameters" below). The live model's sparse TSDF takes
  its own (docs/methods/sparse-tsdf.md): the object class's live voxel size, memory budget and far
  depth. The depth range is 0.25 m to 0.6 m for the tabletop object and 0.25 m to 3.0 m for the
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
  it leaves. A scan ends on it.

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
   (`sparse_tsdf::ray_cast()`), at the frame's intrinsics and the same depth range, every level
   of the pyramid (docs/methods/sparse-tsdf.md, "The ray cast"). Before the first fused frame there
   is no model: see "The first frame".
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
   time.

On tracking loss P_L and the last tracked time stay as they were, and the model is not touched.

**The buffers** of every step are requested from the pool (`src/gpu/gpu_pool.hpp`) by one request
before the first frame, for one frame size, and the pool is closed then; `track()` allocates
nothing (R29). Besides the live model's sparse TSDF, whose device memory is the memory budget,
they take 25.1 MB of device memory and 1.28 MB of pinned memory at 640x400, 100.4 MB and 5.12 MB at
1280x800 [S: computed: the GPU frame 5 bytes a pixel and its staging buffer as much, the pyramid
and the surface 29 bytes a pixel of each level, the surface's tiles 0.125 bytes a pixel of level 0,
the registration 12 bytes a pixel of each level and 280 bytes a block of 32 x 8 pixels; the levels
hold 1 + 1/4 + 1/16 of the pixels]. The live model is requested first, so that a closed pool checks
every parameter without device memory (`sparse_tsdf::request()`).

**The waits**: the upload's wait on its own staging buffer, which returns at once when the frame
before was waited for (`src/gpu/gpu_frame.hpp`), and the registration's one written wait per
iteration, which also waits for the stream's earlier work, the previous frame's fusion included.
Nothing else waits: the fusion is queued and the call returns; the next frame's registration, or
the caller's own wait, finds it done. What comes back to the host is the registration's 280 bytes
per iteration (docs/methods/registration.md, "On the GPU") and nothing more.

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
stay. Each following frame goes through the same path: it is registered against the model's
surface ray cast from the last tracked pose, from the identity, under the same check, Δt counted
from the last tracked frame. The first that passes is fused and tracking goes on from it.
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

