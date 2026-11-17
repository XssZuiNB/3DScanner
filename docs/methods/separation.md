# Separation: the proposal of the support plane and the box

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

After refinement the scanner proposes the separation of the full model (R15; the spec,
"Separation"): the support plane, the dominant plane under the object, and a box that stands
upright on it, one axis along the plane's normal, around what stands on the plane where the views
of the trajectory meet. The operator changes plane and box on the page where needed and confirms;
the cut then keeps the full model's points inside the box and above the plane by one full voxel
(`separate-object-model`). Gravity's up comes from the accelerometer's samples of the scan's
frames and its trajectory, estimated once at refinement and kept in the scan's parameters; the
proposal reads the full model, the trajectory and that estimate, and changes none of them (R14,
R16).

The method composes published pieces: the plane by random sample consensus [S: M. A. Fischler,
R. C. Bolles, "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image
Analysis and Automated Cartography", Communications of the ACM 24(6), 1981] refined by the
orthogonal least-squares plane [S: K. Pearson, "On Lines and Planes of Closest Fit to Systems of
Points in Space", Philosophical Magazine 2(11), 1901]; what stands on it as the connected pieces of
its footprint, joined by union-find [S: R. E. Tarjan, "Efficiency of a Good But Not Linear Set
Union Algorithm", Journal of the ACM 22(2), 1975]; the box's turn from the convex hull [S: A. M.
Andrew, "Another efficient algorithm for convex hulls in two dimensions", Information Processing
Letters 9(5), 1979] and its minimum-area enclosing rectangle, which has a side along an edge of the
hull [S: H. Freeman, R. Shapira, "Determining the minimum-area encasing rectangle for an arbitrary
closed curve", Communications of the ACM 18(7), 1975]. Plane first, then the clusters above it, is
the tabletop pipeline of R. B. Rusu, N. Blodow, Z. C. Marton, M. Beetz, "Close-range Scene
Segmentation and Reconstruction of 3D Point Cloud Maps for Mobile Manipulation in Domestic
Environments", IROS 2009 [?: the paper's pipeline as recalled, not re-read here]; there the robot
knows which way is up, here gravity tells it, through the camera's accelerometer, and the
trajectory tells where the views meet. Up is the mean of the accelerometer's specific force turned
into the models' frame by the trajectory, the levelling of inertial navigation done while the
camera moves, its attitude known from tracking [S: D. H. Titterton, J. L. Weston, "Strapdown
Inertial Navigation Technology", 2nd ed., IET 2004, the navigation equation and the alignment];
the attitude between two frames by spherical linear interpolation [S: K. Shoemake, "Animating
Rotation with Quaternion Curves", SIGGRAPH 1985]. This tree's code is written from these sources;
no reconstruction library's code is read or copied for it (docs/adr/0002).

Built: gravity's estimate (`src/separation/gravity.*`, `estimate_gravity()`), the proposal
(`src/separation/proposal.*`, `propose_separation()`), both on the host in double precision, and
the orthogonal least-squares plane (`src/separation/plane_fit.*`), which distortion_check's
measurement shares; their test `tests/test_separation.cpp`, whose samples come from the synthetic
source's IMU (`tests/synthetic_source.hpp`). The scan interface estimates gravity at refinement and
keeps it in the parameters ("What it reads from a scan"). Left to `separate-object-model`: the
proposal called at refinement's end with that estimate, the cut, the plane and box in the scan's
parameters.

## Inputs and output

Gravity's estimate takes:

* **The trajectory**, as the scan keeps it (`src/types/trajectory_entry.hpp`, docs/SCAN_FILES.md,
  "Trajectory"): for each frame the scanner took, in the order taken, its timestamp t_j in µs on
  the frames' clock and its pose, camera to model (R_j, c_j), or the mark of tracking loss. The
  timestamps rise strictly. The frames in tracking loss bound where samples are taken.
* **The accelerometer's samples** of the scan's frames (`src/types/imu_sample.hpp`,
  docs/methods/imu-samples.md): each a specific force f_k in m/s² along the IMU's axes, at t_k on
  the frames' clock, strictly rising; none when the camera opened without its IMU.
* **Q**, the rotation from the IMU's axes into the camera frame, the camera's
  (`gca::imu_to_camera()`, docs/methods/imu-samples.md, "The axes").
* It gives, a value for every call (`src/types/gravity_estimate.hpp`): how it ended, one key of a
  closed set, and with up the samples taken, the groups, their spread and the mean's length.

The proposal takes:

* **The full model**: its points' positions in the models' frame, the camera frame of the first
  fused frame (docs/SCAN_FILES.md), in m. Normals and colors are not read: the fused model's normals
  are off by a median of 6.6° to 6.8° at the final voxel sizes (docs/methods/sparse-tsdf.md, "What
  the extraction gives").
* **The tracked frames' poses**, camera to model (R_i, c_i), R_i's third column z_i the optical
  axis; frames in tracking loss left out by the caller.
* **Gravity's estimate**, as above or as the scan's parameters keep it.
* **The parameters**: v, the voxel size the full model was fused at (`full_voxel_size_m`), which
  scales every distance below, and D, the farthest depth a frame contributes (`max_depth_m`), which
  bounds how far from the views the proposal looks, both from the scan's object class
  (`src/scan/object_class.cpp`).
* It gives, a value for every call: how it ended, the reason in plain English when it did not
  propose, and with a proposal the plane and the box; and the evidence found on the way: gravity's
  up, the views' meeting point, the window, the plane's points and the object's points.

**The plane** is n · p = d: n the unit normal, pointing to the side the object stands on, d in m.
A point's height above it is h(p) = n · p - d.

**The box** has a pose, box frame to models' frame, and three edge lengths L_x >= L_y and H. Its
frame's origin is the centre of its bottom face, which lies in the plane; its z axis is n; its x axis
lies along the box's longer side in the plane, of either sign. A point q of the box frame is inside when
|q_x| <= L_x / 2, |q_y| <= L_y / 2 and 0 <= q_z <= H.

## Gravity's up

The models' frame is the first camera's, so the support plane lies at any tilt in it and no axis of
it is up. Up is gravity's, from the accelerometer (the user's decision of 2026-11-15, A9).

**What the accelerometer reads.** An accelerometer measures specific force, the acceleration of
its case less gravity: f = a + g u, with g u the reaction to gravity, pointing up; held still it
reads g pointing up [S: Titterton and Weston 2004, the navigation equation;
docs/methods/imu-samples.md, "A sample"]. The camera's reads 9.61 m/s² pointing up in its y-down
frame, held roughly still [M 2026-11-16, early and not reviewed: `hardware-check-orbbec-on-seam`'s
Comments].

**Into the models' frame.** A sample f_k, taken at t_k along the IMU's axes, turns into the camera
frame by Q and into the models' frame by the camera's orientation at t_k:

    s_k = R(t_k) Q f_k = a(t_k) + g u,

a(t_k) the camera's acceleration in the models' frame, u the models' up [S: computed]. R(t_k) comes
from the two consecutive frames of the trajectory whose timestamps enclose it, t_j <= t_k <=
t_{j+1}: the spherical linear interpolation of their quaternions at (t_k - t_j) / (t_{j+1} - t_j),
the shorter way round [S: Shoemake 1985; Eigen's `slerp`]. Both frames must be tracked: a sample
next to a frame in tracking loss, before the trajectory's first frame or after its last is left
out. Slerp is exact for a turn at a constant rate, as the synthetic source turns between frames;
otherwise it errs by about Δt² |ω'| / 8, below 0.01° for a turn rate that changes by 63°/s²
within the Δt = 33 ms between frames [S: computed]. The interpolation counts where the camera
turns about a level axis: on half the tabletop's lap, rolling about the optical axis at 90°/s, u
lies 0.068° from up, and 0.746° when each sample takes the orientation of the frame before it
[M 2026-11-17: `tests/test_separation.cpp`, and with that change made to `orientation_at()`].
Tracking passes a frame only when the camera turned by at most ω_max = 90°/s since the frame before
(docs/methods/tracking.md, "Parameters"). A frame the sequence's writer did not take is in neither
the trajectory nor the sequence, its samples lost with it (docs/SEQUENCE_FILE.md): the samples
there are fewer, and the two frames around the gap interpolate over a longer step. Such gaps are
not "no samples": gravity takes the samples there are (the user's decision of 2026-11-16).

Q is the identity [?: initial] (docs/methods/imu-samples.md, "The axes"); an early fit put the IMU's
axes 5.2° from the camera's, within the fit's error [M 2026-11-16, early and not reviewed: the same
Comments]. Q wrong by δ about an axis e turns each s_k by δ about R(t_k) e: u then leans by at most
δ, and on a lap, where R(t_k) e circles about up, the turned samples lie on a cone about up whose
lean largely cancels [S: computed].

**The mean.** Gravity is the mean of the K samples taken,

    ĝ = (1/K) Σ s_k,   u = ĝ / |ĝ|.

The hand's acceleration averages out: over a stretch of tracked frames from t_a to t_b the
integral of a is v(t_b) - v(t_a), so the mean of a over the samples is the sum of the stretches'
changes of velocity over the time they cover [S: computed]. Tracking passes a frame only below
v_max = 1.0 m/s (docs/methods/tracking.md, "Parameters"), so one stretch of T seconds leans u by at
most atan(2 v_max / (T g)): 2.3° at 5 s, 0.58° at 20 s [S: computed], and less when the hand rests
at the start and the stop. Every stretch between two tracking losses adds its own change; their
sum grows with their count, and the spread below shows it. With the test's hand shaking by 2 cm at
1.5 Hz, 1.78 m/s² at its peak, 10.3° of a single sample's direction [S: computed], u lies 0.073° from
up on the tabletop's lap of 20 s, 0.076° with a second of tracking loss in it, and within 0.0934° on
every walk of the test [M 2026-11-17: `tests/test_separation.cpp`] ("Accuracy").

The camera's acceleration could be taken out instead of averaged out, as visual-inertial
estimators do from the trajectory's positions [S: T. Qin, P. Li, S. Shen, "VINS-Mono: A Robust and
Versatile Monocular Visual-Inertial State Estimator", IEEE Transactions on Robotics 34(4), 2018];
not built: a tracked position's noise of about 1 mm gives its second difference a noise of about
2.2 m/s² a frame [S: computed, √6 · 1 mm / (33 ms)²], which would itself have to average out. It is
built only on a measured need.

What else moves the mean: the time between the IMU's clock and the frames', an offset of +13 ms
not decided [M 2026-11-16, early and not reviewed], turns each sample by the camera's turn in that
time, which averages out as the camera tilts back and forth [S: computed]; a bias b of the
accelerometer adds R(t_k) Q b, up to atan(|b| / g), 1.17° at 0.2 m/s² [S: computed]; an error of its
scale changes |ĝ| alone. |ĝ| is reported, not judged.

**The spread.** The samples taken, in time order, form groups of 200 [?: initial: one second at the
IMU's 200 Hz, `orbbec_imu_rate_hz`, which ctest `separation` holds equal at its build], the last
group of fewer left out; each group's mean m_b, and the spread is the root mean square of the angles
between the m_b and ĝ, each angle atan2(|m_b x ĝ|, m_b · ĝ). A hand's shake averages out within a
second as over the scan: the test's shake leaves a group's mean up to 0.38 m/s², 2.2° off up
[S: computed, the change of its velocity in a second]. A wrong turn of the samples does not average
out: a wrong Q, or a trajectory gone wrong, turns the groups' means apart as the camera turns, Q
wrong by δ by up to δ [S: computed]. **Gravity is not trusted**, and the proposal ends with no
gravity, when the samples fill fewer than 5 groups [?: initial: five seconds of tracked frames,
where one stretch's lean is 2.3° at most], when ĝ is 0, or when the spread exceeds 10° [?: initial:
23 times the recording's below and 6 times the test's shaking hand, far below the 45° of the plane's
cone]. Measured [M 2026-11-17: `tests/test_separation.cpp`, `tests/test_scan_interface.cpp`]: the
test's laps and half laps 0.19° to 0.40°, shaken 1.62° to 1.66°, the scan interface's walk of a lap
in 8 s 1.77°; samples turned by a Q of 90° about (1, 1, 1) and given with the identity 50.4°,
refused, while given with that Q they are right within 0.04°; three seconds of samples, 2 groups,
refused; the lap's first 1000 samples, 5 groups, estimated, and its first 999, 4 groups, refused.

**On the camera.** The sequence of 2026-11-16 (`hardware-check-orbbec-on-seam`'s Comments: 20 s,
held roughly still, then turned slowly about its three axes) and its replay's trajectory, 390 of 601
frames tracked, gave by this method, Q the identity [M 2026-11-16 recording, early: a host-only
scratch script on the replay scan's trajectory, its sequence's samples and its full model, kept
outside the tree]: 2567 of the 4037 accelerometer samples taken, the others next to tracking loss,
12 groups, |ĝ| 9.745 m/s², a spread of 0.43°, the groups' means 0.18° to 0.90° from ĝ, a single
sample 1.53° in the root mean square. The full model's dominant level plane, 0.69 m below the first
camera along u, 321 296 points within 12 mm of its fit, lies 0.86° from u: the sign is right and Q
lies within about a degree of the identity, or the plane is not quite level. Shifting the samples'
times by the undecided ±13 ms moves u by 0.05°.

**No samples.** A scan whose camera opened without its IMU (`No IMU:`, docs/methods/imu-samples.md)
has no accelerometer samples: the proposal ends with no gravity, and the operator sets plane and
box (the user's decision of 2026-11-16). The cameras' axes are not taken instead: there is one way
to know up (AGENTS.md §7 rule 6), and they fail where gravity does not ("The roll").

**The roll.** Before gravity, up came from the cameras' axes, the camera held roughly upright: the
direction most nearly perpendicular to every camera's x axis that stays near the mean of their up
axes, -y [S: the proposal of 1ffa180]. A camera held rolled tilts every x axis out of the level plane
the same way; on a whole lap the tilts cancel, on half a lap that up absorbs them and leans sideways
by about the roll [M 2026-11-15: `tests/test_separation.cpp` at 1ffa180, every camera of a scene's
walk rolled about its optical axis]:

| scene, walk | the cameras' axes' up from up, rolled by 15°, 30°, 40°, 45°, 60° | the proposal |
|---|---|---|
| tabletop, a lap | 0.02°, 0.05°, 0.09°, 0.11°, 0.20° | right at every roll |
| large, a lap | 0.09°, 0.12°, 0.16°, 0.19°, 0.39° | right at every roll |
| tabletop by a wall, half a lap looking down by 60° | 9.62°, 18.4°, 23.9°, 26.4°, 33.4° | right at every roll |
| large by a wall, half a lap | 17.1°, 33.3°, 43.4°, 48.1°, 60.5° | right up to 40°; no plane at 45° and 60° (the band's spread) |
| large in a room's corner, half a lap, rolled by ±30°, ±45°, ±60° | 33.3° and 33.7°, 48.1° and 48.7°, 60.5° and 61.0° | right at ±30°; at ±45° and ±60° a side wall, 90° off |

So the cameras' axes were right up to 30° of roll on every scene tested, and beyond it half a lap
could end with no plane or, in a room's corner, propose a side wall. No measure of the roll could
refuse those and keep every right proposal [?: the orchestrator's choice, 2026-11-15]. Gravity does
not lean with the roll: on the same scenes and rolls it lies where it lies without one, within
0.0934° of up, and every proposal is right, the corner's at ±45° and ±60° among them
[M 2026-11-17: `tests/test_separation.cpp`].

## The views' meeting point

The operator walks around the object and points the camera at it, so the optical axes pass near
it. The meeting point m is the point nearest to the N tracked frames' axes in the least-squares
sense:

    m = A^-1 b,   A = Σ (I - z_i z_i^T),   b = Σ (I - z_i z_i^T) c_i,

the normal equations of Σ |(I - z_i z_i^T)(m - c_i)|², each term the squared distance of m from
axis i [S: computed]. A / N's least eigenvalue is the least mean of sin² of the angles between the
axes and a direction; under sin² 5° = 0.0076 [?: initial] the axes are near parallel, as in a sweep
past the object without turning, and the views do not meet. Every axis counts alike: a view that
looked elsewhere moves m [?: a choice; a robust weight is built on a measured need].

## The window

The proposal looks at the full model's points within W of m,

    W = min(median_i |c_i - m|, D):

the cameras walk around the object, so the object lies nearer to m than the cameras do, and what
stands farther than the median camera is background [?: a choice]. The median, not the largest
distance, so that a few views from afar do not widen it. D bounds W, so that every coordinate below
is bounded; D is refused beyond 10^5 v, which keeps the footprint's cells within ±4·10^4 of m's foot
[S: computed, 2 D / c; the object classes' D / v are 600 and 750].

## The support plane

**Hypotheses.** A sample of 20 000 of the window's points [?: initial], drawn with replacement by
std::mt19937_64 from a fixed seed, each index its 64-bit output modulo the window's count, a bias
below 10^7 / 2^64 = 5.4·10^-13 for windows of up to 10^7 points [S: computed]; all of the window's
points when it holds fewer. 4000 hypotheses [?: initial]: each the plane through three sampled
points, its normal turned so that n · u >= 0. A hypothesis is a candidate only where it lies **under
the object**:

1. its normal within 45° of gravity's up, n · u >= cos 45° [?: initial: far above gravity's error
   and a support's tilt, far below the 90° of a wall];
2. m above it, h(m) > 0: the object stands on the plane, and the views meet on the object.

A candidate's score is the count of sampled points within d_in = 3 v of it [?: initial: the fused
model's points lie within a 99th percentile of 0.60 mm at 1 mm voxels and 4.97 mm at 4 mm,
docs/methods/sparse-tsdf.md, "What the extraction gives"]; the highest score wins, the first on a
tie. The support plane is so the dominant plane under the object: the one with the most points in
the window among the planes that pass both conditions. A plane with more points elsewhere, or a wall
beside the object, does not take its place. With a share w of the window's points on the support
plane, 4000 hypotheses miss it with the chance (1 - w³)^4000: 1.8 % at w = 0.10, 0.1 % at w = 0.12,
10^-6 at w = 0.15 [S: computed; Fischler and Bolles 1981].

**Fits.** The plane is then fitted three times [?: initial] to the window's points within d_in of
it: through their centroid, the normal the eigenvector of the least eigenvalue of their centred
covariance, summed in double (CODING_STANDARDS.md, "Numerics"), turned so that n · u > 0 [S: Pearson
1901; `gca::fit_plane()`]. Both conditions are then checked again on the fitted plane: a candidate
that cuts across a surface at a slant holds a strip of that surface, whose fit is the surface. A
steep ramp alone, 55° from up, gave such a candidate within the 45° cone, whose fit lay 55.0° from
up [M 2026-11-15: `tests/test_separation.cpp`, "a steep ramp alone"].

**No plane.** The proposal ends with no plane when no hypothesis is a candidate; when the plane,
before a fit or after the last, holds fewer than 10^4 of the window's points within d_in
[?: initial]: a patch of 100 x 100 voxels, 10 cm square at 1 mm voxels, 40 cm at 4 mm, at the density
of the extraction on a plane along the grid, one point per voxel column; a tilted plane gives up to
√3 as many [S: computed: the surface crosses |n_x| + |n_y| + |n_z| voxel edges per column]; when the
fitted plane leaves the conditions; or when the last band's points lie farther than v from the plane
in the root mean square [?: initial]. A surface lies within its noise of its plane, and the band
also holds the feet of what stands on it: 0.30 to 0.46 v in the test's scenes, with noise of 0.25 v
[M 2026-11-15: `tests/test_separation.cpp`]. A fit that keeps cutting across surfaces at a slant
spreads its band's points evenly across the band, d_in / √3 = 1.73 v [S: computed], 1.73 v in the
test [M 2026-11-15: the large scene by a wall, rolled by 45° and 60°, whose fits lay 3.5° and 16.9°
off the floor]. A sphere of radius r has no flatter slab of thickness 2 d_in than a zone of
area 4π r d_in [S: computed, Archimedes' hat-box theorem], so a sphere below 10^4 v / (12π) = 265 v
gives no plane: 0.27 m at 1 mm, 1.06 m at 4 mm [S: computed]; the test's sphere of 0.1 m gives 3774
points against the 3770 of the formula [M 2026-11-15].

## What stands on the plane

The window's points higher than d_in above the plane stand on it. In the plane's frame, origin o =
m - h(m) n, the foot of m, axes e1 (the models' axis least aligned with n, projected onto the plane
and normalized) and e2 = n x e1, each such point has the coordinates (a, b). The plane is divided
into square cells of c = 5 v [?: initial], cell (⌊a/c⌋, ⌊b/c⌋); the occupied cells join their eight
neighbours into pieces, by union-find over the cells sorted by their coordinates. Two points nearer
than c always lie in one piece and two farther apart than 2√2 c = 14.1 v never join through their
own cells [S: computed]: 5 mm and 14 mm at 1 mm voxels, 20 mm and 57 mm at 4 mm. A box beside the
object, 3 v from it, joins it, the box holding all of its points, and one 18 v from it does not, the
box holding none, at both object classes' voxel sizes [M 2026-11-15: `tests/test_separation.cpp`].

**The object** is the piece of the occupied cell nearest to o, the distance from o to the cell's
square, 0 for o's own cell; the first cell in sorted order on a tie. When nothing stands on the plane
in the window, the proposal ends with nothing on the plane.

## The box

The object's points' (a, b) give their convex hull [S: Andrew 1979], the hull its minimum-area
enclosing rectangle, which has a side along one of its edges [S: Freeman and Shapira 1975]: each edge
is tried, each one's rectangle measured over the hull's vertices, the least area taken, the first on
a tie. A hull of one point gives a rectangle of no extent along e1, one of a segment a rectangle of no
width along it. The rectangle grows by the margin μ = 5 v on every side [?: initial: the band of d_in
under the footprint's lowest points, whose points the cut keeps above one voxel, and the fused corner
between object and plane], the height is the object's highest point plus μ, and the bottom stays in
the plane. The box's x axis is the rectangle's longer side, of the sign the hull's edge gives;
y = n x x; z = n.

## The ends

* **proposed**: plane, box and evidence.
* **no plane**: no plane under the object ("No plane").
* **nothing on the plane**: a plane, but nothing stands on it in the window. No box, and no plane
  is proposed alone [?: a choice: the operator sets both by hand, as with no plane].
* **the views do not meet**: the tracked frames' optical axes are near parallel.
* **no gravity**: gravity's estimate gives no up: none was estimated, no accelerometer sample,
  the camera having opened without its IMU, the samples between tracked frames fill fewer than 5
  groups, their mean is 0, their spread exceeds 10° ("The spread"), or the estimate refused its
  inputs; the message says which, with the counts. The operator sets plane and box. Checked before
  the views, so that a scan without samples always says why.
* **refused**, before anything is computed: v or D not finite and > 0, or D above 10^5 v;
  gravity's outcome no key of its set, or an estimated up not finite or whose length differs from 1
  by more than 1e-9; no pose; a camera position not finite or a quaternion whose norm differs from 1
  by more than 1e-9 (the trajectory's own check, docs/SCAN_FILES.md); a point not finite.

Gravity's estimate ends refused, before anything is computed, for Q's quaternion whose norm differs
from 1 by more than 1e-9, a frame whose timestamp does not rise, a tracked frame's pose that fails
the trajectory's check, and a sample whose time does not rise or whose value is not finite; the
refusal names it.

## What it reads from a scan

The accelerometer's samples live in the scan's sequence (docs/SEQUENCE_FILE.md), which the service
removes once the scan is refined unless it is kept (docs/SCAN_FILES.md, "The sequence switch"),
while a kept scan can be separated later, after the service has started again. So gravity is
estimated once, at refinement: refinement's fusion reads the sequence anyway and gives the samples
of every frame with the trajectory it read (docs/methods/refinement.md, "The fusion of the
sequence"); the scan interface estimates gravity from them with `gca::imu_to_camera()` and writes
the parameters again with the estimate, before the trajectory, the full model and the sequence's
removal. The parameters keep the estimate as the scan's facts (docs/SCAN_FILES.md, "Parameters",
`gravity`): up, the samples, the groups, the spread, the mean's length, and the outcome, a key of a
closed set that says why a scan has none. The samples themselves are not kept, so what a scan keeps
stays R17's. Whether an unfinished scan's parameters hold the estimate or `not_estimated` depends
on where it ended (docs/SCAN_FILES.md, "Parameters", `not_estimated`); a replay of its kept
sequence (`build/apps/replay`) is a scan of its own and estimates gravity anew. The proposal takes
the estimate as the parameters hold it (`separate-object-model`).

## Constants and parameters

The parameters (`gca::separation_parameters`): v and D, each the scan's object class's; gravity's
estimate takes Q, the camera's. The constants, each [?: initial] until real scans measure them:

| constant | meaning | value |
|---|---|---|
| | the samples of a group of gravity's spread | 200 |
| | the least groups of samples | 5 |
| | gravity's largest spread | 10° |
| | the least of A / N's eigenvalues | sin² 5° = 0.0076 |
| | the sample, the hypotheses | 20 000 points, 4000 |
| θ_up | the largest angle of the plane's normal from up | 45° |
| d_in | the plane's band | 3 v |
| | the plane's fits | 3 |
| | the plane's least points in the window | 10^4 |
| | the band's largest root mean square distance | v |
| c | the footprint's cell | 5 v |
| μ | the box's margin | 5 v |

## Accuracy

On full models `tests/test_separation.cpp` builds: the known surfaces sampled one point per voxel's
square, each moved along its normal by Gaussian noise of 0.25 v [?: a choice: the fused models'
99th percentiles over 2.6, without their tails] from a fixed seed, in the frame of the walk's first
camera; the walks at 30 frames a second, a lap of the tabletop in 20 s and of the large object in
40 s [?: a choice], the accelerometer's samples from the synthetic source at 200 Hz. The plane's
angle from the known one, its distance from it at the known box's foot, the box's faces' excess over
the known box, gravity's up from the known up [M 2026-11-17: `tests/test_separation.cpp`]:

| scene | points | plane | distance | excess, margin μ | up |
|---|---|---|---|---|---|
| tabletop box among four things (one larger in points, one 3 cm beside it), a lap of 600 | 960 908 | 0.0020° | 0.017 v | 5.88 to 6.16 mm, 5 mm | 0.038° |
| large box among three things (one larger in points), a lap of 1200 | 2 048 582 | 0.0018° | 0.017 v | 24.01 to 24.88 mm, 20 mm | 0.052° |
| tabletop box by a wall, half a lap of 299, looking down by 60° | 1 567 998 | 0.011° | 0.027 v | 5.90 to 5.97 mm, 5 mm | 0.053° |
| large box by a wall, half a lap of 599 | 1 624 996 | 0.0063° | 0.037 v | 23.96 to 24.42 mm, 20 mm | 0.093° |
| a flat board on the table, the views meeting under its top, a lap of 600 | 839 120 | 0.0001° | 0.080 v | 5.92 to 6.14 mm, 5 mm | 0.054° |
| large box in a room's corner, half a lap of 599 | 2 599 996 | 0.0064° | 0.050 v | 23.85 to 24.46 mm, 20 mm | 0.093° |

Rolled by 15° to 60° the same, every walk shaken by 2 cm at 1.5 Hz within the same tolerances, its up
within 0.074° [M 2026-11-17: the same test]. Up from the cameras' axes, on walks of 12 and 24
poses, lay 0.01° to 1.75° from up as walked ("The roll").

Every box holds the known box and no point of another thing; its z axis is the plane's normal. The
excess beyond μ is the noise's farthest draw on the box's faces, up to 4.9 σ = 1.22 v. By a wall the
wall holds more of the window's points than the support plane (701 878 against 271 998 on the table,
734 345 against 502 496 on the floor), and the board's top more than the table seen around it
(679 120 against 160 000): the conditions, not the counts, keep the support plane.

## Time on the board

The proposal on a full model of 2 million points takes the time docs/HARDWARE.md, "Separation",
states.

## Limits

* A thing that touches the object, or stands within c of it, joins it; so does a thing above the
  object's footprint, a shelf or a lamp, since the footprint has no height limit. The box then holds
  both, and the operator shrinks it.
* Up is only as right as Q, the rotation from the IMU's axes to the camera's, which is the
  identity until the camera's live check measures it [?] (docs/methods/imu-samples.md, "The axes");
  Q wrong by up to 10° passes the spread and leans up by as much ("Gravity's up").
* A scan without accelerometer samples, or with fewer than five seconds of them between tracked
  frames, gets no proposal; the operator sets plane and box.
* The window stops at the median camera's distance from the meeting point: a part of the object
  farther than that, the ends of a long large object scanned from close by, lies outside the window
  and outside the box, which then stops short; the operator lengthens it.
* A large flat object whose top the views meet above may take its top as the plane.
* Each threshold is [?: initial]; real scans tune them (`separate-object-model`'s acceptance).

## What the prototype did

CUDA_3D's clustering was deleted in the cleanup (.scratch/rewrite/requirements.md, the prototype's
table), and the tree held no plane search or separation [S: grep of `src/` for RANSAC, a plane fit
and clustering, 2026-11-15]. Nothing is carried. distortion_check's measurement fitted its own plane
to a checkerboard's corners; it now calls `gca::fit_plane()`, the one copy.
