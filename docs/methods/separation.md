# Separation: the proposal of the support plane and the box

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

After refinement the scanner proposes the separation of the full model (R15; the spec,
"Separation"): the support plane, the dominant plane under the object, and a box that stands
upright on it, one axis along the plane's normal, around what stands on the plane where the views
of the trajectory meet. The operator changes plane and box on the page where needed and confirms;
the cut then keeps the full model's points inside the box and above the plane by one full voxel
(`separate-object-model`). The proposal reads the full model and its trajectory and changes
neither (R14, R16).

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
knows which way is up, here the trajectory tells it. This tree's code is written from these
sources; no reconstruction library's code is read or copied for it (docs/adr/0002).

Built: the proposal, on the host in double precision (`src/separation/proposal.*`), and the
orthogonal least-squares plane (`src/separation/plane_fit.*`), which distortion_check's measurement
shares; their test `tests/test_separation.cpp`. Left to `separate-object-model`: the cut, the plane
and box in the scan's parameters, the proposal called at refinement's end.

## Inputs and output

* **The full model**: its points' positions in the models' frame, the camera frame of the first
  fused frame (docs/SCAN_FILES.md), in m. Normals and colors are not read: the fused model's normals
  are off by a median of 6.6° to 6.8° at the final voxel sizes (docs/methods/sparse-tsdf.md, "What
  the extraction gives").
* **The trajectory**: the camera's pose at each tracked frame, camera to model (R_i, c_i), its axes
  x_i (right), y_i (down) and z_i (forward, the optical axis) the columns of R_i. Frames in tracking
  loss have no pose and are left out by the caller.
* **The parameters**, both from the scan's object class (`src/scan/object_class.cpp`): v, the voxel
  size the full model was fused at (`full_voxel_size_m`), which scales every distance below; and D,
  the farthest depth a frame contributes (`max_depth_m`), which bounds how far from the views the
  proposal looks.
* **The output**, a value for every call: how it ended, the reason in plain English when it did not
  propose, and with a proposal the plane and the box; and the evidence found on the way, the
  trajectory's up, the views' meeting point, the window, the plane's points and the object's
  points.

**The plane** is n · p = d: n the unit normal, pointing to the side the object stands on, d in m.
A point's height above it is h(p) = n · p - d.

**The box** has a pose, box frame to models' frame, and three edge lengths L_x >= L_y and H. Its
frame's origin is the centre of its bottom face, which lies in the plane; its z axis is n; its x axis
lies along the box's longer side in the plane, of either sign. A point q of the box frame is inside when
|q_x| <= L_x / 2, |q_y| <= L_y / 2 and 0 <= q_z <= H.

## The trajectory's up

The models' frame is the first camera's, so the support plane lies at any tilt in it and no axis of
it is up. **The camera is hand-held and held roughly upright** [?: a choice, the orchestrator's,
2026-11-15; the user decided the same day that up comes from the camera's IMU gravity, and
`separation-up-from-gravity` replaces this assumption]: its x axis lies level up to its roll, at any pitch, and its y
axis points down, leaning back by the pitch θ, y = -cos θ up - sin θ f, f the level direction it
looks along [S: computed]. So the mean of the cameras' up axes,

    ū = -(1/N) Σ y_i,

leans from up toward the mean view direction by β, tan β = tan θ |mean f|: 0 on a whole lap, 24°
on half a lap at θ = 35°, 42° at θ = 55° (|mean f| = 2/π on half a lap) [S: computed]. The level x
axes do not lean with the pitch. Up is taken as the direction most nearly perpendicular to every x_i
that stays near ū: the u that minimizes

    Σ (x_i · u)² / N + γ |u - ū|²,   that is   u ∝ (M + γ I)^-1 ū,   M = (1/N) Σ x_i x_i^T,

normalized [S: computed, the normal equations of the two quadratic terms]. On a lap or half a lap
the x_i span the level plane: M's level eigenvalues lie near 1/2 and its vertical one is the mean of
sin² of the roll, about 0.015 at a roll swinging by 10° [S: computed]; the lean along a level
direction shrinks by γ / (λ + γ) against the vertical's γ / (λ_v + γ), to about a twentieth with
γ = 0.01 [?: initial]. A short arc gives M little weight along the view, and u stays near ū there.
On the test's half lap looking down by 60°, ū leans 50.2° [S: computed, a scratch copy of the walk]
and u 1.75° [M 2026-11-15: `tests/test_separation.cpp`]; with ū taken for up, a mutant, the test
fails there. A mean of the up axes shorter than 0.25 [?: initial: the camera pitched down by 75.5° on
average, cos 75.5° = 0.25] gives no up, and the proposal ends with the views not meeting.

**The roll.** A camera held rolled tilts every x_i out of the level plane the same way. On a whole
lap the tilts cancel and u stays up; on half a lap u absorbs them and leans sideways by about the
roll [M 2026-11-15: `tests/test_separation.cpp`, every camera of a scene's walk rolled about its
optical axis]:

| scene, walk | u from up, rolled by 15°, 30°, 40°, 45°, 60° | the proposal |
|---|---|---|
| tabletop, a lap | 0.02°, 0.05°, 0.09°, 0.11°, 0.20° | right at every roll |
| large, a lap | 0.09°, 0.12°, 0.16°, 0.19°, 0.39° | right at every roll |
| tabletop by a wall, half a lap looking down by 60° | 9.62°, 18.4°, 23.9°, 26.4°, 33.4° | right at every roll |
| large by a wall, half a lap | 17.1°, 33.3°, 43.4°, 48.1°, 60.5° | right up to 40°; no plane at 45° and 60° (the band's spread) |
| large in a room's corner, half a lap, rolled by ±30°, ±45°, ±60° | 33.3° and 33.7°, 48.1° and 48.7°, 60.5° and 61.0° | right at ±30°; at ±45° and ±60° a side wall, 90° off |

**Up to 30° the proposal is right on every scene tested.** Beyond, a whole lap stays right. Half a
lap may end with no plane, where the floor leaves the 45° cone and only a plane cutting across it at
a slant is left, which the band's spread refuses ("No plane"); or, in a room's corner, u leans so
far toward a side wall that the wall passes both conditions and is proposed as the support plane.
The operator then sets plane and box by hand. The way to know up is the camera's IMU, which the spec
leaves unused (A9). A refusal of the proposal beyond a measured roll was tried and taken out: no
measure of the roll kept every right proposal [?: the orchestrator's choice, 2026-11-15, for the
user]. Measured against u, the roll does not show, since half a lap absorbs it. Measured against the
plane the camera centres lie in, it refused upright cameras whose centres are not level: rows along a
wall read 53.4°, a straight sweep bobbing by ±5 cm 36.9°, and a rise of 10 cm in a walk moved it by
5.4° [S: computed, the review's scratch programs].

## The views' meeting point

The operator walks around the object and points the camera at it, so the optical axes pass near
it. The meeting point m is the point nearest to all axes in the least-squares sense:

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

1. its normal within 45° of the trajectory's up, n · u >= cos 45° [?: initial: above the roll and
   the lean left in u, far below the 90° of a wall];
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
* **the views do not meet**: the axes are near parallel, or the mean of the up axes is shorter than
  0.25.
* **refused**, before anything is computed: v or D not finite and > 0, or D above 10^5 v; no pose;
  a camera position not finite or a quaternion whose norm differs from 1 by more than 1e-9 (the
  trajectory's own check, docs/SCAN_FILES.md); a point not finite.

## Constants and parameters

The parameters (`gca::separation_parameters`): v and D, each the scan's object class's. The
constants, each [?: initial] until real scans measure them:

| constant | meaning | value |
|---|---|---|
| γ | the weight of ū against the level x axes | 0.01 |
| | the least length of the mean of the up axes | 0.25 |
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
camera. The plane's angle from the known one, its distance from it at the known box's foot, the box's
faces' excess over the known box, the trajectory's up from the known up [M 2026-11-15]:

| scene | points | plane | distance | excess, margin μ | up |
|---|---|---|---|---|---|
| tabletop box among four things (one larger in points, one 3 cm beside it), a lap of 12 | 960 908 | 0.0020° | 0.017 v | 5.88 to 6.16 mm, 5 mm | 0.01° |
| large box among three things (one larger in points), a lap of 24 | 2 048 582 | 0.0018° | 0.017 v | 24.0 to 24.9 mm, 20 mm | 0.08° |
| tabletop box by a wall, half a lap of 11, looking down by 60° | 1 567 998 | 0.011° | 0.027 v | 5.90 to 5.97 mm, 5 mm | 1.75° |
| large box by a wall, half a lap of 11 | 1 624 996 | 0.0063° | 0.036 v | 23.96 to 24.42 mm, 20 mm | 0.58° |
| a flat board on the table, the views meeting under its top, a lap of 12 | 839 120 | 0.0001° | 0.080 v | 5.92 to 6.14 mm, 5 mm | 0.01° |
| large box in a room's corner, half a lap of 11 | 2 599 996 | 0.0064° | 0.050 v | 23.85 to 24.46 mm, 20 mm | 0.58° |

Every box holds the known box and no point of another thing; its z axis is the plane's normal. The
excess beyond μ is the noise's farthest draw on the box's faces, up to 4.9 σ = 1.22 v. By a wall the
wall holds more of the window's points than the support plane (701 504 against 271 998 on the table,
733 037 against 502 496 on the floor), and the board's top more than the table seen around it
(679 120 against 160 000): the conditions, not the counts, keep the support plane.

## Time on the board

The proposal on a full model of 2 million points takes the time docs/HARDWARE.md, "Separation",
states.

## Limits

* A thing that touches the object, or stands within c of it, joins it; so does a thing above the
  object's footprint, a shelf or a lamp, since the footprint has no height limit. The box then holds
  both, and the operator shrinks it.
* A camera held rolled by more than 30° on half a lap can leave no plane, or in a room's corner
  propose a side wall ("The roll"); the operator corrects it by hand.
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
