# Normals in image space

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The per-frame path's stage after the organized points: each valid pixel gets the normal of the
surface it lies on, from the points of its neighbours in the picture, by the eigenvector of
their covariance, centred first, and turned toward the camera. A pixel at a depth edge or with
too few neighbours on its surface gets no normal and is marked so. The code is
`src/geometry/organized_normals.*`, its test `tests/test_organized_normals.cpp`.

## Input

* **The organized points** (`gca::organized_points`, docs/methods/organized-points.md) of a
  frame, made on the stream the stage runs on or on one the caller has ordered before it: for
  each pixel (u, v) its point p(u, v) in the camera frame (x right, y down, z forward, in m) and
  its validity mark. A valid point's z is > 0; an invalid pixel's point is (0, 0, 0).
* **The parameters** (`gca::organized_normals_parameters`), each a starting value until the
  replay of sequences taken on the hardware measures it [?: initial]:

  | parameter | meaning | range | starting value |
  |---|---|---|---|
  | `radius_px`, R | the window's half size: (2R + 1)² pixels around the pixel | 1 to 3 | 3 |
  | `min_points` | the fewest points of the neighbourhood, the pixel's own included, that give a normal | 2R + 2 to (2R + 1)² | 25 |
  | `max_incidence_deg`, θ | the steepest incidence, the angle between a surface's normal and the line of sight, at which two neighbouring points still count as one surface | above 0, at most 89 | 75 |

  Why these values, and what each range rules out, stands below with the rule it belongs to.
  Each object class's set holds them, the same in both (`src/scan/object_class.cpp`), and a scan
  passes them to the pyramid through tracking (`src/scan/stage_parameters.hpp`); the stage's
  tests name their own.

## Output

Organized, like the points: one entry for each pixel (u, v) at index v · W + u, the frame's own
index.

* **The normal** n(u, v): a unit vector in the camera frame, perpendicular to the surface the
  pixel lies on, pointing toward the camera: n · p <= 0.
* **No normal: (0, 0, 0).** A unit vector is never zero, so the zero vector is the mark, as the
  zero point is an invalid pixel's. A pixel has no normal when it is invalid, when it lies at a
  depth edge, or when its neighbourhood holds too few points. A reader skips such a pixel: a
  point-to-plane residual taken along the zero vector is 0 whatever the distance, so the pixel
  would count as a perfect inlier.

The normals are one GPU image (`src/gpu/gpu_image.hpp`) of the frame's size with dense rows,
12 bytes a pixel: 3.1 MB at 640x400 and 12.3 MB at 1280x800 [S: computed]. It is requested from
the pool (`src/gpu/gpu_pool.hpp`) before it closes; the stage itself asks for no memory and
takes no pool.

## The neighbourhood

The window of a pixel c is the (2R + 1)² pixels (u, v) with |u - u_c| <= R and |v - v_c| <= R,
cut off at the picture's border. Its neighbourhood is the pixel itself and every valid pixel of
the window that lies on the same surface as c by the rule below. The neighbourhood is fixed in
pixels, so it spans R z / fx on each side in m: 4.9 mm at 0.5 m and 19.6 mm at 2 m at 640x400
(fx = 306.18 px), half that at 1280x800 (fx = 612.37 px) [S: computed; the intrinsics,
docs/HARDWARE.md].

### One surface, or a depth edge

Two points a and b lie on one surface unless the segment between them points nearly along the
line of sight:

    |d · m| <= sin θ · |d| · |m|,   d = b - a,   m = a + b,

where m points from the camera toward the pair's midpoint. On a plane seen at the incidence φ,
the angle between its normal and the line of sight, every segment of the plane makes an angle
of at least 90° - φ with the line of sight, and the steepest one, down the plane's slope, makes
exactly that [S: the projection of the line of sight onto the plane has length sin φ]. So the
rule keeps every pair on a plane seen at φ <= θ, and it takes a pair whose segment lies within
90° - θ of the line of sight for a jump in depth between two surfaces, one behind the other.
The rule is symmetric in a and b, and it needs neither the focal length nor the depth unit. θ is
at most 89° [?: a choice, the user's, 2026-11-14]: the stage compares in float with sin² θ,
0.99970 at 89°, 3.0e-4 below 1, far more than the rounding of the rule's products, a few float
steps of 1e-7; from 89.990° on sin² θ rounds to 1, and the rule would take every pair for one
surface, by the Cauchy-Schwarz inequality, a point and the camera's centre included
[S: computed]. Its
code, `src/geometry/one_surface.cuh`, is shared with the frame's pyramid, whose coarser levels
combine only points on one surface by it (docs/methods/frame-pyramid.md).

Across one pixel, near the picture's centre, the rule takes a step in depth larger than
(z / fx) tan θ for an edge: with θ = 75°, 6.1 mm at 0.5 m and 24.4 mm at 2 m at 640x400, 3.0 mm
and 12.2 mm at 1280x800 [S: computed]. Steps that small separate no surfaces of the scenes
scanned, and a surface seen at more than 75° gives few pixels and poor depth [?: initial].
The depth's rounding to whole units makes steps of one unit more or less: at 1 mm and
1280x800, a plane at 0.3 m seen at more than 64° has steps of 2 mm against a limit of 1.8 mm,
and some of its pixels are taken for an edge [S: computed, (0.3 m / 612.37) tan 64° = 1 mm].
How the camera's own depth noise trips the rule is unmeasured [?].

Organized-cloud normal estimation commonly leaves out neighbours across a depth change with a
threshold that grows with depth [S: S. Holzer, R. B. Rusu, M. Dixon, S. Gedikli, N. Navab,
"Adaptive Neighborhood Selection for Real-Time Surface Normal Estimation from Organized Point
Cloud Data Using Integral Images", IROS 2012]; the rule here states that threshold as an angle,
so that one value serves every depth and both stream sizes.

A crease, where two faces of one object meet, is no depth edge: the surface is continuous, and a
pixel within R pixels of a crease gets a normal between the two faces' normals.

### When a pixel gets no normal

A valid pixel c has no normal when either holds:

1. **It lies at a depth edge**: one of its 8 adjacent pixels is valid and not on its surface.
   Both pixels of such a pair have no normal, so every depth edge is marked by a band two
   pixels wide. A pixel farther from the edge leaves a pixel across it out of its neighbourhood
   only when the rule finds the two on different surfaces, and the segment to a pixel two or
   three columns away leans farther from the line of sight: across a step of 7 mm at 0.5 m at
   640x400 it lies 13° from the line of sight across one pixel, an edge, and 25° across two,
   one surface [S: computed, atan(1.63 mm / 7 mm), atan(3.27 mm / 7 mm)]. Beside a step that
   the rule just catches, a window's normal is therefore tilted by the step; beside a step of
   10 cm it is not [M 2026-11-14: `tests/test_organized_normals.cpp`, the flat normals beside
   it]. An adjacent pixel without depth or outside the range is not an edge: what lies there is
   unknown. The pixel keeps its window's pixels on its own side and gets a normal from that cut
   window when they are `min_points` or more. Beside a band without depth one pixel wide, as an
   occluding edge may leave in the camera's depth [?: unmeasured on this camera], it keeps 4 of
   the 7 columns, 28 points, as at the picture's border. A cut window's normal errs more than a
   whole one's: on the test's planes up to 4.90° against 2.29° at 640x400, and 10.6° against
   4.81° at 1280x800 [M 2026-11-14: `tests/test_organized_normals.cpp`; "Accuracy"].
2. **Its neighbourhood holds fewer than `min_points` points.**

`min_points` is at least 2R + 2: a window holds at most 2R + 1 pixels on one line of the
picture, so a neighbourhood that large holds three pixels not on one line. Their points are
then not on one line in space either: a line in space that misses the camera's centre projects
to a line in the picture, and one through it to a single pixel [S: the pinhole model]. So the
covariance below has rank 2 or 3: its two larger eigenvalues are above 0. A smaller
`min_points` is refused. That does not make the smallest eigenvalue a single one: points that
lie on no plane, a blob of noise, may spread as much in two directions, and then every direction
between the two eigenvectors is one. The stage tests no gap between the two smaller eigenvalues
and gives the vector the solver returns, turned toward the camera. On a surface the smallest
eigenvalue is the depth's rounding and noise, far below the window's spread, which grows as
R z / fx [S: computed]; how often the camera's frames hold a window of no plane is unmeasured
[?]. The starting value 25 is half the 7x7 window rounded up
[?: initial]: a pixel two or more pixels from a straight depth edge, or at the picture's border,
keeps at least 4 of the window's 7 columns, 28 pixels, and gets a normal; a pixel near a
convex corner of an object's outline keeps about a quarter of the window and gets none.

## The covariance, centred first

With the neighbourhood's n points p_j, the pixel's own p_c among them:

    p̄ = p_c + (1/n) Σ (p_j - p_c),
    C = (1/n) Σ (p_j - p̄)(p_j - p̄)ᵀ.

The first pass over the window finds the neighbourhood, keeps which pixels belong to it as one
bit each (49 at most, one 64-bit word, which bounds R at 3), and sums their offsets from p_c;
the second sums the outer products of the points less the mean. The arithmetic is float, and
centring makes double unnecessary here:

* A coordinate of p_j - p̄ is a few mm to a few cm, so the rounding of each product is relative
  to the window's spread, not to the distance. The rounding of C is then a few float steps
  (2^-24 = 6e-8) times n relative to the spread, against an in-plane eigenvalue of the order of
  the spread itself [S: float32 ulp; the error bound of a sum of n terms].
* p̄ itself is rounded, by half a float step of its coordinates, 1.2e-7 m at 2 m. A mean off by
  δ adds δ δᵀ, about 1.4e-14 m², to C [S: the sum of squares about a shifted centre], against an
  in-plane eigenvalue of 1.7e-4 m² for a full 7x7 window at 2 m at 640x400, 4 (z / fx)²
  [S: computed].
* An eigenvector moves by at most the perturbation of the matrix over the gap to the next
  eigenvalue [S: C. Davis and W. M. Kahan, "The rotation of eigenvectors by a perturbation.
  III", SIAM J. Numer. Anal. 7(1), 1970]. On a plane the gap is the window's in-plane spread.

The uncentred form, the mean of p pᵀ less p̄ p̄ᵀ, rounds the squares of whole coordinates, of the
order of |p|², and subtracts two such numbers to get the spread: its error relative to the
centred form's grows as |p|² over the spread, about (fx / R)², 10^4 at 640x400 and 4 · 10^4 at
1280x800 [S: computed]. That is the classic cancellation of the one-pass formula for a variance,
and centring the data first is its remedy [S: T. F. Chan, G. H. Golub, R. J. LeVeque,
"Algorithms for Computing the Sample Variance: Analysis and Recommendations", The American
Statistician 37(3), 1983].

For a window fixed in pixels that ratio does not depend on the distance: the window's spread
grows with the distance as |p| does. The uncentred form is as wrong at 0.5 m as at 2 m; only the
depth's rounding, which masks it, shrinks with the distance relative to the window
(see "Accuracy"). Measured [M 2026-11-14: `tests/test_organized_normals.cpp`]:

* The test computes the method in double from the same float points at every valid pixel of
  every picture it makes, apart from pixels whose window holds a pair within 1° of the depth
  edge rule's threshold, where float and double may decide the pair differently: 4627816
  pixels at both stream sizes. The same pixels have a normal, and the largest angle between the
  two normals is 7.1e-4°, on the sphere near the picture's side at 1280x800.
* With the covariance left uncentred (the mean of p pᵀ less p̄ p̄ᵀ, in float; a mutant of the
  code), the normals lie up to 1.40° from the double ones on the plane at 0.40 m to 0.80 m,
  1.52° on the plane at 2 m and 1.94° on the sphere, at 640x400. Against the analytic normal
  the plane at 0.40 m to 0.80 m still passes the test's tolerance, 5.61° against 6°, while the
  plane at 2 m fails it, 2.34° against 2°: the comparison in double catches the uncentred form
  at every distance.

## The eigenvector

The normal is the unit eigenvector of C's smallest eigenvalue: the direction in which the
neighbourhood's points spread least, the normal of the plane fitted to them by least squares
[S: H. Hoppe, T. DeRose, T. Duchamp, J. McDonald, W. Stuetzle, "Surface reconstruction from
unorganized points", SIGGRAPH 1992, the tangent plane from a neighbourhood's centroid and
covariance]. It is found by the tree's closed-form solver for a symmetric 3x3 matrix
(`src/geometry/symmetric_eigen_3x3.cuh`, after Eberly), which returns the eigenvalues in
ascending order except for a diagonal matrix, whose diagonal it returns in its order: the stage
takes the eigenvector of the least of the three values; choosing its column by comparison
instead of at run time showed no gain in time and was not kept (docs/HARDWARE.md). The test
holds a window whose covariance is exactly diagonal, x's and y's spread before z's 0, and gets
the flat normal there. The factor 1/n changes no eigenvector, so the code leaves it out.

## Orientation

The eigenvector has no sign. It is turned toward the camera, at the origin of the camera frame:
when n · p_c > 0 it is negated, so that n · p_c <= 0. A camera frame's surfaces are seen from
the front, so a normal toward the camera points out of the object. Toward the picture's side a
surface that leans outward has normals toward the camera with n_z > 0, which a turn toward
(0, 0, -1), the prototype's, would point away: the test's sphere near the picture's side has
88 of them at 640x400 and 434 at 1280x800 [M 2026-11-14: `tests/test_organized_normals.cpp`].

## Accuracy

On the synthetic source's frames the depth's rounding to whole units, not the arithmetic, sets
the error: a plane's points form steps of one unit, and a window that holds a few steps fits a
plane tilted by them; on a sphere the curvature across the window adds to it. With a depth unit
of 1 mm, the camera's intrinsics at each size and the starting parameters, the largest and the
median angle to the analytic normal, over all normals and over the windows that are not cut,
and the largest over cut windows [M 2026-11-14: `tests/test_organized_normals.cpp`]:

| surface | size | normals | all: largest / median | whole windows: largest / median | cut windows: largest |
|---|---|---|---|---|---|
| a plane seen at a slant, 0.40 m to 0.80 m | 640x400 | 81179 | 4.90° / 0.321° | 2.29° / 0.314° | 4.90° |
| | 1280x800 | 324501 | 10.6° / 0.775° | 4.81° / 0.770° | 10.6° |
| a plane 2 m away | 640x400 | 255980 | 1.49° / 0.118° | 0.846° / 0.117° | 1.49° |
| | 1280x800 | 1023980 | 4.53° / 0.250° | 1.78° / 0.249° | 4.53° |
| a sphere of 8 cm radius, its centre 0.46 m away, in the picture's middle | 640x400 | 8154 | 5.72° / 0.829° | 5.72° / 0.828° | 2.52° |
| | 1280x800 | 34651 | 15.2° / 1.20° | 15.2° / 1.20° | 1.84° |
| the same sphere near the picture's side | 640x400 | 13236 | 8.29° / 1.01° | 8.29° / 1.01° | 3.33° |
| | 1280x800 | 50889 | 25.0° / 2.12° | 25.0° / 2.12° | none |

On the sphere only the pixels whose window lies on it alone are compared. A window is cut at the
picture's border, at the range's limits, beside a depth edge or a pixel without depth; on the
planes its fit errs about twice as much as a whole window's. On the box's plane at 0.39 m a
pixel at the picture's left border, whose window keeps 4 of its 7 columns, is 10.8° off
[M 2026-11-14: `tests/test_organized_normals.cpp` at 74dda68, in the comparison with the oracle,
"What the prototype did, and what is not carried"].

At 1280x800 the errors are two to three times those at 640x400. A window of radius 3 spans half
the surface there, R z / fx with fx twice as large, over steps of the same height, so its fit
tilts more [S: the steps' height against the window's span]. The error would fall with a larger
window and with a finer depth unit; the window is bounded at radius 3 by the membership word
("The covariance, centred first"), so the stream size weighs this. The share of the camera's own
depth noise is unmeasured [?].

At 1280x800 the depth's rounding also trips the depth edge rule where a plane is seen near the
rule's 75°: the tilted plane's far corners are seen at up to 72.8° [S: computed from the pose],
and there 8 of its 324773 pixels lie at a depth edge, none at 640x400 [M 2026-11-14:
`tests/test_organized_normals.cpp`].

## On the GPU

One kernel on the caller's stream, one thread for each pixel, in blocks of 32 x 8 threads
[?: a choice, the user's, 2026-11-14]. Its first pass reads the points and the marks of the
pixel's window, its second only the points of the neighbourhood, and it writes the normal. The
stage allocates nothing, waits for nothing and reads nothing back: the host checks the arguments,
asks CUDA whether an
error is set (`cudaPeekAtLastError`), launches, and asks whether the launch was accepted
(`cudaGetLastError`), the per-frame path's stages' one launch (`src/gpu/kernel_launch.cuh`);
neither waits. Work queued on the stream
after it sees the normals. Its time on the board stands in docs/HARDWARE.md, "The per-frame
path's stages".

## Refusals

Each comes back as a value with its reason; nothing is queued and the normals are left as they
were:

* the parameters: `radius_px` not 1 to 3, `min_points` not 2R + 2 to (2R + 1)²,
  `max_incidence_deg` not a finite number above 0 and at most 89;
* the normals hold no image, or the organized points hold none;
* the organized points' size is not the normals'.

CUDA's errors come back as a failure with CUDA's error, which is then cleared
(`src/gpu/gpu_messages.hpp`): one an earlier call left as CUDA's last error, before the launch,
with nothing queued and the normals as they were; and CUDA's refusal of the launch, with the
normals' contents then unspecified.

## What the prototype did, and what is not carried

The prototype's normals of a point cloud, CUDA_3D's, were the oracle until this stage agreed with
them within a stated tolerance (below); they are deleted since, and their code stands in git at
74dda68 (`src/geometry/`). Their method, the eigenvector of a neighbourhood's covariance, carries
over; these do not:

* **The neighbourhood in space**: every point within a radius, found through a grid built anew
  at each call. Here the neighbourhood is the window in the picture, which the organized points
  give for free, with the depth edge rule in place of the radius.
* **The uncentred covariance**: sums of p and p pᵀ in float, then E[p pᵀ] - E[p] E[p]ᵀ
  [S: the prototype's normals at 74dda68, their sums and their normal]. On a tilted plane at 1 m
  with a radius of 2 cm it was 0.154° off where PCA in double over the same neighbours is
  0.0002° off [M 2026-11-13: the prototype's test of its normals, in git at 74dda68].
* **The orientation toward (0, 0, -1)**: a fixed direction, so that a surface seen at the side
  of the picture, whose normal is perpendicular to the optical axis, has no defined sign. Here
  the normal is turned toward the camera's centre.
* **Three neighbours at least**, which may lie on one line. Here `min_points` rules that out.
* **A new buffer every call**: the neighbour lists, the sums and the normals in Thrust's
  containers, sorted and reduced by key. Here the normals come from the pool before the scan
  and nothing is allocated.

Carried: a pixel without a normal holds the zero vector, as the prototype's normal of a point
with fewer than three neighbours.

**The comparison with the oracle** ran the prototype on the stage's own valid points, so both
saw the same points, and held away from depth edges and creases, where the two neighbourhoods
differ in kind: at pixels whose window lies on one surface of the scene and whose point has no
other surface within the oracle's radius. There both fit a plane to points of one face: on three
frames of a walk around a box, depth range 0.30 m to 1.0 m, with the oracle's radius of 1 cm,
383431 normals agreed with the oracle's as lines within 11.9°, median 0.509°
[M 2026-11-14: `tests/test_organized_normals.cpp` at 74dda68]. The largest lay at the picture's
border, where the stage's window is cut: there the stage's normal was 10.8° off the face's and
the oracle's, from a ball of 1 cm, 1.3° [M 2026-11-14]. This is the record of the agreement;
the prototype and the comparison are deleted.
