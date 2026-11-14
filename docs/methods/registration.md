# The registration of a frame against a surface

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The per-frame path's step after the ray cast: the transform that carries a frame onto the model's
surface as the last pose sees it, with the evidence of how well it is known. The method is colored
ICP [S: J. Park, Q.-Y. Zhou, V. Koltun, "Colored Point Cloud Registration Revisited", ICCV 2017],
whose two residuals, a point-to-plane term and a color term, carry over from the prototype; here it
runs on organized pictures, with the association by projection of KinectFusion [S: R. A. Newcombe et
al., "KinectFusion: Real-time dense surface mapping and tracking", ISMAR 2011, "Sensor pose
estimation"], coarse to fine over the frame's pyramid (docs/methods/frame-pyramid.md). This tree's
code is written from the papers named here; no reconstruction library's code is read or copied for
it (docs/adr/0002-own-reconstruction-code.md).

Built so far: the point-to-plane term on one level, with the numerics and the result of this
document (`src/registration/registration.*`, the solve of one iteration's system
`src/registration/normal_equations.*`, the SE(3) exponential `src/registration/se3.*`; the test
`tests/test_registration.cpp`). The color term and the levels are this document's method too, not
built yet; "The color term" and "Coarse to fine" say what is left to settle there.

## Input

* **The frame**: one level of a frame's pyramid (`gca::frame_pyramid`): for each pixel a point p
  in the frame's camera frame F (x right, y down, z forward, in m) with its validity mark, a unit
  normal m toward the camera or (0, 0, 0) for none, and, for the color term, an intensity I_p in
  [0, 1].
* **The surface**: organized points with normals, in the surface's own camera frame S, seen by a
  pinhole camera whose intrinsics K_S = (fx, fy, cx, cy) in px describe its picture by the one
  pinhole convention (`src/types/camera_param.hpp`): for each pixel a point q with its mark and a
  normal n or (0, 0, 0); for the color term an intensity I_q. It comes from a ray cast of the model
  at a pose (docs/methods/sparse-tsdf.md, "The ray cast"), or, in the tests, from a level of another
  frame's pyramid. The registration reads it through a view of its images and its intrinsics,
  `gca::organized_surface`; it does not take `gca::organized_points`, whose two writers stay the
  deprojection of a frame and the pyramid's combining (docs/methods/organized-points.md), so that a
  ray cast fills images of its own.
* **The initial transform** T_0 = (R_0, t_0), F to S: a point p of the frame lies at R_0 p + t_0 in
  the surface's frame. Tracking passes the identity when the surface is ray cast from the last
  pose.
* **The parameters** ("Parameters").

## Output

The result is a value (`gca::registration_result`) for every call, a refusal included:

* **The transform** T, F to S, in double: the last estimate. It is a pose only when the
  registration converged; tracking gives the frame the pose P_S T, P_S the surface's pose (camera
  to model).
* **How it ended**: converged; the iterations used up; too few correspondences; degenerate; refused
  (the inputs, with the reason); failed (CUDA's error, with its message). Only "converged" says that
  T is the minimum of the method's objective near T_0 and that the last system holds every motion;
  every other end is tracking loss, never a pose (GLOSSARY.md, "Tracking").
* **The fitness**: the share of the frame's candidates, its pixels with a valid point and a normal,
  that found a correspondence ("The association by projection"), 0 to 1; 0 when the frame has no
  candidate.
* **The inlier RMSE**: the root mean square of the correspondences' point-to-plane residuals, in m,
  unweighted; 0 when there is none.
* **The iterations**: the systems built, 0 for a refusal.
* **The conditioning**: the six eigenvalues of the last system, centred, scaled and normalized
  ("The system, centred and scaled"), ascending, each from 0 to 2; zeros when no system was
  solved, for too few correspondences or none.
  A small eigenvalue names a motion the surface does not hold: three near 0 for a single plane,
  its two shifts within itself and its turn about its normal.

The fitness, the RMSE and the conditioning are those of the last system built, at the transform
before the last step; a converged step moved the correspondences by less than the convergence's
thresholds, far below a depth unit ("The iterations and the end").

## The association by projection

Each iteration pairs the frame's pixels with the surface's at the current transform T = (R, t), a
correspondence for each candidate at most. For a candidate pixel with point p and normal m:

    s = R p + t,   u = fx s_x / s_z + cx,   v = fy s_y / s_z + cy,

and its partner is the surface's pixel nearest to (u, v), (floor(u + 1/2), floor(v + 1/2)), pixel
centres at whole numbers. The pair is a correspondence when all of these hold:

* s_z > 0, and the partner lies inside the surface's picture;
* the partner is valid and has a normal n;
* the distance |s - q| <= d_max, the largest distance;
* the angle between n and R m is at most θ_n, the largest angle between normals.

This is projective data association [S: Newcombe et al. 2011, "Sensor pose estimation", with its
thresholds on the distance and on the angle between normals; S. Rusinkiewicz, M. Levoy, "Efficient
Variants of the ICP Algorithm", 3DIM 2001, matching by projection]. It costs one lookup per pixel,
needs no index of the surface, and is right while the frame moves little between the surface's
pose and its own, which a hand-held camera at 30 fps does; the pyramid widens what one pixel
covers ("Coarse to fine"). A frame's point paired across a depth edge of the surface lies far from
its partner, beyond d_max, or faces another way, beyond θ_n, and is left out.

## The residuals and their Jacobians

The objective at fixed correspondences is a sum over them,

    E = λ Σ ρ_G(r_G) + (1 - λ) Σ ρ_C(r_C),

ρ the robust loss of each term ("The robust weight"), λ = 0.968 [S: the prototype,
`src/registration/color_icp.hpp`, "0.968 from paper"; the paper's value is unchecked here [?]].
λ weighs a residual in m against one in intensity units, [0, 1]; the prototype takes both units as
here. Built so far: the point-to-plane term alone, λ = 1.

**The update** is a small motion of the surface's frame applied on the left, T ← exp(ξ^) T, with
ξ = (ω, υ) in R^6, ω the rotation vector in rad and υ the translation in m ("The update"). To first
order it moves s = T p to s + ω × s + υ, so that

    ∂s / ∂ξ = [ -[s]×   I ],

[s]× the matrix of the cross product with s. Park et al. linearize the same motion, the source
moved in the target's frame [S: Park et al. 2017, the linearized T in their Gauss-Newton step]; the
prototype's Jacobians below are of this form and correct [S: the audit of CUDA_3D, 2026-11-13].

### The point-to-plane term

    r_G = n · (s - q),

the distance of s from the surface's tangent plane at q [S: Y. Chen, G. Medioni, "Object Modelling
by Registration of Multiple Range Images", Image and Vision Computing 10(3), 1992]. With
n · (ω × s) = ω · (s × n):

    J_G = ∂r_G / ∂ξ = [ (s × n)^T   n^T ].

### The color term

The surface's intensity near q, on q's tangent plane, is C_q(x) = I_q + d_q · (x - q), with d_q
the intensity's gradient in that plane, d_q · n = 0 [S: Park et al. 2017, the continuous
approximation of the color on the tangent plane and its gradient]. The frame's point is projected
onto that plane, f(s) = s - (n · (s - q)) n, and the term compares the intensities:

    r_C = C_q(f(s)) - I_p = I_q + d_q · (s - q) - I_p,

the second form because d_q · n = 0. With (I - n n^T) d_q = d_q and d_q · (ω × s) = ω · (s × d_q):

    J_C = ∂r_C / ∂ξ = [ (s × d_q)^T   d_q^T ].

The prototype writes J_C with (I - n n^T) d_q in place of d_q, the same vector for a gradient in
the plane [S: `cuda_color_icp_build_least_square.cu`].

**The gradient d_q** is estimated once per surface and level, not per iteration: for each surface
pixel with a normal, by least squares over the pixels of its window that lie on its surface (the
normals' window and rule of one surface, docs/methods/normals.md), each neighbour q' contributing
I_q' - I_q ≈ d_q · (q' - q) in the tangent plane. With a unit basis a, b of the plane, d_q = α a + β
b and (α, β) solve the 2 x 2 normal equations of Σ (I_q' - I_q - α a · (q' - q) - β b · (q' - q))²;
fewer than three neighbours, or a singular 2 x 2 system, give no gradient and the pixel no color
residual [?: initial; Park et al. fit the same constrained gradient over a point's neighbours in
3D]. The window's size is the color term's to measure.

**The intensity** is the frame pyramid's: the Rec. 709 luma of the color, scaled to [0, 1]
(docs/methods/frame-pyramid.md, "The intensity"), the prototype's; for a ray-cast surface the same
luma of its colors. Park et al.'s intensity is the mean (R + G + B) / 3 [?: unchecked here]; it
replaces the luma only on a measured benefit to the registration.

## The robust weight

Each residual carries Huber's weight [S: P. J. Huber, "Robust Estimation of a Location Parameter",
Annals of Mathematical Statistics 35(1), 1964]: its loss is r²/2 up to a threshold δ and grows
linearly beyond, so that a residual far out pulls with a constant force instead of one that grows
with it. It is minimized by iteratively reweighted least squares [S: P. W. Holland, R. E. Welsch,
"Robust regression using iteratively reweighted least-squares", Communications in Statistics -
Theory and Methods 6(9), 1977]: each iteration weighs its residuals by

    w = 1 for |r| <= δ,   w = δ / |r| beyond,

held fixed while it solves. Huber's loss is convex, so the weight adds no minimum of its own; the
association's thresholds still cut the gross outliers, a point across a depth edge or of something
the surface lacks. The point-to-plane term's δ_G is in m ("Parameters"); the color term gets its own
δ_C in intensity units.

## The system, centred and scaled

Each iteration builds, on the GPU, the weighted normal equations of its correspondences [S: the
Gauss-Newton step of a least-squares problem], in a frame centred on a point c:

    H = Σ w J J^T,   g = Σ w J r,   J = [ (s - c) × n ;  n ],

which is the Jacobian of r_G for a rotation about c instead of S's origin. With them come the sums
W = Σ w, Σ w (s - c), Σ w |s - c|², Σ r² and the counts of correspondences and of candidates:
21 + 6 + 8 = 35 numbers, the system and its sums, which is all an iteration reads back (280 bytes).

**Why centred.** About S's origin, the camera's centre, the rotation's columns s × n grow with the
distance of the points while the translation's do not, and rotation and translation mix: a turn
about the camera is nearly a shift of points 2 m away. The eigenvalues of such a system say more
about the distance than about the surface's shape. About the correspondences' centroid, and with
the rotation measured in units of their spread, the six columns are comparable and the eigenvalues
name what the surface holds [S: N. Gelfand, L. Ikemoto, S. Rusinkiewicz, M. Levoy, "Geometrically
Stable Sampling for the ICP Algorithm", 3DIM 2003, the covariance of the point-to-plane
constraints, its points centred and scaled before its eigenvalues are read].

**Centred first.** The kernel centres each point on c before it forms its products: c is the
centroid of the last iteration's correspondences, and S's origin at the first iteration, which has
none before it. The host then shifts the system the remaining way, d = Σ w (s - c) / W, exactly:
with A = [ I  -[d]× ; 0  I ],

    H' = A H A^T,   g' = A g,   ρ² = Σ w |s - c|² / W - |d|²,

since (s - c - d) × n = (s - c) × n - d × n. ρ is the correspondences' RMS distance from their
centroid c' = c + d, in m. From the second iteration on d is the centroid's move in one step, a
few mm; at the first it is the whole centroid, and the shift loses up to (|c'| / ρ)² of the
products' precision ("Precision"). Then the rotation is scaled by ρ and the system normalized by
W:

    Â = S H' S / W,   b̂ = S g' / W,   S = diag(I / ρ, I).

The translation's block of Â is Σ w n n^T / W, of trace 1, and the rotation's at most
Σ w |s - c'|² / (ρ² W) = 1, so each eigenvalue of Â lies in [0, 2] whatever the units and the
distance. For a single plane the shifts within it and the turn about its normal give 0; a sphere's
three turns about its centre give 0, which a scale taken from the rotation's own block would hide.
ρ² <= 0, no spread at all, is degenerate.

### Precision

* **FP64 is slow here**: the GPU runs double-precision arithmetic at 1/32 of the rate of float
  [M 2026-11-14: `cudaGetDeviceProperties`, singleToDoublePrecisionPerfRatio 32, Orin, compute
  capability 8.7, 8 multiprocessors]. A frame of 1280x800 has up to 1024000 candidates, each with
  35 products and the additions that sum them.
* **So the products are float and the sums double above the warp**: each thread forms its pixel's
  35 terms in float from the float points and normals; each warp sums its 32 threads' terms in
  float, by five steps of a shuffle tree, an error of at most 5 u Σ|x| for u = 2^-24 [S: N. J.
  Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002, chapter 4, pairwise
  summation], 3.0e-7 of the sum of the terms' sizes; every sum above the warp, of up to 32000 warps
  at 1280x800, is double, in a fixed order, so that the same input gives the same sums bit for bit.
  This departs from double accumulation within the warp's 32 terms [?: a choice, on the measured
  benefit: double within the warp is 11 to 13 times as slow], on the measured cost of the
  alternative: one iteration takes, by the host's call that waits for its sums, on the tests'
  tabletop object (the pair of "Accuracy"), the medians of 20 s after 5 s [M 2026-11-14, 19:54 to
  19:58, MAXN_SUPER by `nvpmodel -q`, no compiler running, a scratch program]:

  | each warp adds in | 640x400, 30 fps | back to back | 1280x800, 30 fps | back to back |
  |---|---|---|---|---|
  | float, then double above the warp | 1.649 ms | 0.355 ms | 4.348 ms | 1.209 ms |
  | double: each term converted before any addition | 13.718 ms | 3.988 ms | 22.813 ms | 15.712 ms |

  Double within the warp costs 11 to 13 times as much back to back; at 1280x800 it would leave
  about two iterations a frame. The warps' sums become double only after the shuffles, in the
  block's sum, by 35 threads: a conversion runs on the FP64 unit and costs a whole warp's
  instruction even for one lane, and converting each warp's 35 sums in its own lane 0 made an
  iteration take 1.49 ms and 4.34 ms back to back [M 2026-11-14, the same program, 1 s per
  case], for the same sums bit for bit, a float converting to double exactly [S: IEEE 754].
* **The shift's cancellation** multiplies a relative error by up to (|c'| / ρ)²: 400 for points
  2 m away spread over 0.1 m [S: computed], so 1.2e-4 for the float part at the first iteration and
  4e-14 for double's 1.1e-16; later iterations centre in the kernel, d is mm, and the factor is near
  1. The step a system gives is corrected by the next iteration, so its errors slow the iteration at
  most; what a converged transform carries is the last system's g, the residuals' balance, in which
  the float part's error is at most 3.0e-7 of Σ|w J r|.

## The degeneracy test and the damped solve

On the host, in double, Â is decomposed into its eigenvalues λ_1 <= ... <= λ_6 and orthonormal
eigenvectors v_k (Eigen's self-adjoint eigensolver, `Eigen/Eigenvalues`, MPL2):

* **The damped step**, at every iteration, Levenberg's damping of the Gauss-Newton step [S: K.
  Levenberg, "A Method for the Solution of Certain Non-Linear Problems in Least Squares", Quarterly
  of Applied Mathematics 2(2), 1944]:

      ξ̂ = -(Â + μ I)^-1 b̂ = -Σ_k (v_k · b̂) / (λ_k + μ) v_k,

  then ω = ξ̂_rot / ρ in rad and υ_c = ξ̂_trans in m. Â + μ I is positive definite for μ > 0, so the
  solve is defined for every system, and the step along v_k is the Gauss-Newton step times
  λ_k / (λ_k + μ): at λ_k >= τ and μ = τ / 10 at least 0.91 of it, and along a motion the surface
  does not hold, λ_k near 0, at most |v_k · b̂| / μ. A fixed point of the iteration, b̂ = 0, is the
  same with and without the damping; the damping only shortens steps along weakly held motions.
* **Degenerate** when the last system's λ_1 < τ: the surface holds some motion too weakly to fix
  it, and the registration ends as degenerate, whatever its steps did. The eigenvalues of the
  approximated Hessian against a threshold are the test of degeneracy of Zhang et al. [S: J. Zhang,
  M. Kaess, S. Singh, "On Degeneracy of Optimization-based State Estimation Problems", ICRA 2016],
  here on the centred, scaled and normalized system, so that one τ holds at every distance and
  size. The last system decides, not each one: at a start far from the minimum, residuals that
  hold the weakest motion are cut by d_max or weighed down by Huber's weight [?: not measured
  apart], and a first system can look degenerate where the surface is not; the large object of
  the tests, from 2.7 m, gave λ_1 = 1.55e-4 at its first iteration from a turn of 1.5° and 6.5e-3
  aligned [M 2026-11-14: `tests/test_registration.cpp`, the box without its sphere].

## The update: the SE(3) exponential

The step (ω, υ_c) is a twist about c', applied as

    ΔT = Tr(c') exp((ω, υ_c)^) Tr(-c'),   T ← ΔT T,

Tr(c) the translation by c. For θ = |ω| and W_ω = [ω]×, the exponential is

    exp((ω, υ)^) = [ R  V υ ; 0  1 ],   R = I + A W_ω + B W_ω²,   V = I + B W_ω + C W_ω²,
    A = sin θ / θ,   B = (1 - cos θ) / θ²,   C = (θ - sin θ) / θ³,

[S: R. M. Murray, Z. Li, S. S. Sastry, *A Mathematical Introduction to Robotic Manipulation*, 1994,
chapter 2, the exponential of a twist; T. D. Barfoot, *State Estimation for Robotics*, 2017, chapter
7]. Below θ = 1e-4 rad the coefficients are their series, A = 1 - θ²/6, B = 1/2 - θ²/24,
C = 1/6 - θ²/120, whose next terms, below θ⁴ / 120 = 8.3e-19, lie under double's rounding of 1
[S: computed]; above it the closed forms lose to cancellation at most u_d / θ² of a coefficient
that multiplies a term of size θ², an absolute 1.1e-16 [S: computed]. The result is a rotation to
double's rounding at any θ, unlike the prototype's Euler angles, which were a first-order update
only. It is the one copy of the exponential, `src/registration/se3.*`.

## The iterations and the end

The kernel takes T in float, its rotation's and translation's rounding below 1e-7 of a point
[S: float32 ulp]; the host keeps T in double. Each iteration:

1. builds the system and its sums at T on the GPU and reads them back, the one wait;
2. ends as **too few correspondences** when the fitness is below its least value, f_min, or no
   correspondence was found;
3. centres, scales and normalizes the system and decomposes it; ends as **degenerate** when it
   cannot step, ρ² <= 0 or a system or step that is not a number;
4. takes the damped step and applies it;
5. ends when the step turned by |ω| < ε_ω and moved the centroid by |υ_c| < ε_υ: as **converged**
   when this last system's λ_1 >= τ, as **degenerate** when not.

After the most iterations without one of these ends, it ends as **degenerate** when the last
system's λ_1 < τ, else as **the iterations used up**, with the transform after the last step. The
prototype's convergence, a step under 0.081° and 1 mm, was not checked before the pose was used;
here every end but "converged" is tracking loss.

## Coarse to fine

The frame's pyramid has three levels, each with its intrinsics K / 2^l
(docs/methods/frame-pyramid.md). The registration runs from level 2 to level 0, each level from the
transform the coarser one ended with; a transform is metric, so it carries over unchanged, and a
pixel of level 2 covers four of the frame's, so that the association there catches four times the
motion. KinectFusion tracks so over three levels [S: Newcombe et al. 2011, "Sensor pose
estimation"]. Left open until the levels are built, with the color term: the iterations of each
level, d_max per level, what a coarse level that does not converge passes on, and the surface's
coarser levels for a ray cast.

## Parameters

The registration takes, for each call (`gca::registration_parameters`), the values that differ
between the object classes or the levels; the tests name them, and the object class takes them
over once tracking uses them. Starting values [?: initial], and the range outside which a value is
refused as a mistake [?: a choice]:

| parameter | meaning | tabletop object | large object | refused outside |
|---|---|---|---|---|
| d_max | the association's largest distance, m | 0.05 | 0.10 | (0, 0.5] |
| δ_G | the point-to-plane term's Huber threshold, m | 0.005 | 0.030 | (0, d_max] |
| iterations | the most iterations at one level | 20 | 20 | 1 to 100 |

* **d_max**: 5 cm is the prototype's correspondence distance [S: the old loop, CUDA_3D's
  `apps/app.cpp`]; twice that for the large object, whose depth noise by the bound below is 6.8
  times as large at the middle of its range [S: computed].
* **δ_G**: about the datasheet's bound on the depth noise, σ = c z² with c = 0.01 /m
  (docs/methods/sparse-tsdf.md, "The truncation distance"), at the middle of each class's depth
  range: 4.2 mm at 0.65 m and 29 mm at 1.7 m [S: computed].
* **iterations**: 20 at one level, until the levels' counts are measured.

The values that are the same in every call are the registration's constants, θ_n in
`gca::registration` and the others beside the solve of a system
(`src/registration/normal_equations.hpp`), each [?: initial] until measured on sequences:

| constant | meaning | value |
|---|---|---|
| θ_n | the association's largest angle between normals | 30° |
| f_min | the least fitness of a system that is solved | 0.1 |
| τ | the least eigenvalue of a last system that is not degenerate | 0.01 |
| μ | the damping | τ / 10 = 0.001 |
| ε_ω, ε_υ | the step under which the iterations end | 1e-5 rad, 1e-5 m |

τ is dimensionless, a share of the system's information: 0.01 of it along the weakest motion. A
plane's three free motions hold what the noise of its estimated normals puts there, at most
4.7e-4 on the tests' planes at level 0 ("Accuracy"); a frame's noisier normals raise that floor.
Between the floor and τ the result is the measure's and not the noise's, and there the tests
measured both kinds, each ending degenerate with its steps stopped where the error is taken: the
shared tabletop box alone on its plane, λ_1 = 7.5e-4 to 1.1e-3 at its end, 0.84 mm and 8.7 mm off
Δ; the tabletop box with a sphere beside it seen from 2 m, λ_1 = 3.0e-3, 3.1 mm off; the same box
and sphere from 0.4 m to 0.5 m, λ_1 = 1.8e-3 to 7.1e-3, within 0.11 mm; the large box from 2.7 m
without its sphere, 6.5e-3, 0.12 mm [M 2026-11-14:
`tests/test_registration.cpp` and scenes tried while writing it]. The error of a weakly held motion
is a bias of the system divided by its eigenvalue, and the bias, from the association at depth
edges and the normals at creases, is the scene's; so no τ separates the cases cleanly, and 0.01
is the conservative choice: it rejects every case that ended more than 0.15 mm off and some that
did not, since a false pose doubles the model's surfaces where tracking loss only waits for
relocalization. The color term adds the texture's information to the same system.

ε_υ = 10 µm and ε_ω = 1e-5 rad, 10 µm at 1 m, lie far below a depth unit of 1 mm. Tracking's check
of a result, its thresholds of fitness, RMSE and conditioning, is stricter and its own.

## Accuracy

On pairs of the synthetic source's frames whose transform Δ is known, level 0 against level 0, from
the identity, depth unit 1 mm, the camera's intrinsics at each stream size, the normals' starting
parameters and the object classes' starting values [M 2026-11-14: `tests/test_registration.cpp`].
The tabletop object is a box of 0.30 m x 0.20 m x 0.25 m and a sphere of 0.10 m beside it on a
plane, seen from 0.5 m, moved by 1.5° and 10.8 mm; the large object a box of 1.2 m x 0.8 m x 1.0 m
and a sphere of 0.4 m on a floor, seen from 2.7 m and from 2 m, moved by 0.5° and 14.1 mm. The
error is the turn and the shift of Δ^-1 T, the shift at the camera's centre:

| case | size | iterations | error | fitness | inlier RMSE | λ_1 |
|---|---|---|---|---|---|---|
| tabletop object, 0.5 m | 640x400 | 4 | 0.0024°, 0.014 mm | 0.988 | 0.29 mm | 0.029 |
| | 1280x800 | 4 | 0.0025°, 0.031 mm | 0.982 | 0.29 mm | 0.031 |
| large object, 2.7 m | 640x400 | 4 | 0.0034°, 0.142 mm | 0.967 | 0.33 mm | 0.017 |
| | 1280x800 | 4 | 0.0023°, 0.083 mm | 0.967 | 0.31 mm | 0.017 |
| large object, 2 m | 640x400 | 4 | 0.0029°, 0.069 mm | 0.967 | 0.33 mm | 0.035 |
| | 1280x800 | 4 | 0.0007°, 0.018 mm | 0.968 | 0.31 mm | 0.036 |

The inlier RMSE is the depth's rounding, 1/sqrt(12) of a unit, 0.29 mm [S: the uniform
distribution]. A frame against itself converges in one iteration at the identity exactly: the
same points give residuals of 0. At 2 m the centred system is as well held as nearer: λ_1 = 0.035,
against 0.017 from 2.7 m, where the floor beyond the box takes more of the picture.

Degenerate, at both stream sizes, never converged [M 2026-11-14: `tests/test_registration.cpp`]: a
single plane, tilted at 0.4 m to 0.8 m and 2 m away, its three free motions at λ = 3.8e-6 to 4.7e-4,
the others at 0.28 to 1; and the shared tabletop box (0.20 m x 0.14 m x 0.10 m) alone on its plane
of 0.9 m x 0.8 m from about 0.5 m, λ_1 = 1.1e-3 and 7.5e-4, whose steps stopped 0.84 mm and 8.7 mm
off Δ, its weakest motion mostly a turn about the table's normal [M 2026-11-14: its eigenvector at
640x400, printed while writing the test]. A turn of 1.5° of the large box without its sphere from
2.7 m, 70 mm at the box, ended 60 mm off in a degenerate system, its first system at λ_1 = 1.55e-4;
that the box's faces, seen at a slant, lay beyond d_max from their partners at the start is likely
and unmeasured [?], and such a motion is for the pyramid's coarser levels. Few correspondences: a
surface with depth in 16 x 16 pixels gives a fitness of 0.0016; a surface or a frame without depth
and a frame 5 m beside the view give 0.

**Huber's weight**: a patch of 80 x 80 pixels of the tabletop object's frame 20 mm deeper than the
surface, on the plane, within d_max and beyond δ_G, leaves the transform 0.115° and 1.04 mm off Δ,
where plain least squares leaves 0.339° and 3.04 mm; patches of 40 and 120 pixels give 0.25 mm and
2.28 mm against 0.88 mm and 5.41 mm [M 2026-11-14, the test's case and the same with the weight 1].
Huber's weight bounds the pull of each residual, not of a whole object: the tabletop object's sphere
moved by 2 cm between the frames left a converged transform 1.84° and 12.3 mm off, with an inlier
RMSE of 1.79 mm against 0.29 mm [M 2026-11-14, a case tried while writing the test]. Such a result
is for tracking's check of the RMSE to refuse.

**The association's thresholds** [M 2026-11-14: `tests/test_registration.cpp`, and the same with
each test removed]: the patch 100 mm deeper, beyond d_max, leaves the transform 0.0022° and
0.0128 mm off Δ, and 0.092° and 0.79 mm without the test of d_max; a patch of ridges within d_max of
the plane, a sawtooth rising 20 mm over 16 pixels, 0.0015° and 0.0054 mm, and 0.084° and 0.758 mm
without the test of θ_n.

**The last system decides** [M 2026-11-14: `tests/test_registration.cpp`]: the tabletop object's
box without its sphere, from 0.67 m, the frame against itself from a start 25 mm up along the
table's normal, has its first system at λ_1 = 0.0385 and its last, at the identity, at 0.00452; it
ends degenerate.

## On the GPU

Per iteration, on the caller's stream: a kernel of one thread for each pixel of the frame's level,
in the stages' block of 32 x 8 threads (`src/gpu/kernel_launch.cuh`), which associates, forms the 35
terms, sums them in each warp in float and in each block in double, and writes each block's sums;
a kernel of 35 blocks that sums the blocks' sums in double, in a fixed order; a copy of the 280
bytes into pinned memory; and `cudaStreamSynchronize`, the one written wait, which also waits for
the work queued before. The transform and the centre go to the kernel as its arguments; nothing
else is written. The blocks' sums, the totals and the pinned buffer are requested from the pool
(`src/gpu/gpu_pool.hpp`) before it closes, for the largest frame size: 280 bytes for each block,
280 KB at 640x400 and 1.12 MB at 1280x800 [S: computed, 1000 and 4000 blocks]. The time of one
iteration on the board stands in docs/HARDWARE.md, "The per-frame path's stages": 0.355 ms at
640x400 and 1.209 ms at 1280x800 back to back, 1.649 ms and 4.348 ms paced at 30 fps, where the
GPU's clock stays at 306 MHz [M 2026-11-14]. The kernel uses 60 registers a thread, no stack and
no local memory [B 2026-11-14: `cuobjdump -res-usage`].

## Refusals

Each comes back as a result that ended as refused, with its reason, before anything is queued: the
registration holds no buffers; the frame's pyramid is empty, the level is not one of its levels, or
the level is larger than the size the buffers were requested for; the surface's images are empty
or not of one size; its intrinsics are not finite or fx, fy are not > 0; the initial transform is
not finite or its linear part is not a rotation within 1e-6; a parameter lies outside its range.
CUDA's errors end the registration as failed, with CUDA's error, which is cleared: one an earlier
call left as CUDA's last error, before the first launch; a refused launch; a failed copy or wait.

## What the prototype did, and what is not carried

The prototype's `gca::color_icp` (`src/registration/color_icp.*`) is colored ICP at one scale on
unorganized clouds [S: the audit of CUDA_3D, 2026-11-13]. Carried: the two residuals and their
Jacobians, the left update of the source in the target's frame, λ = 0.968. Not carried:

* **The nearest neighbour** within the correspondence distance on a uniform grid rebuilt every
  iteration: here the association by projection, which needs no index.
* **The system in float**, reduced by Thrust's `transform_reduce`, which allocates and waits: here
  float products, double sums above the warp, in a fixed order, from buffers of the pool.
* **The solve**: LDLT in double without damping, an identity transform on its failure, which reads
  as "no motion": here the degeneracy test and the damped step, and a failure as a value.
* **The update** by Euler angles ZYX, first order in the step: here the SE(3) exponential.
* **The convergence**, a step under 0.081° and 1 mm, and a result that is never checked: here the
  result as a value with its fitness, RMSE and conditioning.
* **Plain least squares**: here Huber's weight.
* **The color gradient** from the neighbours within 5 cm in 3D: here from the surface's window.

The prototype has no point-to-plane term alone (λ is a constant), so it is no oracle for this
term: the known transform of the synthetic pairs is. It is the oracle for colored ICP on its fixture
and the textured planes, compared when the color term is built.
