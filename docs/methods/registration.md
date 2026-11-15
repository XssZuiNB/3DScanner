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

Built: both terms, coarse to fine, with the numerics and the result of this document
(`src/registration/registration.*`, the solve of one iteration's system
`src/registration/normal_equations.*`, the SE(3) exponential `src/registration/se3.*`; the test
`tests/test_registration.cpp`). The agreement with the prototype, measured before it was deleted,
stands in "Against the prototype".

## Input

* **The frame**: a frame's pyramid (`gca::frame_pyramid`), at each level for each pixel a point p
  in the frame's camera frame F (x right, y down, z forward, in m) with its validity mark, a unit
  normal m toward the camera or (0, 0, 0) for none, and, for the color term, an intensity I_p in
  [0, 1].
* **The surface**, at each of the pyramid's levels (`gca::surface_levels`): organized points with
  normals and intensity, in the surface's own camera frame S, seen by a pinhole camera whose
  intrinsics K_S,l = (fx, fy, cx, cy) in px describe the level's picture by the one pinhole
  convention (`src/types/camera_param.hpp`): for each pixel a point q with its mark, a normal n or
  (0, 0, 0), and an intensity I_q. It comes from a ray cast of the model at a pose
  (docs/methods/sparse-tsdf.md, "The ray cast"), or, in the tests, from the levels of another
  frame's pyramid. The ray cast gives each level as the cast at K_l = K / 2^l, whose ray at a
  pixel is level 0's at 2^l times its index, a sample of the model at every pixel, not the
  pyramid's combining; through the ray cast the registration recovers the tests' known motions at
  every level as through another frame's pyramid (docs/methods/sparse-tsdf.md, "What the ray cast
  gives"). The registration reads each level through a view of its images and its intrinsics,
  `gca::organized_surface`; it does not take `gca::organized_points`, whose two writers stay the
  deprojection of a frame and the pyramid's combining (docs/methods/organized-points.md), so that
  a ray cast fills images of its own (`gca::model_surface`).
* **The initial transform** T_0 = (R_0, t_0), F to S: a point p of the frame lies at R_0 p + t_0 in
  the surface's frame. Tracking passes the identity when the surface is ray cast from the last
  pose.
* **The parameters** ("Parameters").

## Output

The result is a value (`gca::registration_result`) for every call, a refusal included:

* **The transform** T, F to S, in double: the last estimate, level 0's. It is a pose only when the
  registration converged; tracking gives the frame the pose P_S T, P_S the surface's pose (camera
  to model).
* **How it ended**, level 0's end: converged; the iterations used up; too few correspondences;
  degenerate; refused (the inputs, with the reason); failed (CUDA's error, with its message). Only
  "converged" says that T is the minimum of the method's objective near T_0 and that the last
  system holds every motion; every other end is tracking loss, never a pose (GLOSSARY.md,
  "Tracking"). A coarser level's end is not reported ("Coarse to fine").
* **The fitness**: the share of level 0's candidates, its pixels with a valid point and a normal,
  that found a correspondence ("The association by projection"), 0 to 1; 0 when the frame has no
  candidate.
* **The inlier RMSE**: the root mean square of the correspondences' point-to-plane residuals, in m,
  unweighted; 0 when there is none.
* **The iterations**: the systems built at every level, 0 for a refusal.
* **The conditioning**: the six eigenvalues of the last system, centred, scaled and normalized
  ("The system, centred and scaled"), ascending; the point-to-plane term's part holds each within 0
  to 2, and the color term only adds; zeros when no system was solved, for too few correspondences
  or none. A small eigenvalue names a motion the surface does not hold: three near 0 for a single
  plane without texture, its two shifts within itself and its turn about its normal.

The fitness, the RMSE and the conditioning are those of level 0's last system, at the transform
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

ρ the robust loss of each term ("The robust weight"), λ = 0.968 to start [S: the prototype's
colored ICP at 74dda68, "0.968 from paper"; the paper's value is unchecked here [?]]. λ weighs a
residual in m against one in intensity units, [0, 1]; the prototype took both units as here. λ
is a parameter of each call, in (0, 1] ("Parameters"): λ = 1 leaves the color term out, and the
registration is then the point-to-plane term's alone, its systems' sums bit for bit those of the
registration before the color term joined ("Accuracy").

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

The prototype wrote J_C with (I - n n^T) d_q in place of d_q, the same vector for a gradient in
the plane [S: the prototype's colored ICP at 74dda68, its least squares].

**The gradient d_q** is estimated once per call for each level of the surface that iterates, not
per iteration, by a kernel of its own: for each surface pixel with a normal, by least squares over
the valid pixels of its window, each neighbour q' contributing I_q' - I_q ≈ d_q · (q' - q) in the tangent plane, anchored at q [S: Park et al. 2017
fit the same constrained gradient over a point's neighbours in 3D]. With the unit basis a = n x e /
|n x e|, e the coordinate axis of n's smallest component, and b = n x a, d_q = α a + β b and (α, β)
solve the 2 x 2 normal equations of Σ (I_q' - I_q - α a · (q' - q) - β b · (q' - q))². The pixel has
no gradient, (0, 0, 0), and its correspondences no color residual, when fewer than 3 neighbours
remain or the 2 x 2 system's smaller eigenvalue lies below 1e-3 of its larger, the neighbours
nearly on one line [?: a choice, far below a window's spread over two rows, far above float's
rounding]; a point that is not a number fails that test too and gives no gradient. The intensity
is finite at a valid pixel by the surface's contract ("Input"), and no test guards it: a
gradient made of an intensity that is not a number would make the system not a number, which ends
the registration as degenerate ("The iterations and the end"). A correspondence whose partner has
no gradient carries the point-to-plane term alone.

* **The window**: the 3 x 3 pixels around q, R_C = 1, at every level [M 2026-11-14:
  `tests/test_registration.cpp`, with R_C = 1, 2 and 3]. With the normals' 7 x 7 window, R_C = 3,
  three cases did not converge: the large object from 2.7 m at 640x400 ended with its iterations
  used up 1.04 mm off Δ, its level 1 stepping 13 mm back and forth; the textured plane 2 m away at
  640x400 ended 1.36° and 33.5 mm off; the large box turned by 1.5° at 640x400 used its iterations
  up. A window wide against the texture's waves fits a gradient flatter than the intensity it
  predicts, and Gauss-Newton then overshoots by their ratio [S: the least-squares slope over ±r
  keeps 3 (sin x - x cos x) / x³ of a wave's, x = 2π r / wavelength; 0.18 for the box's 45 mm wave
  over ±26 mm, 2.7 m at 640x400]; at a ratio of 2 it oscillates for good. With R_C = 2 every case
  converged, the large object from 2.7 m at 640x400 1.15 mm off; with R_C = 1 every case converged
  in the fewest iterations, the same case 0.671 mm off, and R_C = 1 costs the least.
* **No rule of one surface** keeps the window off a depth edge. Where the surface's normals mark
  depth edges, as the normals stage's do (docs/methods/normals.md), a pixel whose 3 x 3 window
  holds a pixel across an edge has no normal and so no gradient: the normals' rule of one surface
  at θ_C = 75° in the fit left every result of the tests as it was [M 2026-11-14: the rule
  removed, every printed result the same]. Where they do not, normals at θ = 89° standing in for a
  surface whose normals do not mark edges, the rule made no result better: the tabletop object,
  the shared tabletop box and a board 3 cm high on the plane, at both stream sizes, ended 0.0023°
  to 0.0109° and 0.033 mm to 0.081 mm off Δ with the rule and 0.0014° to 0.0087° and 0.031 mm to
  0.068 mm without it [M 2026-11-14: those cases, written into `tests/test_registration.cpp` for
  the measurement and not kept, run with the rule and without]. So it is not carried (AGENTS.md §7 rule 6): an intensity from across an
  edge gives its pixel a steep gradient, whose residuals Huber's weight on the color term bounds.

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
the surface lacks. The point-to-plane term's δ_G is in m ("Parameters"); the color term's δ_C, in
intensity units, is 0.1 of the intensity's range, a constant of the registration [?: initial]: no
object class differs in it. δ_C = 0.03 made the fixture's real pair, fragment A onto fragment B
("Against the prototype"), end with its iterations used up 1.89° and 55 mm from the prototype's
transform, where 0.1 converged within 0.25° and 4 mm of it [M 2026-11-14:
`tests/test_registration.cpp` at 74dda68, and with δ_C = 0.03]: the fragments' intensities
differ by more than the synthetic frames', and a weight that falls off early leaves the color
term too weak to pull them together. A bright patch of frame B the surface
does not hold, a highlight, leaves the transform 0.00359° and 0.116 mm off Δ with δ_C, 0.0103° and
0.332 mm with the color term's weight 1 ("Accuracy").

## The system, centred and scaled

Each iteration builds, on the GPU, the weighted normal equations of its correspondences, both
terms in one system [S: the Gauss-Newton step of a least-squares problem], in a frame centred on a
point c:

    H = Σ λ w_G J_G J_G^T + Σ (1 - λ) w_C J_C J_C^T,   g = Σ λ w_G J_G r_G + Σ (1 - λ) w_C J_C r_C,
    J_G = [ (s - c) × n ;  n ],   J_C = [ (s - c) × d_q ;  d_q ],

the Jacobians of r_G and r_C for a rotation about c instead of S's origin, w_G and w_C each term's
Huber weight. With them come the point-to-plane term's sums, each weight times λ: W = Σ λ w_G,
Σ λ w_G (s - c), Σ λ w_G |s - c|², and Σ r_G² and the counts of correspondences and of candidates:
21 + 6 + 8 = 35 numbers, the system and its sums, which is all an iteration reads back (280 bytes).
At λ = 1 the color term is left out, not added as 0, so that a color part that is not a number
cannot enter the sums.

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

since (s - c - d) × n = (s - c) × n - d × n, and the same with d_q in place of n: the shift is
exact for both terms. ρ is the correspondences' RMS distance from their centroid c' = c + d, in m,
both by the point-to-plane term's weights. From the second iteration on d is the centroid's move in one step, a
few mm; at the first it is the whole centroid, and the shift loses up to (|c'| / ρ)² of the
products' precision ("Precision"). Then the rotation is scaled by ρ and the system normalized by
W:

    Â = S H' S / W,   b̂ = S g' / W,   S = diag(I / ρ, I).

So Â is the point-to-plane term's system as before and the color term's added to it:

    Â = Â_G + ((1 - λ) / λ) S H_C S / Σ w_G,   H_C = Σ w_C J_C J_C^T,   Â_G = S H_G S / Σ w_G,

H_G = Σ w_G J_G J_G^T. The translation's block of Â_G is Σ w_G n n^T / Σ w_G, of trace 1, and its
rotation's at most Σ w_G |s - c'|² / (ρ² Σ w_G) = 1, so each eigenvalue of Â_G lies in [0, 2]
whatever the units and the distance. For a single plane the shifts within it and the turn about its
normal give 0; a sphere's three turns about its centre give 0, which a scale taken from the
rotation's own block would hide. The color term's part is positive semi-definite, so that no
eigenvalue of Â falls below Â_G's [S: Weyl's inequality]: a motion the point-to-plane term holds
stays held, and the texture can only add what it holds. ρ² <= 0, no spread at all, is degenerate.

**Normalized by the point-to-plane term** [?: a choice, agreed by the orchestrator, 2026-11-14]:
W and ρ are the point-to-plane term's, so that τ keeps one meaning with and without texture, a
motion held by less than τ of what the point-to-plane term holds per correspondence on average
("The degeneracy test and the damped solve"). Normalized by both terms' trace instead, Σ (λ w_G +
(1 - λ) w_C |d_q|²), the eigenvalues would stay within [0, 2], but as shares of the whole: on the
tests' textures, gradients of about 25 /m [S: computed, the plane's green wave, 0.72 · 100/255 ·
2π / 0.07 m], the color term holds about 11 times the point-to-plane term's information per
correspondence [S: computed, 0.032 · 340 /m² against 0.968, 340 /m² the plane's mean |d_q|²], so a
strong texture would push the directions only the point-to-plane term holds, a plane's normal
shift among them, toward τ and below: a degeneracy of the normalization, not of the surface. The
tests' eigenvalues show the color term's weight: up to 10.2 on the textured planes and the
tabletop object, against at most 2 for the point-to-plane term alone ("Accuracy").

### Precision

* **FP64 is slow here**: the GPU runs double-precision arithmetic at 1/32 of the rate of float
  [M 2026-11-14: `cudaGetDeviceProperties`, singleToDoublePrecisionPerfRatio 32, Orin, compute
  capability 8.7, 8 multiprocessors]. A frame of 1280x800 has up to 1024000 candidates, each with
  35 products and the additions that sum them.
* **So the products are float and the sums double above the warp**: each thread forms its pixel's
  35 terms in float from the float points, normals, intensities and gradients, the color term's
  added to the point-to-plane term's; each warp sums its 32 threads' terms in
  float, by five steps of a shuffle tree, an error of at most 5 u Σ|x| for u = 2^-24 [S: N. J.
  Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002, chapter 4, pairwise
  summation], 3.0e-7 of the sum of the terms' sizes; every sum above the warp, of up to 32000 warps
  at 1280x800, is double, in a fixed order, so that the same input gives the same sums bit for bit.
  Float within the warp's 32 terms is what CODING_STANDARDS.md, "Numerics", allows live tracking,
  while refinement sums in double throughout [?: a choice, the user's, 2026-11-14]. Its measured
  benefit: double within the warp is 11 to 13 times as slow; one iteration takes, by the host's
  call that waits for its sums, on the tests'
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
  size: a motion is degenerate when it is held by less than τ of what the point-to-plane term
  holds per correspondence on average, the color term's part counted in those units ("The system,
  centred and scaled"). The last system decides, not each one: at a start far from the minimum,
  residuals that hold the weakest motion are cut by d_max or weighed down by Huber's weight [?: not
  measured apart], and a first system can look degenerate where the surface is not; the large
  object of the tests, from 2.7 m, gave λ_1 = 1.55e-4 at its first iteration from a turn of 1.5°
  and 6.5e-3 aligned [M 2026-11-14: point-to-plane-registration's test, the box without its
  sphere, the point-to-plane term on level 0].

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
[S: float32 ulp]; the host keeps T in double. Each iteration, at each level ("Coarse to fine"):

1. builds the system and its sums at T on the GPU and reads them back, the one wait;
2. ends as **too few correspondences** when the fitness is below its least value, f_min, or no
   correspondence was found;
3. centres, scales and normalizes the system and decomposes it; ends as **degenerate** when it
   cannot step, ρ² <= 0 or a system or step that is not a number;
4. takes the damped step and applies it;
5. ends when the step turned by |ω| < ε_ω and moved the centroid by |υ_c| < ε_υ: as **converged**
   when this last system's λ_1 >= τ, as **degenerate** when not.

After the level's most iterations without one of these ends, it ends as **degenerate** when the
last system's λ_1 < τ, else as **the iterations used up**, with the transform after the last step.
Level 0's end is the registration's. The prototype's convergence, a step under 0.081° and 1 mm, was
not checked before the pose was used; here every end but "converged" is tracking loss.

## Coarse to fine

The frame's pyramid has three levels, each with its intrinsics K / 2^l
(docs/methods/frame-pyramid.md), and the surface comes at the same three levels ("Input"). The
registration runs level 2, then level 1, then level 0, the frame's level l against the surface's
level l, each level from the transform the coarser one ended with: a transform is metric, so it
carries over unchanged, and so does the last centroid, the next system's centre. KinectFusion
tracks so over three levels [S: Newcombe et al. 2011, "Sensor pose estimation"].

* **Each level has its own d_max_l and most iterations n_l** ("Parameters"). A coarser level's
  points are means over a wider window and its texture is smoothed, so that its association and
  its gradients reach further; d_max_l = 2^l d_max lets it pair what a larger misalignment moved,
  which Huber's weight then bounds [?: initial, 2^l]. Its benefit is measured [M 2026-11-14:
  `tests/test_registration.cpp`, and with level 0's d_max at every level]: the large box from 2.7 m
  shifted by 147 mm, beyond level 0's 0.10 m, converges 0.0036° and 0.163 mm off Δ at 640x400 with
  d_max per level, and ends with its iterations used up 2.87° and 143 mm off with 0.10 m at every
  level. At 1280x800 neither recovers it, 109 mm and 115 mm off: that pyramid's level 2, 320x200,
  is 640x400's level 1. The turn of 1.5° below comes out the same either way, its 70 mm within
  0.10 m. n_l = 0 skips a coarser level, so that how many levels the registration needs is a
  parameter to measure; level 0 iterates at least once.
* **A coarser level passes on its last transform whatever its end** [?: a choice]: converged, its
  iterations used up, degenerate, or too few correspondences, which leave the transform it started
  from. A coarse level that ended degenerate leaves its damped steps behind: along a motion it held
  weakly each step was shortened to at most |v_k · b̂| / μ ("The degeneracy test and the damped
  solve"), and level 0 starts from there and judges its own last system. Only a refusal or a
  failure ends the call at a coarser level. The coarser levels' ends are not reported; the
  iterations count every level's systems.
* **What it buys** [M 2026-11-14: `tests/test_registration.cpp`]: the large box from 2.7 m turned
  by 1.5°, 70 mm at the box, ends with level 0's iterations used up, 0.60° and 85 mm off Δ at
  640x400 and 0.87° and 83 mm at 1280x800, when level 0 runs alone; through the pyramid it
  converges, 0.0024° and 0.096 mm and 0.0016° and 0.066 mm off. A coarse level does not converge
  as finely as level 0: on the tabletop object at 640x400 level 2 used its 10 iterations up with
  steps of about 1e-3 rad, level 1 converged in 4 and level 0 in 3 [M 2026-11-14, a run of the
  test with each step printed].

## Parameters

The registration takes, for each call (`gca::registration_parameters`), the values that differ
between the object classes or the levels; the tests name them, and the object class takes them
over once tracking uses them. Starting values [?: initial], and the range outside which a value is
refused as a mistake [?: a choice]:

| parameter | meaning | tabletop object | large object | refused outside |
|---|---|---|---|---|
| d_max_l | the association's largest distance at level l = 0, 1, 2, m | 0.05, 0.10, 0.20 | 0.10, 0.20, 0.40 | (0, 0.5] |
| n_l | the most iterations at level l = 0, 1, 2 | 20, 10, 10 | 20, 10, 10 | level 0: 1 to 100; levels 1, 2: 0 to 100 |
| δ_G | the point-to-plane term's Huber threshold, m | 0.005 | 0.030 | (0, the smallest d_max_l] |
| λ | the point-to-plane term's weight; the color term's is 1 - λ | 0.968 | 0.968 | (0, 1] |

* **d_max_0**: 5 cm is the prototype's correspondence distance [S: the old loop, CUDA_3D's
  `apps/app.cpp`]; twice that for the large object, whose depth noise by the bound below is 6.8
  times as large at the middle of its range [S: computed]. d_max_l = 2^l d_max_0, its benefit
  measured ("Coarse to fine").
* **δ_G**: about the datasheet's bound on the depth noise, σ = c z² with c = 0.01 /m
  (docs/methods/sparse-tsdf.md, "The truncation distance"), at the middle of each class's depth
  range: 4.2 mm at 0.65 m and 29 mm at 1.7 m [S: computed].
* **n_l**: 20 at level 0, where every case of the tests converged within 3 to 4 iterations after
  the coarser levels; 10 at levels 1 and 2, which do not converge as finely and hand on what they
  reach. The tests' recoveries took 7 to 23 systems in all, the fixture's real pair 29
  [M 2026-11-14: `tests/test_registration.cpp`, the fixture's pair at 74dda68], the time per
  frame "On the GPU".
* **λ**: the paper's value by the prototype; the user's starting value (spec, "Registration"),
  kept at 0.968 [?: initial] until real sequences tune it (the user's decision, 2026-11-14, late
  evening; `tracking-thresholds-from-real-sequences`). On the tests' textures it lets the color
  term move the minimum by up to 0.671 mm where the point-to-plane term alone came within 0.142 mm
  ("Accuracy").

The values that are the same in every call are the registration's constants, θ_n, δ_C and R_C in
`gca::registration` and the others beside the solve of a system
(`src/registration/normal_equations.hpp`), each [?: initial] until measured on sequences, R_C
measured on the tests' frames:

| constant | meaning | value |
|---|---|---|
| θ_n | the association's largest angle between normals | 30° |
| δ_C | the color term's Huber threshold, intensity units | 0.1 |
| R_C | the half size of the gradient's window, px of the level | 1 |
| f_min | the least fitness of a system that is solved | 0.1 |
| τ | the least eigenvalue of a last system that is not degenerate | 0.01 |
| μ | the damping | τ / 10 = 0.001 |
| ε_ω, ε_υ | the step under which the iterations end | 1e-5 rad, 1e-5 m |

τ is dimensionless, a share of the point-to-plane term's information per correspondence: 0.01 of
it along the weakest motion. The values below are the point-to-plane term's on level 0, as
point-to-plane-registration measured them. A plane's three free motions hold what the noise of its
estimated normals puts there, at most 4.7e-4 on the tests' planes at level 0 ("Accuracy"); a frame's noisier normals raise that floor.
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
relocalization. The color term adds the texture's information to the same system: with it the
textured planes' smallest eigenvalue is 0.279 to 0.309, the plane's normal shift and its turns about
the in-plane axes, which the point-to-plane term holds, while the color term holds the in-plane
motions at 0.46 to 0.60 [M 2026-11-14: `tests/test_registration.cpp`].

ε_υ = 10 µm and ε_ω = 1e-5 rad, 10 µm at 1 m, lie far below a depth unit of 1 mm. Tracking's check
of a result, its thresholds of fitness, RMSE and conditioning, is stricter and its own.

## Accuracy

On pairs of the synthetic source's frames whose transform Δ is known, from the identity, depth unit
1 mm, the camera's intrinsics at each stream size, the normals' starting parameters and the object
classes' starting values ("Parameters") [M 2026-11-14: `tests/test_registration.cpp`]. The tabletop
object is a box of 0.30 m x 0.20 m x 0.25 m and a sphere of 0.10 m beside it on a plane, seen from
0.5 m, moved by 1.5° and 10.8 mm; the large object a box of 1.2 m x 0.8 m x 1.0 m and a sphere of
0.4 m on a floor, seen from 2.7 m and from 2 m, moved by 0.5° and 14.1 mm; every surface carries the
synthetic source's texture of sine waves 45 mm to 100 mm long (`tests/synthetic_source.cpp`). The
error is the turn and the shift of Δ^-1 T, the shift at the camera's centre. By the full method,
both terms coarse to fine:

| case | size | systems | error | fitness | inlier RMSE | λ_1 |
|---|---|---|---|---|---|---|
| tabletop object, 0.5 m | 640x400 | 15 | 0.00624°, 0.0594 mm | 0.9884 | 0.292 mm | 0.297 |
| | 1280x800 | 11 | 0.00991°, 0.0761 mm | 0.9825 | 0.293 mm | 0.308 |
| large object, 2.7 m | 640x400 | 20 | 0.0136°, 0.671 mm | 0.9671 | 0.346 mm | 0.202 |
| | 1280x800 | 12 | 0.00165°, 0.0693 mm | 0.9674 | 0.309 mm | 0.224 |
| large object, 2 m | 640x400 | 19 | 0.0013°, 0.0464 mm | 0.9673 | 0.326 mm | 0.258 |
| | 1280x800 | 12 | 0.000226°, 0.00377 mm | 0.9681 | 0.314 mm | 0.269 |
| large box, 2.7 m, turned by 1.5° | 640x400 | 19 | 0.00241°, 0.0958 mm | 0.9348 | 0.334 mm | 0.194 |
| | 1280x800 | 17 | 0.00161°, 0.0664 mm | 0.9341 | 0.304 mm | 0.215 |
| shared tabletop box on its plane | 640x400 | 10 | 0.00323°, 0.0351 mm | 0.9976 | 0.337 mm | 0.407 |
| | 1280x800 | 8 | 0.00152°, 0.0267 mm | 0.9981 | 0.343 mm | 0.414 |

The inlier RMSE is the depth's rounding, 1/sqrt(12) of a unit, 0.29 mm [S: the uniform
distribution]. A frame against itself converges in one iteration at each level at the identity
exactly: the same points and intensities give residuals of 0. The point-to-plane term alone came
closer on the box and sphere, 0.0034° and 0.142 mm at worst (below): the color term, holding about
ten times its information here ("The system, centred and scaled"), moves the minimum by what its
linear model of the texture misses within a pixel, most from 2.7 m at 640x400, where the box's
waves of 45 mm to 80 mm span 5 to 9 pixels [S: computed, 8.8 mm a pixel]. Its gain is where the
geometry does not hold a motion: the shared tabletop box on its plane, degenerate for the
point-to-plane term and 0.84 mm to 8.7 mm off, converges within 0.035 mm.

**The color term** [M 2026-11-14: `tests/test_registration.cpp`]:

* **A textured plane moved within itself**, 0.5° about its normal and 3 mm and -2 mm along it, the
  geometry the same in both views: by the full method converged, the tilted plane at 0.4 m to 0.8 m
  0.00091° and 0.036 mm off at 640x400 and 0.00027° and 0.023 mm at 1280x800, the plane 2 m away
  0.00012° and 0.0049 mm and 0.00002° and 0.0015 mm, λ_1 = 0.279 to 0.309; with the color term
  off (λ = 1) degenerate, three eigenvalues of 3.9e-6 to 5.3e-4, 0.5° and 8.1 mm and 9.2 mm off,
  where the steps stopped; the same planes grey, without texture, degenerate with the color term
  on, the same three eigenvalues: where nothing changes along the surface the color term adds
  nothing.
* **λ weighs the terms**: the tabletop object's frame against itself, its color replaced by a grey
  ramp along the picture's rows, 32 to 224 levels over its width, and frame B's ramp shifted by
  20 pixels, so that the color term alone asks for a shift of about 33 mm [S: computed, 20 px at
  0.5 m with fx = 306.18 px]: at λ = 0.968 the geometry holds the transform within 1.94 mm of the
  identity, at λ = 0.5 it moves 21.1 mm, and with λ applied to the wrong term 62.6 mm, its
  iterations used up [M 2026-11-14, and with λ and 1 - λ swapped in the kernel's inputs].
* **Huber's weight on the color term**: a patch of 80 x 80 pixels of frame B 150 levels brighter, a
  highlight the surface does not hold, on the textured tilted plane: 0.00359° and 0.116 mm off,
  where the color term's plain least squares leaves 0.0103° and 0.332 mm [M 2026-11-14, and with
  the color term's weight 1].

**The point-to-plane term on level 0**, the coarser levels skipped and λ = 1, is the registration as
point-to-plane-registration built it: every system's 35 sums are the same bit for bit as that
registration's on the same frames, in 16 cases, the planes, the box on its plane, the few
correspondences, the patches and the ends below, 86 systems in all; so are their steps, their
eigenvalues, fitness, RMSE and iterations. Their transforms agree within 4.4e-16, two units of
double's last place at 1: the product T ← ΔT T, the same Eigen expression on the same numbers,
rounds differently compiled into the new translation unit [M 2026-11-14: a scratch build of the
committed registration and its test, and of this tree's, each printing every system's sums, step
and transform in hexadecimal]. Its values from that ticket stand:

| case | size | iterations | error | fitness | inlier RMSE | λ_1 |
|---|---|---|---|---|---|---|
| tabletop object, 0.5 m | 640x400 | 4 | 0.0024°, 0.014 mm | 0.988 | 0.29 mm | 0.029 |
| | 1280x800 | 4 | 0.0025°, 0.031 mm | 0.982 | 0.29 mm | 0.031 |
| large object, 2.7 m | 640x400 | 4 | 0.0034°, 0.142 mm | 0.967 | 0.33 mm | 0.017 |
| | 1280x800 | 4 | 0.0023°, 0.083 mm | 0.967 | 0.31 mm | 0.017 |
| large object, 2 m | 640x400 | 4 | 0.0029°, 0.069 mm | 0.967 | 0.33 mm | 0.035 |
| | 1280x800 | 4 | 0.0007°, 0.018 mm | 0.968 | 0.31 mm | 0.036 |

At 2 m the centred system is as well held as nearer: λ_1 = 0.035, against 0.017 from 2.7 m, where
the floor beyond the box takes more of the picture.

Degenerate, at both stream sizes, never converged [M 2026-11-14: `tests/test_registration.cpp`],
on level 0 and through the pyramid alike with λ = 1: a single plane, tilted at 0.4 m to 0.8 m and
2 m away, its three free motions at λ = 3.8e-6 to 4.7e-4, the others at 0.28 to 1; and the shared
tabletop box (0.20 m x 0.14 m x 0.10 m) alone on its plane of 0.9 m x 0.8 m from about 0.5 m,
λ_1 = 1.1e-3 and 7.5e-4, whose steps stopped 0.84 mm and 8.7 mm off Δ on level 0, its weakest
motion mostly a turn about the table's normal [M 2026-11-14: its eigenvector at 640x400, printed
while writing point-to-plane-registration's test]. Few correspondences, on level 0 and by the full
method, each level ending at its first system: a surface with depth in 16 x 16 pixels gives a
fitness of 0.0016; a surface or a frame without depth and a frame 5 m beside the view give 0.

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

The patches, the association's thresholds and the ends above are the point-to-plane term's on level
0, as the cases of that ticket.

## Against the prototype

The prototype's colored ICP, CUDA_3D's, was the oracle until this registration agreed with it; it
is deleted since, with its fixture, and its code and the comparison stand in git at 74dda68
(`src/registration/`, `tests/test_registration.cpp`). On the same frames, the prototype given the
points their level 0 held with their colors, this registration's transform lay within a stated
tolerance of the prototype's, about twice the largest difference measured, its turn and its shift
at the camera's centre and at a point of the scene [M 2026-11-14: `tests/test_registration.cpp`
at 74dda68]; this is the record of the agreement:

| case | ours from Δ | the prototype's from Δ | ours from the prototype's | tolerance |
|---|---|---|---|---|
| textured tilted plane, 640x400 | 0.00091°, 0.036 mm | 0.0024°, 0.42 mm | 0.0016°, 0.46 mm, 0.47 mm | 0.004°, 1.0 mm, 1.0 mm |
| fragment B against itself | 0.0012°, 0.023 mm | 0.0072°, 0.46 mm | 0.0060°, 0.45 mm, 0.43 mm | 0.012°, 0.9 mm, 0.85 mm |
| fragment A onto fragment B | | | 0.339°, 2.72 mm, 2.92 mm | 0.7°, 5.5 mm, 6 mm |

* **The textured plane**: the plane moved within itself of "Accuracy", the prototype on 85535
  points of each frame after 3 mm voxels, normals from 2 cm, gradients from 1 cm, correspondences
  within 5 cm and at most 50 iterations, so that its own stop rule ended it [?: a choice].
  CUDA_3D's 5 cm gradient radius would flatten the plane's waves of 5 cm to 10 cm to nothing
  [S: computed, "The color term"]. The scene's point is the point the views look at.
* **The fixture**, two fragments of a room's scan with their triangles, A and B, which the
  prototype's tests used and whose license was never checked (it went with the prototype): each
  fragment was rendered as organized frames at 640x400 by a mesh in the synthetic source, every
  pixel the nearest exact hit of its ray with a triangle, its color the triangle's corners'
  weighted by the hit, depth rounded to 1 mm, seen from a camera 1.2 m from A's centroid along
  the mean of its normals, looking at the centroid [?: a choice]. The full method took the large
  object's values; the prototype CUDA_3D's, 8 mm voxels, normals from 4 cm, correspondences
  within 5 cm, gradients from 5 cm [S: the prototype's colored-ICP test at 74dda68], at most 50
  iterations: 46521 and 46158 points for B against itself, 38197 and 46158 for the real pair. B
  against itself was moved by 1° and 7.8 mm; A onto B started from the identity, 3.64° and
  74.2 mm from the prototype's transform: the fixture is nearly one plane, A's points 4.0 mm RMS
  off it and B's plane near it turned by 3.0° and 36 mm off [M 2026-11-14, a scratch script on
  the PLY files], the case colored ICP is made for.
* **What the gap is**: both methods took the same points, so the gap is the methods': the
  association, by projection here, the nearest neighbour within 5 cm in 3D there; the sampling,
  every pixel here, one point per 8 mm voxel there; the gradient, over 3 x 3 pixels of 4 mm here,
  over 5 cm in 3D there; Huber's weights here, plain least squares there; three levels here, one
  scale there. On B against itself the prototype lay 0.46 mm off the known motion where ours lay
  0.023 mm, so most of that gap was the prototype's own error on these frames. On the real pair
  the 0.339° between them is 2.92 mm at A's centroid, 1.2 m away, mostly a turn about the
  plane's normal that only the texture holds; which one is nearer the truth the fixture cannot
  say. Given every vertex of both fragments instead, about 86 % and 62 % of which the frames
  show [?: a reviewer's projection], the prototype lay 0.246° and 3.95 mm from ours.

## On the GPU

First, for each level that iterates, on the caller's stream, a kernel of one thread for each
pixel of the surface's level, in the stages' block of 32 x 8 threads (`src/gpu/kernel_launch.cuh`),
which fits the pixel's gradient and writes it. Then per iteration: a kernel of one thread for each
pixel of the frame's level, in the same block, which associates, forms the 35 terms of both terms,
sums them in each warp in float and in each block in double, and writes each block's sums; a
kernel of 35 blocks that sums the blocks' sums in double, in a fixed order; a copy of the 280 bytes
into pinned memory; and `cudaStreamSynchronize`, the one written wait, which also waits for the
work queued before. The transform, the centre and the weights go to the kernel as its arguments;
nothing else is written. The blocks' sums, the totals, the pinned buffer and the gradients are
requested from the pool (`src/gpu/gpu_pool.hpp`) before it closes, for the largest frame size: 280
bytes for each block, 280 KB at 640x400 and 1.12 MB at 1280x800 [S: computed, 1000 and 4000
blocks], and 12 bytes for each pixel of each level, 4.03 MB and 16.1 MB [S: computed, a float3 over
1 + 1/4 + 1/16 of the pixels]. The system kernel uses 64 registers a thread, the gradient kernel
46, neither a stack nor local memory [B 2026-11-15: `cuobjdump -res-usage`].

The time of one frame's registration on the board stands in docs/HARDWARE.md, "The per-frame path's
stages": by the full method on the tabletop object's pair, 6.84 ms at 640x400 and 19.5 ms at
1280x800 paced at 30 fps, where the GPU's clock stays at 306 MHz, and 2.31 ms and 6.48 ms back to
back, 15 and 11 systems, the gradient kernels 0.66 ms and 2.44 ms of it paced [M 2026-11-15]. The
time of one point-to-plane iteration without the color term stands there as the evidence of the
float sums within the warp ("Precision"). The stream size is settled by the replay's measurement
of the whole path, not by these times (the user's decision, 2026-11-14, late evening).

## Refusals

Each comes back as a result that ended as refused, with its reason, before anything is queued: the
registration holds no buffers; the frame's pyramid is empty, or its level 0 is larger than the size
the buffers were requested for; a level of the surface has empty images, images not of one size,
a size larger than the buffers' at that level, or intrinsics that are not finite or whose fx, fy
are not > 0; the initial transform is not finite or its linear part is not a rotation within 1e-6;
a parameter lies outside its range.
CUDA's errors end the registration as failed, with CUDA's error, which is cleared: one an earlier
call left as CUDA's last error, before the first launch; a refused launch; a failed copy or wait.

## What the prototype did, and what is not carried

The prototype's colored ICP (in git at 74dda68, `src/registration/`) was colored ICP at one scale
on unorganized clouds [S: the audit of CUDA_3D, 2026-11-13]. Carried: the two residuals and their
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

The prototype had no point-to-plane term alone (λ was a constant), so it was no oracle for that
term: the known transform of the synthetic pairs is. It was the oracle for colored ICP on its
fixture and the textured plane ("Against the prototype").
