# Refinement: the trajectory optimized over keyframes

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The stage after the operator's stop (R12; the spec, "Refinement"): the keyframes' poses are optimized
as a pose graph, every other frame's pose follows its keyframe's, and the frames, read back from
disk, are fused at the full voxel size into a new sparse TSDF, from which the full model is
extracted. The pose graph is the one of graph-based SLAM [S: G. Grisetti, R. Kümmerle, C.
Stachniss, W. Burgard, "A Tutorial on Graph-Based SLAM", IEEE Intelligent Transportation Systems
Magazine 2(4), 2010], solved by this tree's own damped Gauss-Newton on SE(3) with a robust weight,
in double precision (R31). This tree's code is written from the sources named here; no
reconstruction library's code is read or copied for it (docs/adr/0002-own-reconstruction-code.md).

Built so far: the pose graph and its solve, on the host (`src/refinement/pose_graph.*`; the test
`tests/test_pose_graph.cpp`), and the fusion of the sequence at the full voxel size with the
trajectory as tracked (`src/refinement/sequence_fusion.*`; the test
`tests/test_sequence_fusion.cpp`). Left to `refinement-optimizes-the-trajectory`, with its part of
this document: which pairs of keyframes get an edge and the weights, the frames that follow their
keyframes ("How the other frames follow their keyframes" says how), and the optimized poses put
before the fusion.

## The graph

* **A node** is a keyframe k with its pose T_k = (R_k, t_k): from keyframe k's camera frame
  (x right, y down, z forward, in m) into the models' frame, the first fused frame's camera frame
  (the spec). The poses tracking gave are the start.
* **An edge** joins a target node i and a source node j with a measured transform Z_ij, from
  keyframe j's camera frame into keyframe i's, and its information Ω_ij. A registration of
  keyframe j's frame against a surface seen from keyframe i returns such a transform
  (docs/methods/registration.md, "Output": F is j's camera frame, S is i's). Where the poses and the
  edge disagree,

      T_i^-1 T_j = exp(ε^) Z_ij,

  ε = (ω, υ) the edge's error, a twist in keyframe i's camera frame, ω in rad and υ in m. Ω_ij is
  the inverse of ε's covariance: 6 x 6, symmetric, positive definite, rotation first, in rad^-2,
  rad^-1 m^-1 and m^-2. Measured in the edge's standard deviations, the error of an edge that is
  right is about 1 along each of its six axes.
* **Where Ω comes from** is refinement-optimizes-the-trajectory's, with the pairs: an edge's
  information comes from its registration's system (the spec: "each edge is a checked registration
  weighted by its system"). The registration's step is the same left motion exp(ξ^) T in S
  (docs/methods/registration.md, "The update"), so ε is that ξ; the system's scale, which turns it
  into an information in the units above, is that ticket's to settle [?]. Between consecutive
  keyframes whose registration fails its check, the relative pose tracking gave stands in with a
  small Ω (the spec).
* **The fixed node** is node 0, the first keyframe, held at its pose. A pose graph measures only
  relative transforms, so the whole graph may turn and shift without changing an error; one pose
  held fixed removes those six motions [S: Grisetti et al. 2010]. Node 0 holds the models' frame
  where tracking put it, so a model fused before refinement and the full model share the frame.

## The residual

For an edge (i, j, Z, Ω) at poses T_i, T_j:

    E = T_i^-1 T_j Z^-1 = (R_E, t_E),   r = (ω_E, t_E),

ω_E the rotation vector of R_E (its angle θ_E in [0, π], in rad) and t_E its translation (in m), both
in keyframe i's camera frame; E is the identity when the edge and the poses agree. The twist of E
is log E = (ω_E, V(ω_E)^-1 t_E), V = I + B [ω_E]x + C [ω_E]x² as in the exponential
(docs/methods/registration.md, "The update"), so r and ε agree to first order and Ω weighs r as it
weighs ε [S: computed]. This chart of SE(3) near the identity is chosen over log E because its
Jacobian needs only the rotation's inverse left Jacobian, and is exact with it; that of log E needs
SE(3)'s, with the coupling block Q [S: T. D. Barfoot, *State Estimation for Robotics*, 2017,
chapter 7].

The edge's normalized residual is s = sqrt(r^T Ω r): its error in its standard deviations, whose
square is χ²-distributed with 6 degrees of freedom for an edge that is right, at the true poses
[S: the χ² distribution of a Gaussian vector's Mahalanobis norm].

## The update and the Jacobians

Each node moves in its own camera frame, T_k ← T_k exp(δ_k^), δ_k = (ω_k, υ_k), ω_k in rad and υ_k
in m; to first order the camera turns by ω_k about its centre and the centre moves by R_k υ_k in the
models' frame. The exponential is the registration's, `src/registration/se3.*`, the one copy
(AGENTS.md §7 rule 4).

Moving both nodes,

    exp(-δ_i^) T_i^-1 T_j exp(δ_j^) Z^-1 = exp(-δ_i^) exp((Ad(T_ij) δ_j)^) E,
    T_ij = T_i^-1 T_j,   Ad(T) = [ R  0 ; [t]x R  R ] for T = (R, t),

since T exp(δ^) T^-1 = exp((Ad(T) δ)^) [S: Barfoot 2017, chapter 7, the adjoint, here with the
rotation first]. To first order E ← exp(ζ^) E, ζ = Ad(T_ij) δ_j - δ_i. A small left motion ζ = (a, b)
moves R_E to exp([a]x) R_E and t_E to t_E + a x t_E + b, so that

    ∂r/∂ζ = M(r) = [ J_l^-1(ω_E)  0 ; -[t_E]x  I ],
    J_i = ∂r/∂δ_i = -M(r),   J_j = ∂r/∂δ_j = M(r) Ad(T_ij).

J_l^-1 is the inverse of the rotation's left Jacobian, which carries a left turn a of R_E into the
change of its rotation vector, log(exp([a]x) R_E) = ω_E + J_l^-1(ω_E) a to first order [S: Barfoot
2017, chapter 7]:

    J_l^-1(φ) = I - ½ [φ]x + β [φ]x²,   β = 1/θ² - (1 + cos θ) / (2 θ sin θ),   θ = |φ|.

Below θ = 1e-4 rad β is its series 1/12 + θ²/720, whose next term, θ⁴/30240 < 3.4e-21, lies under
double's rounding of 1/12 [S: computed, from β = 1/θ² - cot(θ/2) / (2θ) and the series of the
cotangent]. β stays finite up to θ = π, where it is 1/π², and a rotation vector's angle lies in
[0, π]. These Jacobians agree with central differences of r on 200 pairs of random poses with
edges off by up to about 1.5 rad, the largest difference 5.9e-10 [M 2026-11-14: a scratch numpy
program, steps of 1e-6, near the cube root of double's rounding, 4.8e-6, where a central
difference's truncation and rounding are balanced, S: computed]; the tests check a solved graph's
stationarity the same way ("Accuracy").

## The robust weight

Each edge carries Huber's weight on its normalized residual [S: P. J. Huber, "Robust Estimation of
a Location Parameter", Annals of Mathematical Statistics 35(1), 1964], the registration's weight
(docs/methods/registration.md, "The robust weight"): the loss is

    ρ(s) = s²/2 for s <= k,   k s - k²/2 beyond,

minimized by iteratively reweighted least squares [S: P. W. Holland, R. E. Welsch, "Robust
regression using iteratively reweighted least-squares", Communications in Statistics - Theory and
Methods 6(9), 1977]: each linearization weighs its edge by w = 1 for s <= k and w = k / s beyond,
held fixed while it solves. k = 3.548, the square root of 12.592, the 0.95 quantile of χ² with 6
degrees of freedom: an edge that is right, at the true poses, keeps the weight 1 in 95 of 100 cases
when Ω is its information [S: computed; ?: initial].

**Why Huber's and not a weight that falls to 0.** Weights that redescend, as the line processes of
S. Choi, Q.-Y. Zhou, V. Koltun, "Robust Reconstruction of Indoor Scenes", CVPR 2015, or the dynamic
covariance scaling of P. Agarwal et al., "Robust Map Optimization using Dynamic Covariance Scaling",
ICRA 2013, let a far wrong edge pull with almost no force [?: the papers' weights, not re-derived
here]. But at the start, tracking's trajectory, an edge that closes a loop and is right carries the
whole drift of the loop as its error, many standard deviations; a redescending weight can weigh it
down so far that the loop never closes. Huber's loss is convex, so the weight adds no minimum of its own, and every edge
keeps pulling. Its cost: a wrong edge pulls with a bounded force, k standard deviations' worth, and
bends the graph by what that force moves ("Accuracy" measures it). An edge whose weight ends low
is not dropped and solved again: that is a second method, built only on a measured benefit
(AGENTS.md §7 rule 6).

## The damped Gauss-Newton

**The system.** At the poses T, each edge gives r, s, w, J_i and J_j, and the system

    H = Σ w J^T Ω J,   g = Σ w J^T Ω r,

each edge adding to the blocks (i, i), (i, j), (j, i) and (j, j) of H and to i and j of g; node 0's
rows and columns are left out, 6 (N - 1) unknowns for N nodes. Everything is summed and solved in
double (CODING_STANDARDS.md, "Numerics": refinement sums in double throughout). H is positive
definite when every Ω is and the graph is in one piece: δ^T H δ = 0 asks every edge for J_i δ_i +
J_j δ_j = 0, that is δ_i = Ad(T_ij) δ_j (M(r) is invertible: block triangular, J_l^-1 invertible for
θ < 2π), which is one rigid motion of all poses, and node 0 held fixed makes it none [S: computed].

**The solve.** Each edge fills four blocks of 6 x 6, so H is sparse; it is factorized by a sparse
Cholesky decomposition L L^T with the approximate minimum degree ordering (Eigen's `SimplicialLLT`
with `AMDOrdering`, `Eigen/SparseCholesky`), which reports a matrix that is not positive definite as
a failure. Both are MPL2 in Eigen 3.4.0: the headers of `Amd.h` and `SimplicialCholesky_impl.h` say
so, and the only files that include `NonMPL2.h` are two of `unsupported/Eigen/src/IterativeSolvers/`
[S: Eigen 3.4.0's sources, grep]; a program with `EIGEN_MPL2_ONLY` that factorizes and solves with
them compiles and runs, and one that includes `IterationController.h` stops at its `#error`
[M 2026-11-14: g++ 11.4, a scratch program] (CODING_STANDARDS.md, "Eigen: MPL2 files only").

**The damping** is Marquardt's, scaled by H's diagonal [S: D. W. Marquardt, "An Algorithm for
Least-Squares Estimation of Nonlinear Parameters", Journal of the Society for Industrial and Applied
Mathematics 11(2), 1963]:

    (H + μ D) h = -g,   D = diag(H),

so that each unknown is damped in its own units, rad or m, by the share μ of its own curvature. The
step is judged by its gain ratio, the robust objective's decrease over the decrease the weighted
quadratic model predicts,

    ϱ = (F(T) - F(T ⊕ h)) / (½ h^T (μ D h - g)),   F = Σ ρ(s_e),

T ⊕ h each node moved by its δ_k. ϱ > 0 accepts the step: the poses move, the edges are linearized
anew with their new weights, and μ ← μ max(1/3, 1 - (2ϱ - 1)³), ν ← 2. Otherwise the poses stay and
μ ← μ ν, ν ← 2ν [S: H. B. Nielsen, "Damping Parameter in Marquardt's Method", IMM-REP-1999-05,
Technical University of Denmark, 1999; K. Madsen, H. B. Nielsen, O. Tingleff, *Methods for
Non-Linear Least Squares Problems*, 2nd ed., 2004, the Levenberg-Marquardt method]. μ starts at
1e-4, near the Gauss-Newton step, since tracking's trajectory starts near the minimum [?: initial];
ν at 2; the 2 that doubles ν, the least shrink 1/3 and the cube are Nielsen's [S: Nielsen 1999]. A
factorization that fails counts as a rejected step. No input that passes the checks has been found
that makes one fail, so that path is untested [?: untested]: a connected graph's H is positive
definite, the damping μ D lies far above its rounding while μ is near its start, and a system so
large that it overflows double gave infinite pivots and a finite step, or Huber's weight k / s
shrank it below the overflow [M 2026-11-15: a scratch program, informations of 1e300 to 1.5e308
and spans of eigenvalues to 1e16]. A mutant that ends the solve as converged on such a failure
passes the tests. The weighted model is a fair judge:
Huber's ρ(sqrt(u)) is concave in u, so ρ(s') <= ρ(s) + ½ w(s) (s'² - s²) for every s', and where the
linearization holds the robust objective falls at least as far as the weighted squares do [S:
computed; the half-quadratic bound behind IRLS].

## The end

Each iteration factorizes and solves once and judges its step. The solve ends:

* **converged** at the first step, accepted or not, under which every node turns by less than ε_ω =
  1e-6 rad and moves by less than ε_υ = 1e-6 m, a thousandth of a depth unit [?: initial], when the
  Gauss-Newton step without damping, taken at the poses the solve would end with, lies under them
  too [?: the user's decision, 2026-11-15, option (a) over a refusal of Ω's span]; an accepted
  step is applied. When the undamped step is not small, the solve goes on with its damping as it
  was. A small step that is not accepted comes after rejected steps have grown the damping; with
  exact derivatives that means the objective has no decrease left within double's rounding, since
  along a descent direction a short enough step lowers it. The undamped step guards against a damped
  step that is small only because the damping is large where the curvature is small: Marquardt's D =
  diag(H) does that in the weak directions of an Ω whose eigenvalues span far ("Rejected steps and
  an ill-conditioned Ω"). A wrong Jacobian makes both steps wrong alike, so the tests check that a
  solved graph is a stationary point of the objective by central differences, which no mistake of a
  Jacobian passes (the sign's mutant in "Accuracy").
* **the iterations used up**: max_iterations steps without one, with the poses after the last
  accepted step; not poses refinement takes as optimized.
* **disconnected**, with no poses: the graph has no edge, or a node that no chain of edges joins to
  node 0, whose pose relative to node 0 nothing measures. A breadth-first walk from node 0 over the
  edges finds it before anything is solved.
* **refused**, with the reason and no poses, before anything is solved: no node; max_iterations
  outside 1 to 1000 [?: a choice]; a pose or an edge's transform not finite or not rigid (its linear
  part a rotation within 1e-6, the registration's test); an edge whose node is not in the graph or
  that joins a node to itself; an Ω that is not finite, not symmetric within 1e-9 of its largest
  entry, or not positive definite (a Cholesky decomposition of it fails) [?: a choice].

The result carries the end, the reason, the poses, each edge's last weight w (1 for an edge within
k, below 1 for one weighed down) and the iterations.

## How the other frames follow their keyframes

For refinement-optimizes-the-trajectory to build: a frame with a tracked pose P_f follows the
keyframe k taken last at or before it, keeping the relative pose tracking gave,

    P'_f = T'_k T_k^-1 P_f,

T_k and T'_k the keyframe's pose before and after the solve [?: a choice; the spec: "every other
frame's pose follows its keyframe's"]. A frame marked as tracking loss has no pose and keeps none.
The correction T'_k T_k^-1 changes from one keyframe to the next by that pair's share of the drift;
interpolating between two keyframes' corrections is a candidate built only on a measured benefit.

## The fusion of the sequence

The full model is fused anew from the scan's frames, read back from its sequence, at the object
class's full voxel size (R12; the spec, "Refinement": "The frames are then read back from disk and
fused at the full voxel size into a new sparse TSDF, and the full model is extracted from it").

* **Input**: the sequence (docs/SEQUENCE_FILE.md) and its stream size; the full model's sparse
  TSDF parameters and the depth range, both from the object class's set by their one conversion
  (`src/scan/stage_parameters.hpp`, `fused_model::full`): the memory budget, the full voxel size,
  the truncation's c, k and μ_min, the far depth, and the depth range from `min_depth_m` to
  `max_depth_m`; and the caller's question, asked before each frame, whether to go on.
* **Each frame in the sequence's order** (`src/camera/sequence_reader.hpp`):
  * a frame with a pose is uploaded, its organized points are made within the depth range
    (docs/methods/organized-points.md), and it is integrated at that pose
    (docs/methods/sparse-tsdf.md, "Integration, and the weight of an observation"): the
    allocation of the blocks near its surface, the truncation and the weights are the live
    model's, at the full voxel size. No pyramid, normals, ray cast or registration: the pose is
    given. Until `refinement-optimizes-the-trajectory` the poses are those tracking gave, as the
    sequence holds them, bit for bit;
  * a frame of tracking loss is read and skipped: nothing of it is fused, as nothing of it
    entered the live model;
  * a frame marked as tracked by nothing ends the fusion as failed, with the reason that names
    it: a scan's sequence holds none (only the viewer writes such frames), and fusing the rest
    would hide that the file is not a scan's [?: a choice]. The frame tracking failed on, which
    ends a scan, is not in the sequence at all;
  * the caller's answer no ends the fusion as cancelled, before the frame is read: the scan
    interface asks it for a discard or a stop of the service while refining.
* **A sequence cut off**, as a writer that was stopped leaves it, gives every whole frame and then
  its early end (`sequence_reader::ended_early()`); the fusion takes those frames and ends done,
  saying that the sequence ended early. Any other failure of the reading ends it as failed.
* **A pool used up**: an integration skips a slot the pool had no block for and is done all the
  same (docs/methods/sparse-tsdf.md, "The pool and the memory budget"), so after the last frame the
  pool's usage is read once, and a coordinate that found no block, or an insertion the hash
  refused, ends the fusion as failed, with the counts: the full model would lack surface. A
  failed refinement is a value (the spec), and the scan stays unfinished.
* **Failed or cancelled** gives neither a model nor a trajectory, so that no caller keeps a part of
  either; the counts of the frames fused and skipped up to the end stay.
* **The extraction** (docs/methods/sparse-tsdf.md, "The extraction") runs twice: into a buffer of
  one point, which counts the surface's points, then into a buffer of exactly that count, so that no
  capacity is guessed, with every block and every crossing (`gca::extraction_choice`'s defaults).
  The points are copied to the host and converted to model points (`src/types/model_point.hpp`) by
  `gca::model_point_of()` (`src/fusion/sparse_tsdf.hpp`): the one conversion from the GPU's
  `surface_point` to the host's type, so that the types module needs no CUDA. They come in no stated
  order. A sequence of which no frame was fused gives no point, and the fusion is still done.
* **The trajectory as read**: each frame's index, timestamp and pose, or none for tracking loss,
  bit for bit as the sequence holds them, in its order (`src/types/trajectory_entry.hpp`).
* **Memory**: one sparse TSDF exists at a time (docs/methods/sparse-tsdf.md, "The pool and the
  memory budget"), so the live model's is freed before the full model's is requested; the full
  model's takes the object class's memory budget (the spec: "the budget also has to hold what
  refinement needs at the full voxel size"). Besides it the call holds a GPU frame and its
  organized points at the stream size, the buffer of points, 28 bytes a point on the GPU, and the
  points on the host, 28 bytes as copied and 28 as model points [S: `sizeof`, the static_assert of
  `surface_point`; 3 floats, 3 floats and 3 bytes of a model point, padded].

Its accuracy and time on the test's walk stand in "Accuracy" and "Time on the board".

## Constants and parameters

The solve takes one parameter (`gca::pose_graph_parameters`), the most iterations, 1 to 1000; the
tests name 100 [?: initial], refinement its own. The constants, each [?: initial] until measured on
sequences:

| constant | meaning | value |
|---|---|---|
| k | Huber's threshold on the normalized residual s | 3.548 |
| μ_0 | the damping's start, a share of H's diagonal | 1e-4 |
| ε_ω, ε_υ | the step under which the solve ends | 1e-6 rad, 1e-6 m |

## Accuracy

On graphs `tests/test_pose_graph.cpp` builds from the keyframes of a walk once around the tabletop
box (`tabletop_walk()`, 0.40 m to 0.50 m from the box): an edge between two keyframes carries their
known relative transform with an error put in, and Ω the standard deviations 1 mrad and 1 mm on each
axis unless stated; the start is the known walk with a drift put into each step, 0.15° and 1.5 mm in
the rings of 24 keyframes, a tenth of that in the graph of 300; the most iterations 100. The error is
the largest over the nodes of the turn and the camera centre's shift from the known pose
[M 2026-11-15: `tests/test_pose_graph.cpp`, and the same with each mutant named]:

| graph | start | iterations | end | tolerance |
|---|---|---|---|---|
| ring of 24, an edge to the next and one closing the loop, right edges | 1.34°, 23 mm | 4 | 1.9e-10°, 5.0e-12 m | 1e-8°, 1e-9 m; 6 iterations |
| the same, from the known poses | 0 | 1 | 1.6e-15°, 0 m | the same |
| the same ring, its edges off by 3 standard deviations, Ω correlated (below) | 1.34°, 23 mm | 4 | 0.262°, 5.86 mm | 0.3°, 7 mm |
| ring of 24, edges to 1, 2 and 6 ahead, right, and one wrong edge | 1.34°, 23 mm | 4 | 0.038°, 0.41 mm | 0.05°, 0.6 mm |
| 300 keyframes, edges to 1, 10 and 25 ahead, off by 1 standard deviation | 1.71°, 29 mm | 4 | 0.085°, 1.55 mm | 0.1°, 2 mm |
| ring of 24, right edges, Ω's eigenvalues spanning 1e8 (below) | 1.34°, 23 mm | 61 | 1.4e-8°, 3.9e-10 m | 1e-6°, 1e-8 m |

With right edges the minimum is the known poses, and the solve reaches it to double's rounding in
four iterations, the Gauss-Newton step's quadratic convergence where every residual ends at 0. With
noisy edges the minimum lies where the edges' errors put it, not at the known poses; there the test
checks that the solve found it: **stationarity**, the largest derivative of the robust objective
over every node and axis, by central differences with steps of 1e-6 of the objective as this
document states it, computed in the test apart from the module, is 9.6e-6 at the end against 1.27e4
at the start, 7.5e-10 of it; the tolerance is 2e-8 of it. The share does not depend on the step:
7.4e-10 to 7.5e-10 with steps of 1e-5, 1e-6, 1e-7 and 1e-8, and 9.8e-9 with 1e-4, where the
difference's truncation shows [M 2026-11-15: the test with each step].

**One wrong edge**, from keyframe 3 to keyframe 11 off by a turn of 5° and a shift of 54 mm, about
101 standard deviations, among 72 right edges: Huber's weight ends at 0.035, every right edge's at 1,
and the poses 0.038° and 0.41 mm off the known ones, the bounded pull of k standard deviations
against the right edges. Plain least squares, k taken to 1e300 (a mutant), leaves them 0.779° and
8.39 mm off, 20 times as far, with the wrong edge's weight 1.

**A wrong sign in an edge's Jacobian** (a mutant, J_i = +M(r)): every step is rejected and the
damping grows until a step lies under ε. Without the undamped step's check the solve then ended as
converged where it started, 1.34° and 23 mm off, after 8 iterations on the ring with right edges,
and on the noisy ring after 22 iterations, 1.43° and 24.7 mm off. With the check its undamped step,
as wrong as its damped one, is not small there, and every case ends with its 100 iterations used up,
the ring with right edges still 1.34° and 23 mm off, the noisy ring's largest derivative 0.91 of the
start's [M 2026-11-15: that mutant with and without the check].

**The rotation's inverse left Jacobian**, taken as I (a probe mutant), changes nothing the tests saw
while every Ω was the same on each turn's axis and held no correlation between a turn and a shift:
then J_l^-1(φ)^T Ω_ω φ = Ω_ω φ for the turns' block Ω_ω = c I, since [φ]x φ = 0 and [φ]x² φ = 0
[S: computed], so the term changes no edge's gradient, and the noisy ring's stationarity was 6.7e-12
of the start's with it and without. A registration's information is neither, so the noisy ring's
edges carry a correlated Ω: Ω = (S P S)^-1, S the standard deviations 0.5, 1 and 2 mrad and 2, 1 and
0.5 mm, P the correlations 0.4^|a - b| between axes a and b [?: a choice]. There the probe leaves
9.1e-8 of the start's derivative against the solve's 1.2e-9 with edges off by 1 standard deviation,
and 4.9e-7 against 7.5e-10 off by 3, which the test uses, so its tolerance of 2e-8 catches the
probe. The poses' errors agree with and without the term to the three digits printed: the term is
second order in the edges' errors.

**Rejected steps and an ill-conditioned Ω.** The drifted rings take no rejected step: none from
drifts of up to 8° and 80 mm a step, nor with one node turned by 160° from its known pose, each
converged at the known poses in 4 or 5 iterations [M 2026-11-15: a scratch copy of the module that
counts its rejections]. Every node turned by 90° or by 150° about an axis of its own, far beyond a
drift of tracking, ends with the iterations used up 179° and 160° off: another minimum, without a
rejected step [M 2026-11-15: the same]. Rejections come with an Ω whose eigenvalues span far, one
direction turned against every axis much surer than the others (the test's: five directions of 1
mrad or 1 mm, one 1e4 times as sure, a span of 1e8): Marquardt's D = diag(H) then holds that
direction's curvature in every diagonal entry and damps the weak directions far more than their own
curvature, the accepted steps shrink μ until a step overshoots and is rejected. In the test's case
the 12 rejections come one at a time but for one pair in a row, at iterations 32 and 33, so that ν's
doubling acts once: without it the ring converged after 59 iterations instead of 61, 7.3e-8° and
2.1e-9 m off instead of 1.4e-8° and 3.9e-10 m, within the test's 1e-6° and 1e-8 m; the doubling
changes only the speed there, and the test does not tell it apart, while it catches a damping that
does not grow on a rejection [M 2026-11-15: a mutant of each]. The ring with right edges, by span,
with the undamped step's check [M 2026-11-15: the same scratch copy, which also counts the checks;
the spans 1e8 and 1e12 are the test's cases]:

| span | end | iterations | rejected | undamped step not small | off the known poses |
|---|---|---|---|---|---|
| 1 | converged | 4 | 0 | 0 | 1.9e-10°, 5.0e-12 m |
| 1e6 | converged | 15 | 0 | 0 | 2.7e-8°, 7.4e-10 m |
| 1e8 | converged | 61 | 12 | 0 | 1.4e-8°, 3.9e-10 m |
| 1e9 | iterations used up | 100 | 10 | 0 | 0.0135°, 0.36 mm |
| 1e10 | iterations used up | 100 | 7 | 0 | 0.96°, 22 mm |
| 1e12 | iterations used up | 100 | 1 | 2 | 2.08°, 23.9 mm |
| 1e14, 1e16 | iterations used up | 100 | 97, 96 | 98 | 2.66°, 23.8 mm |

**What the undamped step's check guards and costs.** Without it, from a span of 1e12 on, the solve
ended converged after 3 iterations 2.66° and 24 mm off: the damped step lay under ε at μ = 3.7e-6
while the undamped step was still 0.046 of a rad or a m, against 5e-12 and 3.9e-10 where the end is
right [M 2026-11-15: the same scratch copy before the check, the undamped step solved where the end
was decided; the test's mutant without the check]. With it those ends are the iterations used up, an
honest failure, and no other end changed. It costs one more factorization and solve each time a
damped step is small: once, at the end, in every graph above that converges; in the graph of 300
keyframes a fifth of the solve's time (docs/HARDWARE.md, "Refinement"). It does not make the spans
of 1e9 and more converge: the damping by H's diagonal stays as it is. An Ω of a registration likely
spans far less [?: not measured on an edge]: its centred system's eigenvalues lie within 0.01 to 2
(docs/methods/registration.md), and moving it to the camera's origin multiplies the span by about
(|c| / ρ)², the centroid's distance over the points' spread. How refinement's edges span is
refinement-optimizes-the-trajectory's to measure.

**The fusion of the sequence** [M 2026-11-16: `tests/test_sequence_fusion.cpp`, frames of the
synthetic source without noise at 640x400 with the camera's intrinsics, depth unit 1 mm, the
tabletop object class's full model (1 mm voxels, c 0.01 /m, k 3, μ_min 4 voxels, the depth range
0.25 m to 0.6 m) in a sparse TSDF of 273 MB]: the tabletop box on its plane from 24 known poses
once around it, `tabletop_walk()` with 15° of azimuth between poses, and after every fourth a
frame of a wall 0.35 m before the camera marked as tracking loss, 30 frames in all. The 24 frames
with a pose were fused and the 6 walls skipped; the trajectory came back bit for bit. The full
model's 863311 points lie at a median of 0.0754 mm from the nearest known surface, the 99th
percentile at 0.675 mm and the largest at 8.77 mm; 0.26 % of them lie farther than 2 mm, all
within 5 mm below the box's top edges and outside its sides: the band of the truncation behind a
convex edge, which the extraction makes points of (docs/methods/sparse-tsdf.md, "The extraction").
No point lies at a wall. Every normal is of unit length within 1.34e-7. The walk cut off within
its 14th record gives its 13 whole frames and the same set of points, bit for bit, as a sequence of
just those 13 frames. Four walls at 0.55 m, fused from poses 2 m apart into a budget of 140 MB
(33123 blocks), used it up: 125233 coordinates found no block, and the fusion failed with those
counts.

## Time on the board

The graph of 300 keyframes above is solved in the time docs/HARDWARE.md, "Refinement", states; so
is the fusion of the test's walk of "Accuracy".

## What the prototype did

CUDA_3D optimizes no trajectory after a scan: its tree holds no pose graph [S: grep of `src/`,
`apps/` and `tests/` for a pose graph or a loop closure, 2026-11-14]. Nothing is carried.
