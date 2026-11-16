# The sparse TSDF

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The model of a running scan, and of refinement's fusion at the full voxel size: a truncated
signed distance field (TSDF) kept only near the surfaces, in blocks of voxels that a hash on the
GPU finds by their coordinates (docs/adr/0004-sparse-tsdf-model.md). It has three operations:
integrate a frame at its pose, allocating the blocks near its surface first; ray cast from a
pose, giving the surface the registration takes; extract the colored points with normals that
the full model is made of.

Built so far: the hash of blocks and the pool they come from (`src/fusion/sparse_tsdf.*`, its
test `tests/test_sparse_tsdf_blocks.cpp`), the allocation near a frame's surface (its test
`tests/test_sparse_tsdf_allocation.cpp`), and the integration of a frame and the extraction of the
surface's points (`sparse_tsdf::integrate()` and `sparse_tsdf::extract()`, their test
`tests/test_sparse_tsdf_fusion.cpp`), and the ray cast (`sparse_tsdf::ray_cast()` into a
`gca::model_surface`, its test `tests/test_sparse_tsdf_ray_cast.cpp`).

The method is the volumetric fusion of Curless and Levoy [S: B. Curless, M. Levoy, "A
Volumetric Method for Building Complex Models from Range Images", SIGGRAPH 1996], in the
projective, truncated form of KinectFusion [S: R. A. Newcombe et al., "KinectFusion: Real-time
dense surface mapping and tracking", ISMAR 2011, sections 3.3 and 3.4], on the hash of voxel
blocks of Nießner et al. [S: M. Nießner, M. Zollhöfer, S. Izadi, M. Stamminger, "Real-time 3D
Reconstruction at Scale using Voxel Hashing", ACM TOG 32(6), 2013, sections 4 to 7], whose hash
and allocation are replaced here as "The hash of blocks" says. This tree's code is written from
these papers; no reconstruction library's code is read or copied for it (docs/adr/0002, 0004).

## Space, voxels and blocks

The TSDF lives in the models' frame, the camera frame of the first fused frame
(docs/SCAN_FILES.md, "The models' frame"): x right, y down, z forward, in m. A pose maps the
camera frame of a frame into it (`src/types/camera_pose.hpp`).

* **The voxel size** v, in m, is the TSDF's one length: the live model's TSDF's or the full model's TSDF's
  of the object class (GLOSSARY.md, "Voxel size"; `src/scan/object_class.cpp`). Voxel g, a
  triple of integers, is the cube [g, g + 1) v; its values stand for the cube's centre
  (g + ½) v. The point p lies in voxel floor(p / v).
* **A block** is 8 x 8 x 8 voxels [?: a choice, the block of Nießner et al. 2013, section 4; not
  measured against another size here]: block b holds the voxels 8 b + (i, j, k), i, j, k = 0 to
  7, and is the cube [8 b, 8 b + 8) v. The point p lies in block floor(p / (8 v)). A block's
  edge is 8, 16, 32 and 64 mm at 1, 2, 4 and 8 mm voxels; it holds 512 voxels, 4096 bytes
  [S: computed]. Inside a block, voxel (i, j, k) is element i + 8 j + 64 k.
* **A block's coordinates** b = (x, y, z) are each in [-2^20, 2^20 - 1], 21 bits, so that a
  block's key is one 64-bit word ("The hash of blocks"). That spans ±8.4 km at 1 mm voxels
  [S: computed: 2^20 x 8 mm], far beyond the depth range; a coordinate outside it comes only
  from a broken pose and is refused, counted, and never stored.

## What a voxel holds

Eight bytes, the size of Nießner et al.'s voxel [S: 2013, section 4]:

| field | type | meaning |
|---|---|---|
| distance | IEEE half float | the signed distance D, in voxels: > 0 in front of the surface, towards the cameras that saw it; < 0 behind it |
| weight | unsigned 16 bit | W, the number of observations fused, at most W_max; 0: unobserved |
| color | 3 x unsigned 8 bit | red, green, blue, the frame's color order |

One byte is left over to the alignment of 8 bytes, so that a voxel is read and written in one
64-bit access. A voxel with W = 0 is **unobserved**: its distance and color mean nothing, and the
extraction and the ray cast skip it. The pool's blocks start with every byte 0, so every voxel of
a block just handed out is unobserved, with distance and color 0; the integration's running
average takes them with the weight 0, so that its first observation gives d and c themselves
[?: a choice: no branch for a first observation; the request's clearing is what it starts from].
The request clears the voxels because CUDA does not: "The memory is not cleared" [S:
cuda_runtime_api.h, cudaMalloc, CUDA 12.6]. On this board cudaMalloc gave zeroed memory each
time, also the memory the program had just filled and freed, at the same address [M 2026-11-15:
a scratch program, 268 MB, 2 MB and 4 KB], so no test through the module's interface sees a
request that skips the clearing: its mutant survives `tests/test_sparse_tsdf_fusion.cpp`.

The distance is kept in voxels, not in m, so that its precision follows the voxel size. A half
float has 11 significant bits [S: IEEE 754-2008, binary16]: between 0.5 and 1 voxel its step is
2^-11 voxel, 1 µm at 2 mm voxels; at the largest distance it holds, the largest truncation
("The truncation distance"), 270 mm, 34 voxels at 8 mm or 68 at 4 mm, the step is 2^-5 or
2^-4 voxel, 0.25 mm in both [S: computed]. KinectFusion found 6 bits enough for the SDF value
[S: Newcombe et al. 2011, section 3.3].

## The truncation distance

A depth measurement z_m holds the surface only up to its noise, so the TSDF takes a
measurement's distances only within ±μ of it, the truncation distance, which "is scaled to
ensure that a surface measurement (zero crossing in the SDF) is represented by at least one
non truncated voxel value [...] either side of the surface" and whose "support is increased
linearly with distance from the sensor center to support correct representation of noisier
measurements" [S: Newcombe et al. 2011, section 3.3]; Nießner et al. adapt it "based on the
variance of depth to compensate for larger uncertainty in distant measurements" [S: 2013,
section 5]. Here it follows the camera's depth noise at the measured depth z_m:

    μ(z_m) = max(μ_min, k σ(z_m)),   σ(z) = c z²

* **The noise law.** A stereo camera triangulates depth from disparity d as z = f b / d, focal
  length f in px, baseline b, so a disparity error σ_d gives σ(z) = z² σ_d / (f b): the noise
  grows with the square of the depth [S: the derivative of z = f b / d; Khoshelham and Elberink,
  "Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications", Sensors 12(2),
  2012, which derives the same law for the Kinect's triangulation]. The Gemini 2 L is an active
  stereo camera with a baseline of 100 mm (docs/HARDWARE.md, "What the camera is and reports").
* **c = 0.01 /m** [?: the datasheet's bound, until the camera's depth noise is measured]: the
  datasheet gives a depth accuracy of 2 % or better at 2 m, the RMS distance to a fitted plane
  at 1280x800 (docs/HARDWARE.md), so σ(2 m) <= 40 mm and c <= 40 mm / (2 m)² [S: computed]. It
  is a bound and holds the plane's flatness besides the noise, so μ is likely too large far
  away; no measurement of the noise on this board exists yet: `depth_noise` takes c from a
  sequence of a plane seen obliquely by the camera at rest (docs/methods/depth-noise.md), not yet
  run on the camera. At 640x400 the datasheet gives nothing [?].
* **k = 3** [?: initial]: three standard deviations, so that the true surface lies inside ±μ of
  a measurement for 99.7 % of the measurements of a Gaussian noise [S: the normal distribution].
* **μ_min = 4 voxels** [?: initial]: at least one voxel with a distance below the truncation on
  each side of a zero crossing, along any axis and diagonal, with room to spare; its half is the
  ray cast's step ("The ray cast").

The sparse TSDF takes the voxel size v, the rule's c, k and μ_min, and the far depth z_max, the
farthest depth a frame contributes, as its parameters when it is requested
(`gca::sparse_tsdf_parameters`) and keeps them for every operation, so that the allocation, the
integration and the ray cast use the same μ. Each object class's set holds c, k and μ_min, the
values above in both (`src/scan/object_class.cpp`), with its voxel sizes and its far depth, and
a scan fills them from the set, v the live model's or the full model's
(`src/scan/stage_parameters.hpp`); the stage's tests name the values above. The request refuses a value outside its range, so that a
mistaken value, a unit's factor of 1000 among them, is a refusal by name [?: a choice, the
user's, 2026-11-14, each for the reason given]:

* **v from 0.5 mm to 16 mm**, half the object classes' finest voxel size and twice their
  coarsest. It is not narrowed to the object classes' sizes: the live models' sizes are still to
  be set by measurement, and the check of one frame below limits every size at its budget.
* **c above 0 up to 0.02 /m**, twice the datasheet's bound: the measured c is likely below the
  bound, which holds the plane's flatness besides the noise, and twice it leaves room for a camera
  noisier than its datasheet. At the object classes' budget, 2.0 GB, every full model passes
  the check of one frame over the whole range, the large object's at 0.02 /m with N at 0.80 of
  the room; so does every budget from 1.11 GB, where S becomes 2^21, up to the board's memory.
  Below that the full models pass only part of the range, at 1.0 GB the large object's up to
  0.0132 /m [S: computed, "One frame's blocks"], so a smaller budget can refuse a measured c.
* **k from 1 to 5**: five standard deviations hold all but 6e-7 of a Gaussian
  noise's measurements [S: the normal distribution], and a wider band only lets nearer surfaces
  interfere.
* **μ_min from 1 voxel**, the least the zero crossing needs, **to 8**, twice its starting value:
  the ray cast's step, μ_min / 2, stays at most 4 voxels, half a block, and so does the thickness
  of an object the step can step over ("The ray cast").
* **z_max from 0.25 m to 10 m**, the camera's depth range [S: datasheet, docs/HARDWARE.md, "What
  the camera is and reports"].

The starting values lie inside every range: v of 1, 2, 4 and 8 mm, c = 0.01 /m, k = 3, μ_min = 4
voxels and z_max of 1.0 and 3.0 m. The request also refuses a combination of the parameters with
the memory budget under which one frame could fill the hash ("One frame's blocks"). Within the
ranges a band can still cross far more blocks than the starting values give; the allocation caps
that ("The allocation near a frame's surface").

What the rule gives, in mm [S: computed]:

| voxel size | μ_min | μ at 0.3 m | 0.5 m | 1.0 m | 2.0 m | 3.0 m | 3σ passes μ_min at |
|---|---|---|---|---|---|---|---|
| 1 mm (tabletop, final) | 4 | 4 | 7.5 | 30 | | | 0.37 m |
| 2 mm (tabletop, live) | 8 | 8 | 8 | 30 | | | 0.52 m |
| 4 mm (large, final) | 16 | | 16 | 30 | 120 | 270 | 0.73 m |
| 8 mm (large, live) | 32 | | 32 | 32 | 120 | 270 | 1.03 m |

(Blank: outside the object classes' depth ranges of 2026-11-14, 0.30 m to 1.0 m and 0.4 m to
3.0 m; since 2026-11-15, evening, the ranges are 0.25 m to 0.6 m and 0.25 m to 3.0 m
[?: initial, `src/scan/object_class.cpp`], and μ at 0.25 m is μ_min at every voxel size.) The cost
of a wide band: two surfaces closer than μ along a camera's ray interfere, the nearer one's
negative distances reaching the farther one, and an object thinner than μ is seen through from
both sides. At 3 m that is 270 mm with the datasheet's bound, which is why the bound is to be
replaced by a measurement.

## Integration, and the weight of an observation

A frame with its organized points and validity mask (docs/methods/organized-points.md), its
color picture and its pose updates the voxels of the blocks the allocation listed for it ("The
allocation near a frame's surface"), one GPU thread for each voxel, one thread block for each
voxel block [S: Nießner et al. 2013, section 6]. For voxel g, with centre c_g in the models'
frame:

1. Its centre in the camera frame, q = R^T (c_g - t) for the pose (R, t), and the pixel
   nearest to its projection, (u, v) = (round(fx q_x / q_z + cx), round(fy q_y / q_z + cy)), by
   the one pinhole convention (docs/HARDWARE.md, "Pixel convention"). A nearest pixel, not an
   interpolated depth, "to prevent smearing of measurements at depth discontinuities"
   [S: Newcombe et al. 2011, section 3.3]. No update when q_z <= 0, the pixel lies outside the
   frame, it is invalid, or its depth lies beyond z_max [?: a choice]: z_max is the farthest
   depth a frame contributes, as in the allocation ("The allocation near a frame's surface",
   step 1), and it bounds an observation's distance, |d| <= mu(z_max) / v, at most 20000 voxels
   within the parameters' ranges (mu at most 5 * 0.02 * 10² = 10 m, v at least 0.5 mm), inside
   the largest half float, 65504 [S: computed; IEEE 754-2008, binary16].
2. η = z_m - q_z, the measured depth z_m of that pixel minus the voxel's, along the optical axis
   (KinectFusion found "no considerable difference in using SDF values computed using distances
   along the ray or along the optical axis" [S: 2011, section 3.3]). No update when
   η < -μ(z_m): the voxel lies behind the surface, unseen. Otherwise the observation is
   d = min(η, μ(z_m)) / v, in voxels, and the color c is the frame's color at that pixel.
3. The running average of Curless and Levoy with the weight 1 for each observation, the weight
   limited to W_max [S: Newcombe et al. 2011, equations 11 to 13: "simply letting W_Rk(p) = 1,
   resulting in a simple average, provides good results"]:

       D ← (W D + d) / (W + 1),   C ← (W C + c) / (W + 1),   W ← min(W + 1, W_max)

   The color is averaged with the same weight as the distance; D is rounded to the nearest half
   float, each of C's components to the nearest integer, a tie to the even one in both
   [?: a choice: IEEE 754's default rounding, CUDA's `_rn` conversions].
4. The voxels of the blocks on the frame's list, one thread for each voxel and one thread block
   of 512 for each block; a slot on the list whose key has no block, refused for a used-up pool,
   is skipped. The thread blocks are a fixed grid of 24, as many as the GPU holds at once, each
   taking every 24th entry of the list, so that the kernel reads the list's count on the GPU and
   the frame's fusion reads nothing back and waits for nothing (the grid's measurement:
   `src/fusion/sparse_tsdf.cu`, `voxel_kernel_grid`; its time: docs/HARDWARE.md, "The
   integration of a frame"). `integrate()` allocates the frame's blocks first, so that the list
   is always the frame's own [?: a choice].

* **W_max = 64** [?: initial]. Once a voxel holds W_max observations the average becomes a
  moving one: a new observation counts 1/65. A limit of the color's rounding follows: with
  W = 64 a new color moves a component only when it differs from the stored one by 33 levels or
  more [S: computed, the rounding of (c - C) / 65], so a voxel's color is mostly that of its
  early observations. The integration's tests measure the colors against the texture.
* **What the test measures** (`tests/test_sparse_tsdf_fusion.cpp`), on walls facing the camera at
  whole millimetres, so that every depth is exact, at 2 mm voxels: one frame gives its wall
  exactly, its normals along the optical axis and its color; a wall 4 mm behind, fused once
  after it, puts the surface halfway, within 0.034 µm, and the color at the mean, a tie rounded
  to even; 64 frames of the first wall and then 64 of the second put it at 1 - (64/65)^64 = 0.629
  of the 4 mm, within 0.5 µm, against 0.5 without the limit, and the colors where the rounded
  recursion puts them [M 2026-11-15].
* **Not built without a measured benefit** (AGENTS.md §7 rule 6): a weight that falls with
  depth or with the angle to the surface (KinectFusion's text weights by cos θ / R before it
  settles on 1; Nießner et al. give nearer measurements more weight, and recent colors more
  than old ones), and the update of every allocated block in the view rather than only the
  frame's own band, which Nießner et al. use to remove blocks of outliers.

## The hash of blocks

Open addressing with linear probing [S: D. E. Knuth, *The Art of Computer Programming*, vol. 3,
2nd ed., 1998, section 6.4, "Hashing", algorithm L], on the GPU, its insertions made by an
atomic compare-and-swap on the key.

* **The key** of block b is one 64-bit word: (x + 2^20) · 2^42 + (y + 2^20) · 2^21 + (z + 2^20),
  each term below 2^21, so that two coordinates in range never share a key and the top bit is
  0. The empty slot holds the key with every bit 1, which no coordinate gives.
* **The slots**: S slots, S a power of two and at least four times the pool's blocks ("The pool
  and the memory budget"). Two arrays: the keys (8 bytes each) and the blocks' numbers (4 bytes
  each; the number with every bit 1 is "no block"). A probe reads only the keys, 16 to a
  128-byte cache line [S: computed].
* **The home slot** of a key is the 64-bit finalizer of MurmurHash3 applied to it, its low bits
  taken: h = fmix64(key) mod S [S: A. Appleby, MurmurHash3, `fmix64`, in SMHasher, placed in
  the public domain]. The finalizer spreads every bit of the key over the low bits. The
  multiply-and-xor hash of Teschner et al., which Nießner et al. use with a prime per axis
  [S: 2013, section 4], is not taken: with S a power of two, the low k bits of x · p depend only
  on the low k bits of x [S: arithmetic modulo 2^k], so the slot would see only the low bits of
  each coordinate.
* **Lookup**: the slots h, h + 1, ... (mod S) in turn, until the key (its block's number, or
  none when it has no block), the empty key (none), or S slots (none).
* **Insertion**, by many threads at once, each with one coordinate, in one kernel: each tries
  the slots in the same order, h, h + 1, ..., reading each slot's key by a relaxed atomic load
  first [S: libcu++, `cuda::atomic_ref`]: the key itself ends the insertion and another key
  passes the slot, as the swap's answers below do, without the swap. An empty slot takes
  atomicCAS(slot, empty, key) [S: CUDA C++ Programming Guide, "Atomic Functions", atomicCAS on
  64-bit words], which returns:
  * the empty key: this thread has put the key there and is the one thread that inserts it. It
    takes the next block's number n with atomicAdd on the pool's count of claims; n below the
    pool's blocks is written as the slot's block, otherwise the slot keeps "no block" and the
    key counts as refused for a full pool.
  * the key itself: an earlier call or another thread put it there; nothing to do.
  * another key: the next slot.

  After S slots without room, the coordinate is refused for a full hash and counted.

**Why no key is lost or doubled.** A slot changes once, from empty to a key, and never back.
Every thread that inserts key K tries the same slots in the same order, and passes a slot only
when it holds another key, by the load or by the swap, which it then holds for good. Let s be
the first slot of that order that ever holds K: every slot before s held another key when the
first of these threads passed it, so every later thread passes it too, reaches s and finds K
there or puts it there; the swap on s succeeds for exactly one of them. So K stands in exactly
one slot, gets at most one block's number, and a lookup, which tries the same order, finds it.

**What is wait-free, and what is not.** No thread waits for another and none holds a lock: an
insertion ends after at most S slots, a load and at most one swap each, whatever the other
threads do, a bound on its own steps (wait-free, not only lock-free). What is not atomic: the
block's number is written by a plain store after the swap, so within the kernel that inserts,
another thread can find the key before its number and read the slot's cleared "no block".
Lookups and every reading of a block's number therefore run in a kernel queued after the
insertions on the same stream, which sees all their writes [S: CUDA C++ Programming Guide,
"Streams": the work of one stream runs in order]; the TSDF is used from one stream at a time.
The insertion and the lookup are one device function each (`src/fusion/block_hash.cuh`), which
every kernel that inserts or finds a block calls. Nießner et al. lock a bucket for an insertion
instead, and "all other allocations for the same bucket are staggered until the next frame is
processed" [S: 2013, section 4.2]: a block is then missing for a frame. Here none is.

**Probes.** At a load α of keys per slot, linear probing needs about (1 + 1 / (1 - α)) / 2
probes to find a key and (1 + 1 / (1 - α)²) / 2 to find it absent [S: Knuth, section 6.4]: at
the 2.0 GB budget's full pool, α = 0.23, that is 1.15 and 1.34 [S: computed].

**What is the same in every run.** The set of coordinates that get a block, when the pool holds
them all, and every count. What is not: which slot a key takes when keys collide, and which
block's number it gets, both the order in which the threads' atomics land; and, with a pool too
small, which coordinates get the blocks. A stage that reads the blocks in slot order, such as
the extraction, gives the same points in another order.

## The pool and the memory budget

The pool is one array of blocks in device memory; block n holds voxels 512 n to 512 n + 511.
Blocks are numbered in the order they are claimed and never freed during a scan: the scan's
pool ends with the scan. (Nießner et al. free blocks far from the surface and stream blocks
between GPU and host [S: 2013, sections 6 and 8]; neither is built here.)

**The memory budget** is all of the sparse TSDF's device memory: the pool's blocks (4096 bytes
each); the hash's keys, blocks' numbers and stamps of the last frame (16 bytes a slot); the
frame's list of slots (4 bytes a block, "The allocation near a frame's surface"); and four
32-bit counts, of claims, of refusals, of the list and of the rays not walked. The object class
sets it (`src/scan/object_class.cpp`): 2.0 GB, 2 · 10^9 bytes, for both object classes until a
scan's memory is measured [?: the user's choice, 2026-11-14]. The pool holds the largest number
of blocks B for which

    4100 B + 16 S + 16 <= budget,   S = the least power of two >= 4 B

so at 2.0 GB, B = 479620 blocks and S = 2097152 slots, 1999996448 bytes, of which the keys and
numbers take 25.2 MB and the stamps and the list 10.3 MB [S: computed]. B is at most 2^29, so that
S fits 32 bits. The stamps and the list are the allocation's ("The allocation near a frame's
surface"). S is four times B, not twice, so that one frame of the full model's TSDF fits the hash ("One
frame's blocks") [?: a choice, the user's, 2026-11-14]: at 2.0 GB that costs 4092 blocks, 0.85 %
of the pool, against S at twice B [S: computed].

**One sparse TSDF exists at a time.** Refinement's, at the full voxel size, is requested from a
pool of its own after the live model's is released, within the same budget.

**What the budget holds** [S: computed]: a surface facing the camera at depth z crosses about
2 μ(z) / (8 v) + 1 blocks along each column of blocks, so a square metre of it takes about
(2 μ / (8 v) + 1) / (8 v)² blocks. With c at the datasheet's bound, 2.0 GB holds:

| voxel size | at 0.3 m | at 0.5 m | at 1.0 m |
|---|---|---|---|
| 1 mm (tabletop, final) | 15 m² | 11 m² | 3.6 m² |
| 2 mm (tabletop, live) | 61 m² | 61 m² | 26 m² |

A surface seen at a slant, or from several sides, takes more.

Everything is requested from the GPU pool that closes (`src/gpu/gpu_pool.hpp`) before the scan
starts, and set on the scan's stream: the keys to empty, the blocks' numbers to none, the
stamps to every bit 1 (no frame), the counts and every voxel to 0; the list is written before it
is read. After that no operation of the sparse TSDF allocates.

**A pool too small** is reported as a value. Reading the usage gives the blocks the pool holds,
the blocks in use, the coordinates refused for a full pool and those refused for a full hash or
out of range, the last allocation's slots on the frame's list and those past its room, and the
rays not walked; the share in use is the blocks in use over the blocks held, and the pool is
used up when the two are equal. A reading that fails gives no usage, so it is never taken for a
used-up pool. The reading copies the counts and waits for the stream, and so for all the work
queued on it before; in the per-frame path it comes after the registration's own readback, which
has waited for that work already, so that it costs only its copy of 16 bytes. A coordinate
refused for a full pool keeps its slot without a block, so the count of claims is the count of
distinct coordinates the hash took, and the same set of insertions gives the same counts in
every run; the hash fills only when more distinct coordinates than S arrive, which the request
rules out for one frame ("One frame's blocks").

**One frame's blocks.** Once the pool is used up, a coordinate refused for it keeps its slot, so
the hash's room for one frame's new blocks is S - B; a frame that filled it would make each later
insertion, and each lookup of an absent block, probe all S slots, and a kernel could run for
minutes. A scan is to end on the frame that uses its pool up [?: the scan's rule, not built yet],
so no later frame adds to them.
The request therefore refuses a combination of the parameters and the budget under which one
frame could touch as many distinct blocks as S - B or more. Its model of a frame: a surface facing
the camera that fills the picture at the largest stream size, 1280x800, with the camera's focal
lengths there, fx = 612.37 and fy = 612.29 px [M 2026-11-13: docs/HARDWARE.md, "What the camera
is and reports"] (640x400 sees the same field), every pixel at the far depth z_max, at any pose.
Its bands lie in the slice of the camera's frustum between the depths z_max - μ and z_max + μ,
μ = μ(z_max), inside a box of the camera frame of w = 1280 (z_max + μ) / fx by
h = 800 (z_max + μ) / fy by 2 μ. A block of edge e = 8 v that meets the slice lies inside that
box widened by the cube of ±e of the models' frame, which reaches at most √3 e along each axis of
the camera frame, whatever the pose; the blocks are disjoint, so the frame touches at most

    N = (w + 2√3 e) (h + 2√3 e) (2 μ + 2√3 e) / e³

blocks [S: geometry]. The request refuses N >= S - B and grants N < S - B, so that one frame
touches fewer blocks than the room and an empty slot is left. N grows with the depth, so z_max is
the model's worst depth, and a frame of the model at any nearer depth touches fewer. Float's
rounding moves a segment's end by micrometres, which can add a block only where a band grazes a
border, far fewer than N's margin over the count below. As S - B shrinks while B grows within one
S, a budget one block larger can be refused where a smaller one is granted, until B passes S / 4
and S doubles.

| object class, model | v | z_max | N | blocks the rays' bands cross |
|---|---|---|---|---|
| tabletop, live | 2 mm | 1.0 m | 87200 | 54916 to 57269 |
| tabletop, final | 1 mm | 1.0 m | 513102 | 346482 to 379612 |
| large, live | 8 mm | 3.0 m | 92152 | 61437 to 64889 |
| large, final | 4 mm | 3.0 m | 604739 | 444257 to 452085 |

(z_max the object classes' far depths of 2026-11-14; the tabletop object's is 0.6 m since
2026-11-15, evening, where N is 21381 for the live model and 103746 for the full one [S: computed,
`tests/test_sparse_tsdf_allocation.cpp`'s bound]. The starting values of the
truncation's rule; N [S: computed], rounded up; the last column a
count of the distinct blocks met by the bands of the 1024000 pixels' rays, each sampled every
eighth of a block, at three poses: the identity, a turn of 30° about x and one of 54.7° about
(1, -1, 0) [S: computed, a CPU count].) At 2.0 GB the room is 1617532 blocks, so every model is
granted, the full models' with 3.15 and 2.67 times their N [S: computed]. A frame of the full model's
TSDF of this kind takes 72 % to 79 % of its pool for the tabletop object and 93 % to 94 % for the large
object [S: computed, the counts over B]. The allocation's test measures N's margin at its edge:
a wall at 3.119 m filling 1280x800 at 8 mm voxels, N = 105778.4 against
a room of 105779, touches 68321 blocks at the identity and 74437 turned by 54.7° about
(1, -1, 0), 1.55 and 1.42 times fewer [M 2026-11-14, `tests/test_sparse_tsdf_allocation.cpp`].
A point past z_max is not walked ("The allocation near a frame's surface", step 1), so the model's
depth holds whatever range the points were made with.

What the model does not cover: a surface seen at a slant, whose band's blocks along its surface
grow about as 1 / cos θ of the facing surface's while the band's volume does not; surfaces at
several depths, each edge between them adding the sides of their bands; and a camera of a wider
field of view. Such a frame can touch more than N; how far into the room's margin over N it
reaches is not measured [?].

**Nothing is written outside the pool.** The block numbers handed out are 0 to B - 1, each to
one coordinate; a stage writes a voxel only of a block whose number it found by a lookup, and
the insertion writes only the slots, the blocks' numbers and the counts.

## The allocation near a frame's surface

Before a frame is integrated, every block that its band of ±μ passes through is found or
inserted [S: Nießner et al. 2013, section 5: "For each input depth sample, we instantiate a ray
with an interval bound to the truncation region. [...] we use DDA [Amanatides and Woo 1987] to
determine all the voxel blocks that intersect with the ray"]:

1. For each valid pixel with depth z_m, the segment of its ray between the depths z_m - μ(z_m)
   and z_m + μ(z_m) (docs/methods/organized-points.md, "Deprojection", at each depth), carried
   into the models' frame by the pose. The pixel's organized point p = z_m r lies on the ray, so
   the segment runs from p (z_m - μ) / z_m to p (z_m + μ) / z_m; no intrinsics are needed. The
   pose is rounded to float, its quaternion normalized; a pose that is not finite in float, or a
   quaternion of zero, is refused. A pixel deeper than z_max is not walked and is counted
   (`rays_not_walked` of the usage): the sparse TSDF guards its own bound of one frame ("One
   frame's blocks"), while the organized points' far depth is the caller's filter, which the
   sparse TSDF neither sees nor checks.
2. The blocks the segment passes through, in order, by the grid traversal of Amanatides and
   Woo [S: J. Amanatides, A. Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing",
   Eurographics 1987] on the grid of blocks, each inserted into the hash. Each axis counts its
   steps from the start's block to the end's, |floor(end) - floor(start)|, and the traversal
   takes the axis whose next border comes first among those with a step left, so that float's
   rounding cannot carry it past the end's block or end it early: it visits exactly the sum of
   the steps plus one block. A segment of length L blocks visits at most 1 + Σ⌈|Δ_i|⌉ < √3 L + 4
   blocks, Δ its extent along each axis. With the starting values, at the object classes'
   farthest depths and the camera's longest ray, |r| = 1.591 at either stream size, that is 14
   blocks at 2 mm voxels and 27 at 8 mm (the live models), 24 at 1 mm and 50 at 4 mm (the final
   models) [S: computed, the most over the segment's directions]. **The cap**: a thread walks at
   most 64 blocks [?: a choice, above the 50]; a ray whose band crosses more, or whose segment
   has an end that is not finite in float, is not walked and is counted (`rays_not_walked` of
   the usage), so that no thread runs long whatever the parameters within their ranges and
   whatever the frame, one the bound of one frame does not cover included, such as the picture
   of a far wider field of view in the allocation's test. Before
   the count, each end is clamped to one block past the hash's range on each side, so that it
   converts to an integer; an end beyond the range, which only a broken pose gives, then lies in
   a block the hash refuses, and the in-range part of such a segment may differ from the true
   one's.
3. The list of the frame's blocks, each block once, which the integration runs over: each
   slot keeps the number of the last frame that touched it, exchanged atomically after a relaxed
   load finds another; the thread whose exchange changes it appends the slot to the list. The
   two loads spare the atomics for the blocks a frame finds held, most of them (their times:
   docs/HARDWARE.md, "The per-frame path's stages"). Each allocation is a frame of its own,
   numbered by the sparse TSDF's count of allocations from 0, so that allocating the same frame
   again lists its slots again; the stamp with every bit 1 means "no frame", which the count
   reaches after 2^32 - 1 allocations, 4.5 years at 30 fps [S: computed]. The list's count is
   set to 0 before each allocation. The list holds B slots, as many as the pool has blocks; a
   frame that touches more distinct slots, which only happens once the pool is used up, has the
   rest counted, not written. Reading the usage gives both counts.

Between the rays: a block that the band crosses but no ray does is missed. Neighbouring rays are
z / fx apart: at 640x400 (fx = 306.18 px [M 2026-11-13: the camera's intrinsics at 640x400, as
`tests/test_organized_points.cpp` takes them]) 3.3 mm at 1 m and 9.8 mm at 3 m, at 1280x800 1.6
mm and 4.9 mm, against a block's edge of 16 mm at 2 mm voxels and 64 mm at 8 mm [S: computed],
so only a block grazed at a corner can be missed. The allocation's test bounds such a miss: a
point of the band between the rays, anywhere from one end of the band to the other, in a cell of
four valid pixels, lies in an allocated block or within h of one, h its distance from the
nearest of the four pixels' segments, each narrowed by the depth's rounding, whose points all
lie in allocated blocks [S: geometry]. On the tabletop scene's two frames at 2 mm voxels and a
large box's frame at 8 mm, 640x400, of 19.3 million points between the rays, 17 at three
positions in each cell, every one lies within its bound [M 2026-11-14]. On one surface the
misses lie only at the band's ends, |η| = μ: 21, 0 and 171 points, at most 0.21, 0 and 1.21 mm
from an allocated block. At an edge between two surfaces a sliver that no pixel's ray meets can
be missed whole, its surface included: 4, 46 and 3 points, at most 0.90, 2.68 and 2.47 mm from a
block, the nearest at |η| = 0.125 μ, 0 and μ [M 2026-11-14]. None of 6.5 million points of the
band on the pixels' own rays, narrowed by the depth's rounding, lies outside an allocated block
by more than float's rounding (10 µm) [M 2026-11-14]. Every allocated block there lies within ε
of a point of the band, ε up to 1.6 mm at 2 mm voxels and 4.1 mm at 8 mm (the depth's rounding
to whole millimetres and the test's sampling of the band), so within μ(z) |r| + √3 ε of the
surface [M 2026-11-14].

## The ray cast

The model's surface as a pose sees it, the surface the registration takes at each level of the
frame's pyramid (docs/methods/registration.md, "Input"): KinectFusion's surface prediction
[S: Newcombe et al. 2011, section 3.4] on the hash of blocks [S: Nießner et al. 2013, section 7]
(`sparse_tsdf::ray_cast()`, into the images of a `gca::model_surface`). It takes a pose, the
intrinsics K of level 0's picture and a depth range, and gives for each pixel of each level a point
in the camera frame of the pose with its validity mark, a normal or none, and an intensity.

* **The ray** of pixel (u, v) is r = ((u - cx) / fx, (v - cy) / fy, 1) in the camera frame, by the
  one pinhole convention, and the point at depth z, along the camera's z, is t + z R r in the
  models' frame, (R, t) the pose rounded to float as the allocation rounds it.
* **The depth range** is the caller's, the object class's, the one the frame's organized points
  are made with (`gca::organized_points_parameters`, refused as the organized points refuse it)
  [?: a choice]: the march starts at its near limit and runs one step past its far limit, and a
  crossing deeper than the far limit is not taken, so that the surface covers the depths the
  frame's points do.
* **Empty space**: the allocation's grid traversal on blocks ("The allocation near a frame's
  surface", step 2; the one copy, `block_walk` in `src/fusion/sparse_tsdf.cu`) visits the blocks the
  ray's segment between those depths crosses, in order, one lookup each [S: Amanatides and Woo
  1987]; in a block the hash holds no voxels for, every sample is none, without a voxel read. The
  march steps through those samples' indices: a jump to the block's exit, computed from the
  lattice, took 7 % to 10 % longer back to back, the kernel at 68 registers a thread against 64
  [M 2026-11-15: docs/HARDWARE.md, "The ray cast"; B: `cuobjdump -res-usage`], and is not built.
  A segment of length L visits at most sqrt(3) L / (8 v) + 4 blocks:
  125 at the tabletop object's live model, 0.30 m to 1.0 m at 2 mm voxels, and 116 at the large
  object's, 0.4 m to 3.0 m at 8 mm, along the camera's longest ray, |r| = 1.590 at either stream
  size [S: computed]. No cap: the range is the object class's, and the time follows it. The walk
  looks up only the blocks within its pixel's depth interval (the next point); the others it steps
  past.
* **The depth interval** [S: Kähler et al. 2015, InfiniTAM's ray cast on a hash of voxel blocks,
  and Nießner et al. 2013, section 7, the method from the papers, no library's code read or
  copied]: before the march, every block the hash holds is projected into the picture of level 0,
  and each tile of 8 x 8 pixels [?: a choice; 16 x 16 took within 0.3 % of its times (M 2026-11-15,
  docs/HARDWARE.md, "The ray cast")] keeps the least and the greatest depth, along the camera's z,
  of the corners of the blocks whose projection covers it. A block's projection is the rectangle
  around its 8 corners' projections, rounded out to whole pixels; a block whose corners all lie
  nearer than one block before the near limit, or farther than one block past the march's end, is
  left out; one that reaches nearer than one block before the near limit is clipped there, its
  corners beyond that plane and its edges' crossings of it projected; and when the near limit lies
  within a block and 1 mm of the camera's plane, a block that reaches within 1 mm of it covers the
  whole picture [?: a choice]. A ray then steps past the blocks it leaves before its tile's least
  depth, without a lookup or a sample, takes as its first sample the one the march through every
  block reaches there (the least index whose depth is at or past the block's entry: the lattice's
  depths never fall from one index to the next), and ends at the first block it enters past its
  tile's greatest depth; a tile no block covers leaves its rays invalid without a march. Every
  block a ray passes through is one whose projection covers its pixel and whose corners' depths
  bound the depths the ray has in it [S: a block is the convex hull of its corners, and the depth
  along the camera's z is linear], so the blocks stepped past and those after the end hold no
  voxels: their samples would be none, and the pixel is the one the march through every block
  gives, bit for bit [M 2026-11-15: 1272 casts of four walls and the two scenes of "What the ray
  cast gives", from the walks' poses, poses between them, 24 turned up to 15° and shifted up to
  0.1 m, poses 0.15 m to 1.28 m nearer the scene, and the walls from in front, behind, inside their
  band, turned and shifted, each in the object class's range, from 0 to its far limit and a narrow
  one, at both stream sizes: all four images of all three levels compared with the march through
  every block, 1.07e9 pixels, none differed; `tests/test_sparse_tsdf_ray_cast.cpp` compares a
  cast with one whose tiles a held block in front widens, bit for bit]. The comparisons of depth
  allow for the rounding of the walk's depths and the corners', 4 times its bound
  [?: a choice] (`march_to_surface()`): 0.12 mm and 0.23 mm at the live models' ranges with the
  camera within 1 m of the models' origin [?: an assumption; S: computed], about 0.7 mm at 10 m,
  which costs a few more lookups, not a surface [S: computed]. The rectangle needs no widening: a
  pixel's ray lies at a whole u and v, so rounding the rectangle out to whole pixels takes every
  pixel less than 1 px, less the projection's rounding, outside it, and the walk's rounding across
  the ray, (n + 1) 2^-23 L f / z_near for a walk of n blocks along a segment of length L, stays
  below 0.1 px at the live models' ranges [S: computed]. After a stretch stepped past, the first
  block looked up holds no voxels, as its depths begin before the tile's least depth [S: the bound
  above]: its samples are none. The projection reads every slot of the hash, so its time grows
  with S (docs/HARDWARE.md, "The ray cast").
* **In a block it holds**: samples at the depths near + i Δz, one lattice along the whole ray, with
  Δz = μ_min v / (2 |r|), so that the step is half the least truncation along the ray in space, 2
  voxels at μ_min = 4 [?: initial; Nießner et al. 2013, section 7: "we skip a predefined interval
  (half the minimum truncation value)"; KinectFusion marches in "steps with size < µ" (S: 2011,
  section 3.4)]. The distance at a sample is the trilinear interpolation of the 8 voxels whose
  centres surround it, each found through the hash when it lies in another block
  [S: Nießner et al. 2013, section 7]; a sample with an unobserved or missing voxel among the 8 is
  none. A sample that is none, and a block the hash holds no voxels for, end the pair of samples.
* **The surface** lies between the first two consecutive samples whose distance changes from >= 0
  to < 0, at z* = z + Δz D(z) / (D(z) - D(z + Δz)), their linear interpolation [S: KinectFusion's
  equation 15]; the pixel's point is z* r, on its own ray in the camera frame. Its intensity is the
  Rec. 709 luma of the trilinear interpolation of the 8 voxels' colors there, unrounded, by the
  pyramid's one function (docs/methods/frame-pyramid.md, "The intensity"); a crossing whose own 8
  voxels are not all observed gives an invalid pixel. Its normal is the gradient of the distance
  there by central differences of one voxel, each end interpolated trilinearly, which is the
  trilinear interpolation of the voxels' own central differences [S: the interpolation is linear in
  the voxels' values and the same at every shift of the grid], turned into the camera frame and
  normalized [S: equation 14]. A pixel whose differences need a voxel that is unobserved or
  missing, or whose gradient is 0, is valid without a normal. A change from < 0 to >= 0 first, a
  back face, or no change gives an invalid pixel [S: section 3.4]: its point, normal and intensity
  0 and its mark 0, as the organized points' invalid pixel.
* **The normal's voxels, each read once**: each end's cell is the crossing's moved by one voxel
  along the end's axis, so the six ends' 48 voxels are 32 distinct ones, along each axis the 4
  layers floor(g) - 1 to floor(g) + 2 of the crossing's 2 x 2 voxels across it, 8 of them the
  crossing's own [S: the lattice]. The cast keeps the crossing's 8 distances from the
  interpolation of its intensity, reads each axis's 8 new voxels, 24 in all, each checked before
  the next is read, and sums each end from its 8 values in the interpolation's operations and
  order, so that every normal is the six interpolations' bit for bit (`crossing_gradient()`;
  the comparison under "On the GPU"). Where float's rounding carries g + 1 or g - 1 across a whole
  number into another cell than that, the cast interpolates the six ends as they are: g - 1
  rounds to -1 at g = -2^-24, g + 1 to 2 at g = 1 - 2^-24 [S: float32, rounding to the nearest, a
  tie to the even one], and g, p / v - 1/2, is one fma, rounded once [S: nvcc's `--fmad=true`
  default; B 2026-11-15: `cuobjdump -sass`], so it can lie that near a whole number on either
  side. The voxels whose absence leaves a pixel without a normal are the six interpolations' own:
  the 32, or the 48 where an end's cell is another. A gap of the tests: a mistake that moves the
  same corner of all three axes' ends alike, such as corner 7 of each end after the crossing read
  from the crossing's own layer, scales the gradient's three components by about the same factor
  on a plane, 1 - f_x f_y f_z / 2 for the crossing's fractions f [S: the ends' fractions differ
  from the crossing's by float's rounding only], so the normal's direction stays, and its
  second-order effect on the spheres lies within their tolerances: it survived
  `tools/run_mutant.sh` [M 2026-11-15], while the same mistake on one axis is caught.
* **The coarser levels** [?: a choice]: level l is the cast at K_l = K / 2^l
  (`frame_pyramid::intrinsics_at_level()`; docs/methods/frame-pyramid.md, "The intrinsics of a
  level"), of the pyramid's level size. Its ray at pixel (u, v) is level 0's at (2^l u, 2^l v)
  [S: (u - cx / 2^l) / (fx / 2^l) = (2^l u - cx) / fx, the halving exact in float], so the thread
  of that pixel of level 0 writes it too: three levels for the cost of one, each pixel a sample of
  the model at its own ray, not the pyramid's combining of the finer level. Through the
  registration they do what the frame's pyramid's levels do ("What the ray cast gives"); a
  combining is built only on a measured need.
* **No color** [?: the user's decision, 2026-11-15]: the cast gives the intensity alone, the registration being its one
  reader; a reader that needs the cast's colors adds them. The full model's colors come from the
  extraction.
* **No depth edges** [?: a choice]: the normals do not mark depth edges as the normals stage's do,
  so the registration's gradient fit may take an intensity from across an edge, which was measured
  to lose nothing (docs/methods/registration.md, "The color term").
* **The thinnest surface kept**: a ray keeps a surface when one of its samples falls behind the
  zero crossing among the negative distances whose 8 voxels are all observed; a band of those
  longer along the ray than the step, μ_min v / 2, is never stepped over. A surface fused head-on
  at a depth where μ = μ_min holds negative distances μ_min = 4 voxels deep, the voxels with
  η >= -μ, and a sample there has its 8 voxels observed over 3.5 of them, the last voxel's half
  lacking an observed voxel beyond [S: the integration's rule, step 2]: 1.75 steps. Twice the step,
  μ_min v, steps over that band on 6860 of the 239616 pixels 8 or more inside the picture of such
  a wall, 500 mm away at 2 mm voxels [M 2026-11-15: the doubled step as a mutant of
  `tests/test_sparse_tsdf_ray_cast.cpp`]. An object thinner than about 2 voxels can be stepped
  over, as can a surface fused only at a slant and cast head-on, whose band along the cast's ray is
  thinner by the ratio of the two incidences' cosines [S: geometry].

**What the ray cast gives** [M 2026-11-15: `tests/test_sparse_tsdf_ray_cast.cpp`, the frames at
640x400 with the camera's intrinsics, depth unit 1 mm, the method's starting values]:

* **A wall** facing the camera at 500 mm, fused once at 2 mm voxels, where μ = μ_min: cast from its
  pose, every pixel 8 or more inside the picture holds the wall, its depth within 0.12 µm, its
  normal within 2e-6° of -z and its intensity within 1e-7 of the color's luma; cast from behind it,
  and from its pose looking back at no block, no pixel is valid.
* **A wall fused at a slant**, 500 mm in front of a camera turned by 20° about (1, -1, 0), at 2 mm
  voxels, in three orientations, the camera looking along the models' z, x and y (turns whose
  matrices hold only 0 and 1), and cast at both stream sizes with the principal point at the
  picture's centre from a position whose coordinate along the axis the column's rays keep is
  0.003000000026077032 m and along the axis the row's rays keep -0.005000000819563866 m: every
  crossing on the column lies at g = 1 - 2^-24 on that axis, where g + 1 rounds to 2 (through the
  fma), every crossing on the row at g = -3 - 2^-22, where g - 1 rounds to -4, so that the normals
  there take the six interpolations ("The normal's voxels, each read once"), for the end after along
  x, y and z and the end before along y, z and x [S: computed in exact rationals and float32]. Every
  pixel whose ray meets the fused part of the wall holds it within 0.59 µm of its depth and its
  normal within 0.0121° of the wall's, the column and the row as the rest; its distances are rounded
  to half floats.
* **Every cast**: each coarser level's pixel is level 0's at its centre bit for bit, with the
  intrinsics K / 2^l; every valid point projects within 7.1e-5 px of its own pixel's centre.
* **A sphere on a plane**, the scenes of "The extraction" fused from their walks of 12 frames at the
  live models' voxel sizes, cast from the walk's first pose, from a pose halfway to the next and from
  one a hand's step off the first (1.5° and 10.8 mm for the tabletop, 0.5° and 14.1 mm for the large
  object), each at 640x400 and 1280x800. The distance of a valid point from the analytic surface,
  its normal's angle from the surface's normal and its intensity's difference from the luma of the
  texture there, the worst median / 99th percentile / largest of the six casts:

| model | voxel | distance, mm | normal, deg | intensity | pixels that see the scene valid | without a normal |
|---|---|---|---|---|---|---|
| tabletop, live | 2 mm | 0.057 / 0.86 / 1.83 | 2.04 / 8.36 / 32.4 | 0.0028 / 0.020 / 0.18 | 99.69 % or more | 0.43 % to 0.50 % |
| large, live | 8 mm | 0.24 / 4.71 / 10.8 | 2.14 / 14.0 / 75.2 | 0.028 / 0.104 / 0.48 | 99.60 % or more | 0.33 % to 0.36 % |

  (The pixels that see the scene: those whose ray meets it within the depth range. Without a
  normal: of the valid pixels.) The cast's points lie nearer the surface than the extraction's at
  the median, 0.057 mm against 0.10 mm and 0.24 mm against 0.51 mm ("The extraction"); the largest
  distances and angles lie where the band of a surface reaches past another's silhouette, as the
  extraction's do. A valid pixel whose ray meets nothing lies at the plane's border, where a voxel
  past the edge takes its nearest pixel's depth: 0 to 14 pixels of a cast at 2 mm and 73 to 312 at
  8 mm, within 0.57 and 0.91 voxels of the plane. A view of space the walk never saw has no valid
  pixel.
* **Through the registration**: a frame seen from A Δ, Δ the registration's known motions
  (docs/methods/registration.md, "Accuracy"), registered by the full method from the identity against
  the cast from A, A fused last after 4 frames each Δ before it; with the levels the registration
  ran, its systems and the error from Δ; at 640x400, the live models' voxel sizes:

| case | level 0 alone | levels 1 and 0 | all three | all three, frame A's pyramid as the surface |
|---|---|---|---|---|
| tabletop object, 0.5 m, 2 mm | 5, 0.0092°, 0.077 mm | 8, 0.0093°, 0.077 mm | 12, 0.0093°, 0.077 mm | 15, 0.0062°, 0.059 mm |
| large object, 2.7 m, 8 mm | 6, 0.0195°, 0.96 mm | 12, 0.0196°, 0.96 mm | 21, 0.0196°, 0.96 mm | 20, 0.0136°, 0.67 mm |
| large box turned by 1.5°, 8 mm | iterations used up, 0.69°, 84 mm | 16, 0.0022°, 0.083 mm | 18, 0.0021°, 0.080 mm | 19, 0.0024°, 0.096 mm |

  Every one converged but the large box's level 0 alone, which the pyramid's levels recover as the
  frame's own pyramid does: the coarser levels as samples of the cast do their part, and no
  combining is built. The large object lies 0.96 mm off through the cast against 0.67 mm against
  the frame's pyramid: the 8 mm voxels blur the box's texture, whose waves span 5 to 9 pixels of
  8.8 mm there (docs/methods/registration.md, "Accuracy"), and the color term's minimum moves with
  it; the tabletop object 0.077 mm against 0.059 mm.

**On the GPU**: the clearing of the tiles (one `cudaMemsetAsync`, 8 bytes a tile), the depth
interval's kernel, one thread for each slot of the hash, which reads the hash and writes only the
tiles, and the march's kernel, one thread for each pixel of level 0 in the stages' block of 32 x 8,
which reads the tiles, the hash and the voxels and writes only the surface; they allocate nothing,
wait for nothing and read nothing back, and the tiles come from the pool with the surface. A ray
keeps the block it marches in and the last other block it found a voxel in, so that a sample's 8
voxels and a normal's 24 more, which lie in one or a few blocks, look each block up about once: back
to back 11 % to 16 % shorter than one lookup for every voxel outside the march's block, when the
normal still read 48 [M 2026-11-15: 9.945, 31.711, 12.044 and 36.772 ms against 8.809, 27.982,
10.064 and 30.966 ms, the cases of docs/HARDWARE.md, "The ray cast", before the depth interval]. The
depth interval took 23 % to 36 % off the back-to-back medians, and the normal's voxels read once
19.5 % to 27.9 % more [M 2026-11-15: docs/HARDWARE.md, "The ray cast"]. The march's kernel uses 64
registers a thread and 16 bytes of stack, held at 64 by its launch bounds, 4 thread blocks of 256 a
multiprocessor (74 registers and no stack without them), the depth interval's 43 registers
[B 2026-11-15: `cuobjdump -res-usage`]. The normal's voxels read once keep every pixel the six
interpolations' [M 2026-11-15: the 1272 casts of "The depth interval", 96 of a wall fused at a slant
at 2^-9 m voxels from poses that pin the column and the row at g = -2^-24, and 108 of the wall fused
at a slant of "What the ray cast gives" in its three orientations, all four images of all three
levels against the cast that interpolated the six ends, none differed; 50 of the 1272 casts'
313,336,357 normals and 39,773 of the slanted walls' took the six interpolations, counted in a
scratch copy]. Its time stands in docs/HARDWARE.md, "The ray cast".

**Where the time goes, and what was measured and not built** (`ray-cast-faster-by-measurement`;
back-to-back medians at 640x400 and 1280x800, 2 mm then 8 mm, in the scenes of docs/HARDWARE.md,
"The ray cast"; scratch copies of the kernel with one part taken out [M 2026-11-15, 09:55 to
11:08]):

* Before the depth interval, the march through blocks the hash does not hold, before a ray's first
  held block and after its last, took 25 %, 33 %, 20 % and 28 % of the cast, the part the interval
  takes out: about half the pixels of the scenes see no surface and walked 78 to 89 blocks, nearly
  all without a held one, and 86 % to 88 % of a cast's lookups found no block.
* With the depth interval, while the normal read its 48 voxels, they took 48 %, 48 %, 37 % and
  36 % of the cast (3.302, 9.347, 4.883 and 13.733 ms without them, against 6.354, 17.966, 7.747
  and 21.402), and the samples before the crossing's pair within the held blocks 24 %, 23 %, 33 %
  and 33 % of a cast whose rays start at their first held block (each ray started at that pair
  instead). Each voxel's read is followed by its check before the next voxel's read [B 2026-11-15:
  `cuobjdump -sass`].
* Built since (`ray-cast-normal-reads-once`): the normal's voxels each read once ("The normal's
  voxels, each read once"), the march's kernel held at 64 registers by its launch bounds: 27.9 %,
  26.9 %, 20.9 % and 19.5 % shorter back to back, 4.550, 13.126, 6.115 and 17.237 ms against
  6.314, 17.967, 7.735 and 21.402 ms [M 2026-11-15, 22:30 to 22:49: docs/HARDWARE.md, "The ray
  cast"]. The same session: the launch bounds alone on the cast before, -0.3 % to +0.1 %; the
  reads once without them, at 74 registers, 17.2 %, 17.3 %, 9.2 % and 9.6 % shorter, under 10 % at
  8 mm. That the rest is the occupancy 64 registers allow, 4 thread blocks of 256 a multiprocessor
  against 3 [S: CUDA C++ Programming Guide, compute capability 8.7, 65536 registers a
  multiprocessor], is inferred, not measured [?].
* Measured and not built, with the reads once: one function that hands an observed voxel out to
  the interpolation and to the 24 new reads, 7.6 % to 8.4 % longer than the reads written in each
  (4.898, 14.162, 6.593 and 18.684 ms, the same session); with that function, the crossing's cell
  corner and fractions handed to the normal rather than computed again, within 0.1 %, and the six
  interpolations' gradient computed apart from the 24 reads', 0.3 % to 0.8 % longer [M 2026-11-15,
  21:32 to 21:39, back to back]; on the cast before `ray-cast-faster-by-measurement`'s review fixes,
  an axis's 8 loads issued before their checks, 80 registers a thread, 16.0 % to 8.3 % shorter, less
  than the reads checked one by one [M 2026-11-15, 12:10 to 12:20; B 2026-11-15: `cuobjdump
  -res-usage`].
* Not built, as none shortens the cast by more than 10 % at both stream sizes [?: initial, the
  ticket's threshold], against the cast with the depth interval, each made to keep the cast bit for
  bit (the three as first built compared as above: none differed): the 8 voxels of an interpolation
  read before their checks, 5 % to 9 % longer at 80 registers a thread [B 2026-11-15: `cuobjdump
  -res-usage`], 0.6 % to 2.1 % shorter held at 64 registers by the launch bounds (88 bytes of stack
  [B 2026-11-15: `cuobjdump -res-usage`]), 23 % to 25 % longer together with the next; the crossing
  and its normal computed after the march's loop, where a warp's threads meet again, 2.5 % to 4.0 %
  shorter; the 2 x 2 x 2 blocks around a crossing each looked up once, 42 % to 52 % longer at 76
  registers and 48 bytes of stack [B 2026-11-15: `cuobjdump -res-usage`]. Paced, where the governor
  kept the clock, the second was 2.4 % to 4.1 % shorter; the others' paced times moved with the
  clock the governor chose (the ticket's Comments). Their code is backed up outside the tree.
* Not tried: starting a ray near the depth the last cast found at its pixel, and longer steps where
  the samples' distance is truncated in front of the surface. Neither keeps every surface the march
  through every block finds: the first misses a surface that came before the last depth, the second
  one fused at a slant or thinner than its step, which tracking would show as drift. Their most is
  the samples before the crossing's pair above; at the tabletop object's live model the truncation
  is μ_min out to 0.52 m, so a sample's distance there is at most twice the step and a longer step
  saves at most one sample in two [S: computed, "The truncation distance"].

## The extraction

The colored points with normals of the full model: for each block the hash holds, each voxel
g and each of its neighbours g + e, e = (1, 0, 0), (0, 1, 0), (0, 0, 1), found through the hash
when it lies in another block: when both are observed and their distances D_a, D_b differ in
sign (D_a >= 0 > D_b, or D_a < 0 <= D_b), a point at the fraction s = D_a / (D_a - D_b) of the
way from g's centre to the neighbour's. These are the vertices marching cubes places on the
grid's edges [S: W. E. Lorensen, H. E. Cline, "Marching Cubes: A High Resolution 3D Surface
Construction Algorithm", SIGGRAPH 1987], without its triangles. The point's normal is the
gradient of D by central differences at both voxels, interpolated at s and normalized, so it
points from - to +, towards the cameras that saw the surface; its color the two colors
interpolated at s, each channel rounded to the nearest integer, a tie to the even one. A
crossing whose central differences need a voxel that is missing or unobserved, any of the 12
around the two, gives no point [?: a choice: no one-sided difference], nor does one whose
interpolated gradient is zero. A voxel whose distance is exactly 0 between two negative
neighbours along one axis gives the same point twice, one from each side; no rule leaves one
out, as either sign given to 0 has such a case, and the two points are one. The points go into a
buffer of a stated size from the pool
(`gca::surface_points`) in no stated order, and their count is read back with the extraction's
one wait; a buffer too small is reported as a value, with the count found and the buffer's first
points written. Each point carries its block's slot in the hash, which no other block takes.

**What an extraction takes** (`gca::extraction_choice`). Its defaults take every block and every
crossing, as refinement's full model does. Two choices narrow it, for the live model
(docs/methods/tracking.md, "The live model"), the per-frame path's one use of the extraction, at
most twice a second:

* **The grid's thinning**, a spacing m, a power of two: a crossing gives a point only when both
  coordinates of its voxel g across its edge's axis are multiples of m. So the points of 2m are a
  subset of those of m, a surface keeps the same points from one extraction to the next, and a
  doubling keeps about a quarter of a surface's points: 0.250 at 2 and 0.249 at 4 on the tabletop
  sphere's live model [M 2026-11-16: `tests/test_sparse_tsdf_fusion.cpp`]. A spacing that is no
  power of two is refused. It saves little time past 2: 23.2, 22.8 and 23.9 ms at 2, 4 and 8
  against 30.2 ms at 1 on a scene of 46849 blocks [M 2026-11-16: docs/HARDWARE.md, "The live
  model"], as a held block's time is the latency of its own work, not its edges' count
  [?: inferred, not measured apart].
* **The blocks touched since a stamp**: only the blocks whose slot an allocation near a frame's
  surface of that stamp or a later one touched (each slot's stamp, "The allocation near a frame's
  surface"; `sparse_tsdf::frames_allocated()` gives the next), which the sweep reads with the
  slot. When the buffer has a list, each block the extraction goes through is listed once, whether
  it gives points or not, so that a reader knows whose points the extraction replaces.

**The sweep over the slots** [M 2026-11-16]. The kernel goes through the hash's slots in chunks of
512 consecutive slots on the integration's fixed grid, one chunk to a thread block at a time. Each
of the thread block's 512 threads reads one slot of its chunk (`block_in_slot()`), so a chunk's keys
come in coalesced loads. The slots that hold a block with a number go into a list in shared memory,
in slot order: a warp's ballot gives each of its lanes the count of the earlier lanes that hold one,
and the warps' counts, in shared memory, give each warp the count before it. Then the thread block
extracts the listed blocks one after another, one thread for each voxel, as above. Two barriers a
chunk: one between the warps' counts and their reading, one between the list and its reading. The
next chunk needs none more: it writes the counts only after the second barrier, which every thread
passes once it has read them, and the list only after its own first barrier, which every thread
reaches once it has read this chunk's list. The kernel's launch bounds ask for 3 thread blocks a
multiprocessor, so that all 24 of the grid run at once: 40 registers a thread, 40 bytes spilled to
the stack and 10304 bytes of shared memory, the chunk's 512 blocks of 16 bytes, their 512 slots and
16 counts of 4 bytes [B 2026-11-17: `cuobjdump -res-usage build/src/libfusion.a`, read again
whenever the kernel or its bound changes]. The points are the same set, bit for bit, as when the kernel took one
slot a thread block per step, on the walk's live and full models at S = 2^18 and 2^21, and at 2^20
with a third barrier. Their order is still not stated, and nothing relies on it: it varied between
runs before too, by the order the threads' atomic additions took, the tests and the replay compare
points as sets, and a buffer too small keeps its first points, whichever they are. The time now
follows the blocks held, 0.6 to 1.2 µs a block [S: computed, the times over the blocks], and S adds
about 1 ms at 2^21: 1.97 ms at S = 2^18 and 3.15 ms at 2^21 for the 2682 blocks of the walk's first
frame. The walk's live model of 7932 blocks takes 6.85 ms at S = 2^21 against 360 ms before
(docs/HARDWARE.md, "The extraction").

**Measured and not built** [M 2026-11-16: scratch prototypes beside the committed kernel, the same
point sets bit for bit; docs/HARDWARE.md, "The extraction"]. A third barrier a chunk, after the
list's blocks, made the kernel 0 % to 3.9 % shorter, too little to keep a barrier the code does not
need. The same chunks with three barriers but without the bound, 57 registers and 2 thread blocks a
multiprocessor, 16 of the grid at once, took 1.3 to 1.6 times as long. A pass of one thread per slot
that lists the held slots, then one thread block for each entry of the list, as the integration goes
through the frame's list, and a pass over the blocks in use, each block's slot kept in an array of 4
bytes a block (which takes B at 2.0 GB from 479620 to 479153), took 0.8 to 1.8 times the chunks'
time on the live model, less only at its 2682 blocks, and 1.8 to 5.3 times on the full model's 12599
to 39253 blocks; each step of their thread blocks fetches its block through dependent loads from
global memory, where the chunks read it from shared memory [?: the likely cause, not measured
apart]. An estimate made from the committed kernel's times, 4.6 to 5.0 µs a block in use [S:
`.scratch/rewrite/live-model-extraction-research.md`, not in the tree], was 4 to 8 times what the
chunks take. What a block's time is made of is not measured apart.

**No filter of crossings** [M 2026-11-15]. A crossing between the bands of two different
surfaces, a voxel truncated in front of one beside a negative one of the other, is no surface;
the test measured whether a filter would remove such points, on a sphere resting on a plane
fused from a walk of 12 poses at the four voxel sizes of the object classes (the table below).
The step between the two voxels' distances, |D_a - D_b|, which such a crossing makes large, does
not separate the far points: at 4 mm voxels points up to 11.6 mm off lie among the crossings
of a step below 2 voxels, and the points of a step above 16 voxels lie no farther than the
others (99th percentile 7.4 mm against 2.0 to 9.1 mm). The far points come from the band itself:
with c = 0.01 /m the band at 1.6 m to 2 m is 77 mm to 120 mm deep, and its negative distances
behind the sphere's rim, seen along a ray that grazes it, lie outside the sphere: ℓ² / (2 R) =
12.5 mm for ℓ = 100 mm behind a rim of R = 0.4 m [S: geometry]. With c = 0.002 /m the largest
distances of the large object's models fell from 13.0 mm and 14.9 mm to 4.6 mm and 3.8 mm
[M 2026-11-15, the same test with c changed]. The measured c ("The truncation distance") is the
remedy that stands; a filter is built only on a measured benefit.

**What the extraction gives** [M 2026-11-15: `tests/test_sparse_tsdf_fusion.cpp`, 640x400, the
method's starting values; tabletop: the tabletop sphere, 8 cm, on its plane, walked around from
0.45 m; large: a sphere of 0.4 m on a plane of 3 m x 3 m, walked around from 1.6 m]. The
distance from the analytic surface and the normal's angle from its normal, and the largest
channel's error of the color against the texture there, each as median / 99th percentile /
largest:

| model | voxel | far depth | points | distance, mm | normal, deg | color, levels |
|---|---|---|---|---|---|---|
| tabletop, live | 2 mm | 1.0 m | 202488 | 0.10 / 0.73 / 1.50 | 3.5 / 15.1 / 37.0 | 2.1 / 20.6 / 76 |
| tabletop, final | 1 mm | 0.7 m | 905214 | 0.11 / 0.60 / 1.38 | 6.8 / 30.9 / 95.2 | 1.7 / 21.7 / 115 |
| large, live | 8 mm | 3.0 m | 181794 | 0.51 / 4.41 / 13.0 | 4.2 / 24.3 / 84.6 | 16.1 / 91.8 / 183 |
| large, final | 4 mm | 2.0 m | 772148 | 0.41 / 4.97 / 14.9 | 6.6 / 44.7 / 176 | 7.4 / 98.7 / 199 |

(The far depths of the full models are nearer than those of the ranges the test was measured at,
1.0 m and 3.0 m, so that the test's pool of 65536 blocks holds one frame's bound.) The normals at
1 mm voxels show the depth's steps of 1 mm. The largest angles, flipped up to 176°, lie where the
band reaches past the sphere's rim. The colors follow the textures' waves of 45 mm to 100 mm, up to
14 levels a millimetre [S: computed, 2 pi 100 / 45 mm], at each voxel's centre's projection, so they
blur at 8 mm. Of the surface the frames saw, sampled at every fourth pixel, all but 0.018 % or less
lie within 2 voxels of a point, the 99th percentile within 0.68 to 0.83 voxels; the rest lie at the
plane's border, at the far depth and where the sphere meets the plane, where the two bands interfere
("The truncation distance"). Fused from 12 frames with Gaussian depth noise of sigma = 0.01 z² at
2 mm voxels, the surface spreads 1.01 mm about the plane, one frame's points 2.19 mm, 2.16 times
less (root mean square).

## What the prototype did, and what is not carried

The prototype had no TSDF and no oracle for it: its loop (CUDA_3D's `apps/app.cpp`, deleted in
eabb407) transformed the whole model into the current camera frame each frame, appended the
frame's points, and downsampled the model on an 8 mm voxel grid every 5 fused frames
[S: `apps/app.cpp` at eabb407^]. None of that is carried; the TSDF's tests compare it with
analytic surfaces. Also not carried:

* **The motion gate**: a registration step whose rotation's cosine was above 0.999999 (under
  0.081°) and whose translation was under 1 mm was not applied, while the reference frame
  advanced, so slow motion drifted [S: as above]. Every frame that passes the check is fused at
  its pose.
* **The model in the camera frame**: the TSDF stays in the models' frame, and a frame comes to
  it by its pose.
