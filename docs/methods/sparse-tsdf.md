# The sparse TSDF

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The model of a running scan, and of refinement's fusion at the full voxel size: a truncated
signed distance field (TSDF) kept only near the surfaces, in blocks of voxels that a hash on the
GPU finds by their coordinates (docs/adr/0004-sparse-tsdf-model.md). It has three operations:
integrate a frame at its pose, allocating the blocks near its surface first; ray cast from a
pose, giving the surface the registration takes; extract the colored points with normals that
the full model is made of.

Built so far: the hash of blocks and the pool they come from (`src/fusion/sparse_tsdf.*`, its
test `tests/test_sparse_tsdf_blocks.cpp`). The allocation near a frame's surface, the
integration, the extraction and the ray cast are not built yet; this document is their method,
written before their code.

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

* **The voxel size** v, in m, is the TSDF's one length: the live model's or the final models'
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
64-bit access. A voxel with W = 0 is **unobserved**: its distance and color mean nothing, and
no operation reads them. The pool's blocks start with every byte 0, so every voxel of a block
just handed out is unobserved.

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
  away; no measurement of the noise on this board exists yet: σ(z) is to be measured on still
  views of a plane, and c from it. At 640x400 the datasheet gives nothing [?].
* **k = 3** [?: initial]: three standard deviations, so that the true surface lies inside ±μ of
  a measurement for 99.7 % of the measurements of a Gaussian noise [S: the normal distribution].
* **μ_min = 4 voxels** [?: initial]: at least one voxel with a distance below the truncation on
  each side of a zero crossing, along any axis and diagonal, with room to spare; its half is the
  ray cast's step ("The ray cast").

What the rule gives, in mm [S: computed]:

| voxel size | μ_min | μ at 0.3 m | 0.5 m | 1.0 m | 2.0 m | 3.0 m | 3σ passes μ_min at |
|---|---|---|---|---|---|---|---|
| 1 mm (tabletop, final) | 4 | 4 | 7.5 | 30 | | | 0.37 m |
| 2 mm (tabletop, live) | 8 | 8 | 8 | 30 | | | 0.52 m |
| 4 mm (large, final) | 16 | | 16 | 30 | 120 | 270 | 0.73 m |
| 8 mm (large, live) | 32 | | 32 | 32 | 120 | 270 | 1.03 m |

(Blank: outside the object class's depth range, 0.30 m to 1.0 m and 0.4 m to 3.0 m
[?: initial, `src/scan/object_class.cpp`].) The cost
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
   frame, or it is invalid.
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
   float, each of C's components to the nearest integer.

* **W_max = 64** [?: initial]. Once a voxel holds W_max observations the average becomes a
  moving one: a new observation counts 1/65. A limit of the color's rounding follows: with
  W = 64 a new color moves a component only when it differs from the stored one by 33 levels or
  more [S: computed, the rounding of (c - C) / 65], so a voxel's color is mostly that of its
  early observations. The integration's tests measure the colors against the texture.
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
* **The slots**: S slots, S a power of two and at least twice the pool's blocks ("The pool and
  the memory budget"). Two arrays: the keys (8 bytes each) and the blocks' numbers (4 bytes
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
  the slots in the same order, h, h + 1, ..., with atomicCAS(slot, empty, key) [S: CUDA C++
  Programming Guide, "Atomic Functions", atomicCAS on 64-bit words]. The swap returns:
  * the empty key: this thread has put the key there and is the one thread that inserts it. It
    takes the next block's number n with atomicAdd on the pool's count of claims; n below the
    pool's blocks is written as the slot's block, otherwise the slot keeps "no block" and the
    key counts as refused for a full pool.
  * the key itself: an earlier call or another thread put it there; nothing to do.
  * another key: the next slot.

  After S slots without room, the coordinate is refused for a full hash and counted.

**Why no key is lost or doubled.** A slot changes once, from empty to a key, and never back.
Every thread that inserts key K tries the same slots in the same order, and passes a slot only
when it holds another key, which it then holds for good. Let s be the first slot of that order
that ever holds K: every slot before s held another key when the first of these threads passed
it, so every later thread passes it too, reaches s and finds K there or puts it there; the swap
on s succeeds for exactly one of them. So K stands in exactly one slot, gets at most one
block's number, and a lookup, which tries the same order, finds it.

**What is wait-free, and what is not.** No thread waits for another and none holds a lock: an
insertion ends after at most S swaps whatever the other threads do, a bound on its own steps
(wait-free, not only lock-free). What is not atomic: the block's number is written by a plain
store after the swap, so within the kernel that inserts, another thread can find the key before
its number and read the slot's cleared "no block". Lookups and every reading of a block's number
therefore run in a kernel queued after the insertions on the same stream, which sees all their
writes [S: CUDA C++ Programming Guide, "Streams": the work of one stream runs in order]; the
TSDF is used from one stream at a time. The insertion and the lookup are one device function
each (`src/fusion/block_hash.cuh`), which every kernel that inserts or finds a block calls. Nießner et al.
lock a bucket for an insertion instead, and "all other allocations for the same bucket are
staggered until the next frame is processed" [S: 2013, section 4.2]: a block is then missing for
a frame. Here none is.

**Probes.** At a load α of keys per slot, linear probing needs about (1 + 1 / (1 - α)) / 2
probes to find a key and (1 + 1 / (1 - α)²) / 2 to find it absent [S: Knuth, section 6.4]: at
the 1.0 GB budget's full pool, α = 0.46, that is 1.43 and 2.22 [S: computed].

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

**The memory budget** is all of the sparse TSDF's device memory: the pool's blocks (4096
bytes each); the hash's keys, blocks' numbers and stamps of the last frame (16 bytes a slot);
the frame's list of slots (4 bytes a block, "The allocation near a frame's surface"); and three
32-bit counts, of claims, of refusals and of the list. The object class sets it
(`src/scan/object_class.cpp`), 1.0 GB to start with [?: initial]. The pool holds the largest
number of blocks B for which

    4100 B + 16 S + 12 <= budget,   S = the least power of two >= 2 B

so at 1.0 GB, B = 241856 blocks and S = 524288 slots, 999998220 bytes, of which the keys and
numbers take 6.3 MB and the stamps and the list 3.1 MB [S: computed]. B is at most 2^30, so that
S fits 32 bits. The stamps and the list are counted and requested now, before their first user,
the allocation, so that the budget's division stays as it is.

**One sparse TSDF exists at a time.** Refinement's, at the full voxel size, is requested from a
pool of its own after the live model's is released, within the same budget.

**What the budget holds** [S: computed]: a surface facing the camera at depth z crosses about
2 μ(z) / (8 v) + 1 blocks along each column of blocks, so a square metre of it takes about
(2 μ / (8 v) + 1) / (8 v)² blocks. With c at the datasheet's bound, 1.0 GB holds:

| voxel size | at 0.3 m | at 0.5 m | at 1.0 m |
|---|---|---|---|
| 1 mm (tabletop, final) | 7.7 m² | 5.4 m² | 1.8 m² |
| 2 mm (tabletop, live) | 31 m² | 31 m² | 13 m² |

A surface seen at a slant, or from several sides, takes more.

Everything is requested from the GPU pool that closes (`src/gpu/gpu_pool.hpp`) before the scan
starts, and set on the scan's stream: the keys to empty, the blocks' numbers to none, the
stamps to every bit 1 (no frame), the counts and every voxel to 0; the list is written before it
is read. After that no operation of the sparse TSDF allocates.

**A pool too small** is reported as a value. Reading the usage gives the blocks the pool holds,
the blocks in use, the coordinates refused for a full pool and those refused for a full hash or
out of range; the share in use is the blocks in use over the blocks held, and the pool is used
up when the two are equal. A reading that fails gives no usage, so it is never taken for a
used-up pool. The reading copies the counts and waits for the stream, and so for all the work
queued on it before; in the per-frame path it comes after the registration's own readback, which
has waited for that work already, so that it costs only its copy of 12 bytes. A coordinate refused for
a full pool keeps its slot without a block, so the count of claims is the count of distinct
coordinates the hash took, and the same set of insertions gives the same counts in every run;
the hash fills only when more distinct coordinates than S arrive, which a scan that ends on a
used-up pool never sends.

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
   into the models' frame by the pose.
2. The blocks the segment passes through, in order, by the grid traversal of Amanatides and Woo
   [S: J. Amanatides, A. Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing", Eurographics
   1987] on the grid of blocks, each inserted into the hash.
3. The list of the frame's blocks, each block once, which the integration runs over: each slot
   keeps the number of the last frame that touched it, exchanged atomically; the thread that
   changes it appends the slot to the list. The list holds B slots, as many as the pool has
   blocks; a frame that touches more distinct slots, which only happens once the pool is used
   up, has the rest counted, not written.

Between the rays: a block that the band crosses but no ray does is missed. Neighbouring rays are
z / fx apart: at 640x400 (fx = 306.18 px [M 2026-11-13: the camera's intrinsics at
640x400, as `tests/test_organized_points.cpp` takes them]) 3.3 mm at 1 m and 9.8 mm at 3 m, at 1280x800 1.6 mm
and 4.9 mm, against a block's edge of 16 mm at 2 mm voxels and 64 mm at 8 mm [S: computed], so
only a block grazed at a corner can be missed. The allocation's test states how far from the
surface such a miss lies.

## The ray cast

The model's surface as a pose sees it, for each pixel of a camera of given intrinsics and size,
as organized points, normals and colors with a validity mask, the surface the registration
takes [S: Newcombe et al. 2011, section 3.4]:

* **The ray** of pixel (u, v), r = R ((u - cx) / fx, (v - cy) / fy, 1), from the camera's centre
  t, the point t + z r at depth z, from the object class's nearest depth to its farthest.
* **Empty space**: the grid traversal on blocks [S: Amanatides and Woo 1987]; a block the hash
  does not hold is crossed in one step, to its exit.
* **In a block that it holds**: D at the point by trilinear interpolation of the 8 voxel centres
  around it, each found through the hash, also across a block's border [S: Nießner et al. 2013,
  section 7, which looks up neighbouring voxels through the hash]. A sample with an unobserved
  voxel among the 8 is no sample. The step is half the smallest truncation, μ_min / 2 = 2 voxels
  [?: initial; Nießner et al. 2013, section 7: "we skip a predefined interval (half the minimum
  truncation value)"; KinectFusion marches in "steps with size < µ" [S: 2011, section 3.4]].
* **The surface** is the first sign change from + to - between two samples z and z + Δz, at
  z* = z + Δz D(z) / (D(z) - D(z + Δz)), the linear interpolation of KinectFusion [S: 2011,
  equation 15]. Its normal is the gradient of D there by central differences of one voxel,
  normalized [S: equation 14]; its color the trilinear interpolation of the voxels' colors. A
  sign change from - to +, a back face, or no change before the farthest depth gives an invalid
  pixel [S: section 3.4].
* **Thin surfaces**: a surface whose negative distances, seen from both sides, are thinner than
  the step can be stepped over; at μ_min / 2 that is an object thinner than about 2 voxels.

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
interpolated at s. The points go into a buffer of a stated size from the pool; a buffer too
small is reported as a value. A crossing between the bands of two different surfaces, a voxel
truncated in front of one beside a negative one of the other, is no surface; whether it needs
a filter, and which, the extraction's tests decide by measurement.

## What the prototype did, and what is not carried

The prototype has no TSDF and no oracle for it: its loop (CUDA_3D's `apps/app.cpp`, deleted in
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
