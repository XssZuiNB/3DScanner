# The intensity and the pyramid of a frame

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The per-frame path's step after the upload, as one call: a frame on the GPU becomes three levels,
each with its organized points, validity mask, normals and intensity and its own intrinsics.
Level 0 is the frame's own size: its points are the organized points
(docs/methods/organized-points.md), its normals the normals in image space
(docs/methods/normals.md), its intensity the Rec. 709 luma of the frame's color. Each coarser
level has half the rows and columns of the one before; the registration works through them
coarse to fine. The code is `src/geometry/frame_pyramid.*`, its test
`tests/test_frame_pyramid.cpp`.

## Input

* **A frame on the GPU** (`gca::gpu_frame`, `src/gpu/gpu_frame.hpp`), uploaded on the stream the
  step runs on: its depth picture, its color picture (RGB, 8 bits a channel) and its description
  (size W x H, intrinsics fx, fy, cx, cy in px, depth unit).
* **The parameters** (`gca::frame_pyramid_parameters`): the depth range of the organized points
  and the parameters of the normals (both documents above), the same at every level. The normals'
  steepest incidence θ also sets which points a coarser level combines ("A coarser level").
* **The number of levels**, L = 3 [?: initial]: level 2 is a quarter of the frame's size along
  each side, 160x100 at 640x400 and 320x200 at 1280x800. KinectFusion tracks over three levels
  [S: R. A. Newcombe et al., "KinectFusion: Real-time dense surface mapping and tracking", ISMAR
  2011, "Sensor pose estimation"]; how many levels the registration needs is measured by the
  registration.

## Output

For each level l = 0, 1, 2, organized like the frame: one entry for each pixel (u, v) of the
level, at index v · W_l + u.

* **The size** W_l x H_l: the frame's at level 0; W_{l+1} = ceil(W_l / 2), H_{l+1} =
  ceil(H_l / 2), so that every pixel of level l + 1 has its centre pixel in level l
  ("Which pixels are combined").
* **The intrinsics** K_l ("The intrinsics of a level").
* **The organized points** (`gca::organized_points`): the point in the camera frame (x right,
  y down, z forward, in m) and the validity mark, with the meaning of the organized points at
  every level: an invalid pixel's point is (0, 0, 0) and its mark 0; a valid point's z is > 0.
* **The normals** (`gca::organized_normals`): a unit normal toward the camera, or (0, 0, 0) for
  none, with the meaning of the normals at every level.
* **The intensity**: a float in [0, 1] at a valid pixel; 0 at an invalid one, which a reader
  skips by its mark, as it skips its point.

The pyramid owns every level: 12 bytes a pixel for the points, 1 for the marks, 12 for the
normals and 4 for the intensity, at three levels 9.7 MB at 640x400 and 39.0 MB at 1280x800, of
which the intensity and the coarser levels are 3.3 MB and 13.4 MB [S: computed, 29 bytes a
pixel over 1 + 1/4 + 1/16 of the frame's pixels]. All of it is requested from the pool
(`src/gpu/gpu_pool.hpp`) before it closes; the step itself asks for no memory.

## The intensity

The intensity of a pixel with color (R, G, B), each 0 to 255, is its Rec. 709 luma, scaled to
[0, 1]:

    I = (0.2126 R + 0.7152 G + 0.0722 B) / 255

[S: Recommendation ITU-R BT.709-6 (06/2015), the luminance signal E'_Y = 0.2126 E'_R +
0.7152 E'_G + 0.0722 E'_B]. It is luma, a weighted sum of the encoded values as the camera
delivers them, not the luminance of linear light. The coefficients and the scale are the
prototype's ("What the prototype did"). The colored ICP paper takes the plain mean
(R + G + B) / 3 [?: unchecked here; J. Park, Q.-Y. Zhou, V. Koltun, "Colored Point Cloud
Registration Revisited", ICCV 2017]; it replaces the luma only on a measured benefit to the
registration.

The arithmetic is float: three products and two sums of numbers below 256, and one division by
255, each rounded once, a few float steps of 1, below 1e-6, against the 1/255 = 3.9e-3 of one
step of 8 bits [S: float32 ulp]. At level 0 a pixel's intensity is computed from its own color,
the depth being aligned to the color; at an invalid pixel it is 0.

## A coarser level

### Which pixels are combined

Pixel (u, v) of level l + 1 is centred on pixel c = (2u, 2v) of level l, its centre pixel. Its
window is the 3 x 3 pixels (2u + i, 2v + j), i, j = -1, 0, 1, cut off at the picture's border.
This is the reduction of an image pyramid, each sample of the coarser level a weighted mean of
the finer level around twice its index, with the generating kernel of three taps
(1/4, 1/2, 1/4) along each axis, the triangle [S: P. J. Burt, E. H. Adelson, "The Laplacian
Pyramid as a Compact Image Code", IEEE Trans. Communications 31(4), 1983, the REDUCE operation
and its kernel w with a = 1/2]: pixel (2u + i, 2v + j) has the weight
w = (2 - |i|)(2 - |j|), 4 at the centre, 2 beside it and 1 at a corner. The kernel is
symmetric, so on a surface seen face on the mean of a whole window lies on the centre pixel's
ray, and each finer pixel weighs the same in total, 4, over the coarser pixels it enters.

A pixel of level l + 1 is valid exactly when its centre pixel is. Its window's pixels that are
**combined** are the centre and each valid pixel that lies on one surface with the centre by
the rule below. Its point and its intensity are their weighted means over the combined pixels
only:

    p = p_c + Σ w_j (p_j - p_c) / Σ w_j,   I = Σ w_j I_j / Σ w_j.

The weights are small whole numbers, their sums exact. The mean is taken of offsets from the
centre's point, a few mm, as the normals' mean is, so that it is rounded at the scale of the
point only once [S: float32 ulp; docs/methods/normals.md, "The covariance, centred first"].

So no value of a coarser level comes from an invalid pixel or from across a depth edge: an
invalid pixel is never combined, and a pixel across an edge from the centre is not either. A
pixel of level l is invalid when the frame's pixel (2^l u, 2^l v), its centre at level 0, is
invalid; an area without depth that holds that pixel, as one of 2^l x 2^l pixels or more always
does, stays invalid at level l. A gap narrower than a coarser level's pixels can leave no
invalid pixel there; the values around it are taken from the valid pixels alone.

A crease, where two faces of one object meet, is no depth edge, and a coarser point there is a
mean of both faces' points, off each face by a fraction of the window's span, as a normal there
lies between the faces' normals. On a plane every combined point lies within the depth's
rounding of the plane, and so does their mean, a convex combination of them [S].

### One surface, at every level

Two points a, b of level l lie on one surface for the combining when

    |d · m| <= sin θ_l · |d| · |m|,   d = b - a,   m = a + b,   tan θ_l = tan θ / 2^l,

the normals' rule of one surface (docs/methods/normals.md, "One surface, or a depth edge") with
the steepest incidence θ_l. Adjacent pixels of level l lie 2^l pixels of the frame apart, so the
segment between two points on one surface is 2^l times as long across the line of sight as at
level 0; with tan θ_l = tan θ / 2^l the rule takes for an edge the same step along the line of
sight as across one pixel of the frame: (z / fx) tan θ near the picture's centre, 6.1 mm at
0.5 m and 24.4 mm at 2 m at 640x400, 3.0 mm and 12.2 mm at 1280x800, with θ = 75°
[S: computed; docs/methods/normals.md]. With θ itself at every level, a step that the frame's
pixels show as an edge would be averaged over at level 2 up to twice that height. A smaller step
is one surface at every level, and a point combined across it lies off each side by at most half
the step.

KinectFusion builds its depth pyramid the same way: block averaging and subsampling to half the
size, a depth used in the average only when it lies within a fixed difference of the central
pixel's, so that no smoothing crosses a depth boundary [S: Newcombe et al. 2011, "Surface
measurement"]; the rule here keeps that difference to the frame's own step at every level.

θ_0 = θ = 75° and θ_1 = 61.8° [S: computed, atan(tan 75° / 2)]. On a surface seen at an
incidence above θ_l, the pixels of level l down its slope from the centre lie beyond the rule,
and those along its level line, whose segment lies across the line of sight, within it: level
l + 1 then combines along the level line only, a smoothing in that direction alone, and still
nothing from across an edge. θ is at most 89° [?: a choice, the user's, 2026-11-14]
(docs/methods/normals.md, "Input"), so that sin² θ_l stays below 1 by far more than float's
rounding of the rule, and the point (0, 0, 0) of an invalid pixel never lies on one surface with a
valid one.

### The intrinsics of a level

By the one pinhole convention (`src/types/camera_param.hpp`), pixel centres at whole numbers,
a point x, y, z projects at level l to u_l = fx_l x / z + cx_l. Pixel u of level l + 1 is centred
on pixel 2u of level l, so u_{l+1} = u_l / 2 and

    fx_{l+1} = fx_l / 2,   fy_{l+1} = fy_l / 2,   cx_{l+1} = cx_l / 2,   cy_{l+1} = cy_l / 2,

K_l = K_0 / 2^l, exact in float [S: computed from the projection; a division by a power of 2].
A window of 2 x 2 pixels, centred between four pixels of the finer level, would need
cx_{l+1} = (cx_l - 0.5) / 2 instead under this convention; the centred window needs no shift.

The mean of the combined points projects at K_l onto the mean of their pixels, each weighted by
w_j z_j [S: the projection of Σ w_j z_j r_j]: on the centre pixel when the whole window is
combined on a surface seen face on, and within half a pixel of level l of it otherwise, a
quarter pixel of level l + 1, and a little more where z varies across the window, by the ratio
of two adjacent depths, near the picture's centre at most 1 + tan θ / fx across one pixel and
1 + √2 tan θ / fx across a diagonal [S: the rule]. With θ = 75° at 640x400, 1.012 and 1.017, a
point of level 1 projects within 0.26 px of its pixel's centre, and one of level 2, whose
combined points of level 1 are each off by that much, within 0.38 px; with the largest θ, 89°,
within 0.28 and 0.42 px [S: computed, near the picture's centre]. A point of level 1 projects
within half a pixel of its pixel's centre whatever the depths, its weight on the far side of the
centre pixel at most that of the centre's side [S: the weights]; a point of level 2 does by the
rule's bound on the depths' ratio.

## Normals at every level

Each level's normals are estimated anew from that level's organized points by the normals stage
(`gca::organized_normals::estimate()`), with the same parameters: one method of normals at every
level, as KinectFusion computes each level's normal map from that level's vertex map
[S: Newcombe et al. 2011, "Surface measurement"]. The window of 7 x 7 pixels of a coarser level
spans 2^l times as much of the surface, over points whose depth's rounding the combining has
partly averaged out, so that a plane's normals tilt less than at level 0 (docs/methods/normals.md,
"Accuracy"); near a depth edge and a thin part, more of a coarser level's pixels have no normal.
A pixel of a coarser level has a normal by the rules of the normals stage on that level's points.

Combining the finer level's normals, a weighted mean of the combined pixels' normals made unit
again, would cost a few loads inside the combining's kernel in place of two runs of the normals'
kernel on a quarter and a sixteenth of the frame's pixels, whose times stand in
docs/HARDWARE.md, "The per-frame path's stages". It is a second method of normals, and is not
built without a measured need (AGENTS.md §7 rule 6).

## Accuracy

On the synthetic source's frames, depth unit 1 mm, the camera's intrinsics at each size and the
normals' starting parameters [M 2026-11-14: `tests/test_frame_pyramid.cpp`]:

* **The intensity** of known colors lies within 1.02e-7 of the Rec. 709 luma computed in double,
  and lay within 1.19e-7 of the prototype's before it was deleted (its record, "What the
  prototype did"); a pixel of level 2 inside one color within 1.94e-7.
* **The combining** agrees with the method computed in double from the level before as the GPU
  left it, wherever no pair of a window lies within 1° of the rule's threshold: the same pixels
  are valid, the points within 1.79e-7 m and the intensities within 1.69e-7, on every picture
  of the test.
* **On the planes**, the tilted one at 0.40 m to 0.80 m and one 2 m away, every point lies within
  half a depth unit along its ray of the plane at level 0, as the organized points do, and within
  0.32 of a unit at level 1 and 0.16 at level 2, the depth's rounding averaged.
* **Every point projects onto its own pixel**: within 0.201 px of its centre at level 1 and
  0.282 px at level 2, against the bounds 0.26 and 0.38 px at θ = 75°; where the whole window
  is combined,
  within 0.0235 px, and within 0.0016 px at level 1.
* **No point lies between two surfaces**: on three frames of the walk around the tabletop box at
  640x400 and one at 1280x800, every point of every level lies within 1.87 mm of a known surface
  (0.63 mm at level 0, the depth's rounding, 1.01 mm at level 1, 1.87 mm at level 2, at creases
  and steps the rule takes for one surface). Combined across the box's edges, without the rule,
  points of level 1 and 2 lie up to 62.5 mm off.

The normals of a coarser level tilt far less than the frame's, the largest and the median angle
to the plane's normal:

| plane | size | level 0 | level 1 | level 2 |
|---|---|---|---|---|
| tilted, 0.40 m to 0.80 m | 640x400 | 4.90° / 0.321° | 0.642° / 0.0623° | 0.130° / 0.0109° |
| | 1280x800 | 10.6° / 0.775° | 1.35° / 0.103° | 0.298° / 0.0160° |
| 2 m away | 640x400 | 1.49° / 0.118° | 0.395° / 0.0219° | 0.0449° / 0.00424° |
| | 1280x800 | 4.53° / 0.250° | 0.860° / 0.0386° | 0.209° / 0.00628° |

Level 0's are the normals' own (docs/methods/normals.md, "Accuracy"). On the steep part of the
tilted plane and toward the side of the plane at 2 m, seen above θ_1 = 61.8°, some adjacent
pixels of level 1 lie beyond the rule: there level 2 leaves them out, as the rule says.

## On the GPU

On the caller's stream, in this order, each a kernel of one thread for each pixel of its level,
in the stages' block of 32 x 8 threads [?: a choice, the user's, 2026-11-14]
(`src/gpu/kernel_launch.cuh`): the organized points of level 0 from the frame's depth; the
intensity of level 0 from the frame's color and the level's marks; the
normals of level 0; then for level 1 and level 2 the combining, which writes the level's points,
marks and intensity from the level before, and the level's normals. The step allocates nothing,
waits for nothing and reads nothing back; every launch goes through the stages' one launch
(`src/gpu/kernel_launch.cuh`). Work queued on the stream after it sees every level. Its time on
the board stands in docs/HARDWARE.md, "The per-frame path's stages".

## Refusals

Each comes back as a value with its reason; nothing is queued and every level is left as it was:

* the pyramid holds no levels;
* the normals' parameters (docs/methods/normals.md, "Refusals"), checked before anything is
  queued;
* the depth range and the frame, as the organized points refuse them
  (docs/methods/organized-points.md, "Refusals"); the frame's size must be level 0's.

CUDA's errors come back as a failure with CUDA's error, which is then cleared
(`src/gpu/gpu_messages.hpp`): one an earlier call left as CUDA's last error, before the first
launch, with nothing queued and every level as it was; and CUDA's refusal of a launch, with the
levels' contents then unspecified.

## What the prototype did, and what is not carried

The prototype registered at one scale and had no pyramid; it is deleted since, and its code
stands in git at 74dda68 (`src/geometry/`). Its intensity was the Rec. 709 luma of a point's
color, whose channels its point cloud from a frame scaled by 1/255 in float, computed in double
and rounded to float [S: the prototype's point type and point-cloud kernel at 74dda68]. On the
test's cells of known colors this stage's intensity lay within 1.19e-7 of it, against a
tolerance of 4 float steps, 4.8e-7 [M 2026-11-14: `tests/test_frame_pyramid.cpp` at 74dda68];
this is the record of the agreement. Carried: the coefficients and the scale to [0, 1]. Not
carried: the intensity computed anew from a point's color at every use, in double; here it is
one image of each level, computed once per frame in float.
