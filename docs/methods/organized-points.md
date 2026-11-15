# Organized points from depth

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The per-frame path's first stage after the upload: a frame's depth picture on the GPU becomes
organized points in the camera frame and a validity mask, by the one pinhole convention, inside
a depth range. The code is `src/geometry/organized_points.*`, its test
`tests/test_organized_points.cpp`.

## Input

* **A frame on the GPU** (`gca::gpu_frame`, `src/gpu/gpu_frame.hpp`), uploaded on the stream
  the stage runs on. Its depth picture holds a 16-bit value d for each pixel; d · unit is the
  depth in m along the camera's z axis, and d = 0 means the camera measured none
  (`src/camera/frame.hpp`). Its description gives the size W x H in px, the intrinsics fx, fy,
  cx, cy in px and the depth unit in m.
* **The depth range** [near, far], in m: the stage's parameters
  (`gca::organized_points_parameters`), 0 <= near < far. Each object class looks at its own
  range (GLOSSARY.md, "Object class"); the caller passes it.

## Output

Organized: one entry for each pixel (u, v), u = 0 to W - 1 left to right, v = 0 to H - 1 top
to bottom, at index v · W + u, the frame's own index. The depth is aligned to the color, so a
point's color is the frame's color at the same index; the stage copies no color.

* **The point** p(u, v), in the camera frame (x right, y down, z forward, in m), as three
  floats.
* **The validity mark**: 1 for a valid pixel, 0 for an invalid one. An invalid pixel's point is
  (0, 0, 0); a valid point's z is at least one depth unit and at least near, so z > 0 exactly
  where the mark is 1 [S: the rules below].

Both are GPU images (`src/gpu/gpu_image.hpp`) of the frame's size with dense rows: 12 bytes a
pixel for the points and 1 for the marks, 3.3 MB at 640x400 and 13.3 MB at 1280x800
[S: computed]. They are requested from the pool (`src/gpu/gpu_pool.hpp`) before it closes; the
stage itself asks for no memory and takes no pool.

The class `gca::organized_points` holds the images of one other kind of points: a coarser level
of the frame's pyramid, whose points the pyramid combines from the level before, with the same
meaning of point and mark (docs/methods/frame-pyramid.md). That is its second writer, through
a key only the pyramid makes (`gca::organized_points::pyramid_key`).

## Deprojection

By the one pinhole convention (`src/types/camera_param.hpp`; docs/HARDWARE.md, "Pixel
convention"), pixel centres lie at whole numbers and the ray of pixel (u, v) is
r(u, v) = ((u - cx) / fx, (v - cy) / fy, 1). With z = d · unit, the depth along the camera's z
axis, the pixel's point is

    p(u, v) = z r(u, v) = ((u - cx) z / fx, (v - cy) z / fy, z).

This is the pinhole model's back-projection of a pixel at a known depth [S: Hartley and
Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., 2004, chapter 6, "Camera
Models"], and the vertex map V(u) = D(u) K⁻¹ u̇ of KinectFusion with its validity mask
[S: Newcombe et al., "KinectFusion: Real-time dense surface mapping and tracking", ISMAR 2011,
"Surface measurement"], without that paper's bilateral filter.

The arithmetic is float: d converts exactly (d <= 65535 < 2^24), z = d · unit is rounded once,
and each of x and y takes a few roundings, a relative error of a few times 6e-8, below 1 µm
within 3 m [S: float's 24-bit significand]. Half a depth unit from the depth's own quantization,
0.5 mm at 1 mm, is far larger. The test's tolerances, with the errors measured, stand above
them in `tests/test_organized_points.cpp`.

## Validity

A pixel is valid exactly when both hold:

1. **d ≠ 0**: the camera measured a depth there. A pixel without depth is invalid whatever the
   range, also when near is 0.
2. **near <= z <= far**, compared on z as the stage computes it, the rounded float, so that
   every valid point's z lies within the range as stored.

Nothing else makes a pixel invalid: no filter runs and no neighbour is looked at. Depth edges
and isolated pixels are the business of the stages that look at neighbours, the normals first.

## Refusals

Each comes back as a value with its reason; nothing is queued and the output is left as it
was:

* the parameters: near or far not finite, near < 0, or far <= near;
* the frame: it holds no frame yet (its description's size is not its pictures'), its size is
  not the output's, fx or fy is not a finite number > 0, cx or cy is not finite, or the depth
  unit is not a finite number > 0;
* the output holds no images.

CUDA's errors come back as a failure with CUDA's error, which is then cleared
(`src/gpu/gpu_messages.hpp`). An error an earlier call left as CUDA's last error is reported
before the launch, with nothing queued and the output as it was; otherwise it would come back
after the launch as if the launch had failed. When CUDA refuses the launch itself, the output's
contents are unspecified.

## On the GPU

One kernel on the caller's stream, one thread for each pixel, in blocks of 32 x 8 threads
[?: a choice, the user's, 2026-11-14]. It reads the depth picture and writes both images. The
stage allocates nothing, waits for nothing and reads nothing back: the host checks the arguments,
asks CUDA whether an
error is set (`cudaPeekAtLastError`), launches, and asks whether the launch was accepted
(`cudaGetLastError`); neither waits. Work queued on the stream after it sees the points. Its
time on the board stands in docs/HARDWARE.md, "The per-frame path's stages".

## What the prototype did, and what is not carried

The prototype's point cloud from a frame, CUDA_3D's, was the oracle until this stage agreed with
it within a stated tolerance (below); it is deleted since, and its code stands in git at 74dda68
(`src/geometry/`). Its deprojection and depth range carry over; these do not:

* **The half pixel.** It deprojected pixel (u, v) at (u - 0.5, v - 0.5), half a pixel off the
  convention OrbbecSDK's own code uses for its intrinsics (docs/HARDWARE.md, "Pixel
  convention"), so its point lay (0.5 z / fx, 0.5 z / fy, 0) from this stage's. At 1 m that is
  0.82 mm along x and along y, 1.15 mm apart, with fx = 612.37 px at 1280x800, and 1.63 mm along
  each axis, 2.31 mm apart, with fx = 306.18 px at 640x400 [S: computed; the intrinsics,
  docs/HARDWARE.md].
* **The bilateral filter**, always on, its constants tuned for RealSense: no measured benefit
  on this camera (AGENTS.md §7 rule 6).
* **NaN points at a minimum depth of 0.** A pixel without depth then passed the range and went
  through the filter, whose sum over neighbours with depth was empty, 0 / 0 [S: the
  prototype's point-cloud kernel at 74dda68, its bilateral filter]. Here rule 1 makes it
  invalid.
* **No check of the frame's size** against the camera parameters: the kernel took the size
  from the parameters alone. Here the frame's size, its description's and the output's are
  compared.
* **A new buffer every frame**: an allocation and fill of the point buffer, a
  `cudaDeviceSynchronize`, and a compaction that dropped the invalid points. Here the buffers
  come from the pool before the scan, nothing waits, and the output stays organized.
* **The color in each point.** It projected the point, deprojected at u - 0.5, back to the
  color picture and took the pixel `static_cast<int>(u_c + 0.5f)` of the result u_c, which
  lies on the boundary between two pixels, so that float's rounding picked one of them
  [S: the prototype's point-cloud kernel at 74dda68, its color lookup]. Here the color stays in
  the frame's picture at the point's index.

**The comparison with the oracle** held where its filter was the identity. The filter's radius
was floor((z - near) / 0.5 m) px, at most 5, and a radius of 0 returned the pixel's own depth
[S: the prototype's point-cloud kernel at 74dda68, its adaptive bilateral filter]; outside the
range it was not applied. A range narrower than 0.5 m therefore turned it off on the whole
frame, and the test ran both on such a range. There every valid pixel's point equalled the
oracle's point of the same pixel plus (0.5 z / fx, 0.5 z / fy, 0), and the two kept the same
pixels: on three frames of a walk around a box at 640x400, 377831 points, each within 9.7e-8 m
of the oracle's plus the half pixel, against a tolerance of 1e-6 m, float's rounding of the two
points [M 2026-11-14: `tests/test_organized_points.cpp` at 74dda68, the range 0.3005 m to
0.7995 m]. This is the record of the agreement; the prototype and the comparison are deleted.
