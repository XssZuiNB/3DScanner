# The depth noise: c from a still sequence

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The sparse TSDF's truncation follows the camera's depth noise, σ(z) = c z², at the measured depth z
(docs/methods/sparse-tsdf.md, "The truncation distance"); c stands there at the datasheet's bound
until it is measured. `depth_noise DIR` measures it offline from a sequence taken with the camera
at rest (`DIR/frames.sequence`, docs/SEQUENCE_FILE.md): each pixel's mean depth and its spread over
the frames, σ against z over depth bins, the fit of c with its spread, the depth unit's rounding
removed; beside it the mean frame's distance from a plane, the spatial error. It opens no camera
(AGENTS.md §3). The measured c, its date and its sequences go into "The truncation distance" and the
object classes' parameters, not here.

The camera rests on a firm support, aimed obliquely at a plain wall or the floor, so that one view
spans many depths, about 0.4 m to 3 m [?: a choice, `measure-depth-noise-on-the-camera`], with no
person in view. No distance is measured by hand: z is the camera's own mean depth at each pixel,
which a still sequence gives to a fraction of the noise, and the plane fit needs no known distance
either; a 1 % error in z would move c by about 2 %, since c = σ / z² [S: computed].

Built: the measurement (`apps/depth_noise_measure.*`), the command line
(`apps/depth_noise_options.*`), the program (`apps/depth_noise.cpp`), host only, in double; their
tests `tests/test_depth_noise_measure.cpp` and `tests/test_depth_noise_options.cpp`.

## Each pixel over the frames

The tool reads the sequence twice, through the sequence's one reader (`camera/sequence_reader.hpp`),
since the halves below need N before the sums. The first reading counts its N frames; the second
sums, frame k (0 to N - 1, in the file's order) going to the first half when k < N / 2, rounded
down, and to the second otherwise. A reading of 300 frames at 1280x800, 1.54 GB, takes about 2.4 s
at the 646 MB/s the sequence's writer reaches on the board's NVMe [S: computed from
docs/HARDWARE.md, "Memory, storage, network"; the read itself is not measured]. For each pixel
and each half it counts the frames with depth, n_1 and n_2, and sums the depth values centred on the
pixel's first value x_0, S_1 and S_2, and over both halves the squares Q; the values are whole
numbers, so the sums are exact in 64-bit integers [S: a value is at most 65535, a square below 2^32].

A pixel is kept when it has depth in at least 90 % of the N frames [?: initial] and in at least one
frame of each half. A pixel that loses its depth often lies on an edge, a dark spot or a speckle
the camera's own filter removes (docs/HARDWARE.md, "What the camera is and reports"), whose noise is
not the surface's [?]. For a kept pixel, with n = n_1 + n_2 and the depth unit u in m per value:

    z̄ = u (x_0 + (S_1 + S_2) / n)                    the mean depth, m
    s² = u² (Q - (S_1 + S_2)² / n) / (n - 1)          the variance over the frames, m²
    d  = u (S_2 / n_2 - S_1 / n_1)                     the second half's mean less the first's, m

## The depth unit's rounding

The camera gives depth in whole depth units, u = 1 mm at its setting (docs/HARDWARE.md). Rounding
adds about u²/12 to the variance of a depth whose noise is not small against u [S: W. F. Sheppard,
"On the Calculation of the most Probable Values of Frequency-Constants, for Data arranged according
to Equidistant Division of a Scale", Proceedings of the London Mathematical Society 29, 1898;
recalled, not re-read here], so the noise is taken as σ² = s² - u²/12. Below about u/2 the share
depends on where the true depth lies within its unit, and the correction fails. For Gaussian noise
of deviation σ rounded to u, the error of s² - u²/12 against σ² [S: computed, the rounded
distribution summed exactly at 400 offsets within the unit]:

| σ | at the worst offset | averaged over offsets spread evenly |
|---|---|---|
| u/4 | ±161 % | -6.9 % |
| u/3 | ±55 % | -0.57 % |
| u/2 | ±3.2 % | -0.001 % |
| 3u/4 | ±0.006 % | 0.000 % |

An oblique plane spreads a bin's depths evenly over the unit; a plane seen face-on need not. So a
bin's noise enters the fit only when it lies at or above u/2 [?: a choice]: its σ² then holds within
3.2 %, its σ within 1.6 %, at any offset. At u = 1 mm that is σ ≥ 0.5 mm, reached at 0.50 m for
c = 0.002 /m and at 0.22 m for 0.01 /m [S: computed, z = (0.0005 m / c)^(1/2)]. A bin below it is
listed as below the rounding and left out.

## Depth bins and the fit of c

The kept pixels go into depth bins of 0.1 m [?: initial] by their mean depth: bin j holds
j × 0.1 m ≤ z̄ < (j + 1) × 0.1 m. A bin counts when it holds at least 1000 kept pixels
[?: initial]. For each bin that counts:

    σ_b = (mean(s²) - u²/12)^(1/2)                    its noise, m
    z_b = mean(z̄)                                      its depth, m
    c_b = (mean((s² - u²/12) / z̄⁴))^(1/2)              when σ_b ≥ u/2: each pixel's own c², averaged

Each pixel divides by its own z̄⁴, so the bin's width does not bias c_b: across a bin at 0.5 m,
z² changes by ±20 % [S: computed, (0.55 / 0.5)² = 1.21, (0.45 / 0.5)² = 0.81].

**c** is the least-squares fit of σ = c z² to the fitted bins, at their depths, through the
origin, each bin's σ taken as c_b z_b²:

    c = Σ c_b z_b⁴ / Σ z_b⁴

It weighs the far bins most, where the truncation μ = k c z² is widest and passes μ_min, beyond
0.37 m to 1.03 m by the object classes' voxel sizes [S: computed, docs/methods/sparse-tsdf.md, "The
truncation distance", at c = 0.01 /m]. The fit needs at least 3 bins [?: initial]. **The spread** is the root mean square of
c_b - c over the fitted bins. Where σ follows c z², it holds only the statistical error: one
pixel's s² has a relative deviation of (2 / (n - 1))^(1/2), 8.2 % at n = 300 frames [S: the
chi-square distribution of a Gaussian's sample variance], raised by the rounding's share near u/2,
and a bin averages over at least 1000 pixels, fewer independent ones where the camera's noise is
correlated between neighbours [?]. A spread beyond that says σ does not follow c z² across the
range; the bins' lines show where.

## The spatial error

The mean frame's points, P = z̄ ((u - cx) / fx, (v - cy) / fy, 1) for each kept pixel (u, v) by the
one pinhole convention (types/camera_param.hpp), get the orthogonal least-squares plane
(`fit_plane()`, separation/plane_fit.hpp, the one copy), centroid p_0 and normal n. Each pixel's
offset from it along its ray, in depth as σ is:

    e = z̄ - (n · p_0) / (n · r),   r = ((u - cx) / fx, (v - cy) / fy, 1)

A mean over n frames still holds the temporal noise's share s² / n, which the spatial error leaves
out, so that only the mean frame's own error remains:

    spatial_b = (max(0, mean(e² - s² / n)))^(1/2)       over a bin's pixels; and over every kept pixel

It holds the camera's error that does not average out over time and the surface's own unevenness
alike [?: a wall's flatness is unmeasured]. The truncation does not take it: from one pose it is a
fixed offset of the surface, which other poses see elsewhere.

A plane without noise cannot show this method's own spatial error: each of its pixels keeps one
rounded value in every frame, so its mean frame is a staircase of the depth unit, about u / √12 rms
from the plane, 0.29 mm at 1 mm [S: computed, the rounding error spread evenly over ±u/2], which
noise of σ ≥ u/2 dithers away (the table above). Its noise lies below the rounding everywhere, so
it is refused ("Refusals"); the test takes the spatial error of noisy planes instead.

## Still

A camera that moved between the frames widens every pixel's spread with the motion. The halves'
means test it: for a still camera whose frames' noise is independent, d has the variance
s² (1/n_1 + 1/n_2) [S: the variance of a difference of two independent means], so

    T = (Σ d² / Σ s² (1/n_1 + 1/n_2))^(1/2)            over the kept pixels

lies near 1, and the sequence is refused as not still above 3 [?: initial]. At T = 3 with 300
frames, a drift alike at every pixel moves the halves' means apart by D = 0.35 σ
[S: computed, 3 (2 / 150)^(1/2)]; spread evenly over the sequence it adds D² / 3 to the variance,
which raises σ by 2 % [S: computed, a ramp of 2 D has the variance (2 D)² / 12].

T sums over all kept pixels, so it admits more drift in the near bins than in the far ones when
the drift is not alike at every pixel. A drift along the camera's axis moves each pixel's depth in
proportion to it, while the noise grows with its square: on the test's plane below ("Verification")
seen at 1280x800 over 300 frames with c = 0.002 /m, T = 3 lets the halves differ by 1.1 mm per m
of depth, which raises c_b by 15 % in the bin at 0.5 m, by 0.9 % in the bin at 2.4 m, and c by
1.3 % [S: computed, the ramp's variance added to each pixel's]. The bins' lines show such a rise
toward the near end. Noise correlated
from frame to frame, a temporal filter of the camera's for instance, raises T with no motion [?];
whether the camera has one is unmeasured.

## Refusals

In this order, each as `depth_noise: <the sequence's path>: <reason>`:

1. the sequence cannot be read, with the reader's reason (it cannot be opened, is no sequence, is
   of another version, ends early, holds a record it does not know, among others); a sequence
   that ends early is refused too, since a still sequence is quickly taken again [?: a choice];
2. `its depth unit, <u> m per value, is no positive number`: the depth unit is the sequence's own,
   from its header, never assumed, and z and the rounding's share both take it;
3. `its intrinsics, fx <fx> fy <fy> cx <cx> cy <cy> px, are no pinhole camera's`
   (`usable_intrinsics()`): the plane fit takes them;
4. `too few pixels: <B> depth bins of 0.1 m hold at least 1000 of its <K> pixels kept, with depth
   in at least 90 % of the <N> frames and in both halves; the fit needs 3`;
5. `not still: the halves differ by <D> mm rms, <T> times their standard error; a still sequence
   stays within 3 times`;
6. `no plane fits the mean frame's <K> points`, which cannot happen after 4: the bins that count
   hold at least 3000 points, and a plane needs three;
7. `below the rounding: the noise lies below 0.5 depth units, <u/2> mm, in <k> of the <B> depth
   bins that hold enough pixels; the fit needs 3 above it`.

Each reason is one line; the ones above are broken here only to fit the page. `depth_noise` exits
0 with its lines, 1 with a refusal, 2 for a command line it refuses (an argument it does not know,
`--help` too, a second directory, none, one that does not exist), each reason on stderr, a refused
command line's followed by the usage.

## Output

The lines `measure-depth-noise-on-the-camera` reads, an interface (AGENTS.md §7 rule 5):

    depth_noise: <DIR>: <N> frames of <W>x<H>, depth unit <u> mm
    pixels: <K> of <W*H> kept, with depth in at least 90 % of the frames and in both halves
    still: the halves differ by <D> mm rms, <T> times their standard error
    spatial: <S> mm rms from the plane fitted to the mean frame
    bin <a> to <b> m: <n> pixels, noise <σ_b> mm, spatial <spatial_b> mm, c <c_b> /m
    bin <a> to <b> m: <n> pixels, noise <σ_b> mm, spatial <spatial_b> mm, below the rounding
    c: <c> /m, spread <r> /m over <B> bins

one line for each bin that counts, nearest first; D, σ_b and the spatial errors to 0.01 mm, T to
0.01, the bins' ends to 0.1 m, c, c_b and the spread to 0.000001 /m.

## What it does not cover

* The angle of incidence: an oblique view sees its far depths at a grazing angle, where a stereo
  camera's noise may grow beyond c z² [?]; the bins' c_b then rise with depth, which the spread
  shows. The c measured so is then too large for a surface seen face-on, the safe side for the
  truncation.
* The noise is measured along the camera's axis, as the TSDF's distance η is taken
  (docs/methods/sparse-tsdf.md, "Integration, and the weight of an observation").
* One stream size per sequence: the camera's two sizes are measured from a sequence each.

## Verification

`tests/test_depth_noise_measure.cpp` writes its sequences through the sequence's writer: a plane
seen at 45° from 0.6 m above it [?: a choice], 0.51 m to 2.4 m deep [S: computed], 60 frames of
320x200 (the camera's 640x400 intrinsics halved), rendered by the tests' synthetic source with its Gaussian depth noise of a known
c, independent at every pixel, rounded to 1 mm [M 2026-11-15: the test, seed 7, 5.4 s on the board]:

| known c | c measured | the bins' c_b, largest error | spread | T | spatial error |
|---|---|---|---|---|---|
| 0.002 /m | 0.002003 /m, +0.14 % | 0.38 %, 14 bins | 0.23 % of c | 0.99 | 0.07 mm |
| 0.01 /m | 0.010014 /m, +0.14 % | 0.39 %, 16 bins | 0.23 % of c | 0.99 | 0.35 mm |
| 0.0005 /m | 0.000501 /m, +0.18 % | 0.41 %, 9 bins; the 5 from 0.5 m to 1.0 m below the rounding | 0.27 % of c | 0.99 | 0.08 mm |

The test's tolerances [?: a choice, each from the measurement above]: c within 0.5 %, each c_b within 1.5 %, the spread within 1 % of c, T within
1 ± 0.05, the spatial error within 0.15 mm per 0.002 /m of c, and at 0.0005 /m within 0.3 mm, the
depth unit's staircase u / √12 [S: computed] that the bins below the rounding keep in part. The spatial error of the exactly flat
plane is the estimate's statistical floor: the temporal share removed, s² / n, is itself measured
from n frames, and the floor scales with c as the noise does. A noise-free plane is refused as
below the rounding (its 14 bins of 14); a camera moving 20 mm forward along its axis over the 60
frames [?: a choice] as not still, at T = 6.09; each other refusal and the program's three exits
have their cases, and `tests/test_depth_noise_options.cpp` the command line's refusals.
