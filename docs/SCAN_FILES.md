# The files a scan keeps

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The formats of a scan's model, trajectory and parameters (GLOSSARY.md, "What a scan keeps").
Each format has one writer and one reader, in `src/scan/` (`model_file`, `trajectory_file`,
`parameters_file`); the programs that keep or read a scan's files use these. ctest `scan_files`
pins each format by writing it and reading it back, and reads the examples below as written
here. Each format has one version: a change to a format changes this document, its writer, its
reader and that test together, and a file written before the change is not read (AGENTS.md §7
rule 6).

## Writing and reading

A writer writes the whole file into `<name>.part` beside it (created, or truncated when one is
left), syncs it to disk, renames it onto `<name>`, replacing a file of that name, and syncs the
directory, so that the rename survives a power cut. When a step before the rename fails, the
writer removes `<name>.part` and leaves a file already at `<name>` as it was. So a file found
under its name is whole. A power cut during a write can leave `<name>.part`; no reader takes
it, and the next write of `<name>` truncates it.

A writer and a reader never throw. Each one returns its failure as a value, with a reason that
names the file and, for a reader, the line, the point, the entry or the key. A writer refuses
a value its reader would refuse, and then writes nothing.

## The models' frame

Positions and poses are in the models' frame, the camera frame of the first fused frame: x
right, y down, z forward, in m (`src/types/camera_param.hpp`, the one pinhole convention). A
pose is `camera_pose` (`src/types/camera_pose.hpp`), which a sequence holds too
(docs/SEQUENCE_FILE.md).

## Models

The full model and the object model are binary little-endian PLY. The header is exactly these
lines, each ended by `\n`, with no comment line; `<count>` is the number of points in decimal:

    ply
    format binary_little_endian 1.0
    element vertex <count>
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    property uchar red
    property uchar green
    property uchar blue
    end_header

The data follows, 27 bytes for each point, in the writer's order, each float an IEEE 754 binary32
with its least significant byte first:

| bytes | what |
|---|---|
| 0 to 11 | the position x, y, z, in m |
| 12 to 23 | the unit normal nx, ny, nz; its length, computed in double precision, lies within 1e-6 of 1 [?: a choice: eight float32 steps of 1.19e-7, S: FLT_EPSILON] |
| 24 to 26 | the color red, green, blue, 0 to 255 |

The reader refuses a header that differs in any line, naming the line; a file whose data is not
27 bytes for each point the header declares; and a normal that is not unit within 1e-6, naming
the point. The tests' own PLY reader (`tests/ply_reader.hpp`) reads this layout too.

## Trajectory

Text. The first line is the header, `# 3dscanner trajectory 1`, which names the format and its
version. Then there is one line for each frame the scanner took, in the order taken:

    <index> <timestamp_us> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
    <index> <timestamp_us> lost

The first form is a tracked frame. The second is a frame in tracking loss; `lost` is the mark of
tracking loss.

- `<index>` is the frame's index and `<timestamp_us>` its timestamp in µs on the frame source's
  clock (`src/camera/frame.hpp`): decimal integers from 0 to 2^64 - 1.
- `<tx> <ty> <tz>` is the camera's centre in the models' frame, in m.
- `<qx> <qy> <qz> <qw>` is the camera's orientation, a unit quaternion in Hamilton's convention
  (Eigen's), in the order x, y, z, w. It turns a vector of the camera frame into the models'
  frame (camera to model): a point p of the camera frame lies at q p q* + t in the models'
  frame. q and -q are the same rotation; the file keeps the sign it was given.
- Fields are separated by one space, and every line ends in `\n`, the last one too. Numbers are
  written in their shortest decimal form that reads back to the same double
  (`std::to_chars`), independent of the locale; a reader gets every bit back
  [M 2026-11-14: 10^6 random doubles, g++ 11.4].
- A pose has a finite position, and its quaternion's norm lies within 1e-9 of 1 [?: a choice;
  a pose solve in double precision leaves a few 1e-16]. The writer refuses another pose and
  writes nothing. The reader refuses it, naming the line.

The reader refuses a first line other than the header, a line of neither form, and a last line
without its `\n`, each naming the line.

Example, seven distinct values in the tracked line: frame 7 with the camera's centre at
(0.5, -1.25, 2) m and the quaternion x = 0.2, y = 0.4, z = 0.16, w = 0.88 (a unit one:
0.04 + 0.16 + 0.0256 + 0.7744 = 1), then frame 9 in tracking loss:

    # 3dscanner trajectory 1
    7 233333 0.5 -1.25 2 0.2 0.4 0.16 0.88
    9 300000 lost

## Parameters

JSON, written and read with OpenCV's `cv::FileStorage` on memory (ADR 0002: no other JSON
library). Example, a large object's scan whose set differs from the object class's own in five
values, so that every value of it is another:

    {
        "format_version": 1,
        "commit": "db91e0d",
        "object_class": "large_object",
        "parameters": {
            "live_voxel_size_m": 8.0000000000000002e-03,
            "full_voxel_size_m": 4.0000000000000001e-03,
            "min_depth_m": 2.5000000000000000e-01,
            "max_depth_m": 3.0,
            "memory_budget_bytes": 2000000000.0,
            "too_close_depth_m": 4.0000000000000002e-01,
            "max_distance_level_0_m": 1.0000000000000001e-01,
            "max_distance_level_1_m": 2.0000000000000001e-01,
            "max_distance_level_2_m": 3.4999999999999998e-01,
            "max_iterations_level_0": 20.0,
            "max_iterations_level_1": 10.0,
            "max_iterations_level_2": 5.0,
            "huber_threshold_m": 2.9999999999999999e-02,
            "point_to_plane_weight": 9.6799999999999997e-01,
            "min_fitness": 5.9999999999999998e-01,
            "max_inlier_rmse_m": 2.0000000000000000e-02,
            "max_turn_rate_deg_s": 90.0,
            "max_speed_m_s": 1.5000000000000000e+00
        }
    }

| key | what |
|---|---|
| `format_version` | the integer 1 |
| `commit` | a string, the tree's commit the scanner was built from |
| `object_class` | the object class the operator chose: `tabletop_object` or `large_object` |
| `parameters` | the set of parameters in force, an object of the eighteen keys below |
| `live_voxel_size_m` | the voxel edge of the live model, in m, finite and > 0 |
| `full_voxel_size_m` | the voxel edge of the full models, in m, finite and > 0 |
| `min_depth_m` | the nearest depth a frame contributes, along the camera's z, in m, finite and > 0 |
| `max_depth_m` | the farthest depth, in m, finite and greater than `min_depth_m` |
| `memory_budget_bytes` | all of the device memory of the scan's sparse TSDF, its pool of blocks, its hash and their lists and counts (docs/methods/sparse-tsdf.md, "The pool and the memory budget"), in bytes, a whole number from 1 to 2^53 |
| `too_close_depth_m` | the "too close" notice's threshold, along the camera's z, in m: a frame is too close when at least a quarter of its pixels with depth lie nearer (R36); greater than `min_depth_m` and less than `max_depth_m` |
| `max_distance_level_0_m`, `max_distance_level_1_m`, `max_distance_level_2_m` | the registration's d_max at levels 0, 1 and 2 (docs/methods/registration.md, "Parameters"), in m, finite and > 0 |
| `max_iterations_level_0`, `max_iterations_level_1`, `max_iterations_level_2` | the registration's most iterations n_l at levels 0, 1 and 2, a whole number from 0 to 2^32 - 1 |
| `huber_threshold_m` | the registration's δ_G, in m, finite and > 0 |
| `point_to_plane_weight` | the registration's λ, a share, > 0 and at most 1 |
| `min_fitness` | tracking's f_track, the least fitness (docs/methods/tracking.md, "Parameters"), a share, > 0 and at most 1 |
| `max_inlier_rmse_m` | tracking's e_track, the largest inlier RMSE, in m, finite and > 0 |
| `max_turn_rate_deg_s` | tracking's ω_max, the hand's largest turn rate, in degrees a second, finite and > 0 |
| `max_speed_m_s` | tracking's v_max, the hand's largest speed, in m/s, finite and > 0 |

The ranges each stage takes these values in are the stage's own, narrower than the file's: the
scan's request of its buffers refuses a value outside them, naming it (`src/tracking/tracker.hpp`,
`src/fusion/sparse_tsdf.hpp`), so a file can hold a set the scanner does not run with.

Every key stands in every file; there is no optional key. A field added later that can lack a
value states its empty form here when it is added, so the file keeps one version. A file written
before the thirteen keys from `too_close_depth_m` on joined the set (2026-11-15, evening) lacks
them, and the reader refuses it: `no key "too_close_depth_m" in "parameters"`.

Each object class's own set stands in `src/scan/object_class.cpp`, each value with its
evidence. A scan stores the set it was made with, and a stored set is read back as it stands,
so the set in a file need not be its class's own.

The writer writes each parameter as a double with 17 significant digits, or as a whole number
with `.0`, and the reader gets every bit back [M 2026-11-14: OpenCV 4.8.0, 200 000 random
doubles]. A budget up to 2^53 and a count up to 2^32 - 1 are doubles exactly [S: IEEE 754
binary64, a 53-bit significand].

**Strings.** OpenCV's JSON writer gives text that both its own reader and Python's `json.loads`
take back unchanged only for some strings [M 2026-11-14: OpenCV 4.8.0, Python 3.10]: it writes
an apostrophe as `\'`, which is no JSON escape; a string that starts and ends with a double
quote raw, without quotes of its own, which OpenCV's reader then reads without them; a control
byte other than `\b \f \n \r \t` raw, which is no JSON; and a string of more than 4095 characters
does not read back (4097 make the writer throw). So a string in the file is at most 4095
characters of printable ASCII (0x20 to 0x7e) without the apostrophe, and does not both start and
end with a double quote. A byte above 0x7e is refused too, as the reader checks no UTF-8. A
commit hash and the object classes' names lie in this set. The writer refuses another string
and the reader refuses one, naming the key.

The reader refuses, naming the key:

- a text whose first byte after any white space is not `{` (YAML and XML among them, which
  OpenCV's reader takes whatever format it is told) [M 2026-11-14: OpenCV 4.8.0];
- a `format_version` other than the integer 1; it checks this first, so a file of another
  version is refused as such;
- a key it does not know, a key that stands twice, and a missing key, at the top level and in
  `parameters`;
- a value of another type, an unknown object class, a string outside the set above, a
  parameter out of the range above, and a count of iterations that is no whole number from 0 to
  2^32 - 1;
- a parameter written as an integer: OpenCV reads an integer beyond 2147483647 wrapped to 32
  bits without a word, so every parameter is a number with a decimal point or an exponent;
- a text that OpenCV's parser does not read.

What OpenCV's parser still does that strict JSON does not [M 2026-11-14: OpenCV 4.8.0]:

- it reads an integer modulo 2^32 and with C's prefixes, so a `format_version` of 4294967297,
  `0x1` or `01` reads as 1, and `010` as 8; it reads `true` as the integer 1 and `false` as 0;
- it accepts text after the closing brace, a comma before a closing brace, a `//` comment, a
  number written `+1`, and `.Nan` (which the range check then refuses);
- it refuses `null`, the escapes `\/` and `\uXXXX`, and an exponent without a fraction written
  with a capital `E` (`3E2`).
