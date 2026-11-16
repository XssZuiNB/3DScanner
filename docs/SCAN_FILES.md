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

## A scan's directory

Each scan has a directory of its own in the scans directory (`~/3dscanner-scans/`, a start option
of the service, the spec), named by the scan's start time in UTC, `YYYYMMDDTHHMMSS.ffffffZ`
(`src/scan/utc_time.hpp`), so that the names sort as the scans began. The scan interface
(`src/scan/scan_interface.hpp`) makes it once the frame source has given its first frame, and
writes into it:

| file | what | when |
|---|---|---|
| `frames.sequence` | the sequence (docs/SEQUENCE_FILE.md): every frame taken but the one tracking failed on, with its pose or the mark of tracking loss | while the scan runs |
| `parameters.json` | the parameters ("Parameters" below) | at the scan's stop, before refining |
| `trajectory.txt` | the trajectory ("Trajectory" below), as refinement read it back from the sequence | after refining |
| `full.ply` | the full model ("Models" below) | after refining |

A scan ends as one of these, which its directory shows:

* **refined**: the four files;
* **unfinished**: the sequence and the parameters, without a trajectory or a full model, when
  refinement failed (the scan interface's state says why), when the scan interface ended during
  the scan (`end` `shut_down`) or during refinement, or when the trajectory or the full model could
  not be written (a trajectory written before a full model that failed is removed again); when the
  parameters could not be written, the sequence alone;
* **discarded**: nothing, the directory removed, at a discard while scanning or refining, one
  that comes during refinement's last steps included; a discard given while the scan starts is
  answered after its first frame. When the directory cannot be removed, the scan is unfinished,
  with the reason in the scan interface's state, never reported as discarded.

A start that fails (the GPU's buffers refused, the source refusing to open, giving no first frame
or describing itself with a string the parameters cannot hold) leaves nothing on disk. The
sequence stays after refining; the switch that deletes it then (R18) is not built.

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
version. Then there is one line for each frame the scanner took, in the order taken, but for
the frame tracking failed on, which ends a scan (`end` `tracking_failed` in the parameters) and
has neither a pose nor the mark of tracking loss:

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
library). It holds the set of parameters the scan ran with and the scan's facts. Example, a large
object's scan whose set differs from the object class's own in eight values, so that every value
of it is another, stopped by the operator after five minutes:

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
            "depth_noise_per_m": 4.0000000000000001e-03,
            "truncation_noise_multiple": 2.5000000000000000e+00,
            "truncation_min_voxels": 3.0,
            "normals_radius_px": 2.0,
            "normals_min_points": 16.0,
            "normals_max_incidence_deg": 70.0,
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
        },
        "stream": {
            "width_px": 640.0,
            "height_px": 400.0,
            "fx_px": 5.1725000000000000e+02,
            "fy_px": 5.1750000000000000e+02,
            "cx_px": 3.1975000000000000e+02,
            "cy_px": 2.0125000000000000e+02,
            "depth_unit_m": 1.0000000474974513e-03
        },
        "source": {
            "camera": "Orbbec Gemini 2 L",
            "firmware_version": "1.5.02",
            "sdk_version": "2.10.6"
        },
        "start_utc": "20261116T093012.250000Z",
        "stop_utc": "20261116T093512.750000Z",
        "frames": {
            "taken": 8990.0,
            "fused": 8950.0,
            "lost": 40.0,
            "not_taken": 12.0
        },
        "end": "stopped"
    }

| key | what |
|---|---|
| `format_version` | the integer 1 |
| `commit` | a string, the tree's commit the scanner was built from; a program's is its build's (`apps/build_commit.hpp`): the short hash, with `-dirty` when the tree held changes not committed or files untracked by git, or `unknown` when built without the git repository |
| `object_class` | the object class the operator chose: `tabletop_object` or `large_object` |
| `parameters` | the set of parameters in force, an object of the twenty-four keys below |
| `live_voxel_size_m` | the voxel edge of the live model, in m, finite and > 0 |
| `full_voxel_size_m` | the voxel edge of the full models, in m, finite and > 0 |
| `min_depth_m` | the nearest depth a frame contributes, along the camera's z, in m, finite and > 0 |
| `max_depth_m` | the farthest depth, in m, finite and greater than `min_depth_m` |
| `memory_budget_bytes` | all of the device memory of the scan's sparse TSDF, its pool of blocks, its hash and their lists and counts (docs/methods/sparse-tsdf.md, "The pool and the memory budget"), in bytes, a whole number from 1 to 2^53 |
| `too_close_depth_m` | the "too close" notice's threshold, along the camera's z, in m: a frame is too close when at least a quarter of its pixels with depth lie nearer (R36); greater than `min_depth_m` and less than `max_depth_m` |
| `depth_noise_per_m` | the sparse TSDF's c, the depth noise's coefficient, σ(z) = c z², in 1/m (docs/methods/sparse-tsdf.md, "The truncation distance"), finite and > 0 |
| `truncation_noise_multiple` | the sparse TSDF's k, the depth noise's standard deviations within the truncation distance, finite and > 0 |
| `truncation_min_voxels` | the sparse TSDF's μ_min, the least truncation distance, in voxels, finite and > 0 |
| `normals_radius_px` | the normals' window's half size R, in px (docs/methods/normals.md, "Input"), a whole number from 0 to 2^32 - 1 |
| `normals_min_points` | the normals' fewest points of a neighbourhood that give a normal, a whole number from 0 to 2^32 - 1 |
| `normals_max_incidence_deg` | the normals' steepest incidence at which two neighbouring points still count as one surface, in degrees, finite and > 0 |
| `max_distance_level_0_m`, `max_distance_level_1_m`, `max_distance_level_2_m` | the registration's d_max at levels 0, 1 and 2 (docs/methods/registration.md, "Parameters"), in m, finite and > 0 |
| `max_iterations_level_0`, `max_iterations_level_1`, `max_iterations_level_2` | the registration's most iterations n_l at levels 0, 1 and 2, a whole number from 0 to 2^32 - 1 |
| `huber_threshold_m` | the registration's δ_G, in m, finite and > 0 |
| `point_to_plane_weight` | the registration's λ, a share, > 0 and at most 1 |
| `min_fitness` | tracking's f_track, the least fitness (docs/methods/tracking.md, "Parameters"), a share, > 0 and at most 1 |
| `max_inlier_rmse_m` | tracking's e_track, the largest inlier RMSE, in m, finite and > 0 |
| `max_turn_rate_deg_s` | tracking's ω_max, the hand's largest turn rate, in degrees a second, finite and > 0 |
| `max_speed_m_s` | tracking's v_max, the hand's largest speed, in m/s, finite and > 0 |
| `stream` | the frames' size, intrinsics and depth unit, the stream size the frame source was opened for (`src/types/frame_description.hpp`), an object of the seven keys below, each a float of the frame's description written as a double |
| `width_px`, `height_px` | the frames' size, in px, a whole number from 1 to 2^32 - 1 |
| `fx_px`, `fy_px` | the focal lengths, in px, finite and > 0 as floats |
| `cx_px`, `cy_px` | the principal point, in px, pixel centres at whole numbers (`src/types/camera_param.hpp`), finite as floats |
| `depth_unit_m` | meters per depth value, finite and > 0 as a float |
| `source` | what delivered the frames (`src/camera/frame_source.hpp`, `describe()`), an object of the three strings below; a source without a version gives the empty string |
| `camera` | the camera's name, or the kind of source: a sequence's says `sequence` and its path |
| `firmware_version` | the camera's firmware's version |
| `sdk_version` | OrbbecSDK's version |
| `start_utc`, `stop_utc` | the times the scan started and stopped, in UTC to the microsecond as ISO 8601's basic format, `YYYYMMDDTHHMMSS.ffffffZ`, from `19700101T000000.000000Z` to `99991231T235959.999999Z` (`src/scan/utc_time.hpp`); the system's clock, which counts no leap second and may be set back, so the stop is not checked against the start |
| `frames` | the frames of the scan, an object of the four counts below, each a whole number from 0 to 2^53 |
| `taken` | the frames the scanner took, each of which entered the per-frame path: those fused, those lost, and, when the scan ended because tracking failed (`end` is `tracking_failed`), the frame it failed on, which tracking gave neither a pose nor tracking loss and which is in neither the sequence nor the trajectory; so `taken` is `fused` and `lost` together, and 1 more then |
| `fused` | the frames tracking passed and fused into the live model |
| `lost` | the frames in tracking loss, of which nothing was fused |
| `not_taken` | the frames the scanner did not take: those the frame source delivered while the sequence's writer had no room, and the gaps in the frames' indices, frames the source did not deliver |
| `end` | why the scan ended, one key of the closed set below; a discarded scan has no parameters and so no end |

Why a scan ended is one of seven keys, never free text: OpenCV writes only some strings as JSON
("Strings" below), and a reason in the operator's words belongs to the scan's state, not to its
file (`src/scan/parameters_file.hpp`, `scan_end`).

| `end` | the scan ended because |
|---|---|
| `stopped` | the operator stopped it |
| `source_timeout` | the frame source gave no frame within its time limit |
| `source_failure` | the frame source failed |
| `source_end` | the frame source had no frame left, as a sequence at its end |
| `tracking_failed` | tracking failed on a frame (`src/tracking/tracker.hpp`, `tracking_state::failed`) |
| `writer_failed` | the sequence's writer failed; the frame it failed on was fused into the live model and counted before the writer refused it, so the sequence and the trajectory may lack the last frame counted as fused |
| `shut_down` | the service was asked to stop during the scan |

The ranges each stage takes these values in are the stage's own, narrower than the file's: the
scan's request of its buffers refuses a value outside them, naming it (`src/tracking/tracker.hpp`,
`src/fusion/sparse_tsdf.hpp`, `src/geometry/organized_normals.hpp`), so a file can hold a set the
scanner does not run with. The stages take the set's values through one conversion each
(`src/scan/stage_parameters.hpp`), a double that the stage holds as a float rounded to the
nearest float.

Every key stands in every file; there is no optional key. A field added later that can lack a
value states its empty form here when it is added, so the file keeps one version. A file written
before the thirteen keys from `too_close_depth_m` on joined the set (2026-11-15, evening) lacks
them, and the reader refuses it: `no key "too_close_depth_m" in "parameters"`. A file written
before the set's six keys of the sparse TSDF's truncation and the normals and the scan's facts
joined (2026-11-16) lacks them too: `no key "stream" at the top level`.

Each object class's own set stands in `src/scan/object_class.cpp`, each value with its
evidence. A scan stores the set it was made with, and a stored set is read back as it stands,
so the set in a file need not be its class's own.

The writer writes each number as a double with 17 significant digits, or as a whole number
with `.0`, and the reader gets every bit back [M 2026-11-14: OpenCV 4.8.0, 200 000 random
doubles]. A budget and a count of frames up to 2^53, and a count of iterations or points and a
width or height up to 2^32 - 1, are doubles exactly [S: IEEE 754 binary64, a 53-bit
significand]; so is every float of the stream, which the reader rounds back to the same float. A
number of the stream that is not finite or lies beyond the largest float is refused before it is
rounded; one that rounds to 0 as a float, such as a depth unit of 1e-50, is refused as not > 0.
One bit does not come back: OpenCV writes -0.0 as `0.0` [M 2026-11-16: OpenCV 4.8.0], so a
principal point of -0 reads back as +0, equal by value; no other number of the file can be 0.
ctest `scan_files` compares each round trip by value, which for numbers other than 0 is bit for
bit, and the principal point bit for bit.

**Strings.** OpenCV's JSON writer gives text that both its own reader and Python's `json.loads`
take back unchanged only for some strings [M 2026-11-14: OpenCV 4.8.0, Python 3.10]: it writes
an apostrophe as `\'`, which is no JSON escape; a string that starts and ends with a double
quote raw, without quotes of its own, which OpenCV's reader then reads without them; a control
byte other than `\b \f \n \r \t` raw, which is no JSON; and a string of more than 4095 characters
does not read back (4097 make the writer throw). So a string in the file is at most 4095
characters of printable ASCII (0x20 to 0x7e) without the apostrophe, and does not both start and
end with a double quote. A byte above 0x7e is refused too, as the reader checks no UTF-8. A
commit hash, the object classes' names, a time's text and the keys of why a scan ended lie in
this set. The source's strings are the source's own: a camera's name and versions as OrbbecSDK
reports them, and a sequence's path, which can hold an apostrophe. The writer refuses another
string and the reader refuses one, naming the key, so a scan whose source describes itself with
such a string has to refuse it before it writes anything.

The reader refuses, naming the key:

- a text whose first byte after any white space is not `{` (YAML and XML among them, which
  OpenCV's reader takes whatever format it is told) [M 2026-11-14: OpenCV 4.8.0];
- a `format_version` other than the integer 1; it checks this first, so a file of another
  version is refused as such;
- a key it does not know, a key that stands twice, and a missing key, at the top level and in
  `parameters`, `stream`, `source` and `frames`;
- a value of another type, an unknown object class, a string outside the set above, a value
  out of the range above, a count of iterations or points that is no whole number from 0 to
  2^32 - 1, a width or height that is no whole number from 1 to 2^32 - 1, a count of frames that
  is no whole number from 0 to 2^53, frames taken that are not those fused and lost together and
  the frame tracking failed on when `end` is `tracking_failed`, a
  time of another form or a day the calendar lacks, and an `end` outside the seven keys;
- a number written as an integer: OpenCV reads an integer beyond 2147483647 wrapped to 32
  bits without a word, so every number is written with a decimal point or an exponent;
- a text that OpenCV's parser does not read.

What OpenCV's parser still does that strict JSON does not [M 2026-11-14: OpenCV 4.8.0]:

- it reads an integer modulo 2^32 and with C's prefixes, so a `format_version` of 4294967297,
  `0x1` or `01` reads as 1, and `010` as 8; it reads `true` as the integer 1 and `false` as 0;
- it accepts text after the closing brace, a comma before a closing brace, a `//` comment, a
  number written `+1`, and `.Nan` (which the range check then refuses);
- it refuses `null`, the escapes `\/` and `\uXXXX`, and an exponent without a fraction written
  with a capital `E` (`3E2`).
