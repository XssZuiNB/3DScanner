# The sequence on disk

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The format of a sequence (GLOSSARY.md, "Sequence"): the frames a scan took, in order, each
with its index, timestamp, the IMU's samples taken since the frame before
(docs/methods/imu-samples.md), depth and color, and with the pose tracking gave it, the mark of
tracking loss or the mark that nothing tracked it; once, the frames' size, intrinsics and depth
unit and the format's version. The viewer writes it (`orbbec_viewer --sequence`, AGENTS.md §3),
and refinement reads a scan's back to fuse it at the full voxel size, refusing a frame that
nothing tracked (docs/methods/refinement.md, "The fusion of the sequence"); the scanner and the
replay tool are planned to write or read it. It has one writer and one reader, in `src/camera/` (`sequence_writer`, `sequence_reader`, the
layout in `sequence_format`); the frames are stored uncompressed. ctest `sequence` pins the
format: it writes frames and reads them back bit for bit, and compares the writer's bytes with
bytes it builds from the tables below. The format has one version, 2, which added the IMU's
samples: a change to it changes this document, the writer, the reader and that test together,
and a file written before the change is not read (AGENTS.md §7 rule 6), so a file of version 1
is refused. A change of the layout raises the version in the header, so that the reader refuses
a file of the old layout as such instead of misreading it.

## Writing

The writer creates the file under its name and never over a file that stands there, writes the
header and syncs the directory, so that the name survives a power cut. A thread of the
writer's own then takes the frames from a bounded queue and writes them; the queue holds 16
frames in a scan [?: a choice, `sequence_queue_frames` in `src/camera/sequence_writer.hpp`]. A
frame enters the queue by a swap of buffers, without a copy: the caller gets a frame of the
same size back, its pixels unspecified. When the queue is full the writer says so and does not
take the frame. The writer tells whether its queue has room, and room it has told of stays until
the caller's next frame, because only the caller fills the queue. In the scanner a frame enters
the per-frame path only when the queue has room for it; after tracking it goes to the writer with
its pose or mark, and no frame is fused that is not written. The viewer fuses nothing: it shows
every frame it takes, pushes it after its last use, and counts the frames a full queue refused
(AGENTS.md §3). The writer refuses a frame of another size, intrinsics or depth unit than the
header's (bit for bit, `same_description()` in `src/camera/sequence_format.hpp`), a frame
whose index or timestamp does not increase, one with more than 65535 samples of a sensor, and one
with a sample that does not follow its sensor's sample before, in the frame or in a frame
accepted before, or that lies after the frame's timestamp (`imu_sample_order`), the rule of
the frame source seam (`src/camera/frame_source.hpp`). A frame the writer does not take takes its
samples with it, so its sensors' streams have a gap there [?: a choice: a record stays a whole
frame, and the scanner fuses no frame it does not write; keeping those samples would take a
record of samples without pictures] (docs/methods/imu-samples.md, "Which frame carries a
sample"). `close()` writes every frame the writer took, in
order, then the end, and syncs the file to disk.

A write that fails (a full disk) ends the writing: the writer gives the reason to the next
push and to `close()`, and the frames still in the queue are lost. It tells how many frames it
wrote whole (`frames_written()`), so that its caller can say how many it lost.

The sequence is written under its own name, not beside it as a scan's other files are
(docs/SCAN_FILES.md, "Writing and reading"), because an unfinished scan keeps its frames. A
writer cut off (a crash, a power cut, a full disk) leaves a file that ends early: without its
end, within a record or between two. The reader gives every whole frame of such a file and then
says that it ends early, naming the record, and tells so apart from every other failure. After a
power cut, the frames not yet on disk are lost.

## Layout

Every number is little-endian: an integer's least significant byte first, a float an IEEE 754
binary32 and a double a binary64 with every bit kept, least significant byte first.

**Header**, 50 bytes:

| bytes | what |
|---|---|
| 0 to 17 | the magic, the 18 ASCII bytes `3dscanner sequence` |
| 18 to 21 | the format's version, the unsigned 32-bit integer 2 |
| 22 to 25 | the frames' width, in px, unsigned 32-bit |
| 26 to 29 | their height, in px, unsigned 32-bit |
| 30 to 45 | the intrinsics fx, fy, cx, cy, in px, four floats (`src/types/camera_param.hpp`, the one pinhole convention) |
| 46 to 49 | the depth unit, in m per depth value, a float |

**Records** follow, one for each frame in the order the writer took them, then the end. A
record's first byte is its kind:

| kind | what follows it |
|---|---|
| `P` | a frame with a pose: its index and timestamp, the pose, the counts of its samples, its samples, its depth, its color |
| `L` | a frame in tracking loss: its index and timestamp, the counts, its samples, its depth, its color |
| `N` | a frame that nothing tracked, as the viewer writes every frame: the same as `L` |
| `E` | nothing: the end, the file's last byte |

| bytes after the kind | what |
|---|---|
| 0 to 7 | the frame's index, unsigned 64-bit (`src/camera/frame.hpp`) |
| 8 to 15 | its timestamp, in µs on the frame source's clock, unsigned 64-bit |
| 16 to 71, `P` only | the pose: tx, ty, tz, qx, qy, qz, qw, seven doubles |
| then 2 bytes | a, the count of the accelerometer's samples, unsigned 16-bit |
| then 2 bytes | g, the count of the gyroscope's samples, unsigned 16-bit |
| then 20 a bytes | the accelerometer's samples, each as the table below, in increasing time |
| then 20 g bytes | the gyroscope's samples, the same |
| then | the depth, width × height unsigned 16-bit values, rows top to bottom, pixels left to right; 0 means no depth |
| then | the color, width × height × 3 bytes, red, green, blue, in the same order |

**A sample**, 20 bytes (`src/types/imu_sample.hpp`):

| bytes | what |
|---|---|
| 0 to 7 | its timestamp, in µs on the frames' clock, unsigned 64-bit |
| 8 to 19 | x, y, z, three floats, along the IMU's axes: m/s² for the accelerometer, rad/s for the gyroscope (docs/methods/imu-samples.md) |

Each sensor's samples of all records, in record order, are that sensor's stream, strictly
increasing in time, and each lies at or before its record's timestamp: frame i carries the
samples taken since the frame before, normally those of (t_{i-1}, t_i] (docs/methods/imu-samples.md,
"Which frame carries a sample"). A frame without samples has the counts 0. The values are kept bit
for bit, as the IMU's axes report them; no rotation is applied.

The pose is the trajectory's (docs/SCAN_FILES.md, "Trajectory"; `src/types/camera_pose.hpp`):
the camera's centre in m and its orientation as a unit quaternion, Hamilton's convention, from
the camera frame into the models' frame, in the order x, y, z, w, its sign as given. The
sequence keeps every bit and checks none of it.

So a frame of 1280x800 takes 5 120 077 bytes with a pose and 5 120 021 without one, and 20 bytes
more for each sample [S: computed]. With the camera's IMU at 200 Hz and 30 fps a frame carries
6.7 samples of each sensor, 267 bytes [S: computed, 2 × 200 / 30 × 20], 8 KB/s, 0.5 MB a minute;
a minute of frames takes 2.30 GB at 640x400 and 9.22 GB at 1280x800 [S: computed, 30 fps]. The
time to write a frame on the board stands in docs/HARDWARE.md, "Memory, storage, network".

## Reading

The reader is a frame source (`src/camera/frame_source.hpp`). It is opened for the size of the
sequence's frames and gives them in order, bit for bit with their samples, as fast as they are
asked. With each
frame it tells the pose or the mark. After the end it reports the end at every call.

It refuses an opening, with the reason, naming the file:

- a file that cannot be opened or read;
- a file whose first bytes are not the magic;
- a version other than 2, checked right after the magic, so that a file of another version,
  version 1 without the IMU's samples among them, is refused as such;
- a file that ends within its header;
- a size other than the sequence's, naming both.

After the frames before it, it reports a failure, naming the file and the record (the first
after the header is record 1):

- a file that ends early, before the end, which `ended_early()` tells apart from the other
  failures, so that the frames before it can be used;
- a record of an unknown kind;
- a frame whose index or timestamp does not increase, which the writer refuses too: the frames
  of a frame source's opening have strictly increasing indices and timestamps, and a gap in the
  indices stands for frames not taken;
- a sample that does not follow its sensor's sample before or lies after its frame's timestamp,
  which the writer refuses too;
- bytes after the end.

Each failure stays: every later call reports it again, until the reader is opened again, which
starts at the first frame.
