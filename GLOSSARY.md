# 3DScanner

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The terms of the scanner: what it scans, who works it and what a scan is. One context; the
workflow skills use these words and none of the ones listed under _Avoid_.

## Language

### What is scanned

**Tabletop object**:
A single rigid thing of 10 cm to 50 cm that stands on a table while it is scanned
[?: a choice of the user, 2026-11-13].
_Avoid_: small object, part

**Large object**:
A single rigid thing of 0.5 m to 2 m that stands on the floor while it is scanned
[?: a choice of the user, 2026-11-13]. A room is not one, and the scanner scans no rooms.
_Avoid_: big object, scene

**Object class**:
Tabletop object or large object: the one choice the operator makes before a scan. It sets
how far the scanner looks and how fine the model is.
_Avoid_: mode, preset (the camera's presets are another thing), profile

### Scanning

**Scan**:
One continuous pass of the hand-held camera around one object that stands still, from the
operator's start to the operator's stop, together with everything the scanner keeps of it.
_Avoid_: session, capture, run

**Frame**:
One color picture and the depth picture aligned to it, taken at one time, with its index, its
timestamp, the camera's intrinsics and the depth unit, and with the IMU's samples taken since the
frame before.
_Avoid_: frameset, image pair, snapshot

**Frame source**:
What gives the scanner its frames one after another: the camera, a sequence, or the tests'
synthetic source. When it has no frame it says why: it could not be opened, a timeout, a
failure of the camera, the end.
_Avoid_: device, stream, grabber

**Live model**:
The coarse model the scanner builds while a scan runs and shows to the operator, so that
missed surfaces are seen before the scan stops.
_Avoid_: preview cloud, draft

**Live view**:
The picture of the live model's surface, in its colors, as the camera sees it from where it is,
which the operator watches while a scan runs: one for each tracked frame, the newest only.
_Avoid_: preview, render, video

**Refinement**:
The work on a scan after the operator's stop that turns its frames into the full model.
_Avoid_: post-processing, offline pass

**Voxel size**:
The edge of the small cubes the model's space is divided into; nothing smaller than a voxel is
resolved, in shape or in color. Each object class sets two: the live model's, coarser, and the
full model's, which refinement fuses at.
_Avoid_: resolution, grid size, cell size

**Support plane**:
The table or floor surface the scanned object stands on.
_Avoid_: ground, table plane, turntable plane

**Separation**:
Dividing a scan's model into the object and its background (the support plane and whatever
else stands around). The scanner proposes it and the operator may change it.
_Avoid_: segmentation, clustering, background removal

**Tracking**:
Knowing, for each frame of a running scan, where the camera is relative to the model.
**Tracking loss** is the state in which the scanner does not know it; nothing enters the
model then, until **relocalization** finds the camera again on a part already scanned.
_Avoid_: odometry, registration (the alignment of two point clouds, a step tracking uses)

**Operator**:
The person who works a scan from the scanner's web page and needs no knowledge of the
software.
_Avoid_: user (the user is the person who owns this repository and decides its requirements)

### What a scan keeps

**Full model**:
The colored point cloud of everything a scan reconstructed, the background included.
_Avoid_: scene, raw model, map, final model

**Object model**:
The part of the full model that the separation assigns to the object.
_Avoid_: result, cropped model

**Trajectory**:
The camera's position and orientation at each frame of a scan.
_Avoid_: path, poses

**Keyframe**:
A tracked frame the scanner keeps, with its pose and a small signature of its picture, whenever
the camera has moved far enough since the last one; refinement optimizes the trajectory over the
keyframes, and relocalization tries them as candidates.
_Avoid_: key image, anchor frame, reference frame

**Sequence**:
The color and depth frames of a scan as the camera delivered them, with the IMU's samples taken
since the frame before, kept so that the scan can be run again without the camera.
_Avoid_: recording, bag, dataset
