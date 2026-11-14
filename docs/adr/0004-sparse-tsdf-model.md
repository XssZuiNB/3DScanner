# A scan's frames are fused in a sparse TSDF

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The model of a running scan is a truncated signed distance field on a sparse voxel structure
that holds only the space near surfaces; the full model and the object model are taken from
it. It replaces CUDA_3D's fusion, which appended each frame's points to the model and
downsampled. A TSDF averages the depth noise of many frames, gives the live model and the
surface a new frame is tracked against, and gives the mesh phase its surface. The voxel size
is fixed by the object class. Dense volumes do not fit: 0.6 m cubed at 1 mm and 8 bytes a
voxel is 1.7 GB [?: computed], with about 4 GB available to CPU and GPU together
[M 2026-11-13] (docs/HARDWARE.md, "Memory, storage, network"). Decided by the user,
2026-11-13.

## Considered Options

* Surfels: detail follows the viewing distance and the memory is small, but the mesh phase
  would need a surface reconstruction of its own.

## Consequences

* The sparse structure is written here from the published methods. VoxelHashing's code is
  CC BY-NC-SA 3.0 and is not copied (ADR 0002).
* Detail and color are limited to the voxel size.
