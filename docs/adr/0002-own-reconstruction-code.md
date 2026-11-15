# The reconstruction is this tree's own code

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

Everything that turns frames into a model (point clouds, tracking, fusion, the pose
optimization after a scan, later the meshing) is written in this tree; no reconstruction
library such as Open3D or PCL is linked. The scanner is also the user's platform for GPU
registration and fusion algorithms, it stays usable commercially, and each step has to fit
the Orin's frame time, which needs control of every allocation and kernel. The dependencies
stay OrbbecSDK, Eigen and OpenCV; the web page alone may add small libraries. Every dependency
carries a permissive license (MIT, BSD, Apache, MPL, 0BSD): no GPL and no non-commercial code
or data. Decided by the user, 2026-11-13.

**OrbbecSDK's LGPL parts.** OrbbecSDK v2.10.6 is MIT and carries libusb (LGPL-2.1) and live555
(LGPL) statically inside `libOrbbecSDK.so` [B 2026-11-15: `readelf -d` names only libc, libstdc++,
libm, libgcc_s, libdl, libpthread and ld-linux; `strings` finds 421 lines with "libusb" and the
string "LIVE555 Streaming Media"; the SDK's `licenses/THIRD_PARTY_LICENSES.md` lists both]. The
user accepted this on 2026-11-15: the permissive-only rule covers the code the tree holds and
links as its own. The camera vendor's SDK is linked dynamically and fetched by
`tools/fetch_deps.sh`, not distributed by the tree. A device sold with the SDK installed needs a
license check then [?: not legal advice].

## Considered Options

* Open3D (MIT) for tracking, fusion and meshing: CUDA_3D's algorithms would go, and Open3D
  would first have to build with CUDA and run in real time on this board [?].
* Own code for the per-frame path and a library for the steps after a scan: rejected for the
  platform's purpose.
