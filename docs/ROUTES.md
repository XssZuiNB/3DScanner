# The web layer's routes

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The interface between the scanner's service and the page (the spec, "The web layer and the
page"): the HTTP routes, the WebSocket and the layout of every message, which the page, the
service and the web tests follow (AGENTS.md §7 rule 5). The web layer (`src/web/`) translates a
request into a command of the scan interface (`src/scan/scan_interface.hpp`) or a read of its
state, and the answer into a message; it holds no state of the scan's own. The routes carry the
scan interface's three commands (start a scan of an object class, stop, discard), its state, and
the page's static files; the WebSocket pushes the state and the live model to every browser.
ctest `http_routes` pins the routes: it plays a whole scan and a discarded one through them on a
free port of the loopback interface and reads every message back with OpenCV's JSON reader.
ctest `websocket` pins the WebSocket: it reads the live model back by the offsets below, plays
scans with browsers that read, one that connects in the middle and one that stops reading, and
ends the web layer with a browser connected.
The routes, the WebSocket and their messages have one version until the phase is accepted: a
change changes this document, the web layer, the page and those tests together (the stage brief,
§4).

The routes have no authentication and no encryption (R6): every browser that reaches the port may
read the state and give every command. They are for the operator's own network, never for the
internet. The service chooses the address it listens on; the test listens on `127.0.0.1` alone.

## Routes

| method and target | what it does | answers |
|---|---|---|
| `GET /api/state` | reads the scan interface's state | 200 with the state (below) |
| `POST /api/start?object_class=<name>` | starts a scan of the object class with its own set of parameters (`parameters_of()`, `src/scan/object_class.cpp`); `<name>` is `tabletop_object` or `large_object` | 200 when the scan scans, its first frame taken; 409 when the scan interface refuses it; 400 when the web layer refuses it (below); each with a command's answer |
| `POST /api/stop` | stops the scan: it is refined | 200 when taken, 409 when refused; a command's answer |
| `POST /api/discard` | discards the scan: nothing of it stays on disk | 200 when taken, 409 when refused; a command's answer |
| `GET /api/live`, asking to upgrade to a WebSocket | pushes the state and the live model (below, "The WebSocket") | 101, then the messages; without the upgrade, as any other path |
| `GET /<path>`, any other path | a file of the page's directory (below) | 200 with the file, or 404 |

A command carries its argument in the request line, as a query parameter; no route reads a
body. A request with a body longer than 1024 bytes [?: a choice] is refused with 413 by the
library before any route runs, so that requests cannot fill the board's memory, which the scan's
sparse TSDF needs. A command's request returns once the scan interface has answered it
(`scan_interface.hpp`, `start()`, `stop()`, `discard()`): a start once the scan scans or the
start failed, which takes seconds when the camera opens and a large sparse TSDF is allocated; a
stop and a discard before the scan's next frame, or between two frames of refinement.

The web layer's own refusals of a start, with 400:

| request | refusal |
|---|---|
| no `object_class` parameter | `a start names its object class: object_class=<name>` |
| an unknown name, `giant` for one | `no object class is named "giant"` |

The scan interface's refusals, with 409, are its own words, which the state's `refusal` repeats:
`no scan is in progress` for a stop or a discard while idle, `a scan is in progress (scanning)`
or `(refining)` for a start during a scan, `the scan is refining: a stop is for a scan that scans`,
and a start that failed with its reason (the source refused to open, the GPU's buffers refused).

## The WebSocket

`GET /api/live` with the WebSocket's upgrade (RFC 6455) opens a connection that stays while the
browser does. The web layer sends on it, the browser sends nothing: a message from the browser is
read and dropped. On it come two kinds of message:

- **the state**, a text message: exactly the text `GET /api/state` answers at that moment (below,
  "The state");
- **the live model**, a binary message (below, "The live model").

A browser gets the current state and the current live model as soon as it connects, then each
again when it changes: a state whenever the scan interface publishes one, about once a frame
while a scan scans, a live model at most twice a second [?: initial] (`scan_options`,
`live_model_interval`); the empty one is published at each scan's start, and a browser gets the
newest live model at each poll, which may already be a later one. A message reaches the browser
within 20 ms [?: a choice], the poll of the browser's loop, and the time to lay it out after the
scan interface published it, or once the send in progress to it ends. Each message is laid out
once, by the first loop that needs it, and every browser is sent that same message.

Nothing queues for a browser: the web layer keeps the newest state and the newest live model,
one of each, and at each poll a browser's loop sends the newest it has not sent. A state that a
newer one replaced within one poll is not sent; a browser that reads slowly gets the newest of
each once its send in progress ends, and the ones published meanwhile are dropped for it, never
for another browser, and never hold up the scan (below, "Threads").

The web layer ends a connection:

- when a send makes no progress for 5 s, the library's write timeout [S: httplib.h,
  `CPPHTTPLIB_SERVER_WRITE_TIMEOUT_SECOND`]: a browser that stopped reading; the connection is
  closed without a closing handshake;
- when the browser leaves 2 [?: a choice] of the library's pings unanswered, one every 30 s
  [S: httplib.h, `CPPHTTPLIB_WEBSOCKET_PING_INTERVAL_SECOND`]: a browser gone from the network
  without a word, closed within about 90 s [S: computed, 3 × 30 s] with the library's close,
  1001 "pong timeout";
- when a frame of the browser's arrives split across the read's 20 ms, its first byte within it:
  the library then fails the read and closes the connection [S: httplib.h, `read_exact()` in
  `read_websocket_frame()`], and the page reconnects. The browser's frames are small, a ping's
  answer 6 bytes and a close at most 131 [S: RFC 6455, 5.2 and 5.5], so the risk is low;
- when the browser closes it or a read fails otherwise;
- when the web layer ends, without a closing handshake.

## Messages

Each JSON message is one JSON object (RFC 8259) on one line, UTF-8, served as `application/json`
over HTTP and as a text message on the WebSocket.
Its keys stand in the order below; a reader takes them by name. A string is the scan interface's
text as it stands: a quote and a backslash are escaped, `\b`, `\f`, `\n`, `\r` and `\t` by their
letter, any other control character below 0x20 as `\u00XX`, every other byte as it is. A value
the state does not hold is the empty string, never `null`: OpenCV's JSON reader, which the test
uses, refuses `null` and `\u` [M 2026-11-17: cv2 4.8.0 on the board]; it reads `true` as 1 and
`false` as 0. A whole number is at most 2^53, so that the page, whose numbers are doubles, reads it
exactly [S: IEEE 754 binary64, a 53-bit significand].

### The state

| key | value |
|---|---|
| `version` | whole number: rises by one with each state the scan interface publishes |
| `phase` | `idle`, `starting`, `scanning` or `refining` (the spec, "The scan interface") |
| `scan` | the scan in progress or the last one: its directory's name in the scans directory, its start time (`YYYYMMDDTHHMMSS.ffffffZ`, docs/SCAN_FILES.md, "A scan's directory"); empty before the first scan |
| `tracking` | what tracking made of the last frame taken: `tracking`, `lost` or `failed`; empty before a scan's first frame |
| `lost_cause` | the check's condition that failed when the last frame taken was lost: `registration_end`, `fitness`, `inlier_rmse` or `step`; `passed` otherwise (docs/methods/tracking.md, "The check") |
| `too_close` | `true` when the last frame taken was too close (R36; docs/methods/tracking.md, "Parameters"), else `false`; the page shows the notice and smooths it |
| `frames` | an object: `taken`, `fused`, `lost`, `not_taken`, the scan's frames so far, as its parameters count them (docs/SCAN_FILES.md, "Parameters") |
| `end` | why the scan in progress or the last one ended, one key of docs/SCAN_FILES.md's closed set (`stopped`, `source_timeout`, `source_failure`, `source_end`, `tracking_failed`, `writer_failed`, `shut_down`); empty before it ended and for a scan discarded |
| `end_message` | what ended it, in plain English; empty for the operator's stop and the service's |
| `outcome` | what became of the last scan: `none` before the first ended, `refined`, `unfinished` or `discarded` |
| `outcome_message` | why the last scan is unfinished, in plain English; empty otherwise |
| `refusal` | the last command refused or start failed, with the reason; empty before the first |
| `live_model_failure` | why a live model could not be taken, the last time one could not; empty while each was taken |

Example, a tabletop scan 31 frames in, the camera in view, after a second start was refused:

    {"version":40,"phase":"scanning","scan":"20261117T101112.123456Z","tracking":"tracking","lost_cause":"passed","too_close":false,"frames":{"taken":31,"fused":31,"lost":0,"not_taken":0},"end":"","end_message":"","outcome":"none","outcome_message":"","refusal":"a scan is in progress (scanning)","live_model_failure":""}

### A command's answer

| key | value |
|---|---|
| `accepted` | `true` when the command was taken, else `false` |
| `refusal` | why not, in plain English; empty when it was taken |

Example, a stop while idle, with 409:

    {"accepted":false,"refusal":"no scan is in progress"}

### The live model

The live model (`src/scan/live_model.hpp`; docs/methods/tracking.md, "The live model") as one
binary message on the WebSocket: a header of 80 bytes, then the positions of its n points, then
their colors, in the order of the live model's points. Every number is little-endian: an
integer's least significant byte first, a float an IEEE 754 binary32 and a double a binary64,
least significant byte first, as in docs/SEQUENCE_FILE.md.

| bytes | what |
|---|---|
| 0 to 7 | the live model's version, unsigned 64-bit: rises by one with each live model published, the empty one a scan starts with too |
| 8 to 15 | the index of the frame it was taken at, unsigned 64-bit; 0 when has_frame is 0 |
| 16 to 19 | has_frame, unsigned 32-bit: 1 when it was taken at a frame, 0 for the empty live model, without a frame or a point, that a scan starts with and that stands before the first scan |
| 20 to 23 | n, the number of its points, unsigned 32-bit, at most 200000 [?: initial] (`scan_options`, `live_model_points`) |
| 24 to 79 | the pose tracking gave that frame: tx, ty, tz, qx, qy, qz, qw, seven doubles; the camera's centre in m and its orientation, a unit quaternion in Hamilton's convention, from the camera frame into the models' frame, as the trajectory's (docs/SCAN_FILES.md, "Trajectory"); the identity (0, 0, 0, 0, 0, 0, 1) when has_frame is 0 |
| 80 to 80 + 12n − 1 | the positions: x, y, z of each point, three floats, in m in the models' frame (the camera frame of the first fused frame: x right, y down, z forward) |
| 80 + 12n to 80 + 15n − 1 | the colors: red, green, blue of each point, three bytes, 0 to 255 |

The positions start at a multiple of 8 and lie one block after the other, so that the page can
read them as a `Float32Array` and the colors as a `Uint8Array` on the message's buffer without a
copy; the header is read with a `DataView`, little-endian. The live model's normals, its spacing
and the time it was taken are not sent.

A message takes 80 + 15n bytes: 3.0 MB at 200000 points, so about 48 Mbit/s to each browser at
two a second [S: computed]. The walk of ctest `websocket` sent live models of 41874 to 152987
points, 0.63 MB to 2.29 MB [M 2026-11-17]. Whether a phone on the board's WiFi takes that load is
a live check of `check-the-page-on-devices`.

Example, the empty live model of version 12: 80 bytes, `0c 00 00 00 00 00 00 00`, then 8 bytes
of 0, then has_frame `00 00 00 00` and n `00 00 00 00`, then six doubles 0 and the double 1,
`00 00 00 00 00 00 f0 3f`.

## The page's files

Every `GET` that names no route above serves a file of the page's directory, which the service
is given at its start. `/` serves `index.html`. The path is checked in two steps, as the library
decoded it (`%2F` is a `/`, `%5C` a backslash, `%00` a NUL), and every refusal is a 404 with no
body:

1. As written, after its leading `/`: a path that holds a backslash or a NUL byte, or a segment
   between two `/` that is empty (`//`, a trailing `/`), `.` or `..`, is refused, whether it
   would lead outside the directory or not. So `/sub/../app.js`, `/./app.js`, `/sub//inner.html`,
   `/sub/` and an absolute path (`//etc/hostname`, `/%2Fetc/hostname`) get 404.
2. Resolved: the path is joined to the directory and every link in it resolved; the file is
   served only when it is a regular file inside the directory, compared component by component.
   So a link that points out, a directory and a missing file get 404.

A link that stays inside the directory is followed; the page's files are copied into the
directory, not linked from outside it.

The tree's page is `src/web/page/`. Each build copies it, with four files of three.js 0.187.0
(README.md, "License"), into `<build>/page/`, the directory the service is given
(`src/CMakeLists.txt`, "The page"):

| file | what |
|---|---|
| `index.html` | the page: its parts and words, and the import map that gives three.js's bare name `three` as `./three/three.module.js` |
| `page.css` | its look, usable at the width of a phone |
| `scan_page.js` | the WebSocket `/api/live`, which it opens again after it closes; the state shown by phase; the commands `/api/start?object_class=`, `/api/stop`, `/api/discard`, each refusal from the command's answer |
| `live_view.js` | the 3D view of the live model, read from its binary message |
| `three/three.module.js`, `three/three.core.js`, `three/OrbitControls.js`, `three/LICENSE` | three.js's module, the core it imports, its orbit controls and its MIT notice |

The page reads the state and the live model from the WebSocket alone, not from `GET /api/state`.
ctest `http_routes` fetches `<build>/page/` through a web layer: `/`, then every file the page
names, through the modules' static imports, each served and not empty, none of another host.

| extension | media type |
|---|---|
| `.html` | `text/html; charset=utf-8` |
| `.js` | `text/javascript; charset=utf-8` |
| `.css` | `text/css; charset=utf-8` |
| `.json` | `application/json` |
| any other | `application/octet-stream` |

## Threads

The web layer's own thread accepts the connections. cpp-httplib's pool of workers runs the
requests: 8 threads on this board's 6 cores, the larger of 8 and the cores less one, and up to
4 times that, 32, while every one is busy; a worker beyond the 8 ends after 3 s idle
[S: cpp-httplib v0.66.0, `httplib.h`, `CPPHTTPLIB_THREAD_POOL_COUNT`, `_MAX_COUNT`,
`_IDLE_TIMEOUT`; M 2026-11-17: `nproc` 6]. A connection kept alive holds its worker between its
requests, up to 5 s idle [S: `CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND`]. A command's request holds
its worker until the scan interface answers it: during a start that takes seconds, a burst of
commands holds as many workers, every one answered when the start is decided; a state request
is answered at once by another worker, as `state()` takes only the state's lock. Beyond 32 busy
workers a new connection waits in the library's queue, unbounded, until a worker is free; none
is dropped.

A WebSocket holds its worker while its browser stays, for its loop: every 20 ms [?: a choice] it
takes the newest state and live model, sends those it has not sent, outside every lock, then reads
the browser's frames (a ping's answer, a close) for up to 20 ms. So each browser takes one of the
32 workers, and a browser gone without a word holds its worker until the pings close it.

The scan thread never waits for a browser. The scan interface's listeners only count, with one
atomic addition each, the states and the live models it publishes (`gca::scan_news`,
`src/web/web_server.hpp`), which the service builds before the scan interface and wires into its
listeners. At each poll a browser's loop takes the web layer's lock of the newest messages and
compares the counts with those the newest were laid out at; the first loop after a change copies
the state (`state()`) or the live model (`latest_live_model()`, up to 5.6 MB [S: computed, 200000
points of 28 bytes]) once and lays it out there, and a version already laid out is not laid out
again. Inside that lock the loop takes the scan interface's state
lock or its live model's lock, never the other way round; the scan thread takes neither the web
layer's lock nor a socket's. The scan thread's next live model waits for that copy only when it
comes during it, and it comes at least the live model's interval, 500 ms [?: initial], after the
one copied. No lock is held across a send or a read.

The socket is bound with `SO_REUSEADDR` alone, not the library's `SO_REUSEPORT`: a second server
on the port is refused at its start, instead of the kernel sharing the connections between both
[S: socket(7)], and a restart binds the port its predecessor left in TIME_WAIT. Ending the web
layer closes the listening socket, so the port is free, and waits for every request in progress,
a command's answer too, and for every WebSocket's loop: a loop ends within its 20 ms poll, or
when the send in progress ends, which for a browser that stopped reading is the 5 s write timeout,
and a ping the library sends to that browser at the same moment may wait as long again. The
service ends the web layer before the scan interface, and the scan interface before the news.
