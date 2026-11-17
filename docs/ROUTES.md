# The web layer's routes

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The interface between the scanner's service and the page (the spec, "The web layer and the
page"): the HTTP routes and the layout of every message, which the page, the service and the
HTTP test follow (AGENTS.md §7 rule 5). The web layer (`src/web/`) translates a request into a
command of the scan interface (`src/scan/scan_interface.hpp`) or a read of its state, and the
answer into a message; it holds no state of its own. The routes carry the scan interface's three
commands (start a scan of an object class, stop, discard), its state, and the page's static
files. ctest `http_routes` pins them: it plays a whole scan and a discarded one through the
routes on a free port of the loopback interface and reads every message back with OpenCV's JSON
reader. The routes and their messages have one version until the phase is accepted: a change
changes this document, the web layer, the page and that test together (the stage brief, §4).

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

## Messages

Each message is one JSON object (RFC 8259) on one line, UTF-8, served as `application/json`.
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

The socket is bound with `SO_REUSEADDR` alone, not the library's `SO_REUSEPORT`: a second server
on the port is refused at its start, instead of the kernel sharing the connections between both
[S: socket(7)], and a restart binds the port its predecessor left in TIME_WAIT. Ending the web
layer closes the listening socket, so the port is free, and waits for every request in progress,
a command's answer too; the service ends the web layer before the scan interface.
