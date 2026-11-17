# The scanner service

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The scanner as one program, `build/apps/scanner_service` (R7; the spec, "The service"): it serves
the page (docs/ROUTES.md) on the operator's own network, runs the scans the page starts, and opens
the Orbbec camera only from a scan's start to its stop, so that the development tools can use the
camera while no scan runs. Its options, its stop, its log lines and the command that says whether a
scan is in progress are an interface (AGENTS.md §7 rule 5): the install tool
(`install-the-user-service`) and the live check of the service read them. The program is
`apps/scanner_service.cpp`; its parts and their order `apps/service.hpp`; its command line
`apps/service_options.hpp`. ctest `service_options` pins the command line, ctest
`scanner_service` the parts, the stop, the log lines and `scan_in_progress`, with a stand-in for
the camera; the program itself runs on the camera only on the user's word (AGENTS.md §3).

**No login, the operator's own network only** (R6). The page has no authentication and no
encryption: every device that reaches the port may read the state and give every command. Without
`--address` the service serves every interface of the board (`0.0.0.0`), so the page is for the
board's own network and never for the internet.

**What it scans** (A5): rigid, opaque objects that are not mirror-like. People are not scanned.

## Options

| option | default | what |
|---|---|---|
| `--scans DIR` | `~/3dscanner-scans` (A8), from the environment's `HOME` | the scans directory each scan's directory goes into (docs/SCAN_FILES.md, "A scan's directory"); it exists and can be written; the service does not make it |
| `--port N` | 8080 [?: a choice, the spec's] | the port the page is served on, 1 to 65535 in decimal digits |
| `--address A` | `0.0.0.0`, every interface | the numeric address the page is served on; the board's LAN address serves that network alone |
| `--page DIR` | the build's page directory, `<build>/page` | the page's directory (docs/ROUTES.md, "The page's files"); the install copies it elsewhere and names it |
| `--keep-sequence` | off | a refined scan keeps its sequence; off, the sequence is removed once the scan is refined, and only a scan left unfinished keeps its frames (docs/SCAN_FILES.md, "The sequence switch"; R18) |

An argument the service does not know (`--help` too), an option without its value or given twice,
a port that is no whole number from 1 to 65535, no `--scans` without `HOME`, and a scans directory
that does not exist, is no directory or cannot be written are refused: `scanner_service: <reason>`
and the usage on stderr, exit 2. `main` reads the options before it opens the GPU, a port or the
camera and before it installs a signal handler [S: apps/scanner_service.cpp, `main`]. An address the
board does not have, a port in use or below 1024, and a page directory that does not exist are the
web layer's to refuse when it starts: exit 1 with the reason (below).

## The camera

The camera opens at a scan's start and closes at its stop, before refinement: the scan interface
opens and closes its source (`src/scan/scan_interface.hpp`), and `orbbec_device` creates OrbbecSDK's
context only in its opening and drops it at its close [S: src/camera/orbbec_device.cpp]. The stream
is 640x400 at 30 fps [?: a choice, until `per-stage-times-from-replay` decides the stream size].
During a scan the hold (`src/camera/color_hold.hpp`) writes the two switches of AGENTS.md §3's list,
2000 and 2003: on at the opening, off after the 30th frame, on again at every ending, the service's
stop included. A property the camera refuses does not stop the scan; the state and the scan's
parameters name it (`property_refusals`, docs/ROUTES.md; `refused_properties`,
docs/SCAN_FILES.md). OrbbecSDK writes `Log/OrbbecSDK.log.txt` into the current directory
[M 2026-11-13], which the unit's working directory sets.

## The stop

SIGINT (Ctrl-C), SIGTERM (systemd's stop, a kill) and SIGHUP (a closed terminal or SSH session)
stop the service (`apps/stop_signals.hpp`); `main` blocks them before any thread starts and waits
for one (`apps/service.hpp`, `install_stop_signals()`). A second stop signal during the stop does
nothing: only a SIGKILL ends it sooner, with the camera's controls as held. SIGPIPE is ignored, so a
browser that closes its connection fails a write instead of ending the service. Then the parts end
in the reverse of their order:

1. the web layer: its listening socket closed, so the port is free, then every request in progress
   and every WebSocket's loop ended: within its 20 ms poll, or up to 5 s when a send to a browser
   that stopped reading makes no progress, and as long again for a ping sent to it at that moment
   [S: docs/ROUTES.md, "Threads"]; a start in progress is answered first, after the camera's
   opening and its first frame;
2. the scan interface: a scan in progress ends at its next frame without refinement, `end`
   `shut_down`: the hold switches the camera's automatic controls on again, the camera closes, the
   writer writes the frames still in its queue, at most 16 (`sequence_queue_frames`), and the
   parameters are written; the scan stays on disk unfinished, with its sequence. A refinement in
   progress stops between two frames, the scan unfinished too. The next frame comes within the
   camera's frame time, 33 ms at 30 fps [S: computed], or the camera's time limit of 5 s passes
   when none comes [S: src/camera/orbbec_rules.hpp, `orbbec_frame_time_limit_ms`].

The worst case is about 15 s [S: computed: 5 s + 5 s for the web layer, 5 s for the camera's last
frame], plus the writer's last frames and the parameters' write, which the board's disk sets
[?: unmeasured], and, when a start was in progress, the camera's opening and first frame
[?: unmeasured on the camera]. systemd's stop timeout for the unit must lie above it, or systemd kills the
service with the camera's controls held; `install-the-user-service` sets it (systemd's default is
90 s [?: systemd's documentation, not read here]). A SIGKILL, a crash or a power cut leaves the
controls as they were held until the next scan's opening switches them on, or a replug [?]
(AGENTS.md §3).

Exit statuses: 0 after a stop signal; 1 when the web layer did not start or an exception ended the
service; 2 when the command line is refused.

## Log lines

The service prints each line on standard output, which systemd's journal takes, whole and flushed;
the main thread's lines and the scan thread's do not mix. The scan thread writes its lines itself,
at a change of the phase and when the hold adds a reading or a refusal, so a stalled standard
output or journal stalls the scan thread at those moments, and the camera's frames meanwhile count
as not taken. Every line starts with
`scanner_service: `:

| line | when |
|---|---|
| `serving the page on http://<address>:<port>/ from <page>; scans in <scans>, <sequences>` | the web layer serves; `<sequences>` is `sequences kept` with `--keep-sequence`, else `a refined scan's sequence removed` |
| `the web layer did not start: <reason>` | its start was refused (docs/ROUTES.md; `src/web/web_server.hpp`, `start()`): exit 1 after `stopped` |
| `a scan starts` | a start was accepted |
| `the start failed: <reason>` | the start failed; the reason as the state's `refusal` |
| `scan <scan>: scanning` | the scan's first frame was taken; `<scan>` is its directory's name |
| `scan <scan>: refining, end <end>[ (<message>)], frames taken <T> fused <F> lost <L> not taken <N>` | the scan stopped and refines; `<end>` one key of docs/SCAN_FILES.md's closed set, the message the state's `end_message` when there is one |
| `scan <scan>: <outcome>[: <message>], [end <end>[ (<message>)], ]frames taken <T> fused <F> lost <L> not taken <N>` | the scan ended: `refined`, `unfinished` or `discarded`, with the state's `outcome_message` when there is one; a discarded scan has no end |
| `Color at the start: ...`, `Color before the hold: ...`, `Color while held: ...`, `Color after: ...` | the hold's readings of the five properties (`src/camera/color_hold.hpp`, `reading_line()`), as `orbbec_viewer --hold-color` prints them (AGENTS.md §3) |
| `Color: the camera refused to <read\|switch> <property> (<number>) ...` | a read or write the camera refused (`refusal_line()`) |
| `a live model could not be taken: <reason>` | the state's `live_model_failure` changed to it |
| `stopping on <SIGINT\|SIGTERM\|SIGHUP>` | a stop signal came; the parts end |
| `stopped` | the parts ended, the last line |
| `<what>` | an exception ended the service (`main`): exit 1 |

A command the scan interface refuses is not logged; the page shows it. The camera's own lines that
`orbbec_viewer` prints (its name and link, the streams, the IMU's or `No IMU: ...`, the frames it
skipped) are not in the service's log; the scan's parameters hold the camera's name and versions.

## Whether a scan is in progress

    build/apps/scan_in_progress [--address A] [--port N]    # 127.0.0.1 and 8080 without them

It asks the service's state, `GET /api/state` (docs/ROUTES.md), and reads its `phase`, waiting at
most 2 s for the connection and 5 s for the answer [?: a choice]. A scan is in progress while it
starts, scans or refines: the camera or the GPU is the scan's then, and the service is not to be
stopped or replaced. It links no OrbbecSDK, no CUDA and no OpenCV [M 2026-11-17: `ldd`], so it
touches no hardware.

| exit | line | when |
|---|---|---|
| 0 | `scan_in_progress: yes, <phase>` on stdout | the phase is `starting`, `scanning` or `refining` |
| 1 | `scan_in_progress: no, idle` on stdout | the phase is `idle` |
| 2 | `scan_in_progress: <reason>` and the usage on stderr | the command line is refused: an argument it does not know (`--help` too), an option without its value or given twice, a port that is no whole number from 1 to 65535 |
| 3 | `scan_in_progress: cannot tell: <reason>` on stderr | no answer came (no service on the port, a refused connection, a time limit), the answer was no 200, or it held no phase of the four |

A service started with `--address` set to the board's LAN address does not answer on 127.0.0.1:
ask it at that address.
