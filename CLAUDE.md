# Start here

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

3DScanner: a 3D scanner on a Jetson Orin Nano with an Orbbec Gemini 2 L RGB-D camera, a full
rewrite of CUDA_3D (GPU point clouds, colored ICP); today's library is `src/`, the programs
`apps/`.

**Read [AGENTS.md](AGENTS.md) sections 1 to 3 before running anything.** The hardware is live
and shared. The things that cost a reset or a reboot:

* **One program opens the camera at a time**; check with the commands of AGENTS.md §2 first.
* **Firmware, presets and the camera's flash are the user's**: the Orbbec tools that write
  them run only on the user's word (AGENTS.md §3).
* **Root is the user's**: the one root step, the udev rule, is done; anything else that needs
  `sudo` goes to the user as a command to run (AGENTS.md §2).

Then, depending on what you came for:

| you want | read |
|---|---|
| to work on the next thing | [ROADMAP.md](ROADMAP.md), which says who can do each step |
| what the scanner is to do, in its own words | [GLOSSARY.md](GLOSSARY.md), [docs/adr/](docs/adr/) |
| to write code for the rewrite | [CODING_STANDARDS.md, "The rewrite's code"](CODING_STANDARDS.md#the-rewrites-code) |
| to build and verify | [AGENTS.md §4](AGENTS.md#4-build-and-verify) |
| a hardware fact about this board | [docs/HARDWARE.md](docs/HARDWARE.md) |
| whether a kernel log line is harmful | [docs/HARDWARE.md, "Kernel log"](docs/HARDWARE.md#kernel-log) |

Two habits this tree expects:

* **Every claim carries its evidence**, in the tags AGENTS.md defines at its top. A number
  with no tag is a mistake.
* **Say what actually happened.** If a test fails, quote the output. If a step was skipped,
  say so. Nothing here is shipped from arithmetic alone (AGENTS.md §7 rule 8).

## Agent skills

The workflow skills in `.claude/skills/` (README there: the flow and what was adapted):

* **Tracker:** specs and tickets are local markdown in `.scratch/`, not committed;
  [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md).
* **Domain docs:** `GLOSSARY.md` and `docs/adr/`; read the ones that touch your area before
  exploring, [docs/agents/domain.md](docs/agents/domain.md).
* **Coding standards:** [CODING_STANDARDS.md](CODING_STANDARDS.md) and AGENTS.md §7, what
  `two-axis-review` checks a change against.
