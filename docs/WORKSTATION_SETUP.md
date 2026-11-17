# Setting up the workstation

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The commands that brought the workstation to where an agent can go on there
(`.scratch/rewrite/brief.md`, §7, 55), and what it measured afterwards. The user ran them on
2026-11-17; the lines with `sudo` are the user's. `builds-and-verifies-on-the-workstation` folds
what lasts into README.md and AGENTS.md and deletes this file.

## The workstation

| what | value | evidence |
|---|---|---|
| CPU | Intel Core i9-13900KF, 32 threads | [M 2026-11-17] `lscpu` |
| memory | 30 GiB RAM, 7 GiB swap | [M 2026-11-17] `free -g` |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, compute capability 8.9, driver 595.99.02 | [M 2026-11-17] `nvidia-smi` |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-43-generic, x86_64 | [M 2026-11-17] `/etc/os-release`, `uname` |
| disk | 937 GB NVMe, 853 GB free | [M 2026-11-17] `df -h` |
| camera | Gemini 2 L (2bc5:0673) on a USB 3 port, 5000M | [M 2026-11-17] `lsusb`, `lsusb -t` |

The brief and the tickets say Ubuntu 22.04 or 24.04 and an i9-13900K; the machine is 26.04 and a
13900KF (the K without the integrated GPU) [M 2026-11-17].

## The toolchain beside the Jetson's

| tool | workstation [M 2026-11-17] | Jetson [M 2026-11-17] | from |
|---|---|---|---|
| g++ | 15.2.0 | 11.4.0 | apt `build-essential` |
| nvcc | 13.2.86 | 12.6 | /usr/local/cuda-13.2, there before the setup |
| OpenCV | 4.10.0 | 4.8.0 | apt `libopencv-dev` |
| Python | 3.14.4 | 3.10 | the system's |
| ccache | 4.12.3 | 4.5.1 | apt |
| clang-format | 18.1.8 | 18.1.8 | pipx |
| shellcheck | 0.10.0 | 0.10.0 | pipx (`shellcheck-py`) |
| cmake | 4.4.4 | 4.4.4 | pipx |

The checkers match the Jetson's; the compilers and OpenCV do not. Whether the tree builds without
a warning under g++ 15 and nvcc 13.2 is `builds-and-verifies-on-the-workstation`'s first build
[?]. Whether Ubuntu 26.04 has a `g++-11` package is not checked [?].

## What ran

### 0. On the Jetson

    git -C ~/3DScanner push origin main      # origin/main is at c3f1e36 [M 2026-11-17]

### 1. The repository

    git clone <the repository's URL> ~/3DScanner

### 2. What git does not carry, from the Jetson

`JETSON` is the Jetson's user and address.

    JETSON=<user>@<the Jetson's address>
    rsync -a "$JETSON:3DScanner/.scratch/" ~/3DScanner/.scratch/
    rsync -a "$JETSON:3dscanner-scans/" ~/3dscanner-scans/                # 2.4 GB [M 2026-11-17]
    rsync -a "$JETSON:3dscanner-sequences/" ~/3dscanner-sequences/        # 734 MiB [M 2026-11-17]
    # the agent's notes; Claude Code names a project's directory after its path, '/' becoming '-'
    mkdir -p ~/.claude/projects/"$(echo ~/3DScanner | tr / -)"/memory
    rsync -a "$JETSON:.claude/projects/$(ssh "$JETSON" 'echo ~/3DScanner' | tr / -)/memory/" \
      ~/.claude/projects/"$(echo ~/3DScanner | tr / -)"/memory/

The notes name Jetson paths (the build lock, the memory guard, the demo worktree); those exist
only on the Jetson.

### 3. System packages (root)

    sudo apt update
    sudo apt install -y build-essential git curl rsync ccache pkg-config libopencv-dev python3-pip pipx

nvcc was installed but not on the PATH; `~/.bashrc` gained it:

    export PATH=/usr/local/cuda/bin:$PATH

### 4. The checkers, in the user's home (no root)

On 26.04, as on 24.04, pipx (pip's `--user` is refused):

    pipx install clang-format==18.1.8
    pipx install shellcheck-py==0.10.0.1
    pipx install cmake==4.4.4
    grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

### 5. The camera (root)

The Jetson's udev rule, copied unchanged [?: OrbbecSDK's `shared/` for x86_64 not compared]:

    rsync -a "$JETSON:/etc/udev/rules.d/99-obsensor-libusb.rules" ~/
    sudo cp ~/99-obsensor-libusb.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules && sudo udevadm trigger
    sudo usermod -aG video "$USER"     # adm and plugdev were there; then log out and in

No program has opened the camera on the workstation yet [?]: OrbbecSDK's x86_64 build is not
fetched (tools/fetch_deps.sh fetches the arm64 one).

## What the workstation does not have yet

* **The memory guard and the monitor** (AGENTS.md §4): the Jetson's `memguard.sh` and
  `~/.local/state/3dscanner/monitor.sh` are not here. The lock's directory,
  `~/.local/state/3dscanner/`, is (made 2026-11-17; `flock` makes the file). Whether one heavy
  job at a time holds on 30 GiB and 32 threads is `workstation-as-a-second-platform`'s ADR to
  say; until then the Jetson's rule holds and the lock is taken as there.
* **OrbbecSDK** for x86_64, **GCA_CUDA_ARCHITECTURES 89**, **BUILD_JOBS** for this machine:
  `builds-and-verifies-on-the-workstation`.

## Then

Start Claude Code in ~/3DScanner and have it read `.scratch/rewrite/handoff.md` first. The next
tickets are ROADMAP.md's: `workstation-as-a-second-platform`, then
`builds-and-verifies-on-the-workstation`.
