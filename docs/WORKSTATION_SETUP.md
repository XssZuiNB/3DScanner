# Setting up the workstation

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

The commands that bring the workstation (x86_64, i9-13900K, RTX 4090, Ubuntu 22.04 or 24.04, CUDA
13.2) to where an agent can go on there (`.scratch/rewrite/brief.md`, §7, 55). The user runs them;
the lines with `sudo` are the user's. None of them has run on the workstation yet [?]: a step that
fails goes to the next agent as it failed. `builds-and-verifies-on-the-workstation` folds what
lasts into README.md and AGENTS.md and deletes this file.

## 0. On the Jetson

The commits since the last push go up, the push being the user's:

    git -C ~/3DScanner push origin main

## 1. The repository

    git clone <the repository's URL> ~/3DScanner     # or, where it is cloned: git -C ~/3DScanner pull
    git -C ~/3DScanner log --oneline -1              # the Jetson's newest commit

## 2. What git does not carry, from the Jetson

`JETSON` is the Jetson's user and address, e.g. `user@192.168.1.20`.

    JETSON=<user>@<the Jetson's address>
    rsync -a "$JETSON:3DScanner/.scratch/" ~/3DScanner/.scratch/          # tickets, brief, handoff
    rsync -a "$JETSON:3dscanner-scans/" ~/3dscanner-scans/                # 2.4 GB: imu-clock-offset-from-the-scan reads it
    rsync -a "$JETSON:3dscanner-sequences/" ~/3dscanner-sequences/        # optional, 770 MB
    # optional: the agent's notes on how the user wants it to work; Claude Code names a
    # project's directory after its path, '/' becoming '-'
    mkdir -p ~/.claude/projects/"$(echo ~/3DScanner | tr / -)"/memory
    rsync -a "$JETSON:.claude/projects/$(ssh "$JETSON" 'echo ~/3DScanner' | tr / -)/memory/" \
      ~/.claude/projects/"$(echo ~/3DScanner | tr / -)"/memory/

## 3. System packages (root)

    sudo apt update
    sudo apt install -y build-essential git curl rsync ccache pkg-config libopencv-dev python3-pip pipx
    nvidia-smi                       # the driver sees the RTX 4090
    nvcc --version                   # 13.2; else: export PATH=/usr/local/cuda/bin:$PATH in ~/.bashrc

On the Jetson the tree builds with g++ 11.4.0 and OpenCV 4.8.0 [M 2026-11-17]. Ubuntu 22.04's
packages give g++ 11 and OpenCV 4.5.4, 24.04's g++ 13 and OpenCV 4.6.0 [?: Ubuntu's archive, not
checked here]. Whether the tree builds without a warning with those is
`builds-and-verifies-on-the-workstation`'s first build; on 24.04 g++ 11 is there if needed:

    sudo apt install -y g++-11       # only if that ticket asks for it

## 4. The checkers, in the user's home (no root)

The same versions as the Jetson's (CODING_STANDARDS.md, "Tools"); clang-format's major version
decides the format check.

Ubuntu 22.04:

    pip3 install --user clang-format==18.1.8 shellcheck-py==0.10.0.1 cmake==4.4.4

Ubuntu 24.04 (pip's `--user` is refused there, so pipx):

    pipx install clang-format==18.1.8
    pipx install shellcheck-py==0.10.0.1
    pipx install cmake==4.4.4

Both:

    grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    clang-format --version           # 18.1.8
    shellcheck --version             # 0.10.0
    cmake --version                  # 4.4.4
    ccache --version

## 5. The camera (root)

The Jetson's udev rule (docs/HARDWARE.md, "Permissions") is the same file on x86_64 [?: OrbbecSDK's
`shared/` for x86_64 not compared]:

    rsync -a "$JETSON:/etc/udev/rules.d/99-obsensor-libusb.rules" ~/
    sudo cp ~/99-obsensor-libusb.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules && sudo udevadm trigger
    sudo usermod -aG video,plugdev,adm "$USER"     # then log out and in
    lsusb -t                                       # the camera plugged in: 5000M, USB 3

## 6. Then

Start Claude Code in ~/3DScanner and have it read `.scratch/rewrite/handoff.md` first. The next
tickets are ROADMAP.md's: `workstation-as-a-second-platform`, then
`builds-and-verifies-on-the-workstation`, which fetches OrbbecSDK's x86_64 build
(tools/fetch_deps.sh fetches the arm64 one today) and builds for the RTX 4090.
