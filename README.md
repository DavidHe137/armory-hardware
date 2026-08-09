# Armory Hardware

Hardware-specific fleet control, trial tooling, and the operator TUI used with Armory.

This repository is intentionally separate from the public Armory core. It contains setup-specific
assumptions for the hardware fleet and is not a general-purpose Armory interface.

## Setup

Operator-specific settings -- SSH account, workstation address range, jump host, camera
server -- live in a gitignored `.env` rather than in the committed config:

    cp .env.example .env
    # then fill in the values for your fleet

Run the dashboard from this repository with:

    uv run armory-tui

Set ARMORY_TUI_CONFIG to use a different fleet configuration, or ARMORY_ENV_FILE to use a
different `.env`. Precedence is: explicit key in the fleet YAML, then the environment,
then `.env`, then a safe default. `configs/armory-tui.yaml` defines only fleet layout, so
in practice the environment supplies the connection details.

## Workspace builds

Every workstation bind-mounts the same `user_data/piper_ros` checkout, so the ROS
workspace is one shared tree rather than a per-container one. Its `install/` can lag
`src/` -- a node added to `setup.py` without a rebuild surfaces only as `No executable
found` in a client log, minutes into a session.

Boot therefore checks that the client executable is installed and, when it is not,
incrementally rebuilds that one package on a single workstation (never `rm -rf build
install`, which would pull the overlay out from under every other workstation's running
client). Progress goes to `logs/tui/piper_build.log`. Set `ARMORY_AUTO_BUILD_WORKSPACE=false`
to have boot report a stale overlay instead of building it. `[B] Build WS` in the TUI forces
a rebuild, for when `src/` changes under an executable that is already installed.

Tests run with:

    uv run pytest

The optional run_real.py trial tooling expects this repository to sit beside the Armory core
checkout, as ../armory.
