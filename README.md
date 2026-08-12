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

## Episode data

Each workstation's episode store lives at `ARMORY_EPISODE_HOST_ROOT` (default
`/home/data_collection/<ssh user>`), bind-mounted into the container at `/datasets`. Both
the client node's write path and the SFTP fetch derive from that one setting, so they
cannot point at different directories — a fetch root that disagrees with the mount looks
exactly like a trial that produced nothing.

A trial gets its own subtree, `<root>/armory_episodes/<trial id>/`, where the trial id is
the output directory's name. RealSaver numbers episodes from zero per client process and
creates its folders with `exist_ok`, so a shared directory means each trial overwrites the
last at the same path — and any robot that produced nothing this trial still has the
previous trial's episode there for the recursive fetch to collect as though it were fresh.
Starting a client outside a trial (`[C]` in the TUI) has no trial id and keeps writing to
`<root>/armory_episodes/` as before.

## Per-robot launch overrides

`run_real.py --het-config-path` takes a YAML with `control_hz`, `language_index`,
`min_execution_horizon` and `max_execution_horizon` sections, each mapping workstation id
to a value. Anything else is a hard error: `client_node_armory` parses its flags with
`parse_known_args`, so a flag it does not define is discarded rather than rejected, and an
unrecognised config section would produce a trial that runs homogeneous while reporting
that it applied the config. `language_index` resolves against `configs/real_task_index.json`
(override with `--task-index-path`).

Tests run with:

    uv run pytest

The optional run_real.py trial tooling expects this repository to sit beside the Armory core
checkout, as ../armory.
