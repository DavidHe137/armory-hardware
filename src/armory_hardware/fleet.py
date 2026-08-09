"""Asynchronous SSH/Docker fleet controller for real robot workstations.

Runs an asyncio event loop on a background thread so callers (curses TUIs,
CLIs, notebooks) never block on network calls. The class exposes a thread-safe
public API that returns ``concurrent.futures.Future`` objects.

The controller emits human-readable status events through ``self.logger``.
Callers that want to render those events in a UI should attach a
``logging.Handler`` to the logger they pass in (or to ``logging.getLogger
("armory_hardware") if they pass nothing).
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import re
import shlex
import threading
import time
from collections.abc import Callable

import asyncssh

from armory_hardware.config import FleetConfig, Robot, RobotStatus

DOCKER_CONTAINER = "piper_env"
DATA_COLLECTION_DIR = "/CS4803ARM_Lab/user_data/data_collection"
PIPER_WORKSPACE_DIR = "/CS4803ARM_Lab/user_data/piper_ros"
# Default data dir written by RealSaver inside the container (bind-mounted to the
# workstation host). Override via the run_trial(remote_subdir=...) parameter.
DEFAULT_EPISODE_SUBDIR = "armory_episodes"

# Sentinel files used by the trial startup barrier (see FleetDispatcher._run_trial
# and piper_client_armory --barrier). They live inside the container's /tmp.
BARRIER_READY_FLAG = "/tmp/armory_ready.flag"
BARRIER_GO_FLAG = "/tmp/armory_go.flag"

# Boot (see _boot_script). The whole sequence is one remote script, so its steps
# report back as marker lines on stdout instead of as separate SSH round trips.
BOOT_ENTRYPOINT = "/CS4803ARM_Lab/user_data/piper_ros/entrypoint.sh"
# armory_client, as the container sees it. The lab image pip-installs it
# editable, but records the pre-restructure layout
# (armory/packages/armory-client/src), so `import armory_client` fails outright
# and takes the client node down on its import line. Putting the real location
# on PYTHONPATH at `docker run` is what makes it reachable: the image's stale
# .pth cannot be fixed from here, and entrypoint.sh cannot do it either — boot
# runs that as a one-shot `docker exec`, so its exports never reach the separate
# exec that launches the client. Container env does reach every exec.
ARMORY_CLIENT_SRC = "/CS4803ARM_Lab/user_data/armory/armory-client/src"
BOOT_WAIT_SEC = 15.0
BOOT_POLL_INTERVAL_SEC = 0.25

_BOOT_ALREADY = "@@ALREADY_RUNNING"
_BOOT_STARTING = "@@STARTING"
_BOOT_NOT_UP = "@@NOT_UP"
_BOOT_NO_PIPER_ROS = "@@NO_PIPER_ROS"
_BOOT_ENTRYPOINT = "@@ENTRYPOINT"
_BOOT_OK = "@@BOOTED"

# Markers worth replaying to the operator, phrased in past tense: the script is
# one round trip, so these describe what already happened rather than narrating
# it live. The rest are decided on by _boot_single and reported as an outcome.
_BOOT_PROGRESS = {
    _BOOT_ALREADY: "container already running — skipped boot",
    _BOOT_STARTING: "no container found; ran docker run",
    _BOOT_ENTRYPOINT: "container up; launched entrypoint",
}

# The lab image defines `p` only as an alias, appended to the container's
# ~/.bashrc *below* the stock "if not running interactively, don't do anything"
# guard. Aliases also don't expand in non-interactive shells. Both facts used to
# force `bash -ic` here, which needs a TTY it never gets over asyncssh and buries
# the real error under job-control warnings. We invoke the script the alias points
# at instead, sourcing the same three things the bashrc tail sources.
PIPER_CTRL_SCRIPT = "/piper_ros/scripts/p_ctrl.py"
PIPER_ROS_SETUP = "/piper_ros/install/setup.bash"
# Exports ROS_DOMAIN_ID derived from the workstation hostname — required for ROS 2
# peering, so it must be sourced even though we bypass the alias it also defines.
PIPER_RC_SETUP = "/piper_ros/scripts/setup_rc.sh"
PIPER_LD_LIBRARY_PATH = "/usr/lib/x86_64-linux-gnu"

# Workspace build (see _ensure_workspace_built).
#
# /piper_ros is the same NFS checkout on every workstation — the bind-mount
# source under piper_root is an operator home directory served to the whole lab.
# So `src/` can gain a node (and setup.py an entry point) without `install/`
# ever being rebuilt, and `ros2 run` then fails with a bare "No executable
# found" in the client log, minutes into a session. Boot checks for the
# installed executable directly and rebuilds the one package when it is absent.
#
# One shared tree also means the build is a fleet-level operation, not a
# per-container one: _boot_robots elects a single workstation to run it rather
# than letting 14 containers colcon-build into the same install/ at once.
PIPER_PACKAGE = "piper"
PIPER_WORKSPACE_ROOT = "/piper_ros"
PIPER_CLIENT_EXECUTABLE = "piper_client_armory"
PIPER_CLIENT_EXECUTABLE_PATH = (
    f"{PIPER_WORKSPACE_ROOT}/install/{PIPER_PACKAGE}/lib/{PIPER_PACKAGE}/"
    f"{PIPER_CLIENT_EXECUTABLE}"
)
ROS_UNDERLAY_SETUP = "/opt/ros/humble/setup.bash"
# Second line of defence against two operators booting at once — flock over
# NFSv4 is advisory and not perfectly reliable, so the single-builder election
# in _boot_robots is what actually prevents concurrent builds.
PIPER_BUILD_LOCK = f"{PIPER_WORKSPACE_ROOT}/.armory_build.lock"
# colcon's build space for the package. Its symlinked setup.py is the marker
# colcon reads as "this was last built with --symlink-install" — see
# _build_workspace for why the build clears it.
PIPER_BUILD_DIR = f"{PIPER_WORKSPACE_ROOT}/build/{PIPER_PACKAGE}"
PIPER_DEVELOP_MARKER = f"{PIPER_BUILD_DIR}/setup.py"
# setuptools' build_py staging tree. Copied from, not into, install/ — see
# _build_workspace for why a copy install has to discard it first.
PIPER_STAGING_DIR = f"{PIPER_BUILD_DIR}/build"
# Exit status of the last build, on the workstation host that ran it.
PIPER_BUILD_STATUS_PATH = "/tmp/armory_workspace_build.status"
DEFAULT_BUILD_TIMEOUT_SEC = 900.0
BUILD_POLL_INTERVAL_SEC = 5.0
BUILD_PROGRESS_INTERVAL_SEC = 30.0
# Shorter than the overall cap so a contended lock reports as such rather than
# consuming the whole build budget waiting.
BUILD_LOCK_WAIT_SEC = 300.0
# Distinct exit status for "someone else is already building" (EX_TEMPFAIL).
BUILD_LOCK_BUSY_STATUS = 75

# Emitted by an interactive bash with no controlling terminal. Harmless, but it
# lands on stderr and used to be reported as though the command had failed.
_JOB_CONTROL_NOISE = (
    "cannot set terminal process group",
    "no job control in this shell",
)


def _container_script(command: str) -> str:
    """Expand a fleet command into a self-contained script for the container.

    ``p <args>`` becomes an explicit ``source``-then-``python3`` sequence so it
    works under a non-interactive shell; anything else is passed through
    untouched. Sourcing the setup files by name also means a broken or unmounted
    ``/piper_ros`` fails the command outright, instead of leaving ``p`` quietly
    undefined and surfacing as ``p: command not found`` on stderr.
    """
    if command != "p" and not command.startswith("p "):
        return command
    args = command[1:].strip()
    return (
        f"source {PIPER_ROS_SETUP} && "
        f"source {PIPER_RC_SETUP} && "
        f"export LD_LIBRARY_PATH={PIPER_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH && "
        f"python3 {PIPER_CTRL_SCRIPT} {args}"
    )


def _clean_stderr(stderr: str) -> str:
    """Drop bash's job-control warnings so only real errors are reported."""
    lines = [
        line
        for line in stderr.splitlines()
        if not any(noise in line for noise in _JOB_CONTROL_NOISE)
    ]
    return "\n".join(lines).strip()


class FleetController:
    """Orchestrates a fleet of real robot workstations over SSH + Docker."""

    def __init__(self, config: FleetConfig, logger: logging.Logger | None = None):
        self.config = config
        # Event stream — every human-readable status line goes here. Consumers
        # attach handlers to render in a UI or pipe to stdout. Defaults to the
        # "armory_hardware" logger so standard logging.basicConfig() works.
        self.logger = logger or logging.getLogger("armory_hardware")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        # robot.id -> Task producing that robot's connection. Storing the task
        # rather than the connection is what lets N robots dial concurrently:
        # see _get_connection for why the lock can't wrap the connect itself.
        self._connections: dict[int, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        # Per-robot cache of the resolved host-side user_data path
        # (echo $PIPER_DOCKER_DIR/../user_data). Populated lazily on first fetch.
        self._user_data_host_path: dict[int, str] = {}

        # Per-workstation file loggers (internal — not the event stream).
        self._loggers: dict[int, logging.Logger] = {}
        self._system_logger = self._make_logger(
            "armory_system",
            os.path.join(config.log_dir, "armory_system.log"),
        )
        for robot in config.robots:
            self._loggers[robot.id] = self._make_logger(
                f"workstation_{robot.id}",
                os.path.join(config.log_dir, f"workstation_{robot.id}.log"),
            )

    # ── lifecycle ────────────────────────────────────────────────

    def start(self):
        self._thread.start()

    def stop(self):
        """Gracefully close all SSH connections and stop the loop."""
        future = asyncio.run_coroutine_threadsafe(self._close_all(), self._loop)
        try:
            future.result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # ── public API (thread-safe, returns futures) ────────────────

    def submit(self, coro) -> asyncio.Future:
        """Submit a coroutine to the background event loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def check_all_status(self, callback: Callable | None = None):
        """Check status of all robots concurrently. Returns a Future."""
        return self.submit(self._check_all_status(callback))

    def run_on_robots(
        self,
        robots: list[Robot],
        command: str,
        callback: Callable | None = None,
    ):
        """Run a command inside the Docker container on multiple robots."""
        return self.submit(self._run_on_robots(robots, command, callback))

    def boot_robots(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        """Start the Docker container on multiple robots."""
        return self.submit(self._boot_robots(robots, callback))

    def shutdown_robots(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        """Stop and remove the Docker container on multiple robots."""
        return self.submit(self._shutdown_robots(robots, callback))

    def build_workspace(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        """Rebuild the piper workspace, unconditionally.

        Boot already does this on its own when the client executable is missing;
        this is the explicit form, for when ``src/`` changed under an executable
        that is still installed. ``robots`` selects which workstation runs the
        build — the workspace itself is one shared checkout, so only the first
        is used.
        """
        return self.submit(self._build_workspace_on_first(robots, callback))

    def start_tunnels(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        """Start workstation-level SSH tunnels outside Docker."""
        return self.submit(self._start_tunnels(robots, callback))

    def kill_tunnels(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        """Kill workstation-level SSH tunnels outside Docker."""
        return self.submit(self._kill_tunnels(robots, callback))

    def start_webcam_streams(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        """Start the Logitech webcam RTSP stream on each workstation host."""
        return self.submit(self._start_webcam_streams(robots, callback))

    def kill_webcam_streams(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        """Stop the Logitech webcam RTSP stream on each workstation host."""
        return self.submit(self._kill_webcam_streams(robots, callback))

    def start_data_listeners(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        """Start the data collection listener inside Docker."""
        return self.submit(self._start_data_listeners(robots, callback))

    def start_clients(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
        extra_args_per_robot: dict[int, str] | None = None,
    ):
        """Start the Piper client node inside Docker.

        ``extra_args_per_robot`` maps workstation id to a string appended
        verbatim to the launch command (e.g.
        ``{14: "--ros-args -p control_hz:=20"}``).
        """
        return self.submit(
            self._start_clients(robots, callback, extra_args_per_robot=extra_args_per_robot)
        )

    def kill_data_listeners(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        """Stop the data collection listener inside Docker."""
        return self.submit(self._kill_data_listeners(robots, callback))

    def kill_clients(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
        grace_sec: float = 5.0,
    ):
        """Stop the Piper client node inside Docker.

        Sends SIGINT first (which the client's rclpy.spin loop translates to a
        KeyboardInterrupt → clean destroy_node → RealSaver flush), waits
        ``grace_sec`` seconds, then sends SIGKILL.
        """
        return self.submit(self._kill_clients(robots, callback, grace_sec=grace_sec))

    def check_clients(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        """Check whether Piper client nodes are running inside Docker."""
        return self.submit(self._check_clients(robots, callback))

    def fetch_episode_data(
        self,
        robots: list[Robot],
        local_dir: pathlib.Path,
        remote_subdir: str = DEFAULT_EPISODE_SUBDIR,
        include_video: bool = False,
        callback: Callable | None = None,
    ):
        """Pull each robot's episode tree to ``local_dir/<robot_name>/`` via SFTP.

        ``remote_subdir`` is a path relative to the workstation's user_data
        bind mount (host side of /CS4803ARM_Lab/user_data/). Default
        ``armory_episodes`` matches RealSaver's default.
        """
        return self.submit(
            self._fetch_episode_data(
                robots, local_dir, remote_subdir, include_video, callback
            )
        )

    # ── internal async methods ──────────────────────────────────

    async def _get_connection(self, robot: Robot) -> asyncssh.SSHClientConnection:
        """Return a live SSH connection to ``robot``, dialling if necessary.

        The lock is held only long enough to publish an in-flight connect task,
        never across the connect itself. That distinction is the whole point:
        every fan-out method here gathers one coroutine per robot, and each of
        those calls this first — so a lock spanning ``asyncssh.connect`` would
        quietly serialise all N handshakes and make the gathers decorative.
        Concurrent callers for the *same* robot still share one task, so a
        fleet-wide command opens exactly one session per workstation.

        Liveness is a local ``is_closed()`` check rather than a probe command.
        A probe cost a full round trip on every call and still raced: a
        connection can die between the probe and the real command, so callers
        have to handle that failure anyway.
        """
        for _attempt in range(2):
            async with self._lock:
                task = self._connections.get(robot.id)
                if task is None:
                    task = asyncio.ensure_future(self._connect(robot))
                    self._connections[robot.id] = task

            try:
                # Shielded so one cancelled caller doesn't tear down the
                # connect that its peers are also waiting on.
                conn = await asyncio.shield(task)
            except Exception:
                await self._forget_connection(robot.id, task)
                raise

            if not conn.is_closed():
                return conn
            # Cached session has since dropped — discard and dial once more.
            await self._forget_connection(robot.id, task)

        raise ConnectionError(f"WS-{robot.id}: SSH connection to {robot.ip} keeps closing")

    async def _connect(self, robot: Robot) -> asyncssh.SSHClientConnection:
        return await asyncssh.connect(
            robot.ip,
            username=self.config.ssh_user,
            known_hosts=None,
            connect_timeout=8,
        )

    async def _forget_connection(
        self,
        robot_id: int,
        task: asyncio.Task | None = None,
    ) -> None:
        """Drop a robot's cached connection so the next call redials.

        ``task`` guards against evicting a *replacement* connection opened by a
        concurrent caller after this one already failed.
        """
        async with self._lock:
            current = self._connections.get(robot_id)
            if current is None or (task is not None and current is not task):
                return
            del self._connections[robot_id]

        if current.done() and not current.cancelled():
            try:
                current.result().close()
            except Exception:
                pass

    async def _close_all(self):
        async with self._lock:
            tasks = list(self._connections.values())
            self._connections.clear()

        for task in tasks:
            if not task.done():
                task.cancel()
                continue
            if task.cancelled():
                continue
            try:
                task.result().close()
            except Exception:
                pass

    async def _check_status(self, robot: Robot) -> RobotStatus:
        logger = self._loggers[robot.id]
        try:
            conn = await self._get_connection(robot)
            result = await conn.run(
                f"docker ps --filter name={DOCKER_CONTAINER} -q",
                timeout=10,
            )
            if result.stdout.strip():
                logger.info("Docker container running — status: booted")
                self._emit(f"WS-{robot.id}: container running (booted)")
                return RobotStatus.BOOTED
            else:
                logger.info("Docker container not running — status: offline")
                self._emit(f"WS-{robot.id}: container not running (offline)")
                return RobotStatus.OFFLINE
        except Exception as e:
            logger.error("Status check failed: %s", e)
            self._emit(f"WS-{robot.id}: unreachable — {e}")
            await self._forget_connection(robot.id)
            return RobotStatus.OFFLINE

    async def _check_all_status(self, callback: Callable | None = None):
        tasks = []
        for robot in self.config.robots:
            tasks.append(self._check_and_update(robot))
        await asyncio.gather(*tasks)
        if callback:
            callback()

    async def _check_and_update(self, robot: Robot):
        # Preserve 'online' status (set by dummy Connect to Server)
        if robot.status == RobotStatus.ONLINE:
            status = await self._check_status(robot)
            if status == RobotStatus.OFFLINE:
                robot.status = RobotStatus.OFFLINE
            # else keep ONLINE
        else:
            robot.status = await self._check_status(robot)

    async def _run_on_robots(
        self,
        robots: list[Robot],
        command: str,
        callback: Callable | None = None,
    ):
        results = {}
        tasks = [self._exec_in_docker(r, command) for r in robots]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for robot, out in zip(robots, outputs):
            results[robot.id] = out
        if callback:
            callback(results)
        return results

    async def _exec_in_docker(self, robot: Robot, command: str) -> str:
        logger = self._loggers[robot.id]
        try:
            conn = await self._get_connection(robot)
            script = _container_script(command)
            cmd = f"docker exec {DOCKER_CONTAINER} bash -lc {shlex.quote(script)}"
            logger.info("Executing: %s", cmd)
            # `command` (not `script`) in operator-facing output — the expanded
            # script is several lines of sourcing and only useful in the log.
            self._emit(f"WS-{robot.id}: running '{command}'")
            result = await conn.run(cmd, timeout=30, check=False)
            stdout = result.stdout.strip()
            stderr = _clean_stderr(result.stderr)
            if stdout:
                logger.info("stdout: %s", stdout)
                self._emit(f"WS-{robot.id}: {stdout[:120]}")
            if stderr:
                logger.warning("stderr: %s", stderr)
                self._emit(f"WS-{robot.id} err: {stderr[:120]}")
            if result.exit_status != 0:
                detail = stderr or stdout or "no output"
                logger.error(
                    "Command '%s' exited %s: %s", command, result.exit_status, detail
                )
                self._emit(
                    f"WS-{robot.id}: FAILED '{command}' "
                    f"(exit {result.exit_status}) — {detail[:100]}"
                )
                return f"ERROR: '{command}' exited {result.exit_status} — {detail}"
            return stdout or stderr or "(no output)"
        except Exception as e:
            logger.error("Command '%s' failed: %s", command, e)
            self._emit(f"WS-{robot.id}: FAILED '{command}' — {e}")
            robot.status = RobotStatus.OFFLINE
            return f"ERROR: {e}"

    async def _boot_robots(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        results = {}
        tasks = [self._boot_single(r) for r in robots]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for robot, out in zip(robots, outputs):
            results[robot.id] = out

        # The workspace is shared, so one workstation answers for the fleet.
        booted = [r for r in robots if r.status is RobotStatus.BOOTED]
        if booted:
            note = await self._ensure_workspace_built(booted[0])
            if note:
                results[booted[0].id] = f"{results[booted[0].id]}; {note}"

        if callback:
            callback(results)
        return results

    def _require_piper_root(self, robot: Robot) -> str:
        """Lab checkout root on the workstation, or raise with a fixable message."""
        root = self.config.piper_root
        if not root:
            raise RuntimeError(
                f"WS-{robot.id}: piper_docker_dir is not configured — set "
                "ARMORY_PIPER_DOCKER_DIR in .env (or docker.piper_dir in the "
                "fleet YAML) to the workstation's PIPER_DOCKER_DIR"
            )
        return root

    def _boot_script(self, piper_root: str) -> str:
        """One-shot boot: check, clean, run, wait for up, verify, entrypoint.

        Written as a single remote script rather than the six ``conn.run``
        calls it replaces. Each of those was a full round trip that the next
        step's outcome depended on, so a 14-workstation boot paid 6×RTT×14 in
        latency alone — and the fixed ``sleep 2`` between ``docker run`` and the
        liveness check was charged whether the container took 200ms or didn't
        come up at all.

        Progress is reported by marker lines on stdout (``@@`` prefixed) that
        ``_boot_single`` maps back to operator-facing events, so collapsing the
        round trips doesn't collapse the log detail.
        """
        # piper_start uses `docker run -it` which blocks and needs a TTY.
        # Instead, run in detached mode (-d) with --entrypoint overridden to
        # `sleep` so the container stays alive, then launch the real entrypoint
        # via docker exec.
        #
        # The lab paths are interpolated from config rather than expanded by the
        # remote shell. --mount (not -v) is used for them so Docker rejects a
        # source that does not exist instead of helpfully creating an empty
        # root-owned directory and booting a container with nothing in it.
        # $DISPLAY stays shell-expanded — it is genuinely per-session, and the
        # inner `bash -lc` this script runs under still expands it.
        run_cmd = (
            "docker run -d"
            " --privileged --net=host --runtime=nvidia --gpus all"
            f" --name {DOCKER_CONTAINER}"
            " --entrypoint sleep"
            " -e NVIDIA_DRIVER_CAPABILITIES=all"
            " -e DISPLAY=$DISPLAY"
            f" -e PYTHONPATH={ARMORY_CLIENT_SRC}"
            " -v /tmp/.X11-unix/:/tmp/.X11-unix/:rw"
            f" --mount type=bind,source={piper_root}/user_data,target=/CS4803ARM_Lab/user_data"
            f" --mount type=bind,source={piper_root}/assets,target=/CS4803ARM_Lab/assets"
            f" --mount type=bind,source={piper_root}/user_data/piper_ros,target=/piper_ros"
            " -v /home/data_collection/dhe83/:/datasets/"
            " -v /dev:/dev"
            " -w /CS4803ARM_Lab/user_data/data_collection"
            " firefall/cluster_piper_env:v2"
            " infinity"
        )
        running = f"docker ps --filter name={DOCKER_CONTAINER} -q | grep -q ."
        deadline = int(BOOT_WAIT_SEC / BOOT_POLL_INTERVAL_SEC)

        return (
            f"if {running}; then echo '{_BOOT_ALREADY}'; exit 0; fi; "
            f"docker rm -f {DOCKER_CONTAINER} >/dev/null 2>&1 || true; "
            f"echo '{_BOOT_STARTING}'; "
            f"{run_cmd} || {{ echo '{_BOOT_NOT_UP}'; exit 1; }}; "
            # Poll instead of sleeping a flat interval: a healthy container is
            # usually up in well under a second.
            f"i=0; while [ $i -lt {deadline} ]; do "
            f"  if {running}; then break; fi; "
            f"  sleep {BOOT_POLL_INTERVAL_SEC}; i=$((i+1)); "
            "done; "
            f"if ! {running}; then echo '{_BOOT_NOT_UP}'; exit 1; fi; "
            # A container whose /piper_ros is empty starts fine and fails later
            # on the first `p` command. Catch it here, while the cause is still
            # in view.
            f"if ! docker exec {DOCKER_CONTAINER} test -f {PIPER_ROS_SETUP}; then "
            f"  echo '{_BOOT_NO_PIPER_ROS}'; exit 1; "
            "fi; "
            f"echo '{_BOOT_ENTRYPOINT}'; "
            f"docker exec -d {DOCKER_CONTAINER} bash -c {shlex.quote(BOOT_ENTRYPOINT)}; "
            f"echo '{_BOOT_OK}'"
        )

    async def _boot_single(self, robot: Robot) -> str:
        logger = self._loggers[robot.id]
        try:
            conn = await self._get_connection(robot)
            piper_root = self._require_piper_root(robot)

            logger.info("Booting Docker container")
            self._emit(f"WS-{robot.id}: booting Docker container...")
            result = await conn.run(
                self._bash_command(self._boot_script(piper_root)),
                timeout=BOOT_WAIT_SEC + 60,
                check=False,
            )

            markers = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("@@"):
                    markers.append(line)
                elif line:
                    logger.info("boot output: %s", line)
            for marker in markers:
                note = _BOOT_PROGRESS.get(marker)
                if note:
                    self._emit(f"WS-{robot.id}: {note}")

            stderr = result.stderr.strip()
            if stderr:
                logger.warning("boot stderr: %s", stderr)

            if _BOOT_ALREADY in markers:
                msg = "Container already running — skipping boot"
                logger.info(msg)
                robot.status = RobotStatus.BOOTED
                return msg

            if _BOOT_NO_PIPER_ROS in markers:
                msg = (
                    f"container is up but {PIPER_ROS_SETUP} is missing — "
                    f"check that {piper_root}/user_data/piper_ros is populated"
                )
                logger.error(msg)
                self._emit(f"WS-{robot.id}: {msg}")
                return f"ERROR: {msg}"

            if _BOOT_OK not in markers:
                detail = stderr or "container not detected after docker run"
                logger.warning("Container failed to start: %s", detail)
                self._emit(f"WS-{robot.id}: container failed to start")
                return f"Boot failed — {detail}"

            logger.info("Container started successfully")
            self._emit(f"WS-{robot.id}: boot successful")
            robot.status = RobotStatus.BOOTED
            return "Boot successful"
        except Exception as e:
            logger.error("Boot failed: %s", e)
            self._emit(f"WS-{robot.id}: boot FAILED — {e}")
            robot.status = RobotStatus.OFFLINE
            return f"ERROR: {e}"

    # ── workspace build (fleet-level, shared /piper_ros) ─────────

    async def _build_workspace_on_first(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        results: dict[int, str] = {}
        builders = [r for r in robots if r.status is not RobotStatus.OFFLINE]
        if not builders:
            self._emit("workspace build skipped — no booted workstation to build on")
            if callback:
                callback(results)
            return results

        builder = builders[0]
        note = await self._ensure_workspace_built(builder, force=True)
        results[builder.id] = note or "workspace already up to date"
        if callback:
            callback(results)
        return results

    async def _client_executable_present(
        self,
        conn: asyncssh.SSHClientConnection,
    ) -> bool:
        """True when the installed overlay actually carries the client node.

        A ``test -x`` on the installed script, not ``ros2 pkg executables`` —
        the package resolves either way, and it is the executable's absence
        that ``ros2 run`` reports as the unhelpful "No executable found".
        """
        probe = await conn.run(
            f"docker exec {DOCKER_CONTAINER} "
            f"test -x {shlex.quote(PIPER_CLIENT_EXECUTABLE_PATH)}",
            timeout=15,
            check=False,
        )
        return probe.exit_status == 0

    async def _ensure_workspace_built(
        self,
        robot: Robot,
        force: bool = False,
    ) -> str | None:
        """Build the piper workspace if its overlay is missing the client node.

        Returns a note for the caller's result map, or ``None`` when the
        workspace was already good (the common case — one ``test -x``, so a
        normal boot pays no measurable cost).

        ``force`` builds regardless, for an operator who knows ``src/`` moved
        under an already-installed executable.
        """
        logger = self._loggers[robot.id]
        try:
            conn = await self._get_connection(robot)
            present = await self._client_executable_present(conn)
        except Exception as e:
            logger.error("Workspace check failed: %s", e)
            self._emit(f"WS-{robot.id}: workspace check FAILED — {e}")
            return f"workspace check failed — {e}"

        if present and not force:
            logger.info("Workspace overlay has %s", PIPER_CLIENT_EXECUTABLE)
            return None

        if not present:
            stale = (
                f"workspace overlay is stale — {PIPER_CLIENT_EXECUTABLE} is not "
                f"installed in {PIPER_WORKSPACE_ROOT}/install"
            )
            logger.warning(stale)
            self._emit(f"WS-{robot.id}: {stale}")
            if not self.config.auto_build_workspace:
                return (
                    f"ERROR: {stale}; auto-build is off — enable docker.auto_build "
                    "(or ARMORY_AUTO_BUILD_WORKSPACE) or build the workspace by hand"
                )

        try:
            return await self._build_workspace(robot, conn)
        except Exception as e:
            # Report as a result, never as an exception. Both callers reach
            # their ``callback(results)`` only by returning normally, and that
            # callback is what unlocks the TUI — an SSH timeout escaping from
            # here (the post-build `test -x` probe is the reachable one) left
            # the console permanently wedged with nothing in the log.
            logger.error("Workspace build failed: %s", e)
            self._emit(f"WS-{robot.id}: workspace build FAILED — {e}")
            return f"ERROR: workspace build failed — {e}"

    async def _build_workspace(
        self,
        robot: Robot,
        conn: asyncssh.SSHClientConnection,
    ) -> str:
        """Incrementally build the piper package on one workstation.

        Detached-and-polled rather than a single long ``conn.run``: a colcon
        build outlives any sane SSH timeout, and a dropped connection midway
        through would leave a half-written install/ tree for the whole lab.

        Never ``rm -rf build install`` the way the lab's own entrypoint.sh does
        (commented out there) — that would delete the overlay out from under
        every other workstation's running client.
        """
        logger = self._loggers[robot.id]
        log_path = os.path.join(self.config.log_dir, "piper_build.log")
        log_path_q = shlex.quote(log_path)
        log_dir_q = shlex.quote(os.path.dirname(log_path))
        status_q = shlex.quote(PIPER_BUILD_STATUS_PATH)

        # Lock acquisition is kept separate from the build so a contended lock
        # (another operator's TUI) is distinguishable from a compile error —
        # `flock <file> <cmd>` collapses both into exit 1.
        #
        # A plain copy install, deliberately not --symlink-install. For an
        # ament_python package colcon implements the symlink variant as
        # `setup.py develop --editable`, and the lab image carries setuptools
        # 80, which dropped the develop command's easy_install-derived options
        # outright — so it fails with "option --editable not recognized" before
        # compiling anything. The existing install/ tree holds real .py copies
        # rather than symlinks, so a copy install is also what it was built
        # with. Cost of the difference: editing an *existing* node now needs a
        # rebuild too, where a symlink overlay would have picked it up for free.
        # That is what `[B] Build WS` is for.
        # Two pieces of leftover build state have to go first. Both live under
        # the build space, which is pure derived state — never install/, the
        # overlay every other workstation's running client is reading.
        #
        # The develop marker, or colcon will not get as far as compiling. It
        # treats "<build space>/setup.py is a symlink, with <pkg>.egg-info
        # beside it" as "the last build was --symlink-install" and tries to undo
        # that with `setup.py develop --uninstall --editable` — options
        # setuptools 80 dropped along with --editable. colcon removes the
        # symlink itself after a *successful* develop build; a failed one leaves
        # it behind, which is precisely what a half-finished symlink build (or
        # an older version of this code) hands us.
        #
        # The staging tree, or the build compiles and still installs the wrong
        # thing. A copy install goes src/ -> build/<pkg>/build/lib -> install/,
        # and setuptools' build_py refreshes that middle step only when the
        # source file is *newer* than the staged one. On a shared NFS checkout
        # that does not hold — a pull can land a file whose mtime predates the
        # staged copy, and the build then reinstalls the stale staged version
        # and reports success. Not hypothetical: it is how install/ came to hold
        # a client node written against a previous armory_client API.
        marker_q = shlex.quote(PIPER_DEVELOP_MARKER)
        staging_q = shlex.quote(PIPER_STAGING_DIR)
        container_script = (
            f"source {ROS_UNDERLAY_SETUP} || exit 1; "
            f"cd {shlex.quote(PIPER_WORKSPACE_ROOT)} || exit 1; "
            f"exec 9>{shlex.quote(PIPER_BUILD_LOCK)} || exit 1; "
            f"flock -w {int(BUILD_LOCK_WAIT_SEC)} 9 || exit {BUILD_LOCK_BUSY_STATUS}; "
            f"if [ -L {marker_q} ]; then rm -f {marker_q}; fi; "
            f"rm -rf {staging_q}; "
            f"colcon build --packages-select {PIPER_PACKAGE}"
        )
        build_cmd = (
            f"docker exec {DOCKER_CONTAINER} bash -lc "
            f"{shlex.quote(container_script)} >> {log_path_q} 2>&1; "
            f"echo $? > {status_q}"
        )
        script = (
            f"mkdir -p {log_dir_q}; "
            f"touch {log_path_q} || exit 1; "
            f"rm -f {status_q}; "
            f"printf '\\n[%s] Building {PIPER_PACKAGE} workspace on WS-{robot.id}\\n' "
            f"\"$(date '+%F %T')\" >> {log_path_q}; "
            f"nohup bash -c {shlex.quote(build_cmd)} >/dev/null 2>&1 < /dev/null &"
        )

        self._emit(
            f"WS-{robot.id}: building {PIPER_PACKAGE} workspace "
            f"(shared by the fleet); log {log_path}"
        )
        logger.info("Starting workspace build; log %s", log_path)
        try:
            await conn.run(self._bash_command(script), timeout=30)
        except Exception as e:
            logger.error("Workspace build launch failed: %s", e)
            self._emit(f"WS-{robot.id}: workspace build launch FAILED — {e}")
            return f"workspace build failed to launch — {e}"

        # Module-level intervals looked up at call time so tests can shorten them.
        status = await self._await_build(
            robot,
            conn,
            DEFAULT_BUILD_TIMEOUT_SEC,
            poll_interval_sec=BUILD_POLL_INTERVAL_SEC,
            progress_interval_sec=BUILD_PROGRESS_INTERVAL_SEC,
        )
        if status is None:
            msg = (
                f"workspace build still running after "
                f"{DEFAULT_BUILD_TIMEOUT_SEC:.0f}s — see {log_path}"
            )
            logger.error(msg)
            self._emit(f"WS-{robot.id}: {msg}")
            return f"ERROR: {msg}"
        if status == BUILD_LOCK_BUSY_STATUS:
            msg = (
                f"another workspace build already holds {PIPER_BUILD_LOCK} — "
                "retry once it finishes"
            )
            logger.error(msg)
            self._emit(f"WS-{robot.id}: {msg}")
            return f"ERROR: {msg}"
        if status != 0:
            tail = await self._tail_remote(conn, log_path)
            logger.error("Workspace build exited %s: %s", status, tail)
            self._emit(
                f"WS-{robot.id}: workspace build FAILED (exit {status}); see {log_path}"
            )
            # The *end* of the tail, not the start. colcon prints the reason a
            # build failed last, under a "--- stderr" banner and below its own
            # progress chatter, so truncating from the front reliably discards
            # the one line the operator needs.
            excerpt = tail[-300:]
            if len(tail) > 300:
                excerpt = f"...{excerpt}"
            return f"ERROR: workspace build exited {status}; see {log_path} — {excerpt}"

        if not await self._client_executable_present(conn):
            msg = (
                f"workspace build succeeded but {PIPER_CLIENT_EXECUTABLE} is still "
                f"missing — check setup.py console_scripts in {PIPER_WORKSPACE_ROOT}"
                f"/src/{PIPER_PACKAGE}"
            )
            logger.error(msg)
            self._emit(f"WS-{robot.id}: {msg}")
            return f"ERROR: {msg}"

        logger.info("Workspace build complete")
        self._emit(f"WS-{robot.id}: workspace build complete")
        return "workspace rebuilt"

    async def _await_build(
        self,
        robot: Robot,
        conn: asyncssh.SSHClientConnection,
        timeout_sec: float,
        poll_interval_sec: float = BUILD_POLL_INTERVAL_SEC,
        progress_interval_sec: float = BUILD_PROGRESS_INTERVAL_SEC,
    ) -> int | None:
        """Poll for the build's exit status. ``None`` means it outlasted the cap.

        Emits a progress line periodically: the build blocks boot, so the event
        stream is the operator's only sign that the TUI has not wedged.
        """
        status_q = shlex.quote(PIPER_BUILD_STATUS_PATH)
        started = time.monotonic()
        deadline = started + timeout_sec
        next_progress = started + progress_interval_sec

        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval_sec)
            try:
                result = await conn.run(f"cat {status_q} 2>/dev/null", timeout=10, check=False)
                raw = result.stdout.strip()
            except Exception as e:
                self._loggers[robot.id].warning("Build poll failed: %s", e)
                continue
            if raw:
                try:
                    return int(raw)
                except ValueError:
                    return 1
            now = time.monotonic()
            if now >= next_progress:
                self._emit(
                    f"WS-{robot.id}: workspace build running "
                    f"({now - started:.0f}s elapsed)..."
                )
                next_progress = now + progress_interval_sec
        return None

    async def _tail_remote(
        self,
        conn: asyncssh.SSHClientConnection,
        path: str,
        lines: int = 20,
    ) -> str:
        try:
            result = await conn.run(
                f"tail -n {lines} {shlex.quote(path)}", timeout=15, check=False
            )
            return result.stdout.strip() or "(no output)"
        except Exception as e:
            return f"(could not read {path}: {e})"

    async def _shutdown_robots(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        results = {}
        tasks = [self._shutdown_single(r) for r in robots]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for robot, out in zip(robots, outputs):
            results[robot.id] = out
        if callback:
            callback(results)
        return results

    async def _shutdown_single(self, robot: Robot) -> str:
        logger = self._loggers[robot.id]
        try:
            conn = await self._get_connection(robot)
            self._emit(f"WS-{robot.id}: stopping Docker container...")
            result = await conn.run(
                f"docker rm -f {DOCKER_CONTAINER}",
                timeout=15,
            )
            stderr = result.stderr.strip()
            if stderr:
                logger.warning("shutdown stderr: %s", stderr)
                self._emit(f"WS-{robot.id}: {stderr[:80]}")
            logger.info("Container stopped and removed")
            self._emit(f"WS-{robot.id}: shutdown complete")
            robot.status = RobotStatus.OFFLINE
            return "Shutdown successful"
        except Exception as e:
            logger.error("Shutdown failed: %s", e)
            self._emit(f"WS-{robot.id}: shutdown FAILED — {e}")
            return f"ERROR: {e}"

    async def _start_tunnels(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        results = {}
        tasks = [self._start_tunnel_single(r) for r in robots]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for robot, out in zip(robots, outputs):
            results[robot.id] = out
        if callback:
            callback(results)
        return results

    async def _kill_tunnels(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        results = {}
        tasks = [self._kill_tunnel_single(r) for r in robots]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for robot, out in zip(robots, outputs):
            results[robot.id] = out
        if callback:
            callback(results)
        return results

    async def _start_tunnel_single(self, robot: Robot) -> str:
        logger = self._loggers[robot.id]
        try:
            error = self._tunnel_config_error()
            if error:
                self._emit(f"WS-{robot.id}: tunnel not started — {error}")
                return f"ERROR: {error}"

            conn = await self._get_connection(robot)
            node = self.config.tunnel_node
            port = self.config.tunnel_port
            user = self.config.tunnel_user
            server = self.config.tunnel_server
            socket = self._tunnel_control_path()
            dest = f"{user}@{node}"
            jump = f"{user}@{server}"
            forward = f"{port}:{node}:{port}"

            script = (
                f"mkdir -p {shlex.quote(os.path.dirname(socket))}; "
                f"if ssh -S {shlex.quote(socket)} -O check {shlex.quote(dest)} "
                ">/dev/null 2>&1; then "
                "echo 'Tunnel already running'; exit 0; fi; "
                f"rm -f {shlex.quote(socket)}; "
                f"ssh -f -N -M -S {shlex.quote(socket)} "
                "-o ExitOnForwardFailure=yes "
                "-o ServerAliveInterval=30 "
                "-o ServerAliveCountMax=3 "
                f"-L {shlex.quote(forward)} "
                f"-J {shlex.quote(jump)} "
                f"{shlex.quote(dest)}; "
                "echo 'Tunnel started'"
            )
            cmd = self._bash_command(script)
            logger.info("Starting tunnel with command: %s", cmd)
            self._emit(f"WS-{robot.id}: starting tunnel {forward} via {jump}")
            result = await conn.run(cmd, timeout=20)
            return self._format_tunnel_result(robot, result, "start")
        except Exception as e:
            logger.error("Tunnel start failed: %s", e)
            self._emit(f"WS-{robot.id}: tunnel start FAILED — {e}")
            return f"ERROR: {e}"

    async def _kill_tunnel_single(self, robot: Robot) -> str:
        logger = self._loggers[robot.id]
        try:
            error = self._tunnel_config_error()
            if error:
                self._emit(f"WS-{robot.id}: tunnel kill skipped — {error}")
                return f"ERROR: {error}"

            conn = await self._get_connection(robot)
            node = self.config.tunnel_node
            port = self.config.tunnel_port
            user = self.config.tunnel_user
            socket = self._tunnel_control_path()
            dest = f"{user}@{node}"
            pattern = (
                f"[s]sh .* -L {port}:{re.escape(node)}:{port} "
                f".*{re.escape(user)}@{re.escape(node)}"
            )
            script = (
                f"ssh -S {shlex.quote(socket)} -O exit {shlex.quote(dest)} "
                ">/dev/null 2>&1 || true; "
                f"rm -f {shlex.quote(socket)}; "
                "if command -v pkill >/dev/null 2>&1; then "
                f"pkill -f {shlex.quote(pattern)} >/dev/null 2>&1 || true; "
                "fi; "
                "echo 'Tunnel stopped'"
            )
            cmd = self._bash_command(script)
            logger.info("Killing tunnel with command: %s", cmd)
            self._emit(f"WS-{robot.id}: killing tunnel for {node}:{port}")
            result = await conn.run(cmd, timeout=15)
            return self._format_tunnel_result(robot, result, "kill")
        except Exception as e:
            logger.error("Tunnel kill failed: %s", e)
            self._emit(f"WS-{robot.id}: tunnel kill FAILED — {e}")
            return f"ERROR: {e}"

    async def _start_webcam_streams(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        results = {}
        tasks = [self._start_webcam_single(r) for r in robots]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for robot, out in zip(robots, outputs):
            results[robot.id] = out
        if callback:
            callback(results)
        return results

    async def _kill_webcam_streams(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        results = {}
        tasks = [self._kill_webcam_single(r) for r in robots]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for robot, out in zip(robots, outputs):
            results[robot.id] = out
        if callback:
            callback(results)
        return results

    def _webcam_rtsp_url(self, robot: Robot) -> str:
        base = self.config.webcam_rtsp_base_url
        if not base:
            raise RuntimeError(
                "No webcam RTSP server configured. Set ARMORY_WEBCAM_RTSP_BASE_URL in .env "
                "(or webcam.rtsp_base_url in the fleet YAML)."
            )
        return f"{base.rstrip('/')}/workstation{robot.id}"

    async def _start_webcam_single(self, robot: Robot) -> str:
        logger = self._loggers[robot.id]
        try:
            conn = await self._get_connection(robot)
            # Runs INSIDE the piper_env container: that is where v4l2-ctl and
            # ffmpeg are installed. /dev is bind-mounted so the C922 is visible.
            # The PID file at /tmp/armory_webcam.pid lives inside the container
            # and is what _kill_webcam_single uses to find the running stream.
            rtsp_url = self._webcam_rtsp_url(robot)
            inner = (
                'device=$(v4l2-ctl --list-devices 2>/dev/null '
                '| grep -A1 "C922.*usb-0000:80:14.0-7" '
                '| grep /dev/video | head -1 | tr -d "[:space:]"); '
                'if [ -z "$device" ]; then '
                '  echo "No C922 webcam device found inside container" >&2; '
                '  exit 1; '
                'fi; '
                'if [ -s /tmp/armory_webcam.pid ] '
                '&& kill -0 "$(cat /tmp/armory_webcam.pid)" 2>/dev/null; then '
                '  echo "Webcam stream already running on $device '
                '(pid $(cat /tmp/armory_webcam.pid))"; exit 0; '
                'fi; '
                'rm -f /tmp/armory_webcam.pid; '
                'mkdir -p /tmp/armory_webcam; '
                'nohup ffmpeg -nostdin -hide_banner -loglevel error '
                '-f v4l2 -input_format mjpeg -framerate 30 -video_size 1280x720 '
                '-i "$device" '
                '-c:v h264_nvenc -preset p1 -tune ull -profile baseline '
                '-rc cbr -b:v 2M '
                '-g 15 -bf 0 '
                '-fflags nobuffer -flags low_delay -avioflags direct '
                '-rtsp_transport udp '
                f'-f rtsp {rtsp_url} '
                '>> /tmp/armory_webcam/stream.log 2>&1 & '
                'pid=$!; '
                'disown "$pid" 2>/dev/null || true; '
                'echo "$pid" > /tmp/armory_webcam.pid; '
                'sleep 0.5; '
                'if kill -0 "$pid" 2>/dev/null; then '
                f'  echo "Webcam stream started on $device -> {rtsp_url} (pid $pid)"; '
                'else '
                '  rm -f /tmp/armory_webcam.pid; '
                '  echo "Webcam stream exited immediately; '
                'check container:/tmp/armory_webcam/stream.log" >&2; '
                '  exit 1; '
                'fi'
            )
            docker_cmd = (
                f"docker exec {DOCKER_CONTAINER} bash -lc {shlex.quote(inner)}"
            )
            cmd = self._bash_command(docker_cmd)
            logger.info("Starting webcam stream: %s", cmd)
            self._emit(f"WS-{robot.id}: starting webcam stream -> {rtsp_url}")
            result = await conn.run(cmd, timeout=15)
            return self._format_process_result(robot, result, "webcam", "start")
        except Exception as e:
            logger.error("Webcam start failed: %s", e)
            self._emit(f"WS-{robot.id}: webcam start FAILED — {e}")
            return f"ERROR: {e}"

    async def _kill_webcam_single(self, robot: Robot) -> str:
        logger = self._loggers[robot.id]
        try:
            conn = await self._get_connection(robot)
            # SIGTERM → grace → SIGKILL on the PID inside the container; pkill
            # is a fallback in case the PID file is stale but an ffmpeg with
            # our per-robot RTSP URL is still up. The pattern is anchored to
            # this robot's URL so it won't touch other workstations' streams.
            rtsp_url = self._webcam_rtsp_url(robot)
            pattern = f"[f]fmpeg.*{re.escape(rtsp_url)}"
            inner = (
                'killed=0; '
                'if [ -s /tmp/armory_webcam.pid ]; then '
                '  pid=$(cat /tmp/armory_webcam.pid); '
                '  if kill -0 "$pid" 2>/dev/null; then '
                '    kill -TERM "$pid" 2>/dev/null || true; '
                '    for i in 1 2 3 4 5 6 7 8 9 10; do '
                '      kill -0 "$pid" 2>/dev/null || break; sleep 0.2; '
                '    done; '
                '    kill -KILL "$pid" 2>/dev/null || true; '
                '    killed=1; '
                '  fi; '
                '  rm -f /tmp/armory_webcam.pid; '
                'fi; '
                f'pkill -f {shlex.quote(pattern)} >/dev/null 2>&1 || true; '
                'if [ "$killed" = "1" ]; then '
                '  echo "Webcam stream stopped"; '
                'else '
                '  echo "Webcam stream not running"; '
                'fi'
            )
            docker_cmd = (
                f"docker exec {DOCKER_CONTAINER} bash -lc {shlex.quote(inner)}"
            )
            cmd = self._bash_command(docker_cmd)
            logger.info("Stopping webcam stream: %s", cmd)
            self._emit(f"WS-{robot.id}: stopping webcam stream")
            result = await conn.run(cmd, timeout=15)
            return self._format_process_result(robot, result, "webcam", "stop")
        except Exception as e:
            logger.error("Webcam stop failed: %s", e)
            self._emit(f"WS-{robot.id}: webcam stop FAILED — {e}")
            return f"ERROR: {e}"

    async def _start_data_listeners(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        return await self._start_detached_docker_processes(
            robots,
            label="data listener",
            command="ros2 launch run_follower.launch.py",
            cwd=DATA_COLLECTION_DIR,
            log_suffix="listener",
            callback=callback,
        )

    def _policy_ros_args(self) -> str:
        """``--ros-args`` overriding the client node's policy endpoint.

        The node declares ``host``/``port`` as ROS parameters defaulting to
        ``localhost:8080``, which only resolves when the SSH tunnel is
        forwarding that port. Pointing the parameters straight at the policy
        server removes the tunnel from the path entirely.

        ``rclpy`` merges command-line parameter overrides with the explicit
        ``parameter_overrides`` the node's ``main`` builds from its own flags,
        and that list never contains ``host``/``port`` — so this cannot collide
        with ``--prompt``/``--control-hz``/``--barrier``.

        Empty when no host is configured, which leaves the node's built-in
        default (and therefore the tunnel workflow) untouched.
        """
        host = self.config.policy_host.strip()
        if not host:
            return ""
        return (
            f"--ros-args -p host:={shlex.quote(host)} "
            f"-p port:={int(self.config.policy_port)}"
        )

    async def _start_clients(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
        extra_args_per_robot: dict[int, str] | None = None,
    ):
        return await self._start_detached_docker_processes(
            robots,
            label="Piper client",
            command="ros2 run piper piper_client_armory",
            cwd=PIPER_WORKSPACE_DIR,
            log_suffix="client",
            callback=callback,
            extra_args_per_robot=extra_args_per_robot,
            trailing_args=self._policy_ros_args(),
        )

    async def _kill_data_listeners(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        return await self._kill_detached_docker_processes(
            robots,
            label="data listener",
            command="ros2 launch run_follower.launch.py",
            log_suffix="listener",
            callback=callback,
        )

    async def _kill_clients(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
        grace_sec: float = 5.0,
    ):
        return await self._kill_detached_docker_processes(
            robots,
            label="Piper client",
            command="ros2 run piper piper_client_armory",
            log_suffix="client",
            callback=callback,
            signal_first="INT",
            grace_sec=grace_sec,
        )

    async def _check_clients(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        return await self._check_detached_docker_processes(
            robots,
            command="ros2 run piper piper_client_armory",
            log_suffix="client",
            callback=callback,
        )

    # ── startup barrier (trial-level rendezvous) ────────────────

    async def _clear_barrier_flags(self, robots: list[Robot]) -> None:
        """Remove any stale barrier flags inside each robot's container.

        Called before ``_start_clients`` so a crashed prior trial can't leave a
        ``go.flag`` lying around that lets the next trial's clients race past
        the barrier before the dispatcher has rendezvoused with them.
        """
        script = f"rm -f {BARRIER_READY_FLAG} {BARRIER_GO_FLAG}"
        docker_cmd = (
            f"docker exec {DOCKER_CONTAINER} bash -lc {shlex.quote(script)}"
        )
        cmd = self._bash_command(docker_cmd)
        await asyncio.gather(
            *(self._barrier_run(r, cmd) for r in robots),
            return_exceptions=True,
        )

    async def _await_clients_ready(
        self,
        robots: list[Robot],
        timeout_sec: float,
        poll_interval_sec: float = 0.25,
    ) -> dict[int, bool]:
        """Poll each container for ``BARRIER_READY_FLAG`` until all flip or timeout.

        Returns ``{robot.id: bool}``. ``True`` means the client wrote ready
        within ``timeout_sec``; ``False`` means it didn't (caller decides
        whether to skip that robot or abort the trial).
        """
        check = f"test -f {BARRIER_READY_FLAG} && echo READY || echo NOT_READY"
        docker_cmd = (
            f"docker exec {DOCKER_CONTAINER} bash -lc {shlex.quote(check)}"
        )
        cmd = self._bash_command(docker_cmd)

        ready: dict[int, bool] = {r.id: False for r in robots}
        pending: list[Robot] = list(robots)
        deadline = time.monotonic() + timeout_sec
        while pending and time.monotonic() < deadline:
            outcomes = await asyncio.gather(
                *(self._barrier_run(r, cmd) for r in pending),
                return_exceptions=True,
            )
            still_pending: list[Robot] = []
            for r, out in zip(pending, outcomes):
                if isinstance(out, Exception):
                    still_pending.append(r)
                    continue
                if str(out).strip() == "READY":
                    ready[r.id] = True
                else:
                    still_pending.append(r)
            pending = still_pending
            if pending:
                await asyncio.sleep(poll_interval_sec)
        return ready

    async def _signal_clients_go(
        self,
        robots: list[Robot],
    ) -> dict[int, bool]:
        """Touch ``BARRIER_GO_FLAG`` inside every container in parallel.

        Returns ``{robot.id: bool}`` indicating which touches succeeded.
        Parallel ``asyncio.gather`` so the per-robot start skew is just the
        spread of when each ``touch`` actually returns, not N×SSH-RTT.
        """
        script = f"touch {BARRIER_GO_FLAG}"
        docker_cmd = (
            f"docker exec {DOCKER_CONTAINER} bash -lc {shlex.quote(script)}"
        )
        cmd = self._bash_command(docker_cmd)
        outcomes = await asyncio.gather(
            *(self._barrier_run(r, cmd) for r in robots),
            return_exceptions=True,
        )
        return {
            r.id: not isinstance(out, Exception)
            for r, out in zip(robots, outcomes)
        }

    async def _barrier_run(self, robot: Robot, cmd: str) -> str:
        """Run a short SSH command for the barrier dance.

        Wrapped so callers can ``asyncio.gather`` and have failures land as
        exceptions in the result list instead of bringing the gather down.
        """
        conn = await self._get_connection(robot)
        result = await conn.run(cmd, timeout=10)
        return result.stdout or ""

    async def _start_detached_docker_processes(
        self,
        robots: list[Robot],
        label: str,
        command: str,
        cwd: str,
        log_suffix: str,
        callback: Callable | None = None,
        extra_args_per_robot: dict[int, str] | None = None,
        trailing_args: str = "",
    ):
        results = {}
        connections = []

        # Warm every connection at once, then launch the long-running commands
        # together. Warming still happens up front (a robot we can't reach
        # should fail before any of its peers are launched), but the warming
        # itself has no reason to be sequential.
        conns = await asyncio.gather(
            *(self._get_connection(robot) for robot in robots),
            return_exceptions=True,
        )
        for robot, conn in zip(robots, conns):
            if isinstance(conn, BaseException):
                self._emit(f"WS-{robot.id}: {label} launch FAILED — {conn}")
                results[robot.id] = f"ERROR: {conn}"
            else:
                connections.append((robot, conn))

        tasks = [
            self._start_detached_docker_process(
                robot,
                conn,
                label,
                self._command_for_robot(
                    command, robot, extra_args_per_robot, trailing_args
                ),
                cwd,
                log_suffix,
            )
            for robot, conn in connections
        ]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for robot, out in zip((robot for robot, _ in connections), outputs):
            results[robot.id] = out

        if callback:
            callback(results)
        return results

    def _command_for_robot(
        self,
        base_command: str,
        robot: Robot,
        extra_args_per_robot: dict[int, str] | None,
        trailing_args: str = "",
    ) -> str:
        """Append per-robot args to the base command. Kill matching uses the
        base command as a substring pattern, so any suffix here is invisible
        to ``_kill_detached_docker_process`` — start/kill stay symmetric.

        ``trailing_args`` lands after the per-robot args, which is what makes it
        safe to put a ``--ros-args`` block there: rclpy consumes everything from
        ``--ros-args`` to the end of argv (or to a ``--``), so anything appended
        behind it would be swallowed as a ROS argument instead of reaching the
        node's own argparse. Dispatcher args open with their own ``--``
        separator and must therefore come first.
        """
        parts = [base_command]
        extra = (extra_args_per_robot or {}).get(robot.id)
        if extra:
            parts.append(extra.strip())
        if trailing_args:
            parts.append(trailing_args.strip())
        resolved = " ".join(p for p in parts if p).rstrip()
        if resolved == base_command:
            return base_command
        # Surface the resolved per-robot command so you can verify the override
        # actually reached the SSH layer without tailing each workstation log.
        self.logger.info("WS-%s: resolved launch command: %s", robot.id, resolved)
        return resolved

    async def _start_detached_docker_process(
        self,
        robot: Robot,
        conn: asyncssh.SSHClientConnection,
        label: str,
        command: str,
        cwd: str,
        log_suffix: str,
    ) -> str:
        logger = self._loggers[robot.id]
        log_path = self._process_log_path(robot, log_suffix)
        log_dir = os.path.dirname(log_path)
        container_pid_path = f"/tmp/armory_{log_suffix}.pid"
        host_pid_path = f"/tmp/armory_{log_suffix}_docker_exec.pid"
        log_path_q = shlex.quote(log_path)
        log_dir_q = shlex.quote(log_dir)
        container_pid_path_q = shlex.quote(container_pid_path)
        host_pid_path_q = shlex.quote(host_pid_path)
        docker_filter_q = shlex.quote(f"name={DOCKER_CONTAINER}")
        container_alive_script = (
            f"if [ -s {container_pid_path_q} ]; then "
            f"pid=$(cat {container_pid_path_q}); "
            "kill -0 \"$pid\" >/dev/null 2>&1; "
            "else "
            "exit 1; "
            "fi"
        )
        container_launch_script = (
            "source ~/.bashrc; "
            f"cd {shlex.quote(cwd)} || exit 1; "
            f"echo $$ > {container_pid_path_q}; "
            f"exec {command}"
        )
        container_wrapper = (
            "if command -v setsid >/dev/null 2>&1; then "
            f"exec setsid bash -ic {shlex.quote(container_launch_script)}; "
            "else "
            f"exec bash -ic {shlex.quote(container_launch_script)}; "
            "fi"
        )
        docker_launch_cmd = (
            f"docker exec {DOCKER_CONTAINER} bash -lc "
            f"{shlex.quote(container_wrapper)}"
        )

        script = (
            f"mkdir -p {log_dir_q}; "
            f"touch {log_path_q} || exit 1; "
            f"if ! docker ps --filter {docker_filter_q} -q | grep -q .; then "
            f"echo '{DOCKER_CONTAINER} is not running' >&2; "
            "exit 1; "
            "fi; "
            f"if docker exec {DOCKER_CONTAINER} bash -lc "
            f"{shlex.quote(container_alive_script)}; then "
            f"pid=$(docker exec {DOCKER_CONTAINER} cat {container_pid_path_q}); "
            f"echo '{label} already running (pid '\"$pid\"'); log: {log_path}'; "
            "exit 0; "
            "fi; "
            f"docker exec {DOCKER_CONTAINER} rm -f {container_pid_path_q} "
            ">/dev/null 2>&1 || true; "
            f"printf '\\n[%s] Starting {label}: {command}\\n' "
            f"\"$(date '+%F %T')\" >> {log_path_q}; "
            f"nohup {docker_launch_cmd} >> {log_path_q} 2>&1 < /dev/null & "
            "host_pid=$!; "
            f"echo $host_pid > {host_pid_path_q}; "
            "sleep 0.5; "
            "if kill -0 \"$host_pid\" >/dev/null 2>&1; then "
            f"pid=$(docker exec {DOCKER_CONTAINER} cat {container_pid_path_q} 2>/dev/null || echo \"$host_pid\"); "
            f"echo '{label} started (pid '\"$pid\"'); log: {log_path}'; "
            "else "
            f"rm -f {host_pid_path_q}; "
            f"echo '{label} exited immediately; check log: {log_path}' >&2; "
            f"tail -n 20 {log_path_q} >&2 || true; "
            "exit 1; "
            "fi"
        )
        cmd = self._bash_command(script)
        logger.info("Starting %s", label)
        self._emit(f"WS-{robot.id}: starting {label}; log {log_path}")

        try:
            result = await conn.run(cmd, timeout=15)
            return self._format_process_result(robot, result, label, "launch")
        except Exception as e:
            logger.error("%s launch failed: %s", label, e)
            self._emit(f"WS-{robot.id}: {label} launch FAILED — {e}")
            return f"ERROR: {e}"

    async def _kill_detached_docker_processes(
        self,
        robots: list[Robot],
        label: str,
        command: str,
        log_suffix: str,
        callback: Callable | None = None,
        signal_first: str = "TERM",
        grace_sec: float = 0.5,
    ):
        results = {}
        tasks = [
            self._kill_detached_docker_process(
                robot, label, command, log_suffix,
                signal_first=signal_first, grace_sec=grace_sec,
            )
            for robot in robots
        ]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for robot, out in zip(robots, outputs):
            results[robot.id] = out
        if callback:
            callback(results)
        return results

    async def _kill_detached_docker_process(
        self,
        robot: Robot,
        label: str,
        command: str,
        log_suffix: str,
        signal_first: str = "TERM",
        grace_sec: float = 0.5,
    ) -> str:
        logger = self._loggers[robot.id]
        container_pid_path = f"/tmp/armory_{log_suffix}.pid"
        host_pid_path = f"/tmp/armory_{log_suffix}_docker_exec.pid"
        log_path = self._process_log_path(robot, log_suffix)
        log_dir = os.path.dirname(log_path)
        pattern = self._process_match_pattern(command)
        log_path_q = shlex.quote(log_path)
        log_dir_q = shlex.quote(log_dir)
        container_pid_path_q = shlex.quote(container_pid_path)
        host_pid_path_q = shlex.quote(host_pid_path)
        docker_filter_q = shlex.quote(f"name={DOCKER_CONTAINER}")
        sig_q = shlex.quote(signal_first.lstrip("-"))
        grace_str = f"{max(0.0, float(grace_sec)):.3f}"
        container_stop_script = (
            f"if [ -s {container_pid_path_q} ]; then "
            f"pid=$(cat {container_pid_path_q}); "
            f"kill -{sig_q} -- -\"$pid\" >/dev/null 2>&1 "
            f"|| kill -{sig_q} \"$pid\" >/dev/null 2>&1 || true; "
            f"sleep {grace_str}; "
            "kill -9 -- -\"$pid\" >/dev/null 2>&1 || kill -9 \"$pid\" >/dev/null 2>&1 || true; "
            f"rm -f {container_pid_path_q}; "
            "fi; "
            f"pkill -f {shlex.quote(pattern)} >/dev/null 2>&1 || true"
        )
        script = (
            f"mkdir -p {log_dir_q}; "
            f"touch {log_path_q} || exit 1; "
            f"printf '\\n[%s] Stopping {label}\\n' "
            f"\"$(date '+%F %T')\" >> {log_path_q}; "
            f"if docker ps --filter {docker_filter_q} -q | grep -q .; then "
            f"docker exec {DOCKER_CONTAINER} bash -lc "
            f"{shlex.quote(container_stop_script)} >> {log_path_q} 2>&1 || true; "
            "else "
            f"echo '{DOCKER_CONTAINER} is not running' >> {log_path_q}; "
            "fi; "
            f"if [ -s {host_pid_path_q} ]; then "
            f"host_pid=$(cat {host_pid_path_q}); "
            "kill \"$host_pid\" >/dev/null 2>&1 || true; "
            f"rm -f {host_pid_path_q}; "
            "fi; "
            f"echo '{label} stopped; log: {log_path}'"
        )
        cmd = self._bash_command(script)
        logger.info("Stopping %s", label)
        self._emit(f"WS-{robot.id}: stopping {label}")

        try:
            result = await (await self._get_connection(robot)).run(cmd, timeout=15)
            return self._format_process_result(robot, result, label, "stop")
        except Exception as e:
            logger.error("%s stop failed: %s", label, e)
            self._emit(f"WS-{robot.id}: {label} stop FAILED — {e}")
            return f"ERROR: {e}"

    async def _check_detached_docker_processes(
        self,
        robots: list[Robot],
        command: str,
        log_suffix: str,
        callback: Callable | None = None,
    ):
        results = {}
        tasks = [
            self._check_detached_docker_process(robot, command, log_suffix)
            for robot in robots
        ]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for robot, out in zip(robots, outputs):
            results[robot.id] = False if isinstance(out, Exception) else bool(out)
        if callback:
            callback(results)
        return results

    async def _check_detached_docker_process(
        self,
        robot: Robot,
        command: str,
        log_suffix: str,
    ) -> bool:
        container_pid_path = f"/tmp/armory_{log_suffix}.pid"
        container_pid_path_q = shlex.quote(container_pid_path)
        docker_filter_q = shlex.quote(f"name={DOCKER_CONTAINER}")
        pattern = self._process_match_pattern(command)
        container_check_script = (
            f"if [ -s {container_pid_path_q} ]; then "
            f"pid=$(cat {container_pid_path_q}); "
            "if kill -0 \"$pid\" >/dev/null 2>&1; then "
            "echo RUNNING; "
            "exit 0; "
            "fi; "
            f"rm -f {container_pid_path_q}; "
            "fi; "
            f"if pgrep -f {shlex.quote(pattern)} >/dev/null 2>&1; then "
            "echo RUNNING; "
            "else "
            "echo STOPPED; "
            "fi"
        )
        script = (
            f"if ! docker ps --filter {docker_filter_q} -q | grep -q .; then "
            "echo STOPPED; "
            "exit 0; "
            "fi; "
            f"docker exec {DOCKER_CONTAINER} bash -lc "
            f"{shlex.quote(container_check_script)}"
        )

        try:
            result = await (await self._get_connection(robot)).run(
                self._bash_command(script),
                timeout=10,
            )
            return result.stdout.strip() == "RUNNING"
        except Exception:
            return False

    # ── SFTP fetch ──────────────────────────────────────────────

    async def _resolve_user_data_host_path(self, robot: Robot) -> str:
        """Return the workstation host path that backs /CS4803ARM_Lab/user_data."""
        cached = self._user_data_host_path.get(robot.id)
        if cached:
            return cached
        path = f"{self._require_piper_root(robot)}/user_data"
        # Confirm it exists rather than trusting config — a stale value here would
        # otherwise surface as an empty fetch that looks like "no episodes".
        conn = await self._get_connection(robot)
        check = await conn.run(f"test -d {shlex.quote(path)}", timeout=10, check=False)
        if check.exit_status != 0:
            raise RuntimeError(
                f"WS-{robot.id}: {path} does not exist — check "
                "ARMORY_PIPER_DOCKER_DIR in .env"
            )
        self._user_data_host_path[robot.id] = path
        return path

    async def _fetch_episode_data(
        self,
        robots: list[Robot],
        local_dir: pathlib.Path,
        remote_subdir: str,
        include_video: bool,
        callback: Callable | None,
    ) -> dict[int, str]:
        local_dir = pathlib.Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            self._fetch_one_robot(robot, local_dir, remote_subdir, include_video)
            for robot in robots
        ]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        results: dict[int, str] = {}
        for robot, out in zip(robots, outputs):
            results[robot.id] = (
                f"ERROR: {out}" if isinstance(out, Exception) else str(out)
            )
        if callback:
            callback(results)
        return results

    async def _fetch_one_robot(
        self,
        robot: Robot,
        local_dir: pathlib.Path,
        remote_subdir: str,
        include_video: bool,
    ) -> str:
        logger = self._loggers[robot.id]
        try:
            await self._resolve_user_data_host_path(robot)
        except Exception as e:
            self._emit(f"WS-{robot.id}: fetch FAILED — {e}")
            return f"ERROR: {e}"

        # remote_root = os.path.join(user_data, remote_subdir.lstrip("/"))
        remote_root = "/home/data_collection/dhe83/armory_episodes"
        # Land each robot's tree under <local_dir>/<robot_name>/.
        per_robot_local = pathlib.Path(local_dir) / robot.name
        per_robot_local.mkdir(parents=True, exist_ok=True)

        conn = await self._get_connection(robot)
        try:
            async with conn.start_sftp_client() as sftp:
                if not await sftp.isdir(remote_root):
                    msg = f"no data at {remote_root}"
                    self._emit(f"WS-{robot.id}: {msg}")
                    return msg

                logger.info("SFTP fetch: %s -> %s", remote_root, per_robot_local)
                self._emit(
                    f"WS-{robot.id}: fetching {remote_root} -> {per_robot_local}"
                )
                # Recursive copy of the entire tree.
                await sftp.mget(
                    remote_root, str(per_robot_local), recurse=True
                )
        except Exception as e:
            logger.error("SFTP fetch failed: %s", e)
            self._emit(f"WS-{robot.id}: fetch FAILED — {e}")
            return f"ERROR: {e}"

        if not include_video:
            removed = 0
            for mp4 in per_robot_local.rglob("out.mp4"):
                try:
                    mp4.unlink()
                    removed += 1
                except OSError:
                    pass
            if removed:
                logger.info("Dropped %d video file(s) (include_video=False)", removed)

        msg = f"fetched to {per_robot_local}"
        self._emit(f"WS-{robot.id}: {msg}")
        return msg

    # ── helpers ──────────────────────────────────────────────────

    def _emit(self, msg: str):
        """Emit a human-readable status event on the controller's logger."""
        self.logger.info(msg)

    def _tunnel_config_error(self) -> str | None:
        node = self.config.tunnel_node.strip()
        if not node or node == "CHANGE_ME":
            return "set tunnel.node in config.yaml first"
        if not self.config.tunnel_server.strip():
            return "set tunnel.server in config.yaml first"
        if self.config.tunnel_port <= 0:
            return "tunnel.port must be positive"
        return None

    def _tunnel_control_path(self) -> str:
        safe_node = "".join(
            ch if ch.isalnum() else "_"
            for ch in self.config.tunnel_node
        ).strip("_")
        return f"/tmp/armory_tunnel_{safe_node}_{self.config.tunnel_port}.sock"

    def _process_log_path(self, robot: Robot, suffix: str) -> str:
        return os.path.join(
            self.config.log_dir,
            f"workstation_{robot.id}-{suffix}.log",
        )

    @staticmethod
    def _process_match_pattern(command: str) -> str:
        """Build a pkill/pgrep pattern that does not match its own shell text."""
        if not command:
            return ""
        return f"[{re.escape(command[0])}]{re.escape(command[1:])}"

    @staticmethod
    def _bash_command(script: str) -> str:
        return f"bash -lc {shlex.quote(script)}"

    def _format_tunnel_result(self, robot: Robot, result, action: str) -> str:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.exit_status != 0:
            msg = stderr or stdout or f"tunnel {action} failed"
            self._emit(f"WS-{robot.id}: tunnel {action} FAILED — {msg[:120]}")
            return f"ERROR: {msg}"

        msg = stdout or f"Tunnel {action} complete"
        self._emit(f"WS-{robot.id}: {msg}")
        return msg

    def _format_process_result(
        self,
        robot: Robot,
        result,
        label: str,
        action: str,
    ) -> str:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.exit_status != 0:
            msg = stderr or stdout or f"{label} {action} failed"
            self._emit(f"WS-{robot.id}: {label} {action} FAILED — {msg[:120]}")
            return f"ERROR: {msg}"

        msg = stdout or f"{label} {action} requested"
        self._emit(f"WS-{robot.id}: {msg}")
        return msg

    @staticmethod
    def _make_logger(name: str, filepath: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.FileHandler(filepath)
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            logger.addHandler(handler)
        return logger

    @property
    def system_logger(self) -> logging.Logger:
        return self._system_logger
