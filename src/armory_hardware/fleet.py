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
        self._connections: dict[int, asyncssh.SSHClientConnection] = {}
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
        async with self._lock:
            conn = self._connections.get(robot.id)
            if conn is not None:
                # Check if still alive
                try:
                    await conn.run("true", timeout=3)
                    return conn
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    del self._connections[robot.id]

            conn = await asyncssh.connect(
                robot.ip,
                username=self.config.ssh_user,
                known_hosts=None,
                connect_timeout=8,
            )
            self._connections[robot.id] = conn
            return conn

    async def _close_all(self):
        async with self._lock:
            for _rid, conn in self._connections.items():
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()

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
            async with self._lock:
                self._connections.pop(robot.id, None)
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
            cmd = f"docker exec {DOCKER_CONTAINER} bash -ic 'source ~/.bashrc && {command}'"
            logger.info("Executing: %s", cmd)
            self._emit(f"WS-{robot.id}: running '{command}'")
            result = await conn.run(cmd, timeout=30)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if stdout:
                logger.info("stdout: %s", stdout)
                self._emit(f"WS-{robot.id}: {stdout[:120]}")
            if stderr:
                logger.warning("stderr: %s", stderr)
                self._emit(f"WS-{robot.id} err: {stderr[:120]}")
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
        if callback:
            callback(results)
        return results

    async def _boot_single(self, robot: Robot) -> str:
        logger = self._loggers[robot.id]
        try:
            conn = await self._get_connection(robot)

            # Check if already running
            check = await conn.run(
                f"docker ps --filter name={DOCKER_CONTAINER} -q",
                timeout=10,
            )
            if check.stdout.strip():
                msg = "Container already running — skipping boot"
                logger.info(msg)
                self._emit(f"WS-{robot.id}: {msg}")
                robot.status = RobotStatus.BOOTED
                return msg

            # Remove any exited container with the same name
            await conn.run(
                f"docker rm -f {DOCKER_CONTAINER} 2>/dev/null || true",
                timeout=10,
            )

            logger.info("Starting Docker container in detached mode")
            self._emit(f"WS-{robot.id}: starting Docker container...")

            # piper_start uses `docker run -it` which blocks and needs a TTY.
            # Instead, run in detached mode (-d) with --entrypoint overridden
            # to `sleep` so the container stays alive, then launch the real
            # entrypoint via docker exec.
            boot_cmd = (
                "bash -lc 'source ~/.bashrc && docker run -d"
                " --privileged --net=host --runtime=nvidia --gpus all"
                f" --name {DOCKER_CONTAINER}"
                " --entrypoint sleep"
                " -e NVIDIA_DRIVER_CAPABILITIES=all"
                ' -e DISPLAY=$DISPLAY'
                " -v /tmp/.X11-unix/:/tmp/.X11-unix/:rw"
                ' -v $PIPER_DOCKER_DIR/../user_data/:/CS4803ARM_Lab/user_data/'
                ' -v $PIPER_DOCKER_DIR/../assets/:/CS4803ARM_Lab/assets/'
                ' -v $PIPER_DOCKER_DIR/../user_data/piper_ros/:/piper_ros/'
                " -v /home/data_collection/dhe83/:/datasets/"
                " -v /dev:/dev"
                " -w /CS4803ARM_Lab/user_data/data_collection"
                " firefall/cluster_piper_env:v2"
                " infinity'"
            )
            result = await conn.run(boot_cmd, timeout=30)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if stdout:
                logger.info("docker run output: %s", stdout)
                self._emit(f"WS-{robot.id}: {stdout[:80]}")
            if stderr:
                logger.warning("docker run stderr: %s", stderr)
                self._emit(f"WS-{robot.id}: {stderr[:80]}")

            # Verify container came up
            await asyncio.sleep(2)
            check = await conn.run(
                f"docker ps --filter name={DOCKER_CONTAINER} -q",
                timeout=10,
            )
            if not check.stdout.strip():
                logger.warning("Container failed to start")
                self._emit(f"WS-{robot.id}: container failed to start")
                return "Boot failed — container not detected after docker run"

            # Run the entrypoint inside the container
            self._emit(f"WS-{robot.id}: container up, running entrypoint...")
            entrypoint = "/CS4803ARM_Lab/user_data/piper_ros/entrypoint.sh"
            await conn.run(
                f"docker exec -d {DOCKER_CONTAINER} bash -c '{entrypoint}'",
                timeout=15,
            )

            logger.info("Container started successfully")
            self._emit(f"WS-{robot.id}: boot successful")
            robot.status = RobotStatus.BOOTED
            return "Boot successful"
        except Exception as e:
            logger.error("Boot failed: %s", e)
            self._emit(f"WS-{robot.id}: boot FAILED — {e}")
            robot.status = RobotStatus.OFFLINE
            return f"ERROR: {e}"

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
    ):
        results = {}
        connections = []

        # Warm connections first, then launch the long-running commands together.
        for robot in robots:
            try:
                connections.append((robot, await self._get_connection(robot)))
            except Exception as e:
                self._emit(f"WS-{robot.id}: {label} launch FAILED — {e}")
                results[robot.id] = f"ERROR: {e}"

        tasks = [
            self._start_detached_docker_process(
                robot,
                conn,
                label,
                self._command_for_robot(command, robot, extra_args_per_robot),
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
    ) -> str:
        """Append per-robot args to the base command. Kill matching uses the
        base command as a substring pattern, so any suffix here is invisible
        to ``_kill_detached_docker_process`` — start/kill stay symmetric.
        """
        if not extra_args_per_robot:
            return base_command
        extra = extra_args_per_robot.get(robot.id)
        if not extra:
            return base_command
        resolved = f"{base_command} {extra}".rstrip()
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
        conn = await self._get_connection(robot)
        # PIPER_DOCKER_DIR is set in the workstation's ~/.bashrc; resolve via a
        # login shell so the env var is available, then normalize the path.
        result = await conn.run(
            "bash -lc 'readlink -f \"$PIPER_DOCKER_DIR/../user_data\"'",
            timeout=10,
        )
        path = result.stdout.strip()
        if not path:
            raise RuntimeError(
                f"WS-{robot.id}: could not resolve $PIPER_DOCKER_DIR/../user_data; "
                "is PIPER_DOCKER_DIR set in ~/.bashrc?"
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
