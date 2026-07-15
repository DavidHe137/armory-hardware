"""High-level fleet command dispatcher.

Composes ``FleetController`` primitives into the broadcast operations a
fleet operator actually invokes (enable, disable, goto, boot, shutdown,
client/listener lifecycle, run-trial).
"""

from __future__ import annotations

import asyncio
import pathlib
import shlex
from collections.abc import Callable
from typing import TYPE_CHECKING

from armory_hardware.config import Robot, RobotStatus

if TYPE_CHECKING:
    from armory_hardware.fleet import FleetController


class FleetDispatcher:
    """Dispatches broadcast commands to selected robots."""

    def __init__(self, fleet: FleetController):
        self.fleet = fleet

    # ── broadcast commands ──────────────────────────────────────

    def enable(self, robots: list[Robot], callback: Callable | None = None):
        """Enable: 'p enable' then 'p goto init'."""
        return self.fleet.submit(self._enable(robots, callback))

    def disable(self, robots: list[Robot], callback: Callable | None = None):
        """Disable: 'p goto reset' then 'p disable'."""
        return self.fleet.submit(self._disable(robots, callback))

    def goto_init(self, robots: list[Robot], callback: Callable | None = None):
        """Goto Init: 'p goto init'."""
        return self.fleet.run_on_robots(robots, "p goto init", callback)

    def goto_zero(self, robots: list[Robot], callback: Callable | None = None):
        """Goto Zero: 'p goto zero'."""
        return self.fleet.run_on_robots(robots, "p goto zero", callback)

    def connect_to_server(self, robots: list[Robot], callback: Callable | None = None):
        """Mark selected robots ONLINE.

        Stub today — flips local status only. Replaced in the next phase by a
        real RPC that registers the fleet with the running armory server, so
        scheduling decisions can be tagged with fleet metadata for fairness/
        starvation metrics.
        """
        # TODO(real-fleet-metrics): replace stub with real server registration.
        for r in robots:
            if r.status != RobotStatus.OFFLINE:
                r.status = RobotStatus.ONLINE
        if callback:
            callback({r.id: "Connected (simulated)" for r in robots})

    def boot(self, robots: list[Robot], callback: Callable | None = None):
        """Boot Docker containers on selected robots."""
        return self.fleet.boot_robots(robots, callback)

    def shutdown(self, robots: list[Robot], callback: Callable | None = None):
        """Disable first, kill tunnels, then stop Docker containers."""
        return self.fleet.submit(self._shutdown_safely(robots, callback))

    def start_tunnel(self, robots: list[Robot], callback: Callable | None = None):
        """Start SSH tunnels on selected workstations, outside Docker."""
        return self.fleet.start_tunnels(robots, callback)

    def kill_tunnel(self, robots: list[Robot], callback: Callable | None = None):
        """Kill SSH tunnels on selected workstations, outside Docker."""
        return self.fleet.kill_tunnels(robots, callback)

    def start_webcam(self, robots: list[Robot], callback: Callable | None = None):
        """Start the Logitech webcam RTSP stream on each workstation host."""
        return self.fleet.start_webcam_streams(robots, callback)

    def kill_webcam(self, robots: list[Robot], callback: Callable | None = None):
        """Stop the Logitech webcam RTSP stream on each workstation host."""
        return self.fleet.kill_webcam_streams(robots, callback)

    def start_listener(self, robots: list[Robot], callback: Callable | None = None):
        """Start the local data collection listener inside Docker."""
        return self.fleet.start_data_listeners(robots, callback)

    def start_client(self, robots: list[Robot], callback: Callable | None = None):
        """Start the Piper client node inside Docker."""
        return self.fleet.start_clients(robots, callback)

    def kill_listener(self, robots: list[Robot], callback: Callable | None = None):
        """Stop the local data collection listener inside Docker."""
        return self.fleet.kill_data_listeners(robots, callback)

    def kill_client(self, robots: list[Robot], callback: Callable | None = None):
        """Stop the Piper client node inside Docker."""
        return self.fleet.kill_clients(robots, callback)

    def run_trial(
        self,
        robots: list[Robot],
        duration_sec: float,
        output_dir: pathlib.Path,
        fetch_video: bool = False,
        grace_sec: float = 5.0,
        remote_subdir: str = "armory_episodes",
        callback: Callable | None = None,
        control_hz_overrides: dict[int, int] | None = None,
        prompt_overrides: dict[int, str] | None = None,
        execution_horizon_overrides: dict[int, int] | None = None,
        on_clients_running: Callable[[], None] | None = None,
        on_clients_stopping: Callable[[], None] | None = None,
    ):
        """Run a bounded client trial then fetch each robot's data via SFTP.

        Sequence:
          1. ``start_clients(robots)``
          2. wait ``duration_sec`` seconds
          3. ``kill_clients(robots, grace_sec=grace_sec)`` (SIGINT, then SIGKILL)
          4. brief settle so RealSaver flushes its background writes
          5. ``fetch_episode_data(robots, output_dir, remote_subdir, fetch_video)``

        ``control_hz_overrides`` (ws id → hz), ``prompt_overrides``
        (ws id → prompt string), and ``execution_horizon_overrides``
        (ws id → action-chunk horizon) are forwarded per-robot to
        ``piper_client_armory`` as ``--control-hz <N>``,
        ``--prompt <STRING>``, and ``--execution-horizon <N>`` respectively,
        after a single leading ``--`` separator. Robots not present in a
        given dict use the node's declared default for that parameter.

        ``on_clients_running`` (optional) is invoked once, in the fleet event
        loop thread, immediately after the startup barrier go signal is sent
        and before the trial duration timer starts. Use it to kick off
        trial-scoped side effects (e.g. webcam recording) at the moment the
        robots actually begin executing actions. Exceptions raised inside the
        callback are logged but don't abort the trial.

        ``on_clients_stopping`` (optional) is invoked once, in the fleet event
        loop thread, immediately before ``kill_clients``. Use it to tear down
        the same trial-scoped side effects so they bound to the trial window
        rather than running through the kill grace, settle, and fetch phases.
        Should be non-blocking (or very brief) — it runs inline before the
        SIGINT is sent. Exceptions are logged but don't abort the kill.

        Returns a Future whose result is a summary dict with keys
        ``start``, ``kill``, ``fetch``, and ``output_dir``.
        """
        return self.fleet.submit(
            self._run_trial(
                robots, duration_sec, pathlib.Path(output_dir),
                fetch_video, grace_sec, remote_subdir, callback,
                control_hz_overrides, prompt_overrides, execution_horizon_overrides,
                on_clients_running, on_clients_stopping,
            )
        )

    # ── sequential compound commands ────────────────────────────

    async def _enable(self, robots: list[Robot], callback: Callable | None = None):
        booted = [r for r in robots if r.status != RobotStatus.OFFLINE]
        if not booted:
            if callback:
                callback({r.id: "Skipped — robot offline" for r in robots})
            return

        # Step 1: p enable
        await self.fleet._run_on_robots(booted, "p enable")
        # Step 2: p goto init
        await self.fleet._run_on_robots(booted, "p goto init", callback)

    async def _disable(self, robots: list[Robot], callback: Callable | None = None):
        booted = [r for r in robots if r.status != RobotStatus.OFFLINE]
        if not booted:
            if callback:
                callback({r.id: "Skipped — robot offline" for r in robots})
            return

        # Step 1: p goto reset
        await self.fleet._run_on_robots(booted, "p goto reset")
        # Step 2: p disable
        await self.fleet._run_on_robots(booted, "p disable", callback)

    async def _shutdown_safely(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        active = [r for r in robots if r.status != RobotStatus.OFFLINE]
        if not active:
            if callback:
                callback({r.id: "Skipped — robot offline" for r in robots})
            return

        # Safeguard: never remove Docker until the robot command stack is disabled.
        reset_results = await self.fleet._run_on_robots(active, "p goto reset")
        disable_results = await self.fleet._run_on_robots(active, "p disable")
        disabled = [
            robot
            for robot in active
            if self._result_ok(reset_results.get(robot.id))
            and self._result_ok(disable_results.get(robot.id))
        ]
        tunnel_results = await self.fleet._kill_tunnels(disabled)
        shutdown_results = await self.fleet._shutdown_robots(disabled)

        results = {}
        for robot in active:
            if robot not in disabled:
                results[robot.id] = (
                    f"Reset: {reset_results.get(robot.id, 'unknown')}; "
                    f"Disable: {disable_results.get(robot.id, 'unknown')}; "
                    "Shutdown skipped — disable safeguard failed"
                )
                continue

            results[robot.id] = (
                f"Reset: {reset_results.get(robot.id, 'unknown')}; "
                f"Disable: {disable_results.get(robot.id, 'unknown')}; "
                f"Tunnel: {tunnel_results.get(robot.id, 'unknown')}; "
                f"Shutdown: {shutdown_results.get(robot.id, 'unknown')}"
            )
        if callback:
            callback(results)
        return results

    async def _run_trial(
        self,
        robots: list[Robot],
        duration_sec: float,
        output_dir: pathlib.Path,
        fetch_video: bool,
        grace_sec: float,
        remote_subdir: str,
        callback: Callable | None,
        control_hz_overrides: dict[int, int] | None = None,
        prompt_overrides: dict[int, str] | None = None,
        execution_horizon_overrides: dict[int, int] | None = None,
        on_clients_running: Callable[[], None] | None = None,
        on_clients_stopping: Callable[[], None] | None = None,
    ):
        log = self.fleet.logger
        n = len(robots)

        # Use --control-hz / --prompt / --barrier (argparse layer in
        # client_node_armory) rather than --ros-args -p key:=value: avoids both
        # the int/float type mismatch against declare_parameter("control_hz",
        # 20.0) and any precedence confusion when the node also builds
        # parameter_overrides. The leading `--` is the conventional "end of
        # ros2 args" separator; `ros2 run` strips it before invoking the entry
        # point, so argparse never sees it. ``--barrier`` is added for every
        # target so all clients block on /tmp/armory_go.flag and don't begin
        # publishing observations until the dispatcher signals go.
        target_ids = {r.id for r in robots}
        extra_args_per_robot = self._build_extra_args(
            control_hz_overrides,
            prompt_overrides,
            execution_horizon_overrides=execution_horizon_overrides,
            barrier_robot_ids=target_ids,
        )
        if extra_args_per_robot:
            applied = {r.id: extra_args_per_robot[r.id] for r in robots
                       if r.id in extra_args_per_robot}
            if applied:
                log.info(
                    "trial: per-robot launch overrides applied to %d robot(s): %s",
                    len(applied),
                    applied,
                )
            else:
                log.warning(
                    "trial: overrides were provided but no target robot ids "
                    "matched (override ids=%s, target ids=%s)",
                    sorted(extra_args_per_robot),
                    [r.id for r in robots],
                )
        else:
            log.info("trial: no per-robot launch overrides (using node defaults)")

        # Defensive: wipe any stale barrier flags from a prior crashed trial
        # before launching clients, so dispatcher and clients agree on a clean
        # starting state.
        await self.fleet._clear_barrier_flags(robots)

        log.info(
            f"trial: start_clients on {n} robot(s); will run for {duration_sec:.1f}s"
        )
        start_results = await self.fleet._start_clients(
            robots, callback=None, extra_args_per_robot=extra_args_per_robot,
        )

        # Rendezvous: every client should reach the --barrier and touch
        # /tmp/armory_ready.flag; we wait for all of them, then signal go in
        # parallel so they all start spinning on the same wall-clock tick.
        # Laggards that miss the window get logged but don't block the trial.
        ready_timeout = 60.0
        log.info(
            "trial: waiting up to %.0fs for clients to reach startup barrier",
            ready_timeout,
        )
        ready = await self.fleet._await_clients_ready(robots, timeout_sec=ready_timeout)
        not_ready = [r for r in robots if not ready.get(r.id, False)]
        if not_ready:
            log.warning(
                "trial: %d/%d robot(s) did not signal ready: %s — "
                "they'll start whenever their own barrier timeout expires",
                len(not_ready), len(robots),
                [f"WS-{r.id}" for r in not_ready],
            )
        else:
            log.info("trial: all %d robot(s) ready", len(robots))
        await self.fleet._signal_clients_go(robots)
        log.info("trial: go signal sent; duration timer starts now")

        # Fire the post-barrier hook before the duration sleep so trial-scoped
        # side effects (e.g. webcam recording) line up with the moment robots
        # actually begin executing actions, not the earlier client-spawn moment.
        if on_clients_running is not None:
            try:
                on_clients_running()
            except Exception as e:
                log.warning("trial: on_clients_running callback raised: %s", e)

        await asyncio.sleep(max(0.0, float(duration_sec)))

        # Tear down trial-scoped side effects (e.g. webcam recording) before
        # kill so they bound to the trial window rather than running through
        # the kill grace, settle, and fetch phases that follow.
        if on_clients_stopping is not None:
            try:
                on_clients_stopping()
            except Exception as e:
                log.warning("trial: on_clients_stopping callback raised: %s", e)

        log.info(f"trial: killing clients (SIGINT, grace={grace_sec:.1f}s)")
        kill_results = await self.fleet._kill_clients(
            robots, callback=None, grace_sec=grace_sec
        )

        # Settle so RealSaver's executor finishes flushing after SIGINT.
        await asyncio.sleep(min(grace_sec, 2.0))

        log.info(f"trial: fetching episode data to {output_dir}")
        fetch_results = await self.fleet._fetch_episode_data(
            robots,
            local_dir=pathlib.Path(output_dir),
            remote_subdir=remote_subdir,
            include_video=fetch_video,
            callback=None,
        )

        summary = {
            "start": start_results,
            "kill": kill_results,
            "fetch": fetch_results,
            "output_dir": str(output_dir),
        }
        log.info(f"trial: complete — {output_dir}")
        if callback:
            callback(summary)
        return summary

    @staticmethod
    def _result_ok(result) -> bool:
        return result is not None and not str(result).startswith("ERROR")

    @staticmethod
    def _build_extra_args(
        control_hz_overrides: dict[int, int] | None,
        prompt_overrides: dict[int, str] | None,
        execution_horizon_overrides: dict[int, int] | None = None,
        barrier_robot_ids: set[int] | None = None,
    ) -> dict[int, str] | None:
        """Merge per-robot overrides into a single ``--<flag> <value> …`` suffix.

        Returns ``{robot_id: " -- --control-hz X --prompt 'STRING'
        --execution-horizon N --barrier "}`` for each robot that has at least
        one override (or has the barrier enabled); ``None`` if no robot has
        anything to add. The prompt is shell-quoted because it may contain
        spaces, and the whole string is interpolated verbatim into a bash
        command in fleet.py.
        """
        rids: set[int] = set(control_hz_overrides or {})
        rids |= set(prompt_overrides or {})
        rids |= set(execution_horizon_overrides or {})
        rids |= barrier_robot_ids or set()
        if not rids:
            return None
        out: dict[int, str] = {}
        for rid in rids:
            parts = ["--"]
            if control_hz_overrides and rid in control_hz_overrides:
                parts.append(f"--control-hz {float(control_hz_overrides[rid])}")
            if prompt_overrides and rid in prompt_overrides:
                parts.append(f"--prompt {shlex.quote(prompt_overrides[rid])}")
            if execution_horizon_overrides and rid in execution_horizon_overrides:
                parts.append(
                    f"--execution-horizon {int(execution_horizon_overrides[rid])}"
                )
            if barrier_robot_ids and rid in barrier_robot_ids:
                parts.append("--barrier")
            out[rid] = " ".join(parts)
        return out
