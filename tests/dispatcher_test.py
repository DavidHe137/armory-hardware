"""Unit tests for FleetDispatcher (no network)."""

from __future__ import annotations

import asyncio

import pytest

from armory_hardware import FleetDispatcher, Robot, RobotStatus


class _StubController:
    """Minimal in-memory stand-in for FleetController."""

    def __init__(self):
        self.calls: list[tuple] = []

    # mirror FleetController signatures
    def run_on_robots(self, robots, command, callback=None):
        self.calls.append(("run_on_robots", list(robots), command, callback))
        return "future:run_on_robots"

    def boot_robots(self, robots, callback=None):
        self.calls.append(("boot_robots", list(robots), callback))
        return "future:boot_robots"

    def build_workspace(self, robots, callback=None):
        self.calls.append(("build_workspace", list(robots), callback))
        return "future:build_workspace"

    def start_tunnels(self, robots, callback=None):
        self.calls.append(("start_tunnels", list(robots), callback))
        return "future:start_tunnels"

    def kill_tunnels(self, robots, callback=None):
        self.calls.append(("kill_tunnels", list(robots), callback))
        return "future:kill_tunnels"

    def start_data_listeners(self, robots, callback=None):
        self.calls.append(("start_data_listeners", list(robots), callback))
        return "future:start_data_listeners"

    def start_clients(self, robots, callback=None):
        self.calls.append(("start_clients", list(robots), callback))
        return "future:start_clients"

    def kill_data_listeners(self, robots, callback=None):
        self.calls.append(("kill_data_listeners", list(robots), callback))
        return "future:kill_data_listeners"

    def kill_clients(self, robots, callback=None):
        self.calls.append(("kill_clients", list(robots), callback))
        return "future:kill_clients"

    def kill_all(self, robots, callback=None):
        self.calls.append(("kill_all", list(robots), callback))
        return "future:kill_all"

    def submit(self, coro):
        # Tests that exercise compound commands invoke the underlying
        # coroutines directly via asyncio.run, so submit is unused here.
        coro.close()
        return None


class _AsyncStubController(_StubController):
    """Stub exposing the internal async primitives compound commands use.

    ``step_results`` maps a step name to the per-robot results it should
    return; steps not listed return "ok" for every robot.
    """

    def __init__(self, step_results: dict[str, dict[int, str]] | None = None):
        super().__init__()
        self.step_results = step_results or {}

    async def _step(self, name, robots, *extra):
        self.calls.append((name, list(robots), *extra))
        default = self.step_results.get(name, {})
        return {r.id: default.get(r.id, "ok") for r in robots}

    async def _kill_clients(self, robots, callback=None, grace_sec=5.0):
        return await self._step("_kill_clients", robots)

    async def _kill_data_listeners(self, robots, callback=None):
        return await self._step("_kill_data_listeners", robots)

    async def _kill_webcam_streams(self, robots, callback=None):
        return await self._step("_kill_webcam_streams", robots)

    async def _run_on_robots(self, robots, command, callback=None):
        return await self._step("_run_on_robots", robots, command)

    async def _kill_tunnels(self, robots, callback=None):
        return await self._step("_kill_tunnels", robots)

    async def _shutdown_robots(self, robots, callback=None):
        return await self._step("_shutdown_robots", robots)


def _robot(rid: int, status: RobotStatus = RobotStatus.OFFLINE) -> Robot:
    return Robot(id=rid, name=f"R{rid}", ip=f"10.0.0.{rid}", status=status)


# ── pure delegation ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "method, command",
    [
        ("goto_init", "p goto init"),
        ("goto_zero", "p goto zero"),
    ],
)
def test_goto_methods_forward_to_run_on_robots(method, command):
    fleet = _StubController()
    dispatcher = FleetDispatcher(fleet)
    robots = [_robot(1)]

    getattr(dispatcher, method)(robots)

    assert fleet.calls == [("run_on_robots", robots, command, None)]


@pytest.mark.parametrize(
    "method, expected_call",
    [
        ("boot", "boot_robots"),
        ("build_workspace", "build_workspace"),
        ("start_tunnel", "start_tunnels"),
        ("kill_tunnel", "kill_tunnels"),
        ("start_listener", "start_data_listeners"),
        ("start_client", "start_clients"),
        ("kill_listener", "kill_data_listeners"),
        ("kill_client", "kill_clients"),
    ],
)
def test_simple_dispatch_methods_route_to_controller(method, expected_call):
    fleet = _StubController()
    dispatcher = FleetDispatcher(fleet)
    robots = [_robot(1), _robot(2)]

    getattr(dispatcher, method)(robots)

    assert fleet.calls[0][0] == expected_call
    assert fleet.calls[0][1] == robots


# ── connect_to_server (pure local state) ───────────────────────────


def test_connect_to_server_marks_non_offline_robots_online():
    fleet = _StubController()
    dispatcher = FleetDispatcher(fleet)
    online = _robot(1, status=RobotStatus.ONLINE)
    booted = _robot(2, status=RobotStatus.BOOTED)
    offline = _robot(3, status=RobotStatus.OFFLINE)
    captured = []

    dispatcher.connect_to_server([online, booted, offline], callback=captured.append)

    assert online.status is RobotStatus.ONLINE
    assert booted.status is RobotStatus.ONLINE
    assert offline.status is RobotStatus.OFFLINE
    assert captured == [{1: "Connected (simulated)", 2: "Connected (simulated)", 3: "Connected (simulated)"}]
    assert fleet.calls == []


# ── compound async commands (skip-when-offline branch) ────────────


def test_enable_skips_when_all_robots_offline():
    fleet = _StubController()
    dispatcher = FleetDispatcher(fleet)
    offline = [_robot(1), _robot(2)]
    captured = []

    asyncio.run(dispatcher._enable(offline, callback=captured.append))

    assert captured == [{1: "Skipped — robot offline", 2: "Skipped — robot offline"}]
    assert fleet.calls == []


def test_disable_skips_when_all_robots_offline():
    fleet = _StubController()
    dispatcher = FleetDispatcher(fleet)
    offline = [_robot(1)]
    captured = []

    asyncio.run(dispatcher._disable(offline, callback=captured.append))

    assert captured == [{1: "Skipped — robot offline"}]
    assert fleet.calls == []


def test_shutdown_safely_skips_when_all_robots_offline():
    fleet = _StubController()
    dispatcher = FleetDispatcher(fleet)
    offline = [_robot(1)]
    captured = []

    asyncio.run(dispatcher._shutdown_safely(offline, callback=captured.append))

    assert captured == [{1: "Skipped — robot offline"}]
    assert fleet.calls == []


def test_kill_all_dispatch_routes_to_controller():
    fleet = _StubController()
    dispatcher = FleetDispatcher(fleet)
    robots = [_robot(1)]

    dispatcher.kill_all(robots)

    # kill_all submits its own coroutine rather than delegating to a
    # controller method, so the stub records nothing — but the call must not
    # raise and must go through submit (which closes the coroutine).
    assert fleet.calls == []


def test_kill_all_runs_every_step_even_when_offline_and_failing():
    fleet = _AsyncStubController(
        step_results={"_kill_clients": {1: "ERROR: container gone"}}
    )
    dispatcher = FleetDispatcher(fleet)
    robots = [_robot(1, status=RobotStatus.OFFLINE)]
    captured = []

    asyncio.run(dispatcher._kill_all(robots, callback=captured.append))

    steps = [call[0] for call in fleet.calls]
    assert steps == [
        "_kill_clients",
        "_kill_data_listeners",
        "_kill_webcam_streams",
        "_run_on_robots",
        "_run_on_robots",
        "_kill_tunnels",
        "_shutdown_robots",
    ]
    run_commands = [call[2] for call in fleet.calls if call[0] == "_run_on_robots"]
    assert run_commands == ["p goto reset", "p disable"]

    assert len(captured) == 1
    summary = captured[0][1]
    assert "Client: ERROR: container gone" in summary
    assert "Shutdown: ok" in summary


def test_result_ok_classifies_outputs():
    assert FleetDispatcher._result_ok("Boot successful")
    assert FleetDispatcher._result_ok("(no output)")
    assert not FleetDispatcher._result_ok(None)
    assert not FleetDispatcher._result_ok("ERROR: timed out")
