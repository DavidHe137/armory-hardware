"""The arm-motion commands must never land on an arm something else is driving.

Enable, Disable, Goto Init and Goto Zero all end up issuing ``p goto ...``. If a
client or a follower is still running, it is publishing joint targets too, and
the arm gets commanded from both sides at once. The dashboard's cached
client-running set is no defence: it refreshes on a schedule and never knew
about the follower at all.

So these pin the guard end to end -- the state is re-read from the workstations
before anything reaches an arm, a blocked command stops dead, the stop-first
path re-checks instead of trusting the kill, and a stop that did not take
refuses rather than proceeding.
"""

from __future__ import annotations

import asyncio
import logging
import types
from concurrent.futures import Future

from armory_hardware import FleetDispatcher, Robot, RobotStatus
from armory_tui.dashboard import Dashboard


def _robot(rid: int, status: RobotStatus = RobotStatus.BOOTED) -> Robot:
    return Robot(id=rid, name=f"R{rid}", ip=f"10.0.0.{rid}", status=status)


# ── dispatcher ─────────────────────────────────────────────────────


class _ProcessStubController:
    """Async primitives that the two new compound commands sit on."""

    def __init__(self, clients=None, listeners=None):
        self.calls: list[str] = []
        self._clients = clients or {}
        self._listeners = listeners or {}

    async def _check_clients(self, robots, callback=None):
        self.calls.append("_check_clients")
        return {r.id: self._clients.get(r.id, False) for r in robots}

    async def _check_data_listeners(self, robots, callback=None):
        self.calls.append("_check_data_listeners")
        return {r.id: self._listeners.get(r.id, False) for r in robots}

    async def _kill_clients(self, robots, callback=None, grace_sec=5.0):
        self.calls.append("_kill_clients")
        return {r.id: "client stopped" for r in robots}

    async def _kill_data_listeners(self, robots, callback=None):
        self.calls.append("_kill_data_listeners")
        return {r.id: "listener stopped" for r in robots}


def test_check_runtime_processes_reports_both_processes_per_robot():
    fleet = _ProcessStubController(clients={1: True}, listeners={2: True})
    dispatcher = FleetDispatcher(fleet)
    robots = [_robot(1), _robot(2), _robot(3)]

    results = asyncio.run(dispatcher._check_runtime_processes(robots))

    assert results == {
        1: {"client": True, "listener": False},
        2: {"client": False, "listener": True},
        3: {"client": False, "listener": False},
    }


def test_stop_runtime_processes_kills_the_client_before_the_follower():
    """The client is what drives the arm, and its SIGINT lets RealSaver flush."""
    fleet = _ProcessStubController()
    dispatcher = FleetDispatcher(fleet)

    results = asyncio.run(dispatcher._stop_runtime_processes([_robot(1)]))

    assert fleet.calls == ["_kill_clients", "_kill_data_listeners"]
    assert "client stopped" in results[1]
    assert "listener stopped" in results[1]


# ── dashboard guard ────────────────────────────────────────────────


class _FakeConfig:
    def __init__(self, robots):
        self.robots = robots


class _FakeFleet:
    system_logger = logging.getLogger("motion_guard_test_null")

    def check_clients(self, robots, callback=None):
        return None


class _FakeDispatcher:
    """Answers the preflight from a scripted queue of per-robot states.

    With ``hold=True`` the preflight callback is parked instead of fired, so a
    test can inspect the dashboard while the check is still in flight.
    """

    def __init__(self, checks, hold=False):
        self._checks = list(checks)
        self._hold = hold
        self._parked = None
        self.check_calls = 0
        self.stopped: list[list[int]] = []

    def check_runtime_processes(self, robots, callback=None):
        self.check_calls += 1
        state = self._checks.pop(0) if self._checks else {}
        results = {r.id: state.get(r.id, {}) for r in robots}
        if self._hold:
            self._parked = lambda: callback and callback(results)
            return Future()
        if callback:
            callback(results)
        return None

    def release(self):
        """Fire a parked preflight callback, as a late fleet result would."""
        parked, self._parked = self._parked, None
        if parked:
            parked()

    def stop_runtime_processes(self, robots, callback=None, grace_sec=5.0):
        self.stopped.append([r.id for r in robots])
        if callback:
            callback({r.id: "stopped" for r in robots})
        return None


def _dashboard(checks, menu_choice="cancel", hold=False):
    robots = [_robot(1), _robot(2)]
    dispatcher = _FakeDispatcher(checks, hold=hold)
    dash = Dashboard(_FakeConfig(robots), _FakeFleet(), dispatcher)
    dash._confirm = lambda _message: True
    dash._motion_guard_menu = lambda _name, _detail: menu_choice
    dash._set_notice = lambda *_a, **_k: None

    executed: list[list[int]] = []

    def action(targets, callback=None):
        executed.append([r.id for r in targets])
        if callback:
            callback({r.id: "ok" for r in targets})
        return None

    return dash, dispatcher, action, executed


def _run(dash):
    """Drain the queue the fleet callbacks hand back to the curses thread."""
    for _ in range(6):
        if not dash._deferred:
            break
        dash._drain_deferred()


def _logged(dash, needle: str) -> bool:
    return any(needle in line for line in dash.log_lines)


def test_clean_preflight_runs_the_command():
    dash, _dispatcher, action, executed = _dashboard([{}])

    dash._do_guarded_motion("Goto Init", action)
    _run(dash)

    assert executed == [[1, 2]]
    assert dash._busy is False


def test_running_client_blocks_the_command():
    """The whole point: nothing reaches the arms while a client owns them."""
    dash, _dispatcher, action, executed = _dashboard(
        [{1: {"client": True, "listener": False}}],
        menu_choice="cancel",
    )

    dash._do_guarded_motion("Disable", action)
    _run(dash)

    assert executed == []
    assert _logged(dash, "nothing was sent to the arms")


def test_running_follower_alone_also_blocks():
    """The follower drives the arm too, and the old cache never tracked it."""
    dash, _dispatcher, action, executed = _dashboard(
        [{2: {"client": False, "listener": True}}],
        menu_choice="cancel",
    )

    dash._do_guarded_motion("Goto Init", action)
    _run(dash)

    assert executed == []
    assert _logged(dash, "WS-2 (follower)")


def test_stop_first_kills_only_the_busy_robots_then_runs_on_all_targets():
    dash, dispatcher, action, executed = _dashboard(
        [{1: {"client": True, "listener": False}}, {}],
        menu_choice="stop",
    )

    dash._do_guarded_motion("Disable", action)
    _run(dash)

    assert dispatcher.stopped == [[1]]
    # Re-checked after the kill rather than assumed clean.
    assert dispatcher.check_calls == 2
    assert executed == [[1, 2]]


def test_stop_that_does_not_take_refuses_instead_of_looping():
    dash, dispatcher, action, executed = _dashboard(
        [
            {1: {"client": True, "listener": False}},
            {1: {"client": True, "listener": False}},
        ],
        menu_choice="stop",
    )

    dash._do_guarded_motion("Disable", action)
    _run(dash)

    assert executed == []
    assert dispatcher.stopped == [[1]]  # offered once, not on a loop
    assert _logged(dash, "still running after the stop")


def test_aborting_the_preflight_never_runs_the_command():
    """Aborting 'Disable preflight' must not go on to run Disable.

    The preflight's own late result is the danger: it is what schedules the
    motion, so a result landing after the operator abandoned the command would
    drive the arms with no one expecting it.
    """
    dash, dispatcher, action, executed = _dashboard([{}], hold=True)

    dash._do_guarded_motion("Disable", action)
    assert dash._busy is True

    dash._abort_operation()
    dispatcher.release()  # the abandoned check reports back anyway
    _run(dash)

    assert executed == []
    assert dash._busy is False


def test_offline_targets_never_reach_the_preflight():
    dash = Dashboard(
        _FakeConfig([_robot(1, RobotStatus.OFFLINE)]),
        _FakeFleet(),
        types.SimpleNamespace(),
    )
    dash._set_notice = lambda *_a, **_k: None

    dash._do_guarded_motion("Disable", lambda *_a, **_k: None)

    assert _logged(dash, "No eligible robot(s)")
