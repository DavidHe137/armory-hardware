"""Unit tests for the dashboard's in-flight command tracking (no curses).

The dashboard locks its controls while a fleet command runs. These tests pin
the property that makes that lock safe: it is always released. Before, ``_busy``
was cleared only by the command's own callback, so anything that stopped the
callback from firing — an exception escaping the fleet coroutine, a submit that
never scheduled — left the console permanently unusable, including for the
emergency client stop.
"""

from __future__ import annotations

import logging
import types
from concurrent.futures import Future

import pytest

from armory_hardware import Robot, RobotStatus
from armory_tui.dashboard import Dashboard


class _FakeConfig:
    def __init__(self):
        self.robots = [
            Robot(id=1, name="R1", ip="10.0.0.1", status=RobotStatus.BOOTED),
            Robot(id=2, name="R2", ip="10.0.0.2", status=RobotStatus.BOOTED),
        ]


class _FakeFleet:
    system_logger = logging.getLogger("dashboard_test_null")

    def check_clients(self, robots, callback=None):
        return None


@pytest.fixture
def dash():
    d = Dashboard(_FakeConfig(), _FakeFleet(), types.SimpleNamespace())
    d._confirm = lambda _message: True
    return d


def _logged(dashboard, needle: str) -> bool:
    return any(needle in line for line in dashboard.log_lines)


def test_launch_exception_unlocks(dash):
    """A submit that raises must not leave the controls locked."""

    def boom(_done):
        raise RuntimeError("event loop is not running")

    dash._start_operation("Boot", boom)

    assert dash._busy is False
    assert _logged(dash, "'Boot' failed")


def test_escaping_coroutine_exception_unlocks(dash):
    """The real wedge: the fleet task dies, so its callback never fires."""
    future = Future()
    dash._start_operation("Boot", lambda _done: future)
    assert dash._busy is True

    future.set_exception(RuntimeError("SSH probe timed out"))

    assert dash._busy is False
    assert _logged(dash, "SSH probe timed out")


def test_coroutine_returning_without_callback_unlocks(dash):
    future = Future()
    dash._start_operation("Refresh", lambda _done: future)

    future.set_result(None)

    assert dash._busy is False


def test_callback_and_backstop_do_not_double_report(dash):
    """Normal path: the future settling after the callback must be a no-op."""
    future = Future()
    seen = []
    captured = {}

    def launch(done):
        captured["done"] = done
        return future

    dash._start_operation("Enable", launch, on_results=seen.append)
    captured["done"]({1: "ok"})
    future.set_result({1: "ok"})

    assert dash._busy is False
    assert seen == [{1: "ok"}]
    assert _logged(dash, "WS-1: ok")


def test_abort_unlocks_and_cancels_a_hung_command(dash):
    future = Future()
    dash._start_operation("Build Workspace", lambda _done: future)

    dash._abort_operation()

    assert dash._busy is False
    assert future.cancelled()
    assert _logged(dash, "Aborted 'Build Workspace'")


def test_abandoned_callback_cannot_unlock_a_later_command(dash):
    """A late result from an aborted command must not free a newer one."""
    captured = {}

    def launch(done):
        captured["done"] = done
        return Future()

    dash._start_operation("Build Workspace", launch)
    dash._abort_operation()
    dash._start_operation("Boot", lambda _done: Future())

    captured["done"]({1: "late result"})

    assert dash._busy is True
    assert dash._op_name == "Boot"


def test_emergency_stop_keeps_the_lock_until_it_finishes(dash):
    """The older command landing first must not unlock mid-emergency-stop."""
    captured = {}

    def launch_boot(done):
        captured["boot"] = done
        return Future()

    def launch_estop(done):
        captured["estop"] = done
        return Future()

    dash._start_operation("Boot", launch_boot)
    dash._start_operation("Emergency Stop", launch_estop)

    captured["boot"]({1: "boot done"})
    assert dash._busy is True

    captured["estop"]({1: "killed"})
    assert dash._busy is False


def test_abort_with_nothing_running_is_a_no_op(dash):
    dash._abort_operation()

    assert dash._busy is False
    assert _logged(dash, "nothing to abort")


def test_busy_status_line_advertises_the_abort_key(dash):
    dash._start_operation("Boot", lambda _done: Future())

    line = dash._busy_status_line()

    assert "Boot" in line
    assert "[A]" in line
