"""Unit tests for the shared /piper_ros workspace build (no network, no SSH).

The workspace is one NFS checkout mounted by every workstation, so these tests
pin the two properties that keeps safe: a present executable is never rebuilt,
and a missing one is built exactly once, on one machine.
"""

from __future__ import annotations

import asyncio
import re
from textwrap import dedent

import pytest

from armory_hardware import FleetConfig, FleetController, Robot, RobotStatus
from armory_hardware import fleet as fleet_mod


class _Result:
    def __init__(self, stdout: str = "", stderr: str = "", exit_status: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_status = exit_status


class _FakeConnection:
    """Records commands and answers the two probes the build path issues."""

    def __init__(self, executable_present: bool, build_status: str = "0"):
        self.executable_present = executable_present
        self.build_status = build_status
        self.commands: list[str] = []

    async def run(self, command, timeout=None, check=True):
        self.commands.append(command)
        if f"test -x {fleet_mod.PIPER_CLIENT_EXECUTABLE_PATH}" in command:
            return _Result(exit_status=0 if self.executable_present else 1)
        if fleet_mod.PIPER_BUILD_STATUS_PATH in command and command.startswith("cat "):
            return _Result(stdout=self.build_status)
        if "colcon build" in command:
            # Launching the detached build is what makes it "installed".
            self.executable_present = self.build_status == "0"
        return _Result()

    @property
    def build_launched(self) -> bool:
        return any("colcon build" in c for c in self.commands)


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.delenv("ARMORY_AUTO_BUILD_WORKSPACE", raising=False)
    monkeypatch.setattr(fleet_mod, "BUILD_POLL_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(fleet_mod, "BUILD_PROGRESS_INTERVAL_SEC", 0.01)

    cfg_path = tmp_path / "fleet.yaml"
    cfg_path.write_text(
        dedent(
            f"""
            workstations:
              - id: 11
              - id: 12
            logging:
              log_dir: {tmp_path / "logs"}
            """
        ).lstrip()
    )
    return FleetController(FleetConfig(str(cfg_path)))


def _robot(rid: int = 11) -> Robot:
    return Robot(id=rid, name=f"R{rid}", ip=f"10.0.0.{rid}", status=RobotStatus.BOOTED)


def _attach(controller, conn):
    async def _get_connection(_robot):
        return conn

    controller._get_connection = _get_connection
    return conn


def test_present_executable_is_never_rebuilt(controller):
    """The common case: boot must not touch the overlay other robots are using."""
    conn = _attach(controller, _FakeConnection(executable_present=True))

    note = asyncio.run(controller._ensure_workspace_built(_robot()))

    assert note is None
    assert not conn.build_launched


def test_missing_executable_triggers_build(controller):
    conn = _attach(controller, _FakeConnection(executable_present=False))

    note = asyncio.run(controller._ensure_workspace_built(_robot()))

    assert note == "workspace rebuilt"
    assert conn.build_launched
    build = next(c for c in conn.commands if "colcon build" in c)
    assert f"--packages-select {fleet_mod.PIPER_PACKAGE}" in build
    # Not --symlink-install: colcon implements it for ament_python as
    # `setup.py develop --editable`, which setuptools 80 (the lab image's
    # version) rejects outright — the build fails before compiling anything.
    assert "--symlink-install" not in build
    # Never the entrypoint's clean build — it would delete the overlay out from
    # under every other workstation's running client. Derived state under
    # build/ is fair game; install/ is not.
    assert f"rm -rf {fleet_mod.PIPER_WORKSPACE_ROOT}/install" not in build
    for deleted in re.findall(r"rm -rf (\S+)", build):
        assert deleted.startswith(fleet_mod.PIPER_BUILD_DIR), deleted


def test_build_discards_stale_staging_tree(controller):
    """A copy install reinstalls stale staged files unless staging is dropped.

    setuptools' build_py refreshes build/<pkg>/build/lib only when src is
    *newer*, which a shared NFS checkout cannot guarantee — so a build can
    reinstall an old node and still report success.
    """
    conn = _attach(controller, _FakeConnection(executable_present=False))

    asyncio.run(controller._ensure_workspace_built(_robot()))

    build = next(c for c in conn.commands if "colcon build" in c)
    assert f"rm -rf {fleet_mod.PIPER_STAGING_DIR}" in build
    assert build.index(fleet_mod.PIPER_STAGING_DIR) < build.index("colcon build")
    # Derived state only — install/ is the overlay other workstations read.
    assert fleet_mod.PIPER_STAGING_DIR.startswith(fleet_mod.PIPER_BUILD_DIR)


def test_build_clears_stale_develop_marker(controller):
    """A leftover symlinked setup.py in the build space breaks the copy install.

    colcon reads it as "last built with --symlink-install" and tries to undo
    that with `develop --uninstall --editable`, which setuptools 80 rejects —
    so the build fails on the previous build's debris, not on anything in src/.
    """
    conn = _attach(controller, _FakeConnection(executable_present=False))

    asyncio.run(controller._ensure_workspace_built(_robot()))

    build = next(c for c in conn.commands if "colcon build" in c)
    assert f"[ -L {fleet_mod.PIPER_DEVELOP_MARKER} ]" in build
    # Cleared before colcon runs, or colcon has already read it.
    assert build.index(fleet_mod.PIPER_DEVELOP_MARKER) < build.index("colcon build")
    # Only the regenerable build space — never install/, the overlay every
    # other workstation's running client reads.
    assert fleet_mod.PIPER_DEVELOP_MARKER.startswith(fleet_mod.PIPER_BUILD_DIR)
    assert f"{fleet_mod.PIPER_WORKSPACE_ROOT}/install" not in build


def test_build_is_serialized_by_flock(controller):
    conn = _attach(controller, _FakeConnection(executable_present=False))

    asyncio.run(controller._ensure_workspace_built(_robot()))

    build = next(c for c in conn.commands if "colcon build" in c)
    assert f"flock -w {int(fleet_mod.BUILD_LOCK_WAIT_SEC)} 9" in build
    assert fleet_mod.PIPER_BUILD_LOCK in build


def test_contended_lock_is_reported_distinctly(controller):
    """A busy lock must not read as a compile failure."""
    conn = _attach(
        controller,
        _FakeConnection(
            executable_present=False,
            build_status=str(fleet_mod.BUILD_LOCK_BUSY_STATUS),
        ),
    )

    note = asyncio.run(controller._ensure_workspace_built(_robot()))

    assert note.startswith("ERROR:")
    assert "already holds" in note


def test_auto_build_disabled_reports_instead_of_building(controller):
    controller.config.auto_build_workspace = False
    conn = _attach(controller, _FakeConnection(executable_present=False))

    note = asyncio.run(controller._ensure_workspace_built(_robot()))

    assert note.startswith("ERROR:")
    assert fleet_mod.PIPER_CLIENT_EXECUTABLE in note
    assert not conn.build_launched


def test_force_rebuilds_a_present_executable(controller):
    conn = _attach(controller, _FakeConnection(executable_present=True))

    note = asyncio.run(controller._ensure_workspace_built(_robot(), force=True))

    assert note == "workspace rebuilt"
    assert conn.build_launched


def test_failed_build_reports_exit_status(controller):
    conn = _attach(
        controller, _FakeConnection(executable_present=False, build_status="2")
    )

    note = asyncio.run(controller._ensure_workspace_built(_robot()))

    assert note.startswith("ERROR:")
    assert "exited 2" in note


def test_build_that_leaves_executable_missing_is_an_error(controller):
    conn = _FakeConnection(executable_present=False)
    # Build "succeeds" but never installs the entry point (setup.py not updated).
    conn.build_status = "0"

    async def run(command, timeout=None, check=True):
        conn.commands.append(command)
        if f"test -x {fleet_mod.PIPER_CLIENT_EXECUTABLE_PATH}" in command:
            return _Result(exit_status=1)
        if fleet_mod.PIPER_BUILD_STATUS_PATH in command and command.startswith("cat "):
            return _Result(stdout="0")
        return _Result()

    conn.run = run
    _attach(controller, conn)

    note = asyncio.run(controller._ensure_workspace_built(_robot()))

    assert note.startswith("ERROR:")
    assert "console_scripts" in note


def test_fleet_build_uses_a_single_workstation(controller):
    """14 containers share one install/ — only one may ever run colcon."""
    conn = _attach(controller, _FakeConnection(executable_present=False))
    robots = [_robot(11), _robot(12)]

    results = asyncio.run(controller._build_workspace_on_first(robots))

    assert list(results) == [11]
    assert sum("colcon build" in c for c in conn.commands) == 1


def test_fleet_build_skips_when_nothing_is_booted(controller):
    conn = _attach(controller, _FakeConnection(executable_present=False))
    offline = Robot(id=11, name="R11", ip="10.0.0.11", status=RobotStatus.OFFLINE)

    results = asyncio.run(controller._build_workspace_on_first([offline]))

    assert results == {}
    assert not conn.build_launched


def test_build_error_is_returned_not_raised(controller):
    """An escaping exception here used to wedge the TUI permanently.

    ``_boot_robots`` and ``_build_workspace_on_first`` both reach their
    ``callback(results)`` — the call that unlocks the dashboard — only by
    returning normally. The post-build ``test -x`` probe is a live SSH round
    trip that can time out, so this path has to report as a result.
    """

    class _ExplodingConnection(_FakeConnection):
        async def run(self, command, timeout=None, check=True):
            if f"test -x {fleet_mod.PIPER_CLIENT_EXECUTABLE_PATH}" in command:
                if self.commands:  # the post-build probe, not the initial one
                    raise TimeoutError("SSH probe timed out")
                self.commands.append(command)
                return _Result(exit_status=1)
            return await super().run(command, timeout=timeout, check=check)

    _attach(controller, _ExplodingConnection(executable_present=False))

    note = asyncio.run(controller._ensure_workspace_built(_robot()))

    assert note.startswith("ERROR:")
    assert "SSH probe timed out" in note


def test_boot_still_calls_back_when_the_build_probe_fails(controller):
    """Boot's callback is the dashboard's only unlock signal — it must fire."""

    class _ExplodingConnection(_FakeConnection):
        async def run(self, command, timeout=None, check=True):
            if "docker ps" in command or "@@" in command:
                return _Result(stdout=fleet_mod._BOOT_ALREADY)
            if f"test -x {fleet_mod.PIPER_CLIENT_EXECUTABLE_PATH}" in command:
                raise TimeoutError("SSH probe timed out")
            return await super().run(command, timeout=timeout, check=check)

    _attach(controller, _ExplodingConnection(executable_present=False))
    controller.config.piper_docker_dir = "/lab/docker"
    seen = {}

    asyncio.run(controller._boot_robots([_robot()], callback=seen.update))

    assert seen, "boot must invoke its callback even when the build probe fails"
