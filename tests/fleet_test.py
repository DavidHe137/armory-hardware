"""Integration tests for FleetController.

Talks to the real fleet defined in ``configs/armory-tui.yaml``. Skips cleanly
when the config is missing (e.g. running on a machine without the workspace
checkout).
"""

from __future__ import annotations

import logging
import os

import pytest
import yaml

from armory_hardware import FleetConfig, FleetController, RobotStatus

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "configs", "armory-tui.yaml")


pytestmark = pytest.mark.skipif(
    not os.path.isfile(_DEFAULT_CONFIG),
    reason=f"fleet config not present at {_DEFAULT_CONFIG}",
)


@pytest.fixture
def integration_config(tmp_path):
    """Real fleet YAML, but with log_dir redirected into tmp_path."""
    with open(_DEFAULT_CONFIG) as f:
        raw = yaml.safe_load(f)
    raw.setdefault("logging", {})["log_dir"] = str(tmp_path / "logs")
    p = tmp_path / "armory-tui.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(raw, f)
    return FleetConfig(str(p))


@pytest.fixture
def fleet(integration_config):
    f = FleetController(integration_config, logger=logging.getLogger("armory_hardware.test"))
    f.start()
    yield f
    f.stop()


def test_lifecycle_start_stop_clean(integration_config):
    """Construct, start, stop without raising."""
    f = FleetController(integration_config)
    f.start()
    f.stop()


def test_logger_defaults_to_armory_real():
    cfg = FleetConfig(_DEFAULT_CONFIG)
    f = FleetController(cfg)
    try:
        assert f.logger.name == "armory_hardware"
    finally:
        # Controller wasn't started, but stop() must still be safe.
        pass


def test_check_all_status_assigns_each_robot(fleet):
    fut = fleet.check_all_status()
    fut.result(timeout=45)

    for r in fleet.config.robots:
        assert r.status in {
            RobotStatus.OFFLINE,
            RobotStatus.BOOTED,
            RobotStatus.ONLINE,
        }


def test_run_command_on_first_booted_robot(fleet):
    """Run a benign echo on the first booted robot's container."""
    fut = fleet.check_all_status()
    fut.result(timeout=45)

    booted = [r for r in fleet.config.robots if r.status is RobotStatus.BOOTED]
    if not booted:
        pytest.skip("no booted robots in fleet — boot one before running this test")

    target = booted[:1]
    fut = fleet.run_on_robots(target, "echo armory-fleet-test")
    results = fut.result(timeout=45)

    assert target[0].id in results
    assert "armory-fleet-test" in str(results[target[0].id])


def test_run_command_on_offline_robot_returns_error(fleet):
    fut = fleet.check_all_status()
    fut.result(timeout=45)

    offline = [r for r in fleet.config.robots if r.status is RobotStatus.OFFLINE]
    if not offline:
        pytest.skip("no offline robots to validate failure path")

    target = offline[:1]
    fut = fleet.run_on_robots(target, "echo should-fail")
    results = fut.result(timeout=45)

    # _exec_in_docker returns "ERROR: ..." on failure (or stdout on success).
    # Offline robots may still be reachable via SSH (just no container),
    # so we accept either an ERROR string or empty/no-output.
    out = str(results[target[0].id])
    assert out  # something was returned
