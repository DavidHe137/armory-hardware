"""Hermetic tests for pointing the client node at the policy server.

The client node defaults ``host``/``port`` to ``localhost:8080``, which only
resolves while an SSH tunnel forwards that port. Naming the server directly
takes the tunnel out of the path, so these pin the argv shape that override
depends on -- no fleet, no network.
"""

from __future__ import annotations

import pytest

from armory_hardware import FleetConfig, FleetController

# What the dispatcher contributes per robot, verbatim in shape: a ``--``
# separator followed by the node's own argparse flags.
DISPATCHER_ARGS = "-- --control-hz 25.0 --prompt 'pick up the mug' --barrier"


def _controller(tmp_path, host: str | None = None, port: int = 8080) -> FleetController:
    lines = [
        "workstations:",
        "  - id: 11",
        "logging:",
        f"  log_dir: {tmp_path / 'logs'}",
    ]
    if host is not None:
        lines += ["policy:", f"  host: {host}", f"  port: {port}"]
    cfg_path = tmp_path / "fleet.yaml"
    cfg_path.write_text("\n".join(lines) + "\n")
    return FleetController(FleetConfig(str(cfg_path)))


@pytest.fixture
def controller(tmp_path, monkeypatch):
    """Configured to reach the policy server directly."""
    monkeypatch.delenv("ARMORY_POLICY_HOST", raising=False)
    monkeypatch.delenv("ARMORY_POLICY_PORT", raising=False)
    return _controller(tmp_path, host="grom")


@pytest.fixture
def unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("ARMORY_POLICY_HOST", raising=False)
    monkeypatch.delenv("ARMORY_POLICY_PORT", raising=False)
    return _controller(tmp_path)


def test_policy_args_name_host_and_port(controller):
    args = controller._policy_ros_args()

    assert "--ros-args" in args
    assert "-p host:=grom" in args
    assert "-p port:=8080" in args


def test_unconfigured_host_leaves_the_node_default_alone(unconfigured):
    """No host configured must not mean ``host:=''`` -- the tunnel setup still works."""
    assert unconfigured._policy_ros_args() == ""

    robot = unconfigured.config.robots[0]
    base = "ros2 run piper piper_client_armory"
    resolved = unconfigured._command_for_robot(
        base, robot, None, unconfigured._policy_ros_args()
    )

    assert resolved == base


def test_ros_args_come_after_the_dispatcher_args(controller):
    """Ordering is load-bearing, not cosmetic.

    rclpy consumes everything from ``--ros-args`` to the end of argv, so any
    dispatcher flag appended behind it would be eaten as a ROS argument and
    never reach the node's argparse -- silently dropping the prompt and the
    startup barrier.
    """
    robot = controller.config.robots[0]
    resolved = controller._command_for_robot(
        "ros2 run piper piper_client_armory",
        robot,
        {robot.id: DISPATCHER_ARGS},
        controller._policy_ros_args(),
    )

    assert resolved.index("--barrier") < resolved.index("--ros-args")
    assert resolved.index("--prompt") < resolved.index("--ros-args")
    # The dispatcher's own separator has to stay ahead of both.
    assert resolved.index("-- --control-hz") < resolved.index("--ros-args")
    assert resolved.endswith("-p port:=8080")


def test_policy_args_applied_without_dispatcher_args(controller):
    """Start Client outside a trial still gets the endpoint."""
    robot = controller.config.robots[0]
    resolved = controller._command_for_robot(
        "ros2 run piper piper_client_armory", robot, None, controller._policy_ros_args()
    )

    assert resolved == (
        "ros2 run piper piper_client_armory --ros-args -p host:=grom -p port:=8080"
    )


def test_base_command_stays_a_prefix(controller):
    """Kill matching greps the base command as a substring -- start/kill stay symmetric."""
    robot = controller.config.robots[0]
    base = "ros2 run piper piper_client_armory"
    resolved = controller._command_for_robot(
        base, robot, {robot.id: DISPATCHER_ARGS}, controller._policy_ros_args()
    )

    assert resolved.startswith(base)
