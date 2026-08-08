"""Unit tests for FleetConfig (no network, no SSH)."""

from __future__ import annotations

import getpass
import os
from textwrap import dedent

import pytest

from armory_hardware import FleetConfig, RobotStatus


def _write_yaml(path, body: str) -> str:
    path.write_text(dedent(body).lstrip())
    return str(path)


def test_parses_workstations_and_assigns_ips(tmp_path):
    cfg_path = _write_yaml(
        tmp_path / "fleet.yaml",
        """
        workstations:
          - id: 11
          - id: 12
          - id: 13
        ssh:
          user: tester
          base_ip: "10.0.0"
          ip_offset: 100
        """,
    )

    cfg = FleetConfig(cfg_path)

    assert [r.id for r in cfg.robots] == [11, 12, 13]
    assert [r.ip for r in cfg.robots] == ["10.0.0.111", "10.0.0.112", "10.0.0.113"]
    assert cfg.ssh_user == "tester"
    assert all(r.status is RobotStatus.OFFLINE for r in cfg.robots)
    # Names are randomly assigned but must be unique.
    assert len({r.name for r in cfg.robots}) == 3


def test_get_robot_returns_match_or_none(tmp_path):
    cfg_path = _write_yaml(
        tmp_path / "fleet.yaml",
        """
        workstations:
          - id: 7
        """,
    )

    cfg = FleetConfig(cfg_path)

    assert cfg.get_robot(7).id == 7
    assert cfg.get_robot(999) is None


_ENV_VARS = [
    "ARMORY_SSH_USER",
    "ARMORY_SSH_BASE_IP",
    "ARMORY_SSH_IP_OFFSET",
    "ARMORY_TUNNEL_NODE",
    "ARMORY_TUNNEL_PORT",
    "ARMORY_TUNNEL_USER",
    "ARMORY_TUNNEL_SERVER",
    "ARMORY_WEBCAM_RTSP_BASE_URL",
]


@pytest.fixture
def clean_env(monkeypatch):
    """Drop any ARMORY_* overrides so default resolution is testable."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_tunnel_defaults_when_section_missing(tmp_path, clean_env):
    cfg_path = _write_yaml(
        tmp_path / "fleet.yaml",
        """
        workstations:
          - id: 1
        """,
    )

    cfg = FleetConfig(cfg_path)

    assert cfg.tunnel_node == ""
    assert cfg.tunnel_port == 8080
    assert cfg.tunnel_user == cfg.ssh_user
    assert cfg.tunnel_server == ""
    assert cfg.webcam_rtsp_base_url == ""


def test_ssh_defaults_to_local_user_and_placeholder_range(tmp_path, clean_env):
    """No committed config or env means no borrowed identity: local user, loopback."""
    cfg_path = _write_yaml(
        tmp_path / "fleet.yaml",
        """
        workstations:
          - id: 1
        """,
    )

    cfg = FleetConfig(cfg_path)

    assert cfg.ssh_user == getpass.getuser()
    assert cfg.base_ip == "127.0.0"
    assert cfg.robots[0].ip == "127.0.0.201"


def test_env_supplies_settings_absent_from_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("ARMORY_SSH_USER", "envuser")
    monkeypatch.setenv("ARMORY_SSH_BASE_IP", "192.168.1")
    monkeypatch.setenv("ARMORY_SSH_IP_OFFSET", "10")
    monkeypatch.setenv("ARMORY_TUNNEL_SERVER", "jump.example.edu")
    monkeypatch.setenv("ARMORY_TUNNEL_NODE", "envnode")
    monkeypatch.setenv("ARMORY_TUNNEL_PORT", "9090")
    monkeypatch.delenv("ARMORY_TUNNEL_USER", raising=False)
    monkeypatch.setenv("ARMORY_WEBCAM_RTSP_BASE_URL", "rtsp://cam.example:8554")

    cfg_path = _write_yaml(
        tmp_path / "fleet.yaml",
        """
        workstations:
          - id: 3
        """,
    )

    cfg = FleetConfig(cfg_path)

    assert cfg.ssh_user == "envuser"
    assert cfg.robots[0].ip == "192.168.1.13"
    assert cfg.tunnel_server == "jump.example.edu"
    assert cfg.tunnel_node == "envnode"
    assert cfg.tunnel_port == 9090
    # Falls back to ssh_user, which itself came from the environment.
    assert cfg.tunnel_user == "envuser"
    assert cfg.webcam_rtsp_base_url == "rtsp://cam.example:8554"


def test_yaml_value_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARMORY_SSH_USER", "envuser")
    monkeypatch.setenv("ARMORY_TUNNEL_SERVER", "env.example.edu")

    cfg_path = _write_yaml(
        tmp_path / "fleet.yaml",
        """
        workstations:
          - id: 1
        ssh:
          user: yamluser
        tunnel:
          server: yaml.example.edu
        """,
    )

    cfg = FleetConfig(cfg_path)

    assert cfg.ssh_user == "yamluser"
    assert cfg.tunnel_server == "yaml.example.edu"


def test_no_personal_identifiers_in_committed_config():
    """The checked-in fleet config must not carry an operator's account or hosts."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(repo_root, "configs", "armory-tui.yaml")).read()

    for section in ("ssh:", "tunnel:", "webcam:"):
        assert section not in text


def test_log_dir_defaults_to_repo_root_logs_tui(tmp_path):
    """When config lives in <root>/configs/, default log_dir = <root>/logs/tui."""
    configs = tmp_path / "configs"
    configs.mkdir()
    cfg_path = _write_yaml(
        configs / "fleet.yaml",
        """
        workstations:
          - id: 1
        """,
    )

    cfg = FleetConfig(cfg_path)

    assert cfg.log_dir == str(tmp_path / "logs" / "tui")
    assert os.path.isdir(cfg.log_dir)


def test_log_dir_custom_relative_resolves_against_repo_root(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    cfg_path = _write_yaml(
        configs / "fleet.yaml",
        """
        logging:
          log_dir: custom/sublog
        workstations:
          - id: 1
        """,
    )

    cfg = FleetConfig(cfg_path)

    assert cfg.log_dir == str(tmp_path / "custom" / "sublog")


def test_log_dir_absolute_path_honored(tmp_path):
    target = tmp_path / "elsewhere" / "logs"
    cfg_path = _write_yaml(
        tmp_path / "fleet.yaml",
        f"""
        logging:
          log_dir: {target}
        workstations:
          - id: 1
        """,
    )

    cfg = FleetConfig(cfg_path)

    assert cfg.log_dir == str(target)
    assert os.path.isdir(cfg.log_dir)


def test_log_dir_falls_back_to_config_dir_when_no_configs_parent(tmp_path):
    """Config not inside a configs/ folder → repo root falls back to config dir."""
    cfg_path = _write_yaml(
        tmp_path / "fleet.yaml",
        """
        workstations:
          - id: 1
        """,
    )

    cfg = FleetConfig(cfg_path)

    assert cfg.log_dir == str(tmp_path / "logs" / "tui")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FleetConfig(str(tmp_path / "does_not_exist.yaml"))
