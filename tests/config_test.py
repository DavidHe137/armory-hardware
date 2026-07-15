"""Unit tests for FleetConfig (no network, no SSH)."""

from __future__ import annotations

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


def test_tunnel_defaults_when_section_missing(tmp_path):
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
    assert cfg.tunnel_server == "sky1.cc.gatech.edu"


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
