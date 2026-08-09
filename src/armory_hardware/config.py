"""YAML configuration parser and robot state for a real robot fleet."""

from __future__ import annotations

import getpass
import os
import posixpath
import random
from dataclasses import dataclass
from enum import Enum

import yaml


class RobotStatus(Enum):
    OFFLINE = "offline"
    BOOTED = "booted"
    ONLINE = "online"


@dataclass
class Robot:
    id: int
    name: str
    ip: str
    status: RobotStatus = RobotStatus.OFFLINE


_NAMES = [
    "Shadow", "Neon", "Crimson", "Cobalt", "Phantom", "Onyx",
    "Rogue", "Titan", "Nova", "Blaze", "Frost", "Volt",
    "Echo", "Flux", "Prism", "Viper", "Falcon", "Raven",
    "Lynx", "Mantis",
]


def _armory_repo_root(config_path: str) -> str:
    """Infer Armory workspace root from the config location.

    If the config sits in ``.../<root>/configs/``, ``<root>`` is returned. Otherwise
    ``dirname(config)`` is used so relative ``logging.log_dir`` paths stay predictable.
    """
    d = os.path.dirname(os.path.abspath(config_path))
    if os.path.basename(d) == "configs":
        return os.path.dirname(d)
    return d


def _setting(yaml_value, env_var: str, default: str) -> str:
    """Resolve one setting: explicit YAML value, else ``env_var``, else ``default``.

    YAML wins so a checked-in config stays authoritative and tests stay hermetic; the
    environment fills in the per-operator values that are deliberately absent from the
    committed config. Blank YAML values are treated as absent.
    """
    if yaml_value is not None and str(yaml_value).strip() != "":
        return str(yaml_value).strip()
    return os.environ.get(env_var, "").strip() or default


def _generate_name(used: set[str]) -> str:
    while True:
        name = random.choice(_NAMES)
        if name not in used:
            used.add(name)
            return name


class FleetConfig:
    """Parses a fleet config.yaml and holds runtime robot state.

    The library does not bundle a default config; callers must supply a path.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        self._config_path = os.path.abspath(config_path)

        ssh_cfg = raw.get("ssh") or {}
        self.ssh_user = _setting(ssh_cfg.get("user"), "ARMORY_SSH_USER", getpass.getuser())
        self.base_ip = _setting(ssh_cfg.get("base_ip"), "ARMORY_SSH_BASE_IP", "127.0.0")
        self.ip_offset = int(_setting(ssh_cfg.get("ip_offset"), "ARMORY_SSH_IP_OFFSET", "200"))

        tunnel_cfg = raw.get("tunnel") or {}
        self.tunnel_node = _setting(tunnel_cfg.get("node"), "ARMORY_TUNNEL_NODE", "")
        self.tunnel_port = int(_setting(tunnel_cfg.get("port"), "ARMORY_TUNNEL_PORT", "8080"))
        self.tunnel_user = _setting(tunnel_cfg.get("user"), "ARMORY_TUNNEL_USER", self.ssh_user)
        self.tunnel_server = _setting(tunnel_cfg.get("server"), "ARMORY_TUNNEL_SERVER", "")

        webcam_cfg = raw.get("webcam") or {}
        self.webcam_rtsp_base_url = _setting(
            webcam_cfg.get("rtsp_base_url"), "ARMORY_WEBCAM_RTSP_BASE_URL", ""
        )

        # Workstation-side lab checkout. Named after the workstation's own
        # PIPER_DOCKER_DIR, but read from here rather than the workstation's
        # ~/.bashrc: that file only defines the variable for interactive shells,
        # so every non-interactive lookup silently produced an empty string and
        # bind-mounted the wrong host paths.
        docker_cfg = raw.get("docker") or {}
        self.piper_docker_dir = _setting(
            docker_cfg.get("piper_dir"), "ARMORY_PIPER_DOCKER_DIR", ""
        )

        used_names: set[str] = set()
        self.robots: list[Robot] = []
        for entry in raw.get("workstations", []):
            wid = entry["id"]
            ip = f"{self.base_ip}.{self.ip_offset + wid}"
            name = _generate_name(used_names)
            self.robots.append(Robot(id=wid, name=name, ip=ip))

        self.robots.sort(key=lambda r: r.id)

        repo_root = _armory_repo_root(self._config_path)
        log_section = raw.get("logging") or {}
        raw_log_dir = log_section.get("log_dir")
        if raw_log_dir is None or str(raw_log_dir).strip() == "":
            self._log_dir = os.path.join(repo_root, "logs", "tui")
        else:
            p = os.path.expanduser(str(raw_log_dir).strip())
            self._log_dir = p if os.path.isabs(p) else os.path.normpath(os.path.join(repo_root, p))

    @property
    def piper_root(self) -> str:
        """Parent of ``piper_docker_dir`` — holds ``user_data/`` and ``assets/``.

        Empty when ``piper_docker_dir`` is unset, so callers can report a missing
        setting instead of building a path rooted at ``/``.
        """
        if not self.piper_docker_dir:
            return ""
        return posixpath.normpath(posixpath.join(self.piper_docker_dir, ".."))

    def get_robot(self, robot_id: int) -> Robot | None:
        for r in self.robots:
            if r.id == robot_id:
                return r
        return None

    @property
    def log_dir(self) -> str:
        """Directory for fleet file logs (from ``logging.log_dir`` in YAML)."""
        os.makedirs(self._log_dir, exist_ok=True)
        return self._log_dir
