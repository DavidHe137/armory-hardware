"""YAML configuration parser and robot state for a real robot fleet."""

from __future__ import annotations

import os
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

        ssh_cfg = raw.get("ssh", {})
        self.ssh_user = ssh_cfg.get("user", "rbansal66")
        self.base_ip = ssh_cfg.get("base_ip", "130.207.121")
        self.ip_offset = ssh_cfg.get("ip_offset", 200)

        tunnel_cfg = raw.get("tunnel", {})
        self.tunnel_node = str(tunnel_cfg.get("node", "") or "")
        self.tunnel_port = int(tunnel_cfg.get("port", 8080))
        self.tunnel_user = str(tunnel_cfg.get("user", self.ssh_user) or self.ssh_user)
        self.tunnel_server = str(tunnel_cfg.get("server", "sky1.cc.gatech.edu") or "")

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
