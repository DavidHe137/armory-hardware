"""Unit tests for armory_tui entry-point wiring (no curses, no SSH)."""

from __future__ import annotations

import logging
import os
from collections import deque

from armory_tui.main import _DashboardLogHandler, _resolve_config_path

# ── _resolve_config_path ───────────────────────────────────────────


def test_resolve_config_prefers_env_var(tmp_path, monkeypatch):
    target = tmp_path / "custom.yaml"
    target.write_text("workstations: []\n")
    monkeypatch.setenv("ARMORY_TUI_CONFIG", str(target))
    monkeypatch.chdir(tmp_path)

    assert _resolve_config_path() == str(target)


def test_resolve_config_expands_user_in_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ARMORY_TUI_CONFIG", "~/custom.yaml")

    resolved = _resolve_config_path()
    assert resolved == str(tmp_path / "custom.yaml")


def test_resolve_config_finds_configs_subdir_in_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("ARMORY_TUI_CONFIG", raising=False)
    configs = tmp_path / "configs"
    configs.mkdir()
    target = configs / "armory-tui.yaml"
    target.write_text("workstations: []\n")
    monkeypatch.chdir(tmp_path)

    assert _resolve_config_path() == str(target)


def test_resolve_config_finds_bare_yaml_in_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("ARMORY_TUI_CONFIG", raising=False)
    target = tmp_path / "armory-tui.yaml"
    target.write_text("workstations: []\n")
    monkeypatch.chdir(tmp_path)

    assert _resolve_config_path() == str(target)


def test_resolve_config_falls_back_to_repo_default(tmp_path, monkeypatch):
    """With no env var and an empty cwd, returns the repo default path."""
    monkeypatch.delenv("ARMORY_TUI_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_config_path()

    assert resolved.endswith(os.path.join("configs", "armory-tui.yaml"))
    assert os.path.isabs(resolved)


# ── _DashboardLogHandler ──────────────────────────────────────────


class _FakeDashboard:
    def __init__(self):
        self.log_lines = deque(maxlen=10)
        self.fleet = None

    def _log(self, msg: str) -> None:
        self.log_lines.append(msg)


def test_dashboard_log_handler_forwards_record_message():
    dashboard = _FakeDashboard()
    handler = _DashboardLogHandler(dashboard)
    logger = logging.getLogger("armory_hardware.test_handler")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        logger.info("WS-11: container running (booted)")
        logger.info("WS-12: %s", "unreachable")
    finally:
        logger.removeHandler(handler)

    assert list(dashboard.log_lines) == [
        "WS-11: container running (booted)",
        "WS-12: unreachable",
    ]


def test_dashboard_log_handler_does_not_propagate_emit_errors():
    """Errors inside dashboard._log must not break the logging stack."""

    class _BrokenDashboard:
        def _log(self, msg):
            raise RuntimeError("dashboard exploded")

    handler = _DashboardLogHandler(_BrokenDashboard())
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname="", lineno=0,
        msg="boom", args=(), exc_info=None,
    )

    # Should not raise — the handler swallows internal errors via handleError().
    handler.emit(record)
