"""ARMory — curses console for the real-robot fleet."""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import sys

from armory_hardware import FleetConfig, FleetController, FleetDispatcher
from armory_tui.dashboard import Dashboard

# ── ASCII splash ────────────────────────────────────────────────

ESC = chr(27)
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
PASTEL_PINK = f"{ESC}[38;5;225m"
PASTEL_LAVENDER = f"{ESC}[38;5;183m"
PASTEL_SKY = f"{ESC}[38;5;153m"
MUTED = f"{ESC}[38;5;250m"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BANNER_WIDTH = 54


def _visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def _center_line(text: str, width: int) -> str:
    return " " * max(0, (width - _visible_len(text)) // 2) + text


def _banner_line(text: str) -> str:
    inner_width = BANNER_WIDTH - 2
    padding = max(0, inner_width - _visible_len(text))
    left = padding // 2
    right = padding - left
    return f"{PASTEL_LAVENDER}│{RESET}{' ' * left}{text}{' ' * right}{PASTEL_LAVENDER}│{RESET}"


def _splash_lines() -> list[str]:
    brand = f"{PASTEL_PINK}{BOLD}ARM{RESET}{PASTEL_SKY}{BOLD}ory{RESET}"
    subtitle = f"{MUTED}Advanced Robotic Manipulation{RESET}"
    console = f"{MUTED}fleet console{RESET}"
    return [
        f"{PASTEL_LAVENDER}╭{'─' * (BANNER_WIDTH - 2)}╮{RESET}",
        _banner_line(brand),
        _banner_line(subtitle),
        _banner_line(console),
        f"{PASTEL_LAVENDER}╰{'─' * (BANNER_WIDTH - 2)}╯{RESET}",
    ]


def bootup():
    """Display the ASCII splash screen and welcome message."""
    print(f"{ESC}[2J{ESC}[H", end="")

    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USER", "operator")

    cols, rows = shutil.get_terminal_size(fallback=(100, 30))
    splash = _splash_lines()
    welcome = f"{PASTEL_SKY}Welcome, {BOLD}{user}{RESET}"
    prompt = f"{MUTED}Press Enter to proceed{RESET}"

    block_height = len(splash) + 4
    top_padding = max(0, (rows - block_height) // 2)
    print("\n" * top_padding, end="")

    for line in splash:
        print(_center_line(line, cols))
    print()
    print(_center_line(welcome, cols))
    print()
    input(_center_line(prompt, cols))


# ── logging bridge ──────────────────────────────────────────────


class _DashboardLogHandler(logging.Handler):
    """Forwards FleetController log events into the dashboard's log panel."""

    def __init__(self, dashboard: Dashboard):
        super().__init__()
        self.dashboard = dashboard

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.dashboard._log(record.getMessage())
        except Exception:
            self.handleError(record)


# ── config resolution ──────────────────────────────────────────


def _resolve_config_path() -> str:
    """Locate the Armory fleet YAML (``configs/armory-tui.yaml`` in the repo).

    Resolution order:
      1. ``ARMORY_TUI_CONFIG`` env var (path to any fleet YAML).
      2. ``configs/armory-tui.yaml``, ``armory-tui.yaml``, or ``config.yaml`` in cwd.
      3. Repo default: ``<armory>/configs/armory-tui.yaml`` (four parents up from this file).
    """
    env_path = os.environ.get("ARMORY_TUI_CONFIG")
    if env_path:
        return os.path.abspath(os.path.expanduser(env_path.strip()))

    cwd = os.getcwd()
    for name in (
        os.path.join("configs", "armory-tui.yaml"),
        "armory-tui.yaml",
        "config.yaml",
    ):
        p = os.path.join(cwd, name)
        if os.path.isfile(p):
            return p

    hardware_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(hardware_root, "configs", "armory-tui.yaml")


# ── entry point ────────────────────────────────────────────────


def main():
    config_path = _resolve_config_path()
    config = FleetConfig(config_path)

    dashboard = Dashboard(config, fleet=None, dispatcher=None)

    # Bridge the controller's event stream into the dashboard log panel.
    handler = _DashboardLogHandler(dashboard)
    logger = logging.getLogger("armory_hardware")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    fleet = FleetController(config, logger=logger)
    fleet.start()

    dispatcher = FleetDispatcher(fleet)
    dashboard.fleet = fleet
    dashboard.dispatcher = dispatcher

    original_sigint = signal.getsignal(signal.SIGINT)

    def _shutdown(_sig, _frame):
        fleet.stop()
        signal.signal(signal.SIGINT, original_sigint)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    try:
        dashboard.run()
    finally:
        fleet.stop()
        logger.removeHandler(handler)


def run():
    """Console-script entry point: splash + main."""
    bootup()
    main()


if __name__ == "__main__":
    run()
