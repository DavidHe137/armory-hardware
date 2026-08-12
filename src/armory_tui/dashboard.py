"""Curses-based CLI dashboard for ARMory."""

from __future__ import annotations

import curses
import textwrap
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

from armory_hardware import RobotStatus

if TYPE_CHECKING:
    from armory_hardware import FleetConfig, FleetController, FleetDispatcher


# ── colour pairs ────────────────────────────────────────────────
PAIR_HEADER = 1
PAIR_OFFLINE = 2
PAIR_BOOTED = 3
PAIR_ONLINE = 4
PAIR_BORDER = 5
PAIR_CMD = 6
PAIR_LOG = 7
PAIR_HIGHLIGHT = 8
PAIR_MUTED = 9
PAIR_NOTICE = 10
PAIR_SURFACE = 11

MAX_RESULT_CHARS = 400
MAX_LOG_WRAP_ROWS = 4

PANE_FLEET = "fleet"
PANE_LOG = "log"

MIN_WIDTH = 80
MIN_HEIGHT = 24

CORE_COMMANDS = [
    ("6", "Boot Fleet"),
    ("T", "Start Tunnel"),
    ("X", "Kill Tunnel"),
    ("7", "Shutdown"),
    ("8", "Kill All"),
    ("R", "Refresh"),
    ("Q", "Quit"),
]

# [A] Abort is deliberately not in the grid: it only does anything while a
# command is in flight, and the grid truncates past four rows on a 24-row
# terminal. It is advertised in the header instead, exactly when it applies.
KEY_ABORT = "A"

RUNTIME_COMMANDS = [
    ("L", "Listener"),
    ("C", "Client"),
    ("W", "Webcam"),
    ("5", "Connect"),
    ("1", "Enable"),
    ("2", "Disable"),
    ("3", "Goto Init"),
    ("4", "Goto Zero"),
    # Last: the command grid truncates on short terminals, and this is the
    # rarest action here — boot rebuilds the workspace on its own.
    ("B", "Build WS"),
]

RUNTIME_KEYS = {key for key, _ in RUNTIME_COMMANDS}


class Dashboard:
    """Renders the main ARMory curses dashboard."""

    def __init__(
        self,
        config: FleetConfig,
        fleet: FleetController,
        dispatcher: FleetDispatcher,
    ):
        self.config = config
        self.fleet = fleet
        self.dispatcher = dispatcher
        self.log_lines: deque[str] = deque(maxlen=200)
        self._busy = False
        self._lock = threading.Lock()
        self._last_busy_notice = 0.0
        self._runtime_unlocked_announced = False
        self._focused_robot_idx = 0
        self._selected_robot_ids: set[int] = set()
        self._notice_message = ""
        self._notice_until = 0.0
        self._client_running_robot_ids: set[int] = set()
        self._client_status_pending = False
        self._emergency_stop_active = False
        self._active_pane = PANE_FLEET
        # In-flight command bookkeeping. `_op_token` identifies the operation
        # that currently owns `_busy`, so a late callback from an operation the
        # operator already abandoned can't unlock the UI for a newer one.
        self._op_name = ""
        self._op_started = 0.0
        self._op_token = 0
        self._op_future = None
        self._op_state: dict | None = None
        # Work handed back from a fleet callback to the curses thread; see
        # `_defer`.
        self._deferred: list = []
        # Rows scrolled up from the newest entry; 0 means "follow the tail".
        self._log_scroll = 0
        self._log_rows_total = 0
        self._log_capacity = 1

    # ── public entry point ──────────────────────────────────────

    def run(self):
        curses.wrapper(self._main)

    # ── curses main loop ────────────────────────────────────────

    def _main(self, stdscr: curses.window):
        self._stdscr = stdscr
        self._init_colors()
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.nodelay(True)
        stdscr.keypad(True)

        self._log("ARMory dashboard started.")
        self._refresh_status("Querying workstation status...")

        while True:
            try:
                self._drain_deferred()
                self._render()
                key = stdscr.getch()
                if key == -1:
                    # time.sleep, not curses.napms: napms holds the GIL for the
                    # whole nap, so the fleet's asyncio thread only advanced
                    # between naps. That capped every SSH round trip at 100ms
                    # and made the 14-way cold status check on startup blow its
                    # per-command timeout on all but a workstation or two.
                    time.sleep(0.1)
                    continue
                if self._handle_key(key):
                    break
            except KeyboardInterrupt:
                break

    # ── deferred work ───────────────────────────────────────────

    def _defer(self, action):
        """Queue `action` to run on the curses thread.

        Fleet callbacks land on the dispatcher's asyncio thread, which must not
        touch curses — the main loop is drawing from the other thread and the
        two corrupt each other's screen state. Anything a callback wants to do
        that involves a dialog goes through here and runs on the next loop pass.
        """
        with self._lock:
            self._deferred.append(action)

    def _drain_deferred(self):
        with self._lock:
            pending, self._deferred = self._deferred, []
        for action in pending:
            try:
                action()
            except Exception as e:
                self._log(f"Deferred action failed: {e}")

    # ── input handling ──────────────────────────────────────────

    def _handle_key(self, key: int) -> bool:
        """Handle a keypress. Returns True to quit."""
        ch = chr(key).upper() if 0 <= key < 256 else ""

        if ch == "Q":
            return True

        # Pane switching and log scrolling are read-only, so they stay live even
        # while a command (or an emergency stop) is running.
        if key in (ord("\t"), curses.KEY_BTAB):
            self._toggle_pane()
            return False

        if self._active_pane == PANE_LOG and self._handle_log_scroll(key, ch):
            return False

        # Abort outranks every busy guard below — it exists precisely for the
        # case where the operation holding those guards is never going to
        # finish on its own.
        if ch == KEY_ABORT:
            self._abort_operation()
            return False

        if self._emergency_stop_active:
            self._throttled_busy_log(
                "Emergency client stop is still running. Press [A] to abort and unlock."
            )
            return False

        if key == curses.KEY_UP or ch == "K":
            self._move_robot_focus(-1)
            return False
        if key == curses.KEY_DOWN or ch == "J":
            self._move_robot_focus(1)
            return False
        if ch == " ":
            if self._clients_running():
                self._emergency_stop_clients()
            else:
                self._toggle_focused_robot()
            return False

        if self._busy:
            self._throttled_busy_log(
                f"'{self._op_name or 'Current operation'}' still running "
                f"({time.time() - self._op_started:.0f}s). "
                "Press [A] to abort and unlock the dashboard."
            )
            return False

        if ch in RUNTIME_KEYS and not self._runtime_controls_unlocked():
            self._log(
                "Runtime controls stay locked until the current target is booted "
                f"({len(self._eligible_targets('runtime', self._command_targets()))}/"
                f"{len(self._command_targets())} ready)."
            )
            return False

        if ch == "R":
            self._refresh_status()
        elif ch == "T":
            self._do_start_tunnel()
        elif ch == "X":
            self._do_kill_tunnel()
        elif ch == "L":
            self._open_process_menu(
                "Listener",
                self.dispatcher.start_listener,
                self.dispatcher.kill_listener,
            )
        elif ch == "C":
            self._open_process_menu(
                "Client",
                self.dispatcher.start_client,
                self.dispatcher.kill_client,
                process_key="client",
            )
        elif ch == "W":
            self._open_process_menu(
                "Webcam",
                self.dispatcher.start_webcam,
                self.dispatcher.kill_webcam,
            )
        elif ch == "B":
            # Boot already rebuilds when the client executable is missing; this
            # is for src/ changing under an executable that is still installed.
            self._broadcast_with_confirm(
                "Build Workspace", self.dispatcher.build_workspace
            )
        elif ch == "1":
            self._do_guarded_motion("Enable", self.dispatcher.enable)
        elif ch == "2":
            self._do_guarded_motion("Disable", self.dispatcher.disable)
        elif ch == "3":
            self._do_guarded_motion("Goto Init", self.dispatcher.goto_init)
        elif ch == "4":
            self._do_guarded_motion("Goto Zero", self.dispatcher.goto_zero)
        elif ch == "5":
            self._do_connect_to_server()
        elif ch == "6":
            self._do_boot()
        elif ch == "7":
            self._do_shutdown()
        elif ch == "8":
            self._do_kill_all()

        return False

    # ── operation tracking ──────────────────────────────────────

    def _start_operation(
        self,
        name: str,
        launch,
        start_message: str | None = None,
        on_results=None,
    ):
        """Run a fleet command with a guaranteed unlock.

        Every command used to set ``_busy`` and rely solely on its own callback
        to clear it. Anything that kept that callback from firing — an
        exception escaping the fleet coroutine, a submit that never scheduled —
        locked the operator out of the dashboard permanently, with nothing in
        the log to say why. Killing a client, refreshing, even shutting down
        cleanly all became impossible.

        So the submitted future is tracked here as well: the operation unlocks
        on whichever of the two paths lands first, and ``[A]`` has a handle to
        cancel when neither does.
        """
        with self._lock:
            self._op_token += 1
            token = self._op_token
            state: dict = {"done": False}
            self._busy = True
            self._op_name = name
            self._op_started = time.time()
            self._op_future = None
            self._op_state = state

        if start_message:
            self._log(start_message)

        def finish(results=None, error=None):
            with self._lock:
                if state["done"]:
                    return
                state["done"] = True
                # Only the newest operation owns the busy flag. An emergency
                # stop launched on top of this one must keep the UI locked
                # until *it* finishes, even if this one lands first.
                if token == self._op_token:
                    self._busy = False
                    self._op_name = ""
                    self._op_future = None
                    self._op_state = None

            if error is not None:
                self._log(f"'{name}' failed: {error}")
                self._set_notice(f"'{name}' failed — see the Signal Log.")
            if results:
                for rid, out in results.items():
                    self._log(f"  WS-{rid}: {self._format_result(out)}")
            if on_results is not None:
                on_results(results)

        try:
            future = launch(finish)
        except Exception as e:
            finish(error=e)
            return

        with self._lock:
            if token == self._op_token and not state["done"]:
                self._op_future = future

        if future is not None and hasattr(future, "add_done_callback"):
            future.add_done_callback(
                lambda settled: self._on_operation_settled(settled, finish)
            )

    @staticmethod
    def _on_operation_settled(future, finish):
        """Backstop: unlock when the coroutine ends without calling back.

        A no-op in the normal case — ``finish`` is idempotent and the command's
        own callback has already run by the time the future settles.
        """
        try:
            if future.cancelled():
                finish(error="cancelled")
                return
            error = future.exception()
        except Exception as e:
            finish(error=e)
            return
        finish(error=error)

    def _abort_operation(self):
        """Cancel the in-flight command and unlock the dashboard."""
        with self._lock:
            busy, name, future = self._busy, self._op_name, self._op_future

        if not busy:
            self._log("No operation is running — nothing to abort.")
            return

        if not self._confirm(f"Abort '{name}' and unlock the dashboard?"):
            self._log(f"Abort of '{name}' cancelled.")
            return

        cancelled = False
        if future is not None:
            try:
                cancelled = future.cancel()
            except Exception:
                cancelled = False

        with self._lock:
            # Silence the abandoned operation's callback and hand the busy flag
            # to a fresh token, so its late results can't unlock a later command.
            if self._op_state is not None:
                self._op_state["done"] = True
            self._op_token += 1
            self._busy = False
            self._emergency_stop_active = False
            self._client_status_pending = False
            self._op_name = ""
            self._op_future = None
            self._op_state = None
            # An abandoned preflight must not leave a queued follow-up behind:
            # aborting 'Disable preflight' cannot go on to run Disable.
            self._deferred.clear()

        self._log(
            f"Aborted '{name}'. "
            + (
                "Fleet task cancelled."
                if cancelled
                else "Fleet task was already past the point of cancellation."
            )
            + " Commands already sent to a workstation keep running there — "
            "press [R] to resync status."
        )
        self._set_notice(f"'{name}' aborted. Controls unlocked.")

    def _busy_status_line(self) -> str:
        if not self._busy:
            return ""
        elapsed = time.time() - self._op_started
        return (
            f"Running '{self._op_name or 'operation'}' ({elapsed:.0f}s)"
            " — [A] aborts and unlocks"
        )

    def _throttled_busy_log(self, message: str):
        """Log a 'still running' notice at most once every 1.5s."""
        now = time.time()
        if now - self._last_busy_notice > 1.5:
            self._last_busy_notice = now
            self._log(message)

    # ── command actions ─────────────────────────────────────────

    def _refresh_status(self, start_message: str = "Refreshing status..."):
        self._start_operation(
            "Refresh",
            lambda done: self.fleet.check_all_status(callback=lambda *_: done()),
            start_message=start_message,
            on_results=self._on_status_done,
        )

    def _on_status_done(self, *_args):
        self._announce_runtime_state()
        self._log("Status refresh complete.")
        self._refresh_client_status()

    def _broadcast_with_confirm(self, name: str, action_fn):
        """Show Y/N confirmation, then execute the broadcast command."""
        targets = self._eligible_targets("runtime", self._command_targets())
        if not targets:
            self._set_notice(f"'{name}' needs booted or online targets.")
            self._log(f"No eligible robot(s) for '{name}'.")
            return

        if not self._confirm(f"Execute '{name}' on {self._target_label(targets)}?"):
            self._log(f"{name} cancelled.")
            return

        self._run_broadcast(name, action_fn, targets)

    def _run_broadcast(self, name: str, action_fn, targets: list):
        self._start_operation(
            name,
            lambda done: action_fn(targets, callback=done),
            start_message=f"Executing '{name}' on {self._target_label(targets)}...",
            on_results=lambda _results: self._on_broadcast_done(name),
        )

    def _on_broadcast_done(self, name: str):
        self._announce_runtime_state()
        self._log(f"'{name}' complete.")

    # ── arm-motion guard ────────────────────────────────────────

    def _do_guarded_motion(
        self,
        name: str,
        action_fn,
        allow_stop: bool = True,
        skip_confirm: bool = False,
    ):
        """Run an arm-motion broadcast, but never onto a live client/follower.

        Every command routed through here ends up driving the arm — Disable
        opens with ``p goto reset``, Enable closes with ``p goto init`` — so a
        client or follower that is still running would be commanding the same
        joints from the other side. The two fight, and the arm is the thing that
        loses.

        The dashboard's cached client-running set is not good enough to gate on:
        it is refreshed on a schedule and says nothing about the follower. So
        the state is re-read from the workstations first, and only a clean
        preflight reaches the arms. When something is running the operator gets
        the choice of stopping it first, and that path re-checks (with
        ``allow_stop=False``) rather than assuming the kill worked.
        """
        targets = self._eligible_targets("runtime", self._command_targets())
        if not targets:
            self._set_notice(f"'{name}' needs booted or online targets.")
            self._log(f"No eligible robot(s) for '{name}'.")
            return

        # The preflight results reach us through `holder` rather than through
        # `done(results)`: _start_operation logs whatever it is handed, and a
        # per-robot dict on a 14-robot fleet would bury the log.
        holder: dict = {}

        def launch(done):
            def on_check(results=None):
                holder["results"] = results or {}
                done()

            return self.dispatcher.check_runtime_processes(targets, callback=on_check)

        self._start_operation(
            f"{name} preflight",
            launch,
            start_message=(
                f"Checking for running clients/followers before '{name}' on "
                f"{self._target_label(targets)}..."
            ),
            on_results=lambda _results: self._defer(
                lambda: self._resolve_motion_guard(
                    name,
                    action_fn,
                    targets,
                    holder.get("results", {}),
                    allow_stop=allow_stop,
                    skip_confirm=skip_confirm,
                )
            ),
        )

    def _resolve_motion_guard(
        self,
        name: str,
        action_fn,
        targets: list,
        results: dict,
        allow_stop: bool,
        skip_confirm: bool,
    ):
        """Decide what the preflight result means. Runs on the curses thread."""
        # The listener is the run_follower.launch.py process; the operator calls
        # it the follower, so that is what the blocked dialog names.
        labels = (("client", "client"), ("listener", "follower"))
        blocked: dict[int, list[str]] = {}
        for robot in targets:
            state = results.get(robot.id) or {}
            running = [label for key, label in labels if state.get(key)]
            if running:
                blocked[robot.id] = running

        if not blocked:
            if not skip_confirm and not self._confirm(
                f"Execute '{name}' on {self._target_label(targets)}?"
            ):
                self._log(f"{name} cancelled.")
                return
            self._run_broadcast(name, action_fn, targets)
            return

        detail = ", ".join(
            f"WS-{rid} ({'+'.join(procs)})" for rid, procs in sorted(blocked.items())
        )
        blocked_robots = self._robots_for_ids(set(blocked))

        if not allow_stop:
            # Second pass: the operator already asked for the stop and it did
            # not take. Refuse rather than offer the same button again.
            self._set_notice(f"'{name}' aborted — processes survived the stop.")
            self._log(
                f"'{name}' aborted: still running after the stop — {detail}. "
                "Nothing was sent to the arms. Try [8] Kill All."
            )
            return

        self._log(f"'{name}' blocked — still running: {detail}.")
        if self._motion_guard_menu(name, detail) != "stop":
            self._set_notice(f"'{name}' cancelled — client/follower still running.")
            self._log(f"{name} cancelled; nothing was sent to the arms.")
            return

        self._stop_then_run(name, action_fn, targets, blocked_robots)

    def _stop_then_run(self, name: str, action_fn, targets: list, blocked: list):
        """Stop the clients/followers in the way, then re-run the guard."""

        def on_results(_results=None):
            self._refresh_client_status()
            self._defer(
                lambda: self._do_guarded_motion(
                    name, action_fn, allow_stop=False, skip_confirm=True
                )
            )

        self._start_operation(
            f"Stop client/follower before {name}",
            lambda done: self.dispatcher.stop_runtime_processes(
                blocked, callback=done
            ),
            start_message=(
                "Stopping clients and followers on "
                f"{self._format_robot_ids(blocked)} before '{name}'..."
            ),
            on_results=on_results,
        )

    def _motion_guard_menu(self, name: str, detail: str) -> str:
        """Blocked-command dialog. Returns "stop" or "cancel".

        There is deliberately no "run anyway": the whole point of the guard is
        that this command must not reach an arm something else is driving.
        """
        stdscr = self._stdscr
        h, w = stdscr.getmaxyx()
        box_w = min(max(52, len(detail) + 8), w - 6)
        box_h = 9
        start_y = max(1, h // 2 - box_h // 2)
        start_x = max(2, w // 2 - box_w // 2)
        win = curses.newwin(box_h, box_w, start_y, start_x)
        win.bkgd(" ", curses.color_pair(PAIR_SURFACE))

        stdscr.nodelay(False)
        try:
            while True:
                win.erase()
                win.bkgd(" ", curses.color_pair(PAIR_SURFACE))
                self._draw_box(win, 0, 0, box_h, box_w, f"'{name}' Blocked")
                self._center_text(
                    win,
                    2,
                    1,
                    box_w - 2,
                    "Still running:",
                    curses.color_pair(PAIR_LOG),
                )
                self._center_text(
                    win,
                    3,
                    1,
                    box_w - 2,
                    detail,
                    curses.color_pair(PAIR_LOG) | curses.A_BOLD,
                )
                self._center_text(
                    win,
                    5,
                    1,
                    box_w - 2,
                    f"[K] Stop them, then {name}",
                    curses.color_pair(PAIR_NOTICE),
                )
                self._center_text(
                    win,
                    6,
                    1,
                    box_w - 2,
                    "[C] Cancel",
                    curses.color_pair(PAIR_NOTICE),
                )
                win.refresh()

                key = stdscr.getch()
                ch = chr(key).upper() if 0 <= key < 256 else ""
                if ch == "K":
                    return "stop"
                if ch in ("C", "N", "Q") or key == 27:
                    return "cancel"
        finally:
            stdscr.nodelay(True)

    def _do_boot(self):
        """Boot selected robots, or all robots when none are selected."""
        targets = self._command_targets()
        if not targets:
            self._log("No workstations configured for boot.")
            return
        if not self._confirm(f"Boot Docker on {self._target_label(targets)}?"):
            self._log("Boot cancelled.")
            return

        self._start_operation(
            "Boot",
            lambda done: self.dispatcher.boot(targets, callback=done),
            start_message=f"Booting {self._target_label(targets)}...",
            on_results=lambda _results: self._on_broadcast_done("Boot"),
        )

    def _do_shutdown(self):
        """Disable, kill tunnels, then shutdown selected/all active robots."""
        active = self._eligible_targets("shutdown", self._command_targets())
        if not active:
            self._log("No active robots to shut down.")
            return
        if not self._confirm(
            f"Disable, kill tunnel, then shutdown {self._target_label(active)}?"
        ):
            self._log("Shutdown cancelled.")
            return

        self._start_operation(
            "Shutdown",
            lambda done: self.dispatcher.shutdown(active, callback=done),
            start_message=(
                "Disabling, killing tunnels, and shutting down "
                f"{self._target_label(active)}..."
            ),
            on_results=lambda _results: self._on_broadcast_done("Shutdown"),
        )

    def _do_kill_all(self):
        """Force-stop clients, listeners, webcams, tunnels, and Docker.

        No eligibility filter: the point is recovering workstations whose
        reported status no longer matches reality, so every configured target
        gets the full teardown and reports per-step results.
        """
        targets = self._command_targets()
        if not targets:
            self._log("No workstations configured for kill all.")
            return
        if not self._confirm(
            f"Kill ALL processes and Docker on {self._target_label(targets)}?"
        ):
            self._log("Kill All cancelled.")
            return

        def on_results(_results=None):
            with self._lock:
                self._client_running_robot_ids.difference_update(
                    robot.id for robot in targets
                )
                self._apply_client_robot_status(
                    self._client_running_robot_ids,
                    checked_ids={robot.id for robot in targets},
                )
            self._on_broadcast_done("Kill All")

        self._start_operation(
            "Kill All",
            lambda done: self.dispatcher.kill_all(targets, callback=done),
            start_message=(
                "Killing clients, listeners, webcams, tunnels, and Docker on "
                f"{self._target_label(targets)}..."
            ),
            on_results=on_results,
        )

    def _do_start_tunnel(self):
        """Start SSH tunnels on selected robots, or all when none are selected."""
        targets = self._command_targets()
        if not targets:
            self._log("No workstations configured for tunnel start.")
            return
        if not self._confirm(f"Start SSH tunnel on {self._target_label(targets)}?"):
            self._log("Start Tunnel cancelled.")
            return

        self._start_operation(
            "Start Tunnel",
            lambda done: self.dispatcher.start_tunnel(targets, callback=done),
            start_message=f"Starting SSH tunnel on {self._target_label(targets)}...",
            on_results=lambda _results: self._log("Start Tunnel complete."),
        )

    def _do_kill_tunnel(self):
        """Kill SSH tunnels on selected robots, or all when none are selected."""
        targets = self._command_targets()
        if not targets:
            self._log("No workstations configured for tunnel kill.")
            return
        if not self._confirm(f"Kill SSH tunnel on {self._target_label(targets)}?"):
            self._log("Kill Tunnel cancelled.")
            return

        self._start_operation(
            "Kill Tunnel",
            lambda done: self.dispatcher.kill_tunnel(targets, callback=done),
            start_message=f"Killing SSH tunnel on {self._target_label(targets)}...",
            on_results=lambda _results: self._log("Kill Tunnel complete."),
        )

    def _open_process_menu(
        self,
        label: str,
        start_fn,
        stop_fn,
        process_key: str | None = None,
    ):
        """Show a compact start/stop menu for a runtime process."""
        targets = self._eligible_targets("runtime", self._command_targets())
        if not targets:
            self._set_notice(f"{label} requires booted or online targets.")
            self._log(f"No booted robots for '{label}'.")
            return

        stdscr = self._stdscr
        h, w = stdscr.getmaxyx()
        box_w = min(56, w - 6)
        box_h = 8
        start_y = max(1, h // 2 - box_h // 2)
        start_x = max(2, w // 2 - box_w // 2)
        win = curses.newwin(box_h, box_w, start_y, start_x)
        win.bkgd(" ", curses.color_pair(PAIR_SURFACE))

        stdscr.nodelay(False)
        try:
            while True:
                win.erase()
                win.bkgd(" ", curses.color_pair(PAIR_SURFACE))
                self._draw_box(win, 0, 0, box_h, box_w, f"{label} Control")
                self._center_text(
                    win,
                    2,
                    1,
                    box_w - 2,
                    f"Target: {self._target_label(targets)}",
                    curses.color_pair(PAIR_LOG) | curses.A_BOLD,
                )
                self._center_text(
                    win,
                    4,
                    1,
                    box_w - 2,
                    "[S] Start   [K] Stop   [Q] Cancel",
                    curses.color_pair(PAIR_NOTICE),
                )
                win.refresh()

                key = stdscr.getch()
                ch = chr(key).upper() if 0 <= key < 256 else ""
                if ch == "S":
                    self._run_targeted_process_action(
                        f"Start {label}",
                        start_fn,
                        targets,
                        f"Starting {label.lower()}",
                        process_key=process_key,
                        is_start=True,
                    )
                    return
                if ch == "K":
                    self._run_targeted_process_action(
                        f"Stop {label}",
                        stop_fn,
                        targets,
                        f"Stopping {label.lower()}",
                        process_key=process_key,
                        is_start=False,
                    )
                    return
                if ch == "Q" or key == 27:
                    self._log(f"{label} menu closed.")
                    return
        finally:
            stdscr.nodelay(True)

    def _run_targeted_process_action(
        self,
        name: str,
        action_fn,
        targets: list,
        present_participle: str,
        process_key: str | None = None,
        is_start: bool = False,
    ):
        if process_key == "client" and is_start:
            self._client_running_robot_ids.update(robot.id for robot in targets)
            self._apply_client_robot_status({robot.id for robot in targets})
            self._set_notice("Clients starting. Space is now emergency-stop.")

        def on_results(results=None):
            if process_key == "client":
                self._update_client_state_from_results(
                    targets,
                    results,
                    running=is_start,
                )
            self._log(f"{name} complete.")
            if process_key == "client":
                self._refresh_client_status()

        self._start_operation(
            name,
            lambda done: action_fn(targets, callback=done),
            start_message=f"{present_participle} on {self._target_label(targets)}...",
            on_results=on_results,
        )

    def _do_connect_to_server(self):
        booted = self._eligible_targets("connect", self._command_targets())
        if not booted:
            self._log("No booted robots to connect.")
            return
        if not self._confirm(f"Connect {self._target_label(booted)} to server?"):
            self._log("Connect to Server cancelled.")
            return

        self.dispatcher.connect_to_server(booted)
        self._announce_runtime_state()
        self._log(f"Connected {self._target_label(booted)} to server (simulated).")

    def _eligible_targets(self, action: str, targets: list) -> list:
        if action in ("boot", "start_tunnel", "kill_tunnel"):
            return targets
        if action in ("runtime", "start_listener", "start_client"):
            return [
                r
                for r in targets
                if r.status in (RobotStatus.BOOTED, RobotStatus.ONLINE)
            ]
        if action == "connect":
            return [r for r in targets if r.status == RobotStatus.BOOTED]
        if action == "shutdown":
            return [r for r in targets if r.status != RobotStatus.OFFLINE]
        return [r for r in targets if r.status != RobotStatus.OFFLINE]

    def _refresh_client_status(self):
        """Refresh the client-running set without blocking the dashboard UI."""
        if self._client_status_pending:
            return

        targets = self._eligible_targets("runtime", list(self.config.robots))
        if not targets:
            self._client_running_robot_ids.clear()
            return

        self._client_status_pending = True
        target_ids = {robot.id for robot in targets}

        def on_done(results=None):
            previous = set(self._client_running_robot_ids)
            running = {
                rid
                for rid, is_running in (results or {}).items()
                if is_running
            }
            with self._lock:
                self._client_status_pending = False
                self._client_running_robot_ids.difference_update(target_ids)
                self._client_running_robot_ids.update(running)
                self._apply_client_robot_status(running, checked_ids=target_ids)

            if running != previous:
                if running:
                    self._set_notice(
                        f"Clients running on {self._format_robot_ids(self._robots_for_ids(running))}. "
                        "Space = emergency-stop.",
                        seconds=4.0,
                    )
                    self._log(
                        "Clients detected on "
                        f"{self._format_robot_ids(self._robots_for_ids(running))}. "
                        "Space is emergency-stop."
                    )
                elif previous:
                    self._log("No running clients detected.")

        try:
            future = self.fleet.check_clients(targets, callback=on_done)
        except Exception as e:
            self._client_status_pending = False
            self._log(f"Client status check could not be submitted: {e}")
            return

        # Backstop, same reasoning as _start_operation: a check that ends
        # without calling back would otherwise leave `_client_status_pending`
        # stuck True and freeze the client-running set for the rest of the
        # session — including the set that Space's emergency stop reads.
        if future is not None and hasattr(future, "add_done_callback"):
            future.add_done_callback(lambda _f: self._clear_client_status_pending())

    def _clear_client_status_pending(self):
        with self._lock:
            self._client_status_pending = False

    def _emergency_stop_clients(self):
        """Immediately kill all known running clients; no confirmation by design."""
        if self._emergency_stop_active:
            self._log("Emergency client stop is already in progress.")
            return

        targets = self._client_running_targets()
        if not targets:
            self._client_running_robot_ids.clear()
            self._log("No running clients detected for emergency stop.")
            return

        self._emergency_stop_active = True
        self._set_notice("EMERGENCY STOP: killing clients now.", seconds=4.0)
        try:
            curses.beep()
        except curses.error:
            pass

        def on_results(_results=None):
            with self._lock:
                self._emergency_stop_active = False
                self._client_running_robot_ids.difference_update(
                    robot.id for robot in targets
                )
                self._apply_client_robot_status(
                    self._client_running_robot_ids,
                    checked_ids={robot.id for robot in targets},
                )
            self._set_notice("Emergency client stop complete.", seconds=3.0)
            self._log("Emergency client stop complete.")
            self._refresh_client_status()

        self._start_operation(
            "Emergency Stop",
            lambda done: self.dispatcher.kill_client(targets, callback=done),
            start_message=(
                f"EMERGENCY STOP: killing clients on {self._target_label(targets)}."
            ),
            on_results=on_results,
        )

    def _update_client_state_from_results(
        self,
        targets: list,
        results,
        running: bool,
    ):
        target_ids = {robot.id for robot in targets}
        if not results:
            if running:
                self._client_running_robot_ids.update(target_ids)
                self._apply_client_robot_status(target_ids)
            else:
                self._client_running_robot_ids.difference_update(target_ids)
                self._apply_client_robot_status(
                    self._client_running_robot_ids,
                    checked_ids=target_ids,
                )
            return

        successful = {
            rid
            for rid, out in results.items()
            if not isinstance(out, Exception)
            and not str(out).startswith("ERROR:")
        }
        failed = target_ids - successful
        if running:
            self._client_running_robot_ids.update(successful)
            self._client_running_robot_ids.difference_update(failed)
            self._apply_client_robot_status(successful)
            self._apply_client_robot_status(
                self._client_running_robot_ids,
                checked_ids=failed,
            )
            if successful:
                self._set_notice("Clients running. Space = emergency-stop.")
        else:
            self._client_running_robot_ids.difference_update(successful)
            self._apply_client_robot_status(
                self._client_running_robot_ids,
                checked_ids=successful,
            )

    def _apply_client_robot_status(
        self,
        running_ids: set[int],
        checked_ids: set[int] | None = None,
    ):
        """Reflect client process state in the dashboard robot status."""
        for robot in self.config.robots:
            if robot.id in running_ids:
                robot.status = RobotStatus.ONLINE
            elif checked_ids is not None and robot.id in checked_ids:
                if robot.status == RobotStatus.ONLINE:
                    robot.status = RobotStatus.BOOTED

    # ── confirmation dialog ─────────────────────────────────────

    def _confirm(self, message: str) -> bool:
        """Show a blocking Y/N confirmation prompt."""
        stdscr = self._stdscr
        h, w = stdscr.getmaxyx()
        box_w = min(max(40, len(message) + 8), w - 6)
        box_h = 7
        start_y = max(1, h // 2 - box_h // 2)
        start_x = max(2, w // 2 - box_w // 2)

        win = curses.newwin(box_h, box_w, start_y, start_x)
        win.bkgd(" ", curses.color_pair(PAIR_SURFACE))
        self._draw_box(win, 0, 0, box_h, box_w, "Confirm Action")
        self._center_text(
            win,
            2,
            1,
            box_w - 2,
            message,
            curses.color_pair(PAIR_LOG) | curses.A_BOLD,
        )
        self._center_text(
            win,
            4,
            1,
            box_w - 2,
            "[Y] Confirm   [N] Cancel",
            curses.color_pair(PAIR_NOTICE),
        )
        win.refresh()

        stdscr.nodelay(False)
        while True:
            ch = stdscr.getch()
            if ch in (ord("y"), ord("Y")):
                stdscr.nodelay(True)
                return True
            if ch in (ord("n"), ord("N"), 27):  # 27 = ESC
                stdscr.nodelay(True)
                return False

    # ── rendering ───────────────────────────────────────────────

    def _render(self):
        stdscr = self._stdscr
        try:
            h, w = stdscr.getmaxyx()
        except Exception:
            return

        if h < MIN_HEIGHT or w < MIN_WIDTH:
            stdscr.erase()
            self._center_text(
                stdscr,
                h // 2,
                0,
                w,
                f"Resize to at least {MIN_WIDTH}x{MIN_HEIGHT} for the polished ARMory layout.",
                curses.color_pair(PAIR_NOTICE) | curses.A_BOLD,
            )
            stdscr.refresh()
            return

        stdscr.erase()
        self._clamp_robot_focus()

        content_w = min(w - 2, 124)
        content_h = min(h - 1, 34)
        origin_x = max(0, (w - content_w) // 2)
        origin_y = max(0, (h - content_h) // 2)

        header_h = 2
        gap = 1
        desired_log_h = max(7, min(10, content_h // 3))
        # 14 = header rows + a four-row core grid + a three-row runtime grid;
        # the log pane gives up rows first so no core command falls off.
        min_body_h = 14 if self._runtime_controls_unlocked() else 10
        log_h = min(desired_log_h, content_h - header_h - (gap * 2) - min_body_h)
        body_h = content_h - header_h - log_h - (gap * 2)

        side_w = min(44, max(36, content_w // 3))
        right_w = content_w - side_w - gap

        self._draw_header(origin_y, origin_x, content_w)
        self._draw_side_panel(origin_y + header_h + gap, origin_x, body_h, side_w)

        right_x = origin_x + side_w + gap
        right_y = origin_y + header_h + gap
        if self._runtime_controls_unlocked():
            fleet_h = min(7, body_h - gap - 6)
            fleet_h = max(5, fleet_h)
            runtime_h = body_h - fleet_h - gap
            self._draw_core_panel(right_y, right_x, fleet_h, right_w)
            self._draw_runtime_panel(right_y + fleet_h + gap, right_x, runtime_h, right_w)
        else:
            self._draw_core_panel(right_y, right_x, body_h, right_w, show_notice=True)

        log_y = right_y + body_h + gap
        self._draw_log_panel(log_y, origin_x, log_h, content_w)

        stdscr.refresh()

    def _draw_header(self, y: int, x: int, w: int):
        stdscr = self._stdscr
        title = "ARMory"
        self._center_text(
            stdscr,
            y,
            x,
            w,
            title,
            curses.color_pair(PAIR_HIGHLIGHT) | curses.A_BOLD,
        )

        clients_running = self._clients_running()
        badge = (
            "CLIENTS LIVE"
            if clients_running
            else "SYNCING"
            if self._busy
            else "RUNTIME READY"
            if self._runtime_controls_unlocked()
            else f"BOOT {self._ready_robot_count()}/{len(self.config.robots)}"
        )
        badge_pair = (
            PAIR_OFFLINE
            if clients_running
            else PAIR_BOOTED
            if self._busy
            else PAIR_ONLINE
            if self._runtime_controls_unlocked()
            else PAIR_NOTICE
        )
        badge_attr = curses.color_pair(badge_pair) | curses.A_BOLD
        self._right_text(stdscr, y, x, w, badge, badge_attr)

        notice = self._active_notice()
        notice_attr = curses.color_pair(PAIR_OFFLINE) | curses.A_BOLD
        if not notice and clients_running:
            notice = (
                f"Clients running on {self._format_robot_ids(self._client_running_targets())}. "
                "SPACE = EMERGENCY STOP"
            )
        if not notice:
            # A running command is ordinary, not an alarm — and the elapsed
            # counter is what tells the operator the console is still alive
            # rather than wedged.
            busy_line = self._busy_status_line()
            if busy_line:
                notice = busy_line
                notice_attr = curses.color_pair(PAIR_NOTICE) | curses.A_BOLD
        if notice:
            self._center_text(stdscr, y + 1, x, w, notice, notice_attr)
        else:
            offline, booted, online = self._status_counts()
            segments = [
                ("target ", curses.color_pair(PAIR_MUTED)),
                (self._target_label(), curses.color_pair(PAIR_HIGHLIGHT) | curses.A_BOLD),
                ("   offline ", curses.color_pair(PAIR_MUTED)),
                (str(offline), curses.color_pair(PAIR_OFFLINE) | curses.A_BOLD),
                ("   booted ", curses.color_pair(PAIR_MUTED)),
                (str(booted), curses.color_pair(PAIR_BOOTED) | curses.A_BOLD),
                ("   online ", curses.color_pair(PAIR_MUTED)),
                (str(online), curses.color_pair(PAIR_ONLINE) | curses.A_BOLD),
            ]
            self._draw_centered_segments(stdscr, y + 1, x, w, segments)

    def _draw_side_panel(self, y: int, x: int, h: int, w: int):
        stdscr = self._stdscr
        self._draw_box(
            stdscr, y, x, h, w, "Fleet Matrix", active=self._active_pane == PANE_FLEET
        )

        total = len(self.config.robots)
        ready = self._ready_robot_count()
        selected = len(self._selected_robot_ids)
        self._center_text(
            stdscr,
            y + 1,
            x + 1,
            w - 2,
            f"{ready}/{total} ready | {selected} selected",
            curses.color_pair(PAIR_NOTICE),
        )

        show_ip = h >= (len(self.config.robots) * 2 + 5)
        row = y + 3
        for idx, robot in enumerate(self.config.robots):
            if row >= y + h - 2:
                break

            focused = idx == self._focused_robot_idx
            selected = robot.id in self._selected_robot_ids
            status_text = robot.status.value.upper()
            pair = self._status_pair(robot.status)
            focus_marker = ">" if focused else " "
            select_marker = "*" if selected else " "
            label = f"{focus_marker}{select_marker} WS-{robot.id}  {robot.name}"
            status_x = x + w - len(status_text) - 3
            label_w = max(0, status_x - (x + 2) - 1)
            row_attr = curses.A_REVERSE if focused else 0
            label_pair = PAIR_HIGHLIGHT if selected else PAIR_CMD

            self._add_text(
                stdscr,
                row,
                x + 2,
                label[:label_w],
                curses.color_pair(label_pair) | curses.A_BOLD | row_attr,
            )
            self._add_text(
                stdscr,
                row,
                status_x,
                status_text,
                curses.color_pair(pair) | curses.A_BOLD | row_attr,
            )
            row += 1

            if show_ip and row < y + h - 2:
                self._add_text(
                    stdscr,
                    row,
                    x + 2,
                    robot.ip[: w - 4],
                    curses.color_pair(PAIR_MUTED) | row_attr,
                )
                row += 1

        if h >= 8:
            if self._clients_running():
                footer_pair = PAIR_OFFLINE
                choices = ["SPACE = KILL CLIENTS NOW", "SPACE = KILL CLIENTS"]
            else:
                footer_pair = PAIR_MUTED
                choices = [
                    "Up/Down focus | Space selects | Tab = Signal Log",
                    "Up/Down focus | Space select | Tab = Log",
                    "Up/Down | Space | Tab = Log",
                    "Tab = Log",
                ]
            footer = self._fit_text(choices, w - 2)
            self._center_text(
                stdscr,
                y + h - 2,
                x + 1,
                w - 2,
                footer,
                curses.color_pair(footer_pair) | curses.A_BOLD,
            )

    def _draw_core_panel(self, y: int, x: int, h: int, w: int, show_notice: bool = False):
        stdscr = self._stdscr
        self._draw_box(stdscr, y, x, h, w, f"Core Controls [{self._target_label()}]")
        self._draw_command_grid(y + 2, x + 2, w - 4, h - 3, CORE_COMMANDS)

        if not show_notice or h < 8:
            return

        targets = self._command_targets()
        ready = len(self._eligible_targets("runtime", targets))
        total = len(targets)
        self._center_text(
            stdscr,
            y + h - 3,
            x + 1,
            w - 2,
            "Runtime controls appear once the current target is fully booted.",
            curses.color_pair(PAIR_MUTED),
        )
        self._draw_centered_segments(
            stdscr,
            y + h - 2,
            x + 1,
            w - 2,
            [
                ("ready ", curses.color_pair(PAIR_MUTED)),
                (f"{ready}/{total}", curses.color_pair(PAIR_HIGHLIGHT) | curses.A_BOLD),
                ("  press [6] to boot target robots", curses.color_pair(PAIR_NOTICE)),
            ],
        )

    def _draw_runtime_panel(self, y: int, x: int, h: int, w: int):
        stdscr = self._stdscr
        self._draw_box(stdscr, y, x, h, w, f"Runtime Controls [{self._target_label()}]")
        self._draw_command_grid(y + 2, x + 2, w - 4, h - 3, RUNTIME_COMMANDS)

    def _draw_command_grid(
        self,
        y: int,
        x: int,
        w: int,
        max_rows: int,
        commands: list[tuple[str, str]],
    ):
        stdscr = self._stdscr
        cols = 2 if w >= 30 and len(commands) > 1 else 1
        gap = 3
        col_w = max(12, (w - (gap * (cols - 1))) // cols)

        for idx, (key, label) in enumerate(commands):
            row = y + (idx // cols)
            if row >= y + max_rows:
                break
            col = idx % cols
            cell_x = x + col * (col_w + gap)

            self._add_text(stdscr, row, cell_x, "[", curses.color_pair(PAIR_CMD))
            self._add_text(
                stdscr,
                row,
                cell_x + 1,
                key,
                curses.color_pair(PAIR_HIGHLIGHT) | curses.A_BOLD,
            )
            self._add_text(stdscr, row, cell_x + 1 + len(key), "] ", curses.color_pair(PAIR_CMD))
            text_x = cell_x + len(key) + 3
            label_w = max(0, col_w - (text_x - cell_x))
            self._add_text(
                stdscr,
                row,
                text_x,
                label[:label_w],
                curses.color_pair(PAIR_CMD),
            )

    def _draw_log_panel(self, y: int, x: int, h: int, w: int):
        stdscr = self._stdscr
        active = self._active_pane == PANE_LOG

        capacity = max(1, h - 2)
        text_w = max(1, w - 4)
        rows: list[str] = []
        for line in self.log_lines:
            rows.extend(self._wrap_log_line(line, text_w))

        # Keep the view anchored to the same content when entries arrive while
        # the user is scrolled up; snap back to the tail once they scroll down.
        if self._log_scroll > 0:
            self._log_scroll += max(0, len(rows) - self._log_rows_total)
        self._log_rows_total = len(rows)
        self._log_capacity = capacity
        self._log_scroll = max(0, min(self._log_scroll, self._max_log_scroll()))

        title = "Signal Log"
        if self._log_scroll > 0:
            title = f"Signal Log — PAUSED ({self._log_scroll} newer)"
        self._draw_box(stdscr, y, x, h, w, title, active=active)

        if not self.log_lines:
            self._center_text(
                stdscr,
                y + 2,
                x + 1,
                w - 2,
                "No activity yet.",
                curses.color_pair(PAIR_MUTED),
            )
            return

        end = len(rows) - self._log_scroll
        visible = rows[max(0, end - capacity) : end]

        row = y + 1
        for line in visible:
            if row >= y + h - 1:
                break
            self._add_text(
                stdscr,
                row,
                x + 2,
                line,
                curses.color_pair(PAIR_LOG),
            )
            row += 1

        if active and h >= 4:
            self._draw_border_hint(
                y + h - 1,
                x,
                w,
                [
                    "Up/Down scroll · PgUp/PgDn page · End = live · Tab = Fleet Matrix",
                    "Up/Down scroll · End = live · Tab = Fleet",
                    "End = live · Tab = Fleet",
                ],
            )

    def _draw_border_hint(self, y: int, x: int, w: int, hints: list[str]):
        """Right-align the longest hint that fits on a panel's bottom border."""
        for hint in hints:
            label = f" {hint} "
            if len(label) + 4 <= w:
                self._add_text(
                    self._stdscr,
                    y,
                    x + w - len(label) - 2,
                    label,
                    curses.color_pair(PAIR_MUTED),
                )
                return

    @staticmethod
    def _fit_text(choices: list[str], width: int) -> str:
        """Return the first choice that fits in `width`, else the shortest, clipped."""
        for text in choices:
            if len(text) <= width:
                return text
        return choices[-1][: max(0, width)]

    @staticmethod
    def _wrap_log_line(line: str, width: int) -> list[str]:
        """Wrap one log entry, hanging-indenting continuations under the message."""
        # Align continuations past the "[HH:MM:SS] " stamp, but never so far that
        # a narrow panel leaves no room for text.
        indent = " " * min(11, max(0, width - 8))
        wrapped = textwrap.wrap(
            line,
            width=width,
            subsequent_indent=indent,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        if len(wrapped) > MAX_LOG_WRAP_ROWS:
            wrapped = wrapped[:MAX_LOG_WRAP_ROWS]
            wrapped[-1] = wrapped[-1][: max(1, width - 1)].rstrip() + "…"
        return wrapped

    # ── helpers ─────────────────────────────────────────────────

    def _runtime_controls_unlocked(self) -> bool:
        targets = self._command_targets()
        return (
            bool(targets)
            and len(self._eligible_targets("runtime", targets)) == len(targets)
        )

    def _clamp_robot_focus(self):
        if not self.config.robots:
            self._focused_robot_idx = 0
            self._selected_robot_ids.clear()
            return

        self._focused_robot_idx = max(
            0,
            min(self._focused_robot_idx, len(self.config.robots) - 1),
        )
        valid_ids = {robot.id for robot in self.config.robots}
        self._selected_robot_ids.intersection_update(valid_ids)

    def _move_robot_focus(self, delta: int):
        self._clamp_robot_focus()
        if not self.config.robots:
            return
        self._focused_robot_idx = (self._focused_robot_idx + delta) % len(self.config.robots)

    # ── pane focus & log scrolling ──────────────────────────────

    def _toggle_pane(self):
        if self._active_pane == PANE_FLEET:
            self._active_pane = PANE_LOG
        else:
            self._active_pane = PANE_FLEET
            self._log_scroll = 0

    def _handle_log_scroll(self, key: int, ch: str) -> bool:
        """Apply a scroll key to the log. Returns True if the key was consumed."""
        page = max(1, self._log_capacity - 1)
        if key == curses.KEY_UP or ch == "K":
            delta = 1
        elif key == curses.KEY_DOWN or ch == "J":
            delta = -1
        elif key == curses.KEY_PPAGE:
            delta = page
        elif key == curses.KEY_NPAGE:
            delta = -page
        elif key == curses.KEY_HOME:
            delta = self._max_log_scroll()
        elif key == curses.KEY_END:
            delta = -self._log_scroll
        else:
            return False

        self._log_scroll = max(0, min(self._log_scroll + delta, self._max_log_scroll()))
        return True

    def _max_log_scroll(self) -> int:
        return max(0, self._log_rows_total - self._log_capacity)

    def _toggle_focused_robot(self):
        robot = self._focused_robot()
        if robot is None:
            self._log("No robot available to select.")
            return

        if robot.id in self._selected_robot_ids:
            self._selected_robot_ids.remove(robot.id)
            self._runtime_unlocked_announced = False
            self._log(f"Unselected WS-{robot.id}.")
            return

        selected = [
            r
            for r in self.config.robots
            if r.id in self._selected_robot_ids
        ]
        if selected and any(self._is_ready(r) != self._is_ready(robot) for r in selected):
            self._set_notice("Selection blocked: do not mix booted/online and offline robots.")
            self._log("Selection blocked: mixed ready/offline robot sets are not permitted.")
            try:
                curses.beep()
            except curses.error:
                pass
            return

        self._selected_robot_ids.add(robot.id)
        self._runtime_unlocked_announced = False
        self._log(f"Selected WS-{robot.id}.")

    def _focused_robot(self):
        self._clamp_robot_focus()
        if not self.config.robots:
            return None
        return self.config.robots[self._focused_robot_idx]

    def _command_targets(self) -> list:
        self._clamp_robot_focus()
        if self._selected_robot_ids:
            return [
                robot
                for robot in self.config.robots
                if robot.id in self._selected_robot_ids
            ]
        return list(self.config.robots)

    def _clients_running(self) -> bool:
        valid_ids = {robot.id for robot in self.config.robots}
        self._client_running_robot_ids.intersection_update(valid_ids)
        return bool(self._client_running_robot_ids)

    def _client_running_targets(self) -> list:
        return self._robots_for_ids(self._client_running_robot_ids)

    def _robots_for_ids(self, robot_ids: set[int]) -> list:
        return [
            robot
            for robot in self.config.robots
            if robot.id in robot_ids
        ]

    def _target_label(self, robots: list | None = None) -> str:
        if robots is not None:
            return self._format_robot_ids(robots)
        if self._selected_robot_ids:
            return self._format_robot_ids(self._command_targets())
        return "ALL"

    def _set_notice(self, message: str, seconds: float = 3.5):
        self._notice_message = message
        self._notice_until = time.time() + seconds

    def _active_notice(self) -> str:
        if self._notice_message and time.time() < self._notice_until:
            return self._notice_message
        self._notice_message = ""
        return ""

    @staticmethod
    def _is_ready(robot) -> bool:
        return robot.status in (RobotStatus.BOOTED, RobotStatus.ONLINE)

    @staticmethod
    def _format_robot_ids(robots: list) -> str:
        if not robots:
            return "none"
        labels = [f"WS-{robot.id}" for robot in robots]
        if len(labels) <= 5:
            return ", ".join(labels)
        return f"{', '.join(labels[:5])}, +{len(labels) - 5} more"

    def _ready_robot_count(self) -> int:
        return sum(
            robot.status in (RobotStatus.BOOTED, RobotStatus.ONLINE)
            for robot in self.config.robots
        )

    def _status_counts(self) -> tuple[int, int, int]:
        offline = sum(robot.status == RobotStatus.OFFLINE for robot in self.config.robots)
        booted = sum(robot.status == RobotStatus.BOOTED for robot in self.config.robots)
        online = sum(robot.status == RobotStatus.ONLINE for robot in self.config.robots)
        return offline, booted, online

    def _announce_runtime_state(self):
        unlocked = self._runtime_controls_unlocked()
        if unlocked and not self._runtime_unlocked_announced:
            self._runtime_unlocked_announced = True
            self._log(f"{self._target_label()} is ready. Runtime controls unlocked.")
        elif not unlocked:
            self._runtime_unlocked_announced = False

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {msg}")
        if self.fleet is not None:
            self.fleet.system_logger.info(msg)

    @staticmethod
    def _format_result(out) -> str:
        """Flatten a command result into a single loggable line."""
        text = " ".join(str(out).split())
        if len(text) > MAX_RESULT_CHARS:
            text = text[: MAX_RESULT_CHARS - 1] + "…"
        return text

    def _draw_box(
        self,
        target: curses.window,
        y: int,
        x: int,
        h: int,
        w: int,
        title: str,
        active: bool = False,
    ):
        if h < 3 or w < 6:
            return

        border_attr = curses.color_pair(PAIR_HIGHLIGHT if active else PAIR_BORDER)
        if active:
            border_attr |= curses.A_BOLD
            tl, tr, bl, br, horiz, vert = "╔", "╗", "╚", "╝", "═", "║"
        else:
            tl, tr, bl, br, horiz, vert = "╭", "╮", "╰", "╯", "─", "│"

        self._add_text(target, y, x, f"{tl}{horiz * (w - 2)}{tr}", border_attr)
        for row in range(y + 1, y + h - 1):
            self._add_text(target, row, x, vert, border_attr)
            self._add_text(target, row, x + w - 1, vert, border_attr)
        self._add_text(target, y + h - 1, x, f"{bl}{horiz * (w - 2)}{br}", border_attr)

        label = f" {title} "
        label_x = x + max(2, min(w - len(label) - 2, 3))
        self._add_text(
            target,
            y,
            label_x,
            label[: max(0, w - 4)],
            border_attr | curses.A_BOLD,
        )

    def _draw_centered_segments(
        self,
        target: curses.window,
        y: int,
        x: int,
        w: int,
        segments: list[tuple[str, int]],
    ):
        total = sum(len(text) for text, _ in segments)
        cursor = x + max(0, (w - total) // 2)
        for text, attr in segments:
            self._add_text(target, y, cursor, text, attr)
            cursor += len(text)

    def _center_text(
        self,
        target: curses.window,
        y: int,
        x: int,
        w: int,
        text: str,
        attr: int = 0,
    ):
        if w <= 0:
            return
        clipped = text[:w]
        start_x = x + max(0, (w - len(clipped)) // 2)
        self._add_text(target, y, start_x, clipped, attr)

    def _right_text(
        self,
        target: curses.window,
        y: int,
        x: int,
        w: int,
        text: str,
        attr: int = 0,
    ):
        start_x = x + max(0, w - len(text))
        self._add_text(target, y, start_x, text[:w], attr)

    def _status_pair(self, status: RobotStatus) -> int:
        if status == RobotStatus.ONLINE:
            return PAIR_ONLINE
        if status == RobotStatus.BOOTED:
            return PAIR_BOOTED
        return PAIR_OFFLINE

    @staticmethod
    def _add_text(target: curses.window, y: int, x: int, text: str, attr: int = 0):
        if not text:
            return
        try:
            target.addstr(y, x, text, attr)
        except curses.error:
            pass

    @staticmethod
    def _init_colors():
        curses.start_color()
        curses.use_default_colors()

        if curses.COLORS >= 256:
            curses.init_pair(PAIR_HEADER, 16, 225)
            curses.init_pair(PAIR_OFFLINE, 210, -1)
            curses.init_pair(PAIR_BOOTED, 223, -1)
            curses.init_pair(PAIR_ONLINE, 151, -1)
            curses.init_pair(PAIR_BORDER, 183, -1)
            curses.init_pair(PAIR_CMD, 225, -1)
            curses.init_pair(PAIR_LOG, 252, -1)
            curses.init_pair(PAIR_HIGHLIGHT, 219, -1)
            curses.init_pair(PAIR_MUTED, 246, -1)
            curses.init_pair(PAIR_NOTICE, 117, -1)
            curses.init_pair(PAIR_SURFACE, 255, 236)
            return

        curses.init_pair(PAIR_HEADER, curses.COLOR_BLACK, curses.COLOR_MAGENTA)
        curses.init_pair(PAIR_OFFLINE, curses.COLOR_RED, -1)
        curses.init_pair(PAIR_BOOTED, curses.COLOR_YELLOW, -1)
        curses.init_pair(PAIR_ONLINE, curses.COLOR_GREEN, -1)
        curses.init_pair(PAIR_BORDER, curses.COLOR_MAGENTA, -1)
        curses.init_pair(PAIR_CMD, curses.COLOR_WHITE, -1)
        curses.init_pair(PAIR_LOG, curses.COLOR_WHITE, -1)
        curses.init_pair(PAIR_HIGHLIGHT, curses.COLOR_CYAN, -1)
        curses.init_pair(PAIR_MUTED, curses.COLOR_WHITE, -1)
        curses.init_pair(PAIR_NOTICE, curses.COLOR_CYAN, -1)
        curses.init_pair(PAIR_SURFACE, curses.COLOR_WHITE, curses.COLOR_BLACK)
