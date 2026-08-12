"""Run a bounded real-robot trial and compute the same metrics as run_libero.

Workflow:
  1. Load FleetConfig (defaults to ``configs/armory-tui.yaml``).
  2. Filter to selected robots; require they're all reachable.
  3. ``FleetDispatcher.run_trial(...)`` → start clients, wait, kill (SIGINT
     with grace), SFTP each robot's RealSaver output back.
  4. Optionally copy the server's own metrics files into ``<trial>/server/``
     (``--server-metrics-dir``); the server writes them itself, so there is
     nothing to fetch over HTTP.
  5. Run ``calculate_metrics`` and ``generate_all_plots`` on the assembled
     output dir — the same offline pass run_libero.py uses for sim.

The on-disk layout RealSaver produces matches the sim Saver, so the metrics
code parses the real output without modification.

Needs an interpreter that can import ``evaluation.metrics`` for step 5 — i.e.
one with ``armory[evaluation]`` installed, which pins Python >=3.11,<3.12.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import imageio_ffmpeg
import requests
import tqdm
import tqdm.contrib.logging
import tyro
import yaml

from armory_hardware import FleetConfig, FleetController, FleetDispatcher, RobotStatus
from armory_hardware.env import load_env

logger = logging.getLogger("run_real")

# The publisher URL the workstation-side ffmpeg writes to comes from the fleet config
# (webcam.rtsp_base_url / ARMORY_WEBCAM_RTSP_BASE_URL), mirroring
# FleetController._webcam_rtsp_url. Per-robot path is ``{base}/workstation{robot.id}``.
WEBCAM_RECORDER_GRACE_SEC = 3.0


@dataclass
class Args:
    #################################################################################
    # Fleet selection
    #################################################################################
    robots: list[int] = field(default_factory=list)
    """Workstation IDs to include. Empty = every robot in the config."""

    config_path: str | None = None
    """Path to the fleet YAML. Defaults to <armory>/configs/armory-tui.yaml."""

    #################################################################################
    # Trial parameters
    #################################################################################
    duration_sec: float = 60.0
    """How long to let the clients run before killing them."""

    grace_sec: float = 5.0
    """SIGINT grace period before SIGKILL — needs to cover RealSaver flush."""

    output_dir: pathlib.Path = pathlib.Path("data/real")
    """Parent dir; this script appends ``trial_<timestamp>``."""

    fetch_video: bool = False
    """Pull MP4s back too. Adds ~50–500MB per episode."""

    save_videos: bool = False
    """Record each workstation's RTSP webcam stream to videos/workstationN.mp4
    for the duration of the trial. The workstation-side publisher must already
    be streaming (W key in the TUI). Off by default because mp4s are large."""

    debug: bool = False
    """Enable verbose logging: per-workstation SSH/Docker chatter and asyncssh
    transport events. Off by default; only the dispatcher's trial-phase
    messages and run_real's own status lines are shown."""

    remote_subdir: str = "armory_episodes"
    """Subdir under the workstation's user_data bind mount where RealSaver writes."""

    require_status: bool = True
    """If true, skip robots that aren't BOOTED before starting."""

    het_config_path: str | None = None
    """Optional YAML with per-workstation launch overrides. Supports four
    top-level sections, all optional (must have at least one):

    ``control_hz: {<station_id>: <hz>, ...}`` — forwarded as ``--control-hz``.
    ``language_index: {<station_id>: <idx>, ...}`` — looked up in
    ``configs/real_task_index.json`` (override path with --task-index-path)
    and forwarded as ``--prompt <STRING>``.
    ``min_execution_horizon: {<station_id>: <N>, ...}`` and
    ``max_execution_horizon: {<station_id>: <N>, ...}`` — forwarded as
    ``--min-execution-horizon <N>`` / ``--max-execution-horizon <N>``.

    Any other top-level key is an error: an unrecognised section would
    otherwise be dropped in silence and the trial would run homogeneous while
    looking like it applied the config."""

    task_index_path: str | None = None
    """Path to the language-index → prompt JSON. Defaults to
    <armory>/configs/real_task_index.json. Only consulted when the het
    config contains a ``language_index:`` section."""

    #################################################################################
    # Server metrics (optional)
    #################################################################################
    reset_server_metrics: bool = True
    """POST /reset before the trial so the scheduler's in-flight state doesn't
    carry over from the previous run."""

    server_host: str | None = None
    """Host serving /reset. Defaults to the fleet config's policy host
    (``ARMORY_POLICY_HOST``) — the same server the robots' tunnels point at.
    Not ``localhost``: the fleet creates its SSH forwards *on the
    workstations*, so nothing is listening on this machine's 8080 unless you
    opened your own. Use --no-reset-server-metrics to skip."""

    server_port: int = 8080

    server_metrics_dir: str | None = None
    """Path to the policy server's metrics dir — ``<its output_dir>/server``,
    which defaults to ``output/serve/server`` relative to wherever the server
    was launched. Its contents (metadata.json, batches.jsonl, events.jsonl,
    scheduler_decisions.jsonl) are copied into ``<trial>/server/``, where the
    metrics pass looks for them; without it the batch/Gantt/scheduler plots
    are skipped. Must be readable from *this* machine, so it only works when
    the server writes somewhere shared — otherwise copy the dir across
    yourself and point the metrics pass at the trial afterwards.

    There is no HTTP fetch: the old ``/save-metrics`` endpoint is gone, and
    the server now writes these files itself."""

    #################################################################################
    # Trial provenance
    #################################################################################
    control_hz: int = 30
    """Used only to estimate max_steps in runtime_metadata.json."""

    broker_type: str = "naive_async"

    max_execution_horizon: int = 20

    resize_size: int = 224


class _SuppressWSChatter(logging.Filter):
    """Drop per-workstation INFO chatter from the armory_hardware logger.

    Every per-WS line in fleet.py is emitted via ``_emit`` and starts with
    ``WS-{id}: ...``; the dispatcher's trial-phase messages start with
    ``trial: ...`` and we want to keep those. WARNINGs and ERRORs always pass
    through regardless of prefix.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > logging.INFO:
            return True
        return not record.getMessage().startswith("WS-")


def _setup_logging(debug: bool) -> None:
    """Default: clean log stream + tqdm-friendly. ``--debug``: full verbosity."""
    fmt = "%(asctime)s [%(name)s] %(message)s"
    if debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
        logging.getLogger("asyncssh").setLevel(logging.INFO)
        return
    logging.basicConfig(level=logging.INFO, format=fmt)
    logging.getLogger("armory_hardware").addFilter(_SuppressWSChatter())
    logging.getLogger("asyncssh").setLevel(logging.WARNING)


def _default_config_path() -> str:
    """Locate configs/armory-tui.yaml relative to this script."""
    here = pathlib.Path(__file__).resolve().parent
    return str(here.parent / "configs" / "armory-tui.yaml")


def _default_task_index_path() -> str:
    """Locate configs/real_task_index.json relative to this script."""
    here = pathlib.Path(__file__).resolve().parent
    return str(here.parent / "configs" / "real_task_index.json")


HET_SECTIONS = (
    "control_hz",
    "language_index",
    "min_execution_horizon",
    "max_execution_horizon",
)


def _load_het_config(path: str) -> dict[str, dict[int, int]]:
    """Parse the heterogeneous per-robot launch YAML.

    Schema (all sections optional, at least one required), keyed by station id:
      ``control_hz``, ``language_index``,
      ``min_execution_horizon``, ``max_execution_horizon``

    Returns ``{section: {station_id: value}}`` with ints on both sides, one
    entry per section in ``HET_SECTIONS`` (empty dict when absent). Language
    indices are returned as-is; resolution against real_task_index.json happens
    in ``main()``.

    Unknown sections are rejected rather than ignored. The horizon sections
    were once parsed under a different name, and because the file still had a
    ``control_hz`` section to satisfy the at-least-one check, configs whose
    entire fast/slow split lived in the horizons parsed clean and ran
    homogeneous.
    """
    raw = yaml.safe_load(pathlib.Path(path).read_text())
    if not isinstance(raw, dict):
        sys.exit(f"het config {path}: top-level must be a mapping")
    unknown = sorted(set(raw) - set(HET_SECTIONS))
    if unknown:
        sys.exit(
            f"het config {path}: unknown section(s) {', '.join(repr(s) for s in unknown)} — "
            f"known sections are {', '.join(repr(s) for s in HET_SECTIONS)}"
        )
    if not any(s in raw for s in HET_SECTIONS):
        sys.exit(
            f"het config {path}: must define at least one of "
            f"{', '.join(repr(s) for s in HET_SECTIONS)}"
        )

    def _parse_int_int(section: str) -> dict[int, int]:
        mapping = raw.get(section)
        if mapping is None:
            return {}
        if not isinstance(mapping, dict):
            sys.exit(f"het config {path}: '{section}' must be a mapping")
        out: dict[int, int] = {}
        for k, v in mapping.items():
            try:
                out[int(k)] = int(v)
            except (TypeError, ValueError):
                sys.exit(f"het config {path}: bad {section} entry {k!r}: {v!r} (need int → int)")
        return out

    return {section: _parse_int_int(section) for section in HET_SECTIONS}


def _load_task_index(path: str) -> dict[int, str]:
    """Parse configs/real_task_index.json into ``{index: prompt}``."""
    raw = json.loads(pathlib.Path(path).read_text())
    if not isinstance(raw, dict):
        sys.exit(f"task index {path}: top-level must be a JSON object")
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            sys.exit(f"task index {path}: bad entry {k!r}: {v!r}")
    return out


def _select_targets(cfg: FleetConfig, args: Args) -> list:
    if args.robots:
        wanted = set(args.robots)
        targets = [r for r in cfg.robots if r.id in wanted]
        missing = wanted - {r.id for r in targets}
        if missing:
            sys.exit(f"unknown workstation id(s) in --robots: {sorted(missing)}")
    else:
        targets = list(cfg.robots)
    return targets


def _filter_to_booted(fleet: FleetController, targets: list, timeout_sec: float = 30.0) -> list:
    """Refresh statuses; keep only booted/online robots."""
    fleet.check_all_status().result(timeout=timeout_sec)
    eligible = [r for r in targets if r.status in (RobotStatus.BOOTED, RobotStatus.ONLINE)]
    skipped = [r for r in targets if r not in eligible]
    if skipped:
        logger.warning(
            "skipping %d robot(s) not in BOOTED/ONLINE: %s",
            len(skipped),
            [f"WS-{r.id}({r.status.value})" for r in skipped],
        )
    return eligible


def _write_experiment_args(
    out: pathlib.Path,
    robots: list,
    args: Args,
    het: dict[str, dict[int, int]] | None = None,
) -> None:
    """Record what this trial was actually launched with, per robot.

    The horizons and control rate come from the het config when it sets them,
    falling back to the ``Args`` defaults otherwise — a flat provenance record
    would describe a homogeneous run no matter what the robots were told.
    """
    het = het or {}
    het_hz = het.get("control_hz") or {}
    het_min = het.get("min_execution_horizon") or {}
    het_max = het.get("max_execution_horizon") or {}
    estimated_max_steps = int(round(args.duration_sec * args.control_hz))
    data = {
        "experiment_config": {
            "task_suite_name": "real",
            "num_trials_per_task": 1,
            "max_steps": estimated_max_steps,
            "seed": 0,
            "resize_size": args.resize_size,
            "num_robots": len(robots),
            "control_hz": args.control_hz,
            "broker_type": args.broker_type,
            "execution_horizons": [
                {
                    "min": het_min.get(r.id, 0),
                    "max": het_max.get(r.id, args.max_execution_horizon),
                }
                for r in robots
            ],
        },
        "duration_sec": args.duration_sec,
        "output_dir": str(out),
        "robots": [r.name for r in robots],
        "per_robot": [
            {
                "workstation_id": r.id,
                "robot_name": r.name,
                "control_hz": het_hz.get(r.id, args.control_hz),
                "min_execution_horizon": het_min.get(r.id, 0),
                "max_execution_horizon": het_max.get(r.id, args.max_execution_horizon),
            }
            for r in robots
        ],
    }
    (out / "experiment_args.json").write_text(json.dumps(data, indent=2))


SERVER_METRICS_FILES = (
    "metadata.json",
    "batches.jsonl",
    "events.jsonl",
    "scheduler_decisions.jsonl",
)


def _copy_server_metrics(src_dir: str, out: pathlib.Path) -> None:
    """Copy the server's metrics files into ``<out>/server/``.

    That is the layout ``evaluation.metrics`` reads: ``load_server_metadata``
    and the batch/event/decision loaders all resolve ``<trial>/server/<file>``.

    Note the jsonl logs are opened ``"w"`` when the *server process* starts and
    are not truncated by ``/reset``, so back-to-back trials against one server
    each copy a cumulative log, not a per-trial slice. Splitting them means
    restarting the server between trials or slicing by timestamp after.
    """
    src = pathlib.Path(src_dir)
    if not src.is_dir():
        logger.warning(
            "server metrics dir %s not readable from this machine — skipping "
            "(batch/Gantt/scheduler plots will be absent)",
            src,
        )
        return
    dest = out / "server"
    dest.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for name in SERVER_METRICS_FILES:
        path = src / name
        if not path.is_file():
            missing.append(name)
            continue
        shutil.copy2(path, dest / name)
        copied.append(name)
    if copied:
        logger.info("copied server metrics %s -> %s", copied, dest)
    if missing:
        logger.warning("server metrics dir %s had no %s", src, missing)


def _reset_server_metrics(args: Args, host: str) -> None:
    url = f"http://{host}:{args.server_port}/reset"
    try:
        requests.post(url, timeout=5.0)
        logger.info("reset server metrics at %s", url)
    except Exception as e:
        logger.warning("could not reset server metrics: %s", e)


def _start_webcam_recorders(
    targets: list,
    out_dir: pathlib.Path,
    rtsp_base: str,
) -> list[tuple]:
    """Spawn one host-side ffmpeg per robot to record its mediamtx RTSP stream.

    The workstation-side publisher (started via the TUI ``W`` key /
    ``FleetDispatcher.start_webcam``) must already be live; these consumers
    fail fast if the path isn't being published. ``-c copy`` skips re-encoding,
    so the recorder is cheap.

    Written as *fragmented* mp4. A plain mp4 keeps its moov atom in memory and
    writes it only at the end, so a recorder that dies without finalizing
    leaves ftyp+free+mdat and an unplayable file — and this ffmpeg build
    (imageio-ffmpeg's static v7.0.2) installs SIGINT/SIGTERM handlers but does
    not act on them, so every recorder *does* die that way, on the SIGKILL at
    the end of ``_stop_webcam_recorders``. Fragmenting writes an empty moov up
    front and self-contained moof+mdat fragments at each keyframe, so the file
    on disk is playable no matter how the process exits. ``-g 15`` on the
    publisher puts a keyframe every 0.5s, bounding what a kill can lose.

    Returns ``[(robot, popen), ...]`` for ``_stop_webcam_recorders``.
    """
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    procs: list[tuple] = []
    for robot in targets:
        url = f"{rtsp_base}/workstation{robot.id}"
        mp4 = videos_dir / f"workstation{robot.id}.mp4"
        cmd = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-rtsp_transport",
            "udp",
            "-i",
            url,
            "-c",
            "copy",
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            "-y",
            str(mp4),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append((robot, proc))
        logger.info("recording WS-%d: %s -> %s", robot.id, url, mp4)
    return procs


def _request_recorder_stop(procs: list) -> None:
    """Send SIGTERM to each running ffmpeg recorder. Non-blocking.

    Best-effort only: the pinned ffmpeg build catches SIGTERM but keeps
    running, so in practice the recorders are reaped by the SIGKILL in
    ``_stop_webcam_recorders``. That is safe because the output is fragmented
    mp4 (see ``_start_webcam_recorders``) — the file is playable either way.
    Sending it here anyway bounds the recording at the trial window on any
    build that does honour the signal.

    Splitting this from ``_stop_webcam_recorders`` (idempotent on
    already-signaled procs) lets the trial-end callback signal at the kill
    instant without blocking the fleet event loop on the grace period.
    """
    for _, proc in procs:
        if proc.poll() is None:
            proc.terminate()


def _stop_webcam_recorders(
    procs: list,
    grace_sec: float = WEBCAM_RECORDER_GRACE_SEC,
) -> None:
    """SIGTERM each recorder, then SIGKILL whatever is still up after ``grace_sec``.

    The deadline is shared across all recorders on purpose: they shut down
    concurrently, so the budget is wall-clock from the SIGTERM, not per
    process. Reaching it is the normal path rather than an error — the pinned
    ffmpeg ignores both signals — and the fragmented-mp4 output means a killed
    recorder still leaves a playable file.
    """
    if not procs:
        return
    for _, proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + grace_sec
    for robot, proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            logger.warning(
                "recorder WS-%d did not stop in %.1fs, sending SIGKILL",
                robot.id,
                grace_sec,
            )
            proc.kill()
            proc.wait()
        # Expected exits: 0, ffmpeg's own 255 on signal-driven teardown, and a
        # negative code (Popen's encoding of "died on signal N") for the
        # SIGTERM/SIGKILL path that is the norm here. Anything else is a real
        # failure — most often a publisher that was never streaming, which is
        # the case worth a warning since it yields an empty file.
        if proc.returncode in (0, 255) or proc.returncode < 0:
            logger.info("recorder WS-%d finished (rc=%d)", robot.id, proc.returncode)
        else:
            logger.warning(
                "recorder WS-%d exited rc=%d (publisher likely wasn't streaming)",
                robot.id,
                proc.returncode,
            )


def _stop_clients_after_interrupt(
    fleet: FleetController,
    dispatcher: FleetDispatcher,
    targets: list,
    grace_sec: float,
    trial_future: concurrent.futures.Future | None,
) -> None:
    """SIGINT/KeyboardInterrupt path: stop Piper clients before ``fleet.stop()`` closes SSH.

    Without this, ``fut.result()`` unwinds while ``_run_trial`` is still in
    ``asyncio.sleep(duration)``, so robots never get ``kill_clients`` and the
    next ``run_real`` can wedge against still-running clients. After the kill,
    we also send the robots back to init so the next trial starts from a
    known pose rather than wherever the policy happened to leave them.
    """
    if not targets:
        return
    logger.warning(
        "interrupt received — sending client stop (SIGINT + grace) to %d robot(s)",
        len(targets),
    )
    kill_timeout = max(90.0, grace_sec * 4 + 30.0)
    try:
        fleet.kill_clients(targets, grace_sec=grace_sec).result(timeout=kill_timeout)
    except Exception as e:
        logger.warning("emergency kill_clients failed: %s", e)
    if trial_future is not None and not trial_future.done():
        trial_future.cancel()
        try:
            trial_future.result(timeout=5.0)
        except (concurrent.futures.CancelledError, concurrent.futures.TimeoutError, Exception):
            pass
    # chunx_client owns the joint topic during a trial; only safe to issue
    # ``p goto init`` once kill_clients has actually torn it down.
    try:
        logger.warning("sending %d robot(s) to init after interrupt", len(targets))
        dispatcher.goto_init(targets).result(timeout=60.0)
    except Exception as e:
        logger.warning("post-interrupt goto init failed: %s", e)


def _await_trial_with_progress(
    trial_future: concurrent.futures.Future,
    total_timeout_sec: float,
) -> object:
    """Block on ``trial_future`` with a tqdm progress line and clean log redirect.

    Uses ``logging_redirect_tqdm`` so log records appear above the progress
    line instead of overwriting it. The bar tracks elapsed time only — phase
    transitions show up as log lines from the dispatcher.
    """
    deadline = time.monotonic() + total_timeout_sec
    with tqdm.contrib.logging.logging_redirect_tqdm():
        with tqdm.tqdm(
            desc="Trial",
            bar_format="{desc}: {elapsed} elapsed",
            leave=False,
        ) as bar:
            while not trial_future.done():
                if time.monotonic() > deadline:
                    raise concurrent.futures.TimeoutError(
                        f"trial future did not complete in {total_timeout_sec:.0f}s"
                    )
                time.sleep(0.5)
                bar.refresh()
    return trial_future.result(timeout=1.0)


def main(args: Args) -> None:
    _setup_logging(args.debug)

    load_env()
    cfg = FleetConfig(args.config_path or _default_config_path())
    fleet = FleetController(cfg, logger=logging.getLogger("armory_hardware"))
    fleet.start()
    targets: list = []
    trial_future: concurrent.futures.Future | None = None
    recorders: list = []
    try:
        dispatcher = FleetDispatcher(fleet)

        targets = _select_targets(cfg, args)
        if not targets:
            sys.exit("no targets selected")

        if args.require_status:
            targets = _filter_to_booted(fleet, targets)
            if not targets:
                sys.exit("no eligible (BOOTED/ONLINE) robots — boot the fleet first")

        het: dict[str, dict[int, int]] = {section: {} for section in HET_SECTIONS}
        if args.het_config_path:
            het = _load_het_config(args.het_config_path)
        het_hz = het["control_hz"]
        het_lang = het["language_index"]
        het_min_exec = het["min_execution_horizon"]
        het_max_exec = het["max_execution_horizon"]

        ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        out = args.output_dir / f"trial_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        logger.info("output dir: %s", out.resolve())
        logger.info(
            "running trial on %d robot(s): %s",
            len(targets),
            [f"WS-{r.id}/{r.name}" for r in targets],
        )

        _write_experiment_args(out, targets, args, het)

        server_host = args.server_host or cfg.policy_host
        if args.reset_server_metrics and not server_host:
            logger.warning(
                "skipping server reset: no --server-host and no policy host in "
                "the fleet config (set ARMORY_POLICY_HOST in .env)"
            )

        # Clear the scheduler's carry-over state so this trial starts clean.
        if args.reset_server_metrics and server_host:
            _reset_server_metrics(args, server_host)

        target_ids = {r.id for r in targets}
        for section in ("control_hz", "min_execution_horizon", "max_execution_horizon"):
            mapping = het[section]
            if not mapping:
                continue
            applied = {rid: mapping[rid] for rid in mapping if rid in target_ids}
            unmatched = sorted(set(mapping) - target_ids)
            logger.info("%s overrides applied: %s", section, applied)
            if unmatched:
                logger.warning(
                    "%s config has entries for non-target ids: %s", section, unmatched
                )

        prompt_overrides: dict[int, str] = {}
        if het_lang:
            task_index = _load_task_index(args.task_index_path or _default_task_index_path())
            bad = sorted((rid, idx) for rid, idx in het_lang.items() if idx not in task_index)
            if bad:
                sys.exit(
                    f"language_index entries refer to unknown task ids "
                    f"(not in {args.task_index_path or _default_task_index_path()}): "
                    f"{bad}"
                )
            prompt_overrides = {rid: task_index[idx] for rid, idx in het_lang.items()}
            applied_lang = {rid: het_lang[rid] for rid in het_lang if rid in target_ids}
            unmatched_lang = sorted(set(het_lang) - target_ids)
            logger.info(
                "language_index overrides applied: %s -> %s",
                applied_lang,
                {rid: task_index[idx] for rid, idx in applied_lang.items()},
            )
            if unmatched_lang:
                logger.warning(
                    "language_index config has entries for non-target ids: %s",
                    unmatched_lang,
                )

        # Webcam recorders bracket the actual action-execution window:
        # _on_clients_running starts them after barrier release;
        # _on_clients_stopping SIGTERMs them right before kill_clients so
        # they don't bleed through the kill grace, settle, and fetch phases.
        def _on_clients_running() -> None:
            if not args.save_videos:
                return
            recorders.extend(
                _start_webcam_recorders(targets, out, cfg.webcam_rtsp_base_url)
            )

        def _on_clients_stopping() -> None:
            _request_recorder_stop(recorders)

        # Run the trial — start, wait, kill, fetch.
        trial_future = dispatcher.run_trial(
            targets,
            duration_sec=args.duration_sec,
            output_dir=out,
            fetch_video=args.fetch_video,
            grace_sec=args.grace_sec,
            remote_subdir=args.remote_subdir,
            control_hz_overrides=het_hz or None,
            prompt_overrides=prompt_overrides or None,
            min_execution_horizon_overrides=het_min_exec or None,
            max_execution_horizon_overrides=het_max_exec or None,
            on_clients_running=_on_clients_running,
            on_clients_stopping=_on_clients_stopping,
        )
        # Total time: trial duration + grace + fetch overhead. Add a 60s buffer.
        try:
            summary = _await_trial_with_progress(
                trial_future,
                total_timeout_sec=args.duration_sec + args.grace_sec * 2 + 120,
            )
        except KeyboardInterrupt:
            # Same bracketing as the normal path: SIGTERM recorders at the
            # moment we issue the emergency client stop, not in the finally
            # block after fleet teardown.
            _request_recorder_stop(recorders)
            _stop_clients_after_interrupt(
                fleet,
                dispatcher,
                targets,
                args.grace_sec,
                trial_future,
            )
            raise

        # Reset robots to init now that the trial is over. Safe here because
        # the dispatcher already ran kill_clients before returning, so
        # chunx_client is no longer publishing to the joint topic.
        try:
            logger.info("sending %d robot(s) to init", len(targets))
            dispatcher.goto_init(targets).result(timeout=60.0)
        except Exception as e:
            logger.warning("post-trial goto init failed: %s", e)

        logger.info("trial summary:")
        for stage, results in summary.items():
            if isinstance(results, dict):
                for rid, msg in results.items():
                    logger.info("  %s WS-%s: %s", stage, rid, str(msg)[:120])
            else:
                logger.info("  %s: %s", stage, results)

        # Collect the server's own metrics files, if they're reachable.
        if args.server_metrics_dir:
            _copy_server_metrics(args.server_metrics_dir, out)

        # Restructure: each robot's data landed under <out>/<robot.name>/<robot_idx>/...
        # but calculate_metrics expects <out>/<robot_idx>/... directly.
        # _fetch_one_robot mgets the trial's remote directory, so it writes to
        # <local_dir>/<robot_name>/<trial_id>/<robot_idx>/... — the nested level
        # is named for the leaf of the remote path, which is the trial id.
        # Flatten: hoist the per-robot trees so the metrics pass sees the layout it expects.
        _flatten_fetched_layout(out, out.name)

        # Finally: same offline metrics pass as run_libero.py.
        try:
            from evaluation.metrics import calculate_metrics, generate_all_plots

            calculate_metrics(out)
            generate_all_plots(out)
            logger.info("metrics + plots written to %s", out)
        except Exception as e:
            logger.warning("metrics/plot pass failed (data is still on disk): %s", e)

        print(f"trial complete: {out.resolve()}")
    finally:
        _stop_webcam_recorders(recorders)
        fleet.stop()


def _flatten_fetched_layout(out: pathlib.Path, nested_name: str) -> None:
    """Lift per-robot episode trees so calculate_metrics sees <out>/<robot_idx>/.

    fetch lands data at <out>/<robot.name>/<nested_name>/<robot_idx>/<episode>/,
    where <nested_name> is the leaf of the remote path SFTP copied — the trial id.
    Move <robot_idx> up so the layout matches the sim Saver: <out>/<robot_idx>/<episode>/.
    """
    for robot_dir in list(out.iterdir()):
        if not robot_dir.is_dir():
            continue
        # Skip already-flattened or non-fetched dirs (e.g. server_metrics dir).
        nested_root = robot_dir / nested_name.lstrip("/")
        if not nested_root.is_dir():
            continue
        for robot_idx_dir in list(nested_root.iterdir()):
            if not robot_idx_dir.is_dir():
                continue
            target = out / robot_idx_dir.name
            if target.exists():
                logger.warning(
                    "flatten: %s already exists, merging episodes from %s",
                    target,
                    robot_idx_dir,
                )
                for ep in robot_idx_dir.iterdir():
                    dest = target / ep.name
                    if dest.exists():
                        continue
                    ep.rename(dest)
                robot_idx_dir.rmdir()
            else:
                robot_idx_dir.rename(target)
        # Clean up empty intermediates.
        try:
            nested_root.rmdir()
            robot_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main(tyro.cli(Args))
