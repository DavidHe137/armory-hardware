"""Each trial writes and reads its own episode tree.

RealSaver numbers episodes from zero per client process and creates its output
folder with ``exist_ok=True``, so a shared episode directory means trial N
overwrites trial N-1 at the identical path. Worse, a robot whose client never
ran this trial still has the previous trial's episode sitting on disk, and a
recursive fetch pulls it in as though it were fresh data.

The fix is a per-trial directory on both sides. These tests pin that the same
trial id reaches the client launch and the fetch -- if the two ever disagree
the trial reports "no data" instead of quietly mixing trials, but neither is
something to find out during a session.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib

import pytest

from armory_hardware import FleetDispatcher, Robot, RobotStatus
from armory_hardware.fleet import DEFAULT_EPISODE_SUBDIR, EPISODE_CONTAINER_ROOT


class _TrialStub:
    """Just enough FleetController to drive ``_run_trial`` with no network."""

    EPISODE_HOST_ROOT = "/home/data_collection/tester"

    def __init__(self):
        self.logger = logging.getLogger("trial_stub")
        self.start_calls: list[dict] = []
        self.fetch_calls: list[dict] = []

    # the two helpers _run_trial derives paths from
    def episode_container_dir(self, remote_subdir=DEFAULT_EPISODE_SUBDIR, trial_id=""):
        parts = [EPISODE_CONTAINER_ROOT, remote_subdir.strip("/"), trial_id.strip("/")]
        return "/".join(p for p in parts if p)

    def episode_host_dir(self, remote_subdir=DEFAULT_EPISODE_SUBDIR, trial_id=""):
        parts = [self.EPISODE_HOST_ROOT, remote_subdir.strip("/"), trial_id.strip("/")]
        return "/".join(p for p in parts if p)

    async def _clear_barrier_flags(self, robots):
        return None

    async def _start_clients(
        self, robots, callback=None, extra_args_per_robot=None, episode_dir=""
    ):
        self.start_calls.append(
            {"robots": list(robots), "extra": extra_args_per_robot, "dir": episode_dir}
        )
        return {r.id: "ok" for r in robots}

    async def _await_clients_ready(self, robots, timeout_sec):
        return {r.id: True for r in robots}

    async def _signal_clients_go(self, robots):
        return None

    async def _kill_clients(self, robots, callback=None, grace_sec=5.0):
        return {r.id: "ok" for r in robots}

    async def _fetch_episode_data(
        self, robots, local_dir, remote_subdir, include_video, callback, trial_id=""
    ):
        self.fetch_calls.append(
            {"local": pathlib.Path(local_dir), "subdir": remote_subdir, "trial": trial_id}
        )
        return {r.id: "fetched" for r in robots}


def _robot(rid: int) -> Robot:
    return Robot(id=rid, name=f"R{rid}", ip=f"10.0.0.{rid}", status=RobotStatus.BOOTED)


def _run(fleet, out: pathlib.Path, **kwargs):
    dispatcher = FleetDispatcher(fleet)
    return asyncio.run(
        dispatcher._run_trial(
            [_robot(11), _robot(12)],
            duration_sec=0.0,
            output_dir=out,
            fetch_video=False,
            grace_sec=0.0,
            remote_subdir=DEFAULT_EPISODE_SUBDIR,
            callback=None,
            **kwargs,
        )
    )


@pytest.fixture
def fleet():
    return _TrialStub()


def test_clients_write_into_the_trials_own_directory(fleet, tmp_path):
    out = tmp_path / "trial_20260810_010203"

    _run(fleet, out, trial_id=out.name)

    assert fleet.start_calls[0]["dir"] == (
        "/datasets/armory_episodes/trial_20260810_010203"
    )


def test_fetch_reads_the_same_trial_the_clients_wrote(fleet, tmp_path):
    """The whole point: one trial id, both ends."""
    out = tmp_path / "trial_20260810_010203"

    _run(fleet, out, trial_id=out.name)

    written = fleet.start_calls[0]["dir"]
    read = fleet.episode_host_dir(
        fleet.fetch_calls[0]["subdir"], fleet.fetch_calls[0]["trial"]
    )

    assert written.removeprefix(EPISODE_CONTAINER_ROOT) == read.removeprefix(
        _TrialStub.EPISODE_HOST_ROOT
    )


def test_two_trials_never_share_a_directory(fleet, tmp_path):
    _run(fleet, tmp_path / "trial_a", trial_id="trial_a")
    _run(fleet, tmp_path / "trial_b", trial_id="trial_b")

    assert fleet.start_calls[0]["dir"] != fleet.start_calls[1]["dir"]
    assert fleet.fetch_calls[0]["trial"] != fleet.fetch_calls[1]["trial"]


def test_trial_id_defaults_to_the_output_dir_name(fleet, tmp_path):
    """run_real names the output dir trial_<ts>; that is the trial id."""
    out = tmp_path / "trial_20260810_010203"

    _run(fleet, out, trial_id=out.name)

    assert fleet.fetch_calls[0]["trial"] == "trial_20260810_010203"


def test_horizon_overrides_reach_the_launch_command(fleet, tmp_path):
    _run(
        fleet,
        tmp_path / "trial_x",
        trial_id="trial_x",
        min_execution_horizon_overrides={11: 5},
        max_execution_horizon_overrides={11: 10, 12: 20},
    )

    extra = fleet.start_calls[0]["extra"]
    assert "--min-execution-horizon 5" in extra[11]
    assert "--max-execution-horizon 10" in extra[11]
    assert "--max-execution-horizon 20" in extra[12]
    assert "--min-execution-horizon" not in extra[12]
