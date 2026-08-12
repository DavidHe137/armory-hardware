"""The het config → launch flag path, end to end but hermetic.

Every hop here has silently dropped overrides before. The config parser ignored
sections it did not recognise, and the flags it produced were then handed to an
argparse layer that runs ``parse_known_args`` and discards what it does not
define. Both failures produce a trial that runs, saves data, and is homogeneous.
So these tests assert on the exact flag names the ROS node defines, not on
"an override was applied".
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

from armory_hardware import FleetDispatcher

# The flags client_node_armory's argparse actually defines. Changing one of
# these without changing the node means the override stops arriving, in silence.
NODE_FLAGS = frozenset(
    {
        "--control-hz",
        "--prompt",
        "--min-execution-horizon",
        "--max-execution-horizon",
        "--barrier",
    }
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def run_real():
    """Import scripts/run_real.py by path -- scripts/ is not a package."""
    pytest.importorskip("tyro", reason="run_real needs the 'trial' extra")
    pytest.importorskip("imageio_ffmpeg", reason="run_real needs the 'trial' extra")
    spec = importlib.util.spec_from_file_location(
        "run_real", REPO_ROOT / "scripts" / "run_real.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves its annotations through
    # sys.modules[cls.__module__], which is None for a module that only exists
    # as a local variable.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(tmp_path, body: str) -> str:
    path = tmp_path / "het.yaml"
    path.write_text(body)
    return str(path)


# ── parsing ──────────────────────────────────────────────────────


def test_min_max_horizons_are_parsed(run_real, tmp_path):
    het = run_real._load_het_config(
        _write(
            tmp_path,
            "control_hz:\n  11: 30\nmin_execution_horizon:\n  11: 5\n"
            "max_execution_horizon:\n  11: 10\n  12: 20\n",
        )
    )

    assert het["control_hz"] == {11: 30}
    assert het["min_execution_horizon"] == {11: 5}
    assert het["max_execution_horizon"] == {11: 10, 12: 20}


def test_unknown_section_is_rejected(run_real, tmp_path):
    """The regression that made 1f9s and 5f5s run homogeneous.

    ``execution_horizon`` (singular) was the name the parser knew while the
    configs on disk said min/max. The file still had ``control_hz``, so the
    at-least-one-section check passed and the horizons -- the entire fast/slow
    split -- were dropped without a word.
    """
    with pytest.raises(SystemExit) as excinfo:
        run_real._load_het_config(
            _write(tmp_path, "control_hz:\n  11: 30\nexecution_horizon:\n  11: 10\n")
        )

    assert "execution_horizon" in str(excinfo.value)
    assert "unknown section" in str(excinfo.value)


def test_empty_config_is_rejected(run_real, tmp_path):
    with pytest.raises(SystemExit):
        run_real._load_het_config(_write(tmp_path, "{}\n"))


def test_non_integer_entry_is_rejected(run_real, tmp_path):
    with pytest.raises(SystemExit):
        run_real._load_het_config(_write(tmp_path, "control_hz:\n  11: fast\n"))


def test_shipped_task_index_covers_the_configs(run_real):
    """language_index used to crash on a task index this repo never shipped."""
    task_index = run_real._load_task_index(run_real._default_task_index_path())

    assert {0, 1} <= set(task_index)
    assert all(isinstance(v, str) and v for v in task_index.values())


# ── flag construction ────────────────────────────────────────────


def _flags(text: str) -> set[str]:
    return {tok for tok in text.split() if tok.startswith("--") and tok != "--"}


def test_horizon_overrides_use_the_flags_the_node_defines():
    args = FleetDispatcher._build_extra_args(
        None,
        None,
        min_execution_horizon_overrides={11: 5},
        max_execution_horizon_overrides={11: 10},
    )

    assert args is not None
    assert "--min-execution-horizon 5" in args[11]
    assert "--max-execution-horizon 10" in args[11]
    assert _flags(args[11]) <= NODE_FLAGS


def test_every_emitted_flag_exists_on_the_node():
    """parse_known_args means an unrecognised flag is dropped, not rejected."""
    args = FleetDispatcher._build_extra_args(
        {11: 30},
        {11: "put the red legos in the red mug"},
        min_execution_horizon_overrides={11: 5},
        max_execution_horizon_overrides={11: 10},
        barrier_robot_ids={11},
    )

    assert _flags(args[11]) <= NODE_FLAGS


def test_prompt_with_spaces_stays_one_argument():
    args = FleetDispatcher._build_extra_args(
        None, {11: "put the red legos in the red mug"}
    )

    assert "--prompt 'put the red legos in the red mug'" in args[11]


def test_robot_without_overrides_still_gets_the_barrier():
    args = FleetDispatcher._build_extra_args(
        {11: 30}, None, barrier_robot_ids={11, 12}
    )

    assert "--control-hz" not in args[12]
    assert args[12].strip() == "-- --barrier"


def test_only_min_horizon_set_leaves_max_at_the_node_default():
    args = FleetDispatcher._build_extra_args(
        None, None, min_execution_horizon_overrides={11: 5}
    )

    assert "--min-execution-horizon 5" in args[11]
    assert "--max-execution-horizon" not in args[11]


def test_no_overrides_at_all_is_none():
    assert FleetDispatcher._build_extra_args(None, None) is None
