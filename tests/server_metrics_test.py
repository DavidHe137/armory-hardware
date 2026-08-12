"""The server-metrics hand-off from the policy server to the metrics pass.

This path fails silently in both directions. run_real used to GET a
``/save-metrics`` endpoint that no longer exists and write
``server_metrics_history.json``, a name nothing reads; the trial still ran, the
plots were just quietly absent. So these tests assert on the exact filenames
and the exact directory ``evaluation.metrics`` resolves, not on "files were
copied".
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# What evaluation.metrics reads, and where. loading.py resolves each of these
# as ``<trial>/server/<name>``: load_server_metadata -> metadata.json,
# _read_jsonl -> batches.jsonl / events.jsonl, and the scheduler decisions
# loader -> scheduler_decisions.jsonl. Renaming one here without renaming it in
# the server drops that plot with only a warning.
LOADER_FILES = frozenset(
    {
        "metadata.json",
        "batches.jsonl",
        "events.jsonl",
        "scheduler_decisions.jsonl",
    }
)


@pytest.fixture(scope="module")
def run_real():
    """Import scripts/run_real.py by path -- scripts/ is not a package."""
    pytest.importorskip("tyro", reason="run_real needs the 'trial' extra")
    pytest.importorskip("imageio_ffmpeg", reason="run_real needs the 'trial' extra")
    spec = importlib.util.spec_from_file_location(
        "run_real", REPO_ROOT / "scripts" / "run_real.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _server_dir(tmp_path, names=LOADER_FILES) -> pathlib.Path:
    src = tmp_path / "serve" / "server"
    src.mkdir(parents=True)
    for name in names:
        (src / name).write_text(f"payload::{name}\n")
    return src


def test_copies_every_file_the_loader_reads(run_real, tmp_path):
    src = _server_dir(tmp_path)
    out = tmp_path / "trial"
    out.mkdir()

    run_real._copy_server_metrics(str(src), out)

    # The metrics pass looks under <trial>/server/, not the trial root.
    assert {p.name for p in (out / "server").iterdir()} == LOADER_FILES
    assert (out / "server" / "metadata.json").read_text() == "payload::metadata.json\n"


def test_declared_file_list_matches_the_loader(run_real):
    """The module's list is what gets copied; drift here loses plots in silence."""
    assert set(run_real.SERVER_METRICS_FILES) == LOADER_FILES


def test_unreadable_dir_is_a_warning_not_a_crash(run_real, tmp_path, caplog):
    """A server on a host we cannot read must not fail the trial after the fact.

    The episode data is already fetched by this point; aborting here would
    throw away a completed run over absent plots.
    """
    out = tmp_path / "trial"
    out.mkdir()

    run_real._copy_server_metrics(str(tmp_path / "nope"), out)

    assert not (out / "server").exists()
    assert "not readable" in caplog.text


def test_partial_dir_copies_what_exists(run_real, tmp_path, caplog):
    """A server restarted mid-run may not have written every log yet."""
    src = _server_dir(tmp_path, names={"metadata.json", "batches.jsonl"})
    out = tmp_path / "trial"
    out.mkdir()

    run_real._copy_server_metrics(str(src), out)

    assert {p.name for p in (out / "server").iterdir()} == {
        "metadata.json",
        "batches.jsonl",
    }
    assert "scheduler_decisions.jsonl" in caplog.text
