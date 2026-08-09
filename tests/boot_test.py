"""Hermetic tests for the boot script.

``_boot_script`` is pure string building, so these need no fleet and no
network — deliberately not in ``fleet_test.py``, which only runs against real
hardware.
"""

from __future__ import annotations

from textwrap import dedent

import pytest

from armory_hardware import FleetConfig, FleetController
from armory_hardware import fleet as fleet_mod


@pytest.fixture
def controller(tmp_path):
    cfg_path = tmp_path / "fleet.yaml"
    cfg_path.write_text(
        dedent(
            f"""
            workstations:
              - id: 11
            logging:
              log_dir: {tmp_path / "logs"}
            """
        ).lstrip()
    )
    return FleetController(FleetConfig(str(cfg_path)))


def test_boot_puts_armory_client_on_pythonpath(controller):
    """The container must be able to import armory_client.

    The lab image installs it editable but records a pre-restructure path, so
    the import fails and takes the client node down on its import line. Boot
    supplies the real location as container env — reachable from every
    ``docker exec``, unlike an export inside the one-shot entrypoint, which
    boot runs as its own short-lived exec.
    """
    script = controller._boot_script("/lab/CS4803ARM_Lab")

    assert f"-e PYTHONPATH={fleet_mod.ARMORY_CLIENT_SRC}" in script


def test_armory_client_path_is_container_side(controller):
    """It resolves through the user_data bind mount, not the host layout.

    The host path differs per operator (``ARMORY_PIPER_DOCKER_DIR``); the
    mount target does not, so this one must not be built from piper_root.
    """
    script = controller._boot_script("/lab/CS4803ARM_Lab")

    assert fleet_mod.ARMORY_CLIENT_SRC.startswith("/CS4803ARM_Lab/user_data/")
    assert "/lab/CS4803ARM_Lab/armory" not in script
    # The bind mount that makes the path resolve at all.
    assert (
        "source=/lab/CS4803ARM_Lab/user_data,"
        "target=/CS4803ARM_Lab/user_data" in script
    )
