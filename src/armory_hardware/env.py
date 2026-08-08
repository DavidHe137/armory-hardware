"""Loads operator-specific settings from a ``.env`` file.

The values covered here are deployment identity -- SSH account, jump host, camera
server -- which differ per operator and should not be committed. See ``.env.example``
for the full list.

Entry points call :func:`load_env` once at startup. ``FleetConfig`` deliberately does
not, so library callers and tests stay hermetic.
"""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv

_loaded = False


def _repo_root_env() -> str:
    """``<repo>/.env``, derived from this file's location in ``src/armory_hardware/``."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(pkg_dir)), ".env")


def load_env() -> str | None:
    """Load ``.env`` into ``os.environ``; return the file used, if any.

    Real environment variables always win over the file, so an exported
    ``ARMORY_SSH_USER`` overrides whatever ``.env`` says. Repeat calls are no-ops.

    Set ``ARMORY_ENV_FILE`` to point at a specific file; otherwise the repo-root
    ``.env`` is preferred, falling back to a search upward from the working directory.
    """
    global _loaded
    if _loaded:
        return None
    _loaded = True

    explicit = os.environ.get("ARMORY_ENV_FILE")
    if explicit:
        load_dotenv(explicit, override=False)
        return explicit

    root_env = _repo_root_env()
    if os.path.isfile(root_env):
        load_dotenv(root_env, override=False)
        return root_env

    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
        return found
    return None
