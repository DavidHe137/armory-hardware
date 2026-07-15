# Armory Hardware

Hardware-specific fleet control, trial tooling, and the operator TUI used with Armory.

This repository is intentionally separate from the public Armory core. It contains setup-specific
assumptions for the hardware fleet and is not a general-purpose Armory interface.

Run the dashboard from this repository with:

    uv run armory-tui

Set ARMORY_TUI_CONFIG to use a different fleet configuration.

The optional run_real.py trial tooling expects this repository to sit beside the Armory core
checkout, as ../armory.
