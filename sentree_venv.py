from __future__ import annotations

import os
import sys


def _in_virtualenv() -> bool:
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    real_prefix = getattr(sys, "real_prefix", None)  # virtualenv (legacy)
    return bool(real_prefix) or (sys.prefix != base_prefix)


def ensure_venv(*, allow_env_var: str = "SENTREE_ALLOW_NO_VENV") -> None:
    """
    Enforce running inside a Python virtual environment.

    This repo intentionally avoids global `pip install -r requirements.txt`.
    Use `setup.sh` / `setup.ps1` to create `.venv` and install dependencies.
    """
    if os.environ.get(allow_env_var) in {"1", "true", "TRUE", "yes", "YES"}:
        return

    if _in_virtualenv():
        return

    raise RuntimeError(
        "SenTree requires a virtual environment.\n\n"
        "Windows:   powershell -ExecutionPolicy Bypass -File .\\setup.ps1\n"
        "macOS/Linux:  ./setup.sh\n\n"
        "If you know what you're doing, you can bypass this check by setting:\n"
        f"  {allow_env_var}=1\n"
    )

