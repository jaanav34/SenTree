"""Download pre-computed pipeline outputs from GitHub Releases on first boot.

Streamlit Community Cloud doesn't persist files outside the repo, so the
dashboard's videos, ROI data, and embeddings are stored as a tarball in a
GitHub Release and extracted on startup.

Usage (automatic — called by app.py):
    python bootstrap_outputs.py

Environment variables:
    SENTREE_ASSETS_URL  — Full URL to the .tar.gz release asset.
                          Defaults to the latest release on jaanav34/SenTree.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import urllib.request
import shutil
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

REPO = "jaanav34/SenTree"
ASSET_NAME = "sentree-outputs.tar.gz"
TAG = "v1.0-outputs"

DEFAULT_URL = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET_NAME}"

# Marker file: if it exists, outputs are already present → skip download.
MARKER = Path("outputs/.bootstrap_done")

# Files the dashboard absolutely requires to render anything useful.
REQUIRED_FILES = [
    "outputs/roi/roi_results.json",
    "outputs/roi/risk_timeseries.json",
]


def _outputs_present() -> bool:
    """Return True if all critical outputs exist on disk."""
    if MARKER.exists():
        return True
    return all(Path(f).exists() for f in REQUIRED_FILES)


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress counter."""
    print(f"[bootstrap] Downloading {url}")
    print(f"[bootstrap]   → {dest}")

    req = urllib.request.Request(url, headers={"User-Agent": "SenTree-Bootstrap/1.0"})
    dest.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as fp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1 << 20  # 1 MiB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            fp.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb = downloaded / 1e6
                print(f"\r[bootstrap]   {mb:.1f} MB / {total/1e6:.1f} MB  ({pct:.0f}%)", end="", flush=True)
        print()  # newline after progress


def _extract(archive: Path) -> None:
    """Extract the tarball into the repo root."""
    print(f"[bootstrap] Extracting {archive} …")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(".")
    print(f"[bootstrap] Extraction complete.")


def bootstrap() -> None:
    """Main entry point. Downloads + extracts outputs if not already present."""
    if _outputs_present():
        return

    url = os.environ.get("SENTREE_ASSETS_URL", DEFAULT_URL).strip()
    archive = Path(".cache") / ASSET_NAME
    archive.parent.mkdir(parents=True, exist_ok=True)

    try:
        _download(url, archive)
        _extract(archive)

        # Write marker so future boots skip the download.
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text("ok\n")

        # Clean up the tarball to save disk.
        archive.unlink(missing_ok=True)
        print("[bootstrap] Done. All outputs ready.")

    except Exception as exc:
        print(f"[bootstrap] ERROR: {exc}", file=sys.stderr)
        print("[bootstrap] The dashboard will still load but some tabs may be empty.", file=sys.stderr)


if __name__ == "__main__":
    bootstrap()
