"""Find earlier runs to resume from (pure python)."""

from __future__ import annotations

import os
import re


def find_saved_motion(run_dir: str) -> str | None:
    """The motion file a run was trained on (``train.py`` copies it into ``<run_dir>/params``)."""
    params = os.path.join(run_dir, "params")
    if not os.path.isdir(params):
        return None
    clips = sorted(f for f in os.listdir(params) if f.endswith(".npz"))
    return os.path.join(params, clips[0]) if clips else None


def latest_run_with_checkpoint(log_root: str, run_name: str) -> str | None:
    """Name of the newest run folder ``<timestamp>_<run_name>`` below ``log_root`` that holds a checkpoint.

    Run folders start with a sortable timestamp, so the lexicographic maximum is the newest one. Runs that
    crashed before their first save are skipped.
    """
    if not os.path.isdir(log_root):
        return None
    pattern = re.compile(rf"\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}_{re.escape(run_name)}")
    candidates = sorted(
        d
        for d in os.listdir(log_root)
        if pattern.fullmatch(d)
        and os.path.isdir(os.path.join(log_root, d))
        and any(re.fullmatch(r"model_\d+\.pt", f) for f in os.listdir(os.path.join(log_root, d)))
    )
    return candidates[-1] if candidates else None
