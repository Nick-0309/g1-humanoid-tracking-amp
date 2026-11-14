"""Collect the eval results of all runs into one markdown table.

    python scripts/summarize_runs.py [logs_dir] [--out logs/summary.md]

Reads ``<logs>/<experiment>/<run>/eval_model_*.json`` (written by ``scripts/eval.py``; the newest per run) and
``params/features.json``. Pure python.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re


def newest_eval(run_dir: str) -> str | None:
    files = glob.glob(os.path.join(run_dir, "eval_model_*.json"))
    if not files:
        return None
    return max(files, key=lambda f: int(re.search(r"eval_model_(\d+)\.json", f).group(1)))


def fmt(value, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def collect(logs_dir: str) -> list[dict]:
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(logs_dir, "*", "*"))):
        path = newest_eval(run_dir) if os.path.isdir(run_dir) else None
        if path is None:
            continue
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        features_file = os.path.join(run_dir, "params", "features.json")
        tag = "?"
        if os.path.isfile(features_file):
            from g1dance.tasks.features import TaskFeatures

            tag = TaskFeatures.load(run_dir).tag
        diag = result.get("diagnostics") or {}
        metrics = result.get("metrics_mean", {})
        touch_policy = diag.get("touchdowns_policy_mean_over_finished_envs")
        rows.append(
            {
                "run": f"{os.path.basename(os.path.dirname(run_dir))}/{os.path.basename(run_dir)}",
                "checkpoint": os.path.basename(result["checkpoint"].replace("\\", "/")),
                "features": tag,
                "success": result["success_rate"],
                "fail_time": result.get("mean_fail_time_s"),
                "anchor_pos": metrics.get("error_anchor_pos"),
                "body_pos": metrics.get("error_body_pos"),
                "joint_pos": metrics.get("error_joint_pos"),
                "touchdowns": (
                    f"{sum(touch_policy):.0f} / {sum(diag['touchdowns_reference'])}" if touch_policy else "-"
                ),
                "mismatch": diag.get("contact_mismatch_rate"),
                "swing_contact": (diag.get("landing") or {}).get("swing_contact_rate"),
                "target_delta_p95": diag.get("joint_target_delta_max_p95_rad"),
            }
        )
    return rows


def to_markdown(rows: list[dict]) -> str:
    header = (
        "| run | checkpoint | features | success | fail time [s] | anchor pos [m] | body pos [m] | joint pos [rad] "
        "| touchdowns (policy / ref) | contact mismatch | swing contact | target delta p95 [rad] |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {r['run']} | {r['checkpoint']} | {r['features']} | {fmt(r['success'], '.2f')} | {fmt(r['fail_time'], '.1f')} "
        f"| {fmt(r['anchor_pos'])} | {fmt(r['body_pos'])} | {fmt(r['joint_pos'])} | {r['touchdowns']} "
        f"| {fmt(r['mismatch'], '.1%')} | {fmt(r['swing_contact'], '.1%')} | {fmt(r['target_delta_p95'])} |"
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "logs"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    table = to_markdown(collect(args.logs_dir))
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"wrote {args.out}")
