import importlib.util
import json
import os

spec = importlib.util.spec_from_file_location(
    "summarize_runs", os.path.join(os.path.dirname(__file__), "..", "scripts", "summarize_runs.py")
)
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)


def make_run(logs, experiment, run, evals, features=None):
    d = logs / experiment / run
    (d / "params").mkdir(parents=True)
    for iteration, result in evals.items():
        (d / f"eval_model_{iteration}.json").write_text(json.dumps(result))
    if features is not None:
        (d / "params" / "features.json").write_text(json.dumps(features))


RESULT = {
    "checkpoint": "C:\\x\\model_999.pt",
    "success_rate": 0.75,
    "mean_fail_time_s": 12.5,
    "metrics_mean": {"error_anchor_pos": 0.1, "error_body_pos": 0.05, "error_joint_pos": 0.4},
    "diagnostics": {
        "touchdowns_policy_mean_over_finished_envs": [3.0, 4.0],
        "touchdowns_reference": [2, 7],
        "contact_mismatch_rate": 0.1,
        "joint_target_delta_max_p95_rad": 0.3,
    },
}


def test_newest_checkpoint_eval_is_used_and_table_is_built(tmp_path):
    older = dict(RESULT, success_rate=0.1)
    make_run(tmp_path, "g1d_amp", "2026-10-22_10-00-00_amp_v1", {250: older, 1000: RESULT},
             {"future_ref": [5], "residual": True, "residual_scale": None, "slew_limit": False, "foot_rewards": False})
    make_run(tmp_path, "g1d_tracking", "2026-10-22_11-00-00_track", {}, None)  # no eval yet: skipped
    rows = summarize.collect(str(tmp_path))
    assert len(rows) == 1
    assert rows[0]["success"] == 0.75
    assert rows[0]["features"] == "fut5+res"
    table = summarize.to_markdown(rows)
    assert "g1d_amp/2026-10-22_10-00-00_amp_v1" in table
    assert "7 / 9" in table
    assert "10.0%" in table


def test_missing_diagnostics_do_not_break_the_table(tmp_path):
    old = {k: v for k, v in RESULT.items() if k != "diagnostics"}
    make_run(tmp_path, "g1d_amp", "2026-10-22_10-00-00_old", {5: old})
    table = summarize.to_markdown(summarize.collect(str(tmp_path)))
    assert "| ? |" in table
