from g1dance.utils.runs import find_saved_motion, latest_run_with_checkpoint


def test_saved_motion_is_found_in_params(tmp_path):
    assert find_saved_motion(str(tmp_path)) is None
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "env.yaml").write_text("x")
    assert find_saved_motion(str(tmp_path)) is None
    (tmp_path / "params" / "my_dance.npz").write_bytes(b"x")
    assert find_saved_motion(str(tmp_path)).endswith("my_dance.npz")


def make_run(root, name, files):
    d = root / name
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_bytes(b"x")


def test_picks_newest_run_with_a_checkpoint(tmp_path):
    make_run(tmp_path, "2026-10-22_10-00-00_exp", ["model_0.pt", "model_250.pt"])
    make_run(tmp_path, "2026-10-22_12-00-00_exp", ["model_500.pt"])
    make_run(tmp_path, "2026-10-22_14-00-00_exp", ["events.out.tfevents.1"])  # crashed before the first save
    make_run(tmp_path, "2026-10-22_15-00-00_other", ["model_0.pt"])
    make_run(tmp_path, "2026-10-22_16-00-00_pre_exp", ["model_0.pt"])  # a different run whose name ends in "exp"
    assert latest_run_with_checkpoint(str(tmp_path), "exp") == "2026-10-22_12-00-00_exp"


def test_returns_none_without_a_match(tmp_path):
    assert latest_run_with_checkpoint(str(tmp_path / "missing"), "exp") is None
    make_run(tmp_path, "2026-10-22_10-00-00_exp", ["notes.txt"])
    assert latest_run_with_checkpoint(str(tmp_path), "exp") is None
    assert latest_run_with_checkpoint(str(tmp_path), "ex") is None
