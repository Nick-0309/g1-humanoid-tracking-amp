import pytest

from g1dance.tasks.features import TaskFeatures, parse_offsets


def test_round_trip_through_the_run_directory(tmp_path):
    features = TaskFeatures(future_ref=(5, 10), residual=True, residual_scale=0.25, slew_limit=True, foot_rewards=True,
                            contact_obs=True, foot_termination=0.15)
    features.save(str(tmp_path))
    assert TaskFeatures.load(str(tmp_path)) == features
    assert (tmp_path / "params" / "features.json").is_file()


def test_missing_file_means_an_older_run(tmp_path):
    assert TaskFeatures.load(str(tmp_path)) is None


def test_invalid_combinations_are_rejected():
    with pytest.raises(ValueError):
        TaskFeatures(slew_limit=True)
    with pytest.raises(ValueError):
        TaskFeatures(residual_scale=0.25)
    with pytest.raises(ValueError):
        TaskFeatures(future_ref=(0,))
    with pytest.raises(ValueError):
        TaskFeatures(foot_termination=0.0)


def test_tag_and_offsets():
    assert TaskFeatures().tag == "base"
    assert TaskFeatures(future_ref=(5, 10), residual=True, foot_rewards=True).tag == "fut5-10+res+foot"
    assert TaskFeatures(foot_rewards=True, contact_obs=True, foot_termination=0.15).tag == "foot+cobs+fterm0.15"
    assert parse_offsets("5,10") == (5, 10)
    assert parse_offsets(None) == ()


def test_reward_features_do_not_change_the_network_layout():
    assert TaskFeatures(foot_rewards=True, foot_termination=0.15).layout == TaskFeatures().layout
    assert TaskFeatures(future_ref=(5,)).layout != TaskFeatures().layout
    assert TaskFeatures(contact_obs=True).layout != TaskFeatures().layout


def test_init_from_keeps_the_source_layout_and_the_new_rewards():
    saved = TaskFeatures(future_ref=(5,), residual=True, residual_scale=0.2)
    merged = TaskFeatures(foot_rewards=True, foot_termination=0.15).with_layout_from(saved)
    assert merged.layout == saved.layout
    assert merged.foot_rewards and merged.foot_termination == 0.15
    # a reward-only change survives a warm start from a plain tracking run
    assert TaskFeatures(foot_rewards=True).with_layout_from(TaskFeatures()) == TaskFeatures(foot_rewards=True)
