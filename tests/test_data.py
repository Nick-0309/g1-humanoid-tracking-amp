import numpy as np
import pytest

from g1dance.data.convert import G1_29DOF_SDK_ORDER, convert_hh_csv, hh_csv_to_unitree_array
from g1dance.data.motion_check import check_motion


def write_hh_csv(path, *, rate="30.000000", shuffle=True, drop=None, extra=None, with_rate=True):
    rng = np.random.default_rng(0)
    names = [n.removesuffix("_joint") for n in G1_29DOF_SDK_ORDER]
    order = list(range(len(names)))
    if shuffle:
        order = rng.permutation(len(names)).tolist()
    cols = [f"dof_{names[i]}" for i in order]
    if drop:
        cols.remove(f"dof_{drop}")
    if extra:
        cols.append(f"dof_{extra}")
    frames = 6
    root = rng.normal(size=(frames, 7))
    dof = {c: rng.normal(size=frames) for c in cols}
    with open(path, "w", encoding="utf-8") as f:
        f.write("# robot: unitree_g1__g1_29dof\n")
        if with_rate:
            f.write(f"# sample_rate: {rate}\n")
        f.write("time,root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw," + ",".join(cols) + "\n")
        for i in range(frames):
            row = [i / 30.0, *root[i], *[dof[c][i] for c in cols]]
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")
    return root, dof


def test_columns_are_reordered_by_name(tmp_path):
    src = tmp_path / "hh.csv"
    root, dof = write_hh_csv(src)
    motion, rate = hh_csv_to_unitree_array(src)
    assert rate == 30.0
    assert motion.shape == (6, 7 + 29)
    np.testing.assert_allclose(motion[:, :3], root[:, :3], atol=1e-5)
    for j, name in enumerate(G1_29DOF_SDK_ORDER):
        np.testing.assert_allclose(motion[:, 7 + j], dof["dof_" + name.removesuffix("_joint")], atol=1e-5)


def test_quaternions_are_normalised_and_continuous(tmp_path):
    src = tmp_path / "hh.csv"
    write_hh_csv(src)
    motion, _ = hh_csv_to_unitree_array(src)
    q = motion[:, 3:7]
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-9)
    assert np.all(np.sum(q[1:] * q[:-1], axis=1) >= 0)


def test_convert_writes_headerless_csv_and_returns_fps(tmp_path):
    src, dst = tmp_path / "hh.csv", tmp_path / "out" / "g1.csv"
    write_hh_csv(src)
    assert convert_hh_csv(src, dst) == 30.0
    loaded = np.loadtxt(dst, delimiter=",")
    assert loaded.shape == (6, 36)


def test_rejects_other_robots(tmp_path):
    src = tmp_path / "hh.csv"
    write_hh_csv(src, drop="left_knee")
    with pytest.raises(ValueError, match="missing joints"):
        hh_csv_to_unitree_array(src)
    write_hh_csv(src, extra="head_yaw")
    with pytest.raises(ValueError, match="unexpected joints"):
        hh_csv_to_unitree_array(src)


def test_non_integer_sample_rate_is_rejected(tmp_path):
    src = tmp_path / "hh.csv"
    write_hh_csv(src, rate="25.500000")
    with pytest.raises(ValueError, match="not an integer"):
        convert_hh_csv(src, tmp_path / "o.csv")


def test_rounded_time_stamps_still_give_an_integer_rate(tmp_path):
    src = tmp_path / "hh.csv"
    write_hh_csv(src, with_rate=False)
    assert convert_hh_csv(src, tmp_path / "o.csv") == 30.0


def test_sample_rate_falls_back_to_time_column(tmp_path):
    src = tmp_path / "hh.csv"
    write_hh_csv(src, with_rate=False)
    _, rate = hh_csv_to_unitree_array(src)
    assert rate == pytest.approx(30.0, rel=1e-3)


def make_npz(path, *, root_z=0.75, fps=50, penetration=0.0):
    t, b, j = 100, 30, 29
    pos = np.zeros((t, b, 3))
    pos[:, :, 2] = 0.3 + penetration
    pos[:, 0, 2] = root_z
    pos[:, 5, 2] = 0.03 + penetration
    quat = np.zeros((t, b, 4))
    quat[..., 3] = 1.0
    np.savez(
        path, fps=[fps], joint_pos=np.zeros((t, j)), joint_vel=np.zeros((t, j)), body_pos_w=pos,
        body_quat_w=quat, body_lin_vel_w=np.zeros((t, b, 3)), body_ang_vel_w=np.zeros((t, b, 3)),
    )


def test_motion_check_accepts_a_sane_clip(tmp_path):
    make_npz(tmp_path / "ok.npz")
    report, warnings = check_motion(str(tmp_path / "ok.npz"))
    assert warnings == []
    assert report["frames"] == 100 and report["duration_s"] == 2.0


def test_motion_check_flags_problems(tmp_path):
    make_npz(tmp_path / "bad.npz", root_z=75.0, fps=30, penetration=-0.1)
    _, warnings = check_motion(str(tmp_path / "bad.npz"))
    text = " ".join(warnings)
    assert "fps" in text and "pelvis height" in text and "below the ground" in text
