from scripts.check_dist import wheel_for_version


def test_distribution_check_ignores_stale_wheels(tmp_path):
    stale = tmp_path / "tinyharness-9.9.9-py3-none-any.whl"
    current = tmp_path / "tinyharness-0.1.0-py3-none-any.whl"
    stale.touch()
    current.touch()

    assert wheel_for_version("0.1.0", tmp_path) == current
