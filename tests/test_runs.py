import pytest

from eegmmidb_to_nwb.runs import get_run_info


@pytest.mark.parametrize(
    "run, task, movement_type, t1, t2",
    [
        (3, "left_right_fist_executed", "executed", "left_fist", "right_fist"),
        (4, "left_right_fist_imagined", "imagined", "left_fist", "right_fist"),
        (5, "fists_feet_executed", "executed", "both_fists", "both_feet"),
        (6, "fists_feet_imagined", "imagined", "both_fists", "both_feet"),
        (11, "left_right_fist_executed", "executed", "left_fist", "right_fist"),
        (14, "fists_feet_imagined", "imagined", "both_fists", "both_feet"),
    ],
)
def test_task_runs(run, task, movement_type, t1, t2):
    info = get_run_info(run)
    assert info.task == task
    assert info.movement_type == movement_type
    assert info.conditions == {"T0": "rest", "T1": t1, "T2": t2}


@pytest.mark.parametrize("run, task", [(1, "baseline_eyes_open"), (2, "baseline_eyes_closed")])
def test_baseline_runs_only_have_rest(run, task):
    info = get_run_info(run)
    assert info.task == task
    assert info.conditions == {"T0": "rest"}


@pytest.mark.parametrize("run", [0, 15, -1])
def test_invalid_run_number(run):
    with pytest.raises(ValueError):
        get_run_info(run)
