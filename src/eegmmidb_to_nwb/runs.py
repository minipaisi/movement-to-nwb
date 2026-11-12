"""What each run of the EEG Motor Movement/Imagery Dataset contains.

The EDF annotations only say T0, T1 or T2. Their meaning depends on the run number,
which is documented on the PhysioNet dataset page but not stored in the files:

    Runs 1, 2           baseline, eyes open / eyes closed (T0 only)
    Runs 3, 7, 11       open and close left or right fist         (executed)
    Runs 4, 8, 12       imagine opening and closing left or right fist (imagined)
    Runs 5, 9, 13       open and close both fists or both feet    (executed)
    Runs 6, 10, 14      imagine opening and closing both fists or both feet (imagined)

T0 is rest in every run. T1 is the left fist (runs 3, 4, 7, 8, 11, 12) or both fists
(runs 5, 6, 9, 10, 13, 14). T2 is the right fist or both feet, respectively.
"""

from dataclasses import dataclass

N_RUNS = 14


@dataclass(frozen=True)
class RunInfo:
    run: int
    task: str  # short machine-readable name, used in session_id and descriptions
    description: str
    movement_type: str  # "executed", "imagined" or "none"
    conditions: dict  # annotation code -> condition name


_BASELINE_EYES_OPEN = ("baseline_eyes_open", "Baseline, eyes open, one minute.")
_BASELINE_EYES_CLOSED = ("baseline_eyes_closed", "Baseline, eyes closed, one minute.")
_LEFT_RIGHT = {"T0": "rest", "T1": "left_fist", "T2": "right_fist"}
_FISTS_FEET = {"T0": "rest", "T1": "both_fists", "T2": "both_feet"}


def get_run_info(run: int) -> RunInfo:
    """Return the task, movement type and annotation meanings for a run number (1 to 14)."""
    if not 1 <= run <= N_RUNS:
        raise ValueError(f"Run number must be between 1 and {N_RUNS}, got {run}.")

    if run == 1:
        return RunInfo(run, *_BASELINE_EYES_OPEN, movement_type="none", conditions={"T0": "rest"})
    if run == 2:
        return RunInfo(run, *_BASELINE_EYES_CLOSED, movement_type="none", conditions={"T0": "rest"})

    # Runs 3 to 14 cycle through four tasks: 3, 7, 11 / 4, 8, 12 / 5, 9, 13 / 6, 10, 14
    task_index = (run - 3) % 4
    if task_index == 0:
        return RunInfo(
            run,
            "left_right_fist_executed",
            "Open and close the left or right fist as a target appears on the left or right "
            "side of the screen, then relax.",
            "executed",
            _LEFT_RIGHT,
        )
    if task_index == 1:
        return RunInfo(
            run,
            "left_right_fist_imagined",
            "Imagine opening and closing the left or right fist as a target appears on the "
            "left or right side of the screen, then relax.",
            "imagined",
            _LEFT_RIGHT,
        )
    if task_index == 2:
        return RunInfo(
            run,
            "fists_feet_executed",
            "Open and close both fists (target at the top of the screen) or both feet "
            "(target at the bottom), then relax.",
            "executed",
            _FISTS_FEET,
        )
    return RunInfo(
        run,
        "fists_feet_imagined",
        "Imagine opening and closing both fists (target at the top of the screen) or both "
        "feet (target at the bottom), then relax.",
        "imagined",
        _FISTS_FEET,
    )
