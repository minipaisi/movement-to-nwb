"""Interface that writes the EDF annotations of one run as the NWB trials table."""

from pathlib import Path

import mne
from neuroconv.basedatainterface import BaseDataInterface
from pynwb import NWBFile

from .runs import get_run_info


class EEGMMIDBTrialsInterface(BaseDataInterface):
    """Reads the T0/T1/T2 annotations of one EDF run and adds them as trials.

    Each annotation becomes one trial with its start and stop time (seconds from the
    start of the run), the raw annotation code and the condition it stands for in this
    run (for example T1 is "left_fist" in run 3 but "both_fists" in run 5).
    """

    def __init__(self, file_path: str | Path, run: int, verbose: bool = False):
        super().__init__(file_path=file_path, run=run, verbose=verbose)
        self.run_info = get_run_info(run)

    def read_annotations(self) -> mne.Annotations:
        raw = mne.io.read_raw_edf(self.source_data["file_path"], preload=False, verbose="error")
        return raw.annotations

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict | None = None, **conversion_options) -> None:
        annotations = self.read_annotations()
        unknown = set(annotations.description) - set(self.run_info.conditions)
        if unknown:
            raise ValueError(
                f"Run {self.run_info.run} has annotation codes {sorted(unknown)} that this run "
                f"should not contain (expected {sorted(self.run_info.conditions)})."
            )

        nwbfile.add_trial_column(name="annotation_code", description="Annotation code in the EDF file (T0, T1 or T2).")
        nwbfile.add_trial_column(
            name="condition",
            description="What the annotation code means in this run: rest, left_fist, right_fist, both_fists or both_feet.",
        )
        nwbfile.add_trial_column(
            name="movement_type",
            description="executed (real movement), imagined (motor imagery) or none (rest and baseline).",
        )
        for onset, duration, code in zip(annotations.onset, annotations.duration, annotations.description):
            condition = self.run_info.conditions[code]
            movement_type = "none" if condition == "rest" else self.run_info.movement_type
            nwbfile.add_trial(
                start_time=float(onset),
                stop_time=float(onset + duration),
                annotation_code=str(code),
                condition=condition,
                movement_type=movement_type,
            )
