"""Convert one run of the PhysioNet EEG Motor Movement/Imagery Dataset to NWB."""

import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from neuroconv import NWBConverter
from neuroconv.datainterfaces import EDFRecordingInterface
from neuroconv.utils import dict_deep_update, load_dict_from_file

from .channels import clean_channel_name, get_template_position
from .runs import get_run_info
from .trials import EEGMMIDBTrialsInterface

# The recordings were made with BCI2000 at the Wadsworth Center, Albany, New York.
RECORDING_TIMEZONE = ZoneInfo("America/New_York")
METADATA_FILE = Path(__file__).parent / "metadata.yaml"
EDF_NAME_PATTERN = re.compile(r"S(\d{3})R(\d{2})\.edf")


def parse_edf_name(file_path: str | Path) -> tuple[str, int]:
    """Return the subject id ("S001") and run number (3) from a file name such as S001R03.edf."""
    match = EDF_NAME_PATTERN.fullmatch(Path(file_path).name)
    if match is None:
        raise ValueError(f"Expected a file name like S001R03.edf, got {Path(file_path).name!r}.")
    return f"S{match.group(1)}", int(match.group(2))


class EEGMMIDBRecordingInterface(EDFRecordingInterface):
    """EDF recording with cleaned channel names and template electrode positions."""

    def __init__(self, file_path: str | Path, verbose: bool = False):
        super().__init__(file_path=file_path, verbose=verbose)
        recording = self.recording_extractor
        raw_names = list(recording.get_channel_ids())
        names = [clean_channel_name(name) for name in raw_names]
        if len(set(names)) != len(names):
            raise ValueError(f"Channel names are not unique after cleaning: {names}")

        positions = [get_template_position(name) for name in names]
        recording.set_property("channel_name", names)
        recording.set_property("edf_label", raw_names)
        recording.set_property("group_name", ["EEG"] * len(names))
        recording.set_property("brain_area", ["scalp"] * len(names))
        for axis, values in zip("xyz", zip(*positions, strict=True), strict=True):
            recording.set_property(f"template_{axis}", list(values))


class EEGMMIDBConverter(NWBConverter):
    data_interface_classes = dict(Recording=EEGMMIDBRecordingInterface, Trials=EEGMMIDBTrialsInterface)

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata = dict_deep_update(metadata, load_dict_from_file(METADATA_FILE))

        file_path = self.data_interface_objects["Recording"].source_data["file_path"]
        subject_id, run = parse_edf_name(file_path)
        run_info = get_run_info(run)

        # EDF headers have no time zone; all runs of a subject carry the same start time.
        start_time = metadata["NWBFile"]["session_start_time"]
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        metadata["NWBFile"]["session_start_time"] = start_time.replace(tzinfo=RECORDING_TIMEZONE)

        metadata["NWBFile"]["session_id"] = f"{subject_id}R{run:02d}"
        metadata["NWBFile"]["session_description"] = (
            f"Run {run} of 14 for subject {subject_id}: {run_info.description} Task: {run_info.task}."
        )
        metadata["Subject"]["subject_id"] = subject_id
        return metadata


def convert_run(
    edf_path: str | Path,
    output_dir: str | Path,
    subject_age: str | None = None,
    overwrite: bool = False,
    stub_test: bool = False,
) -> Path:
    """Convert one EDF run to sub-<subject>/sub-<subject>_ses-<run>_ecephys.nwb under output_dir.

    subject_age is an ISO 8601 duration such as "P25Y". PhysioNet does not publish subject ages,
    so it is left empty unless the data owner provides it (DANDI requires it before upload).
    """
    edf_path = Path(edf_path)
    subject_id, run = parse_edf_name(edf_path)
    converter = EEGMMIDBConverter(
        source_data=dict(Recording=dict(file_path=edf_path), Trials=dict(file_path=edf_path, run=run))
    )
    metadata = converter.get_metadata()
    if subject_age is not None:
        metadata["Subject"]["age"] = subject_age

    nwbfile_path = Path(output_dir) / f"sub-{subject_id}" / f"sub-{subject_id}_ses-R{run:02d}_ecephys.nwb"
    nwbfile_path.parent.mkdir(parents=True, exist_ok=True)
    converter.run_conversion(
        nwbfile_path=nwbfile_path,
        metadata=metadata,
        overwrite=overwrite,
        conversion_options=dict(Recording=dict(stub_test=stub_test)),
    )
    return nwbfile_path


def convert_subject(
    subject_dir: str | Path, output_dir: str | Path, subject_age: str | None = None, overwrite: bool = False
) -> list[Path]:
    """Convert every run (SxxxRyy.edf) found in one subject folder."""
    edf_paths = sorted(p for p in Path(subject_dir).glob("S*R*.edf") if EDF_NAME_PATTERN.fullmatch(p.name))
    if not edf_paths:
        raise FileNotFoundError(f"No SxxxRyy.edf files in {subject_dir}")
    return [convert_run(path, output_dir, subject_age=subject_age, overwrite=overwrite) for path in edf_paths]
