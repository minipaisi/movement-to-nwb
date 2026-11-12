"""Convert one Pose2Sim trial to NWB."""

from datetime import datetime
from pathlib import Path

from neuroconv import NWBConverter
from neuroconv.utils import dict_deep_update, load_dict_from_file

from .interface import Pose2SimTrialInterface

METADATA_FILE = Path(__file__).parent / "metadata.yaml"
SUBJECT_ID = "Pose2SimDemo"
SESSION_ID = "S01"


class Pose2SimConverter(NWBConverter):
    data_interface_classes = dict(Pose=Pose2SimTrialInterface)

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata = dict_deep_update(metadata, load_dict_from_file(METADATA_FILE))
        metadata["NWBFile"]["session_id"] = SESSION_ID
        metadata["NWBFile"]["session_description"] = "A person balancing on a beam, filmed by four calibrated cameras."
        metadata["Subject"]["subject_id"] = SUBJECT_ID
        return metadata


def convert_trial(
    trial_dir: str | Path,
    output_dir: str | Path,
    session_start_time: datetime,
    subject_age: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert one trial to sub-<subject>/sub-<subject>_ses-<session>_behavior.nwb under output_dir.

    The demo has no recording date and NWB requires one, so session_start_time must come from the caller
    with a time zone. subject_age is an ISO 8601 duration such as "P25Y"; the demo does not publish it.
    """
    if session_start_time.tzinfo is None:
        raise ValueError("session_start_time needs a time zone.")
    converter = Pose2SimConverter(source_data=dict(Pose=dict(trial_dir=trial_dir)))
    metadata = converter.get_metadata()
    metadata["NWBFile"]["session_start_time"] = session_start_time
    if subject_age is not None:
        metadata["Subject"]["age"] = subject_age

    nwbfile_path = Path(output_dir) / f"sub-{SUBJECT_ID}" / f"sub-{SUBJECT_ID}_ses-{SESSION_ID}_behavior.nwb"
    nwbfile_path.parent.mkdir(parents=True, exist_ok=True)
    converter.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=overwrite)
    return nwbfile_path
