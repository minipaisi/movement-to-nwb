"""Command line entry point: pose2sim-to-nwb TRIAL_DIR OUTPUT_DIR --session-start-time ISO."""

import argparse
from datetime import datetime
from pathlib import Path

from .converter import convert_trial


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert one Pose2Sim trial (OpenPose JSON + calibration) to NWB.")
    parser.add_argument("trial_dir", type=Path, help="Folder with Config.toml, calibration/ and pose/")
    parser.add_argument("output_dir", type=Path, help="Where to write sub-<subject>/*.nwb")
    parser.add_argument(
        "--session-start-time",
        type=datetime.fromisoformat,
        required=True,
        help="ISO 8601 with time zone, e.g. 2023-05-01T12:00:00+00:00 (the demo does not publish one)",
    )
    parser.add_argument("--subject-age", help="ISO 8601 duration, e.g. P25Y (not published)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing NWB file")
    args = parser.parse_args(argv)
    print(convert_trial(args.trial_dir, args.output_dir, args.session_start_time, args.subject_age, args.overwrite))


if __name__ == "__main__":
    main()
