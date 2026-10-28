"""Command line entry point: eegmmidb-to-nwb SUBJECT_DIR OUTPUT_DIR."""

import argparse
from pathlib import Path

from .converter import convert_subject


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert one subject of the PhysioNet EEG Motor Movement/Imagery Dataset to NWB.")
    parser.add_argument("subject_dir", type=Path, help="Folder with the subject's EDF files, e.g. data/S001")
    parser.add_argument("output_dir", type=Path, help="Where to write sub-<subject>/*.nwb")
    parser.add_argument("--subject-age", help="ISO 8601 duration, e.g. P25Y (not published by PhysioNet)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing NWB files")
    args = parser.parse_args(argv)

    for path in convert_subject(args.subject_dir, args.output_dir, subject_age=args.subject_age, overwrite=args.overwrite):
        print(path)


if __name__ == "__main__":
    main()
