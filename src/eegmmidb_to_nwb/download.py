"""Download the EDF files of one subject from PhysioNet's open-data bucket."""

import argparse
import urllib.request
from pathlib import Path

# physionet.org/files/... redirects here; downloading from the bucket directly is much faster.
BASE_URL = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0"


def download_subject(subject: int, data_dir: str | Path, overwrite: bool = False) -> list[Path]:
    """Download S{subject:03d}R01.edf to R14.edf into data_dir/S{subject:03d}/."""
    subject_id = f"S{subject:03d}"
    subject_dir = Path(data_dir) / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for run in range(1, 15):
        name = f"{subject_id}R{run:02d}.edf"
        path = subject_dir / name
        if overwrite or not path.exists() or path.stat().st_size == 0:
            urllib.request.urlretrieve(f"{BASE_URL}/{subject_id}/{name}", path)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download one subject of the EEG Motor Movement/Imagery Dataset.")
    parser.add_argument("subject", type=int, help="Subject number, 1 to 109")
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args(argv)
    for path in download_subject(args.subject, args.data_dir):
        print(path)


if __name__ == "__main__":
    main()
