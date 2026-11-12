"""Download the single-trial demo from the Pose2Sim 0.8.0 release, pinned by checksum."""

import argparse
import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

ARCHIVE_URL = "https://github.com/perfanalytics/pose2sim/archive/refs/tags/v0.8.0.tar.gz"
ARCHIVE_SHA256 = "87907beed2f6dd3c0c4f8558a9df45929a4d84a324fe804b5b64c2d9df1ed122"
ARCHIVE_ROOT = PurePosixPath("pose2sim-0.8.0")
TRIAL_IN_ARCHIVE = ARCHIVE_ROOT / "Pose2Sim" / "S01_Demo_SingleTrial"
TRIAL_NAME = TRIAL_IN_ARCHIVE.name


def extract_trial(archive_path: str | Path, data_dir: str | Path, expected_sha256: str = ARCHIVE_SHA256) -> Path:
    """Check the archive's checksum and extract the trial and the Pose2Sim license into data_dir."""
    archive_path = Path(archive_path)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Checksum mismatch for {archive_path}: expected {expected_sha256}, got {digest}.")

    trial_dir = Path(data_dir) / TRIAL_NAME
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if not member.isfile() or ".." in path.parts:
                continue
            if path == ARCHIVE_ROOT / "LICENSE":
                target = trial_dir / "LICENSE"
            elif path.is_relative_to(TRIAL_IN_ARCHIVE):
                target = trial_dir / path.relative_to(TRIAL_IN_ARCHIVE)
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    return trial_dir


def download_trial(data_dir: str | Path) -> Path:
    """Download the release archive (about 43 MB) and extract the demo trial into data_dir."""
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "pose2sim.tar.gz"
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)
        return extract_trial(archive_path, data_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the Pose2Sim 0.8.0 single-trial demo.")
    parser.add_argument("data_dir", type=Path)
    print(download_trial(parser.parse_args(argv).data_dir))


if __name__ == "__main__":
    main()
