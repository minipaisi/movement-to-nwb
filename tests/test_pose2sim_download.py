import hashlib
import io
import tarfile

import pytest

from pose2sim_to_nwb.download import ARCHIVE_ROOT, TRIAL_IN_ARCHIVE, extract_trial


def add_file(archive, name, content=b"data"):
    member = tarfile.TarInfo(str(name))
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


@pytest.fixture
def archive_path(tmp_path):
    path = tmp_path / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        add_file(archive, ARCHIVE_ROOT / "LICENSE", b"license")
        add_file(archive, ARCHIVE_ROOT / "README.md")
        add_file(archive, TRIAL_IN_ARCHIVE / "Config.toml", b"config")
        add_file(archive, TRIAL_IN_ARCHIVE / "pose" / "cam01_json" / "cam01.0000.json", b"{}")
        add_file(archive, f"{TRIAL_IN_ARCHIVE}/../../../../escape.txt")
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extracts_only_the_trial_and_license(archive_path, tmp_path):
    trial_dir = extract_trial(archive_path, tmp_path / "data", expected_sha256=sha256(archive_path))
    files = sorted(str(path.relative_to(trial_dir)) for path in trial_dir.rglob("*") if path.is_file())
    assert files == ["Config.toml", "LICENSE", "pose/cam01_json/cam01.0000.json"]
    assert (trial_dir / "LICENSE").read_bytes() == b"license"


def test_path_traversal_is_ignored(archive_path, tmp_path):
    extract_trial(archive_path, tmp_path / "a" / "b" / "c" / "data", expected_sha256=sha256(archive_path))
    assert not list(tmp_path.rglob("escape.txt"))


def test_wrong_checksum_is_rejected(archive_path, tmp_path):
    with pytest.raises(ValueError, match="Checksum mismatch"):
        extract_trial(archive_path, tmp_path / "data")
    assert not (tmp_path / "data").exists()
