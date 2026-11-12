import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from nwbinspector import Importance, inspect_nwbfile, load_config
from pynwb import NWBHDF5IO, validate

from pose2sim_to_nwb import convert_trial
from pose2sim_to_nwb.association import associate_participant
from pose2sim_to_nwb.calibration import read_qca
from pose2sim_to_nwb.cli import main as cli_main
from pose2sim_to_nwb.openpose import read_camera_folder
from pose2sim_to_nwb.skeleton import KEYPOINT_NAMES, SKELETON_EDGES

TRIAL = Path(__file__).parent / "data" / "S01_Demo_SingleTrial"
START = datetime(2023, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
CAMERAS = ["cam_01", "cam_02", "cam_03", "cam_04"]


@pytest.fixture(scope="module")
def nwb_path(tmp_path_factory):
    return convert_trial(TRIAL, tmp_path_factory.mktemp("nwb"), START)


@pytest.fixture
def nwbfile(nwb_path):
    with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
        yield io.read()


@pytest.fixture(scope="module")
def track():
    cameras = read_qca(TRIAL / "calibration" / "Calib.qca.txt")
    frames = [read_camera_folder(TRIAL / "pose" / f"cam0{number}_json").frames for number in range(1, 5)]
    return cameras, associate_participant(cameras, frames)


def critical_messages(path):
    return [
        message.check_function_name
        for message in inspect_nwbfile(nwbfile_path=path, config=load_config("dandi"))
        if message.importance == Importance.CRITICAL
    ]


def test_output_path_follows_the_dandi_layout(nwb_path):
    assert nwb_path.parts[-2:] == ("sub-Pose2SimDemo", "sub-Pose2SimDemo_ses-S01_behavior.nwb")


def test_session_metadata(nwbfile):
    assert nwbfile.session_start_time == START
    assert nwbfile.session_id == "S01"
    assert nwbfile.subject.subject_id == "Pose2SimDemo"
    assert nwbfile.subject.species == "Homo sapiens"
    assert nwbfile.subject.age is None
    assert "10.21105/joss.04362" in nwbfile.related_publications[0]


def test_file_passes_schema_validation(nwb_path):
    assert validate(path=str(nwb_path)) == []


def test_camera_calibration_round_trips(nwbfile, track):
    cameras, _ = track
    assert set(CAMERAS) <= set(nwbfile.devices)
    for camera in cameras:
        device = nwbfile.devices[camera.name]
        assert np.array_equal(device.intrinsic_matrix[:], camera.intrinsic_matrix)
        assert np.array_equal(device.rotation_matrix[:], camera.rotation)
        assert np.array_equal(device.translation_vector[:], camera.translation)
        assert np.array_equal(device.distortion_coefficients[:], camera.distortion)
        assert "1088 x 1920 pixels" in device.description


def test_skeleton_round_trips(nwbfile):
    skeleton = nwbfile.processing["behavior"]["Skeletons"].skeletons["BODY_25B"]
    assert list(skeleton.nodes[:]) == list(KEYPOINT_NAMES)
    assert skeleton.edges[:].tolist() == [list(edge) for edge in SKELETON_EDGES]
    assert skeleton.subject.subject_id == "Pose2SimDemo"


def test_2d_series_round_trip(nwbfile, track):
    cameras, result = track
    multi = nwbfile.processing["behavior"]["MultiCameraPoseEstimation"]
    assert set(multi.pose_estimations) == {f"PoseEstimation_{name}" for name in CAMERAS}
    for camera_index, camera in enumerate(cameras):
        pose = multi.pose_estimations[f"PoseEstimation_{camera.name}"]
        assert set(pose.pose_estimation_series) == set(KEYPOINT_NAMES)
        assert pose.device.name == camera.name
        assert pose.skeleton.name == "BODY_25B"
        for keypoint, name in enumerate(KEYPOINT_NAMES):
            series = pose.pose_estimation_series[name]
            expected = result.keypoints_2d[:, camera_index, keypoint]
            assert np.array_equal(series.data[:], expected[:, :2], equal_nan=True)
            assert np.array_equal(series.confidence[:], expected[:, 2])
            assert series.unit == "pixels"
            assert series.rate == 60.0
            assert series.starting_time == 0.0
            assert series.reference_frame.startswith("(0, 0) is the top-left")


def test_3d_series_round_trip(nwbfile, track):
    _, result = track
    series = nwbfile.processing["behavior"]["MultiCameraPoseEstimation"].pose_estimation_series
    assert set(series) == set(KEYPOINT_NAMES)
    for keypoint, name in enumerate(KEYPOINT_NAMES):
        assert np.array_equal(series[name].data[:], result.keypoints_3d[:, keypoint], equal_nan=True)
        assert np.array_equal(series[name].confidence[:], result.confidence_3d[:, keypoint])
        assert series[name].unit == "meters"
        assert series[name].rate == 60.0
        assert "Calibration world frame" in series[name].reference_frame


def test_association_provenance_round_trips(nwbfile, track):
    _, result = track
    behavior = nwbfile.processing["behavior"]
    assert np.array_equal(behavior["AssociationSelectedDetection"].data[:], result.selected)
    assert np.array_equal(behavior["AssociationReprojectionError"].data[:], result.reprojection_error)
    assert "cam_01, cam_02, cam_03, cam_04" in behavior["AssociationSelectedDetection"].description


def test_confidence_stays_in_the_range_the_extension_defines(nwbfile):
    multi = nwbfile.processing["behavior"]["MultiCameraPoseEstimation"]
    for pose in multi.pose_estimations.values():
        for series in pose.pose_estimation_series.values():
            assert 0.0 <= np.min(series.confidence[:]) and np.max(series.confidence[:]) <= 1.0


def test_nwb_inspector_only_flags_the_missing_age(nwb_path):
    assert critical_messages(nwb_path) == ["check_subject_age"]


def test_nwb_inspector_passes_when_age_is_known(tmp_path):
    path = convert_trial(TRIAL, tmp_path, START, subject_age="P30Y")
    assert critical_messages(path) == []


def test_start_time_without_time_zone_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="time zone"):
        convert_trial(TRIAL, tmp_path, datetime(2023, 1, 2))


def test_existing_file_is_not_overwritten_by_default(nwb_path):
    with pytest.raises(ValueError, match="already exists"):
        convert_trial(TRIAL, nwb_path.parents[1], START)


@pytest.fixture
def trial_copy(tmp_path):
    return Path(shutil.copytree(TRIAL, tmp_path / "trial"))


def test_other_pose_models_are_rejected(trial_copy, tmp_path):
    config = trial_copy / "Config.toml"
    config.write_text(config.read_text().replace("pose_model = 'BODY_25B'", "pose_model = 'COCO'"))
    with pytest.raises(ValueError, match="Only BODY_25B"):
        convert_trial(trial_copy, tmp_path / "nwb", START)


def test_missing_camera_folder_is_rejected(trial_copy, tmp_path):
    shutil.rmtree(trial_copy / "pose" / "cam04_json")
    with pytest.raises(ValueError, match="do not match"):
        convert_trial(trial_copy, tmp_path / "nwb", START)


def test_cameras_covering_different_frames_are_rejected(trial_copy, tmp_path):
    (trial_copy / "pose" / "cam02_json" / "cam02.0099.json").unlink()
    with pytest.raises(ValueError, match="different frames"):
        convert_trial(trial_copy, tmp_path / "nwb", START)


def test_frame_rate_comes_from_the_config(trial_copy, tmp_path):
    config = trial_copy / "Config.toml"
    config.write_text(config.read_text().replace("frame_rate = 60", "frame_rate = 120"))
    path = convert_trial(trial_copy, tmp_path / "nwb", START)
    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        series = io.read().processing["behavior"]["MultiCameraPoseEstimation"].pose_estimation_series["Neck"]
        assert series.rate == 120.0


def test_first_frame_number_sets_the_start_time(trial_copy, tmp_path):
    for number in range(1, 5):
        for frame in range(10):
            (trial_copy / "pose" / f"cam0{number}_json" / f"cam0{number}.{frame:04d}.json").unlink()
    path = convert_trial(trial_copy, tmp_path / "nwb", START)
    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        series = io.read().processing["behavior"]["MultiCameraPoseEstimation"].pose_estimation_series["Neck"]
        assert series.starting_time == pytest.approx(10 / 60)
        assert series.data.shape == (90, 3)


def test_command_line(tmp_path, capsys):
    cli_main([str(TRIAL), str(tmp_path), "--session-start-time", "2023-01-02T03:04:05+00:00", "--subject-age", "P30Y"])
    path = Path(capsys.readouterr().out.strip())
    assert path.exists()
    assert critical_messages(path) == []
