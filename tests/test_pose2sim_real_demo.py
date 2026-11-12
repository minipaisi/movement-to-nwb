"""Checks on the real Pose2Sim demo: 100 frames from four cameras, stored in tests/data."""

import json
from pathlib import Path

import numpy as np
import pytest

from pose2sim_to_nwb.association import (
    MAX_REPROJECTION_ERROR_PIXELS,
    MIN_AGREEING_FRACTION,
    MIN_CONFIDENCE,
    associate_participant,
    triangulate_views,
)
from pose2sim_to_nwb.calibration import read_qca
from pose2sim_to_nwb.openpose import read_camera_folder
from pose2sim_to_nwb.skeleton import KEYPOINT_NAMES

DATA = Path(__file__).parent / "data"
TRIAL = DATA / "S01_Demo_SingleTrial"
CAMERA_FOLDERS = ["cam01", "cam02", "cam03", "cam04"]
# Pose2Sim 0.4.0's own 3D result for the same 2D files and calibration: tests/data/reference/README.md
REFERENCE_TRC = DATA / "reference" / "Balancing_for_IK.trc"


def index(name):
    return KEYPOINT_NAMES.index(name)


@pytest.fixture(scope="module")
def cameras():
    return read_qca(TRIAL / "calibration" / "Calib.qca.txt")


@pytest.fixture(scope="module")
def detections():
    return [read_camera_folder(TRIAL / "pose" / f"{name}_json") for name in CAMERA_FOLDERS]


@pytest.fixture(scope="module")
def track(cameras, detections):
    return associate_participant(cameras, [detection.frames for detection in detections])


def read_reference():
    """Return {keypoint: array (100, 3)} from the .trc file, in this repository's axes."""
    rows = REFERENCE_TRC.read_text().splitlines()
    names = [name for name in rows[3].split("\t")[2:] if name]
    values = np.array([[float(x) for x in row.split("\t")[2 : 2 + 3 * len(names)]] for row in rows[5:] if row.strip()])
    # The .trc is in OpenSim axes, with Y up: (X, Y, Z) here is (y, z, x) in the calibration's world frame.
    return {
        {"Head": "HeadTop"}.get(name, name): values[:, 3 * position : 3 * position + 3][:, [2, 0, 1]]
        for position, name in enumerate(names)
    }


def test_demo_has_the_quirks_the_reader_handles(detections):
    assert [len(detection.frames) for detection in detections] == [100, 100, 100, 100]
    assert {detection.first_frame for detection in detections} == {0}
    # cam01 and cam02 pick up someone else in every frame; cam01 has a third detection in frame 37
    assert {len(frame) for frame in detections[0].frames} == {2, 3}
    assert len(detections[0].frames[37]) == 3
    assert {len(frame) for frame in detections[1].frames} == {2}
    assert {len(frame) for frame in detections[2].frames} == {1}
    # The placeholder files really are in the folders that were read
    assert (TRIAL / "pose" / "cam03_json" / "del_000000_0.json").exists()


def test_every_frame_finds_the_participant(track):
    # In this demo OpenPose happens to list the participant first; test_pose2sim_association.py covers other orders.
    assert (track.selected == 0).all()


def test_2d_values_are_exactly_those_in_the_json_files(track):
    for camera, name in enumerate(CAMERA_FOLDERS):
        for frame in range(100):
            with open(TRIAL / "pose" / f"{name}_json" / f"{name}.{frame:04d}.json") as file:
                person = json.load(file)["people"][int(track.selected[frame, camera])]
            raw = np.reshape(person["pose_keypoints_2d"], (25, 3))
            stored = track.keypoints_2d[frame, camera]
            not_found = (raw == 0).all(axis=1)
            assert np.isnan(stored[not_found, :2]).all()
            assert (stored[not_found, 2] == 0).all()
            assert (stored[~not_found] == raw[~not_found]).all()


def test_3d_matches_pose2sim_on_the_same_inputs(track):
    reference = read_reference()
    assert len(reference) == 21
    medians = []
    for name, expected in reference.items():
        distance = np.linalg.norm(track.keypoints_3d[:, index(name)] - expected, axis=1)
        medians.append(np.nanmedian(distance))
        # Pose2Sim also filters its output, so they are not expected to be identical.
        assert medians[-1] < 0.03, name
    assert np.median(medians) < 0.01


@pytest.mark.parametrize("shift", [-2, -1, 1, 2])
def test_frames_are_aligned_with_pose2sim(track, shift):
    def error(ours, reference):
        return np.nanmedian(np.linalg.norm(ours - reference, axis=-1))

    reference = np.stack(list(read_reference().values()), axis=1)
    ours = track.keypoints_3d[:, [index(name) for name in read_reference()]]
    shifted = error(ours[shift:], reference[:-shift]) if shift > 0 else error(ours[:shift], reference[-shift:])
    assert error(ours, reference) < shifted


def test_body_makes_anatomical_sense(track):
    height = np.nanmedian(track.keypoints_3d[..., 2], axis=0)
    assert (
        height[index("HeadTop")]
        > height[index("Neck")]
        > height[index("LHip")]
        > height[index("LKnee")]
        > height[index("LAnkle")]
    )
    assert (
        height[index("HeadTop")]
        > height[index("Neck")]
        > height[index("RHip")]
        > height[index("RKnee")]
        > height[index("RAnkle")]
    )
    # Limb lengths in metres should stay almost constant even though the person moves.
    for a, b in [("LHip", "LKnee"), ("RHip", "RKnee"), ("LKnee", "LAnkle"), ("RKnee", "RAnkle")]:
        length = np.linalg.norm(track.keypoints_3d[:, index(a)] - track.keypoints_3d[:, index(b)], axis=1)
        assert 0.3 < np.nanmedian(length) < 0.5, (a, b)
        assert np.nanstd(length) < 0.035, (a, b)


def test_participant_is_in_front_of_every_camera(cameras, track):
    for camera in cameras:
        depth = (track.keypoints_3d @ camera.rotation.T + camera.translation)[..., 2]
        assert np.nanmin(depth) > 1.0


def test_reprojection_error_is_small_in_every_frame(track):
    assert np.nanmax(track.reprojection_error) < 15.0
    assert np.isfinite(track.reprojection_error).all()


def test_agreement_threshold_separates_true_detections_from_bystanders(cameras, detections):
    matrices = np.stack([camera.projection_matrix for camera in cameras])

    def agreeing_fraction(person_in_cam01, frame):
        views = np.stack([detections[0].frames[frame][person_in_cam01]] + [d.frames[frame][0] for d in detections[1:]])
        errors = triangulate_views(cameras, matrices, views, MIN_CONFIDENCE, MAX_REPROJECTION_ERROR_PIXELS)[2]
        usable = (views[0, :, 2] >= MIN_CONFIDENCE) & np.isfinite(views[0, :, 0])
        return np.isfinite(errors[0]).sum() / usable.sum()

    frames = [frame for frame in range(100) if frame != 37]  # frame 37's extra detections are fragments of the person
    assert min(agreeing_fraction(0, frame) for frame in frames) > MIN_AGREEING_FRACTION
    assert max(agreeing_fraction(1, frame) for frame in frames) < MIN_AGREEING_FRACTION


def test_the_x_equals_zero_glitch_does_not_reach_3d(track):
    nose = index("Nose")
    assert track.keypoints_2d[4, 0, nose, 0] == 0.0  # present in the source
    distance = np.linalg.norm(track.keypoints_3d[:, nose] - read_reference()["Nose"], axis=1)
    assert np.nanmax(distance) < 0.05
