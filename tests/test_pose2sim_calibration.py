from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from pose2sim_to_nwb.calibration import read_qca
from pose2sim_to_nwb.skeleton import KEYPOINT_NAMES, SKELETON_EDGES

CALIBRATION_FILE = Path(__file__).parent / "data" / "S01_Demo_SingleTrial" / "calibration" / "Calib.qca.txt"


@pytest.fixture
def camera():
    return read_qca(CALIBRATION_FILE)[0]


def test_values_of_the_first_camera_match_the_file(camera):
    assert camera.name == "cam_01"
    assert camera.image_size == (1088, 1920)
    # focalLengthU="107599.671875" focalLengthV="107588.828125", in units of 1/64 pixel
    assert camera.intrinsic_matrix[0, 0] == pytest.approx(107599.671875 / 64)
    assert camera.intrinsic_matrix[1, 1] == pytest.approx(107588.828125 / 64)
    # centerPointU="34110.316406" centerPointV="60680.792969"
    assert camera.intrinsic_matrix[0, 2] == pytest.approx(34110.316406 / 64)
    assert camera.intrinsic_matrix[1, 2] == pytest.approx(60680.792969 / 64)
    # x="1460.2323709212087" y="-1909.1590482454608" z="1896.5058524092062", in millimetres
    assert camera.position == pytest.approx([1.4602323709212087, -1.9091590482454608, 1.8965058524092062])


def test_cameras_are_sorted_and_have_proper_rotations():
    cameras = read_qca(CALIBRATION_FILE)
    assert [camera.name for camera in cameras] == ["cam_01", "cam_02", "cam_03", "cam_04"]
    for camera in cameras:
        assert camera.rotation @ camera.rotation.T == pytest.approx(np.eye(3))
        assert np.linalg.det(camera.rotation) == pytest.approx(1.0)


def test_cameras_are_read_in_numeric_order(tmp_path):
    text = CALIBRATION_FILE.read_text().replace('serial="cam_01"', 'serial="cam_10"')
    path = tmp_path / "Calib.qca.txt"
    path.write_text(text)
    assert [camera.name for camera in read_qca(path)] == ["cam_02", "cam_03", "cam_04", "cam_10"]


def test_invalid_rotation_is_rejected(tmp_path):
    text = CALIBRATION_FILE.read_text().replace('r11="0.5536380477336265"', 'r11="0.9"')
    path = tmp_path / "Calib.qca.txt"
    path.write_text(text)
    with pytest.raises(ValueError, match="valid rotation"):
        read_qca(path)


def test_file_without_cameras_is_rejected(tmp_path):
    path = tmp_path / "Calib.qca.txt"
    path.write_text("<calibration><cameras/></calibration>")
    with pytest.raises(ValueError, match="No cameras"):
        read_qca(path)


def test_project_matches_opencv(camera):
    points = np.random.default_rng(0).uniform([-2, -1, 0], [0, 1, 2], size=(50, 3))
    rotation_vector = cv2.Rodrigues(camera.rotation)[0]
    expected = cv2.projectPoints(
        points, rotation_vector, camera.translation, camera.intrinsic_matrix, camera.distortion
    )[0]
    assert camera.project(points) == pytest.approx(expected.reshape(-1, 2), abs=1e-8)


def opencv_undistort(camera, pixels):
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    undistorted = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        camera.intrinsic_matrix,
        camera.distortion,
        P=camera.intrinsic_matrix,
        criteria=criteria,
    )
    return undistorted.reshape(-1, 2)


def test_undistort_matches_opencv(camera):
    pixels = np.random.default_rng(0).uniform([0, 0], [1088, 1920], size=(50, 2))
    assert camera.undistort(pixels) == pytest.approx(opencv_undistort(camera, pixels), abs=1e-6)


def test_undistort_converges_with_strong_distortion(camera):
    strong = replace(camera, distortion=np.array([-0.2, 0.05, 0.001, -0.002]))
    pixels = np.random.default_rng(0).uniform([0, 0], [1088, 1920], size=(50, 2))
    assert strong.undistort(pixels, n_iterations=50) == pytest.approx(opencv_undistort(strong, pixels), abs=1e-6)


def test_undistort_inverts_project(camera):
    points = np.random.default_rng(1).uniform([-2, -1, 0], [0, 1, 2], size=(50, 3))
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ camera.projection_matrix.T
    assert camera.undistort(camera.project(points)) == pytest.approx(homogeneous[:, :2] / homogeneous[:, 2:], abs=1e-8)


def test_skeleton_has_a_connected_body_and_face():
    assert len(KEYPOINT_NAMES) == len(set(KEYPOINT_NAMES)) == 25
    assert len(SKELETON_EDGES) == len({frozenset(edge) for edge in SKELETON_EDGES})

    def connected_to(start):
        reached = {start}
        while True:
            grown = reached | {node for edge in SKELETON_EDGES if reached & set(edge) for node in edge}
            if grown == reached:
                return {KEYPOINT_NAMES[node] for node in reached}
            reached = grown

    face = {"Nose", "LEye", "REye", "LEar", "REar"}
    assert connected_to(KEYPOINT_NAMES.index("Nose")) == face
    assert connected_to(KEYPOINT_NAMES.index("Neck")) == set(KEYPOINT_NAMES) - face
