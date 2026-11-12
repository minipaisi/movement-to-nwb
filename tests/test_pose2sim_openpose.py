import json

import numpy as np
import pytest

from pose2sim_to_nwb.openpose import read_camera_folder, read_frame


def write_frame(path, people):
    path.write_text(
        json.dumps({"version": 1.3, "people": [{"person_id": [-1], "pose_keypoints_2d": p} for p in people]})
    )


def keypoints(value=1.0):
    return [value] * 75


def test_keypoint_order_and_columns(tmp_path):
    values = list(np.arange(1.0, 76.0))
    write_frame(tmp_path / "cam01.0000.json", [values])
    frame = read_frame(tmp_path / "cam01.0000.json")
    assert frame.shape == (1, 25, 3)
    assert frame[0, 0].tolist() == [1.0, 2.0, 3.0]
    assert frame[0, 24].tolist() == [73.0, 74.0, 75.0]


def test_not_found_keypoint_becomes_nan_but_a_real_zero_does_not(tmp_path):
    values = keypoints()
    values[0:3] = [0.0, 0.0, 0.0]  # not found
    values[3:6] = [0.0, 436.77, 0.68]  # on the image edge, seen in the real demo
    write_frame(tmp_path / "cam01.0000.json", [values])
    frame = read_frame(tmp_path / "cam01.0000.json")
    assert np.isnan(frame[0, 0, :2]).all() and frame[0, 0, 2] == 0.0
    assert frame[0, 1].tolist() == [0.0, 436.77, 0.68]


def test_frame_without_people_has_shape_zero(tmp_path):
    write_frame(tmp_path / "cam01.0000.json", [])
    assert read_frame(tmp_path / "cam01.0000.json").shape == (0, 25, 3)


def test_wrong_keypoint_count_is_rejected(tmp_path):
    write_frame(tmp_path / "cam01.0000.json", [[1.0] * 54])  # COCO, not BODY_25B
    with pytest.raises(ValueError, match="expected 75"):
        read_frame(tmp_path / "cam01.0000.json")


def test_frames_are_ordered_by_number_not_by_file_name(tmp_path):
    for number, value in ((10, 3.0), (9, 2.0), (8, 1.0)):
        write_frame(tmp_path / f"cam01.{number}.json", [keypoints(value)])
    detections = read_camera_folder(tmp_path)
    assert detections.first_frame == 8
    assert [frame[0, 0, 0] for frame in detections.frames] == [1.0, 2.0, 3.0]


def test_empty_placeholder_files_are_skipped(tmp_path):
    write_frame(tmp_path / "cam03.0000.json", [keypoints()])
    write_frame(tmp_path / "del_000000_0.json", [])
    assert len(read_camera_folder(tmp_path).frames) == 1


def test_placeholder_with_detections_is_rejected(tmp_path):
    write_frame(tmp_path / "cam03.0000.json", [keypoints()])
    write_frame(tmp_path / "del_000000_0.json", [keypoints()])
    with pytest.raises(ValueError, match="placeholder"):
        read_camera_folder(tmp_path)


def test_unexpected_file_is_rejected(tmp_path):
    write_frame(tmp_path / "cam01.0000.json", [keypoints()])
    (tmp_path / "notes.txt").write_text("hello")
    with pytest.raises(ValueError, match="Unexpected file"):
        read_camera_folder(tmp_path)


def test_gap_in_frames_is_rejected(tmp_path):
    write_frame(tmp_path / "cam01.0000.json", [keypoints()])
    write_frame(tmp_path / "cam01.0002.json", [keypoints()])
    with pytest.raises(ValueError, match=r"missing frames \[1\]"):
        read_camera_folder(tmp_path)


def test_empty_folder_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_camera_folder(tmp_path)
