"""Read per-frame OpenPose JSON files from one camera folder."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .skeleton import KEYPOINT_NAMES

N_KEYPOINTS = len(KEYPOINT_NAMES)
FRAME_FILE_PATTERN = re.compile(r".+\.(\d+)\.json")
# Pose2Sim's synchronisation step leaves these placeholders behind in the demo's cam03 and cam04 folders.
PLACEHOLDER_PATTERN = re.compile(r"del_\d+_\d+\.json")


@dataclass(frozen=True)
class CameraDetections:
    """All OpenPose detections from one camera.

    frames[i] has shape (n_people, 25, 3) with columns x, y, confidence. OpenPose writes (0, 0, 0)
    for a keypoint it did not find; those are stored as (NaN, NaN, 0), because (0, 0) is a valid pixel.
    """

    first_frame: int
    frames: list[np.ndarray]


def read_frame(file_path: str | Path) -> np.ndarray:
    """Read the detections of one frame as an array of shape (n_people, 25, 3)."""
    with open(file_path) as file:
        people = json.load(file)["people"]
    detections = np.empty((len(people), N_KEYPOINTS, 3))
    for index, person in enumerate(people):
        values = person["pose_keypoints_2d"]
        if len(values) != N_KEYPOINTS * 3:
            raise ValueError(f"{file_path}: expected {N_KEYPOINTS * 3} values for BODY_25B, got {len(values)}.")
        detections[index] = np.reshape(values, (N_KEYPOINTS, 3))
    not_found = np.all(detections == 0.0, axis=2)
    detections[not_found, :2] = np.nan
    return detections


def read_camera_folder(folder: str | Path) -> CameraDetections:
    """Read one <camera>_json folder, rejecting gaps and anything unexpected in it."""
    folder = Path(folder)
    frames_by_number = {}
    for path in sorted(folder.iterdir()):
        if path.name.startswith("."):
            continue
        match = FRAME_FILE_PATTERN.fullmatch(path.name)
        if match is not None:
            frames_by_number[int(match.group(1))] = read_frame(path)
        elif PLACEHOLDER_PATTERN.fullmatch(path.name):
            if len(read_frame(path)) != 0:
                raise ValueError(f"{path} looks like an empty placeholder but contains detections.")
        else:
            raise ValueError(f"Unexpected file in {folder}: {path.name}")
    if not frames_by_number:
        raise FileNotFoundError(f"No OpenPose frame files in {folder}")

    first_frame, last_frame = min(frames_by_number), max(frames_by_number)
    if len(frames_by_number) != last_frame - first_frame + 1:
        missing = sorted(set(range(first_frame, last_frame + 1)) - set(frames_by_number))
        raise ValueError(f"{folder} is missing frames {missing}.")
    return CameraDetections(first_frame, [frames_by_number[number] for number in range(first_frame, last_frame + 1)])
