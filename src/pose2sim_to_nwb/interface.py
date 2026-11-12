"""NeuroConv interface for one Pose2Sim trial: OpenPose JSON from several calibrated cameras."""

import sys
from functools import cached_property
from pathlib import Path

import numpy as np
from ndx_pose import (
    CalibratedCamera,
    MultiCameraPoseEstimation,
    PoseEstimation,
    PoseEstimationSeries,
    Skeleton,
    Skeletons,
)
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools.nwb_helpers import get_module
from pynwb import NWBFile, TimeSeries

from .association import ParticipantTrack, associate_participant
from .calibration import Camera, read_qca
from .openpose import read_camera_folder
from .skeleton import KEYPOINT_NAMES, SKELETON_EDGES

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

SUPPORTED_POSE_MODEL = "BODY_25B"
CONFIDENCE_DEFINITION_2D = "OpenPose keypoint confidence; 0 means the keypoint was not found."
CONFIDENCE_DEFINITION_3D = "Mean OpenPose confidence of the views used; 0 means there is no 3D position."


class Pose2SimTrialInterface(BaseDataInterface):
    """Adds the 2D keypoints of each camera, the triangulated 3D keypoints and the calibration to an NWB file.

    The trial folder holds Config.toml, calibration/Calib.qca.txt and pose/<camera>_json/.
    """

    def __init__(self, trial_dir: str | Path, verbose: bool = False):
        super().__init__(trial_dir=trial_dir, verbose=verbose)
        self.trial_dir = Path(trial_dir)

    @cached_property
    def config(self) -> dict:
        with open(self.trial_dir / "Config.toml", "rb") as file:
            config = tomllib.load(file)
        pose_model = config["pose"]["pose_model"]
        if pose_model != SUPPORTED_POSE_MODEL:
            raise ValueError(f"Only {SUPPORTED_POSE_MODEL} is supported, but this trial uses {pose_model}.")
        return config

    @property
    def frame_rate(self) -> float:
        return float(self.config["project"]["frame_rate"])

    @cached_property
    def cameras(self) -> list[Camera]:
        return read_qca(self.trial_dir / "calibration" / "Calib.qca.txt")

    @cached_property
    def participant_track(self) -> tuple[int, ParticipantTrack]:
        """Return the first frame number and the participant's keypoints in every camera."""
        folders = {path.name.removesuffix("_json"): path for path in (self.trial_dir / "pose").glob("*_json")}
        camera_folders = [camera.name.replace("_", "") for camera in self.cameras]
        if sorted(folders) != sorted(camera_folders):
            raise ValueError(f"Calibrated cameras {camera_folders} do not match the pose folders {sorted(folders)}.")
        detections = [read_camera_folder(folders[name]) for name in camera_folders]
        ranges = {(detection.first_frame, len(detection.frames)) for detection in detections}
        if len(ranges) != 1:
            raise ValueError(f"Cameras cover different frames (first frame, number of frames): {sorted(ranges)}.")
        return detections[0].first_frame, associate_participant(self.cameras, [d.frames for d in detections])

    def _series(self, name: str, data: np.ndarray, starting_time: float, **fields) -> PoseEstimationSeries:
        return PoseEstimationSeries(name=name, data=data, starting_time=starting_time, rate=self.frame_rate, **fields)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict | None = None, **conversion_options) -> None:
        first_frame, track = self.participant_track
        starting_time = first_frame / self.frame_rate
        behavior = get_module(nwbfile, "behavior", "Pose estimation of the participant from calibrated video.")

        skeleton = Skeleton(
            name="BODY_25B",
            nodes=list(KEYPOINT_NAMES),
            edges=np.array(SKELETON_EDGES, dtype="uint8"),
            subject=nwbfile.subject,
        )
        behavior.add(Skeletons(skeletons=[skeleton]))

        per_camera = []
        for index, camera in enumerate(self.cameras):
            device = CalibratedCamera(
                name=camera.name,
                description=(
                    f"Calibrated video camera, {camera.image_size[0]} x {camera.image_size[1]} pixels. Intrinsics and "
                    "world-to-camera transform in the OpenCV convention, translation in metres; distortion is "
                    "(k1, k2, p1, p2). Converted from a Qualisys .qca.txt calibration."
                ),
                intrinsic_matrix=camera.intrinsic_matrix,
                rotation_matrix=camera.rotation,
                translation_vector=camera.translation,
                distortion_coefficients=camera.distortion,
            )
            nwbfile.add_device(device)
            series = [
                self._series(
                    name=name,
                    description=f"{name} in {camera.name}, as detected by OpenPose (lens distortion not removed).",
                    data=track.keypoints_2d[:, index, keypoint, :2],
                    confidence=track.keypoints_2d[:, index, keypoint, 2],
                    confidence_definition=CONFIDENCE_DEFINITION_2D,
                    reference_frame="(0, 0) is the top-left corner of the image; +x right, +y down.",
                    unit="pixels",
                    starting_time=starting_time,
                )
                for keypoint, name in enumerate(KEYPOINT_NAMES)
            ]
            per_camera.append(
                PoseEstimation(
                    name=f"PoseEstimation_{camera.name}",
                    description=(
                        f"2D keypoints of the participant in {camera.name}. NaN where OpenPose did not find a keypoint "
                        "or where this camera had no detection of the participant."
                    ),
                    pose_estimation_series=series,
                    scorer="BODY_25B",
                    source_software="OpenPose",
                    device=device,
                    skeleton=skeleton,
                )
            )

        world = [
            self._series(
                name=name,
                description=f"{name} triangulated from the cameras that agree on it; not filtered.",
                data=track.keypoints_3d[:, keypoint],
                confidence=track.confidence_3d[:, keypoint],
                confidence_definition=CONFIDENCE_DEFINITION_3D,
                reference_frame="Calibration world frame: +z is up, in metres; the origin is set by the calibration.",
                unit="meters",
                starting_time=starting_time,
            )
            for keypoint, name in enumerate(KEYPOINT_NAMES)
        ]
        behavior.add(
            MultiCameraPoseEstimation(
                name="MultiCameraPoseEstimation",
                description=(
                    "3D keypoints of the participant, from a confidence-weighted direct linear transform over the "
                    "views selected by multi-view association. NaN where fewer than two views agree on a point."
                ),
                pose_estimation_series=world,
                pose_estimations=per_camera,
                scorer="BODY_25B",
                source_software="movement-to-nwb (2D keypoints from OpenPose)",
                skeleton=skeleton,
            )
        )

        camera_order = ", ".join(camera.name for camera in self.cameras)
        behavior.add(
            TimeSeries(
                name="AssociationSelectedDetection",
                description=(
                    f"Index of the detection chosen in each camera's OpenPose file, columns in order {camera_order}. "
                    "-1 means no detection in that camera agreed with the others in that frame."
                ),
                data=track.selected,
                unit="n.a.",
                starting_time=starting_time,
                rate=self.frame_rate,
            )
        )
        behavior.add(
            TimeSeries(
                name="AssociationReprojectionError",
                description=(
                    "Median reprojection error of the views used for the 3D keypoints in each frame, "
                    "in undistorted pixels."
                ),
                data=track.reprojection_error,
                unit="pixels",
                starting_time=starting_time,
                rate=self.frame_rate,
            )
        )
