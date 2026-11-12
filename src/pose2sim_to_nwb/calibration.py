"""Read a Qualisys .qca.txt calibration and express it in the OpenCV camera convention."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# QCA intrinsics are stored in sensor units of 1/64 pixel.
QCA_UNITS_PER_PIXEL = 64.0
MILLIMETRES_PER_METRE = 1000.0
# A QCA camera looks along its -z axis with +y up; an OpenCV camera looks along +z with +y down.
QCA_TO_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class Camera:
    """A pinhole camera with radial-tangential distortion, in the OpenCV convention.

    A world point X (metres) maps to camera coordinates as rotation @ X + translation.
    distortion holds (k1, k2, p1, p2).
    """

    name: str
    image_size: tuple[int, int]  # (width, height) in pixels
    intrinsic_matrix: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    distortion: np.ndarray

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates (metres)."""
        return -self.rotation.T @ self.translation

    @property
    def projection_matrix(self) -> np.ndarray:
        """The 3x4 matrix that maps homogeneous world points to undistorted pixels."""
        return self.intrinsic_matrix @ np.column_stack([self.rotation, self.translation])

    def _distort_normalized(self, normalized: np.ndarray) -> np.ndarray:
        k1, k2, p1, p2 = self.distortion
        x, y = normalized[..., 0], normalized[..., 1]
        r2 = x**2 + y**2
        radial = 1 + k1 * r2 + k2 * r2**2
        return np.stack(
            [x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2), y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y],
            axis=-1,
        )

    def project(self, world_points: np.ndarray) -> np.ndarray:
        """Project world points of shape (..., 3) to distorted pixel coordinates of shape (..., 2)."""
        in_camera = np.asarray(world_points) @ self.rotation.T + self.translation
        distorted = self._distort_normalized(in_camera[..., :2] / in_camera[..., 2:])
        focal = self.intrinsic_matrix[[0, 1], [0, 1]]
        centre = self.intrinsic_matrix[[0, 1], [2, 2]]
        return distorted * focal + centre

    def undistort(self, pixels: np.ndarray, n_iterations: int = 10) -> np.ndarray:
        """Remove lens distortion from pixels of shape (..., 2), keeping them in pixel units.

        The distortion model has no closed-form inverse, so this solves it by fixed-point iteration.
        """
        focal = self.intrinsic_matrix[[0, 1], [0, 1]]
        centre = self.intrinsic_matrix[[0, 1], [2, 2]]
        target = (np.asarray(pixels) - centre) / focal
        normalized = target.copy()
        for _ in range(n_iterations):
            normalized = normalized + target - self._distort_normalized(normalized)
        return normalized * focal + centre


def read_qca(file_path: str | Path) -> list[Camera]:
    """Read every camera of a .qca.txt file, sorted by camera number."""
    cameras = []
    for element in ET.parse(file_path).getroot().findall("cameras/camera"):
        name = element.attrib["serial"]
        field_of_view = element.find("fov_video").attrib
        intrinsic = element.find("intrinsic").attrib
        transform = element.find("transform").attrib

        left, right = float(field_of_view["left"]), float(field_of_view["right"])
        top, bottom = float(field_of_view["top"]), float(field_of_view["bottom"])
        intrinsic_matrix = np.array(
            [
                [
                    float(intrinsic["focalLengthU"]) / QCA_UNITS_PER_PIXEL,
                    0.0,
                    float(intrinsic["centerPointU"]) / QCA_UNITS_PER_PIXEL - left,
                ],
                [
                    0.0,
                    float(intrinsic["focalLengthV"]) / QCA_UNITS_PER_PIXEL,
                    float(intrinsic["centerPointV"]) / QCA_UNITS_PER_PIXEL - top,
                ],
                [0.0, 0.0, 1.0],
            ]
        )
        # The format does not document how distortion is scaled. This follows Pose2Sim, which divides by 64;
        # on the demo either choice changes the median reprojection error by less than 0.1 pixel.
        distortion = np.array(
            [
                float(intrinsic[key]) / QCA_UNITS_PER_PIXEL
                for key in ("radialDistortion1", "radialDistortion2", "tangentalDistortion1", "tangentalDistortion2")
            ]
        )

        world_to_qca = np.array([[float(transform[f"r{row}{column}"]) for column in "123"] for row in "123"])
        position = np.array([float(transform[axis]) for axis in "xyz"]) / MILLIMETRES_PER_METRE
        if not np.allclose(world_to_qca @ world_to_qca.T, np.eye(3), atol=1e-9) or not np.isclose(
            np.linalg.det(world_to_qca), 1.0
        ):
            raise ValueError(f"Camera {name!r} in {file_path} does not have a valid rotation matrix.")
        rotation = QCA_TO_OPENCV_CAMERA @ world_to_qca

        cameras.append(
            Camera(
                name=name,
                image_size=(int(right - left + 1), int(bottom - top + 1)),
                intrinsic_matrix=intrinsic_matrix,
                rotation=rotation,
                translation=-rotation @ position,
                distortion=distortion,
            )
        )
    if not cameras:
        raise ValueError(f"No cameras found in {file_path}.")
    return sorted(cameras, key=lambda camera: int(re.sub(r"\D", "", camera.name) or 0))
