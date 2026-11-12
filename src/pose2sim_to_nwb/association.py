"""Find the same person across cameras in each frame, and triangulate that person's keypoints.

OpenPose does not track identity: person 0 in one file can be someone else in the next, and bystanders
appear as extra detections. The cameras are calibrated, so a true match is the combination of one
detection per camera whose keypoints agree on the same 3D points.
"""

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from .calibration import Camera

# Same values as the demo's Config.toml (likelihood_threshold_association, reproj_error_threshold_association).
MIN_CONFIDENCE = 0.3
MAX_REPROJECTION_ERROR_PIXELS = 20.0
MIN_VIEWS = 2
# On the demo a true detection has at least 33% of its usable keypoints in agreement, and a bystander at most 19%.
MIN_AGREEING_FRACTION = 0.25


@dataclass(frozen=True)
class ParticipantTrack:
    """The result for n_frames frames, n_cameras cameras and n_keypoints keypoints.

    selected[f, c] is the index of the chosen detection in that camera's frame, or -1 if none agrees with the
    other cameras.
    keypoints_2d has shape (n_frames, n_cameras, n_keypoints, 3): x, y, confidence as read; NaN, NaN, 0 where
    selected is -1.
    keypoints_3d has shape (n_frames, n_keypoints, 3) in metres, NaN without at least MIN_VIEWS views that agree.
    confidence_3d has shape (n_frames, n_keypoints): mean confidence of the views used, 0 if none.
    reprojection_error is the median over keypoints and views for each frame, in undistorted pixels.
    """

    selected: np.ndarray
    keypoints_2d: np.ndarray
    keypoints_3d: np.ndarray
    confidence_3d: np.ndarray
    reprojection_error: np.ndarray


def triangulate(projection_matrices: np.ndarray, pixels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted direct linear transform for one point seen in n views. Returns x, y, z."""
    x_rows = pixels[:, 0, None] * projection_matrices[:, 2] - projection_matrices[:, 0]
    y_rows = pixels[:, 1, None] * projection_matrices[:, 2] - projection_matrices[:, 1]
    system = np.concatenate([x_rows, y_rows]) * np.concatenate([weights, weights])[:, None]
    homogeneous = np.linalg.svd(system)[2][-1]
    return homogeneous[:3] / homogeneous[3]


def triangulate_views(
    cameras: list[Camera],
    projection_matrices: np.ndarray,
    views: np.ndarray,
    min_confidence: float,
    max_reprojection_error: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate every keypoint from views of shape (n_cameras, n_keypoints, 3).

    Returns the 3D points, the mean confidence of the views behind them and the reprojection error
    of each view (NaN where a view was not used), with shape (n_cameras, n_keypoints).
    """
    n_cameras, n_keypoints = views.shape[:2]
    points = np.full((n_keypoints, 3), np.nan)
    confidence = np.zeros(n_keypoints)
    errors = np.full((n_cameras, n_keypoints), np.nan)
    undistorted = np.stack([camera.undistort(views[index, :, :2]) for index, camera in enumerate(cameras)])
    for keypoint in range(n_keypoints):
        usable = np.flatnonzero((views[:, keypoint, 2] >= min_confidence) & np.isfinite(views[:, keypoint, 0]))
        # OpenPose occasionally puts a confident keypoint in the wrong place (the demo has some at exactly
        # x = 0), so this keeps the largest set of views that agree, and the closest match among sets of that
        # size. Two rays rarely meet exactly, so a pair can disagree too, and then there is no position at all.
        for size in range(len(usable), MIN_VIEWS - 1, -1):
            best_error = max_reprojection_error
            for subset in combinations(usable, size):
                views_used = list(subset)
                point = triangulate(
                    projection_matrices[views_used], undistorted[views_used, keypoint], views[views_used, keypoint, 2]
                )
                projected = projection_matrices[views_used] @ np.append(point, 1.0)
                view_errors = np.linalg.norm(
                    projected[:, :2] / projected[:, 2:] - undistorted[views_used, keypoint], axis=1
                )
                if view_errors.max() <= best_error:
                    best_error = view_errors.max()
                    points[keypoint] = point
                    confidence[keypoint] = views[views_used, keypoint, 2].mean()
                    errors[:, keypoint] = np.nan
                    errors[views_used, keypoint] = view_errors
            if np.isfinite(points[keypoint, 0]):
                break
    return points, confidence, errors


def associate_participant(
    cameras: list[Camera],
    frames_per_camera: list[list[np.ndarray]],
    min_confidence: float = MIN_CONFIDENCE,
    max_reprojection_error: float = MAX_REPROJECTION_ERROR_PIXELS,
) -> ParticipantTrack:
    """Choose one detection per camera in each frame.

    Each combination is scored by how many keypoint views end up agreeing on a triangulated point.
    Counting agreeing views, rather than taking the smallest error, stops a sparse bystander from
    winning: fewer points make a lower error easy, but they cannot agree more often. Ties go to the
    lower median error.
    """
    n_frames = len(frames_per_camera[0])
    n_cameras = len(cameras)
    n_keypoints = next(frame.shape[1] for frames in frames_per_camera for frame in frames)
    projection_matrices = np.stack([camera.projection_matrix for camera in cameras])

    selected = np.full((n_frames, n_cameras), -1, dtype=int)
    keypoints_2d = np.full((n_frames, n_cameras, n_keypoints, 3), np.nan)
    keypoints_2d[..., 2] = 0.0
    keypoints_3d = np.full((n_frames, n_keypoints, 3), np.nan)
    confidence_3d = np.zeros((n_frames, n_keypoints))
    reprojection_error = np.full(n_frames, np.nan)
    missing_view = np.zeros((n_keypoints, 3))
    missing_view[:, :2] = np.nan

    for frame_index in range(n_frames):
        frames = [frames_per_camera[camera][frame_index] for camera in range(n_cameras)]
        best_score, best_views, best_combination = None, None, None
        for combination in product(*[range(len(frame)) if len(frame) else (-1,) for frame in frames]):
            views = np.stack(
                [
                    frame[person] if person >= 0 else missing_view
                    for frame, person in zip(frames, combination, strict=True)
                ]
            )
            errors = triangulate_views(cameras, projection_matrices, views, min_confidence, max_reprojection_error)[2]
            finite = errors[np.isfinite(errors)]
            if finite.size == 0:
                continue
            score = (finite.size, -float(np.median(finite)))
            if best_score is None or score > best_score:
                best_score, best_views, best_combination = score, views, np.array(combination)
        if best_views is None:
            continue

        # A camera whose only detection is someone else leaves almost none of its views in agreement. It must not
        # count as the participant, so it is removed and the 3D points are computed without it.
        _, _, errors = triangulate_views(
            cameras, projection_matrices, best_views, min_confidence, max_reprojection_error
        )
        usable_views = ((best_views[..., 2] >= min_confidence) & np.isfinite(best_views[..., 0])).sum(axis=1)
        agreeing_views = np.isfinite(errors).sum(axis=1)
        agrees = (agreeing_views > 0) & (agreeing_views >= MIN_AGREEING_FRACTION * usable_views)
        participant_views = np.where(agrees[:, None, None], best_views, missing_view)
        points, confidence, errors = triangulate_views(
            cameras, projection_matrices, participant_views, min_confidence, max_reprojection_error
        )
        selected[frame_index] = np.where(agrees, best_combination, -1)
        keypoints_2d[frame_index] = participant_views
        keypoints_3d[frame_index] = points
        confidence_3d[frame_index] = confidence
        if np.isfinite(errors).any():
            reprojection_error[frame_index] = np.nanmedian(errors)

    return ParticipantTrack(selected, keypoints_2d, keypoints_3d, confidence_3d, reprojection_error)
