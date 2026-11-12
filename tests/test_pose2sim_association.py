import numpy as np
import pytest

from pose2sim_to_nwb.association import associate_participant, triangulate
from pose2sim_to_nwb.calibration import Camera

N_KEYPOINTS = 25


def look_at(position, target=(0.0, 0.0, 1.0), name="cam"):
    """A camera in the OpenCV convention at position, looking at target, with some lens distortion."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    intrinsic = np.array([[1500.0, 0.0, 540.0], [0.0, 1500.0, 960.0], [0.0, 0.0, 1.0]])
    return Camera(name, (1088, 1920), intrinsic, rotation, -rotation @ position, np.array([-0.05, 0.01, 0.001, -0.001]))


@pytest.fixture
def cameras():
    positions = [(3, 0, 1.8), (0, 3, 1.8), (-3, 0, 1.8), (0, -3, 1.8)]
    return [look_at(position, name=f"cam_0{index + 1}") for index, position in enumerate(positions)]


def body(centre=(0.0, 0.0, 1.0), seed=0):
    """25 points spread over a body-sized volume."""
    return np.asarray(centre) + np.random.default_rng(seed).uniform(
        [-0.3, -0.3, -0.9], [0.3, 0.3, 0.8], (N_KEYPOINTS, 3)
    )


def detect(camera, points, confidence=0.8):
    return np.column_stack([camera.project(points), np.full(len(points), confidence)])


def test_triangulate_recovers_a_point(cameras):
    point = np.array([0.1, -0.2, 1.3])
    matrices = np.stack([camera.projection_matrix for camera in cameras])
    homogeneous = matrices @ np.append(point, 1.0)
    pixels = homogeneous[:, :2] / homogeneous[:, 2:]
    assert triangulate(matrices, pixels, np.array([0.9, 0.5, 0.7, 1.0])) == pytest.approx(point)


def test_clean_views_recover_the_3d_points_through_lens_distortion(cameras):
    points = body()
    frames = [[detect(camera, points)[None]] for camera in cameras]
    track = associate_participant(cameras, frames)
    assert track.selected.tolist() == [[0, 0, 0, 0]]
    assert track.keypoints_3d[0] == pytest.approx(points, abs=1e-6)
    assert track.reprojection_error[0] < 1e-5
    assert track.confidence_3d[0] == pytest.approx(0.8)


def test_participant_is_found_wherever_openpose_lists_them(cameras):
    participant = body()
    expected = [1, 0, 1, 0]
    frames = []
    for index, (camera, position) in enumerate(zip(cameras, expected, strict=True)):
        # A different bystander in each camera, so the only 3D agreement is on the participant.
        people = [detect(camera, body(centre=(0.8, -0.6, 1.0), seed=index + 1))]
        people.insert(position, detect(camera, participant))
        frames.append([np.stack(people)])
    track = associate_participant(cameras, frames)
    assert track.selected.tolist() == [expected]
    assert track.keypoints_3d[0] == pytest.approx(participant, abs=1e-6)


def test_identity_is_decided_per_frame(cameras):
    participant = body()
    bystander = body(centre=(1.2, 1.2, 1.0), seed=1)
    sparse = detect(cameras[0], bystander)
    sparse[3:, 2] = 0.0
    frames = [[] for _ in cameras]
    for frame_order in (0, 1):
        for index, camera in enumerate(cameras):
            people = [detect(camera, participant)]
            if index == 0:
                people.insert(frame_order, sparse)
            frames[index].append(np.stack(people))
    track = associate_participant(cameras, frames)
    assert track.selected[:, 0].tolist() == [1, 0]


def test_a_sparse_bystander_cannot_win_with_a_lower_error(cameras):
    participant = body()
    noisy = [detect(camera, participant) for camera in cameras]
    noisy[0][:, :2] += np.random.default_rng(3).normal(scale=4.0, size=(N_KEYPOINTS, 2))
    # This detection has only the participant's first three points, without noise, so its error is the lowest.
    sparse = detect(cameras[0], participant)
    sparse[3:, 2] = 0.0
    frames = [[np.stack([sparse, noisy[0]])], [noisy[1][None]], [noisy[2][None]], [noisy[3][None]]]
    track = associate_participant(cameras, frames)
    assert track.selected.tolist() == [[1, 0, 0, 0]]


def test_camera_without_detection_is_marked_and_the_others_still_work(cameras):
    points = body()
    frames = [[detect(camera, points)[None]] for camera in cameras[:3]] + [[np.empty((0, N_KEYPOINTS, 3))]]
    track = associate_participant(cameras, frames)
    assert track.selected.tolist() == [[0, 0, 0, -1]]
    assert np.isnan(track.keypoints_2d[0, 3, :, :2]).all()
    assert (track.keypoints_2d[0, 3, :, 2] == 0).all()
    assert track.keypoints_3d[0] == pytest.approx(points, abs=1e-6)


def test_frame_with_nobody_is_all_nan(cameras):
    frames = [[np.empty((0, N_KEYPOINTS, 3))] for _ in cameras]
    track = associate_participant(cameras, frames)
    assert track.selected.tolist() == [[-1, -1, -1, -1]]
    assert np.isnan(track.keypoints_3d).all()
    assert (track.confidence_3d == 0).all()
    assert np.isnan(track.reprojection_error).all()


def test_confident_outlier_is_dropped_from_3d_but_kept_in_2d(cameras):
    points = body()
    views = [detect(camera, points) for camera in cameras]
    views[0][4] = [0.0, 436.77, 0.9]  # the glitch seen in the real demo
    track = associate_participant(cameras, [[view[None]] for view in views])
    assert track.keypoints_3d[0, 4] == pytest.approx(points[4], abs=1e-6)
    assert track.keypoints_2d[0, 0, 4].tolist() == [0.0, 436.77, 0.9]


def test_two_views_that_disagree_give_no_3d_point(cameras):
    points = body()
    views = [detect(camera, points) for camera in cameras]
    views[2][7, 2] = views[3][7, 2] = 0.0  # keypoint 7 only reaches the first two cameras...
    views[0][7, :2] += 300.0  # ...and they do not agree
    track = associate_participant(cameras, [[view[None]] for view in views])
    assert np.isnan(track.keypoints_3d[0, 7]).all()
    assert track.confidence_3d[0, 7] == 0.0
    assert np.isfinite(track.keypoints_3d[0, 6]).all()


def test_low_confidence_views_are_ignored(cameras):
    points = body()
    views = [detect(camera, points) for camera in cameras]
    views[0][2, :2] += 300.0
    views[0][2, 2] = 0.1  # below the threshold, so this view is never used
    track = associate_participant(cameras, [[view[None]] for view in views])
    assert track.keypoints_3d[0, 2] == pytest.approx(points[2], abs=1e-6)
    assert track.confidence_3d[0, 2] == pytest.approx(0.8)


def test_a_bystander_who_is_the_only_detection_in_a_camera_is_not_taken_for_the_participant(cameras):
    participant = body()
    bystander = body(centre=(1.0, -0.8, 1.0), seed=1)
    frames = [[detect(camera, participant)[None]] for camera in cameras[:3]] + [[detect(cameras[3], bystander)[None]]]
    track = associate_participant(cameras, frames)
    assert track.selected.tolist() == [[0, 0, 0, -1]]
    assert np.isnan(track.keypoints_2d[0, 3, :, :2]).all()
    assert track.keypoints_3d[0] == pytest.approx(participant, abs=1e-6)


def test_person_seen_by_one_camera_cannot_be_associated(cameras):
    frames = [[detect(cameras[0], body())[None]]] + [[np.empty((0, N_KEYPOINTS, 3))] for _ in cameras[1:]]
    track = associate_participant(cameras, frames)
    assert track.selected.tolist() == [[-1, -1, -1, -1]]
    assert np.isnan(track.keypoints_3d).all()


def test_a_stray_agreeing_view_does_not_make_a_bystander_the_participant(cameras):
    participant = body()
    bystander = detect(cameras[3], body(centre=(1.0, -0.8, 1.0), seed=1))
    bystander[0] = detect(cameras[3], participant)[0]  # one of its 25 keypoints lands on the participant by chance
    frames = [[detect(camera, participant)[None]] for camera in cameras[:3]] + [[bystander[None]]]
    assert associate_participant(cameras, frames).selected.tolist() == [[0, 0, 0, -1]]


def test_a_rejected_camera_does_not_contribute_to_the_3d_points(cameras):
    participant = body()
    bystander = detect(cameras[3], body(centre=(1.0, -0.8, 1.0), seed=1))
    # Keypoint 0 of the bystander lands near the participant's, close enough to agree but not exactly.
    bystander[0] = detect(cameras[3], participant)[0] + [10.0, 10.0, 0.0]
    frames = [[detect(camera, participant)[None]] for camera in cameras[:3]] + [[bystander[None]]]
    track = associate_participant(cameras, frames)
    assert track.selected.tolist() == [[0, 0, 0, -1]]
    assert track.keypoints_3d[0, 0] == pytest.approx(participant[0], abs=1e-6)
