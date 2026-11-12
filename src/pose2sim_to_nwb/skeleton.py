"""The OpenPose BODY_25B keypoint layout, as defined by OpenPose itself.

Names and order: POSE_BODY_25B_BODY_PARTS in src/openpose/pose/poseParameters.cpp.
Edges: POSE_BODY_25B_PAIRS_RENDER_GPU in include/openpose/pose/poseParametersRender.hpp. These are the
lines OpenPose draws, and they do not join the face (nose, eyes, ears) to the rest of the body.
"""

KEYPOINT_NAMES = (
    "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "Neck", "HeadTop", "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
)  # fmt: skip

SKELETON_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14),
    (13, 15), (14, 16), (15, 19), (19, 20), (15, 21), (16, 22), (22, 23), (16, 24), (5, 17), (6, 17), (17, 18),
    (11, 12),
)  # fmt: skip
