# movement-to-nwb

Two conversions of open human movement data to [NWB](https://www.nwb.org/) with
[NeuroConv](https://github.com/catalystneuro/neuroconv), each ready for [NWB
Inspector](https://github.com/NeurodataWithoutBorders/nwbinspector) and [DANDI](https://dandiarchive.org/):

| Package | Source data | NWB content |
|---|---|---|
| [`eegmmidb_to_nwb`](#eeg-motor-movementimagery-eegmmidb_to_nwb) | 64-channel EEG in EDF+, annotations whose meaning depends on the run | `ElectricalSeries`, electrodes, trials |
| [`pose2sim_to_nwb`](#multi-camera-pose-estimation-pose2sim_to_nwb) | OpenPose JSON from four cameras, Qualisys calibration | 2D and 3D keypoints with [`ndx-pose`](https://github.com/rly/ndx-pose), calibrated cameras |

```bash
pip install -e ".[test]"
pytest
```

The EEG data needs a download; a real multi-camera trial ships in `tests/data/`, so all of the pose tests run on it
without one.

---

# Multi-camera pose estimation (`pose2sim_to_nwb`)

Converts a trial from the [Pose2Sim](https://github.com/perfanalytics/pose2sim) 0.8.0 demo to NWB: a person
balancing on a beam, filmed by four calibrated cameras at 60 fps for 100 frames, with OpenPose BODY_25B keypoints.
I spent my Master's thesis turning clinical video into 3D movement, where most of the difficulty was metadata that
nobody had kept: which camera, which calibration, which person. That code is not public, so this is the same kind
of problem on open data, stored in a format where the metadata stays with the keypoints.

```bash
pose2sim-demo-download data/                   # about 43 MB, checked against a SHA-256
pose2sim-to-nwb data/S01_Demo_SingleTrial nwb/ --session-start-time 2024-01-01T00:00:00+00:00
nwbinspector nwb/ --config dandi
```

Output: `nwb/sub-Pose2SimDemo/sub-Pose2SimDemo_ses-S01_behavior.nwb` (1.3 MB).

## What ends up in the NWB file

| NWB location | Content |
|---|---|
| `general/devices/cam_01` to `cam_04` | `CalibratedCamera`: intrinsics, world-to-camera rotation and translation (metres), distortion, in the OpenCV convention |
| `processing/behavior/MultiCameraPoseEstimation` | 3D keypoints in metres, with a confidence for each |
| `.../MultiCameraPoseEstimation/PoseEstimation_cam_01` to `_cam_04` | The participant's 2D keypoints exactly as OpenPose wrote them, with confidence, linked to their camera |
| `processing/behavior/Skeletons` | OpenPose's BODY_25B names and edges |
| `processing/behavior/AssociationSelectedDetection` | Which detection in each camera's JSON file was chosen, frame by frame, so every value traces back to its source |
| `processing/behavior/AssociationReprojectionError` | Median reprojection error per frame, to judge each frame's quality |

## Decisions, and the quirks behind them

**OpenPose does not say who is who.** Each JSON file is a list of detections in no guaranteed order. cam01 and
cam02 see a second person in every frame, and in frame 37 OpenPose splits the participant into two detections in
cam01, while cam03 and cam04 see one person. Taking the first detection happens to work on this demo, because
OpenPose lists the participant first in all 100 frames, but nothing promises that. Since the cameras are
calibrated, `association.py` tries each combination of one detection per camera and keeps the one with the most
keypoint views agreeing on a 3D point. It counts agreeing views rather than taking the lowest error, because a
detection with few keypoints gets a low error easily. A camera whose only detection is someone else is marked `-1`
instead of being accepted because nothing else was available. The tests shuffle the order and add bystanders.

**Confident keypoints can still be wrong.** cam01's nose has x = 0 exactly in six frames (4 to 8 and 15), with a
real y and a confidence above 0.6. These values are kept in the 2D data, because that is what OpenPose reported,
but not used for 3D: each 3D point comes from the largest set of views that agree, and if a final pair of cameras
disagrees the point is NaN rather than wrong (12 of 2500 points).

**(0, 0, 0) means "not found", but (0, y, c) does not.** Zeros in a missing keypoint would plot as a real position
in the corner of the image, so they become NaN with confidence 0. Only an exact (0, 0, 0) is treated that way.

**The calibration needed converting, and the conversion is checked rather than assumed.** A Qualisys `.qca.txt`
stores intrinsics in 1/64 pixel, positions in millimetres, and a camera that looks along -z with +y up.
`calibration.py` converts this to the OpenCV convention that `CalibratedCamera` documents. The result is tested by
comparing with `cv2.projectPoints` and `cv2.undistortPoints`, and on real data by every keypoint landing in front
of every camera. The format does not document how distortion is scaled; I follow Pose2Sim in dividing by 64, and on
this demo the choice changes the median reprojection error by less than 0.1 pixel.

**The release has no recording date.** NWB requires one, and I would rather not invent it, so
`--session-start-time` is required and has no default. The date in the example above is a placeholder: use the real
one if the authors can provide it. The same goes for subject age, which is left empty unless passed in.

**Stray files and gaps stop the conversion, with one exception.** cam03 and cam04 contain `del_000000_0.json` and
`del_000001_0.json`, empty placeholders left in the release. They are skipped only if they really are empty. A
missing frame, an unknown file, a different pose model, or cameras covering different frames raise an error instead
of shifting the time base.

**Time comes from the config.** The 60 fps rate is read from `Config.toml`, and the trial has no video or per-frame
clock to check it against. Frames with the same number are treated as simultaneous across cameras. I did not apply
any offset between cameras: shifting the output against Pose2Sim's result by 1 or 2 frames in either direction
makes the match worse, so the frames line up with theirs.

## Validation

- **Against Pose2Sim itself.** Release 0.4.0 includes Pose2Sim's 3D result for this trial, and its 400 JSON files
  and calibration have the same git hashes as the ones here. Over the 21 keypoints it reports, the median distance
  to my points is 9.1 mm, and 23.9 mm for the worst keypoint. They should not be identical: Pose2Sim also filters
  its result, and these points are not filtered.
- **Lossless 2D.** Every value in the NWB file's selected detections is compared with the JSON files.
- **Anatomy.** The head is above the neck, hips, knees and ankles. Median thigh and shin lengths are between 0.3
  and 0.5 m, with a standard deviation below 3.5 cm while the person moves: that spread is the jitter of unfiltered
  3D points.
- **Reprojection.** The median error per frame is 8.5 pixels and never above 10.6 pixels.
- **Format.** PyNWB schema validation passes. NWB Inspector's DANDI configuration reports one critical issue, the
  missing subject age, and none once `--subject-age` is given. It also suggests adding an experimenter and an
  institution; the release does not state them, so they are left empty.

## Limits

These numbers come from one trial. The 20-pixel reprojection threshold and the 0.3 confidence threshold are the
demo's own settings, while the 25% agreement rule for accepting a camera's detection was set from this trial alone:
a true detection reaches at least 33%, a bystander at most 19%. cam03 agrees least with the others (a median of 62%
of its usable keypoints). The 3D points are for checking and quick use, not a replacement for Pose2Sim's filtering
and OpenSim steps. Only BODY_25B is supported.

---

# EEG Motor Movement/Imagery (`eegmmidb_to_nwb`)

Converts the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
(EEGMMIDB: 109 subjects, 64-channel EEG, 14 runs each, EDF+) to [NWB](https://www.nwb.org/) with
[NeuroConv](https://github.com/catalystneuro/neuroconv), ready for validation with [NWB
Inspector](https://github.com/NeurodataWithoutBorders/nwbinspector) and upload to
[DANDI](https://dandiarchive.org/).

I used this dataset for my bachelor thesis (cross-task transfer learning between motor imagery and motor
execution). Back then I packed it into a custom HDF5 file; this repository does it properly, with a community
standard that keeps the metadata next to the signal.

## Quick start

```bash
eegmmidb-download 1 data/        # 14 EDF files for subject S001 (about 32 MB)
eegmmidb-to-nwb data/S001 nwb/   # one NWB file per run
nwbinspector nwb/ --config dandi
```

Or from Python:

```python
from eegmmidb_to_nwb import convert_run

convert_run("data/S001/S001R03.edf", "nwb/", subject_age="P25Y")  # age if you know it
```

Output follows the DANDI layout: `nwb/sub-S001/sub-S001_ses-R03_ecephys.nwb`.

## What ends up in each NWB file

| NWB location | Content |
|---|---|
| `acquisition/ElectricalSeries` | Raw EEG, 64 channels, stored as the original int16 values with the conversion factor to volts (lossless, about 30% smaller than the EDF) |
| `electrodes` | One row per channel: cleaned 10-10 name (`FC5`), original EDF label (`Fc5.`), template x/y/z position, electrode group `EEG` |
| `trials` | One row per EDF annotation: start/stop time, `annotation_code` (T0/T1/T2), `condition` (rest, left_fist, right_fist, both_fists, both_feet), `movement_type` (executed, imagined, none) |
| `session_description`, `session_id` | Run number, task and instructions, e.g. `S001R04`: imagined left/right fist |
| `subject` | Subject id, species; sex unknown (`U`) |

## Decisions, and the quirks behind them

**The meaning of an annotation depends on the run number.** The EDF files only say T0, T1 or T2. T1 is the left
fist in runs 3, 4, 7, 8, 11, 12 but both fists in runs 5, 6, 9, 10, 13, 14, and whether the movement was executed
or imagined is also only implied by the run number. That mapping lives on the PhysioNet page, not in the data.
`runs.py` encodes it once, and the trials table stores both the raw code and its meaning, so nobody has to look it
up again. A run containing a code it should not have (e.g. T1 in a baseline run) is rejected instead of silently
mislabelled.

**One NWB file per run, not per subject.** Every run of a subject carries the same start time in its EDF header
(S001: 2009-08-12 16:15:00 for all 14 runs). The real gaps between runs were not recorded, so putting the runs on
one timeline would mean inventing them. Each run is its own session instead.

**Time zone.** EDF headers have none. The data were recorded at the Wadsworth Center (Albany, NY), so the start
time is stored as America/New_York. Lab and experimenter come from the PhysioNet page.

**Channel names.** The headers pad labels with dots and use mixed case (`Fc5.`, `C5..`, `Fpz.`). They are converted
to standard 10-10 names (`FC5`, `C5`, `Fpz`); the original label is kept in `edf_label`.

**Electrode positions are a template.** Positions come from MNE's 10-05 template montage (`colin27_1005`, formerly
`standard_1005`), not from measurements on each subject, and are stored in custom `template_x/y/z` columns with
that stated in their descriptions, rather than in the `x/y/z` columns that NWB reserves for measured positions.

**Not every subject is the same.** Subjects 88, 92 and 100 have their task runs recorded at 128 Hz (their baseline
runs are at 160 Hz) with a different trial structure (shorter rest periods, more trials). The converter reads the
sampling rate and the annotations from each file instead of assuming 160 Hz, and a test covers the 128 Hz case.

**Subject age is not published.** NWB Inspector's DANDI configuration flags missing age as critical, and the
dataset does not provide it. Rather than guess, the converter leaves it empty and accepts `subject_age=` (or
`--subject-age`) for when the data owner supplies it. With an age, the files pass with no critical issues. The only
remaining suggestion is "trials table has a single row" for the two baseline runs, which is correct: a one-minute
baseline is one trial.

## Validation

- `pytest -k "not pose2sim"` (29 tests, plus the optional one below): run mapping, channel name cleaning against
  all 64 PhysioNet labels, a full conversion of a synthetic EDF file with a sample-by-sample comparison of the
  signal against MNE's reading of the same file, the 128 Hz case, rejection of unexpected annotations, and NWB
  Inspector (DANDI config) on the output.
- An optional test converts a real subject: `EEGMMIDB_DATA=data pytest -k real`.
- Checked by hand on subjects S001, S002, S088, S100 and S104.

## Possible next steps

- Convert all 109 subjects and upload them as a DANDI dataset (`dandi organize` / `dandi upload`), after confirming
  with the dataset maintainers what subject metadata can be added.
- Add the BCI2000 stimulus details (target position on screen) if the original `.dat` files are available.

---

# How it was built

I built this with Claude Code as a pair programmer and reviewed the conversion decisions above. Every behaviour is
pinned by a test or checked against an outside reference (MNE, OpenCV, Pose2Sim's own output, NWB Inspector) rather
than taken on trust.

# Data citations

**Pose2Sim demo.** Pagnon D, Domalain M, Reveret L. Pose2Sim: An open-source Python package for multiview
markerless kinematics. *Journal of Open Source Software* 7(77):4362, 2022.
[doi:10.21105/joss.04362](https://doi.org/10.21105/joss.04362). Pagnon D, Domalain M, Reveret L. Pose2Sim: An
End-to-End Workflow for 3D Markerless Sports Kinematics, Part 2: Accuracy. *Sensors* 22(7):2712, 2022.
[doi:10.3390/s22072712](https://doi.org/10.3390/s22072712). Data and reference result used under the BSD 3-Clause
license, copied in `tests/data/S01_Demo_SingleTrial/LICENSE`.

**EEG Motor Movement/Imagery.** Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. BCI2000: A
General-Purpose Brain-Computer Interface (BCI) System. *IEEE Transactions on Biomedical Engineering*
51(6):1034-1043, 2004. [doi:10.1109/TBME.2004.827072](https://doi.org/10.1109/TBME.2004.827072)

Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215-e220, 2000. Dataset:
[doi:10.13026/C28G6P](https://doi.org/10.13026/C28G6P), Open Data Commons Attribution License v1.0.

Code: MIT license.
