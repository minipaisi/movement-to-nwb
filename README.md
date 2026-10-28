# eegmmidb-to-nwb

Converts the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
(EEGMMIDB: 109 subjects, 64-channel EEG, 14 runs each, EDF+) to [NWB](https://www.nwb.org/) with
[NeuroConv](https://github.com/catalystneuro/neuroconv), ready for validation with
[NWB Inspector](https://github.com/NeurodataWithoutBorders/nwbinspector) and upload to [DANDI](https://dandiarchive.org/).

I used this dataset for my bachelor thesis (cross-task transfer learning between motor imagery and
motor execution). Back then I packed it into a custom HDF5 file; this repository does it properly, with
a community standard that keeps the metadata next to the signal.

## Quick start

```bash
pip install -e ".[test]"

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

**The meaning of an annotation depends on the run number.** The EDF files only say T0, T1 or T2. T1 is the
left fist in runs 3, 4, 7, 8, 11, 12 but both fists in runs 5, 6, 9, 10, 13, 14, and whether the movement
was executed or imagined is also only implied by the run number. That mapping lives on the PhysioNet page,
not in the data. `runs.py` encodes it once, and the trials table stores both the raw code and its meaning,
so nobody has to look it up again. A run containing a code it should not have (e.g. T1 in a baseline run)
is rejected instead of silently mislabelled.

**One NWB file per run, not per subject.** Every run of a subject carries the same start time in its EDF
header (S001: 2009-08-12 16:15:00 for all 14 runs). The real gaps between runs were not recorded, so
putting the runs on one timeline would mean inventing them. Each run is its own session instead.

**Time zone.** EDF headers have none. The data were recorded at the Wadsworth Center (Albany, NY), so the
start time is stored as America/New_York. Lab and experimenter come from the PhysioNet page.

**Channel names.** The headers pad labels with dots and use mixed case (`Fc5.`, `C5..`, `Fpz.`). They are
converted to standard 10-10 names (`FC5`, `C5`, `Fpz`); the original label is kept in `edf_label`.

**Electrode positions are a template.** Positions come from MNE's 10-05 template montage (`colin27_1005`,
formerly `standard_1005`), not from measurements on each subject, and are stored in custom
`template_x/y/z` columns with that stated in their descriptions, rather than in the `x/y/z` columns that
NWB reserves for measured positions.

**Not every subject is the same.** Subjects 88, 92 and 100 have their task runs recorded at 128 Hz (their
baseline runs are at 160 Hz) with a different trial structure (shorter rest periods, more trials). The
converter reads the sampling rate and the annotations from each file instead of assuming 160 Hz, and a test
covers the 128 Hz case.

**Subject age is not published.** NWB Inspector's DANDI configuration flags missing age as critical, and
the dataset does not provide it. Rather than guess, the converter leaves it empty and accepts
`subject_age=` (or `--subject-age`) for when the data owner supplies it. With an age, the files pass with
no critical issues. The only remaining suggestion is "trials table has a single row" for the two baseline
runs, which is correct: a one-minute baseline is one trial.

## Validation

- `pytest` (29 tests): run mapping, channel name cleaning against all 64 PhysioNet labels, a full conversion
  of a synthetic EDF file with a sample-by-sample comparison of the signal against MNE's reading of the same
  file, the 128 Hz case, rejection of unexpected annotations, and NWB Inspector (DANDI config) on the output.
- An optional test converts a real subject: `EEGMMIDB_DATA=data pytest -k real`.
- Checked by hand on subjects S001, S002, S088, S100 and S104.

## Possible next steps

- Convert all 109 subjects and upload them as a DANDI dataset (`dandi organize` / `dandi upload`), after
  confirming with the dataset maintainers what subject metadata can be added.
- Add the BCI2000 stimulus details (target position on screen) if the original `.dat` files are available.

## How it was built

I wrote this with Claude Code as a pair programmer: I chose the structure and the conversion decisions
above, and every behaviour is pinned by a test or checked against MNE and NWB Inspector rather than taken
on trust.

## Data citation

Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. BCI2000: A General-Purpose Brain-Computer
Interface (BCI) System. *IEEE Transactions on Biomedical Engineering* 51(6):1034-1043, 2004.
[doi:10.1109/TBME.2004.827072](https://doi.org/10.1109/TBME.2004.827072)

Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215-e220, 2000.
Dataset: [doi:10.13026/C28G6P](https://doi.org/10.13026/C28G6P), Open Data Commons Attribution License v1.0.

Code: MIT license.
