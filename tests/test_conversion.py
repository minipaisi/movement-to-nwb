import os
from pathlib import Path

import mne
import numpy as np
import pytest
from conftest import write_synthetic_edf
from nwbinspector import Importance, inspect_nwbfile, load_config
from pynwb import NWBHDF5IO

from eegmmidb_to_nwb import convert_run, convert_subject, parse_edf_name


def read_nwb(path):
    io = NWBHDF5IO(path, "r")
    return io, io.read()


def test_parse_edf_name():
    assert parse_edf_name("S001R03.edf") == ("S001", 3)
    with pytest.raises(ValueError):
        parse_edf_name("S1R3.edf")


def test_conversion_round_trip(synthetic_run, tmp_path):
    nwb_path = convert_run(synthetic_run, tmp_path / "nwb")
    assert nwb_path == tmp_path / "nwb" / "sub-S001" / "sub-S001_ses-R03_ecephys.nwb"

    io, nwbfile = read_nwb(nwb_path)
    try:
        series = nwbfile.acquisition["ElectricalSeries"]
        assert series.rate == 160.0
        assert series.data.shape == (int(12 * 160), 64)

        # Values in volts match what MNE reads from the same EDF file
        expected = mne.io.read_raw_edf(synthetic_run, verbose="error").get_data().T
        volts = series.data[:] * series.conversion + series.offset
        np.testing.assert_allclose(volts, expected, atol=1e-8)

        electrodes = nwbfile.electrodes.to_dataframe()
        assert electrodes["channel_name"].iloc[0] == "FC5"
        assert electrodes["edf_label"].iloc[0] == "Fc5."
        assert set(electrodes["group_name"]) == {"EEG"}

        trials = nwbfile.trials.to_dataframe()
        assert list(trials["annotation_code"]) == ["T0", "T2", "T0", "T1"]
        assert list(trials["condition"]) == ["rest", "right_fist", "rest", "left_fist"]
        assert list(trials["movement_type"]) == ["none", "executed", "none", "executed"]
        np.testing.assert_allclose(trials["start_time"], [0, 3, 6, 9])

        assert nwbfile.session_id == "S001R03"
        assert nwbfile.subject.subject_id == "S001"
        assert nwbfile.session_start_time.tzinfo is not None
    finally:
        io.close()


def test_imagined_run_and_other_sampling_rate(tmp_path):
    # Run 6 = imagined fists/feet; subjects 88, 92 and 100 have task runs at 128 Hz
    edf = write_synthetic_edf(tmp_path / "S088R06.edf", sampling_rate=128.0)
    io, nwbfile = read_nwb(convert_run(edf, tmp_path / "nwb"))
    try:
        assert nwbfile.acquisition["ElectricalSeries"].rate == 128.0
        trials = nwbfile.trials.to_dataframe()
        assert list(trials["condition"]) == ["rest", "both_feet", "rest", "both_fists"]
        assert set(trials["movement_type"]) == {"none", "imagined"}
    finally:
        io.close()


def test_unexpected_annotation_is_rejected(tmp_path):
    # Baseline run 1 should only contain T0
    edf = write_synthetic_edf(tmp_path / "S001R01.edf", codes=("T0", "T1"))
    with pytest.raises(ValueError, match="annotation codes"):
        convert_run(edf, tmp_path / "nwb")


def critical_messages(nwb_path):
    return [
        message
        for message in inspect_nwbfile(nwbfile_path=nwb_path, config=load_config("dandi"))
        if message.importance == Importance.CRITICAL
    ]


def test_nwb_inspector_only_flags_the_missing_age(synthetic_run, tmp_path):
    nwb_path = convert_run(synthetic_run, tmp_path / "nwb")
    assert [message.check_function_name for message in critical_messages(nwb_path)] == ["check_subject_age"]


def test_nwb_inspector_passes_when_age_is_known(synthetic_run, tmp_path):
    nwb_path = convert_run(synthetic_run, tmp_path / "nwb", subject_age="P30Y")
    assert critical_messages(nwb_path) == []


@pytest.mark.skipif("EEGMMIDB_DATA" not in os.environ, reason="set EEGMMIDB_DATA to a folder with S001/ to run")
def test_real_subject(tmp_path):
    paths = convert_subject(Path(os.environ["EEGMMIDB_DATA"]) / "S001", tmp_path / "nwb")
    assert len(paths) == 14
    io, nwbfile = read_nwb(paths[2])  # run 3
    try:
        trials = nwbfile.trials.to_dataframe()
        assert len(trials) == 30
        assert set(trials["condition"]) == {"rest", "left_fist", "right_fist"}
    finally:
        io.close()
