import numpy as np
import mne
import pytest

# The 64 labels exactly as they appear in the PhysioNet EDF headers, in file order
PHYSIONET_LABELS = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.", "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..",
    "C6..", "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.", "Fp1.", "Fpz.", "Fp2.", "Af7.", "Af3.",
    "Afz.", "Af4.", "Af8.", "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..", "Ft7.",
    "Ft8.", "T7..", "T8..", "T9..", "T10.", "Tp7.", "Tp8.", "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..",
    "P4..", "P6..", "P8..", "Po7.", "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2..", "Iz..",
]


def write_synthetic_edf(path, sampling_rate=160.0, duration_s=12.0, codes=("T0", "T2", "T0", "T1")):
    """Write a small EDF file shaped like a PhysioNet run: 64 channels, padded labels, T0/T1/T2 annotations."""
    rng = np.random.default_rng(0)
    n_samples = int(duration_s * sampling_rate)
    data = rng.normal(scale=20e-6, size=(len(PHYSIONET_LABELS), n_samples))  # about 20 microvolts
    info = mne.create_info(PHYSIONET_LABELS, sfreq=sampling_rate, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    step = duration_s / len(codes)
    raw.set_annotations(mne.Annotations(onset=[i * step for i in range(len(codes))], duration=[step] * len(codes), description=list(codes)))
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="error")
    return path


@pytest.fixture
def synthetic_run(tmp_path):
    """A synthetic S001R03.edf (run 3: executed left/right fist)."""
    return write_synthetic_edf(tmp_path / "S001R03.edf")
