"""EEG channel names and template positions.

The EDF headers store labels padded with dots to four characters and in mixed case
("Fc5.", "C5..", "Fpz."). This module turns them into standard 10-10 labels ("FC5",
"C5", "Fpz") and looks up template scalp positions for them.
"""

import re
from functools import lru_cache

import mne

# Labels whose standard spelling is not just "upper-case the letters": the z suffix
# for midline electrodes and the Fp prefix stay mixed case.
_PREFIX_CASE = {"FP": "Fp", "AF": "AF", "FC": "FC", "FT": "FT", "CP": "CP", "TP": "TP", "PO": "PO"}


def clean_channel_name(raw_name: str) -> str:
    """Convert an EDF label such as "Fc5." or "Fpz." to its 10-10 name ("FC5", "Fpz")."""
    name = raw_name.strip().rstrip(".")
    match = re.fullmatch(r"([A-Za-z]+?)(\d+|[zZ])", name)
    if match is None:
        raise ValueError(f"Unexpected EEG channel label: {raw_name!r}")
    prefix, suffix = match.groups()
    prefix = _PREFIX_CASE.get(prefix.upper(), prefix.upper())
    suffix = "z" if suffix in ("z", "Z") else suffix
    return prefix + suffix


@lru_cache(maxsize=1)
def _template_positions() -> dict:
    """Template 10-05 positions in metres (MNE's 10-05 montage, head coordinate frame)."""
    # MNE 1.13 renamed "standard_1005" to "colin27_1005"; the positions are the same.
    name = "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages() else "standard_1005"
    montage = mne.channels.make_standard_montage(name)
    return montage.get_positions()["ch_pos"]


def get_template_position(channel_name: str) -> tuple[float, float, float]:
    """Return the (x, y, z) template position of a cleaned 10-10 channel name, in metres."""
    positions = _template_positions()
    if channel_name not in positions:
        raise KeyError(f"{channel_name!r} is not in the 10-05 template montage.")
    x, y, z = positions[channel_name]
    return float(x), float(y), float(z)
