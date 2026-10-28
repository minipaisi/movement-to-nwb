import pytest

from eegmmidb_to_nwb.channels import clean_channel_name, get_template_position
from conftest import PHYSIONET_LABELS


@pytest.mark.parametrize(
    "raw, clean",
    [("Fc5.", "FC5"), ("Fpz.", "Fpz"), ("Cz..", "Cz"), ("Iz..", "Iz"), ("T10.", "T10"), ("Afz.", "AFz"), ("Po7.", "PO7")],
)
def test_clean_channel_name(raw, clean):
    assert clean_channel_name(raw) == clean


def test_every_physionet_label_has_a_unique_template_position():
    names = [clean_channel_name(label) for label in PHYSIONET_LABELS]
    assert len(set(names)) == 64
    positions = [get_template_position(name) for name in names]
    assert len(set(positions)) == 64
    # Sanity check on the frame: C3 is on the left (x < 0), C4 on the right
    assert get_template_position("C3")[0] < 0 < get_template_position("C4")[0]


@pytest.mark.parametrize("bad", ["", "...", "EOG", "123"])
def test_unexpected_labels_raise(bad):
    with pytest.raises(ValueError):
        clean_channel_name(bad)
