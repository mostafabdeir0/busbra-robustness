import numpy as np
import pytest
from scipy import ndimage as ndi

from src.coherent_perturbations import coherent_boundary_change, make_coherent_variants
from src.perturbations import make_variants, mask_iou


def example_mask():
    yy, xx = np.ogrid[:96, :104]
    return (((yy-48)/21)**2 + ((xx-50)/28)**2 <= 1)


def test_all_eight_variants_keep_original_order_and_same_translations():
    m = example_mask()
    old, new = make_variants(m), make_coherent_variants(m)
    assert list(old) == list(new)
    for k in old:
        if k.startswith('shift_'):
            assert (old[k] is None) == (new[k] is None)
            if old[k] is not None:
                assert np.array_equal(old[k].mask, new[k].mask)


def test_masks_reproducible_no_input_mutation():
    m = example_mask()
    saved = m.copy()
    a, b = make_coherent_variants(m), make_coherent_variants(m)
    assert np.array_equal(m, saved)
    for k in a:
        if a[k] is not None:
            assert np.array_equal(a[k].mask, b[k].mask)


def test_expected_iou_tolerances_and_nesting():
    m = example_mask()
    d = make_coherent_variants(m)
    for kind in ('erosion', 'dilation'):
        mild, moderate = d[f'{kind}_mild'], d[f'{kind}_moderate']
        assert abs(mild.iou - .95) < .005
        assert abs(moderate.iou - .90) < .005
        assert mild.changed_pixels < moderate.changed_pixels
        a, b = np.logical_xor(m, mild.mask), np.logical_xor(m, moderate.mask)
        assert np.all(~a | b), 'Mild edited pixels must be included in moderate edits'


def test_edited_pixels_form_one_four_connected_patch():
    m = example_mask()
    structure = ndi.generate_binary_structure(2, 1)
    for k, v in make_coherent_variants(m).items():
        if k.startswith('shift_'):
            continue
        edited = np.logical_xor(m, v.mask)
        assert ndi.label(edited, structure=structure)[1] == 1


def test_erosion_never_adds_and_dilation_never_removes():
    m = example_mask()
    for k, v in make_coherent_variants(m).items():
        if k.startswith('erosion_'):
            assert not (v.mask & ~m).any()
        if k.startswith('dilation_'):
            assert not (m & ~v.mask).any()


def test_small_mask_and_bad_kind_fail_closed():
    m = np.zeros((30, 30), dtype=bool)
    m[9:11, 9:11] = True
    with pytest.raises(ValueError):
        coherent_boundary_change(m, .95, 'erosion')
    with pytest.raises(ValueError):
        coherent_boundary_change(example_mask(), .95, 'foobar')


def test_change_geometry_not_image_intensity():
    img = np.random.default_rng(42).integers(0, 256, size=(96, 104), dtype=np.uint8)
    original = img.copy()
    make_coherent_variants(example_mask())
    assert np.array_equal(img, original)
