"""Unit tests for eight label-blind mask perturbations."""
import numpy as np
import pytest
from src.perturbations import mask_iou, make_variants, translate_no_wrap


def specimen(h=128, w=128):
    y, x = np.ogrid[:h, :w]
    return ((y - h//2)**2 / 25**2 + (x - w//2)**2 / 35**2 <= 1)


def test_identical_masks_have_iou_one():
    m = specimen()
    assert mask_iou(m, m) == 1


def test_small_shift_does_not_wrap():
    m = np.zeros((20, 20), dtype=bool)
    m[3:8, 3:8] = True
    out, clipped = translate_no_wrap(m, dy=2, dx=3)
    assert clipped == 0
    assert out.sum() == m.sum()
    assert out[5:10, 6:11].all()
    assert not out[:5, :].any()


def test_border_shift_reports_clipping():
    m = np.zeros((20,20), dtype=bool)
    m[:,0] = True
    out, clipped = translate_no_wrap(m, dy=0, dx=-1)
    assert clipped == 20
    assert not out.any()


def test_eight_distinct_named_variants():
    variants = make_variants(specimen())
    assert len(variants) == 8
    assert all(v is not None for v in variants.values())
    assert all(0 < v.iou < 1 for v in variants.values())


def test_morphological_direction_changes_area():
    m = specimen()
    v = make_variants(m)
    assert v["erosion_mild"].area_ratio < 1
    assert v["erosion_moderate"].area_ratio < 1
    assert v["dilation_mild"].area_ratio > 1
    assert v["dilation_moderate"].area_ratio > 1


def test_translation_preserves_shape_and_area_if_not_clipped():
    m = specimen()
    variants = make_variants(m)
    for direction in ("left", "right", "up", "down"):
        shifted = variants[f"shift_{direction}_moderate"]
        assert shifted is not None
        assert shifted.mask.sum() == m.sum()


def test_perturbations_do_not_mutate_original():
    m = specimen()
    original = m.copy()
    make_variants(m)
    assert np.array_equal(m, original)


def test_invalid_masks_are_rejected():
    with pytest.raises(ValueError):
        make_variants(np.zeros((64,64), dtype=bool))


def test_mild_and_moderate_are_distinct_and_nested():
    m = specimen()
    variants = make_variants(m)
    for kind in ("erosion", "dilation"):
        mild = variants[f"{kind}_mild"].mask
        moderate = variants[f"{kind}_moderate"].mask
        assert not np.array_equal(mild, moderate)
        if kind == "erosion":
            assert np.all(~moderate | mild)
        else:
            assert np.all(~mild | moderate)
        assert abs(variants[f"{kind}_mild"].iou-0.95) < 0.01
        assert abs(variants[f"{kind}_moderate"].iou-0.90) < 0.01
