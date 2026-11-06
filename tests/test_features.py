"""Tests for features.py. Run: python -m pytest -q tests/test_features.py"""
import numpy as np
import pytest

from src.features import FEATURE_COLUMNS, extract_features, preprocess_pair


def toy_input():
    image = np.full((96, 96), 50, dtype=np.uint8)
    yy, xx = np.ogrid[:96, :96]
    mask = (yy - 48) ** 2 + (xx - 48) ** 2 <= 18 ** 2
    image[mask] = 160
    return image, mask


def test_schema_and_finite_values():
    im, mask = toy_input()
    features = extract_features(im, mask)
    assert len(features) == 23
    assert tuple(features) == FEATURE_COLUMNS
    assert np.isfinite(list(features.values())).all()
    assert features["shape_area_fraction"] > 0
    assert 0 < features["shape_circularity"] < 1.10


def test_translation_preserves_intrinsic_shape_descriptors():
    im, mask = toy_input()
    moved = np.zeros_like(mask)
    moved[2:, 3:] = mask[:-2, :-3]
    before = extract_features(im, mask)
    after = extract_features(im, moved)
    for col in FEATURE_COLUMNS[:6]:
        assert before[col] == pytest.approx(after[col], abs=1e-10), col


def test_outside_pixels_do_not_affect_intensity_or_texture():
    im, mask = toy_input()
    altered = im.copy()
    altered[~mask] = 220
    f1, f2 = extract_features(im, mask), extract_features(altered, mask)
    for col in FEATURE_COLUMNS:
        assert f1[col] == pytest.approx(f2[col], abs=1e-12), col


def test_preprocessing_shapes_and_mask_binary():
    im, mask = toy_input()
    a, b = preprocess_pair(im, mask, max_side=256)
    assert a.shape == b.shape == (256, 256)
    assert b.dtype == np.bool_
    assert b.sum() > 0


def test_reject_mismatched_dimensions():
    im, mask = toy_input()
    with pytest.raises(ValueError, match="matching size"):
        extract_features(im, mask[:20])


def test_reject_empty_mask():
    im, mask = toy_input()
    with pytest.raises(ValueError, match="fewer than 15"):
        extract_features(im, np.zeros_like(mask))
