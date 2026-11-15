"""Unit/consistency tests for post-hoc coherent-boundary feature extraction."""
import numpy as np
import pandas as pd
import pytest

from src.coherent_features import (
    COHERENT_CONDITIONS, coherent_image_rows, verify_coherent_geometry
)
from src.features import FEATURE_COLUMNS, extract_features
from src.perturbations import TARGET_IOUS


def specimen():
    yy, xx = np.ogrid[:96, :104]
    mask = ((yy - 48) / 21) ** 2 + ((xx - 50) / 28) ** 2 <= 1
    image = np.tile(np.arange(104, dtype=np.uint8), (96, 1))
    return image, mask


def sample_rows():
    image, mask = specimen()
    rows = coherent_image_rows("bus_example", 23, "benign", 2, image, mask)
    return image, mask, pd.DataFrame(rows)


def test_four_conditions_and_manifest_metadata():
    _, _, frame = sample_rows()
    assert tuple(frame.condition) == COHERENT_CONDITIONS
    assert len(frame) == 4
    assert frame.Case.eq(23).all()
    assert frame.dev_fold.eq(2).all()
    assert frame.split.eq("development").all()


def test_valid_finite_23_features_and_iou():
    _, _, frame = sample_rows()
    assert frame.valid.all()
    assert np.isfinite(frame[list(FEATURE_COLUMNS)].to_numpy(float)).all()
    for condition in COHERENT_CONDITIONS:
        row = frame.loc[frame.condition.eq(condition)].iloc[0]
        severity = condition.split("_")[1]
        assert abs(row.actual_iou - TARGET_IOUS[severity]) < 0.01
        assert row.changed_pixels > 0


def test_does_not_mutate_original_grayscale_or_mask():
    image, mask = specimen()
    i, m = image.copy(), mask.copy()
    coherent_image_rows("bus_example", 23, "benign", 2, image, mask)
    assert np.array_equal(image, i) and np.array_equal(mask, m)


def test_geometry_consistency_and_clean_circularity():
    image, mask, frame = sample_rows()
    geometry = frame[["ID", "condition", "valid", "actual_iou", "changed_pixels"]].copy()
    geometry["circularity_coherent_protocol"] = frame.shape_circularity
    geometry["circularity_clean"] = extract_features(image, mask)["shape_circularity"]
    clean = pd.DataFrame({"ID": ["bus_example"],
                          "shape_circularity": [extract_features(image, mask)["shape_circularity"]]})
    verify_coherent_geometry(frame, geometry, clean)


def test_reject_corrupt_coherent_geometry():
    image, mask, frame = sample_rows()
    geometry = frame[["ID", "condition", "valid", "actual_iou", "changed_pixels"]].copy()
    geometry["circularity_coherent_protocol"] = frame.shape_circularity
    geometry["circularity_clean"] = extract_features(image, mask)["shape_circularity"]
    clean = pd.DataFrame({"ID": ["bus_example"],
                          "shape_circularity": [extract_features(image, mask)["shape_circularity"]]})
    geometry.loc[0, "actual_iou"] -= 0.1
    with pytest.raises(ValueError, match="mismatch"):
        verify_coherent_geometry(frame, geometry, clean)


def test_mismatched_image_shape_rejected():
    image, mask = specimen()
    with pytest.raises(ValueError, match="matching"):
        coherent_image_rows("bus_example", 23, "benign", 2, image[:, 1:], mask)
