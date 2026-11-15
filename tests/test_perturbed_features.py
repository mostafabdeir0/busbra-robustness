"""Unit tests for Step 13 — perturbation feature data integrity."""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS
from src.perturbed_features import VARIANT_NAMES, check_geometry_match, image_rows


def _specimen():
    yy, xx = np.ogrid[:128, :128]
    lesion = (yy-64)**2 / 22**2 + (xx-64)**2 / 28**2 <= 1
    gray = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    return gray, lesion


def test_exactly_eight_prespecified_rows():
    gray, mask = _specimen()
    rows = image_rows("bus_dummy", 12, "benign", 2, gray, mask)
    assert [r["condition"] for r in rows] == list(VARIANT_NAMES)
    assert all(r["split"] == "development" and r["dev_fold"] == 2 for r in rows)
    assert all(r["valid"] for r in rows)


def test_all_valid_measurements_have_23_finite_features():
    gray, mask = _specimen()
    rows = image_rows("bus_dummy", 12, "benign", 2, gray, mask)
    for r in rows:
        assert np.isfinite([r[f] for f in FEATURE_COLUMNS]).all()
        assert 0 < r["actual_iou"] < 1


def test_no_mutation_of_original_pair():
    gray, mask = _specimen()
    before_img, before_mask = gray.copy(), mask.copy()
    image_rows("bus_dummy", 12, "benign", 2, gray, mask)
    assert np.array_equal(gray, before_img)
    assert np.array_equal(mask, before_mask)


def test_border_variant_remains_explicitly_missing():
    gray, mask = _specimen()
    mask = np.roll(mask, -np.where(mask)[1].min(), axis=1)
    rows = image_rows("bus_dummy", 12, "benign", 2, gray, mask)
    left = next(row for row in rows if row["condition"] == "shift_left_moderate")
    assert left["valid"] is False
    assert left["failure_reason"] == "translation_would_clip_lesion"
    assert np.isnan([left[f] for f in FEATURE_COLUMNS]).all()
    assert sum(r["valid"] for r in rows) == 7


def test_geometry_report_matches_recomputed_values():
    gray, mask = _specimen()
    df = pd.DataFrame(image_rows("bus_dummy", 12, "benign", 2, gray, mask))
    audit = df[["ID", "condition", "valid", "actual_iou", "changed_pixels", "area_ratio", "amount_pixels"]].copy()
    check_geometry_match(df, audit)


def test_reject_audit_mismatch():
    gray, mask = _specimen()
    df = pd.DataFrame(image_rows("bus_dummy", 12, "benign", 2, gray, mask))
    audit = df[["ID", "condition", "valid", "actual_iou", "changed_pixels", "area_ratio", "amount_pixels"]].copy()
    audit.loc[0, "actual_iou"] -= 0.1
    with pytest.raises(ValueError, match="mismatch"):
        check_geometry_match(df, audit)
