"""Tests for sample_weight behaviour in evaluation metrics."""

import math

import numpy as np
import pytest

from rating_systems.evaluation.metrics import (
    accuracy,
    brier_score,
    calibration_error,
    log_loss,
)


@pytest.fixture
def small_case():
    p = np.array([0.8, 0.6, 0.3, 0.9, 0.5])
    y = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    return p, y


def test_all_ones_equals_unweighted(small_case):
    p, y = small_case
    ones = np.ones_like(p)
    assert brier_score(p, y, sample_weight=ones) == pytest.approx(brier_score(p, y))
    assert log_loss(p, y, sample_weight=ones) == pytest.approx(log_loss(p, y))
    assert accuracy(p, y, sample_weight=ones) == pytest.approx(accuracy(p, y))


def test_masking_drops_observation(small_case):
    p, y = small_case
    # Drop the most costly log-loss observation (index 1 is a mispredict at 0.6 → 1.0)
    w = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    assert log_loss(p, y, sample_weight=w) < log_loss(p, y)
    assert brier_score(p, y, sample_weight=w) < brier_score(p, y)


def test_scale_invariance(small_case):
    p, y = small_case
    w = np.array([1.0, 2.0, 0.5, 1.5, 1.0])
    # Weighted mean is scale-invariant in the weights
    assert brier_score(p, y, sample_weight=w) == pytest.approx(
        brier_score(p, y, sample_weight=w * 10.0)
    )
    assert log_loss(p, y, sample_weight=w) == pytest.approx(
        log_loss(p, y, sample_weight=w * 10.0)
    )


def test_zero_weights_return_nan(small_case):
    p, y = small_case
    zeros = np.zeros_like(p)
    assert math.isnan(brier_score(p, y, sample_weight=zeros))
    assert math.isnan(log_loss(p, y, sample_weight=zeros))
    assert math.isnan(accuracy(p, y, sample_weight=zeros))
    assert math.isnan(calibration_error(p, y, sample_weight=zeros))


def test_calibration_error_weighted(small_case):
    p, y = small_case
    # Drop observation 3 (p=0.9, correct) — reduces the hit rate in the top bin
    w = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
    ce_unweighted = calibration_error(p, y)
    ce_weighted = calibration_error(p, y, sample_weight=w)
    assert ce_weighted != pytest.approx(ce_unweighted)
    assert 0.0 <= ce_weighted <= 1.0
