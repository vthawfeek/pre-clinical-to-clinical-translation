"""Day 15 tests: Wilson + bootstrap confidence intervals and seed aggregation."""

import numpy as np
import pytest

from pctrans.evaluation.stats import (
    aggregate_seeds,
    bootstrap_ci,
    bootstrap_metric_ci,
    permutation_test,
    wilson_ci,
)


def test_wilson_ci_known_value():
    # 38/38 successes -> Wilson lower bound ~0.908, upper bound capped at 1.0.
    low, high = wilson_ci(38, 38)
    assert abs(low - 0.908) < 0.005
    assert high == pytest.approx(1.0)


def test_wilson_ci_bounds_within_unit():
    # A messy proportion still yields bounds strictly inside [0, 1], low < p < high.
    low, high = wilson_ci(19, 38)
    assert 0.0 < low < 0.5 < high < 1.0


def test_wilson_ci_zero_n():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.7, scale=0.1, size=200)
    res = bootstrap_ci(values, np.mean, n_boot=2000, seed=0)
    assert res["ci_low"] <= res["point"] <= res["ci_high"]


def test_bootstrap_ci_shrinks_with_n():
    # Same underlying signal, more samples -> narrower interval.
    rng = np.random.default_rng(1)
    small = rng.binomial(1, 0.8, size=38).astype(float)
    large = rng.binomial(1, 0.8, size=380).astype(float)
    ci_small = bootstrap_ci(small, np.mean, n_boot=2000, seed=0)
    ci_large = bootstrap_ci(large, np.mean, n_boot=2000, seed=0)
    width_small = ci_small["ci_high"] - ci_small["ci_low"]
    width_large = ci_large["ci_high"] - ci_large["ci_low"]
    assert width_small > width_large


def test_bootstrap_metric_ci_perfect_retrieval():
    # Every anchor matches all its neighbours -> point 1.0, degenerate CI at 1.0.
    match_fraction = np.ones(38)
    res = bootstrap_metric_ci(match_fraction)
    assert res["point"] == 1.0
    assert res["ci_low"] == 1.0 and res["ci_high"] == 1.0
    assert res["n"] == 38


def test_bootstrap_metric_ci_thresholds_majority():
    # 0.6 (3/5) counts as a hit; 0.4 (2/5) does not -> point estimate 0.5.
    match_fraction = np.array([0.6, 0.6, 0.4, 0.4])
    res = bootstrap_metric_ci(match_fraction, n_boot=500, seed=0)
    assert res["point"] == 0.5


def test_aggregate_seeds_summary():
    values = [0.95, 1.0, 0.97, 0.92, 1.0]
    agg = aggregate_seeds(values, n_boot=1000, seed=0)
    assert agg["n"] == 5
    assert abs(agg["mean"] - np.mean(values)) < 1e-9
    assert agg["min"] == 0.92 and agg["max"] == 1.0
    assert agg["ci_low"] <= agg["mean"] <= agg["ci_high"]


def test_permutation_null_near_chance():
    # Day 21: a "shuffled-label model" cannot do better than randomly matching
    # a permuted label vector against the true one -- for a balanced n_lineages
    # class label, that expected match rate is ~1/n_lineages.
    n_lineages = 15
    rng = np.random.default_rng(0)
    true_labels = rng.integers(0, n_lineages, size=2000)

    def null_generator(rng):
        shuffled = rng.permutation(true_labels)
        return float((shuffled == true_labels).mean())

    result = permutation_test(1.0, null_generator, n_perm=20, seed=1)
    assert abs(result["null_mean"] - 1.0 / n_lineages) < 0.03
    assert result["null_max"] < 0.5  # nowhere near the real (correspondence-intact) value


def test_permutation_pvalue_range():
    # p must always land in [0, 1]; a real value that beats every permutation
    # draw should get a small p-value (real-signal case), not exactly 0. With
    # the add-one correction the smallest reachable p at n_perm draws is
    # 1/(n_perm + 1), so use enough draws to get comfortably under 0.01.
    def chance_generator(rng):
        return float(rng.uniform(0.0, 0.15))

    result = permutation_test(0.95, chance_generator, n_perm=200, seed=2)
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["p_value"] < 0.01
    assert len(result["null_values"]) == 200

    # A value squarely inside the null distribution should NOT get a small p.
    def wide_generator(rng):
        return float(rng.uniform(0.0, 1.0))

    mid_result = permutation_test(0.5, wide_generator, n_perm=200, seed=3)
    assert 0.0 <= mid_result["p_value"] <= 1.0
    assert mid_result["p_value"] > 0.1
