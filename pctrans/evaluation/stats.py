"""Confidence intervals & resampling — the Day 15 statistical-hardening toolkit.

Phase 1 reported bare point metrics ("kNN@5 = 100%") on a 38-cell-line test set.
That is easy to attack: a proportion of 38/38 has a wide sampling interval, and a
single split could be lucky. This module attaches an honest interval to every
headline number.

Three entry points:

- `wilson_ci` — the analytic score interval for a binomial proportion. Correct for
  small n (the 38-anchor case) where the normal approximation over-narrows and can
  spill outside [0, 1]; the Wilson interval stays inside and never collapses to a
  zero-width band at p = 1.
- `bootstrap_ci` — a generic percentile bootstrap. Resample the per-unit values
  (CCLE anchors for kNN, pooled samples for silhouette) with replacement, recompute
  the statistic on each resample, and read the interval off the empirical
  distribution. Makes no distributional assumption.
- `bootstrap_metric_ci` — a convenience wrapper that turns the per-CCLE
  `match_fraction` already produced by `knn.py` into a kNN point estimate + 95% CI,
  so the evaluate CLI can print an interval without re-embedding anything.

Day 21 adds one more entry point, `permutation_test`, for the label-shuffle
negative control: it turns a real metric value plus a null-generating callback
into an empirical p-value. The callback is intentionally generic (it just has to
return a float) so the same function scores both Day 21 variants -- the cheap
eval-only label shuffle (recompute kNN@5 on already-embedded test points with
permuted labels) and the expensive retrain-based shuffle (break the CCLE<->TCGA
lineage correspondence, retrain a short schedule, evaluate) -- without stats.py
importing anything about models or datasets.
"""

import numpy as np
from scipy.stats import norm

# A CCLE anchor counts as a retrieval "hit" when at least half of its k nearest
# TCGA neighbours share its lineage. For k=5 the admissible match fractions are
# {0, .2, .4, .6, .8, 1}, so ">= 0.5" is exactly ">= 3/5" — a strict majority,
# matching the majority-vote accuracy computed in knn.py.
HIT_THRESHOLD = 0.5


def wilson_ci(successes, n, alpha=0.05):
    """Wilson score confidence interval for a binomial proportion.

    Returns ``(low, high)`` for ``successes`` correct out of ``n`` at confidence
    ``1 - alpha``. Unlike the Wald interval, the bounds stay within [0, 1] and the
    interval keeps a finite width even at ``p = 1`` (e.g. 38/38 -> ~0.908–1.0).
    ``n == 0`` returns the vacuous ``(0.0, 1.0)``.
    """
    if n <= 0:
        return (0.0, 1.0)
    z = norm.ppf(1.0 - alpha / 2.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values, statistic, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for ``statistic`` applied to ``values``.

    Resamples ``values`` (a 1-D array of per-unit measurements) with replacement
    ``n_boot`` times, recomputes ``statistic`` on each resample, and returns the
    ``alpha/2`` and ``1 - alpha/2`` percentiles as the interval. The returned dict
    carries the point estimate (statistic on the original sample), the interval,
    and the resampling parameters.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    point = float(statistic(values)) if n else 0.0
    if n == 0:
        return {"point": point, "ci_low": 0.0, "ci_high": 0.0, "n_boot": n_boot, "n": 0}

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        resample = values[rng.integers(0, n, size=n)]
        boots[b] = statistic(resample)
    lo = float(np.percentile(boots, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_boot": n_boot, "n": n}


def bootstrap_metric_ci(match_fraction, n_boot=2000, seed=0, alpha=0.05):
    """kNN point estimate + percentile CI from the per-CCLE ``match_fraction``.

    Converts each anchor's neighbour ``match_fraction`` to a binary hit
    (``>= HIT_THRESHOLD``) and bootstraps the mean over anchors. The point estimate
    is the majority-vote retrieval accuracy; the interval reflects the small anchor
    count. Returns the `bootstrap_ci` dict.
    """
    match_fraction = np.asarray(match_fraction, dtype=np.float64)
    hits = (match_fraction >= HIT_THRESHOLD).astype(np.float64)
    return bootstrap_ci(hits, np.mean, n_boot=n_boot, seed=seed, alpha=alpha)


def aggregate_seeds(values, alpha=0.05, n_boot=2000, seed=0):
    """Summarise a per-seed metric list: mean, sd, and a bootstrap CI of the mean.

    Used by the multi-seed runner to turn 10 per-seed kNN@5 / silhouette / TFS
    values into ``{mean, sd, min, max, ci_low, ci_high, n}``. The CI here is the
    bootstrap interval of the *mean across seeds* (reproducibility spread), not the
    within-split anchor interval.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0,
                "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    ci = bootstrap_ci(values, np.mean, n_boot=n_boot, seed=seed, alpha=alpha)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if n > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "n": n,
    }


def permutation_test(real_value, null_generator, n_perm=20, seed=0):
    """Empirical one-sided p-value: is ``real_value`` reachable by chance?

    Calls ``null_generator(rng)`` ``n_perm`` times to build the null
    distribution -- each call shuffles the label correspondence some way and
    returns the resulting metric as a float -- then reports the fraction of
    null draws at least as large as ``real_value``. Uses the standard
    add-one (Laplace) correction ``(count + 1) / (n_perm + 1)`` so a real
    value that beats every permutation still gets a finite, non-zero p-value
    rather than the unjustified ``0.0`` a naive ratio would report.
    """
    rng = np.random.default_rng(seed)
    null_values = np.array(
        [float(null_generator(rng)) for _ in range(n_perm)], dtype=np.float64
    )
    count = int(np.sum(null_values >= real_value))
    p_value = (count + 1) / (n_perm + 1)
    return {
        "real_value": float(real_value),
        "null_values": null_values.tolist(),
        "null_mean": float(null_values.mean()) if n_perm else 0.0,
        "null_max": float(null_values.max()) if n_perm else 0.0,
        "n_perm": n_perm,
        "p_value": float(p_value),
    }
