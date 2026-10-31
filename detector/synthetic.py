"""Synthetic kinematic sequences (numpy only) used to train the classifier.

Each sequence is SEQ_LEN x NUM_FEATURES in the same units as
tracker.frame_features().  Normal traffic includes some HARD cases (gentle
braking, stopped queues, vehicles overlapping with no sudden stop) and the
accident class includes some HARD cases (merged boxes, single-vehicle crash),
so the task is not trivially separable.
"""
import numpy as np
from .tracker import SEQ_LEN, NUM_FEATURES


def _normal(rng):
    n = int(rng.integers(1, 5))
    s0 = rng.uniform(0.2, 1.6)
    if rng.random() < 0.15:                      # stopped / slow queue
        s0 = rng.uniform(0.0, 0.12)
    mean = np.clip(s0 + np.cumsum(rng.normal(0, 0.03, SEQ_LEN)), 0, None)
    mx = mean * (rng.uniform(1.0, 1.6) if n > 1 else 1.0)
    decel = np.abs(rng.normal(0, 0.03, SEQ_LEN))
    if rng.random() < 0.25:                      # gentle braking
        a = int(rng.integers(0, SEQ_LEN - 6)); b = a + int(rng.integers(4, 8))
        rate = rng.uniform(0.04, 0.18)
        decel[a:b] += rate
        mean[a:] = np.clip(mean[a:] - np.cumsum(np.r_[np.zeros(a), np.full(SEQ_LEN - a, rate)])[a:] * 0.6, 0, None)
        mx = np.maximum(mx, mean)
    iou = np.zeros(SEQ_LEN)
    min_dist = np.full(SEQ_LEN, 1.0)
    if n > 1:
        min_dist = np.clip(rng.uniform(0.08, 0.6) + np.cumsum(rng.normal(0, 0.01, SEQ_LEN)), 0.04, 1)
        if rng.random() < 0.2:                   # close / overlapping but no collision
            iou = np.clip(rng.uniform(0.03, 0.35) + rng.normal(0, 0.02, SEQ_LEN), 0, 1)
            min_dist = np.clip(min_dist * 0.4, 0.03, 1)
    return np.stack([mx, mean, decel, iou, min_dist, np.full(SEQ_LEN, n / 10.0)], 1)


def _accident(rng):
    n = 1 if rng.random() < 0.1 else int(rng.integers(2, 5))
    s0 = rng.uniform(0.4, 1.6)
    t0 = int(rng.integers(3, 13))
    mean = np.full(SEQ_LEN, s0) + rng.normal(0, 0.03, SEQ_LEN)
    after = s0 * rng.uniform(0.0, 0.3)
    mean[t0:] = after * np.exp(-0.3 * np.arange(SEQ_LEN - t0))
    mx = mean * (rng.uniform(1.0, 1.4) if n > 1 else 1.0)
    decel = np.abs(rng.normal(0, 0.03, SEQ_LEN))
    d0 = s0 * rng.uniform(0.35, 0.85)
    decel[t0] += d0
    if t0 + 1 < SEQ_LEN:
        decel[t0 + 1] += d0 * rng.uniform(0.2, 0.6)
    if n >= 3 and rng.random() < 0.6:            # a bystander vehicle keeps moving
        other = rng.uniform(0.2, 1.0)
        mx[t0:] = np.maximum(mx[t0:], other)
        mean[t0:] += other / n
    iou = np.zeros(SEQ_LEN)
    min_dist = np.full(SEQ_LEN, 1.0)
    if n > 1:
        min_dist = np.clip(np.linspace(rng.uniform(0.3, 0.6), 0.1, t0 + 1).tolist() +
                           [rng.uniform(0.0, 0.1)] * (SEQ_LEN - t0 - 1), 0.0, 1)
        if rng.random() < 0.8:                   # boxes overlap after impact
            iou[t0:] = np.clip(rng.uniform(0.08, 0.7) + rng.normal(0, 0.02, SEQ_LEN - t0), 0, 1)
    return np.stack([mx, mean, decel, iou, min_dist, np.full(SEQ_LEN, n / 10.0)], 1)


def make_dataset(n_samples=1600, seed=42):
    """Balanced dataset -> X (N, SEQ_LEN, NUM_FEATURES) float32, y (N,) float32."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n_samples):
        label = i % 2
        seq = _accident(rng) if label else _normal(rng)
        seq = np.clip(seq + rng.normal(0, 0.015, seq.shape), 0, None)
        X.append(seq); y.append(label)
    X = np.array(X, np.float32); y = np.array(y, np.float32)
    assert X.shape[1:] == (SEQ_LEN, NUM_FEATURES)
    return X, y
