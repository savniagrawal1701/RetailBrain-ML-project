"""Hybrid recommender scoring (module 8), shared by training and the web app."""
import numpy as np

# blend weights (collaborative, content, popularity); train.py overwrites these after tuning
HYBRID_W = {"warm": (0.6, 0.1, 0.3), "cold": (0.3, 0.2, 0.5)}


def minmax(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def hybrid_scores(hist, collab, content, pop, weights=None):
    """Blend three signals; lean on popularity + content when we know little about the user."""
    hist = np.asarray(hist)
    w = weights or HYBRID_W["warm" if len(hist) >= 3 else "cold"]
    s = w[0] * minmax(collab[:, hist].sum(1)) + w[1] * minmax(content[:, hist].mean(1)) + w[2] * pop
    s = s.astype(float)
    s[hist] = -np.inf
    return s
