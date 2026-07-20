"""Scoring for the injected-corruption sweep."""
from __future__ import annotations
import numpy as np


def offset_recovery_error(injected: list[float], recovered: list[float], bias: float = 0.0):
    """|recovered - bias - injected| in frames. Returns dict of med/max/within1."""
    e = np.abs(np.array(recovered) - bias - np.array(injected))
    return dict(median=float(np.median(e)), max=float(e.max()),
                within1=float(np.mean(e <= 1.0)), n=len(e))


def prf(truth: list[bool], pred: list[bool]) -> dict:
    """Precision / recall / F1 for episode-level flagging."""
    t, p = np.array(truth), np.array(pred)
    tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn)
