"""
Utility functions for the Battery Usage Analyzer model.

This module contains helper functions for data processing, signal analysis,
and segmentation operations used by the BatteryUsageAnalyzer.
"""

from typing import Dict
import pandas as pd
import numpy as np


def infer_dt_s(index: pd.DatetimeIndex) -> float:
    """Median sampling interval in seconds (fallback to 1s)."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 1.0
    diffs = index.to_series().diff().dt.total_seconds().dropna()
    if diffs.empty:
        return 1.0
    return float(np.median(diffs.values))


def robust_z(series: pd.Series) -> pd.Series:
    """Median/MAD z-score with fallback to std; caps extreme values."""
    x = series.astype(float).values
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad < 1e-12:
        std = np.nanstd(x)
        scale = std if std > 1e-12 else 1.0
    else:
        scale = 1.4826 * mad
    z = (x - med) / (scale + 1e-12)
    z = np.clip(np.where(np.isfinite(z), z, 0.0), -10.0, 10.0)
    return pd.Series(z, index=series.index)


def enforce_min_gap(boundaries: pd.Series, min_gap_samples: int) -> pd.Series:
    """
    Keep boundary candidates but enforce minimal spacing (samples) between them.
    Always keeps the first sample as a boundary.
    """
    n = len(boundaries)
    if n == 0:
        return boundaries
    idx_true = np.flatnonzero(boundaries.values)
    keep = []
    last = -10**12
    for i in idx_true:
        if i - last >= max(1, min_gap_samples):
            keep.append(i)
            last = i
    if len(keep) == 0 or keep[0] != 0:
        keep = [0] + keep  # ensure first point is a boundary
    out = np.zeros(n, dtype=bool)
    out[np.unique(np.clip(keep, 0, n - 1))] = True
    return pd.Series(out, index=boundaries.index)


def ids_from_boundaries(boundaries: pd.Series) -> pd.Series:
    """Cumulative sum of boolean boundaries -> 1-based integer segment IDs."""
    ids = np.cumsum(boundaries.astype(int).values)
    return pd.Series(ids, index=boundaries.index, dtype="int64")


def majority_label_per_segment(raw_labels: pd.Series, seg_ids: pd.Series) -> pd.Series:
    """
    For each segment ID, assign the majority label (mode) within the slice,
    then broadcast back as a series aligned to the index.
    """
    labels = raw_labels.astype("object")
    ids = seg_ids.values
    out = labels.copy()
    # compute run-length segments using seg_ids
    v = ids
    starts = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1]
    ends = np.r_[starts[1:], [len(v)]]
    for s, e in zip(starts, ends):
        segment = labels.iloc[s:e]
        if segment.empty:
            continue
        # majority vote
        vals, counts = np.unique(segment.values, return_counts=True)
        maj = vals[np.argmax(counts)]
        out.iloc[s:e] = maj
    return out


def rolling_ewm(series: pd.Series, win: int) -> pd.Series:
    """Simple EWM smoothing with span ~ win."""
    win = max(1, int(win))
    return series.ewm(span=win, adjust=False).mean()


def detect_peaks_over_threshold(score: pd.Series, thr: float) -> pd.Series:
    """Local maxima above threshold -> boundary candidates."""
    prev = score.shift(1).fillna(score.iloc[0])
    nxt = score.shift(-1).fillna(score.iloc[-1])
    is_peak = (score >= thr) & (score >= prev) & (score >= nxt)
    return is_peak


def safe_quantile(values: np.ndarray, q: float, default: float) -> float:
    """Calculate quantile safely, returning default if no valid values."""
    arr = values[np.isfinite(values)]
    if arr.size == 0:
        return default
    return float(np.quantile(arr, q))


def soc_to_unit(soc: pd.Series) -> pd.Series:
    """Normalize SoC to [0,1] if given in %."""
    s = soc.astype(float)
    if s.max(skipna=True) > 1.5:
        return (s / 100.0).clip(0.0, 1.0)
    return s.clip(0.0, 1.0)


def derivative(x: pd.Series, dt_s: float) -> pd.Series:
    """Calculate time derivative of a series."""
    return x.diff().divide(dt_s).fillna(0.0)


def compose_change_score(df_deriv: pd.DataFrame, weights: Dict[str, float], smooth_n: int) -> pd.Series:
    """
    Compose weighted change score from derivative signals.
    
    Args:
        df_deriv: DataFrame with derivative columns
        weights: Dictionary mapping column names to weights
        smooth_n: Smoothing window size
        
    Returns:
        Smoothed composite change score
    """
    # Robust z on each derivative column, weight, and take L2 norm
    comp = []
    for col, w in weights.items():
        if col not in df_deriv.columns:
            continue
        comp.append((w * robust_z(df_deriv[col])) ** 2)
    if not comp:
        return pd.Series(0.0, index=df_deriv.index)
    score = np.sqrt(np.sum(comp, axis=0))
    score_series = pd.Series(score, index=df_deriv.index)
    return rolling_ewm(score_series, smooth_n)


def label_operating_mode(current: pd.Series,
                        i_charge_on: float,
                        i_discharge_on: float) -> pd.Series:
    """Basic mode: charge / discharge / idle (sign & magnitude)."""
    I = current.astype(float)
    mode = pd.Series(np.where(I >= i_charge_on, "charge",
                              np.where(I <= -i_discharge_on, "discharge", "idle")),
                     index=I.index, dtype="object")
    return mode


def phase_labels(current: pd.Series,
                vmax: pd.Series,
                dI_dt: pd.Series,
                dVmax_dt: pd.Series,
                i_rest_th: float,
                i_charge_on: float,
                i_discharge_on: float,
                cv_voltage_window_V: float,
                dv_dt_small: float) -> pd.Series:
    """
    Domain-specific phase labels:
      rest, cv_charge, cc_charge, cc_discharge, discharge, idle
    """
    I = current.astype(float)
    V = vmax.astype(float)
    # Estimate "near-max-voltage" window from 99.5th percentile
    vmax_q = safe_quantile(V.values, 0.995, default=float(V.max()))
    near_cv = V >= (vmax_q - cv_voltage_window_V)

    rest = I.abs() <= i_rest_th
    charging = I >= i_charge_on
    discharging = I <= -i_discharge_on

    cv_charge = charging & near_cv & (dVmax_dt.abs() <= dv_dt_small) & (dI_dt < 0.0)
    cc_charge = charging & ~cv_charge
    cc_discharge = discharging  # we keep a single CC discharge bucket
    base_idle = (~charging) & (~discharging) & (~rest)

    # Initialize with 'idle' then override per priority
    phase = pd.Series(np.where(base_idle, "idle", "idle"), index=I.index, dtype="object")
    phase[cc_discharge] = "cc_discharge"
    phase[cc_charge] = "cc_charge"
    phase[cv_charge] = "cv_charge"
    phase[rest] = "rest"
    # For remaining discharging but not captured (e.g., weak discharge)
    phase[(~rest) & (~charging) & (I < 0.0)] = "discharge"
    return phase
