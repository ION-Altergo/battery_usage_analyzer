from typing import Dict, Union
import numpy as np
import pandas as pd
from altergo_sdk.boiler_plate import Model, register_model


def _validate_series(name: str, s: pd.Series) -> pd.Series:
    if not isinstance(s, pd.Series):
        raise TypeError(f"Input '{name}' must be a pandas Series with a DateTimeIndex.")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(f"Input '{name}' must have a DateTimeIndex.")
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    return s.astype(float)


def _first_order_smoother(x: np.ndarray, dt_h: np.ndarray, tau_h: float) -> np.ndarray:
    """Causal exponential smoother for irregular dt."""
    y = np.empty_like(x)
    if len(x) == 0:
        return y
    y[0] = x[0]
    if tau_h <= 0:
        y[1:] = x[1:]
        return y
    for k in range(1, len(x)):
        dt = max(dt_h[k], 0.0)
        alpha = 1.0 - np.exp(-dt / tau_h)
        y[k] = y[k-1] + alpha * (x[k] - y[k-1])
    return y


def _smoothstep01(t: np.ndarray) -> np.ndarray:
    """Cubic smoothstep on [0,1] with zero slope at ends."""
    tt = np.clip(t, 0.0, 1.0)
    return tt * tt * (3.0 - 2.0 * tt)


@register_model("enhanced_equivalent_cycles", metadata={
    "category": "Health",
    "complexity": "Advanced", 
    "computational_cost": "Medium"
})
class EnhancedEquivalentCycles(Model):
    """
    Counts standard equivalent full cycles and condition-weighted cycles.
    This revision implements a continuous, chemistry-tunable SOC stress model
    using smooth ramps near the upper and (weakly) the very low SOC regions.
    """

    def process(self, data: Dict[str, Union[pd.Series, float]]) -> Dict[str, pd.Series]:
        """
        Args:
            data:
              Series (DateTimeIndex):
                - "current": A (abs used)
                - "soc": p.u. [0..1] (optional; default 0.5)
                - "temperature": °C (optional; default 25°C)
              Params (float):
                - "capacity": Ah (required)
                - "rated_cycle_count": cycles (required)
        Returns:
            - "std_cycle_count"
            - "equivalent_cycle_count"
            - "cycle_life_fraction"
        """
        try:
            # --- Inputs ---
            current = _validate_series("current", data["current"])
            soc = _validate_series("soc", data["soc"]) if "soc" in data else pd.Series(
                0.5, index=current.index, dtype=float
            )
            temperature = _validate_series("temperature", data["temperature"]) if "temperature" in data else pd.Series(
                25.0, index=current.index, dtype=float
            )
            capacity = float(data["capacity"])
            rated_cycle_count = float(data["rated_cycle_count"])
            if capacity <= 0:
                raise ValueError("Parameter 'capacity' must be > 0.")
            if rated_cycle_count <= 0:
                raise ValueError("Parameter 'rated_cycle_count' must be > 0.")

            # --- Config (LFP-focused toned-down defaults) ---
            cfg = {
                # SOC weighting mode & smoothing
                "soc_weight_mode": "smoothstep",   # "smoothstep" | "off"
                "soc_sustain_tau_hours": 1.5,      # EMA of SOC to capture operating window

                # High-SOC ramp (dominant for LFP): onset -> full
                "soc_high_onset": 0.80,            # start ramping penalty at ~80%
                "soc_high_full": 0.96,             # reach full penalty at ~96%
                "soc_high_gain": 0.45,             # adds up to +0.45x at full (w=1.45)
                "soc_high_pow": 1.0,               # shape exponent (>=1)

                # Very-low-SOC ramp (weak for LFP): onset -> full
                "soc_low_onset": 0.08,             # begin small penalty below ~8%
                "soc_low_full": 0.02,              # full by ~2%
                "soc_low_gain": 0.10,              # tiny +0.10x at full (w=1.10)
                "soc_low_pow": 1.0,

                # Apply SOC weight during charge/discharge/both
                "soc_apply": "both",               # "charge" | "discharge" | "both"

                # C-rate weighting (toned-down penalties)
                "c_rate_ref": 0.50,                # reference ~0.5C
                "c_rate_exponent": 1.0,
                "alpha_c": 1.0,                    # reduced penalty: 0.5C -> 1C ≈ +30%
                "beta_c": 0.20,                    # reduced relief below ref
                "sustain_tau_hours": 0.5,          # EMA horizon for C-rate

                # Temperature weighting (cyclic aging; gentler than calendar Q10)
                "q10_cyclic": 1.30,                # ~1.3x per +10°C above tref
                "temp_ref_c": 25.0,
                "lowT_charge_on": True,            # add small charge-only penalty below lowT_ref
                "lowT_ref_c": 15.0,
                "lowT_charge_gain_per_10C": 0.10,  # +10% per -10°C during charge only

                # General
                "eps_current": 1e-3,
                "min_weight": 0.2,
                "max_weight": 3.0                  # clamp combined multiplier
            }
            cfg.update(self.config or {})

            # --- Timebase & throughput ---
            dt_h = current.index.to_series().diff().dt.total_seconds().fillna(0.0).astype(float).values / 3600.0
            thr_Ah = current.abs().values * dt_h

            # --- Sustained C-rate ---
            c_rate_inst = current.abs().values / capacity
            c_rate_sust = _first_order_smoother(c_rate_inst, dt_h, float(cfg["sustain_tau_hours"]))

            # --- Temperature multiplier ---
            tref = float(cfg["temp_ref_c"])
            q10 = float(cfg["q10_cyclic"])
            temp_arr = temperature.reindex(current.index).fillna(method="ffill").fillna(tref).values
            w_t = np.power(q10, (temp_arr - tref) / 10.0)  # >1 above tref; <1 below

            # Low-temperature charge-only surcharge (plating risk)
            if bool(cfg.get("lowT_charge_on", True)):
                lowT_ref = float(cfg["lowT_ref_c"])
                gain_per_10C = float(cfg["lowT_charge_gain_per_10C"])
                below = np.maximum(0.0, (lowT_ref - temp_arr) / 10.0) * gain_per_10C
                # apply only during net charging (current > +eps)
                w_t = np.where(current.values > float(cfg["eps_current"]), w_t * (1.0 + below), w_t)

            # --- C-rate multiplier ---
            c_ref = float(cfg["c_rate_ref"])
            p_c = float(cfg["c_rate_exponent"])
            alpha_c = float(cfg["alpha_c"])
            beta_c = float(cfg["beta_c"])
            above = np.maximum(0.0, c_rate_sust - c_ref)
            below = np.maximum(0.0, c_ref - c_rate_sust)
            w_c = np.where(
                c_rate_sust >= c_ref,
                1.0 + alpha_c * (above / max(c_ref, 1e-9)) ** p_c,
                1.0 - beta_c * (below / max(c_ref, 1e-9))
            )

            # --- Continuous SOC multiplier (new) ---
            soc_aligned = soc.reindex(current.index).fillna(method="ffill").fillna(0.5).values
            soc_sust = _first_order_smoother(soc_aligned, dt_h, float(cfg["soc_sustain_tau_hours"]))

            if cfg.get("soc_weight_mode", "smoothstep") == "off":
                w_soc = np.ones_like(soc_sust)
            else:
                # High-SOC smooth ramp: 0 below onset, 1 at full, cubic smoothstep
                sh0 = float(cfg["soc_high_onset"])
                sh1 = float(cfg["soc_high_full"])
                if not (0.0 <= sh0 < sh1 <= 1.0):
                    raise ValueError("Require 0 <= soc_high_onset < soc_high_full <= 1.")
                t_high = (soc_sust - sh0) / max(sh1 - sh0, 1e-9)
                g_high = _smoothstep01(t_high) ** float(cfg["soc_high_pow"])
                # Low-SOC smooth ramp (reverse): 0 above onset, 1 at full (toward 0%)
                sl0 = float(cfg["soc_low_onset"])
                sl1 = float(cfg["soc_low_full"])
                if not (0.0 <= sl1 < sl0 <= 1.0):
                    raise ValueError("Require 0 <= soc_low_full < soc_low_onset <= 1.")
                t_low = (sl0 - soc_sust) / max(sl0 - sl1, 1e-9)
                g_low = _smoothstep01(t_low) ** float(cfg["soc_low_pow"])

                w_soc = 1.0 + float(cfg["soc_high_gain"]) * g_high + float(cfg["soc_low_gain"]) * g_low

            # Optionally apply SOC penalty only for a current direction
            apply_mode = str(cfg["soc_apply"]).lower()
            i = current.values
            if apply_mode == "charge":
                w_soc = np.where(i > float(cfg["eps_current"]), w_soc, 1.0)
            elif apply_mode == "discharge":
                w_soc = np.where(i < -float(cfg["eps_current"]), w_soc, 1.0)
            # else "both": use as-is

            # --- Combine multipliers only when current flows ---
            active = (current.abs().values >= float(cfg["eps_current"]))
            w_total = np.clip(w_c * w_soc * w_t, float(cfg["min_weight"]), float(cfg["max_weight"]))
            w_total = np.where(active, w_total, 1.0)

            # --- Cycle counts ---
            std_cycles_cum = np.cumsum(thr_Ah) / (2.0 * capacity)
            eq_cycles_cum = np.cumsum(thr_Ah * w_total) / (2.0 * capacity)
            life_frac = eq_cycles_cum / rated_cycle_count

            idx = current.index
            return {
                "std_cycle_count": pd.Series(std_cycles_cum, index=idx, name="std_cycle_count"),
                "equivalent_cycle_count": pd.Series(eq_cycles_cum, index=idx, name="equivalent_cycle_count"),
                "cycle_life_fraction": pd.Series(life_frac, index=idx, name="cycle_life_fraction")
            }

        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(
                f"Missing required input or parameter: '{missing}'. "
                f"Provide 'current' series and parameters 'capacity' & 'rated_cycle_count'."
            ) from e
        except Exception as e:
            raise RuntimeError(f"[enhanced_equivalent_cycles] Failed during processing: {e}") from e
