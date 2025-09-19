from typing import Dict, Union
import numpy as np
import pandas as pd
from altergo_interface import Model, register_model


def _validate_series(name: str, s: pd.Series) -> pd.Series:
    if not isinstance(s, pd.Series):
        raise TypeError(f"Input '{name}' must be a pandas Series with a DateTimeIndex.")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(f"Input '{name}' must have a DateTimeIndex.")
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    return s.astype(float)


def _ema_irregular(x: np.ndarray, dt_h: np.ndarray, tau_h: float) -> np.ndarray:
    y = np.empty_like(x)
    if len(x) == 0:
        return y
    y[0] = x[0]
    if tau_h <= 0:
        y[1:] = x[1:]
        return y
    for k in range(1, len(x)):
        alpha = 1.0 - np.exp(-max(dt_h[k], 0.0) / tau_h)
        y[k] = y[k-1] + alpha * (x[k] - y[k-1])
    return y


def _smoothstep01(t: np.ndarray) -> np.ndarray:
    tt = np.clip(t, 0.0, 1.0)
    return tt * tt * (3.0 - 2.0 * tt)


@register_model("enhanced_equivalent_cycles")
class EnhancedEquivalentCycles(Model):
    """
    Standard EFCs and condition-weighted equivalent cycles.
    LFP-tuned defaults: continuous SOC stress, gentler cyclic temperature slope (~1.3x/10°C),
    modest 0.5C->1C C-rate penalty, and a low-temperature charge-only penalty.
    """

    def process(self, data: Dict[str, Union[pd.Series, float]]) -> Dict[str, pd.Series]:
        """
        Inputs (pd.Series, DateTimeIndex):
          - current [A]     (sign ignored for throughput; sign used for charge/discharge gating)
          - soc [p.u.]      optional; default 0.5
          - temperature [°C] optional; default 25°C
        Parameters (float):
          - capacity [Ah]
          - rated_cycle_count [cycles]
        Outputs:
          - std_cycle_count
          - equivalent_cycle_count
          - cycle_life_fraction
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
            C_Ah = float(data["capacity"])
            rated_cycles = float(data["rated_cycle_count"])
            if C_Ah <= 0 or rated_cycles <= 0:
                raise ValueError("Parameters 'capacity' (>0) and 'rated_cycle_count' (>0) are required.")

            # --- Config (LFP-focused toned-down defaults) ---
            cfg = {
                # SOC weighting (smooth ramps; see discussion)
                "soc_weight_mode": "smoothstep",
                "soc_sustain_tau_hours": 1.5,
                "soc_high_onset": 0.80,         # start ramp ~80%
                "soc_high_full": 0.96,          # saturate ~96%
                "soc_high_gain": 0.45,          # +45% at top (=> ~1.45x at 96% SOC)
                "soc_high_pow": 1.0,
                "soc_low_onset": 0.08,
                "soc_low_full": 0.02,
                "soc_low_gain": 0.10,           # small penalty only near 0–2%
                "soc_low_pow": 1.0,
                "soc_apply": "both",            # or "charge" | "discharge"

                # C-rate weighting (sustained)
                "c_rate_ref": 0.50,             # reference ~0.5C
                "c_rate_exponent": 1.0,
                "alpha_c": 1.0,                 # 0.5C -> 1C ≈ +30%
                "beta_c": 0.20,                 # relief below ref
                "sustain_tau_hours": 0.5,       # EMA horizon for C-rate

                # Temperature weighting (cyclic aging; gentler than calendar Q10)
                "q10_cyclic": 1.30,             # ~1.3x per +10°C above tref
                "temp_ref_c": 25.0,
                "lowT_charge_on": True,         # add small charge-only penalty below lowT_ref
                "lowT_ref_c": 15.0,
                "lowT_charge_gain_per_10C": 0.10, # +10% per -10°C during charge only

                # General
                "eps_current": 1e-3,
                "min_weight": 0.2,
                "max_weight": 3.0               # clamp combined multiplier
            }
            cfg.update(self.config or {})

            # --- Timebase & throughput ---
            dt_h = current.index.to_series().diff().dt.total_seconds().fillna(0.0).astype(float).values / 3600.0
            abs_I = current.abs().values
            thr_Ah = abs_I * dt_h

            # --- Sustained C-rate ---
            c_rate_inst = abs_I / C_Ah
            c_rate_sust = _ema_irregular(c_rate_inst, dt_h, float(cfg["sustain_tau_hours"]))

            # --- Temperature multiplier ---
            T = temperature.reindex(current.index).fillna(method="ffill").fillna(cfg["temp_ref_c"]).values
            Tref = float(cfg["temp_ref_c"])
            q10 = float(cfg["q10_cyclic"])
            w_t = np.power(q10, (T - Tref) / 10.0)  # >1 above Tref; <1 below (we'll add low-T charge penalty separately)

            # Low-temperature charge-only surcharge (plating risk)
            if bool(cfg.get("lowT_charge_on", True)):
                lowT_ref = float(cfg["lowT_ref_c"])
                gain_per_10C = float(cfg["lowT_charge_gain_per_10C"])
                below = np.maximum(0.0, (lowT_ref - T) / 10.0) * gain_per_10C
                # apply only during net charging (current > +eps)
                w_t = np.where(current.values > float(cfg["eps_current"]), w_t * (1.0 + below), w_t)

            # --- C-rate multiplier (toned-down) ---
            c_ref = float(cfg["c_rate_ref"])
            p = float(cfg["c_rate_exponent"])
            alpha_c = float(cfg["alpha_c"])
            beta_c = float(cfg["beta_c"])
            above = np.maximum(0.0, c_rate_sust - c_ref)
            below = np.maximum(0.0, c_ref - c_rate_sust)
            w_c = np.where(
                c_rate_sust >= c_ref,
                1.0 + alpha_c * (above / max(c_ref, 1e-9)) ** p,
                1.0 - beta_c * (below / max(c_ref, 1e-9))
            )

            # --- SOC multiplier (smoothstep continuous) ---
            soc_vals = soc.reindex(current.index).fillna(method="ffill").fillna(0.5).values
            soc_sust = _ema_irregular(soc_vals, dt_h, float(cfg["soc_sustain_tau_hours"]))
            mode = str(cfg.get("soc_weight_mode", "smoothstep")).lower()
            if mode == "off":
                w_soc = np.ones_like(soc_sust)
            else:
                sh0, sh1 = float(cfg["soc_high_onset"]), float(cfg["soc_high_full"])
                sl0, sl1 = float(cfg["soc_low_onset"]), float(cfg["soc_low_full"])
                if not (0.0 <= sh0 < sh1 <= 1.0 and 0.0 <= sl1 < sl0 <= 1.0):
                    raise ValueError("SOC ramp bounds invalid.")
                t_high = (soc_sust - sh0) / max(sh1 - sh0, 1e-9)
                t_low  = (sl0 - soc_sust) / max(sl0 - sl1, 1e-9)
                g_high = _smoothstep01(t_high) ** float(cfg["soc_high_pow"])
                g_low  = _smoothstep01(t_low)  ** float(cfg["soc_low_pow"])
                w_soc = 1.0 + float(cfg["soc_high_gain"]) * g_high + float(cfg["soc_low_gain"]) * g_low

            apply_mode = str(cfg["soc_apply"]).lower()
            if apply_mode == "charge":
                w_soc = np.where(current.values > float(cfg["eps_current"]), w_soc, 1.0)
            elif apply_mode == "discharge":
                w_soc = np.where(current.values < -float(cfg["eps_current"]), w_soc, 1.0)

            # --- Combine multipliers while current flows ---
            active = (abs_I >= float(cfg["eps_current"]))
            w_total = np.clip(w_t * w_c * w_soc, float(cfg["min_weight"]), float(cfg["max_weight"]))
            w_total = np.where(active, w_total, 1.0)

            # --- Cycle counts ---
            std_cycles = np.cumsum(thr_Ah) / (2.0 * C_Ah)
            eq_cycles = np.cumsum(thr_Ah * w_total) / (2.0 * C_Ah)
            life_frac = eq_cycles / rated_cycles

            idx = current.index
            return {
                "std_cycle_count": pd.Series(std_cycles, index=idx, name="std_cycle_count"),
                "equivalent_cycle_count": pd.Series(eq_cycles, index=idx, name="equivalent_cycle_count"),
                "cycle_life_fraction": pd.Series(life_frac, index=idx, name="cycle_life_fraction"),
            }

        except KeyError as e:
            miss = str(e).strip("'")
            raise KeyError(f"Missing required input or parameter: '{miss}'. "
                           f"Provide 'current' series and parameters 'capacity' & 'rated_cycle_count'.") from e
        except Exception as e:
            raise RuntimeError(f"[enhanced_equivalent_cycles] Failed during processing: {e}") from e
