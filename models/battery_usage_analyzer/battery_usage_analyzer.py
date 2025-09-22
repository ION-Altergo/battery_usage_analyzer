"""
Multi-Layer Battery Usage Analyzer

Analyzes battery usage patterns through multi-layer segmentation:
- Layer 0: Operating modes (charge/discharge/idle)
- Layer 1: Data-driven change points
- Layer 2: Domain-specific phases (rest, CC/CV charge, discharge)
"""

from typing import Dict, Union
import pandas as pd
import numpy as np

from altergo_sdk.boiler_plate import Model, register_model
try:
    from .utils import (
        infer_dt_s, robust_z, enforce_min_gap, ids_from_boundaries,
        majority_label_per_segment, rolling_ewm, detect_peaks_over_threshold,
        safe_quantile, soc_to_unit, derivative, compose_change_score,
        label_operating_mode, phase_labels
    )
except ImportError:
    # When running as standalone module
    from utils import (
        infer_dt_s, robust_z, enforce_min_gap, ids_from_boundaries,
        majority_label_per_segment, rolling_ewm, detect_peaks_over_threshold,
        safe_quantile, soc_to_unit, derivative, compose_change_score,
        label_operating_mode, phase_labels
    )


@register_model("battery_usage_analyzer", metadata={
    "category": "Other",
    "complexity": "Medium", 
    "computational_cost": "Medium"
})
class BatteryUsageAnalyzer(Model):
    """
    Multi-Layer Battery Usage Analyzer
    
    Features:
    - Layer 0: Operating modes (charge/discharge/idle)
    - Layer 1: Data-driven change points on multi-signal derivatives
    - Layer 2: Domain-specific phases (rest, CC/CV charge, discharge)
    - Optional Layer 3: Rainflow cycle segmentation on SoC
    - All metadata defined in model.json manifest
    """

    def process(self, data: Dict[str, Union[pd.Series, float]]) -> Dict[str, pd.Series]:
        """
        Multi-layer segmentation of battery usage timeline.
        
        Args:
            data: Dictionary containing:
                - current: Battery current time series [A] (required)
                - soc: State of charge [0-1 or %] (required)
                - v_cell_min: Minimum cell voltage [V] (required)
                - v_cell_max: Maximum cell voltage [V] (required)
                - t_cell_min: Minimum cell temperature [°C] (required)
                - t_cell_max: Maximum cell temperature [°C] (required)
                
        Returns:
            Dictionary containing segmentation results with logical names:
            - seg_l0_id: Operating mode segment IDs
            - mode: Operating mode labels
            - boundary_l1: Data-driven boundary markers
            - seg_l1_id: Data-driven segment IDs
            - phase: Domain phase labels
            - seg_l2_id: Domain phase segment IDs
        """
        
        try:
            # Get required data
            I = data["current"].copy()
            soc = data["soc"].copy()
            vmin = data["v_cell_min"].copy()
            vmax = data["v_cell_max"].copy()
            tmin = data["t_cell_min"].copy()
            tmax = data["t_cell_max"].copy()
            
            # Assemble and clean frame
            df = pd.DataFrame({
                "current": pd.to_numeric(I, errors="coerce"),
                "soc": pd.to_numeric(soc, errors="coerce"),
                "v_cell_min": pd.to_numeric(vmin, errors="coerce"),
                "v_cell_max": pd.to_numeric(vmax, errors="coerce"),
                "t_cell_min": pd.to_numeric(tmin, errors="coerce"),
                "t_cell_max": pd.to_numeric(tmax, errors="coerce"),
            }).sort_index()
            df = df.dropna(how="all")
            if df.empty:
                raise ValueError("All input series are empty after alignment/cleaning.")

            # Basic setup
            dt_s = infer_dt_s(df.index)
            window_s = float(self.config.get("window_s", 5.0))
            win_n = max(1, int(round(window_s / max(dt_s, 1e-9))))
            # Smoothing for derivative stability
            df_s = df.apply(lambda s: rolling_ewm(s, win_n))

            # Normalize SoC if needed
            df_s["soc"] = soc_to_unit(df_s["soc"])

            # Derivatives (per second)
            dI_dt = derivative(df_s["current"], dt_s)
            dSOC_dt = derivative(df_s["soc"], dt_s)
            dVmin_dt = derivative(df_s["v_cell_min"], dt_s)
            dVmax_dt = derivative(df_s["v_cell_max"], dt_s)
            dTmin_dt = derivative(df_s["t_cell_min"], dt_s)
            dTmax_dt = derivative(df_s["t_cell_max"], dt_s)

            deriv = pd.DataFrame({
                "current": dI_dt,
                "soc": dSOC_dt,
                "v_cell_min": dVmin_dt,
                "v_cell_max": dVmax_dt,
                "t_cell_min": dTmin_dt,
                "t_cell_max": dTmax_dt
            })

            # --- Layer 0: Operating Mode ----------------------------------------------------------
            # Thresholds with sensible fallbacks against data scale
            i99 = float(np.nanquantile(df_s["current"].abs().values, 0.99)) if df_s["current"].notna().any() else 1.0
            i_rest_th = float(self.config.get("i_rest_th_A", max(0.01 * i99, 1.5)))
            i_charge_on = float(self.config.get("i_charge_on_A", max(2.5 * i_rest_th, 1.0)))
            i_discharge_on = float(self.config.get("i_discharge_on_A", max(2.5 * i_rest_th, 1.0)))
            min_state_duration_s = float(self.config.get("min_state_duration_s", 120.0))
            min_state_gap = max(1, int(round(min_state_duration_s / max(dt_s, 1e-9))))

            raw_mode = label_operating_mode(df_s["current"], i_charge_on, i_discharge_on)
            # Raw boundaries when mode changes
            l0_raw_bound = pd.Series(False, index=df.index)
            mode_shift = (raw_mode != raw_mode.shift(1))
            l0_raw_bound[mode_shift.fillna(True)] = True
            # Enforce minimum dwell time
            l0_bound = enforce_min_gap(l0_raw_bound, min_state_gap)
            seg_l0_id = ids_from_boundaries(l0_bound)
            mode = majority_label_per_segment(raw_mode, seg_l0_id)

            # --- Layer 1: Data-driven change points ----------------------------------------------
            weights = {
                "current": float(self.config.get("w_current", 1.0)),
                "soc": float(self.config.get("w_soc", 0.5)),
                "v_cell_min": float(self.config.get("w_vmin", 0.8)),
                "v_cell_max": float(self.config.get("w_vmax", 0.8)),
                "t_cell_min": float(self.config.get("w_tmin", 0.3)),
                "t_cell_max": float(self.config.get("w_tmax", 0.3)),
            }
            score = compose_change_score(deriv, weights, smooth_n=win_n)
            q = float(self.config.get("change_quantile", 0.97))
            default_thr = float(np.nanmean(score.values) + 2.0 * np.nanstd(score.values))
            thr = safe_quantile(score.values, q, default_thr)
            peaks = detect_peaks_over_threshold(score, thr)

            # Candidate boundaries: union of (peaks) and (L0 boundaries)
            l1_candidates = peaks | l0_bound
            min_seg_duration_s = float(self.config.get("min_seg_duration_s", 120.0))
            min_seg_gap = max(1, int(round(min_seg_duration_s / max(dt_s, 1e-9))))
            l1_bound = enforce_min_gap(l1_candidates, min_seg_gap)
            seg_l1_id = ids_from_boundaries(l1_bound)

            # --- Layer 2: Domain phases (CC/CV/Rest/Discharge) -----------------------------------
            cv_voltage_window_V = float(self.config.get("cv_voltage_window_V", 0.03))
            dv_dt_small = float(self.config.get("dv_dt_small_V_per_s", 2e-4))
            phase = phase_labels(df_s["current"], df_s["v_cell_max"], dI_dt, dVmax_dt,
                                i_rest_th, i_charge_on, i_discharge_on,
                                cv_voltage_window_V, dv_dt_small)
            # Boundaries when phase changes; union with L1 to maintain data-driven cuts
            l2_raw_bound = pd.Series(False, index=df.index)
            phase_shift = (phase != phase.shift(1))
            l2_raw_bound[phase_shift.fillna(True)] = True
            l2_candidates = l2_raw_bound | l1_bound
            min_phase_duration_s = float(self.config.get("min_phase_duration_s", 60.0))
            min_phase_gap = max(1, int(round(min_phase_duration_s / max(dt_s, 1e-9))))
            l2_bound = enforce_min_gap(l2_candidates, min_phase_gap)
            seg_l2_id = ids_from_boundaries(l2_bound)

            # Optional: Rainflow cycle segmentation on SoC (Layer 3)
            # We expose a simple binary that marks extra boundaries if enabled.
            if bool(self.config.get("enable_rainflow", False)):
                try:
                    rf_min_range = float(self.config.get("rf_min_range_soc_pct", 5.0)) / 100.0
                    # Turning points on SoC
                    s = soc_to_unit(df_s["soc"])
                    tp = s[(s.shift(1) < s) & (s.shift(-1) < s) | (s.shift(1) > s) & (s.shift(-1) > s)].index
                    # Very lightweight: mark large reversals as boundaries
                    rf_bound = pd.Series(False, index=s.index)
                    last_idx = None
                    last_val = None
                    for idx in tp:
                        v = s.loc[idx]
                        if last_val is None:
                            last_val, last_idx = v, idx
                            continue
                        if abs(v - last_val) >= rf_min_range:
                            rf_bound.loc[idx] = True
                            last_val, last_idx = v, idx
                    # Merge into L2 boundaries (most conservative)
                    l2_bound = enforce_min_gap(l2_bound | rf_bound, min_phase_gap)
                    seg_l2_id = ids_from_boundaries(l2_bound)
                except Exception:
                    # Keep segmentation without rainflow if anything goes wrong
                    pass

            # Post-process Layer 0: Consolidate segments with same mode
            l0_consolidated_bound, seg_l0_consolidated_id, mode_consolidated = self._consolidate_same_mode_segments(
                l0_bound, seg_l0_id, mode, raw_mode
            )
            
            # Generate segment summary DataFrame
            segment_summary = self._generate_segment_summary(
                df_s, deriv, seg_l0_id, seg_l0_consolidated_id, mode_consolidated, raw_mode
            )

            # Return with logical output names (framework handles blueprint mapping for upload)
            return {
                "seg_l0_id": seg_l0_consolidated_id,
                "mode": mode_consolidated.astype("object"),
                "boundary_l1": l1_bound.astype(int),
                "seg_l1_id": seg_l1_id,
                "phase": phase.astype("object"),
                "seg_l2_id": seg_l2_id,
                "segment_summary": segment_summary
            }
        
        except KeyError as e:
            raise ValueError(f"Missing required input: {e}. "
                             f"Expected keys: current, soc, v_cell_min, v_cell_max, t_cell_min, t_cell_max")
        except Exception as e:
            raise RuntimeError(f"Failed to analyze battery usage: {str(e)}") from e
    
    def _consolidate_same_mode_segments(self, boundaries, segment_ids, modes, raw_modes):
        """
        Post-process to merge consecutive segments with the same mode.
        
        Returns:
            consolidated_boundaries, consolidated_segment_ids, consolidated_modes
        """
        # Find boundaries between segments with different modes
        new_boundaries = boundaries.copy()
        
        # Get unique segment boundaries and their modes
        unique_seg_ids = segment_ids.unique()
        
        for i in range(1, len(unique_seg_ids)):
            current_seg_id = unique_seg_ids[i]
            previous_seg_id = unique_seg_ids[i-1]
            
            # Get mode for current and previous segments
            current_mode = modes[segment_ids == current_seg_id].iloc[0]
            previous_mode = modes[segment_ids == previous_seg_id].iloc[0]
            
            # If modes are the same, remove this boundary
            if current_mode == previous_mode:
                boundary_idx = (segment_ids == current_seg_id).idxmax()
                new_boundaries.loc[boundary_idx] = False
        
        # Recreate segment IDs and modes with consolidated boundaries
        new_seg_ids = ids_from_boundaries(new_boundaries)
        new_modes = majority_label_per_segment(raw_modes, new_seg_ids)
        
        return new_boundaries, new_seg_ids, new_modes
    
    def _generate_segment_summary(self, df_signals, df_derivatives, 
                                  original_seg_ids, consolidated_seg_ids, 
                                  consolidated_modes, raw_modes):
        """
        Generate a DataFrame summarizing each consolidated segment with enriched battery analytics.
        """
        segments_data = []
        unique_seg_ids = consolidated_seg_ids.unique()
        
        # Get configuration for enriched analysis
        nominal_capacity_Ah = float(self.config.get("nominal_capacity_Ah", 100.0))
        nominal_voltage_V = float(self.config.get("nominal_voltage_V", 3.7))
        
        for i, seg_id in enumerate(unique_seg_ids):
            seg_mask = (consolidated_seg_ids == seg_id)
            seg_data = df_signals[seg_mask]
            seg_deriv = df_derivatives[seg_mask]
            seg_raw_modes = raw_modes[seg_mask]
            seg_orig_ids = original_seg_ids[seg_mask]
            
            if seg_data.empty:
                continue
                
            # Basic segment info
            start_time = seg_data.index[0]
            end_time = seg_data.index[-1]
            duration_s = (end_time - start_time).total_seconds() if hasattr(start_time, 'total_seconds') else float(len(seg_data))
            duration_h = duration_s / 3600.0
            mode = consolidated_modes[seg_mask].iloc[0]
            
            # Count original segments and mode transitions that were filtered
            raw_segments_count = len(seg_orig_ids.unique())
            mode_changes = (seg_raw_modes != seg_raw_modes.shift(1)).sum()
            mode_transitions_filtered = max(0, mode_changes - 1)  # -1 because first change is segment start
            
            # Signal extractions
            current = seg_data['current']
            soc = seg_data['soc']
            v_min = seg_data['v_cell_min']
            v_max = seg_data['v_cell_max']
            v_avg = (v_min + v_max) / 2
            t_min = seg_data['t_cell_min']
            t_max = seg_data['t_cell_max']
            t_avg = (t_min + t_max) / 2
            
            # Derivative signals
            current_rate = seg_deriv['current']
            soc_rate = seg_deriv['soc']
            v_rate = (seg_deriv['v_cell_min'] + seg_deriv['v_cell_max']) / 2
            t_rate = (seg_deriv['t_cell_min'] + seg_deriv['t_cell_max']) / 2
            
            # Context from adjacent segments
            prev_mode = segments_data[-1]['mode'] if i > 0 else None
            next_mode = None  # Will be filled in next iteration if exists
            
            # === ENRICHED BATTERY ANALYSIS ===
            
            # 1. Energy & Power Analysis
            avg_voltage = float(v_avg.mean())
            avg_current = float(current.mean())
            avg_power_W = avg_voltage * avg_current
            energy_throughput_Wh = abs(avg_power_W * duration_h)
            
            # C-Rate analysis (current relative to capacity)
            c_rate_mean = abs(avg_current) / nominal_capacity_Ah
            c_rate_max = float(current.abs().max()) / nominal_capacity_Ah
            
            # 2. Mode-Specific Characterization
            charge_efficiency = None
            discharge_capacity_Ah = None
            cc_cv_analysis = {}
            
            if mode == 'charge':
                soc_gain = float(soc.iloc[-1] - soc.iloc[0])
                if energy_throughput_Wh > 0:
                    charge_efficiency = soc_gain / 100.0 * nominal_capacity_Ah / energy_throughput_Wh
                
                # CC/CV phase detection
                current_variation = float(current.std() / current.mean()) if current.mean() != 0 else 0
                voltage_rise = float(v_avg.iloc[-1] - v_avg.iloc[0])
                
                if current_variation < 0.1 and voltage_rise > 0.1:
                    charge_pattern = 'CC'
                elif current_variation > 0.3 and voltage_rise < 0.05:
                    charge_pattern = 'CV'
                elif current_variation > 0.2 and voltage_rise > 0.1:
                    charge_pattern = 'CC-CV'
                else:
                    charge_pattern = 'irregular'
                    
                cc_cv_analysis = {
                    'charge_pattern': charge_pattern,
                    'current_variation_coeff': current_variation,
                    'voltage_rise_V': voltage_rise,
                    'taper_ratio': float(current.iloc[-1] / current.iloc[0]) if current.iloc[0] != 0 else 1.0
                }
                
            elif mode == 'discharge':
                discharge_capacity_Ah = abs(float(current.mean() * duration_h))
                voltage_drop = float(v_avg.iloc[0] - v_avg.iloc[-1])
                power_fade = None
                if len(current) > 2:
                    initial_power = float(current.iloc[0] * v_avg.iloc[0])
                    final_power = float(current.iloc[-1] * v_avg.iloc[-1])
                    if initial_power != 0:
                        power_fade = (initial_power - final_power) / abs(initial_power)
            
            # 3. Thermal Analysis
            thermal_rise_C = float(t_max.iloc[-1] - t_min.iloc[0])
            thermal_gradient_C_per_min = thermal_rise_C / (duration_s / 60.0) if duration_s > 0 else 0
            max_cell_temp_delta_C = float((t_max - t_min).max())
            thermal_stability = 1.0 / (1.0 + float(t_avg.std()))
            
            # 4. Voltage & Cell Balance Analysis
            voltage_stability = 1.0 / (1.0 + float(v_avg.std()))
            cell_imbalance_V = float((v_max - v_min).mean())
            cell_balance_quality = 1.0 / (1.0 + cell_imbalance_V)
            
            # Estimate internal resistance (rough approximation)
            internal_resistance_ohm = None
            if len(current) > 1 and current.std() > 0.1:
                # Simple linear correlation between voltage and current changes
                current_change = current.iloc[-1] - current.iloc[0]
                voltage_change = v_avg.iloc[-1] - v_avg.iloc[0]
                if current_change != 0:
                    internal_resistance_ohm = abs(voltage_change / current_change)
            
            # 5. Operational Quality Metrics
            current_smoothness = 1.0 / (1.0 + float(current.std() / abs(current.mean()))) if current.mean() != 0 else 1.0
            
            # Stress indicators
            current_stress = c_rate_max  # High C-rates are stressful
            thermal_stress = max(0, max_cell_temp_delta_C - 5.0) / 10.0  # >5°C delta is stress
            voltage_stress = max(0, abs(avg_voltage - nominal_voltage_V) - 0.5) / 1.0  # >0.5V from nominal
            operational_stress = (current_stress + thermal_stress + voltage_stress) / 3.0
            
            # 6. SoC & Capacity Analysis
            soc_delta = float(soc.iloc[-1] - soc.iloc[0])
            soc_efficiency = abs(soc_delta) / (energy_throughput_Wh + 1e-6)  # SoC change per Wh
            
            # Self-discharge estimate for idle periods
            self_discharge_rate_pct_per_h = None
            if mode == 'idle' and duration_h > 0.1:
                soc_loss = -soc_delta if soc_delta < 0 else 0  # Only count losses
                self_discharge_rate_pct_per_h = soc_loss / duration_h
            
            # 7. Transition Analysis
            transition_smoothness_in = 1.0  # Default smooth
            transition_smoothness_out = 1.0
            if len(current) > 5:
                # Measure abruptness of current changes at segment boundaries
                initial_variation = float(current.iloc[:3].std())
                final_variation = float(current.iloc[-3:].std())
                transition_smoothness_in = 1.0 / (1.0 + initial_variation)
                transition_smoothness_out = 1.0 / (1.0 + final_variation)
            
            segment_info = {
                # === BASIC SEGMENT INFO ===
                'segment_id': int(seg_id),
                'start_time': start_time,
                'end_time': end_time,
                'duration_s': duration_s,
                'duration_h': duration_h,
                'mode': mode,
                
                # === CONSOLIDATION METADATA ===
                'raw_segments_count': raw_segments_count,
                'mode_transitions_filtered': mode_transitions_filtered,
                'prev_segment_mode': prev_mode,
                
                # === BASIC SIGNAL STATISTICS ===
                'current_mean_A': float(current.mean()),
                'current_std_A': float(current.std()),
                'current_min_A': float(current.min()),
                'current_max_A': float(current.max()),
                'soc_start_pct': float(soc.iloc[0]),
                'soc_end_pct': float(soc.iloc[-1]),
                'soc_delta_pct': soc_delta,
                'soc_mean_pct': float(soc.mean()),
                'voltage_avg_V': avg_voltage,
                'voltage_min_V': float(v_min.min()),
                'voltage_max_V': float(v_max.max()),
                'temperature_avg_C': float(t_avg.mean()),
                'temperature_min_C': float(t_min.min()),
                'temperature_max_C': float(t_max.max()),
                
                # === ENERGY & POWER ANALYSIS ===
                'avg_power_W': avg_power_W,
                'energy_throughput_Wh': energy_throughput_Wh,
                'c_rate_mean': c_rate_mean,
                'c_rate_max': c_rate_max,
                'soc_efficiency_pct_per_Wh': soc_efficiency,
                
                # === MODE-SPECIFIC METRICS ===
                'charge_efficiency': charge_efficiency,
                'discharge_capacity_Ah': discharge_capacity_Ah,
                'charge_pattern': cc_cv_analysis.get('charge_pattern') if mode == 'charge' else None,
                'current_variation_coeff': cc_cv_analysis.get('current_variation_coeff') if mode == 'charge' else None,
                'voltage_rise_V': cc_cv_analysis.get('voltage_rise_V') if mode == 'charge' else None,
                'taper_ratio': cc_cv_analysis.get('taper_ratio') if mode == 'charge' else None,
                'voltage_drop_V': locals().get('voltage_drop') if mode == 'discharge' else None,
                'power_fade_ratio': locals().get('power_fade') if mode == 'discharge' else None,
                'self_discharge_rate_pct_per_h': self_discharge_rate_pct_per_h,
                
                # === THERMAL ANALYSIS ===
                'thermal_rise_C': thermal_rise_C,
                'thermal_gradient_C_per_min': thermal_gradient_C_per_min,
                'max_cell_temp_delta_C': max_cell_temp_delta_C,
                'thermal_stability': thermal_stability,
                
                # === ELECTRICAL HEALTH & QUALITY ===
                'voltage_stability': voltage_stability,
                'cell_imbalance_V': cell_imbalance_V,
                'cell_balance_quality': cell_balance_quality,
                'internal_resistance_ohm': internal_resistance_ohm,
                'current_smoothness': current_smoothness,
                
                # === STRESS & OPERATIONAL QUALITY ===
                'current_stress': current_stress,
                'thermal_stress': thermal_stress,
                'voltage_stress': voltage_stress,
                'operational_stress': operational_stress,
                
                # === TRANSITION ANALYSIS ===
                'transition_smoothness_in': transition_smoothness_in,
                'transition_smoothness_out': transition_smoothness_out,
                
                # === DERIVATIVE RATES ===
                'current_rate_mean_A_per_s': float(current_rate.mean()),
                'soc_rate_mean_pct_per_s': float(soc_rate.mean()),
                'voltage_rate_mean_V_per_s': float(v_rate.mean()),
                'temperature_rate_mean_C_per_s': float(t_rate.mean()),
            }
            
            # Update previous segment's next_mode
            if i > 0:
                segments_data[-1]['next_segment_mode'] = mode
            
            segments_data.append(segment_info)
        
        return pd.DataFrame(segments_data)


if __name__ == "__main__":
    """Run tests when script is executed directly"""
    try:
        from .test_battery_usage_analyzer import run_all_tests
    except ImportError:
        from test_battery_usage_analyzer import run_all_tests
    
    passed, failed = run_all_tests()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if failed == 0 else 1)