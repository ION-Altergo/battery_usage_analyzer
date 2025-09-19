# Enhanced Equivalent Cycles (Continuous SOC)

## Overview

The Enhanced Equivalent Cycles model provides advanced battery cycle counting with realistic stress factors that account for operating conditions. Unlike simple Ah-throughput counting, this model applies condition-weighted penalties based on:

- **Continuous SOC stress** - Penalties for sustained high/low state of charge
- **C-rate stress** - Impact of charge/discharge rates 
- **Temperature stress** - Arrhenius-based aging acceleration
- **Chemistry-specific tuning** - Optimized defaults for LFP batteries

The model outputs both standard equivalent cycles (for comparison) and condition-weighted cycles that better reflect real-world battery degradation.

## How It Works

### Core Algorithm

1. **Throughput Calculation**: Converts current measurements to Ah throughput over time
2. **Stress Multipliers**: Calculates individual stress factors for SOC, C-rate, and temperature
3. **Smoothing**: Applies exponential moving averages to capture sustained operating conditions
4. **Weighting**: Combines stress multipliers when current is flowing (above activity threshold)
5. **Cycle Counting**: Accumulates both standard and weighted equivalent cycles

### Key Features

- **Causal Processing**: Uses only past data, suitable for real-time applications
- **Irregular Time Series**: Handles non-uniform sampling intervals
- **Smooth Penalties**: Uses cubic smoothstep functions to avoid discontinuities
- **Chemistry Tuning**: LFP-optimized defaults with full configurability

## Inputs

### Required Time Series
- **`current`** (A): Battery current measurements (absolute value used)
- **`capacity`** (Ah): Nominal battery capacity for normalization
- **`rated_cycle_count`** (cycles): Expected cycle life at reference conditions

### Optional Time Series
- **`soc`** (p.u.): State of charge [0..1] (defaults to 0.5 if missing)
- **`temperature`** (°C): Cell/pack temperature (defaults to 25°C if missing)

## Outputs

- **`std_cycle_count`**: Baseline equivalent cycles from Ah throughput only
- **`equivalent_cycle_count`**: Condition-weighted equivalent cycles
- **`cycle_life_fraction`**: Ratio of equivalent cycles to rated cycle count

## Configuration Parameters

### SOC Stress Parameters

The model applies smooth penalties for sustained operation at extreme SOC levels:

#### High SOC Stress (dominant for LFP)
```json
{
  "soc_high_onset": 0.80,    // Start penalty ramp at 80% SOC
  "soc_high_full": 0.96,     // Full penalty at 96% SOC  
  "soc_high_gain": 0.45,     // +45% stress multiplier at full penalty
  "soc_high_pow": 1.0        // Linear ramp shape (>=1 for convex)
}
```

#### Low SOC Stress (weak for LFP)
```json
{
  "soc_low_onset": 0.08,     // Start penalty at 8% SOC
  "soc_low_full": 0.02,      // Full penalty at 2% SOC
  "soc_low_gain": 0.10,      // +10% stress multiplier at full penalty  
  "soc_low_pow": 1.0         // Linear ramp shape
}
```

#### SOC Control
```json
{
  "soc_weight_mode": "smoothstep",  // "smoothstep" or "off"
  "soc_sustain_tau_hours": 1.5,    // EMA time constant for SOC smoothing
  "soc_apply": "both"               // Apply during "charge", "discharge", or "both"
}
```

### C-Rate Stress Parameters

Higher charge/discharge rates increase stress:

```json
{
  "c_rate_ref": 0.50,        // Reference C-rate (rated condition)
  "c_rate_exponent": 1.0,    // Power law exponent for high C-rate penalty
  "alpha_c": 1.0,            // Penalty factor above reference (30% increase)
  "beta_c": 0.20,            // Relief factor below reference (20% reduction)
  "sustain_tau_hours": 0.5   // EMA time constant for C-rate smoothing
}
```

### Temperature Stress Parameters

Cyclic aging temperature acceleration with low-temperature charge penalty:

```json
{
  "temp_ref_c": 25.0,               // Reference temperature (°C)
  "q10_cyclic": 1.30,               // Factor increase per 10°C (gentler than calendar aging)
  "lowT_charge_on": true,           // Enable low-temperature charge penalty
  "lowT_ref_c": 15.0,               // Temperature below which charge penalty applies
  "lowT_charge_gain_per_10C": 0.10  // +10% penalty per -10°C during charging
}
```

#### Low-Temperature Charge Penalty

The model includes a specialized penalty for charging at low temperatures to account for lithium plating risks:

- **Activation**: Only applies when `current > eps_current` (net charging)
- **Temperature threshold**: Penalty increases below `lowT_ref_c` (default 15°C)
- **Linear scaling**: Each 10°C below reference adds `lowT_charge_gain_per_10C` (default 10%) 
- **Example**: Charging at 5°C adds ~10% penalty, at -5°C adds ~20% penalty

### General Parameters

```json
{
  "eps_current": 0.001,      // Current activity threshold (A)
  "min_weight": 0.2,         // Lower clamp on stress multiplier
  "max_weight": 3.0          // Upper clamp on stress multiplier
}
```

## Example Configurations

### LFP-Optimized (Default)
Toned-down penalties optimized for LFP chemistry:
```json
{
  "soc_high_onset": 0.80,
  "soc_high_gain": 0.45,
  "soc_low_gain": 0.10,
  "c_rate_ref": 0.50,
  "alpha_c": 1.0,
  "q10_cyclic": 1.30,
  "max_weight": 3.0
}
```

### High-Performance Application
Further reduced penalties for demanding applications:
```json
{
  "soc_high_onset": 0.90,    // Allow very high SOC before penalty
  "soc_high_gain": 0.20,     // Minimal high-SOC penalty
  "alpha_c": 0.8,            // Reduced C-rate penalty
  "max_weight": 2.5,         // Lower maximum stress multiplier
  "lowT_charge_on": false    // Disable low-temp charge penalty
}
```

### SOC-Agnostic Operation
Disable SOC penalties entirely:
```json
{
  "soc_weight_mode": "off"
}
```

## Stress Multiplier Examples

| Condition | Multiplier | Impact |
|-----------|------------|---------|
| 25°C, 0.5C, 50% SOC | 1.0× | Baseline (no penalty) |
| 25°C, 0.5C, 90% SOC | 1.3× | High SOC penalty (reduced) |
| 25°C, 1.0C, 50% SOC | 1.3× | High C-rate penalty (reduced) |
| 35°C, 0.5C, 50% SOC | 1.3× | Temperature penalty (Q10=1.3) |
| 10°C, 0.5C charging | 1.15× | Low-temp charge penalty |
| 35°C, 1.0C, 90% SOC | 3.0× | Combined stress (clamped at max) |

## Usage Notes

### Time Series Alignment
- All inputs are automatically aligned to the `current` time index
- Missing values are forward-filled or use defaults
- Irregular sampling intervals are handled correctly

### Real-Time Processing
- Algorithm is strictly causal (no look-ahead)
- Suitable for streaming/incremental processing
- Memory requirements scale with time series length only

### Validation
- Input validation ensures proper data types and ranges
- Configuration parameters are bounds-checked
- Clear error messages for missing or invalid inputs

### Performance
- Computational cost: Medium (due to smoothing operations)
- Memory usage: Linear with time series length
- Suitable for datasets up to millions of points

## References

- SOC stress factors based on LFP degradation studies
- Temperature acceleration follows Arrhenius relationship
- C-rate penalties derived from cycle life testing
- Smoothing prevents unrealistic step changes in stress assessment
