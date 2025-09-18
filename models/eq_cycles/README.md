# Equivalent Cycles Model

## Overview

The Equivalent Cycles Model is a sophisticated battery analytics tool that tracks cumulative battery usage by calculating equivalent charge/discharge cycles. This model is fundamental for battery health monitoring, lifecycle prediction, and maintenance scheduling in battery energy storage systems (BESS).

## Battery Science Background

### What are Equivalent Cycles?

An **equivalent cycle** represents a standardized measure of battery usage that normalizes different partial charge/discharge operations into full cycle equivalents. In battery science:

- **One full cycle** = Complete discharge from 100% to 0% + complete charge from 0% to 100%
- **Equivalent cycles** = Cumulative measure accounting for partial cycles and varying depths of discharge

### Why Equivalent Cycles Matter

1. **Battery Degradation**: Most battery degradation models use cycle count as a primary factor
2. **Capacity Fade**: Each cycle contributes to irreversible capacity loss
3. **Warranty Tracking**: Many battery warranties are based on cycle limits
4. **Maintenance Planning**: Predicts when battery replacement or refurbishment is needed
5. **Performance Analysis**: Correlates usage patterns with battery health metrics

### Scientific Principles

#### Coulomb Counting
The model uses **coulomb counting** (ampere-hour integration) to track charge throughput:
```
Charge_transferred = ∫ |I(t)| × dt
```
Where:
- `I(t)` = instantaneous current
- `dt` = time differential

#### Cycle Calculation
Equivalent cycles are calculated as:
```
Equivalent_Cycles = Total_Charge_Transferred / (2 × Effective_Capacity)
```

The factor of 2 accounts for the fact that one complete cycle involves both charge AND discharge operations.

## Model Features

### 1. Coulombic Efficiency Correction

Real batteries don't achieve 100% efficiency during charge/discharge operations due to:
- **Internal resistance losses** (heat generation)
- **Side reactions** (especially during charging)
- **Electrochemical inefficiencies**

The model applies configurable efficiency factors:
- **Charge efficiency** (default: 98%): Accounts for energy lost during charging
- **Discharge efficiency** (default: 99%): Accounts for energy lost during discharging

### 2. State of Health (SOH) Compensation

As batteries age, their effective capacity decreases. The model optionally compensates for this by:
- Using SOH data to calculate **effective capacity** = nominal_capacity × (SOH/100)
- Providing more accurate cycle counting as the battery degrades
- Accounting for capacity fade in equivalent cycle calculations

### 3. Incremental Processing

The model supports incremental updates, essential for:
- **Real-time monitoring** in operational systems
- **Historical data processing** without recomputing entire datasets
- **Efficient computation** in continuous monitoring applications

## Technical Implementation

### Input Requirements

#### Required Inputs
- **Current** (`A`): Battery current measurements
  - Positive values = discharge
  - Negative values = charge
- **Capacity** (`Ah`): Battery nominal capacity parameter

#### Optional Inputs
- **SOH** (`%`): State of Health for capacity compensation
  - Enables more accurate cycle counting for aged batteries
  - Defaults to 100% if not provided

### Configuration Parameters

| Parameter | Default | Description | Battery Science Context |
|-----------|---------|-------------|------------------------|
| `charge_efficiency` | 0.98 | Coulombic efficiency during charging | Accounts for charging losses (2% typical) |
| `discharge_efficiency` | 0.99 | Coulombic efficiency during discharging | Accounts for discharge losses (1% typical) |
| `enable_efficiency_correction` | true | Enable/disable efficiency corrections | Critical for accurate degradation modeling |
| `initial_eq_cycle` | 0 | Starting equivalent cycle count | For continuing previous calculations |

### Algorithm Details

1. **Time-based Integration**: Uses timestamp differences to calculate charge transferred over time
2. **Efficiency Application**: Applies different efficiency factors for charge vs discharge
3. **Capacity Compensation**: Adjusts for battery aging using SOH data
4. **Cumulative Tracking**: Maintains running total of equivalent cycles

## Battery Chemistry Considerations

### Efficiency Factors by Chemistry

Different battery chemistries have characteristic efficiency ranges:

| Chemistry | Charge Efficiency | Discharge Efficiency | Notes |
|-----------|------------------|---------------------|-------|
| Li-ion (NMC) | 95-99% | 98-99% | High efficiency, minimal losses |
| Li-ion (LFP) | 96-99% | 98-99% | Very stable, good round-trip efficiency |
| Lead-acid | 85-95% | 85-95% | Lower efficiency, significant losses |
| Li-ion (LTO) | 95-98% | 98-99% | Fast charging capable |

### Degradation Mechanisms

The equivalent cycles metric correlates with several degradation mechanisms:

1. **Calendar Aging**: Time-based degradation (independent of cycles)
2. **Cycle Aging**: Usage-based degradation (proportional to equivalent cycles)
3. **SEI Growth**: Solid Electrolyte Interface formation (cycle-dependent)
4. **Active Material Loss**: Physical degradation from repeated cycling

## Applications

### Primary Use Cases

1. **Fleet Management**: Track usage across multiple battery systems
2. **Warranty Validation**: Verify cycle-based warranty claims
3. **Predictive Maintenance**: Schedule maintenance based on cycle count
4. **Performance Benchmarking**: Compare efficiency across different systems
5. **Research & Development**: Analyze battery performance in controlled studies

### Integration Examples

```python
# Basic usage tracking
eq_cycles = model.process({
    'current': current_timeseries,
    'capacity': 100.0  # 100 Ah battery
})

# Advanced usage with SOH compensation
eq_cycles = model.process({
    'current': current_timeseries,
    'capacity': 100.0,
    'soh': soh_timeseries  # Enables capacity compensation
})
```

## Validation and Accuracy

### Expected Accuracy
- **With SOH compensation**: ±2-3% for equivalent cycle counting
- **Without SOH compensation**: ±5-10% (depending on battery age)
- **Efficiency correction**: Improves accuracy by 2-5% in real systems

### Validation Methods
1. Compare with manufacturer cycle test data
2. Cross-validate with capacity fade measurements
3. Verify against known charge/discharge patterns

## Limitations and Considerations

### Model Limitations
1. **Depth of Discharge**: Doesn't account for varying degradation rates at different DOD
2. **Temperature Effects**: No temperature compensation for efficiency
3. **C-rate Dependencies**: Efficiency factors don't vary with charge/discharge rates
4. **Calendar Aging**: Only tracks cycle-based aging, not time-based degradation

### Best Practices
1. Calibrate efficiency factors for specific battery chemistry and operating conditions
2. Update SOH regularly for accurate capacity compensation
3. Validate cycle counts against known test patterns
4. Consider temperature and C-rate effects in system-level analysis

## Version History

- **v2.1.0**: Enhanced with SOH compensation and efficiency corrections
- **v2.0.0**: Added incremental processing and improved configuration
- **v1.0.0**: Basic equivalent cycles calculation

## References

1. Battery University - Cycle Life and Testing
2. IEC 61427 - Battery Test Procedures
3. IEEE 1679 - Battery Characterization Methods
4. Scientific literature on battery degradation modeling
