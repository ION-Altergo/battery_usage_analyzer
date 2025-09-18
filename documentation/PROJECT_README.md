# Battery Digital Twin Models

A clean 2-layer architecture for battery digital twin models that separates platform concerns from battery science.

## Architecture Overview

```
├── altergo_interface/          # Layer 1: Digital Twin Interface
│   ├── base_model.py          # Base class for all models
│   ├── models.py              # Model discovery & management
│   ├── config.py              # Configuration extraction
│   ├── data.py                # Data loading & execution
│   └── output.py              # Result uploading
│
├── models/                    # Layer 2: Battery Science Models
│   ├── eq_cycles/            # Equivalent cycles calculation
│   ├── voltage_monitor/      # Voltage monitoring & alerts
│   ├── imbalance_analysis/   # Cell imbalance detection
│   └── soc_ocv_estimation/   # State of charge estimation
│
└── entrypoint.py             # Main execution entry point
```

## Key Design Principles

1. **Clear Separation**: Platform interface code is isolated from battery science logic
2. **Simple Contract**: Models only need to implement a `process()` method
3. **Minimal Dependencies**: Models import only from `altergo_interface`
4. **Easy Testing**: Models can be tested independently of platform

## Available Models

### Equivalent Cycles (`eq_cycles`)
Tracks battery usage by calculating equivalent full charge-discharge cycles.
- Supports coulombic efficiency corrections
- Optional SOH-based capacity compensation
- Configurable for incremental or full processing

### Voltage Monitor (`voltage_monitor`)
Monitors cell voltages and generates alerts for out-of-range conditions.
- Configurable voltage thresholds
- Temperature compensation support
- Real-time alert generation

### Imbalance Analysis (`imbalance_analysis`)
Analyzes cell voltage imbalances and estimates SOC differences.
- Detects voltage and SOC imbalances
- Confidence-based reporting
- Chemistry-specific OCV curves

### SOC/OCV Estimation (`soc_ocv_estimation`)
Estimates state of charge from open circuit voltage during rest periods.
- Automatic rest period detection
- Chemistry-specific lookup tables
- Available energy calculation

## Quick Start

1. **Configure Models**: Edit `altergo-settings.json` to enable desired models
2. **Set Parameters**: Configure model-specific parameters in the settings file
3. **Run Locally**: Use `dev-parameters.json` for local testing
4. **Deploy**: Push to repository and configure in Altergo platform

## Adding New Models

1. Create a new directory under `models/`
2. Implement a class inheriting from `Model`:
   ```python
   from altergo_interface import Model, register_model
   
   @register_model("your_model")
   class YourModel(Model):
       def process(self, data):
           # Your battery science logic here
           return {"output_name": result_series}
   ```
3. Create a `model.json` manifest file
4. The model will be automatically discovered

## Configuration

Models are configured through `altergo-settings.json`:
- Enable/disable models
- Set model-specific parameters
- Configure sensor mappings
- Control debug output

## Recent Improvements

- **Simplified from 3 to 2 layers**: Removed unnecessary abstraction
- **Reduced logging noise**: Only essential messages remain
- **Cleaner configuration**: Simplified nested dictionary access
- **Better separation**: Clear boundary between platform and science
