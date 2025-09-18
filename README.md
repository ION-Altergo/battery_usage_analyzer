# Battery Digital Twin Models

A clean 2-layer architecture for battery digital twin models on the Altergo platform, with clear separation between platform integration and battery science implementations.

## Overview

This repository provides a framework for developing and deploying battery models as digital twins. The architecture emphasizes simplicity and maintainability with just two layers:

1. **Digital Twin Interface** (`altergo_interface/`) - Handles all platform interactions
2. **Battery Science Models** (`models/`) - Pure battery algorithm implementations

## Quick Start

### Prerequisites

- Python 3.8+
- Altergo SDK
- Required Python packages (see `requirements.txt`)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd demo-eq-cycle-model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure for local testing**
   Create a `dev-parameters.json` file:
   ```json
   {
       "altergoUserApiKey": "YOUR_API_KEY",
       "altergoFactoryApi": "https://YOUR_COMPANY.altergo.io",
       "altergoIotApi": "https://iot.YOUR_COMPANY.altergo.io",
       "assetId": "YOUR_ASSET_ID"
   }
   ```

4. **Run the models**
   ```bash
   python entrypoint.py dev-parameters.json
   ```

## Architecture

### Layer 1: Digital Twin Interface (`altergo_interface/`)

The unified interface layer that handles all interactions with the Altergo platform:

- **`base_model.py`** - Base class that all models inherit from
- **`models.py`** - Simple model discovery and instantiation
- **`config.py`** - Configuration extraction and sensor mapping
- **`data.py`** - Data loading, preparation, and model execution
- **`output.py`** - Result uploading to the platform
- **`utils.py`** - Common utilities (datetime parsing, etc.)

### Layer 2: Battery Science Models (`models/`)

Pure battery science implementations, each in its own directory:

```
models/
├── eq_cycles/          # Equivalent cycles tracking
├── voltage_monitor/    # Voltage monitoring and alerts
├── imbalance_analysis/ # Cell imbalance detection
└── soc_ocv_estimation/ # State of charge estimation
```

## Available Models

### 1. Equivalent Cycles (`eq_cycles`)
Tracks cumulative battery usage by calculating equivalent full charge-discharge cycles.

**Features:**
- Coulombic efficiency corrections
- Optional SOH-based capacity compensation
- Incremental or full processing modes

**Configuration:**
- `charge_efficiency`: Charging efficiency (0.95-0.99)
- `discharge_efficiency`: Discharging efficiency (0.95-0.99)
- `enable_efficiency_correction`: Enable/disable efficiency corrections

### 2. Voltage Monitor (`voltage_monitor`)
Monitors cell voltages and generates alerts for out-of-range conditions.

**Features:**
- Configurable voltage thresholds
- Temperature compensation support
- Real-time status and event tracking

### 3. Imbalance Analysis (`imbalance_analysis`)
Analyzes voltage imbalances between cells and estimates SOC differences.

**Features:**
- Voltage range monitoring
- SOC imbalance estimation with confidence levels
- Chemistry-specific OCV curves

### 4. SOC/OCV Estimation (`soc_ocv_estimation`)
Estimates state of charge from open circuit voltage during rest periods.

**Features:**
- Automatic rest period detection
- Chemistry-specific lookup tables
- Available energy calculation

## Configuration

Models are configured through `altergo-settings.json`:

```json
{
    "parameters": {
        "configuration": {
            "execution": {
                "enabled_models": "eq_cycles,voltage_monitor",
                "max_days_period_compute": 7
            },
            "models": {
                "eq_cycles": {
                    "debug_mode": true,
                    "compute_type": "incremental",
                    "charge_efficiency": 0.98
                }
            }
        }
    }
}
```

### Key Configuration Options

- **`enabled_models`**: Comma-separated list of models to run
- **`compute_type`**: "incremental" or "full" processing mode
- **`debug_mode`**: Enable debug dashboard generation
- **Sensor mappings**: Map logical names to blueprint sensor names

## Creating New Models

1. **Create a new directory** under `models/`
   ```bash
   mkdir models/your_model
   ```

2. **Implement the model** in `your_model.py`:
   ```python
   from altergo_interface import Model, register_model
   
   @register_model("your_model")
   class YourModel(Model):
       def process(self, data):
           # Your battery science logic here
           return {"output_name": result_series}
   ```

3. **Create a manifest** in `model.json`:
   ```json
   {
       "name": "Your Model",
       "version": "1.0.0",
       "inputs": [
           {"logical_name": "current", "unit": "A", "required": true}
       ],
       "outputs": [
           {"logical_name": "output_name", "unit": "unit"}
       ]
   }
   ```

4. **Add to __init__.py**:
   ```python
   from .your_model import YourModel
   ```

## Deployment

1. **Push to repository**
   ```bash
   git add .
   git commit -m "Add new model"
   git push
   ```

2. **Configure in Altergo Platform**
   - Create a new function in the Altergo platform
   - Point it to your repository
   - Configure authentication if needed

3. **Deploy and Run**
   - The platform will automatically pull and execute your models
   - Monitor results through the Altergo dashboard

## Debugging

### Debug Dashboard
Enable debug mode for any model to generate interactive HTML dashboards:

```json
{
    "models": {
        "eq_cycles": {
            "debug_mode": true
        }
    }
}
```

This creates visualizations of:
- Input sensor data quality
- Parameter values
- Model outputs
- Data gaps and issues

### Logging
The framework provides clean console output with:
- Essential status messages
- Clear error reporting
- No excessive logging noise

## Best Practices

1. **Model Development**
   - Keep models focused on battery science
   - Use the `process()` method for all calculations
   - Let the framework handle platform interactions

2. **Data Handling**
   - Models receive data with logical names
   - The framework handles all mapping and conversion
   - Use pandas Series for time series data

3. **Configuration**
   - Use sensible defaults in model.json
   - Allow configuration overrides through altergo-settings.json
   - Document all configuration options

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your License Here]

## Support

For questions or issues:
- Check the documentation in the `documentation/` folder
- Open an issue in the repository
- Contact the Altergo support team
