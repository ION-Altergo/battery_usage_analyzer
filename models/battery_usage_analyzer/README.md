# Battery Usage Analyzer Model

Multi-layer battery usage timeline segmentation model that analyzes battery usage patterns through multiple layers:

- **Layer 0**: Operating modes (charge/discharge/idle)
- **Layer 1**: Data-driven change points
- **Layer 2**: Domain-specific phases (rest, CC/CV charge, discharge)
- **Layer 3**: Optional rainflow cycle segmentation on SoC

## File Structure

- `battery_usage_analyzer.py` - Main model implementation
- `utils.py` - Utility functions for data processing and signal analysis
- `test_battery_usage_analyzer.py` - Comprehensive test suite
- `model.json` - Model metadata and configuration schema
- `__init__.py` - Package initialization
- `run_tests.py` - Standalone test runner
- `README.md` - This documentation

## Usage

### Import the model
```python
from models.battery_usage_analyzer import BatteryUsageAnalyzer

# Create model instance
model = BatteryUsageAnalyzer({})

# Process data
result = model.process(data)
```

### Required Inputs
- `current`: Battery current time series [A]
- `soc`: State of charge [0-1 or %]
- `v_cell_min`: Minimum cell voltage [V]
- `v_cell_max`: Maximum cell voltage [V]
- `t_cell_min`: Minimum cell temperature [°C]
- `t_cell_max`: Maximum cell temperature [°C]

### Outputs
- `seg_l0_id`: Operating mode segment IDs
- `mode`: Operating mode labels
- `boundary_l1`: Data-driven boundary markers
- `seg_l1_id`: Data-driven segment IDs
- `phase`: Domain phase labels
- `seg_l2_id`: Domain phase segment IDs

## Testing

Run tests using any of these methods:

```bash
# Run from the model directory
cd models/battery_usage_analyzer
python run_tests.py

# Or run test file directly
python test_battery_usage_analyzer.py

# Or run main model file (includes tests)
python battery_usage_analyzer.py
```

## Configuration

The model supports 16 configuration parameters for fine-tuning behavior. See `model.json` for complete parameter descriptions and default values.

Key parameters:
- `window_s`: Smoothing window (default: 5.0s)
- `i_rest_th_A`: Rest current threshold
- `enable_rainflow`: Enable SoC rainflow segmentation (default: false)
- Various weights for change score computation
