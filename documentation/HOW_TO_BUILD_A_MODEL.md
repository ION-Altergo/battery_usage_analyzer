# How to Build a Model

This guide explains how to create a new battery model using the Altergo Digital Twin framework.

## Quick Start

### 1. Folder Structure
Create a new folder under `models/` with your model name:
```
models/
└── your_model_name/
    ├── __init__.py
    ├── your_model_name.py
    └── model.json
```

### 2. Key Imports
```python
from typing import Dict, Union
import pandas as pd
import numpy as np
from altergo_interface import Model, register_model
```

### 3. Model Class
Extend the `Model` base class and implement the `process()` method:

```python
@register_model("your_model_name")
class YourModelName(Model):
    """Your model description"""
    
    def process(self, data: Dict[str, Union[pd.Series, float]]) -> Dict[str, pd.Series]:
        """
        Core processing logic.
        
        Args:
            data: Dictionary with input sensor data and parameters
                  - Time series data: pd.Series with datetime index
                  - Parameters: float values
                  
        Returns:
            Dictionary with output time series (logical names as keys)
        """
        # Access inputs
        current = data["current"]  # pd.Series
        capacity = data["capacity"]  # float
        
        # Access configuration
        param = self.config.get("param_name", default_value)
        
        # Process data
        result = your_calculation_logic(current, capacity)
        
        # Return outputs with logical names
        return {
            "output_name": result  # pd.Series
        }
```

### 4. Model Manifest (model.json)
Define model metadata, inputs, outputs, and configuration:

```json
{
  "name": "Equivalent Cycles Estimator",
  "version": "2.1.0",
  "description": "Enhanced equivalent cycles estimation with efficiency corrections and SOH compensation. Backward compatible.",
  "category": "Performance",
  "complexity": "Simple",
  "computational_cost": "Low",
  "inputs": {
    "current":{
      "unit": "A",
      "type": "pd.Series",
      "description": "Battery current measurement",
      "required": true
    },
    "soh":{
      "unit": "%",
      "type": "pd.Series",
      "description": "State of Health (optional) - used for capacity compensation",
      "required": false
    },
    "capacity":{
      "unit": "Ah",
      "type": "float",
      "description": "Battery nominal capacity parameter",
      "required": true
    },
    "initial_equivalent_cycles":{
      "unit": "cycles",
      "type": "float",
      "initialisation_value": "equivalent_cycles",
      "description": "Initial equivalent cycle count",
      "required": false
    },
    "OCV_table":{
      "unit": "V",
      "type": "dict",
      "description": "OCV table",
      "required": false
    }
  },
  "outputs": {
    "equivalent_cycles":{
      "unit": "cycles",
      "type": "pd.Series",
      "description": "Calculated equivalent charge/discharge cycles",
      "decimation_threshold": 0.01
    }
  },
  
  "configuration": {
    "charge_efficiency": {
      "description": "Coulombic efficiency for charging (0.95-0.99)",
      "type": "number",
      "default": 0.98
    },
    "discharge_efficiency": {
      "description": "Coulombic efficiency for discharging (0.95-0.99)",
      "type": "number",
      "default": 0.99
    },
    "enable_efficiency_correction": {
      "description": "Enable coulombic efficiency correction",
      "type": "boolean",
      "default": true
    }
  }
}

```

### 5. model/__init__.py
Export your model class:

```python
from .your_model_name import YourModelName
__all__ = ['YourModelName']
```

### 6. models/__init__.py
add the model to the list of registered models
```python
"""
Models Package

Contains all model implementations organized by functionality.
Each model uses JSON manifests for metadata and simplified implementation.
"""

# Import all models to trigger registration decorators
from .eq_cycles.eq_cycles_model import EqCyclesModel


__all__ = [
    'EqCyclesModel'
]

```

### 7. altergo-settings.json
enable the model & add the necessary key / value pair for the configuration as needed (if you don't specify the conf parameters, default values from the manifest will be used. Required inputs must be provided are required.)
## Key Concepts

- **Logical Names**: Use consistent names for inputs/outputs (e.g., "current", "voltage", "soh")
- **Time Series**: All sensor data comes as `pd.Series` with datetime index
- **Parameters**: Static values (like battery capacity) come as floats
- **Configuration**: Model-specific settings accessed via `self.config`
- **Incremental Mode**: For stateful models, use `self.config.get("initial_value_name", default)`

## Best Practices

1. **Error Handling**: Wrap processing in try-except with descriptive errors
2. **Performance**: Vectorize operations using numpy/pandas
3. **Documentation**: Include docstrings and inline comments

## Example Reference
See `models/eq_cycles/` for a complete working example that demonstrates:
- Handling required and optional inputs
- Configuration parameters
- Efficiency corrections
- SOH compensation
- Incremental mode support
- Comprehensive self-testing
