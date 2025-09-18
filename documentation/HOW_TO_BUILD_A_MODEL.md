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
  "name": "Your Model Name",
  "version": "1.0.0",
  "description": "Model description",
  "category": "Performance|Safety|Health|Other",
  "inputs": [
    {
      "logical_name": "current",
      "unit": "A",
      "description": "Battery current",
      "required": true
    }
  ],
  "outputs": [
    {
      "logical_name": "output_name",
      "unit": "unit",
      "description": "Output description"
    }
  ],
  "parameters": [
    {
      "logical_name": "capacity",
      "unit": "Ah",
      "description": "Battery capacity",
      "required": true
    }
  ],
  "configuration": {
    "param_name": {
      "description": "Configuration parameter",
      "type": "number|string|boolean",
      "default": 0
    }
  }
}
```

### 5. __init__.py
Export your model class:

```python
from .your_model_name import YourModelName
__all__ = ['YourModelName']
```

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
