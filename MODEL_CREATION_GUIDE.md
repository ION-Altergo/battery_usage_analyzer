# Creating New Models - Developer Guide

This guide explains how to create new battery analysis models for the Altergo Platform using the model boilerplate framework.

## Overview

The Altergo model framework uses a clean 2-layer architecture:

1. **Platform Integration Layer**: Handled automatically by the Altergo SDK
2. **Battery Science Layer**: Your model implementations in the `models/` directory

Your models focus purely on battery science algorithms while the framework handles all platform interactions, data fetching, sensor mapping, and result uploading.

## Model Architecture

Each model consists of:

- **Python Implementation**: Core algorithm in a Python class
- **JSON Manifest**: Metadata defining inputs, outputs, and configuration
- **Registration**: Decorator-based registration with the framework

## Step-by-Step Guide

### Step 1: Create Model Directory

Create a new directory under `models/` for your model:

```bash
mkdir models/your_model_name
cd models/your_model_name
```

Use descriptive names like:
- `voltage_monitor` (for voltage monitoring)
- `cell_imbalance` (for imbalance analysis)
- `thermal_model` (for temperature modeling)

### Step 2: Create the Model Manifest (`model.json`)

The `model.json` file defines your model's interface. Here's a template:

```json
{
  "name": "Your Model Name",
  "version": "1.0.0",
  "description": "Description of what your model does",
  "category": "Performance",
  "complexity": "Simple",
  "computational_cost": "Low",
  "inputs": {
    "input_name": {
      "unit": "A",
      "type": "pd.Series",
      "description": "Description of this input",
      "required": true
    },
    "optional_input": {
      "unit": "V",
      "type": "float",
      "description": "Optional parameter",
      "required": false
    }
      "complex_input": {
      "unit": "V",
      "type": "Dict",
      "description": "Optional parameter",
      "required": false
    }
  },
  "outputs": {
    "output_name": {
      "unit": "cycles",
      "type": "pd.Series",
      "description": "Description of this output",
      "decimation_threshold": 0.01
    }
  },
  "configuration": {
    "param_name": {
      "description": "Configuration parameter description",
      "type": "number",
      "default": 1.0
    }
  }
}
```

**Key Fields:**

- **`category`**: "Performance", "Health", "Safety", "Efficiency"
- **`complexity`**: "Simple", "Advanced", "Expert"
- **`computational_cost`**: "Low", "Medium", "High"
- **`inputs`**: Data your model needs (time series or parameters)
- **`outputs`**: Results your model produces
- **`configuration`**: Tunable parameters with defaults

### Step 3: Implement the Model Class

Create your model implementation file (e.g., `your_model.py`):

```python
"""
Your Model Implementation

Description of what your model does and its key features.
"""

from typing import Dict, Union
import pandas as pd
import numpy as np

from altergo_sdk.boiler_plate import Model, register_model


@register_model("your_model_name", metadata={
    "category": "Performance",
    "complexity": "Simple", 
    "computational_cost": "Low"
})
class YourModel(Model):
    """
    Your Model Class
    
    Brief description of the model's purpose and methodology.
    
    Features:
    - Feature 1
    - Feature 2
    - Feature 3
    """

    def process(self, data: Dict[str, Union[pd.Series, float]]) -> Dict[str, pd.Series]:
        """
        Main processing method for your model.
        
        Args:
            data: Dictionary containing:
                - input_name: Description (required)
                - optional_input: Description (optional)
                
        Returns:
            Dictionary containing output_name time series
        """
        
        try:
            # Extract required inputs
            input_data = data["input_name"]
            
            # Extract optional inputs with defaults
            optional_param = data.get("optional_input", 0.0)
            
            # Get configuration parameters
            param_value = self.config.get("param_name", 1.0)
            
            # Validate inputs
            if input_data.empty:
                raise ValueError("Input data cannot be empty")
            
            # Your model logic here
            result = self._your_algorithm(input_data, optional_param, param_value)
            
            # Return results with logical names
            return {
                "output_name": result
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to process {self.__class__.__name__}: {str(e)}") from e
    
    def _your_algorithm(self, input_data: pd.Series, param1: float, param2: float) -> pd.Series:
        """
        Core algorithm implementation.
        
        Args:
            input_data: Primary input time series
            param1: First parameter
            param2: Second parameter
            
        Returns:
            Processed result as pandas Series
        """
        # Implement your battery science algorithm here
        # Example: simple moving average
        window_size = int(param1 * param2)
        result = input_data.rolling(window=window_size, min_periods=1).mean()
        
        return result
```

### Step 4: Register the Model

Add your model to the package's `__init__.py`:

```python
# In models/__init__.py

# Import all models to trigger registration decorators
from .eq_cycles.eq_cycles_model import EqCyclesModel
from .adv_eq_cycles.adv_eq_cycles import EnhancedEquivalentCycles
from .your_model_name.your_model import YourModel  # Add this line

__all__ = [
    'EqCyclesModel',
    'EnhancedEquivalentCycles',
    'YourModel'  # Add this line
]
```

### Step 5: Configure the Model

Add your model configuration to `altergo-settings.json`:

```json
{
    "parameters": {
        "execution": {
            "enabled_models": "eq_cycles,your_model_name"
        },
        "models": {
            "your_model_name": {
                "inputs": {
                    "input_name": {
                        "default": "Blueprint Sensor Name"
                    },
                    "optional_input": {
                        "default": "Optional Sensor Name"
                    }
                },
                "outputs": {
                    "output_name": {
                        "default": "Output Sensor Name"
                    }
                },
                "configuration": {
                    "param_name": 2.0
                }
            }
        }
    }
}
```

**Key Configuration Sections:**

- **`inputs`**: Maps logical names to blueprint sensor names
- **`outputs`**: Maps logical names to blueprint output sensor names  
- **`configuration`**: Overrides default parameter values

### Step 6: Test Your Model

Test your model locally:

```bash
python entrypoint.py
```

Enable debug mode to see detailed output:

```json
{
    "execution": {
        "debug_mode": true
    }
}
```

## Real Examples

### Example 1: Simple Voltage Monitor

**model.json:**
```json
{
  "name": "Voltage Monitor",
  "version": "1.0.0",
  "description": "Monitors cell voltages and detects out-of-range conditions",
  "category": "Safety",
  "inputs": {
    "cell_voltages": {
      "unit": "V",
      "type": "pd.Series",
      "description": "Individual cell voltages",
      "required": true
    }
  },
  "outputs": {
    "voltage_status": {
      "unit": "dimensionless",
      "type": "pd.Series",
      "description": "Status: 0=OK, 1=Low, 2=High"
    },
    "min_voltage": {
      "unit": "V", 
      "type": "pd.Series",
      "description": "Minimum cell voltage"
    },
    "max_voltage": {
      "unit": "V",
      "type": "pd.Series", 
      "description": "Maximum cell voltage"
    }
  },
  "configuration": {
    "voltage_low_threshold": {
      "description": "Low voltage threshold (V)",
      "type": "number",
      "default": 2.5
    },
    "voltage_high_threshold": {
      "description": "High voltage threshold (V)",
      "type": "number",
      "default": 3.6
    }
  }
}
```

**voltage_monitor.py:**
```python
@register_model("voltage_monitor", metadata={
    "category": "Safety",
    "complexity": "Simple",
    "computational_cost": "Low"
})
class VoltageMonitor(Model):
    """Voltage monitoring and alerting model."""

    def process(self, data: Dict[str, Union[pd.Series, float]]) -> Dict[str, pd.Series]:
        cell_voltages = data["cell_voltages"]
        
        # Get thresholds from configuration
        v_low = self.config.get("voltage_low_threshold", 2.5)
        v_high = self.config.get("voltage_high_threshold", 3.6)
        
        # Calculate min and max voltages
        min_voltage = cell_voltages.min(axis=1)
        max_voltage = cell_voltages.max(axis=1)
        
        # Determine status
        status = pd.Series(0, index=cell_voltages.index)  # 0 = OK
        status[min_voltage < v_low] = 1  # 1 = Low
        status[max_voltage > v_high] = 2  # 2 = High
        
        return {
            "voltage_status": status,
            "min_voltage": min_voltage,
            "max_voltage": max_voltage
        }
```

### Example 2: Advanced SOC Estimator

**model.json:**
```json
{
  "name": "SOC Estimator", 
  "version": "1.0.0",
  "description": "State of charge estimation using OCV lookup",
  "category": "Performance",
  "complexity": "Advanced",
  "inputs": {
    "voltage": {
      "unit": "V",
      "type": "pd.Series", 
      "description": "Pack voltage",
      "required": true
    },
    "current": {
      "unit": "A",
      "type": "pd.Series",
      "description": "Pack current", 
      "required": true
    },
    "ocv_table": {
      "unit": "V",
      "type": "dict",
      "description": "OCV lookup table",
      "required": true
    }
  },
  "outputs": {
    "soc": {
      "unit": "p.u.",
      "type": "pd.Series",
      "description": "State of charge (0-1)"
    },
    "ocv": {
      "unit": "V", 
      "type": "pd.Series",
      "description": "Estimated open circuit voltage"
    }
  },
  "configuration": {
    "rest_threshold": {
      "description": "Current threshold for rest detection (A)",
      "type": "number",
      "default": 0.1
    },
    "rest_duration": {
      "description": "Minimum rest duration (minutes)",
      "type": "number", 
      "default": 30
    }
  }
}
```

## Best Practices

### 1. Input Validation

Always validate your inputs:

```python
def process(self, data):
    # Check for required inputs
    if "required_input" not in data:
        raise ValueError("Missing required input: required_input")
    
    input_data = data["required_input"]
    
    # Validate data quality
    if input_data.empty:
        raise ValueError("Input data cannot be empty")
    
    if input_data.isna().all():
        raise ValueError("Input data contains only NaN values")
```

### 2. Configuration Management

Use sensible defaults and validate configuration:

```python
def process(self, data):
    # Get configuration with defaults
    threshold = self.config.get("threshold", 1.0)
    
    # Validate configuration
    if threshold <= 0:
        raise ValueError(f"Threshold must be positive, got {threshold}")
```

### 3. Error Handling

Provide clear error messages:

```python
try:
    result = complex_calculation(data)
except Exception as e:
    raise RuntimeError(f"Calculation failed in {self.__class__.__name__}: {str(e)}") from e
```

### 4. Time Series Handling

Ensure proper time series alignment:

```python
# Align different time series
aligned_data = input2.reindex(input1.index, method='nearest')

# Handle time gaps
time_diff = input1.index.to_series().diff().dt.total_seconds()
```

### 5. Performance Optimization

For large datasets:

```python
# Use numpy operations when possible
values = input_data.values  # Extract numpy array
result_values = np.some_operation(values)

# Create result series
result = pd.Series(result_values, index=input_data.index)
```

## Testing Your Models

### Local Testing

1. **Enable Debug Mode**: Set `debug_mode: true` in configuration
2. **Check Debug Output**: Review generated HTML dashboards
3. **Validate Results**: Ensure outputs make physical sense
4. **Performance Test**: Test with realistic data sizes

### Self-Test Framework

Add a self-test section to your model:

```python
if __name__ == "__main__":
    """Self-test routine"""
    # Generate test data
    test_data = generate_test_data()
    
    # Create model instance
    model = YourModel({"param": 1.0})
    
    # Test basic functionality
    result = model.process(test_data)
    
    # Validate results
    assert len(result["output"]) == len(test_data["input"])
    print("✓ Self-test passed")
```

## Common Patterns

### Pattern 1: Sliding Window Analysis

```python
def _sliding_window_analysis(self, data: pd.Series, window_minutes: int) -> pd.Series:
    """Apply sliding window analysis."""
    window_size = f"{window_minutes}T"  # T = minutes
    return data.rolling(window_size, min_periods=1).apply(your_function)
```

### Pattern 2: State Detection

```python
def _detect_state_changes(self, current: pd.Series, threshold: float) -> pd.Series:
    """Detect charge/discharge state changes."""
    states = pd.Series('idle', index=current.index)
    states[current > threshold] = 'discharge'
    states[current < -threshold] = 'charge'
    return states
```

### Pattern 3: Cumulative Calculations

```python
def _cumulative_calculation(self, data: pd.Series, initial_value: float = 0) -> pd.Series:
    """Calculate cumulative values with time weighting."""
    time_diff = data.index.to_series().diff().dt.total_seconds() / 3600  # hours
    time_diff.iloc[0] = 0
    
    incremental = data.values * time_diff.values
    cumulative = np.cumsum(incremental) + initial_value
    
    return pd.Series(cumulative, index=data.index)
```

## Deployment

### 1. Version Control

Commit your model to the repository:

```bash
git add models/your_model_name/
git commit -m "Add your_model_name model"
git push
```

### 2. Platform Configuration

Update your Altergo platform function to:
- Point to your repository
- Include your model in `enabled_models`
- Configure appropriate sensor mappings

### 3. Production Testing

- Start with small time ranges
- Monitor execution logs
- Validate outputs against expectations
- Gradually increase to full production datasets

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure model is registered in `__init__.py`
2. **Missing Data**: Check sensor mappings in configuration
3. **Performance Issues**: Profile code and optimize bottlenecks
4. **Unexpected Results**: Enable debug mode and examine intermediate values

### Debugging Tips

- Use `print()` statements for debugging (they appear in platform logs)
- Generate intermediate outputs for validation
- Test with synthetic data first
- Compare results with manual calculations

---

This guide provides the foundation for creating robust, maintainable battery analysis models on the Altergo platform. Start with simple models and gradually add complexity as you become familiar with the framework.
