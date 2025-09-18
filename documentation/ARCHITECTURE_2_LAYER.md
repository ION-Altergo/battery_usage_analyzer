# 2-Layer Architecture: Digital Twin Interface + Battery Science Models

## Overview

The codebase has been restructured from a 3-layer to a cleaner 2-layer architecture that better reflects the actual separation of concerns.

## New Architecture

### Layer 1: Digital Twin Interface (`altergo_interface/`)
**Purpose**: Unified interface between Altergo platform and battery science models

**Components**:
- `base_model.py` - Base class defining the model contract
- `models.py` - Simple model discovery and instantiation
- `config.py` - Configuration extraction and mapping
- `data.py` - Data loading and model execution
- `output.py` - Result uploading to platform

**Key Responsibilities**:
1. Handle all Altergo platform interactions
2. Manage configuration and sensor mappings
3. Load and prepare data for models
4. Execute models and upload results
5. Provide simple base class for models

### Layer 2: Battery Science Models (`models/`)
**Purpose**: Pure battery science implementations

**Structure**:
```
models/
├── eq_cycles/
│   ├── eq_cycles_model.py      # Model implementation
│   └── model.json              # Model metadata
├── voltage_monitor/
├── imbalance_analysis/
└── soc_ocv_estimation/
```

**Key Characteristics**:
- Import only from `altergo_interface`
- Focus purely on battery algorithms
- No platform-specific code
- Simple `process()` method implementation

## Benefits of 2-Layer Architecture

1. **Clearer Separation**
   - Digital Twin platform interface vs Battery science logic
   - No ambiguous "framework" layer

2. **Reduced Complexity**
   - Removed `ModelRegistry` complexity
   - Simple model discovery by directory scanning
   - Direct model instantiation

3. **Better Cohesion**
   - Related functionality grouped together
   - Platform concerns isolated in one layer

4. **Easier to Understand**
   - New developers immediately see the boundary
   - Clear import hierarchy: models import from interface

## Migration Summary

### What Changed:
1. `altergo_platform/` → `altergo_interface/` (renamed for clarity)
2. `framework/` merged into `altergo_interface/`
3. Complex `ModelRegistry` replaced with simple `discover_models()`
4. Models now import from `altergo_interface` instead of `framework`

### What Stayed the Same:
1. Model implementations unchanged
2. Configuration structure unchanged
3. Data flow unchanged
4. All functionality preserved

## Code Example

### Before (3 layers):
```python
# Model imports from framework
from framework.base_model import Model
from framework import register_model

# Entrypoint uses both framework and platform
from framework import ModelRegistry
from altergo_platform import load_sensor_data
```

### After (2 layers):
```python
# Model imports from interface
from altergo_interface import Model, register_model

# Entrypoint uses only interface
from altergo_interface import (
    discover_models, create_model, load_sensor_data
)
```

## Future Improvements

1. **Add Type Hints**: Improve type safety across the interface
2. **Standardize Logging**: Use Python's logging module
3. **Add Tests**: Unit tests for both layers
4. **Document API**: Clear documentation of the interface contract
