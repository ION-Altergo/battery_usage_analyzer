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
