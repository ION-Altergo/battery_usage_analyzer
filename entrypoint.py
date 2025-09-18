"""
Model Execution Entrypoint

Main entry point for executing battery digital twin models.
"""

import sys

# Altergo SDK
from altergo_sdk.tools.utils import extract_altergo_parameters

# Boilerplate execution logic
from altergo_sdk.boiler_plate.boiler_plate import execute_altergo_models

# Import models to trigger registration
import models


def main():
    """Main execution function."""
    print("Starting Model Execution Framework")
    
    # Extract Altergo parameters from altergo-settings.json and return it into altergo_arguments
    altergo_arguments = extract_altergo_parameters()
    
    # Execute the main logic using boilerplate
    execute_altergo_models(altergo_arguments)


if __name__ == "__main__":
    main()