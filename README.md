# Altergo Model Boilerplate - Setup Guide

This guide explains how to clone and use the Altergo Model Boilerplate to develop battery digital twin models for the Altergo Platform.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed on your system
- **Git** for version control
- **Altergo Platform Access** with valid API credentials
- **Asset ID** from your Altergo digital twin setup

## 1. Clone the Repository

```bash
git clone <repository-url>
cd model_boilerplate
```

Replace `<repository-url>` with the actual repository URL provided by your team.

## 2. Install Dependencies

The boilerplate uses the Altergo SDK and several scientific computing libraries:

```bash
pip install -r requirements.txt
```

This will install:
- `altergo-sdk` (from Altergo's private repository)
- `numpy` >= 1.20.0 (numerical computing)
- `scipy` >= 1.7.0 (scientific computing)
- `pandas` >= 1.3.0 (data manipulation)
- `plotly` >= 5.0.0 (visualizations for debug mode)

## 3. Configure Development Environment

### Create Dev Parameters File

Copy the template configuration file:

```bash
cp template.dev-parameters.json dev-parameters.json
```

Edit `dev-parameters.json` with your specific Altergo credentials:

```json
{
    "altergoUserApiKey": "YOUR_API_KEY_HERE",
    "altergoFactoryApi": "https://YOUR_COMPANY.altergo.io",
    "altergoIotApi": "https://iot.YOUR_COMPANY.altergo.io",
    "assetId": "YOUR_ASSET_ID_HERE"
}
```

**Important:** 
- Replace `YOUR_API_KEY_HERE` with your actual Altergo API key
- Replace `YOUR_COMPANY` with your company's Altergo subdomain
- Replace `YOUR_ASSET_ID_HERE` with the ID of the digital twin asset you want to analyze

### Configure Model Settings

The main model configuration is in `altergo-settings.json`. Key settings include:

```json
{
    "parameters": {
        "execution": {
            "enabled_models": "eq_cycles,adv_eq_cycles",
            "compute_type": "manual",
            "max_days_period_compute": 1,
            "debug_mode": true,
            "upload_output": false
        },
        "models": {
            "eq_cycles": {
                "inputs": {
                    "current": {"default": "Current"},
                    "capacity": {"default": "Capacity"}
                },
                "configuration": {
                    "charge_efficiency": 0.98,
                    "discharge_efficiency": 0.99
                }
            }
        }
    }
}
```

**Key Configuration Options:**

- **`enabled_models`**: Comma-separated list of models to run (e.g., "eq_cycles", "adv_eq_cycles")
- **`compute_type`**: 
  - `"manual"` - Process specific date range (set manual_start_date/manual_end_date)
  - `"incremental"` - Process new data since last run
  - `"full"` - Reprocess all available data
- **`debug_mode`**: Set to `true` to generate HTML debug dashboards
- **`upload_output`**: Set to `false` for local testing, `true` to upload results to platform

## 4. Test Your Setup

### Run Local Test

Execute the models locally with your development configuration:

```bash
python entrypoint.py
```

The system will:
1. Load configuration from `altergo-settings.json`
2. Connect to Altergo APIs using credentials from `dev-parameters.json`
3. Fetch data for the configured time period
4. Execute enabled models
5. Generate debug outputs (if enabled)
6. Optionally upload results to the platform

### Verify Output

If successful, you should see output similar to:

```
Starting Model Execution Framework
Loading configuration from altergo-settings.json
Connecting to Altergo APIs...
Fetching data for asset: 65e7487cad25e34679d71b66
Processing time range: 2023-10-16T00:18:04+08:00 to 2023-10-16T18:59:59+00:00
Executing model: eq_cycles
Model eq_cycles completed successfully
Generated debug dashboard: debug_eq_cycles.html
Execution completed
```

### Debug Mode Output

When `debug_mode` is enabled, the framework generates interactive HTML dashboards for each model:

- **Input Data Quality**: Visualizes sensor data gaps and quality
- **Model Parameters**: Shows configuration values used
- **Model Outputs**: Interactive plots of calculated results
- **Data Statistics**: Summary statistics and validation checks

Open the generated HTML files in your browser to inspect the results.

## 5. Directory Structure

After setup, your project structure should look like:

```
model_boilerplate/
├── entrypoint.py              # Main execution script
├── requirements.txt           # Python dependencies
├── altergo-settings.json      # Model configuration
├── dev-parameters.json        # Local dev credentials (gitignored)
├── template.dev-parameters.json # Template for credentials
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── eq_cycles/            # Basic equivalent cycles model
│   │   ├── eq_cycles_model.py
│   │   ├── model.json
│   │   └── README.md
│   └── adv_eq_cycles/        # Advanced equivalent cycles
│       ├── adv_eq_cycles.py
│       ├── model.json
│       └── README.md
└── documentation/            # Project documentation
```

## 6. Common Issues and Troubleshooting

### Authentication Errors

If you see authentication errors:
- Verify your API key is correct and active
- Check that the factory and IoT API URLs match your company's setup
- Ensure your account has access to the specified asset ID

### Data Connection Issues

If models can't fetch data:
- Verify the asset ID exists and you have access
- Check that the specified date range contains data
- Ensure your network can reach the Altergo APIs

### Missing Dependencies

If you see import errors:
- Ensure all requirements are installed: `pip install -r requirements.txt`
- Verify you have access to the private Altergo SDK repository
- Check Python version compatibility (3.8+)

### Model Execution Errors

If models fail to execute:
- Check the logs for specific error messages
- Verify sensor mappings in `altergo-settings.json` match your blueprint
- Enable debug mode to see detailed data flow information

## 7. Next Steps

Once your setup is working:

1. **Explore Existing Models**: Study the `eq_cycles` and `adv_eq_cycles` implementations to understand the pattern
2. **Create Custom Models**: Follow the model creation guide to build your own battery analysis models
3. **Configure for Your Data**: Adjust sensor mappings and parameters for your specific battery setup
4. **Deploy to Production**: Configure for automatic execution on the Altergo platform

## 8. Development Workflow

For ongoing development:

1. **Local Testing**: Always test changes locally with `debug_mode: true`
2. **Version Control**: Commit your changes to git (excluding `dev-parameters.json`)
3. **Model Validation**: Verify outputs make sense using debug dashboards
4. **Platform Deployment**: Push changes and update platform configuration
5. **Monitoring**: Monitor execution logs and results on the Altergo platform

## Support

For questions or issues:
- Check the model creation guide for implementation details
- Review existing model examples in the `models/` directory
- Consult the documentation in the `documentation/` folder
- Contact your Altergo support team for platform-specific issues

---

**Security Note**: Never commit `dev-parameters.json` to version control as it contains sensitive API credentials. The file is already in `.gitignore`.
