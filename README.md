# At Risk Transformation

A forecasting package that implements the methodology from the paper "At Risk Transformation for U.S. Recession Prediction".

## Project Structure

```text
at_risk/                # Core Package
  ├── forecasting.py    # Recursive OOS Loop
  ├── models.py         # ModelRegistry & Templates
  ├── evaluation.py     # Metrics
  ├── data.py           # Data Prep
  ├── config.py         # Configuration
  └── cli.py            # Command Line Interface
data/                   # Data Directory
  ├── raw/              # Original CSVs
  └── processed/        # Pickle files
results/                # Outputs
  ├── predictions/      # Forecast .pkl files
  └── figures/          # Exported Plots
notebooks/              # Visualization
  └── Analysis.ipynb    # Figures and tables for Publication
```

## Quick Start

### 1. Installation
Install the package in editable mode:
```bash
pip install -e .
```

### 2. Run an Experiment
Use the CLI to generate OOS forecasts:
```bash
python -m at_risk --horizons 3 6 12 --specific-sets Deter_States
```

### 3. Generate Figures
Open `notebooks/Analysis.ipynb` to view performance tables and generate plots.
