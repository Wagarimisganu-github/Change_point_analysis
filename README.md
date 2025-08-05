## Brent Crude Oil Price Analysis: Bayesian Change Point Detection
https://github.com/zekikobe/brent-oil-analysis/workflows/CI/badge.svg
https://codecov.io/gh/zekikobe/brent-oil-analysis/branch/main/graph/badge.svg
https://img.shields.io/badge/python-3.9+-blue.svg

A comprehensive analytical framework for detecting structural breaks in Brent crude oil prices using Bayesian statistical methods. This project helps identify significant market shifts caused by geopolitical events, economic policies, and supply-demand changes.

## Table of Contents
-Features

-Installation

-Usage

-Project Structure

-Methodology

-Results

-Contributing


## Features
Advanced Analytics
Bayesian Change Point Detection: Identifies structural breaks using PyMC's probabilistic programming

Event Correlation Analysis: Maps detected changes to real-world geopolitical and economic events

Volatility Modeling: Analyzes price fluctuations across different market regimes

## Professional Workflow
Automated Data Pipeline: From raw data ingestion to processed outputs

Reproducible Research: Jupyter notebooks for exploratory analysis

CI/CD Integration: GitHub Actions for automated testing and validation

Visualization Tools
Interactive Dashboard: Plotly Dash interface for exploring results

Publication-Quality Plots: Matplotlib/Seaborn visualizations

Automatic Report Generation: PDF and HTML output options

## Installation
Prerequisites
Python 3.9 or higher

pip package manager

Setup
bash
# Clone the repository
git clone https://github.com/yourusername/brent-oil-analysis.git
cd brent-oil-analysis

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
For development:

bash
pip install -r requirements-dev.txt
Usage
Data Processing Pipeline
bash
# Run the complete analysis pipeline
python scripts/run_pipeline.py

# Or execute individual steps:
python scripts/data_processing.py   # Clean and prepare data
python scripts/modeling.py          # Run Bayesian analysis
python scripts/visualization.py     # Generate figures
Interactive Exploration
bash
# Launch Jupyter notebook server
jupyter notebook

# Start the Dash dashboard
python app.py
## Project Structure
text
brent-oil-analysis/
├── .github/              # CI/CD workflows
├── data/                 # Data storage
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned data
├── notebooks/            # Research notebooks
│   ├── 01_eda.ipynb      # Exploratory analysis
│   └── 02_model_testing.ipynb
├── scripts/              # Production code
│   ├── data_processing.py
│   ├── modeling.py
│   └── visualization.py
├── outputs/              # Generated artifacts
│   ├── models/           # Saved model states
│   ├── figures/          # Visualizations
│   └── reports/          # Analysis outputs
├── tests/                # Unit tests
├── app.py                # Dashboard
└── docs/                 # Documentation
## Methodology
Bayesian Change Point Detection
Our approach uses Markov Chain Monte Carlo (MCMC) sampling to identify:

Number of structural breaks

Breakpoint locations

Posterior distributions of regime parameters

## Statistical Models
Gaussian Process Model: For smooth transitions between regimes

Hierarchical Model: Captures uncertainty across multiple change points

Volatility Clustering: GARCH components for heteroskedasticity

## Validation
Posterior predictive checks

Convergence diagnostics (R-hat, effective sample size)

Cross-validation on holdout periods

## Results
Key Findings
Major Change Points detected during:

2008 Financial Crisis

2014 OPEC Price War

2020 COVID-19 Pandemic

## Volatility Regimes:

High volatility during geopolitical crises

Lower volatility during stable market periods

## Event Impact:

OPEC decisions show strongest correlation with breaks

Geopolitical events cause short-term spikes

## Sample Output
https://outputs/figures/change_points.png

## Contributing
We welcome contributions! Please follow these steps:

## Fork the repository

Create a feature branch (git checkout -b feature/your-feature)

Commit your changes (git commit -am 'Add some feature')

Push to the branch (git push origin feature/your-feature)

Open a Pull Request

## Development Setup
bash
pip install -r requirements-dev.txt
pre-commit install

secrenshoot of the app 
<img width="2158" height="1183" alt="image" src="https://github.com/user-attachments/assets/3b03b60c-0899-47da-b4d3-99f697b034db" />
