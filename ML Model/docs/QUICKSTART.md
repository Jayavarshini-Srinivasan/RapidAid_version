# Quick Start Guide

## Setup

1. **Create and activate virtual environment** (if not already done):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Option 1: Run with your data file
```bash
source venv/bin/activate  # Activate virtual environment first
python main.py india_traffic_accidents.csv
```

### Option 2: Run without specifying file (auto-detects CSV/Excel files)
```bash
source venv/bin/activate
python main.py
```

## Output Files

After successful execution, you'll find:

- **`models/accident_severity_model.pkl`** - Trained Random Forest model
- **`output/predictions.json`** - Predictions with coordinates for map visualization
- **`output/summary_stats.json`** - Model performance metrics and statistics

## Note

Make sure to activate the virtual environment before running the script:
```bash
source venv/bin/activate
```

