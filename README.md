# VectorBT Tutorial

This repository contains a comprehensive tutorial for using VectorBT, a Python library for backtesting and trading strategy analysis.

## Description

The project includes examples of:

- Financial data download using Yahoo Finance
- Technical indicator calculations (RSI, Moving Averages)
- Trading strategy implementation
- Backtesting and performance analysis
- Results visualization

## System Requirements

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd vector-bt
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**On macOS/Linux:**

```bash
source venv/bin/activate
```

**On Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Main Dependencies

The main libraries used in this project are:

- **vectorbt (0.27.3)**: Main library for backtesting
- **pandas (2.2.3)**: Data manipulation
- **numpy (2.2.6)**: Numerical calculations
- **matplotlib (3.10.3)**: Chart visualization
- **plotly (5.24.1)**: Interactive charts
- **yfinance (0.2.61)**: Financial data download
- **ta-lib (0.6.3)**: Technical indicators
- **numba (0.61.2)**: JIT compilation for optimization

## Usage

Once dependencies are installed, you can run the tutorial:

```bash
python tutorial.py
```

## Project Structure

```
vector-bt/
├── tutorial.py          # Main script with examples
├── requirements.txt     # Project dependencies
├── README.md           # This file
├── venv/              # Virtual environment (not included in git)
└── .gitignore         # Files ignored by git
```

## Tutorial Features

The `tutorial.py` file includes examples of:

1. **Data download**: Getting BTC and ETH prices from Yahoo Finance
2. **Technical indicators**: RSI and Moving Averages calculation
3. **Trading signals**: Buy and sell signal generation
4. **Backtesting**: Trading strategy simulation
5. **Visualization**: Price, indicator, and signal charts
6. **Optimization**: Testing different parameters

## Important Notes

- Make sure you have an active internet connection to download financial data
- Data is downloaded in real-time, so results may vary
- The tutorial includes both long and short strategies

## Troubleshooting

If you encounter problems during installation:

1. **ta-lib error**: On some systems, ta-lib requires manual installation:

   ```bash
   # macOS with Homebrew
   brew install ta-lib

   # Ubuntu/Debian
   sudo apt-get install libta-lib-dev
   ```

2. **numba issues**: Make sure you have a compatible version of LLVM installed

3. **Memory errors**: Some calculations may require significant RAM, especially with large datasets
