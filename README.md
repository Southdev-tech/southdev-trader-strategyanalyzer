# Automated Trading Backtesting Module

This repository contains an automated backtesting system using VectorBT for strategy evaluation and optimization. The module is designed to run as a scheduled job on Railway, evaluating RSI and VWAP-based trading strategies across multiple tickers and selecting the best-performing parameters for each.

## Description

The backtesting module provides:

- **Automated Strategy Evaluation**: Grid search optimization for RSI and VWAP strategies
- **Multi-Asset Support**: Concurrent backtesting across multiple tickers
- **Data Integration**: Seamless integration with Alpaca Markets API and PostgreSQL
- **Performance Optimization**: High-speed vectorized backtesting using VectorBT
- **Extensible Framework**: Easy addition of new trading strategies

## Architecture Overview

The system operates as a batch job that:

1. Fetches historical price data from Alpaca API
2. Runs grid search optimization on predefined strategies
3. Evaluates performance metrics (Sharpe ratio, max drawdown, returns)
4. Stores optimal strategy parameters in PostgreSQL database
5. Provides results for the live trading system

## System Requirements

- Python 3.9 or higher
- PostgreSQL database
- Alpaca Markets API credentials
- Railway account (for deployment)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd backtesting-module
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

### 5. Configure environment variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required environment variables:

- `ALPACA_API_KEY`: Your Alpaca API key
- `ALPACA_SECRET_KEY`: Your Alpaca secret key

## Main Dependencies

- **vectorbt-pro**: Advanced backtesting and portfolio analysis
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **sqlalchemy**: Database ORM
- **psycopg2**: PostgreSQL adapter
- **alpaca-py**: Alpaca Markets API client
- **python-dotenv**: Environment variable management
- **schedule**: Task scheduling (for local development)

## Project Structure

```
backtesting_module/
│
├── requirements.txt
├── .env.example
├── main.py
│
├── data/
│   ├── alpaca_client.py
│   ├── database_client.py
│
├── strategies/
│   ├── rsi_strategy.py
│   ├── vwap_strategy.py
│
├── database/
│   ├── models.py
│
├── utils/
│   ├── logger.py
│   └── helpers.py
```

## Usage

### Local Development

Run the backtesting process locally:

```bash
python main.py
```

Or run specific strategies to try locally, this will run it as modules

````bash
python -m strategies.vwap_implementation
```

### Strategy Configuration

Modify strategy parameters in `config/strategies_config.py`:

```python
RSI_STRATEGIES = {
    'rsi_windows': [14, 21, 28],
    'entry_thresholds': [20, 25, 30],
    'exit_thresholds': [70, 75, 80],
    'lookback_period': 252  # trading days
}

VWAP_STRATEGIES = {
    'reversion_threshold': [0.005, 0.01, 0.015],  # % deviation from VWAP
    'momentum_confirmation': [0.002, 0.005],
    'intraday_data_period': 30  # days of intraday data
}
````

## Strategy Types

### RSI

- **Entry**: RSI falls below oversold threshold
- **Exit**: RSI rises above overbought threshold
- **Parameters**: RSI period, entry/exit thresholds
- **Timeframe**: Daily data

### VWAP Strategies

- **Mean Reversion**: Buy below VWAP, sell above VWAP
- **Momentum**: Buy on upward VWAP cross, sell on downward cross
- **Parameters**: Deviation thresholds, confirmation levels
- **Timeframe**: Intraday data (1-minute bars)

## Performance Metrics

The system evaluates strategies using:

- **Total Return**: Cumulative strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return-to-drawdown ratio
- **Win Rate**: Percentage of profitable trades
- **Sortino Ratio**: Downside deviation-adjusted returns

## Monitoring and Logging

- **Structured Logging**: JSON-formatted logs for easy parsing
- **Performance Tracking**: Execution time and resource usage metrics
- **Error Handling**: Comprehensive error catching and reporting
- **Health Checks**: Database connectivity and API status validation

## License

This project is part of an automated trading system. Please ensure compliance with your broker's API terms of service and applicable financial regulations.
