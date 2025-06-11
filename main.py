import os
import sys
from dotenv import load_dotenv

# Load environment variables FIRST (including NUMBA_DISABLE_JIT=1) before importing VectorBT
load_dotenv()

import vectorbtpro as vbt
import datetime
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from data.alpaca_client import download_stock_data_both
from strategies.rsi_implementation import (
    test_rsi_strategy_on_stock,
    optimize_rsi_for_stock,
)
from strategies.vwap_implementation import (
    test_vwap_strategy_on_stock,
    optimize_vwap_for_stock,
)

warnings.filterwarnings("ignore")

vbt.AlpacaData.set_custom_settings(
    client_config=dict(
        api_key=os.getenv("ALPACA_API_KEY"), secret_key=os.getenv("ALPACA_SECRET_KEY")
    )
)


def get_timeframe_slice(df, days):
    if len(df) < days:
        return df
    return df.iloc[-days:]


def multi_timeframe_backtest(
    strategy_func, data, symbol, param_grid, is_vwap=False, timeframes=None
):
    results = []
    valid_results = 0
    total_attempts = 0

    for params in param_grid:
        metrics_per_tf = {}
        for tf_name, tf_days in timeframes.items():
            total_attempts += 1

            if is_vwap:
                df = data.get()
                # Get the most recent date and calculate start date for timeframe
                end_date = df.index.max()
                start_date = end_date - pd.Timedelta(days=tf_days)
                # Filter data for this timeframe
                sliced_df = df[df.index >= start_date]

                class Wrapper:
                    def __init__(self, df):
                        self.df = df

                    def get(self, col=None):
                        if col is None:
                            return self.df
                        return self.df[col]

                sliced_data = Wrapper(sliced_df)
            else:
                # Get the most recent date and calculate start date for timeframe
                end_date = data.index.max()
                start_date = end_date - pd.Timedelta(days=tf_days)
                # Filter data for this timeframe
                sliced_data = data[data.index >= start_date]

            # Skip if we don't have enough data points for analysis
            min_points_needed = 50 if is_vwap else 30
            actual_points = len(sliced_data.get()) if is_vwap else len(sliced_data)

            if actual_points < min_points_needed:
                print(
                    f"        ⚠️  {tf_name}: Only {actual_points} points, need {min_points_needed} minimum"
                )
                metrics_per_tf[tf_name] = {
                    "total_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "calmar_ratio": 0,
                    "win_rate": 0,
                    "sortino_ratio": 0,
                }
                continue

            try:
                if is_vwap:
                    res = strategy_func(sliced_data, symbol, **params)
                else:
                    res = strategy_func(
                        sliced_data, symbol, show_warnings=False, **params
                    )
            except Exception as e:
                res = None

            if res and "portfolio" in res:
                valid_results += 1
                try:
                    stats = res["portfolio"].stats()
                    # Handle different stat formats - convert to dict if needed
                    if hasattr(stats, "to_dict"):
                        stats_dict = stats.to_dict()
                    elif isinstance(stats, dict):
                        stats_dict = stats
                    else:
                        # Fallback if stats are not in expected format
                        stats_dict = {}

                    metrics_per_tf[tf_name] = {
                        "total_return": stats_dict.get("Total Return [%]", 0) / 100,
                        "sharpe_ratio": stats_dict.get("Sharpe Ratio", 0),
                        "max_drawdown": stats_dict.get("Max Drawdown [%]", 0) / 100,
                        "calmar_ratio": stats_dict.get("Calmar Ratio", 0),
                        "win_rate": stats_dict.get("Win Rate [%]", 0) / 100,
                        "sortino_ratio": stats_dict.get("Sortino Ratio", 0),
                    }
                except Exception as e:
                    # Fallback to zero metrics if stats fail
                    metrics_per_tf[tf_name] = {
                        "total_return": 0,
                        "sharpe_ratio": 0,
                        "max_drawdown": 0,
                        "calmar_ratio": 0,
                        "win_rate": 0,
                        "sortino_ratio": 0,
                    }
            else:
                metrics_per_tf[tf_name] = {
                    "total_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "calmar_ratio": 0,
                    "win_rate": 0,
                    "sortino_ratio": 0,
                }
        avg_metrics = {
            k: np.mean([v[k] for v in metrics_per_tf.values()])
            for k in [
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "calmar_ratio",
                "win_rate",
                "sortino_ratio",
            ]
        }
        results.append(
            {"params": params, "metrics": metrics_per_tf, "avg_metrics": avg_metrics}
        )

    print(
        f"      📊 Multi-timeframe: {valid_results}/{total_attempts} successful strategy tests"
    )
    return results


def get_rsi_param_grid():
    rsi_windows = np.arange(10, 25, 2)
    entry_levels = np.arange(20, 40, 5)
    exit_levels = np.arange(60, 85, 5)
    grid = []
    for rsi_window in rsi_windows:
        for entry_level in entry_levels:
            for exit_level in exit_levels:
                if entry_level < exit_level:
                    grid.append(
                        {
                            "rsi_window": rsi_window,
                            "entry_level": entry_level,
                            "exit_level": exit_level,
                        }
                    )
    return grid


def get_vwap_param_grid(symbol):
    if symbol == "SPY":
        entry_thresholds = np.arange(0.0005, 0.0025, 0.0003)
        exit_thresholds = np.arange(0.0002, 0.0018, 0.0003)
    else:
        entry_thresholds = np.arange(0.002, 0.014, 0.0005)
        exit_thresholds = np.arange(0.001, 0.009, 0.0005)
    grid = []
    for entry_threshold in entry_thresholds:
        for exit_threshold in exit_thresholds:
            if entry_threshold > exit_threshold:
                grid.append(
                    {
                        "entry_threshold": entry_threshold,
                        "exit_threshold": exit_threshold,
                    }
                )
    return grid


def download_symbol_data(symbol, start_date, end_date):
    """Downloads data for a specific symbol - SUPER OPTIMIZED"""
    print(f"   🔄 Downloading {symbol}...")

    try:
        rsi_data_single, vwap_data_single = download_stock_data_both(
            symbol, start_date, end_date
        )

        if rsi_data_single is not None:
            print(f"   ✅ RSI: {len(rsi_data_single)} data points")
        else:
            print(f"   ❌ Failed to get RSI data for {symbol}")

        if vwap_data_single is not None:
            print(f"   ✅ VWAP: {len(vwap_data_single.get('Close'))} data points")
        else:
            print(f"   ❌ Failed to get VWAP data for {symbol}")

        return symbol, rsi_data_single, vwap_data_single

    except Exception as e:
        print(f"   ❌ Error downloading {symbol}: {e}")
        return symbol, None, None


def analyze_symbol_individual(symbol, rsi_data, vwap_data):
    """Analyzes a specific symbol in Phase 1"""
    if symbol not in rsi_data or symbol not in vwap_data:
        return None

    print(f"   🔍 Analyzing {symbol}...")

    try:
        # Unpack the tuples returned by optimization functions
        best_rsi_result, rsi_all_results = optimize_rsi_for_stock(
            rsi_data[symbol], symbol
        )

        best_vwap_result, vwap_all_results = optimize_vwap_for_stock(
            vwap_data[symbol], symbol
        )

        rsi_return = best_rsi_result["total_return"] if best_rsi_result else 0
        vwap_return = best_vwap_result["total_return"] if best_vwap_result else 0

        winner = "RSI" if rsi_return > vwap_return else "VWAP"

        return {
            "symbol": symbol,
            "rsi": best_rsi_result,
            "vwap": best_vwap_result,
            "winner": winner,
        }

    except Exception as e:
        print(f"   ❌ Error analyzing {symbol}: {e}")
        return None


def analyze_symbol_multiframe(symbol, rsi_data_dict, vwap_data_dict, timeframes):
    """Analyzes a specific symbol with multi-timeframe in Phase 2"""
    if symbol not in rsi_data_dict or symbol not in vwap_data_dict:
        return None

    print(f"   🔍 Multi-timeframe {symbol}...")
    print(f"      📅 Testing timeframes: {', '.join(timeframes.keys())}")

    try:
        # RSI
        print(f"      ⚡ RSI multi-timeframe analysis...")
        rsi_param_grid = get_rsi_param_grid()
        rsi_results = multi_timeframe_backtest(
            test_rsi_strategy_on_stock,
            rsi_data_dict[symbol],
            symbol,
            rsi_param_grid,
            is_vwap=False,
            timeframes=timeframes,
        )
        best_rsi = max(rsi_results, key=lambda x: x["avg_metrics"]["total_return"])

        # VWAP
        print(f"      ⚡ VWAP multi-timeframe analysis...")
        vwap_param_grid = get_vwap_param_grid(symbol)
        vwap_results = multi_timeframe_backtest(
            test_vwap_strategy_on_stock,
            vwap_data_dict[symbol],
            symbol,
            vwap_param_grid,
            is_vwap=True,
            timeframes=timeframes,
        )
        best_vwap = max(vwap_results, key=lambda x: x["avg_metrics"]["total_return"])

        # Comparison
        winner = (
            "RSI"
            if best_rsi["avg_metrics"]["total_return"]
            > best_vwap["avg_metrics"]["total_return"]
            else "VWAP"
        )

        result = {
            "symbol": symbol,
            "rsi": best_rsi,
            "vwap": best_vwap,
            "winner": winner,
        }

        print(
            f"   ✓ Multi-timeframe {symbol}: {winner} won (RSI: {best_rsi['avg_metrics']['total_return']:.2%}, VWAP: {best_vwap['avg_metrics']['total_return']:.2%})"
        )
        return result

    except Exception as e:
        print(f"   ❌ Error multi-timeframe {symbol}: {e}")
        return None


def main():
    STOCKS = ["SATL", "MSFT", "NVDA", "EVLV", "AMZN", "GOOG", "F", "SCHW", "SPY"]

    TIMEFRAMES = {
        "1_day": 1,  # Last 1 trading day
        "5_days": 5,  # Last 5 trading days
        "2_weeks": 10,  # Last 2 weeks (10 trading days)
        "1_month": 22,  # Last 1 month (22 trading days)
    }

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(
        days=30
    )  # Get 30 days of data for multi-timeframe analysis
    print("=" * 100)
    print("📥 Downloading historical data (30 days) - PARALLEL", STOCKS)
    print("=" * 100)

    rsi_data = {}
    vwap_data = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_symbol = {
            executor.submit(download_symbol_data, symbol, start_date, end_date): symbol
            for symbol in STOCKS
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                symbol_result, rsi_result, vwap_result = future.result()
                if rsi_result is not None:
                    rsi_data[symbol_result] = rsi_result
                if vwap_result is not None:
                    vwap_data[symbol_result] = vwap_result

            except Exception as e:
                print(f"   ❌ Error downloading {symbol}: {e}")

    print(f"\n✅ Download completed: RSI ({len(rsi_data)}), VWAP ({len(vwap_data)})")

    print("\n" + "=" * 100)
    print("🔍 PHASE 1: INDIVIDUAL OPTIMIZATION BY SYMBOL - PARALLEL")
    print("=" * 100)

    # Parallel analysis for each symbol (RSI and VWAP simultaneously)
    with ThreadPoolExecutor(max_workers=2) as executor:  # 2 workers for analysis
        individual_results = []
        for symbol in STOCKS:
            try:
                future = executor.submit(
                    analyze_symbol_individual, symbol, rsi_data, vwap_data
                )
                individual_results.append((symbol, future))
            except Exception as e:
                print(f"   ❌ Error in individual analysis {symbol}: {e}")

    # Collect results
    print("📊 PHASE 1 REPORT: INDIVIDUAL OPTIMIZATION")
    final_individual_results = []

    for symbol, future in individual_results:
        try:
            res = future.result()
            if res is not None:  # Check if result is valid
                final_individual_results.append(res)
                rsi_ret = res["rsi"]["total_return"] if res["rsi"] else 0
                vwap_ret = res["vwap"]["total_return"] if res["vwap"] else 0
                winner = "RSI" if rsi_ret > vwap_ret else "VWAP"
                print(
                    f"   {symbol}: {winner} wins (RSI: {rsi_ret:.2%}, VWAP: {vwap_ret:.2%})"
                )
            else:
                print(f"   {symbol}: Analysis failed - no valid data")
        except Exception as e:
            print(f"   ❌ Error processing results for {symbol}: {e}")

    print("\n" + "=" * 100)
    print("🕒 PHASE 2: MULTI-TIMEFRAME ANALYSIS - PARALLEL")
    print("=" * 100)

    # Phase 2: Multi-timeframe analysis in parallel
    with ThreadPoolExecutor(max_workers=len(STOCKS)) as executor:
        multi_tf_futures = []
        for symbol in STOCKS:
            try:
                future = executor.submit(
                    analyze_symbol_multiframe,
                    symbol,
                    rsi_data,
                    vwap_data,
                    timeframes=TIMEFRAMES,
                )
                multi_tf_futures.append((symbol, future))
            except Exception as e:
                print(f"   ❌ Error in multi-timeframe analysis {symbol}: {e}")

    # Collect multi-timeframe results
    final_multi_tf_results = []
    for symbol, future in multi_tf_futures:
        try:
            res = future.result()
            if res is not None:  # Check if result is valid
                final_multi_tf_results.append(res)
                if res["rsi"] and res["vwap"]:
                    print(f"   {symbol} Multi-TF completed:")
                    print(
                        f"   RSI Avg Return: {res['rsi']['avg_metrics']['total_return']:.2%}"
                    )
                    print(
                        f"   VWAP Avg Return: {res['vwap']['avg_metrics']['total_return']:.2%}"
                    )
                else:
                    print(f"   {symbol}: Analysis completed but missing data")
            else:
                print(f"   {symbol}: Multi-timeframe analysis failed")
        except Exception as e:
            print(f"   ❌ Error processing multi-timeframe results for {symbol}: {e}")

    # Final comparison
    print("🏆 FINAL MULTI-TIMEFRAME REPORT")
    print(
        f"{'Symbol':<8} {'Winner':<8} {'RSI Ret':<9} {'VWAP Ret':<10} {'RSI Sharpe':<10} {'VWAP Sharpe':<11} {'RSI Calmar':<10} {'VWAP Calmar':<11}"
    )
    print("-" * 90)

    for res in final_multi_tf_results:
        if res["rsi"] and res["vwap"]:
            print(
                f"{res['symbol']:<8} {res['winner']:<8} {res['rsi']['avg_metrics']['total_return']:<9.2%} {res['vwap']['avg_metrics']['total_return']:<9.2%} {res['rsi']['avg_metrics']['sharpe_ratio']:<10.2f} {res['vwap']['avg_metrics']['sharpe_ratio']:<10.2f} {res['rsi']['avg_metrics']['calmar_ratio']:<10.2f} {res['vwap']['avg_metrics']['calmar_ratio']:<10.2f}"
            )

    print("✅ MULTI-TIMEFRAME ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
