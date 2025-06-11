import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data.alpaca_client import download_stock_data
from strategies.rsi_implementation import plot_best_strategy as plot_rsi_strategy
from strategies.vwap_implementation import plot_best_strategy as plot_vwap_strategy
from utils.terminal import print_header, print_separator

load_dotenv()
warnings.filterwarnings("ignore")


def optimize_rsi_single_stock(params_tuple):
    """Module-level function for RSI optimization - can be pickled for multiprocessing"""
    symbol, price_data, rsi_params, initial_capitals = params_tuple
    print(f"🚀 Starting RSI optimization for {symbol}...")

    best_return = -np.inf
    best_params = None
    all_results = []
    total_tested = 0

    # Test all parameter combinations
    for initial_capital in initial_capitals:
        for rsi_window in rsi_params["rsi_windows"]:
            for entry_level in rsi_params["entry_levels"]:
                for exit_level in rsi_params["exit_levels"]:
                    if entry_level >= exit_level:
                        continue
                    for take_profit in rsi_params["take_profits"]:
                        for stop_loss in rsi_params["stop_losses"]:
                            total_tested += 1

                            # Import here to avoid circular imports
                            from strategies.rsi_implementation import test_rsi_strategy_on_stock

                            result = test_rsi_strategy_on_stock(
                                price_data,
                                symbol,
                                rsi_window,
                                entry_level,
                                exit_level,
                                show_warnings=False,
                                take_profit=take_profit,
                                stop_loss=stop_loss,
                                initial_capital=initial_capital,
                            )

                            if result:
                                result["strategy_type"] = "RSI"
                                all_results.append(result)

                                # Multi-criteria scoring: return * (1 - max_drawdown) * win_rate
                                score = result["total_return"] * (1 - abs(result["max_drawdown"])) * result["win_rate"]

                                if score > best_return:
                                    best_return = score
                                    best_params = result.copy()
                                    best_params["optimization_score"] = score

                            if total_tested % 100 == 0:
                                print(f"   [{symbol}] Progress: {total_tested:,} combinations tested...")

    print(f"✅ RSI optimization [{symbol}]: {len(all_results)} valid results from {total_tested:,} tests")
    return symbol, best_params, all_results, price_data


def optimize_vwap_single_stock(params_tuple):
    """Module-level function for VWAP optimization - can be pickled for multiprocessing"""
    symbol, ohlcv_data, vwap_params, initial_capitals = params_tuple
    print(f"🚀 Starting VWAP optimization for {symbol}...")

    best_return = -np.inf
    best_params = None
    all_results = []
    total_tested = 0

    # Test all parameter combinations
    for initial_capital in initial_capitals:
        for entry_threshold in vwap_params["entry_thresholds"]:
            for exit_threshold in vwap_params["exit_thresholds"]:
                if entry_threshold <= exit_threshold:
                    continue
                for take_profit in vwap_params["take_profits"]:
                    for stop_loss in vwap_params["stop_losses"]:
                        total_tested += 1

                        # Import here to avoid circular imports
                        from strategies.vwap_implementation import test_vwap_strategy_on_stock

                        result = test_vwap_strategy_on_stock(
                            ohlcv_data,
                            symbol,
                            entry_threshold,
                            exit_threshold,
                            show_warnings=False,
                            take_profit=take_profit,
                            stop_loss=stop_loss,
                            initial_capital=initial_capital,
                        )

                        if result:
                            result["strategy_type"] = "VWAP"
                            all_results.append(result)

                            # Multi-criteria scoring: return * (1 - max_drawdown) * win_rate
                            score = result["total_return"] * (1 - abs(result["max_drawdown"])) * result["win_rate"]

                            if score > best_return:
                                best_return = score
                                best_params = result.copy()
                                best_params["optimization_score"] = score

                        if total_tested % 50 == 0:  # More frequent updates for VWAP
                            print(f"   [{symbol}] Progress: {total_tested:,} combinations tested...")

    print(f"✅ VWAP optimization [{symbol}]: {len(all_results)} valid results from {total_tested:,}")
    return symbol, best_params, all_results, ohlcv_data


class StrategyOptimizer:
    def __init__(self):
        # Comprehensive parameter configurations
        self.INITIAL_CAPITALS = [1000, 2500, 5000, 10000, 15000, 20000]  # Multiple capital amounts to test

        # RSI Strategy Parameters - Extensive ranges
        self.RSI_PARAMS = {
            "rsi_windows": list(range(8, 26, 2)),  # [8, 10, 12, 14, 16, 18, 20, 22, 24]
            "entry_levels": list(range(15, 40, 5)),  # [15, 20, 25, 30, 35]
            "exit_levels": list(range(60, 90, 5)),  # [60, 65, 70, 75, 80, 85]
            "take_profits": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06],
            "stop_losses": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06],
        }

        # VWAP Strategy Parameters - Extensive ranges
        self.VWAP_PARAMS = {
            "entry_thresholds": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02],
            "exit_thresholds": [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01],
            "take_profits": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06],
            "stop_losses": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06],
        }

        # Stock symbols to test
        self.TICKERS = ["TSLA"]

        # Results storage
        self.all_results = []
        self.rsi_results = []
        self.vwap_results = []

    def calculate_total_combinations(self):
        """Calculate total number of combinations for progress tracking"""
        rsi_combinations = (
            len(self.RSI_PARAMS["rsi_windows"])
            * len(self.RSI_PARAMS["entry_levels"])
            * len(self.RSI_PARAMS["exit_levels"])
            * len(self.RSI_PARAMS["take_profits"])
            * len(self.RSI_PARAMS["stop_losses"])
            * len(self.INITIAL_CAPITALS)
        )

        vwap_combinations = (
            len(self.VWAP_PARAMS["entry_thresholds"])
            * len(self.VWAP_PARAMS["exit_thresholds"])
            * len(self.VWAP_PARAMS["take_profits"])
            * len(self.VWAP_PARAMS["stop_losses"])
            * len(self.INITIAL_CAPITALS)
        )

        return rsi_combinations, vwap_combinations

    def download_all_data(self):
        """Download data for all symbols"""
        print_header("DOWNLOADING MARKET DATA")

        rsi_data = {}
        vwap_data = {}

        # Define time ranges
        end_date = datetime.now()
        rsi_start_date = end_date - timedelta(days=1)  # 1 day for RSI (minute bars)
        vwap_start_date = end_date - timedelta(hours=2)  # 2 hours for VWAP (second bars)

        for symbol in self.TICKERS:
            print(f"📊 Downloading data for {symbol}...")

            # Download RSI data (minute bars)
            rsi_data_symbol = download_stock_data(symbol, rsi_start_date, end_date, "rsi")
            if rsi_data_symbol is not None:
                rsi_data[symbol] = rsi_data_symbol
                print(f"   ✓ RSI data: {len(rsi_data_symbol)} minute bars")
            else:
                print(f"   ✗ RSI data: Failed to download")

            # Download VWAP data (4-second bars from trades)
            vwap_data_symbol = download_stock_data(symbol, vwap_start_date, end_date, "vwap")
            if vwap_data_symbol is not None:
                vwap_data[symbol] = vwap_data_symbol
                print(f"   ✓ VWAP data: {len(vwap_data_symbol.get('Close'))} 4-second bars")
            else:
                print(f"   ✗ VWAP data: Failed to download")

        print(f"\n✅ Data download completed")
        print(f"   RSI-ready symbols: {len(rsi_data)}")
        print(f"   VWAP-ready symbols: {len(vwap_data)}")

        return rsi_data, vwap_data

    def optimize_rsi_strategy(self, rsi_data):
        """Run comprehensive RSI optimization"""
        print_header("RSI STRATEGY COMPREHENSIVE OPTIMIZATION")

        rsi_combinations, _ = self.calculate_total_combinations()
        print(f"🔍 RSI Strategy: {rsi_combinations:,} total parameter combinations per symbol")
        print(f"📊 Testing {len(rsi_data)} symbols with {len(self.INITIAL_CAPITALS)} capital levels")

        # Run parallel optimization using module-level function
        with ProcessPoolExecutor(max_workers=min(len(rsi_data), 6)) as executor:
            future_to_symbol = {
                executor.submit(
                    optimize_rsi_single_stock, (symbol, price_data, self.RSI_PARAMS, self.INITIAL_CAPITALS)
                ): symbol
                for symbol, price_data in rsi_data.items()
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result[1] is not None:  # best_params
                        self.rsi_results.append(result)
                except Exception as exc:
                    print(f"❌ RSI optimization failed for {symbol}: {exc}")

        print(f"\n🎯 RSI Strategy optimization completed for {len(self.rsi_results)} symbols")
        return self.rsi_results

    def optimize_vwap_strategy(self, vwap_data):
        """Run comprehensive VWAP optimization"""
        print_header("VWAP STRATEGY COMPREHENSIVE OPTIMIZATION")

        _, vwap_combinations = self.calculate_total_combinations()
        print(f"🔍 VWAP Strategy: {vwap_combinations:,} total parameter combinations per symbol")
        print(f"📊 Testing {len(vwap_data)} symbols with {len(self.INITIAL_CAPITALS)} capital levels")

        # Run parallel optimization using module-level function
        with ProcessPoolExecutor(max_workers=min(len(vwap_data), 6)) as executor:
            future_to_symbol = {
                executor.submit(
                    optimize_vwap_single_stock, (symbol, ohlcv_data, self.VWAP_PARAMS, self.INITIAL_CAPITALS)
                ): symbol
                for symbol, ohlcv_data in vwap_data.items()
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result[1] is not None:  # best_params
                        self.vwap_results.append(result)
                except Exception as exc:
                    print(f"❌ VWAP optimization failed for {symbol}: {exc}")

        print(f"\n🎯 VWAP Strategy optimization completed for {len(self.vwap_results)} symbols")
        return self.vwap_results

    def compare_strategies(self):
        """Compare RSI and VWAP strategies to find the absolute best"""
        print_header("STRATEGY COMPARISON & FINAL RESULTS")

        # Combine all results
        all_strategy_results = []

        # Process RSI results
        for symbol, best_params, all_results, price_data in self.rsi_results:
            if best_params:
                all_strategy_results.append(best_params)

        # Process VWAP results
        for symbol, best_params, all_results, ohlcv_data in self.vwap_results:
            if best_params:
                all_strategy_results.append(best_params)

        if not all_strategy_results:
            print("❌ No valid results from either strategy")
            return None

        # Sort by optimization score (multi-criteria)
        all_strategy_results.sort(key=lambda x: x.get("optimization_score", 0), reverse=True)

        print(f"🏆 TOP 20 STRATEGIES ACROSS ALL SYMBOLS AND METHODS:")
        print(
            f"{'Rank':<4} {'Strategy':<8} {'Symbol':<8} "
            f"{'Return':<10} {'Score':<8} {'Profit':<12} "
            f"{'Capital':<10} {'Trades':<8}"
        )
        print_separator()

        for i, result in enumerate(all_strategy_results[:20], 1):
            print(
                f"{i:<4} {result['strategy_type']:<8} {result['symbol']:<8} "
                f"{result['total_return']:<9.2%} {result.get('optimization_score', 0):<7.3f} "
                f"${result['profit']:<10,.0f} ${result['initial_capital']:<9,.0f} {result['num_trades']:<8}"
            )

        # Find best overall
        best_overall = all_strategy_results[0]

        print_header("🥇 ABSOLUTE BEST STRATEGY CONFIGURATION")
        print(f"Strategy Type: {best_overall['strategy_type']}")
        print(f"Symbol: {best_overall['symbol']}")
        print(f"Optimization Score: {best_overall.get('optimization_score', 0):.4f}")
        print(f"Total Return: {best_overall['total_return']:.2%}")
        print(f"Profit: ${best_overall['profit']:,.2f}")
        print(f"Initial Capital: ${best_overall['initial_capital']:,.0f}")
        print(f"Final Capital: ${best_overall['final_capital']:,.2f}")
        print(f"Sharpe Ratio: {best_overall['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {best_overall['max_drawdown']:.2%}")
        print(f"Win Rate: {best_overall['win_rate']:.1%}")
        print(f"Number of Trades: {best_overall['num_trades']}")
        print(f"Take Profit: {best_overall['take_profit']:.2%}")
        print(f"Stop Loss: {best_overall['stop_loss']:.2%}")

        if best_overall["strategy_type"] == "RSI":
            print(f"RSI Window: {best_overall['rsi_window']}")
            print(f"Entry Level: {best_overall['entry_level']}")
            print(f"Exit Level: {best_overall['exit_level']}")
        else:
            print(f"Entry Threshold: {best_overall['entry_threshold']*100:.2f}%")
            print(f"Exit Threshold: {best_overall['exit_threshold']*100:.2f}%")

        # Strategy type comparison
        rsi_results_count = len([r for r in all_strategy_results if r["strategy_type"] == "RSI"])
        vwap_results_count = len([r for r in all_strategy_results if r["strategy_type"] == "VWAP"])

        print(f"\n📊 STRATEGY PERFORMANCE SUMMARY:")
        print(f"   RSI Strategy: {rsi_results_count} successful optimizations")
        print(f"   VWAP Strategy: {vwap_results_count} successful optimizations")

        # Calculate averages by strategy type
        rsi_avg_return = (
            np.mean([r["total_return"] for r in all_strategy_results if r["strategy_type"] == "RSI"])
            if rsi_results_count > 0
            else 0
        )
        vwap_avg_return = (
            np.mean([r["total_return"] for r in all_strategy_results if r["strategy_type"] == "VWAP"])
            if vwap_results_count > 0
            else 0
        )

        print(f"   RSI Average Return: {rsi_avg_return:.2%}")
        print(f"   VWAP Average Return: {vwap_avg_return:.2%}")

        # Save results to file
        self.save_results(all_strategy_results[:50])  # Save top 50 results

        return best_overall, all_strategy_results

    def save_results(self, results):
        """Save optimization results to JSON file"""
        try:
            # Convert results to JSON-serializable format
            json_results = []
            for result in results:
                json_result = {}
                for key, value in result.items():
                    if key == "portfolio":  # Skip portfolio object
                        continue
                    if isinstance(value, (np.integer, np.floating)):
                        json_result[key] = float(value)
                    elif pd.isna(value):
                        json_result[key] = None
                    else:
                        json_result[key] = value
                json_results.append(json_result)

            filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(json_results, f, indent=2, default=str)

            print(f"💾 Results saved to {filename}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")

    def plot_best_strategies(self, rsi_data, vwap_data, best_overall):
        """Plot the best performing strategies"""
        print_header("GENERATING VISUALIZATION FOR BEST STRATEGY")

        symbol = best_overall["symbol"]
        strategy_type = best_overall["strategy_type"]

        try:
            if strategy_type == "RSI" and symbol in rsi_data:
                print(f"📈 Generating RSI strategy plot for {symbol}...")
                plot_rsi_strategy(rsi_data[symbol], symbol, best_overall)
            elif strategy_type == "VWAP" and symbol in vwap_data:
                print(f"📈 Generating VWAP strategy plot for {symbol}...")
                plot_vwap_strategy(vwap_data[symbol], symbol, best_overall)
            else:
                print(f"⚠️ Could not plot: Missing data for {symbol} {strategy_type}")
        except Exception as e:
            print(f"❌ Error generating plot: {e}")

    def run_full_optimization(self):
        """Run the complete optimization process"""
        print_header("🚀 COMPREHENSIVE STRATEGY OPTIMIZATION SYSTEM")
        print("Testing RSI and VWAP strategies with extensive parameter variations")

        rsi_combinations, vwap_combinations = self.calculate_total_combinations()
        total_combinations = (rsi_combinations + vwap_combinations) * len(self.TICKERS)

        print(f"📊 Total combinations to test: {total_combinations:,}")
        print(f"   RSI combinations per symbol: {rsi_combinations:,}")
        print(f"   VWAP combinations per symbol: {vwap_combinations:,}")
        print(f"   Testing {len(self.TICKERS)} symbols")
        print(f"   Capital variations: {self.INITIAL_CAPITALS}")

        # Step 1: Download data
        rsi_data, vwap_data = self.download_all_data()

        if not rsi_data and not vwap_data:
            print("❌ No data available for optimization")
            return

        # Step 2: Optimize RSI strategy
        if rsi_data:
            self.optimize_rsi_strategy(rsi_data)

        # Step 3: Optimize VWAP strategy
        if vwap_data:
            self.optimize_vwap_strategy(vwap_data)

        # Step 4: Compare and find best
        best_overall, all_results = self.compare_strategies()

        # Step 5: Generate plots for best strategy
        if best_overall:
            self.plot_best_strategies(rsi_data, vwap_data, best_overall)

        print_header("✅ COMPREHENSIVE OPTIMIZATION COMPLETED")
        return best_overall, all_results


if __name__ == "__main__":
    optimizer = StrategyOptimizer()
    best_strategy, all_results = optimizer.run_full_optimization()

    if best_strategy:
        print(f"\n🎯 FINAL RECOMMENDATION:")
        print(f"   Use {best_strategy['strategy_type']} strategy on {best_strategy['symbol']}")
        print(f"   Expected return: {best_strategy['total_return']:.2%}")
        print(f"   With capital: ${best_strategy['initial_capital']:,.0f}")
    else:
        print("\n❌ No successful optimization found")
