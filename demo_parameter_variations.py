#!/usr/bin/env python3
"""
Demonstration of Comprehensive Parameter Variations for RSI and VWAP Strategies

This script shows how to configure and run extensive parameter variations
from the main code scope for both RSI and VWAP trading strategies.
"""

from main import StrategyOptimizer


def demonstrate_parameter_configurations():
    """Demonstrate the extensive parameter configurations available"""

    print("=" * 80)
    print("COMPREHENSIVE PARAMETER VARIATIONS DEMONSTRATION")
    print("=" * 80)

    # Create optimizer instance to show configurations
    optimizer = StrategyOptimizer()

    print("\n1. INITIAL CAPITAL VARIATIONS:")
    print(f"   Testing {len(optimizer.INITIAL_CAPITALS)} different capital amounts:")
    for i, capital in enumerate(optimizer.INITIAL_CAPITALS, 1):
        print(f"   {i}. ${capital:,}")

    print("\n2. RSI STRATEGY PARAMETER RANGES:")
    print(f"   RSI Windows: {len(optimizer.RSI_PARAMS['rsi_windows'])} values")
    print(f"   Values: {optimizer.RSI_PARAMS['rsi_windows']}")

    print(f"\n   Entry Levels: {len(optimizer.RSI_PARAMS['entry_levels'])} values")
    print(f"   Values: {optimizer.RSI_PARAMS['entry_levels']}")

    print(f"\n   Exit Levels: {len(optimizer.RSI_PARAMS['exit_levels'])} values")
    print(f"   Values: {optimizer.RSI_PARAMS['exit_levels']}")

    print(f"\n   Take Profit Levels: {len(optimizer.RSI_PARAMS['take_profits'])} values")
    print(f"   Values: {[f'{tp:.1%}' for tp in optimizer.RSI_PARAMS['take_profits']]}")

    print(f"\n   Stop Loss Levels: {len(optimizer.RSI_PARAMS['stop_losses'])} values")
    print(f"   Values: {[f'{sl:.1%}' for sl in optimizer.RSI_PARAMS['stop_losses']]}")

    print("\n3. VWAP STRATEGY PARAMETER RANGES:")
    print(f"   Entry Thresholds: {len(optimizer.VWAP_PARAMS['entry_thresholds'])} values")
    print(f"   Values: {[f'{et:.1%}' for et in optimizer.VWAP_PARAMS['entry_thresholds']]}")

    print(f"\n   Exit Thresholds: {len(optimizer.VWAP_PARAMS['exit_thresholds'])} values")
    print(f"   Values: {[f'{et:.2%}' for et in optimizer.VWAP_PARAMS['exit_thresholds']]}")

    print(f"\n   Take Profit Levels: {len(optimizer.VWAP_PARAMS['take_profits'])} values")
    print(f"   Values: {[f'{tp:.1%}' for tp in optimizer.VWAP_PARAMS['take_profits']]}")

    print(f"\n   Stop Loss Levels: {len(optimizer.VWAP_PARAMS['stop_losses'])} values")
    print(f"   Values: {[f'{sl:.1%}' for sl in optimizer.VWAP_PARAMS['stop_losses']]}")

    # Calculate total combinations
    rsi_combinations, vwap_combinations = optimizer.calculate_total_combinations()
    total_combinations = (rsi_combinations + vwap_combinations) * len(optimizer.TICKERS)

    print("\n4. TOTAL COMBINATIONS ANALYSIS:")
    print(f"   RSI combinations per symbol: {rsi_combinations:,}")
    print(f"   VWAP combinations per symbol: {vwap_combinations:,}")
    print(f"   Total symbols: {len(optimizer.TICKERS)}")
    print(f"   GRAND TOTAL combinations: {total_combinations:,}")

    print(f"\n5. STOCK SYMBOLS TO TEST:")
    for i, symbol in enumerate(optimizer.TICKERS, 1):
        print(f"   {i:2}. {symbol}")

    return optimizer, total_combinations


def demonstrate_custom_parameter_ranges(total_combinations):
    """Show how to customize parameter ranges from main code scope"""

    print("\n" + "=" * 80)
    print("CUSTOM PARAMETER RANGE CONFIGURATION")
    print("=" * 80)

    # Create a custom optimizer with different parameters
    class CustomStrategyOptimizer(StrategyOptimizer):
        def __init__(self):
            super().__init__()

            # Example: Smaller capital range for faster testing
            self.INITIAL_CAPITALS = [1000, 5000, 10000]

            # Example: More focused RSI parameters
            self.RSI_PARAMS = {
                "rsi_windows": [10, 14, 20],  # Just 3 most common windows
                "entry_levels": [20, 25, 30],  # Focused entry range
                "exit_levels": [70, 75, 80],  # Focused exit range
                "take_profits": [0.02, 0.03, 0.04],  # Conservative profits
                "stop_losses": [0.02, 0.03, 0.04],  # Conservative stops
            }

            # Example: More aggressive VWAP parameters
            self.VWAP_PARAMS = {
                "entry_thresholds": [0.005, 0.01, 0.015],  # Higher thresholds
                "exit_thresholds": [0.002, 0.004, 0.006],  # Quicker exits
                "take_profits": [0.01, 0.02, 0.03],  # Faster profits
                "stop_losses": [0.015, 0.025, 0.035],  # Wider stops
            }

            # Example: Focus on specific high-volume stocks
            self.TICKERS = ["TSLA", "MSFT", "NVDA", "SPY"]

    custom_optimizer = CustomStrategyOptimizer()

    print("\nCUSTOM CONFIGURATION EXAMPLE:")
    print(f"   Capital variations: {custom_optimizer.INITIAL_CAPITALS}")
    print(f"   RSI Windows: {custom_optimizer.RSI_PARAMS['rsi_windows']}")
    print(f"   Focused symbols: {custom_optimizer.TICKERS}")

    # Calculate reduced combinations
    rsi_comb, vwap_comb = custom_optimizer.calculate_total_combinations()
    total_comb = (rsi_comb + vwap_comb) * len(custom_optimizer.TICKERS)

    print(f"\nREDUCED COMPLEXITY:")
    print(f"   RSI combinations per symbol: {rsi_comb:,}")
    print(f"   VWAP combinations per symbol: {vwap_comb:,}")
    print(f"   Total combinations: {total_comb:,}")
    print(f"   Reduction factor: {total_combinations/total_comb:.1f}x faster")

    return custom_optimizer


def demonstrate_parameter_impact():
    """Show the impact of different parameter choices"""

    print("\n" + "=" * 80)
    print("PARAMETER IMPACT ANALYSIS")
    print("=" * 80)

    print("\nRSI STRATEGY PARAMETER IMPACT:")
    print("   RSI Window:")
    print("     • Lower values (8-12): More sensitive, more signals, higher noise")
    print("     • Medium values (14-20): Balanced sensitivity and reliability")
    print("     • Higher values (22-24): Less sensitive, fewer but stronger signals")

    print("\n   Entry/Exit Levels:")
    print("     • Narrow spread (30/70): More frequent trades, lower profit per trade")
    print("     • Wide spread (20/80): Fewer trades, higher profit potential")

    print("\n   Take Profit/Stop Loss:")
    print("     • Conservative (1-2%): Lower risk, more frequent small wins/losses")
    print("     • Aggressive (4-6%): Higher risk, fewer but larger wins/losses")

    print("\nVWAP STRATEGY PARAMETER IMPACT:")
    print("   Entry/Exit Thresholds:")
    print("     • Small thresholds (0.1-0.5%): Very sensitive to price movements")
    print("     • Large thresholds (1-2%): Only trades on significant deviations")

    print("\n   Capital Amount Impact:")
    print("     • $1,000: Minimal position sizes, good for testing")
    print("     • $5,000: Moderate positions, realistic retail trading")
    print("     • $20,000: Larger positions, institutional-like exposure")


def demonstrate_usage_examples():
    """Show practical examples of how to use the system"""

    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)

    print("\n1. QUICK TEST RUN (for development):")
    print(
        """
    # Modify TICKERS in main.py for quick testing
    self.TICKERS = ["TSLA", "SPY"]  # Just 2 stocks
    self.INITIAL_CAPITALS = [5000]  # Just one capital amount
    """
    )

    print("\n2. COMPREHENSIVE PRODUCTION RUN:")
    print(
        """
    # Use default settings in main.py
    python main.py
    # This will test all combinations and save results
    """
    )

    print("\n3. FOCUS ON SPECIFIC STRATEGY:")
    print(
        """
    # Comment out one strategy in run_full_optimization():
    # if rsi_data:
    #     self.optimize_rsi_strategy(rsi_data)

    if vwap_data:  # Only run VWAP
        self.optimize_vwap_strategy(vwap_data)
    """
    )

    print("\n4. PARAMETER SENSITIVITY ANALYSIS:")
    print(
        """
    # Test impact of specific parameters by creating focused ranges:
    self.RSI_PARAMS['rsi_windows'] = [14]  # Fix RSI window
    self.RSI_PARAMS['entry_levels'] = range(15, 36, 1)  # Vary entry finely
    """
    )


if __name__ == "__main__":
    # Run all demonstrations
    print("COMPREHENSIVE PARAMETER VARIATIONS SYSTEM")
    print("This system tests extensive combinations to find optimal strategies")

    # Show default configuration
    optimizer, total_combinations = demonstrate_parameter_configurations()

    # Show custom configuration
    custom_optimizer = demonstrate_custom_parameter_ranges(total_combinations)

    # Show parameter impact
    demonstrate_parameter_impact()

    # Show usage examples
    demonstrate_usage_examples()

    print("\n" + "=" * 80)
    print("READY TO RUN COMPREHENSIVE OPTIMIZATION")
    print("=" * 80)
    print("\nTo start the full optimization, run:")
    print("   python main.py")
    print("\nThe system will:")
    print("   1. Download market data for all symbols")
    print("   2. Test all RSI parameter combinations")
    print("   3. Test all VWAP parameter combinations")
    print("   4. Compare strategies and find the absolute best")
    print("   5. Generate visualizations and save results")

    # Calculate combinations safely
    rsi_comb, vwap_comb = optimizer.calculate_total_combinations()
    total_comb = (rsi_comb + vwap_comb) * len(optimizer.TICKERS)
    print(f"\nTotal combinations to test: {total_comb:,}")
