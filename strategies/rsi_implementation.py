import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from utils.terminal import print_header, print_separator, get_terminal_width

load_dotenv()

import vectorbtpro as vbt
import datetime
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
import exchange_calendars as ecals
from plotly.subplots import make_subplots
from data.alpaca_client import download_stock_data_rsi

warnings.filterwarnings("ignore")

vbt.AlpacaData.set_custom_settings(
    client_config=dict(
        api_key=os.getenv("ALPACA_API_KEY"), secret_key=os.getenv("ALPACA_SECRET_KEY")
    )
)


def test_rsi_strategy_on_stock(
    price_data, symbol, rsi_window=20, entry_level=30, exit_level=75, show_warnings=True
):
    try:
        rsi = vbt.RSI.run(price_data, window=rsi_window)

        # Pick your exchange, e.g. New York Stock Exchange
        calendar = ecals.get_calendar("XNYS")

        # Build valid trading sessions for your data range
        sessions = calendar.sessions_in_range(
            price_data.index.min().date(), price_data.index.max().date()
        )

        # Fix holiday mask - convert sessions to date format that matches index
        sessions_dates = pd.to_datetime(sessions).date
        index_dates = price_data.index.date
        holiday_mask = pd.Series(
            [d in sessions_dates for d in index_dates], index=price_data.index
        )

        # Create time mask - using between_time to get boolean mask
        time_mask = price_data.index.indexer_between_time("9:30", "16:00")
        # Convert indexer to boolean mask
        time_bool_mask = pd.Series(False, index=price_data.index)
        if len(time_mask) > 0:
            time_bool_mask.iloc[time_mask] = True

        weekday_mask = price_data.index.weekday.isin([0, 1, 2, 3, 4])

        final_mask = time_bool_mask & weekday_mask & holiday_mask

        # If no valid trading times, skip time filtering for now
        if final_mask.sum() == 0:
            if show_warnings:
                print(
                    f"WARNING: No valid trading times found for {symbol}, using weekday filter only"
                )
            final_mask = weekday_mask & holiday_mask

        entries = rsi.rsi_crossed_below(entry_level) & final_mask
        exits = rsi.rsi_crossed_above(exit_level) & final_mask

        if not entries.any() or not exits.any():
            return None

        pf = vbt.Portfolio.from_signals(
            price_data,
            entries,
            exits,
            init_cash=INITIAL_CAPITAL,
            sl_stop=0.05,
            tp_stop=0.03,
            accumulate=False,
        )

        try:
            stats = pf.stats()

            # Handle different stat formats - convert to dict if needed
            if hasattr(stats, "to_dict"):
                stats_dict = stats.to_dict()
            elif isinstance(stats, dict):
                stats_dict = stats
            else:
                # Fallback if stats are not in expected format
                stats_dict = {}

            total_return = (
                stats_dict.get("Total Return [%]", 0) / 100
                if "Total Return [%]" in stats_dict
                else (
                    pf.value.iloc[-1] / pf.value.iloc[0] - 1 if len(pf.value) > 0 else 0
                )
            )
            sharpe_ratio = (
                stats_dict.get("Sharpe Ratio", 0) if "Sharpe Ratio" in stats_dict else 0
            )
            max_drawdown = (
                stats_dict.get("Max Drawdown [%]", 0) / 100
                if "Max Drawdown [%]" in stats_dict
                else 0
            )
            num_trades = (
                stats_dict.get("Total Trades", 0) if "Total Trades" in stats_dict else 0
            )
            win_rate = (
                stats_dict.get("Win Rate [%]", 0) / 100
                if "Win Rate [%]" in stats_dict
                else 0
            )

            initial_capital = pf.init_cash
            final_capital = pf.value.iloc[-1] if len(pf.value) > 0 else initial_capital
            profit = final_capital - initial_capital

        except Exception as e:
            print(f"Error getting stats for {symbol}: {e}")
            # Fallback calculations
            try:
                initial_capital = (
                    pf.init_cash if hasattr(pf, "init_cash") else INITIAL_CAPITAL
                )
                final_capital = (
                    pf.value.iloc[-1] if len(pf.value) > 0 else initial_capital
                )
                total_return = (
                    (final_capital / initial_capital) - 1 if initial_capital > 0 else 0
                )
                profit = final_capital - initial_capital
            except:
                initial_capital = INITIAL_CAPITAL
                final_capital = INITIAL_CAPITAL
                total_return = 0
                profit = 0

            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
            num_trades = 0

        return {
            "symbol": symbol,
            "rsi_window": rsi_window,
            "entry_level": entry_level,
            "exit_level": exit_level,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio if not pd.isna(sharpe_ratio) else 0,
            "max_drawdown": max_drawdown if not pd.isna(max_drawdown) else 0,
            "win_rate": win_rate if not pd.isna(win_rate) else 0,
            "num_trades": num_trades,
            "num_entries": entries.sum(),
            "num_exits": exits.sum(),
            "data_points": len(price_data),
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "profit": profit,
            "portfolio": pf,  # Store the portfolio for trade details
        }
    except Exception as e:
        print(f"Error testing RSI strategy on {symbol}: {e}")
        return None


def optimize_rsi_for_stock(price_data, symbol):
    print(f"\n🔍Optimizing RSI parameters for {symbol}...")

    rsi_windows = np.arange(10, 25, 2)  # RSI windows: 10, 12, 14, 16, 18, 20, 22, 24
    entry_levels = np.arange(20, 40, 5)  # Entry levels: 20, 25, 30, 35
    exit_levels = np.arange(60, 85, 5)  # Exit levels: 60, 65, 70, 75, 80

    best_return = -np.inf
    best_params = None
    all_results = []

    total_combinations = len(rsi_windows) * len(entry_levels) * len(exit_levels)
    current_combination = 0
    warning_shown = False

    for rsi_window in rsi_windows:
        for entry_level in entry_levels:
            for exit_level in exit_levels:
                current_combination += 1

                if entry_level >= exit_level:
                    continue

                # Only show warnings for the first combination to avoid spam
                show_warnings = not warning_shown
                result = test_rsi_strategy_on_stock(
                    price_data,
                    symbol,
                    rsi_window,
                    entry_level,
                    exit_level,
                    show_warnings,
                )
                warning_shown = True

                if result:
                    all_results.append(result)
                    if result["total_return"] > best_return:
                        best_return = result["total_return"]
                        best_params = result.copy()

                if current_combination % 20 == 0:
                    print(
                        f"   Progress: {current_combination}/{total_combinations} combinations tested..."
                    )

    print(f"   ✅ Completed: {len(all_results)} valid configurations tested")
    return best_params, all_results


def plot_best_strategy(price_data, symbol, best_params):
    try:
        rsi = vbt.RSI.run(price_data, window=best_params["rsi_window"])
        entries = rsi.rsi_crossed_below(best_params["entry_level"])
        exits = rsi.rsi_crossed_above(best_params["exit_level"])

        # Use the portfolio from optimization results instead of recreating it
        pf = best_params["portfolio"]

        # Create cleaner subplot layout
        fig = make_subplots(
            rows=3,
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            subplot_titles=(
                f"{symbol} - Price & Signals",
                "Trade Details",
                "RSI Indicator",
                "Performance Summary",
                "Portfolio Value",
                "",
            ),
            row_heights=[0.4, 0.3, 0.3],
            column_widths=[0.55, 0.45],  # More space for right column
            specs=[
                [{"type": "xy"}, {"type": "table"}],
                [{"type": "xy"}, {"type": "table"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
        )

        # Main price chart with clean signals (no text labels)
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data.values,
                mode="lines",
                name="Price",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Clean entry signals (no overlapping text)
        entry_prices = price_data[entries]
        if len(entry_prices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=entry_prices.index,
                    y=entry_prices.values,
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(color="green", size=14, symbol="triangle-up"),
                ),
                row=1,
                col=1,
            )

        # Clean exit signals (no overlapping text)
        exit_prices = price_data[exits]
        if len(exit_prices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=exit_prices.index,
                    y=exit_prices.values,
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(color="red", size=14, symbol="triangle-down"),
                ),
                row=1,
                col=1,
            )

        # RSI indicator
        fig.add_trace(
            go.Scatter(
                x=rsi.rsi.index,
                y=rsi.rsi.values,
                mode="lines",
                name="RSI",
                line=dict(color="purple", width=2),
            ),
            row=2,
            col=1,
        )

        # RSI levels
        fig.add_hline(
            y=best_params["entry_level"],
            line_dash="dash",
            line_color="green",
            annotation_text=f"Entry ({best_params['entry_level']})",
            row=2,
            col=1,
        )
        fig.add_hline(
            y=best_params["exit_level"],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Exit ({best_params['exit_level']})",
            row=2,
            col=1,
        )
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

        # Portfolio value
        portfolio_value = pf.value
        fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="orange", width=3),
            ),
            row=3,
            col=1,
        )

        # Create detailed trade information table
        try:
            trades = pf.trades.records

            if len(trades) > 0:
                trade_details = []
                for i, (idx, trade) in enumerate(trades.iterrows()):
                    entry_idx = int(trade["entry_idx"])
                    exit_idx = int(trade["exit_idx"])
                    entry_price = float(trade["entry_price"])
                    exit_price = float(trade["exit_price"])
                    pnl = float(trade["pnl"])
                    return_pct = float(trade["return"]) * 100

                    entry_date = (
                        pf.wrapper.index[entry_idx]
                        if entry_idx < len(pf.wrapper.index)
                        else "N/A"
                    )
                    exit_date = (
                        pf.wrapper.index[exit_idx]
                        if exit_idx < len(pf.wrapper.index) and exit_idx >= 0
                        else "Open"
                    )

                    trade_details.append(
                        [
                            f"#{i+1}",
                            f"{str(entry_date)[5:16]}",
                            f"${entry_price:.2f}",
                            f"{str(exit_date)[5:16]}",
                            f"${exit_price:.2f}",
                            f"${pnl:.2f}",
                            f"{return_pct:.2f}%",
                        ]
                    )

                # Add trade details table in right column
                headers = [
                    "#",
                    "Entry Date",
                    "Entry $",
                    "Exit Date",
                    "Exit $",
                    "PnL",
                    "Return%",
                ]

                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=headers,
                            fill_color="lightsteelblue",
                            align="center",
                            font=dict(size=12, color="black", family="Arial"),
                        ),
                        cells=dict(
                            values=list(zip(*trade_details)),
                            fill_color="lightgray",
                            align="center",
                            font=dict(size=11, color="black", family="Arial"),
                        ),
                    ),
                    row=1,
                    col=2,
                )

            # Create performance summary table data
            performance_metrics = [
                ["Strategy Configuration", ""],
                ["RSI Window", f"{best_params['rsi_window']} periods"],
                ["Entry Level", f"{best_params['entry_level']} RSI"],
                ["Exit Level", f"{best_params['exit_level']} RSI"],
                ["", ""],
                ["Trading Results", ""],
                ["Total Return", f"{best_params['total_return']:.2%}"],
                ["Profit/Loss", f"${best_params['profit']:.2f}"],
                ["Sharpe Ratio", f"{best_params['sharpe_ratio']:.2f}"],
                ["Max Drawdown", f"{best_params['max_drawdown']:.2%}"],
                ["Win Rate", f"{best_params['win_rate']:.1%}"],
                ["Total Trades", f"{best_params['num_trades']}"],
                ["", ""],
                ["Capital Summary", ""],
                ["Initial Capital", f"${best_params['initial_capital']:,.0f}"],
                ["Final Capital", f"${best_params['final_capital']:,.0f}"],
            ]

            # Add performance summary as a professional table
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["Metric", "Value"],
                        fill_color="lightsteelblue",
                        align="left",
                        font=dict(size=12, color="black", family="Arial"),
                    ),
                    cells=dict(
                        values=list(zip(*performance_metrics)),
                        fill_color=[
                            [
                                "lightblue" if i % 2 == 0 else "white"
                                for i in range(len(performance_metrics))
                            ],
                            [
                                "lightblue" if i % 2 == 0 else "white"
                                for i in range(len(performance_metrics))
                            ],
                        ],
                        align=["left", "right"],
                        font=dict(size=11, color="black", family="Arial"),
                    ),
                ),
                row=2,
                col=2,
            )

        except Exception as e:
            print(f"Error creating trade table: {e}")

        # Update layout with cleaner design
        fig.update_layout(
            title=dict(
                text=f"<b>{symbol} - RSI Strategy Analysis</b><br>"
                + f'<sub>Return: {best_params["total_return"]:.2%} | Profit: ${best_params["profit"]:.2f} | Trades: {best_params["num_trades"]}</sub>',
                x=0.5,
                font=dict(size=18),
            ),
            autosize=True,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        )

        # Update axes with better formatting
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Portfolio Value ($)", row=3, col=1)
        fig.update_xaxes(title_text="Date/Time", row=3, col=1)

        # Add subtle grid
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")

        fig.show()

    except Exception as e:
        print(f"Error plotting enhanced strategy for {symbol}: {e}")


def show_detailed_trades(optimization_results):
    """Show detailed trade information for each symbol"""
    print_separator("=")
    print("DETAILED TRADES BY SYMBOL".center(get_terminal_width()))
    print_separator("=")

    for result in optimization_results:
        symbol = result["symbol"]
        pf = result["portfolio"]

        print(f"🔍 DETAILED TRADES FOR {symbol}")
        print(
            f"Configuration: RSI Window={result['rsi_window']}, Entry={result['entry_level']}, Exit={result['exit_level']}"
        )
        print_separator()

        try:
            trades = pf.trades.records

            if len(trades) > 0:
                print(f"📈 Total trades: {len(trades)}")
                print(
                    f"{'#':<3} {'Type':<6} {'Quantity':<12} {'Entry Date':<20} {'Entry Price':<15} {'Exit Date':<20} {'Exit Price':<15} {'PnL':<12} {'Return %':<10}"
                )
                print_separator()

                for i, (idx, trade) in enumerate(trades.iterrows()):
                    try:
                        entry_idx = int(trade["entry_idx"])
                        exit_idx = int(trade["exit_idx"])
                        entry_price = float(trade["entry_price"])
                        exit_price = float(trade["exit_price"])
                        pnl = float(trade["pnl"])
                        size = float(trade["size"])
                        return_pct = float(trade["return"]) * 100

                        entry_date = (
                            pf.wrapper.index[entry_idx]
                            if entry_idx < len(pf.wrapper.index)
                            else "N/A"
                        )
                        exit_date = (
                            pf.wrapper.index[exit_idx]
                            if exit_idx < len(pf.wrapper.index) and exit_idx >= 0
                            else "Open"
                        )

                        trade_type = "LONG" if size > 0 else "SHORT"

                        exit_price_str = (
                            f"${exit_price:.2f}" if exit_price > 0 else "Open"
                        )

                        print(
                            f"{i+1:<3} {trade_type:<6} {abs(size):<11.2f} {str(entry_date):<20} ${entry_price:<14.2f} {str(exit_date):<20} {exit_price_str:<15} ${pnl:<11.2f} {return_pct:<9.2f}%"
                        )

                    except Exception as trade_error:
                        print(f"Error processing trade {i+1}: {trade_error}")
                        print(f"   Trade data: {trade}")

            else:
                print("❌ No trades registered")

                try:
                    print(f"\n📊 Portfolio statistics:")
                    stats = pf.stats()
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                except Exception as stats_error:
                    print(f"Error showing statistics: {stats_error}")

        except Exception as e:
            print(f"Error showing trades for {symbol}: {e}")

            try:
                print(f"\n📊 Portfolio statistics for {symbol}:")
                stats = pf.stats()
                for key, value in stats.items():
                    print(f"   {key}: {value}")
            except Exception as stats_error:
                print(f"Error showing statistics: {stats_error}")

        print_separator()


if __name__ == "__main__":
    # Configuration constants
    INITIAL_CAPITAL = 5000  # Single source of truth for initial capital
    STOCKS = [
        "TSLA",
        "MSFT",
        "NVDA",
        "EVLV",
        "SATL",
        "AMZN",
        "GOOG",
        "F",
        "SCHW",
        "SPY",
    ]

    print_header("INDIVIDUAL STOCK RSI PARAMETER OPTIMIZATION")

    stock_data = {}
    for symbol in STOCKS:
        data = download_stock_data_rsi(
            symbol,
            datetime.datetime.now() - datetime.timedelta(days=30),
            datetime.datetime.now(),
        )
        if data is not None:
            stock_data[symbol] = data
            print(f"✓ {symbol}: {len(data)} data points downloaded")
        else:
            print(f"✗ {symbol}: No data could be obtained")

    print(f"\n✅ Data downloaded successfully for {len(stock_data)} stocks")

    optimization_results = []

    for symbol, price_data in stock_data.items():
        print_separator("=")
        print(f"OPTIMIZING {symbol}".center(get_terminal_width()))
        print_separator("=")

        best_params, all_results = optimize_rsi_for_stock(price_data, symbol)

        if best_params:
            optimization_results.append(best_params)

            print(f"\n🏆 BEST CONFIGURATION FOR {symbol}:")
            print(f"   RSI Window: {best_params['rsi_window']}")
            print(f"   Entry Level: {best_params['entry_level']}")
            print(f"   Exit Level: {best_params['exit_level']}")
            print(f"   Total Return: {best_params['total_return']:.2%}")
            print(f"   Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {best_params['max_drawdown']:.2%}")
            print(f"   Win Rate: {best_params['win_rate']:.1%}")
            print(f"   Number of Trades: {best_params['num_trades']}")
            print(f"   Entry Signals: {best_params['num_entries']}")
            print(f"   Exit Signals: {best_params['num_exits']}")
            print(f"   Initial Capital: ${best_params['initial_capital']:,.2f}")
            print(f"   Final Capital: ${best_params['final_capital']:,.2f}")
            print(f"   Profit: ${best_params['profit']:,.2f}")

            if len(all_results) > 1:
                sorted_results = sorted(
                    all_results, key=lambda x: x["total_return"], reverse=True
                )[:5]
                print(f"\n📊 TOP 5 CONFIGURATIONS FOR {symbol}:")
                width = get_terminal_width()
                rank_width = max(5, (width - 55) // 7)  # Dynamic column width
                print(
                    f"{'Rank':<5} {'RSI':<5} {'Entry':<7} {'Exit':<6} {'Return':<10} {'Profit':<12} {'Trades':<8}"
                )
                print_separator()
                for i, result in enumerate(sorted_results, 1):
                    print(
                        f"{i:<5} {result['rsi_window']:<5} {result['entry_level']:<7} {result['exit_level']:<6} {result['total_return']:<9.2%} ${result['profit']:<10,.0f} {result['num_trades']:<8}"
                    )

            print(f"\n📈 Generating graph for {symbol}...")
            plot_best_strategy(price_data, symbol, best_params)

            show_detailed_trades([best_params])

        else:
            print(f"❌ No valid configurations found for {symbol}")

    print_header("FINAL SUMMARY - BEST CONFIGURATIONS BY STOCK")

    if optimization_results:
        optimization_results.sort(key=lambda x: x["total_return"], reverse=True)

        print(
            f"{'Stock':<8} {'RSI':<5} {'Entry':<7} {'Exit':<6} {'Return':<10} {'Profit':<12} {'Sharpe':<8} {'Trades':<8}"
        )
        print_separator()

        for result in optimization_results:
            print(
                f"{result['symbol']:<8} {result['rsi_window']:<5} {result['entry_level']:<7} {result['exit_level']:<6} {result['total_return']:<9.2%} ${result['profit']:<10,.0f} {result['sharpe_ratio']:<7.2f} {result['num_trades']:<8}"
            )

        best_overall = optimization_results[0]
        print(f"\n🥇 BEST STOCK GENERAL: {best_overall['symbol']}")
        print(
            f"   Configuration: RSI={best_overall['rsi_window']}, Entry={best_overall['entry_level']}, Exit={best_overall['exit_level']}"
        )
        print(f"   Return: {best_overall['total_return']:.2%}")
        print(f"   Profit: ${best_overall['profit']:.2f}")

        profitable_stocks = [r for r in optimization_results if r["total_return"] > 0]
        print(
            f"\n📈 Profitable stocks: {len(profitable_stocks)}/{len(optimization_results)}"
        )

        if profitable_stocks:
            avg_return = sum(r["total_return"] for r in profitable_stocks) / len(
                profitable_stocks
            )
            avg_profit = sum(r["profit"] for r in profitable_stocks) / len(
                profitable_stocks
            )
            total_profit = sum(r["profit"] for r in profitable_stocks)
            print(f"📊 Average return (profitable stocks): {avg_return:.2%}")
            print(f"💰 Average profit (profitable stocks): ${avg_profit::.2f}")
            print(f"💎 Total profit combined: ${total_profit:,.2f}")

        show_detailed_trades(optimization_results)

    else:
        print("❌ No valid configurations found for any stock")

    print_header("OPTIMIZATION COMPLETED")
