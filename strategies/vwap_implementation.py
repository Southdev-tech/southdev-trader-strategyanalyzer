import sys
import os
from dotenv import load_dotenv

# Load environment variables FIRST (including NUMBA_DISABLE_JIT=1) before importing VectorBT
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import vectorbtpro as vbt
import datetime
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.alpaca_client import download_stock_data_vwap
import shutil
import exchange_calendars as ecals
warnings.filterwarnings('ignore')

# Get terminal width for dynamic formatting
def get_terminal_width():
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80  # fallback to 80 if unable to detect

def print_header(text, char="="):
    """Print a centered header with full terminal width"""
    width = get_terminal_width()
    print(char * width)
    print(text.center(width))
    print(char * width)

def print_separator(char="-"):
    """Print a separator line with full terminal width"""
    width = get_terminal_width()
    print(char * width)

# Configuration constants
INITIAL_CAPITAL = 20000  # Single source of truth for initial capital
STOCKS = ['SATL']

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=10)

def calculate_vwap(data):
    try:
        df = data.get()

        vwap_indicator = vbt.VWAP.run(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume'],
            anchor='1D'  # Daily anchor for session-based VWAP
        )
        
        return vwap_indicator.vwap
        
    except Exception as e:
        print(f"Error calculating VWAP: {e}")
        return None

def test_vwap_strategy_on_stock(data, symbol, entry_threshold=0.02, exit_threshold=0.015, show_warnings=True, take_profit=0.03, stop_loss=0.05):
    try:
        close_price = data.get('Close')
        vwap = calculate_vwap(data)
        
        if vwap is None:
            return None
        
        price_deviation = (close_price - vwap) / vwap
        
        # Generate initial entry/exit signals
        initial_entries = price_deviation < -entry_threshold
        initial_exits = price_deviation > exit_threshold
        
        # Pick your exchange, e.g. New York Stock Exchange
        calendar = ecals.get_calendar("XNYS")

        # Build valid trading sessions for your data range
        sessions = calendar.sessions_in_range(
            close_price.index.min().date(),
            close_price.index.max().date()
        )
        
        # Fix holiday mask - convert sessions to date format that matches index
        sessions_dates = pd.to_datetime(sessions).date
        index_dates = close_price.index.date
        holiday_mask = pd.Series([d in sessions_dates for d in index_dates], index=close_price.index)
        
        # Create time mask - using between_time to get boolean mask
        time_mask = close_price.index.indexer_between_time("9:30", "16:00")
        # Convert indexer to boolean mask
        time_bool_mask = pd.Series(False, index=close_price.index)
        if len(time_mask) > 0:
            time_bool_mask.iloc[time_mask] = True
        
        weekday_mask = close_price.index.weekday.isin([0, 1, 2, 3, 4])

        final_mask = time_bool_mask & weekday_mask & holiday_mask
        
        # If no valid trading times, skip time filtering for now
        if final_mask.sum() == 0:
            if show_warnings:
                print(f"WARNING: No valid trading times found for {symbol}, using weekday filter only")
            final_mask = weekday_mask & holiday_mask

        # Apply time filtering to entry/exit signals
        entries = initial_entries & final_mask
        exits = initial_exits & final_mask
        
        if not entries.any() or not exits.any():
            return None
            
        pf = vbt.Portfolio.from_signals(
            close_price,
            entries,
            exits,
            init_cash=INITIAL_CAPITAL,
            sl_stop=stop_loss,
            tp_stop=take_profit,
            accumulate=False,
        )
        
        try:
            stats = pf.stats()
            
            # Handle different stat formats - convert to dict if needed
            if hasattr(stats, 'to_dict'):
                stats_dict = stats.to_dict()
            elif isinstance(stats, dict):
                stats_dict = stats
            else:
                # Fallback if stats are not in expected format
                stats_dict = {}
            
            total_return = stats_dict.get('Total Return [%]', 0) / 100 if 'Total Return [%]' in stats_dict else (pf.value.iloc[-1] / pf.value.iloc[0] - 1 if len(pf.value) > 0 else 0)
            sharpe_ratio = stats_dict.get('Sharpe Ratio', 0) if 'Sharpe Ratio' in stats_dict else 0
            max_drawdown = stats_dict.get('Max Drawdown [%]', 0) / 100 if 'Max Drawdown [%]' in stats_dict else 0
            num_trades = stats_dict.get('Total Trades', 0) if 'Total Trades' in stats_dict else 0
            win_rate = stats_dict.get('Win Rate [%]', 0) / 100 if 'Win Rate [%]' in stats_dict else 0
            
            initial_capital = pf.init_cash
            final_capital = pf.value.iloc[-1] if len(pf.value) > 0 else initial_capital
            profit = final_capital - initial_capital
            
        except Exception as e:
            print(f"Error getting stats for {symbol}: {e}")
            # Fallback calculations
            try:
                initial_capital = pf.init_cash if hasattr(pf, 'init_cash') else INITIAL_CAPITAL
                final_capital = pf.value.iloc[-1] if len(pf.value) > 0 else initial_capital
                total_return = (final_capital / initial_capital) - 1 if initial_capital > 0 else 0
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
            'symbol': symbol,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'total_return': total_return,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'sharpe_ratio': sharpe_ratio if not pd.isna(sharpe_ratio) else 0,
            'max_drawdown': max_drawdown if not pd.isna(max_drawdown) else 0,
            'win_rate': win_rate if not pd.isna(win_rate) else 0,
            'num_trades': num_trades,
            'num_entries': entries.sum(),
            'num_exits': exits.sum(),
            'data_points': len(close_price),
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'profit': profit,
            'portfolio': pf  # Store the portfolio for trade details
        }
    except Exception as e:
        print(f"Error testing VWAP strategy on {symbol}: {e}")
        return None

def optimize_vwap_for_stock(ohlcv_data, symbol):
    print(f"\n🔍 Optimizing VWAP parameters for {symbol} (4-second bars from trades)...")

    entry_thresholds = np.arange(0.002, 0.015, 0.002)    # Entry: 0.2%, 0.4%, 0.6%, 0.8%, 1.0%, 1.2%, 1.4%
    exit_thresholds = np.arange(0.001, 0.01, 0.001)      # Exit thresholds: 0.5%, 1.0%, 1.5%, 2.0%, 2.5%
    take_profits = np.arange(0.01, 0.05, 0.01)  # Take profits: 0.01, 0.02, 0.03, 0.04, 0.05
    stop_losses = np.arange(0.01, 0.05, 0.01)  # Stop losses: 0.01, 0.02, 0.03, 0.04, 0.05

    best_return = -np.inf
    best_params = None
    all_results = []

    total_combinations = len(entry_thresholds) * len(exit_thresholds) * len(take_profits) * len(stop_losses)
    current_combination = 0
    warning_shown = False

    for entry_threshold in entry_thresholds:
        for exit_threshold in exit_thresholds:
            for take_profit in take_profits:
                for stop_loss in stop_losses:
                    current_combination += 1

                    if entry_threshold <= exit_threshold:
                        continue

                    # Only show warnings for the first combination to avoid spam
                    show_warnings = not warning_shown
                    result = test_vwap_strategy_on_stock(ohlcv_data, symbol, entry_threshold, exit_threshold, show_warnings, take_profit, stop_loss)
                    warning_shown = True

                    if result:
                        all_results.append(result)
                        if result['total_return'] > best_return:
                            best_return = result['total_return']
                            best_params = result.copy()

                    if current_combination % 5 == 0:
                        print(f"   Progress: {current_combination}/{total_combinations} combinations tested...")

    print(f"   ✅ Completed: {len(all_results)} valid configurations tested")
    return best_params, all_results

def plot_best_strategy(data, symbol, best_params):
    try:
        close_price = data.get('Close')
        vwap = calculate_vwap(data)
        
        price_deviation = (close_price - vwap) / vwap
        
        # Generate initial entry/exit signals
        initial_entries = price_deviation < -best_params['entry_threshold']
        initial_exits = price_deviation > best_params['exit_threshold']
        
        # Apply same time filtering as in strategy
        calendar = ecals.get_calendar("XNYS")
        sessions = calendar.sessions_in_range(
            close_price.index.min().date(),
            close_price.index.max().date()
        )
        
        sessions_dates = pd.to_datetime(sessions).date
        index_dates = close_price.index.date
        holiday_mask = pd.Series([d in sessions_dates for d in index_dates], index=close_price.index)
        
        time_mask = close_price.index.indexer_between_time("9:30", "16:00")
        time_bool_mask = pd.Series(False, index=close_price.index)
        if len(time_mask) > 0:
            time_bool_mask.iloc[time_mask] = True
        
        weekday_mask = close_price.index.weekday.isin([0, 1, 2, 3, 4])
        final_mask = time_bool_mask & weekday_mask & holiday_mask
        
        if final_mask.sum() == 0:
            final_mask = weekday_mask & holiday_mask
        
        # Apply time filtering to signals
        entries = initial_entries & final_mask
        exits = initial_exits & final_mask
        
        # Use the portfolio from optimization results instead of recreating it
        pf = best_params['portfolio']
        
        # Create cleaner subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            subplot_titles=(
                f'{symbol} - Price & Signals', 'Trade Details',
                'VWAP vs Price & Deviation', 'Performance Summary', 
                'Portfolio Value', ''
            ),
            row_heights=[0.4, 0.3, 0.3],
            column_widths=[0.55, 0.45],  # More space for right column
            specs=[[{"type": "xy"}, {"type": "table"}],
                   [{"type": "xy"}, {"type": "table"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )

        # Main price chart with clean signals (no text labels)
        fig.add_trace(
            go.Scatter(x=close_price.index, y=close_price.values,
                      mode='lines', name='Price', line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # Clean entry signals (no overlapping text)
        entry_prices = close_price[entries]
        if len(entry_prices) > 0:
            fig.add_trace(
                go.Scatter(x=entry_prices.index, y=entry_prices.values,
                          mode='markers', name='Buy Signal',
                          marker=dict(color='green', size=14, symbol='triangle-up')),
                row=1, col=1
            )

        # Clean exit signals (no overlapping text)
        exit_prices = close_price[exits]
        if len(exit_prices) > 0:
            fig.add_trace(
                go.Scatter(x=exit_prices.index, y=exit_prices.values,
                          mode='markers', name='Sell Signal',
                          marker=dict(color='red', size=14, symbol='triangle-down')),
                row=1, col=1
            )

        # VWAP vs Price chart
        fig.add_trace(
            go.Scatter(x=close_price.index, y=close_price.values,
                      mode='lines', name='Price', line=dict(color='blue', width=2)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=vwap.index, y=vwap.values,
                      mode='lines', name='VWAP', line=dict(color='orange', width=2)),
            row=2, col=1
        )

        # Portfolio value
        portfolio_value = pf.value
        fig.add_trace(
            go.Scatter(x=portfolio_value.index, y=portfolio_value.values,
                      mode='lines', name='Portfolio Value', 
                      line=dict(color='darkgreen', width=3)),
            row=3, col=1
        )

        # Create detailed trade information table
        try:
            trades = pf.trades.records
            
            if len(trades) > 0:
                trade_details = []
                for i, (idx, trade) in enumerate(trades.iterrows()):
                    entry_idx = int(trade['entry_idx'])
                    exit_idx = int(trade['exit_idx'])
                    entry_price = float(trade['entry_price'])
                    exit_price = float(trade['exit_price'])
                    pnl = float(trade['pnl'])
                    return_pct = float(trade['return']) * 100
                    
                    entry_date = pf.wrapper.index[entry_idx] if entry_idx < len(pf.wrapper.index) else "N/A"
                    exit_date = pf.wrapper.index[exit_idx] if exit_idx < len(pf.wrapper.index) and exit_idx >= 0 else "Open"
                    
                    trade_details.append([
                        f"#{i+1}",
                        f"{str(entry_date)[5:16]}",
                        f"${entry_price:.2f}",
                        f"{str(exit_date)[5:16]}",
                        f"${exit_price:.2f}",
                        f"${pnl:.2f}",
                        f"{return_pct:.2f}%"
                    ])

                # Add trade details table in right column
                headers = ['#', 'Entry Date', 'Entry $', 'Exit Date', 'Exit $', 'PnL', 'Return%']
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=headers,
                                   fill_color='lightsteelblue',
                                   align='center',
                                   font=dict(size=12, color='black', family="Arial")),
                        cells=dict(values=list(zip(*trade_details)),
                                  fill_color='lightgray',
                                  align='center',
                                  font=dict(size=11, color='black', family="Arial"))
                    ),
                    row=1, col=2
                )

            # Create performance summary table data
            performance_metrics = [
                ["Strategy Configuration", ""],
                ["Entry Threshold", f"{best_params['entry_threshold']*100:.1f}%"],
                ["Exit Threshold", f"{best_params['exit_threshold']*100:.1f}%"],
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
                ["Final Capital", f"${best_params['final_capital']:,.0f}"]
            ]

            # Add performance summary as a professional table
            fig.add_trace(
                go.Table(
                    header=dict(values=["Metric", "Value"],
                               fill_color='lightsteelblue',
                               align='left',
                               font=dict(size=12, color='black', family="Arial")),
                    cells=dict(values=list(zip(*performance_metrics)),
                              fill_color=[['lightblue' if i % 2 == 0 else 'white' for i in range(len(performance_metrics))],
                                         ['lightblue' if i % 2 == 0 else 'white' for i in range(len(performance_metrics))]],
                              align=['left', 'right'],
                              font=dict(size=11, color='black', family="Arial"))
                ),
                row=2, col=2
            )

        except Exception as e:
            print(f"Error creating trade table: {e}")

        # Update layout with cleaner design
        fig.update_layout(
            title=dict(
                text=f'<b>{symbol} - VWAP Strategy Analysis</b><br>' +
                     f'<sub>Return: {best_params["total_return"]:.2%} | Profit: ${best_params["profit"]:.2f} | Trades: {best_params["num_trades"]}</sub>',
                x=0.5,
                font=dict(size=18)
            ),
            height=900,
            width=1300,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
        )

        # Update axes with better formatting
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price/VWAP ($)", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=3, col=1)
        fig.update_xaxes(title_text="Date/Time", row=3, col=1)

        # Add subtle grid
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')

        fig.show()

    except Exception as e:
        print(f"Error plotting enhanced strategy for {symbol}: {e}")

def show_detailed_trades(optimization_results):
    """Show detailed trade information for each symbol"""
    print_separator("=")
    print("DETAILED TRADES BY SYMBOL".center(get_terminal_width()))
    print_separator("=")
    
    for result in optimization_results:
        symbol = result['symbol']
        pf = result['portfolio']
        
        print(f"🔍 DETAILED TRADES FOR {symbol}")
        print(f"Configuration: Entry={result['entry_threshold']*100:.1f}%, Exit={result['exit_threshold']*100:.1f}%")
        print_separator()
        
        try:
            trades = pf.trades.records
            
            if len(trades) > 0:
                print(f"📈 Total trades: {len(trades)}")
                print(f"{'#':<3} {'Type':<6} {'Quantity':<12} {'Entry Date':<20} {'Entry Price':<15} {'Exit Date':<20} {'Exit Price':<15} {'PnL':<12} {'Return %':<10}")
                print_separator()
                
                for i, (idx, trade) in enumerate(trades.iterrows()):
                    try:
                        entry_idx = int(trade['entry_idx'])
                        exit_idx = int(trade['exit_idx'])
                        entry_price = float(trade['entry_price'])
                        exit_price = float(trade['exit_price'])
                        pnl = float(trade['pnl'])
                        size = float(trade['size'])
                        return_pct = float(trade['return']) * 100
                        
                        entry_date = pf.wrapper.index[entry_idx] if entry_idx < len(pf.wrapper.index) else "N/A"
                        exit_date = pf.wrapper.index[exit_idx] if exit_idx < len(pf.wrapper.index) and exit_idx >= 0 else "Open"
                        
                        trade_type = "LONG" if size > 0 else "SHORT"
                        
                        exit_price_str = f"${exit_price:.2f}" if exit_price > 0 else "Open"
                        
                        print(f"{i+1:<3} {trade_type:<6} {abs(size):<11.2f} {str(entry_date):<20} ${entry_price:<14.2f} {str(exit_date):<20} {exit_price_str:<15} ${pnl:<11.2f} {return_pct:<9.2f}%")
                        
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

print_header("VWAP PARAMETER OPTIMIZATION - ANALYSIS WITH TRADES DATA")
print("📊 Downloading individual trades data and converting to 4-second OHLCV bars")
print("🤖 Simulating real-time bot analysis with trades data")
print_separator("=")

stock_data = {}
for symbol in STOCKS:
    print(f"Downloading trades data for {symbol}...")
    data = download_stock_data_vwap(symbol, start_date, end_date)
    if data is not None:
        stock_data[symbol] = data
        close_data = data.get('Close')
        print(f"✓ {symbol}: {len(close_data)} 4-second OHLCV bars generated from trades")
    else:
        print(f"✗ {symbol}: Unable to get trades data")

print(f"\n✅ Trades data processed successfully for {len(stock_data)} stocks")
print(f"🔄 Converted to 4-second OHLCV bars for VWAP analysis")

optimization_results = []

for symbol, data in stock_data.items():
    print_separator("=")
    print(f"OPTIMIZING {symbol}".center(get_terminal_width()))
    print_separator("=")
    
    best_params, all_results = optimize_vwap_for_stock(data, symbol)
    
    if best_params:
        optimization_results.append(best_params)
        
        print(f"\n🏆 BEST CONFIGURATION FOR {symbol}:")
        print(f"   Entry Threshold: {best_params['entry_threshold']*100:.1f}%")
        print(f"   Exit Threshold: {best_params['exit_threshold']*100:.1f}%")
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
            sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)[:5]
            print(f"\n📊 TOP 5 CONFIGURATIONS FOR {symbol}:")
            width = get_terminal_width()
            print(f"{'Rank':<5} {'Entry%':<8} {'Exit%':<7} {'Return':<10} {'Profit':<12} {'Trades':<8}")
            print_separator()
            for i, result in enumerate(sorted_results, 1):
                print(f"{i:<5} {result['entry_threshold']*100:<7.1f} {result['exit_threshold']*100:<6.1f} {result['total_return']:<9.2%} ${result['profit']:<10,.0f} {result['num_trades']:<8}")
        
        print(f"\n📈 Generating graph for {symbol}...")
        plot_best_strategy(data, symbol, best_params)
        
        show_detailed_trades([best_params])
        
    else:
        print(f"❌ No valid configurations found for {symbol}")

print_header("FINAL SUMMARY - BEST CONFIGURATIONS BY STOCK")

if optimization_results:
    optimization_results.sort(key=lambda x: x['total_return'], reverse=True)
    
    print(f"{'Stock':<8} {'Entry%':<8} {'Exit%':<7} {'Return':<10} {'Profit':<12} {'Sharpe':<8} {'Trades':<8}")
    print_separator()
    
    for result in optimization_results:
        print(f"{result['symbol']:<8} {result['entry_threshold']*100:<7.1f} {result['exit_threshold']*100:<6.1f} {result['total_return']:<9.2%} ${result['profit']:<10,.0f} {result['sharpe_ratio']:<7.2f} {result['num_trades']:<8}")
    
    best_overall = optimization_results[0]
    print(f"\n🥇 BEST OVERALL STOCK: {best_overall['symbol']}")
    print(f"   Configuration: Entry={best_overall['entry_threshold']*100:.1f}%, Exit={best_overall['exit_threshold']*100:.1f}%")
    print(f"   Return: {best_overall['total_return']:.2%}")
    print(f"   Profit: ${best_overall['profit']:.2f}")
    print(f"   Take Profit: {best_overall['take_profit']:.2%}")
    print(f"   Stop Loss: {best_overall['stop_loss']:.2%}")
    
    profitable_stocks = [r for r in optimization_results if r['total_return'] > 0]
    print(f"\n📈 Profitable stocks: {len(profitable_stocks)}/{len(optimization_results)}")
    
    if profitable_stocks:
        avg_return = sum(r['total_return'] for r in profitable_stocks) / len(profitable_stocks)
        avg_profit = sum(r['profit'] for r in profitable_stocks) / len(profitable_stocks)
        total_profit = sum(r['profit'] for r in profitable_stocks)
        print(f"📊 Average return (profitable stocks): {avg_return:.2%}")
        print(f"💰 Average profit (profitable stocks): ${avg_profit:,.2f}")
        print(f"💎 Total combined profit: ${total_profit:,.2f}")

    show_detailed_trades(optimization_results)

else:
    print("❌ No valid configurations found for any stock")

print_header("OPTIMIZATION COMPLETED")
