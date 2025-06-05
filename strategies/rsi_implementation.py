import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import vectorbtpro as vbt
import datetime
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.alpaca_client import download_stock_data_rsi
warnings.filterwarnings('ignore')
import exchange_calendars as ecals


STOCKS = ['SPY']

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=1)

vbt.AlpacaData.set_custom_settings(
    client_config=dict(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY")
    )
)

def test_rsi_strategy_on_stock(price_data, symbol, rsi_window=20, entry_level=30, exit_level=75):
    try:
        rsi = vbt.RSI.run(price_data, window=rsi_window)
        
        # Pick your exchange, e.g. New York Stock Exchange
        calendar = ecals.get_calendar("XNYS")

        # Build valid trading sessions for your data range
        sessions = calendar.sessions_in_range(
            price_data.index.min().date(),
            price_data.index.max().date()
        )

        # DEBUG: Let's check each mask component
        print(f"DEBUG - Price data index range: {price_data.index.min()} to {price_data.index.max()}")
        print(f"DEBUG - Sessions range: {sessions.min()} to {sessions.max()}")
        
        # Fix holiday mask - convert sessions to date format that matches index
        sessions_dates = pd.to_datetime(sessions).date
        index_dates = price_data.index.date
        holiday_mask = pd.Series([d in sessions_dates for d in index_dates], index=price_data.index)
        
        print(f"DEBUG - Holiday mask True count: {holiday_mask.sum()}/{len(holiday_mask)}")

        # Create time mask - using between_time to get boolean mask
        time_mask = price_data.index.indexer_between_time("9:30", "16:00")
        # Convert indexer to boolean mask
        time_bool_mask = pd.Series(False, index=price_data.index)
        if len(time_mask) > 0:
            time_bool_mask.iloc[time_mask] = True
        
        print(f"DEBUG - Time mask True count: {time_bool_mask.sum()}/{len(time_bool_mask)}")
        
        weekday_mask = price_data.index.weekday.isin([0, 1, 2, 3, 4])
        print(f"DEBUG - Weekday mask True count: {weekday_mask.sum()}/{len(weekday_mask)}")

        final_mask = time_bool_mask & weekday_mask & holiday_mask
        print(f"DEBUG - Final mask True count: {final_mask.sum()}/{len(final_mask)}")
        
        # If no valid trading times, skip time filtering for now
        if final_mask.sum() == 0:
            print(f"WARNING: No valid trading times found, using weekday filter only")
            final_mask = weekday_mask & holiday_mask
            print(f"DEBUG - Simplified final mask True count: {final_mask.sum()}/{len(final_mask)}")

        entries = rsi.rsi_crossed_below(entry_level) & final_mask
        exits = rsi.rsi_crossed_above(exit_level) & final_mask

        if not entries.any() or not exits.any():
            return None

        pf = vbt.Portfolio.from_signals(
            price_data,
            entries,
            exits,
            init_cash=20000,
            sl_stop=0.05,
            tp_stop=0.03,
            accumulate=False,
        )

        try:
            stats = pf.stats()
            
            total_return = stats['Total Return [%]'] / 100 if 'Total Return [%]' in stats else pf.total_return
            sharpe_ratio = stats['Sharpe Ratio'] if 'Sharpe Ratio' in stats else 0
            max_drawdown = stats['Max Drawdown [%]'] / 100 if 'Max Drawdown [%]' in stats else 0
            num_trades = stats['Total Trades'] if 'Total Trades' in stats else 0
            win_rate = stats['Win Rate [%]'] / 100 if 'Win Rate [%]' in stats else 0
            
            initial_capital = pf.init_cash
            final_capital = pf.value.iloc[-1]
            profit = final_capital - initial_capital
            
        except Exception as e:
            print(f"Error getting stats for {symbol}: {e}")
            total_return = (pf.value.iloc[-1] / pf.value.iloc[0]) - 1
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
            num_trades = 0
            initial_capital = 100000
            final_capital = pf.value.iloc[-1] if len(pf.value) > 0 else 100000
            profit = final_capital - initial_capital

        return {
            'symbol': symbol,
            'rsi_window': rsi_window,
            'entry_level': entry_level,
            'exit_level': exit_level,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio if not pd.isna(sharpe_ratio) else 0,
            'max_drawdown': max_drawdown if not pd.isna(max_drawdown) else 0,
            'win_rate': win_rate if not pd.isna(win_rate) else 0,
            'num_trades': num_trades,
            'num_entries': entries.sum(),
            'num_exits': exits.sum(),
            'data_points': len(price_data),
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'profit': profit,
            'portfolio': pf  # Store the portfolio for trade details
        }
    except Exception as e:
        print(f"Error testing RSI strategy on {symbol}: {e}")
        return None

def optimize_rsi_for_stock(price_data, symbol):
    print(f"\n🔍 Optimizando parámetros RSI para {symbol}...")

    rsi_windows = np.arange(10, 25, 2)      # RSI windows: 10, 12, 14, 16, 18, 20, 22, 24
    entry_levels = np.arange(20, 40, 5)     # Entry levels: 20, 25, 30, 35
    exit_levels = np.arange(60, 85, 5)      # Exit levels: 60, 65, 70, 75, 80

    best_return = -np.inf
    best_params = None
    all_results = []

    total_combinations = len(rsi_windows) * len(entry_levels) * len(exit_levels)
    current_combination = 0

    for rsi_window in rsi_windows:
        for entry_level in entry_levels:
            for exit_level in exit_levels:
                current_combination += 1

                if entry_level >= exit_level:
                    continue

                result = test_rsi_strategy_on_stock(price_data, symbol, rsi_window, entry_level, exit_level)

                if result:
                    all_results.append(result)
                    if result['total_return'] > best_return:
                        best_return = result['total_return']
                        best_params = result.copy()

                if current_combination % 20 == 0:
                    print(f"   Progreso: {current_combination}/{total_combinations} combinaciones probadas...")

    print(f"   ✅ Completado: {len(all_results)} configuraciones válidas probadas")
    return best_params, all_results

def plot_best_strategy(price_data, symbol, best_params):
    try:
        rsi = vbt.RSI.run(price_data, window=best_params['rsi_window'])
        entries = rsi.rsi_crossed_below(best_params['entry_level'])
        exits = rsi.rsi_crossed_above(best_params['exit_level'])

        pf = vbt.Portfolio.from_signals(
            price_data,
            entries,
            exits,
            init_cash=100000,
            sl_stop=0.05,
            tp_stop=0.03,
            accumulate=False
        )

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} - Mejor Estrategia RSI', 'RSI Indicator', 'Portfolio Value'),
            row_heights=[0.5, 0.25, 0.25]
        )

        fig.add_trace(
            go.Scatter(x=price_data.index, y=price_data.values,
                      mode='lines', name='Price', line=dict(color='blue')),
            row=1, col=1
        )

        entry_prices = price_data[entries]
        if len(entry_prices) > 0:
            fig.add_trace(
                go.Scatter(x=entry_prices.index, y=entry_prices.values,
                          mode='markers', name='Buy Signal',
                          marker=dict(color='green', size=10, symbol='triangle-up')),
                row=1, col=1
            )

        exit_prices = price_data[exits]
        if len(exit_prices) > 0:
            fig.add_trace(
                go.Scatter(x=exit_prices.index, y=exit_prices.values,
                          mode='markers', name='Sell Signal',
                          marker=dict(color='red', size=10, symbol='triangle-down')),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(x=rsi.rsi.index, y=rsi.rsi.values,
                      mode='lines', name='RSI', line=dict(color='purple')),
            row=2, col=1
        )

        fig.add_hline(y=best_params['entry_level'], line_dash="dash", line_color="green",
                     annotation_text=f"Entry ({best_params['entry_level']})", row=2, col=1)
        fig.add_hline(y=best_params['exit_level'], line_dash="dash", line_color="red",
                     annotation_text=f"Exit ({best_params['exit_level']})", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

        portfolio_value = pf.value
        fig.add_trace(
            go.Scatter(x=portfolio_value.index, y=portfolio_value.values,
                      mode='lines', name='Portfolio Value', line=dict(color='orange')),
            row=3, col=1
        )

        fig.update_layout(
            title=f'{symbol} - Mejor Configuración RSI (Window={best_params["rsi_window"]}, Entry={best_params["entry_level"]}, Exit={best_params["exit_level"]})',
            height=800,
            showlegend=True
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Portfolio Value", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        fig.show()

    except Exception as e:
        print(f"Error plotting best strategy for {symbol}: {e}")

def show_detailed_trades(optimization_results):
    """Show detailed trade information for each symbol"""
    print(f"\n" + "="*100)
    print("DETALLES DE TRADES POR SÍMBOLO")
    print("="*100)
    
    for result in optimization_results:
        symbol = result['symbol']
        pf = result['portfolio']
        
        print(f"\n🔍 TRADES DETALLADOS PARA {symbol}")
        print(f"Configuración: RSI Window={result['rsi_window']}, Entry={result['entry_level']}, Exit={result['exit_level']}")
        print("-" * 80)
        
        try:
            trades = pf.trades.records
            
            if len(trades) > 0:
                print(f"📈 Total de trades: {len(trades)}")
                print(f"{'#':<3} {'Tipo':<6} {'Cantidad':<12} {'Fecha Entrada':<20} {'Precio Entrada':<15} {'Fecha Salida':<20} {'Precio Salida':<15} {'PnL':<12} {'Return %':<10}")
                print("-" * 135)
                
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
                        print(f"Error procesando trade {i+1}: {trade_error}")
                        print(f"   Trade data: {trade}")
                            
            else:
                print("❌ No se encontraron trades registrados")
                
                try:
                    print(f"\n📊 Estadísticas del portfolio:")
                    stats = pf.stats()
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                except Exception as stats_error:
                    print(f"Error mostrando estadísticas: {stats_error}")
                    
        except Exception as e:
            print(f"Error mostrando trades para {symbol}: {e}")
            
            try:
                print(f"\n📊 Estadísticas del portfolio para {symbol}:")
                stats = pf.stats()
                for key, value in stats.items():
                    print(f"   {key}: {value}")
            except Exception as stats_error:
                print(f"Error mostrando estadísticas: {stats_error}")
            
        print("\n" + "-" * 80)

print("="*80)
print("OPTIMIZACIÓN DE PARÁMETROS RSI POR STOCK INDIVIDUAL")
print("="*80)

stock_data = {}
for symbol in STOCKS:
    print(f"Descargando datos para {symbol}...")
    data = download_stock_data_rsi(symbol, start_date, end_date)
    if data is not None:
        stock_data[symbol] = data
        print(f"✓ {symbol}: {len(data)} puntos de datos descargados")
    else:
        print(f"✗ {symbol}: No se pudieron obtener datos")

print(f"\n✅ Datos descargados exitosamente para {len(stock_data)} stocks")

optimization_results = []

for symbol, price_data in stock_data.items():
    print(f"\n" + "="*60)
    print(f"OPTIMIZANDO {symbol}")
    print("="*60)

    best_params, all_results = optimize_rsi_for_stock(price_data, symbol)

    if best_params:
        optimization_results.append(best_params)

        print(f"\n🏆 MEJOR CONFIGURACIÓN PARA {symbol}:")
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
        print(f"   Capital Inicial: ${best_params['initial_capital']:,.2f}")
        print(f"   Capital Final: ${best_params['final_capital']:,.2f}")
        print(f"   Ganancia: ${best_params['profit']:,.2f}")

        if len(all_results) > 1:
            sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)[:5]
            print(f"\n📊 TOP 5 CONFIGURACIONES PARA {symbol}:")
            print(f"{'Rank':<5} {'RSI':<5} {'Entry':<7} {'Exit':<6} {'Return':<10} {'Profit':<12} {'Trades':<8}")
            print("-" * 65)
            for i, result in enumerate(sorted_results, 1):
                print(f"{i:<5} {result['rsi_window']:<5} {result['entry_level']:<7} {result['exit_level']:<6} {result['total_return']:<9.2%} ${result['profit']:<10,.0f} {result['num_trades']:<8}")

        print(f"\n📈 Generando gráfico para {symbol}...")
        plot_best_strategy(price_data, symbol, best_params)
        
        show_detailed_trades([best_params])

    else:
        print(f"❌ No se encontraron configuraciones válidas para {symbol}")

print(f"\n" + "="*80)
print("RESUMEN FINAL - MEJORES CONFIGURACIONES POR STOCK")
print("="*80)

if optimization_results:
    optimization_results.sort(key=lambda x: x['total_return'], reverse=True)

    print(f"{'Stock':<8} {'RSI':<5} {'Entry':<7} {'Exit':<6} {'Return':<10} {'Profit':<12} {'Sharpe':<8} {'Trades':<8}")
    print("-" * 80)

    for result in optimization_results:
        print(f"{result['symbol']:<8} {result['rsi_window']:<5} {result['entry_level']:<7} {result['exit_level']:<6} {result['total_return']:<9.2%} ${result['profit']:<10,.0f} {result['sharpe_ratio']:<7.2f} {result['num_trades']:<8}")

    best_overall = optimization_results[0]
    print(f"\n🥇 MEJOR STOCK GENERAL: {best_overall['symbol']}")
    print(f"   Configuración: RSI={best_overall['rsi_window']}, Entry={best_overall['entry_level']}, Exit={best_overall['exit_level']}")
    print(f"   Retorno: {best_overall['total_return']:.2%}")
    print(f"   Ganancia: ${best_overall['profit']:.2f}")

    profitable_stocks = [r for r in optimization_results if r['total_return'] > 0]
    print(f"\n📈 Stocks rentables: {len(profitable_stocks)}/{len(optimization_results)}")

    if profitable_stocks:
        avg_return = sum(r['total_return'] for r in profitable_stocks) / len(profitable_stocks)
        avg_profit = sum(r['profit'] for r in profitable_stocks) / len(profitable_stocks)
        total_profit = sum(r['profit'] for r in profitable_stocks)
        print(f"📊 Retorno promedio (stocks rentables): {avg_return:.2%}")
        print(f"💰 Ganancia promedio (stocks rentables): ${avg_profit:,.2f}")
        print(f"💎 Ganancia total combinada: ${total_profit:,.2f}")

    show_detailed_trades(optimization_results)

else:
    print("❌ No se encontraron configuraciones válidas para ningún stock")

print(f"\n" + "="*80)
print("OPTIMIZACIÓN COMPLETADA")
print("="*80)
