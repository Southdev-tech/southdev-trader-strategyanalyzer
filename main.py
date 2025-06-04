import vectorbtpro as vbt
import datetime
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
warnings.filterwarnings('ignore')

from rsi_implementation import (
    download_stock_data as download_rsi_data,
    optimize_rsi_for_stock,
    test_rsi_strategy_on_stock,
    plot_best_strategy as plot_rsi_strategy
)

from vwap_implementation import (
    download_stock_data as download_vwap_data,
    optimize_vwap_for_stock,
    test_vwap_strategy_on_stock,
    plot_best_strategy as plot_vwap_strategy
)

STOCKS = ['EVLV', 'MSFT', 'GOOGL', 'AMZN', 'SPY']

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=30)

vbt.AlpacaData.set_custom_settings(
    client_config=dict(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY")
    )
)

def compare_strategies_for_symbol(symbol, rsi_data, vwap_data):
    """
    Compare RSI and VWAP strategies for a specific symbol and return the best one.
    """
    print(f"\n🔄 Comparando estrategias para {symbol}...")
    
    print(f"   📈 Optimizando RSI para {symbol}...")
    rsi_best_params, rsi_all_results = optimize_rsi_for_stock(rsi_data, symbol)
    
    print(f"   📊 Optimizando VWAP para {symbol}...")
    vwap_best_params, vwap_all_results = optimize_vwap_for_stock(vwap_data, symbol)
    
    comparison_result = {
        'symbol': symbol,
        'rsi_best': rsi_best_params,
        'vwap_best': vwap_best_params,
        'rsi_all_results': rsi_all_results,
        'vwap_all_results': vwap_all_results
    }
    
    if rsi_best_params and vwap_best_params:
        rsi_return = rsi_best_params['total_return']
        vwap_return = vwap_best_params['total_return']
        
        if rsi_return > vwap_return:
            comparison_result['winner'] = 'RSI'
            comparison_result['best_strategy'] = rsi_best_params
            comparison_result['winner_return'] = rsi_return
            comparison_result['loser_return'] = vwap_return
        else:
            comparison_result['winner'] = 'VWAP'
            comparison_result['best_strategy'] = vwap_best_params
            comparison_result['winner_return'] = vwap_return
            comparison_result['loser_return'] = rsi_return
            
    elif rsi_best_params:
        comparison_result['winner'] = 'RSI'
        comparison_result['best_strategy'] = rsi_best_params
        comparison_result['winner_return'] = rsi_best_params['total_return']
        comparison_result['loser_return'] = None
        
    elif vwap_best_params:
        comparison_result['winner'] = 'VWAP'
        comparison_result['best_strategy'] = vwap_best_params
        comparison_result['winner_return'] = vwap_best_params['total_return']
        comparison_result['loser_return'] = None
        
    else:
        comparison_result['winner'] = 'NONE'
        comparison_result['best_strategy'] = None
        comparison_result['winner_return'] = None
        comparison_result['loser_return'] = None
    
    return comparison_result

def plot_strategy_comparison(symbol, comparison_result, rsi_data, vwap_data):
    """
    Create a comprehensive comparison plot for both strategies.
    """
    try:
        if comparison_result['winner'] == 'NONE':
            print(f"❌ No hay estrategias válidas para graficar en {symbol}")
            return
            
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            subplot_titles=(
                f'{symbol} - RSI Strategy', 
                f'{symbol} - VWAP Strategy',
                'RSI Portfolio Performance', 
                'VWAP Portfolio Performance'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # RSI Strategy Plot
        if comparison_result['rsi_best']:
            rsi_params = comparison_result['rsi_best']
            close_price = rsi_data
            rsi = vbt.RSI.run(close_price, window=rsi_params['rsi_window'])
            rsi_entries = rsi.rsi_crossed_below(rsi_params['entry_level'])
            rsi_exits = rsi.rsi_crossed_above(rsi_params['exit_level'])
            
            rsi_pf = vbt.Portfolio.from_signals(
                close_price, rsi_entries, rsi_exits, init_cash=100000,
                sl_stop=0.05, tp_stop=0.03, accumulate=False
            )
            
            # Price and signals
            fig.add_trace(
                go.Scatter(x=close_price.index, y=close_price.values,
                          mode='lines', name='Price', line=dict(color='blue')),
                row=1, col=1
            )
            
            rsi_entry_prices = close_price[rsi_entries]
            if len(rsi_entry_prices) > 0:
                fig.add_trace(
                    go.Scatter(x=rsi_entry_prices.index, y=rsi_entry_prices.values,
                              mode='markers', name='RSI Buy', 
                              marker=dict(color='green', size=8, symbol='triangle-up')),
                    row=1, col=1
                )
            
            rsi_exit_prices = close_price[rsi_exits]
            if len(rsi_exit_prices) > 0:
                fig.add_trace(
                    go.Scatter(x=rsi_exit_prices.index, y=rsi_exit_prices.values,
                              mode='markers', name='RSI Sell',
                              marker=dict(color='red', size=8, symbol='triangle-down')),
                    row=1, col=1
                )
            
            # RSI Portfolio performance
            fig.add_trace(
                go.Scatter(x=rsi_pf.value.index, y=rsi_pf.value.values,
                          mode='lines', name='RSI Portfolio', line=dict(color='green')),
                row=2, col=1
            )
        
        # VWAP Strategy Plot
        if comparison_result['vwap_best']:
            vwap_params = comparison_result['vwap_best']
            close_price = vwap_data.get('Close')
            
            # Calculate VWAP
            df = vwap_data.get()
            vwap_indicator = vbt.VWAP.run(
                high=df['High'], low=df['Low'], close=df['Close'], 
                volume=df['Volume'], anchor='1D'
            )
            vwap = vwap_indicator.vwap
            
            price_deviation = (close_price - vwap) / vwap
            vwap_entries = price_deviation < -vwap_params['entry_threshold']
            vwap_exits = price_deviation > vwap_params['exit_threshold']
            
            vwap_pf = vbt.Portfolio.from_signals(
                close_price, vwap_entries, vwap_exits, init_cash=100000,
                sl_stop=0.05, tp_stop=0.03, accumulate=False
            )
            
            # Price and signals
            fig.add_trace(
                go.Scatter(x=close_price.index, y=close_price.values,
                          mode='lines', name='Price', line=dict(color='blue')),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=vwap.index, y=vwap.values,
                          mode='lines', name='VWAP', line=dict(color='orange')),
                row=1, col=2
            )
            
            vwap_entry_prices = close_price[vwap_entries]
            if len(vwap_entry_prices) > 0:
                fig.add_trace(
                    go.Scatter(x=vwap_entry_prices.index, y=vwap_entry_prices.values,
                              mode='markers', name='VWAP Buy',
                              marker=dict(color='green', size=8, symbol='triangle-up')),
                    row=1, col=2
                )
            
            vwap_exit_prices = close_price[vwap_exits]
            if len(vwap_exit_prices) > 0:
                fig.add_trace(
                    go.Scatter(x=vwap_exit_prices.index, y=vwap_exit_prices.values,
                              mode='markers', name='VWAP Sell',
                              marker=dict(color='red', size=8, symbol='triangle-down')),
                    row=1, col=2
                )
            
            # VWAP Portfolio performance
            fig.add_trace(
                go.Scatter(x=vwap_pf.value.index, y=vwap_pf.value.values,
                          mode='lines', name='VWAP Portfolio', line=dict(color='orange')),
                row=2, col=2
            )
        
        # Add baseline (buy and hold)
        if comparison_result['rsi_best']:
            baseline_value = 100000 * (close_price / close_price.iloc[0])
            fig.add_trace(
                go.Scatter(x=baseline_value.index, y=baseline_value.values,
                          mode='lines', name='Buy & Hold', 
                          line=dict(color='gray', dash='dash')),
                row=2, col=1
            )
            
        if comparison_result['vwap_best']:
            close_price = vwap_data.get('Close')
            baseline_value = 100000 * (close_price / close_price.iloc[0])
            fig.add_trace(
                go.Scatter(x=baseline_value.index, y=baseline_value.values,
                          mode='lines', name='Buy & Hold',
                          line=dict(color='gray', dash='dash')),
                row=2, col=2
            )
        
        winner = comparison_result['winner']
        winner_return = comparison_result['winner_return']
        
        fig.update_layout(
            title=f'{symbol} - Comparación de Estrategias | 🏆 Ganador: {winner} ({winner_return:.2%})',
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=2)
        fig.update_yaxes(title_text="Portfolio Value", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value", row=2, col=2)
        
        fig.show()
        
    except Exception as e:
        print(f"Error plotting comparison for {symbol}: {e}")

def generate_final_report(all_comparisons):
    """
    Generate a comprehensive final report comparing all strategies.
    """
    print(f"\n" + "="*100)
    print("🏆 REPORTE FINAL - COMPARACIÓN DE ESTRATEGIAS RSI vs VWAP")
    print("="*100)
    
    # Summary table
    print(f"{'Symbol':<8} {'Winner':<8} {'Best Return':<12} {'Best Profit':<15} {'Strategy Details':<40}")
    print("-" * 100)
    
    rsi_wins = 0
    vwap_wins = 0
    total_rsi_profit = 0
    total_vwap_profit = 0
    
    for comparison in all_comparisons:
        symbol = comparison['symbol']
        winner = comparison['winner']
        
        if winner == 'NONE':
            print(f"{symbol:<8} {'NONE':<8} {'N/A':<12} {'N/A':<15} {'No valid strategies found':<40}")
            continue
            
        best_strategy = comparison['best_strategy']
        winner_return = comparison['winner_return']
        profit = best_strategy['profit']
        
        if winner == 'RSI':
            rsi_wins += 1
            total_rsi_profit += profit
            details = f"RSI(w={best_strategy['rsi_window']}, e={best_strategy['entry_level']}, x={best_strategy['exit_level']})"
        else:
            vwap_wins += 1
            total_vwap_profit += profit
            details = f"VWAP(e={best_strategy['entry_threshold']*100:.1f}%, x={best_strategy['exit_threshold']*100:.1f}%)"
        
        print(f"{symbol:<8} {winner:<8} {winner_return:<11.2%} ${profit:<13,.0f} {details:<40}")
    
    print("-" * 100)
    print(f"\n📊 RESUMEN GENERAL:")
    print(f"   🥇 RSI Wins: {rsi_wins}/{len(all_comparisons)} stocks")
    print(f"   🥈 VWAP Wins: {vwap_wins}/{len(all_comparisons)} stocks")
    
    if rsi_wins > 0:
        avg_rsi_profit = total_rsi_profit / rsi_wins
        print(f"   💰 RSI Average Profit: ${avg_rsi_profit:,.2f}")
        print(f"   💎 RSI Total Profit: ${total_rsi_profit:,.2f}")
    
    if vwap_wins > 0:
        avg_vwap_profit = total_vwap_profit / vwap_wins
        print(f"   💰 VWAP Average Profit: ${avg_vwap_profit:,.2f}")
        print(f"   💎 VWAP Total Profit: ${total_vwap_profit:,.2f}")
    
    # Best performing stocks
    profitable_comparisons = [c for c in all_comparisons if c['winner'] != 'NONE' and c['winner_return'] > 0]
    
    if profitable_comparisons:
        profitable_comparisons.sort(key=lambda x: x['winner_return'], reverse=True)
        
        print(f"\n🌟 TOP 3 MEJORES CONFIGURACIONES:")
        for i, comp in enumerate(profitable_comparisons[:3], 1):
            symbol = comp['symbol']
            winner = comp['winner']
            return_pct = comp['winner_return']
            profit = comp['best_strategy']['profit']
            
            print(f"   {i}. {symbol} - {winner}: {return_pct:.2%} (${profit:,.0f})")
    
    # Strategy recommendations
    print(f"\n🎯 RECOMENDACIONES:")
    if rsi_wins > vwap_wins:
        print(f"   ✅ RSI es la estrategia dominante ({rsi_wins}/{len(all_comparisons)} stocks)")
        print(f"   📈 Considerar usar RSI como estrategia principal")
    elif vwap_wins > rsi_wins:
        print(f"   ✅ VWAP es la estrategia dominante ({vwap_wins}/{len(all_comparisons)} stocks)")
        print(f"   📈 Considerar usar VWAP como estrategia principal")
    else:
        print(f"   ⚖️  Empate entre estrategias - usar análisis por stock individual")
    
    print(f"\n" + "="*100)
    print("ANÁLISIS COMPLETADO")
    print("="*100)

# Main execution
print("="*100)
print("🚀 COMPARACIÓN COMPLETA DE ESTRATEGIAS: RSI vs VWAP")
print("="*100)
print("📊 Descargando datos y ejecutando ambas estrategias para comparación...")
print("="*100)

# Download data for both strategies
print("\n📥 Descargando datos...")
rsi_stock_data = {}
vwap_stock_data = {}

for symbol in STOCKS:
    print(f"   Descargando {symbol}...")
    
    # Download for RSI (1-minute data)
    rsi_data = download_rsi_data(symbol, start_date, end_date)
    if rsi_data is not None:
        rsi_stock_data[symbol] = rsi_data
        print(f"   ✓ RSI {symbol}: {len(rsi_data)} puntos de datos")
    
    # Download for VWAP (4-second resampled data)
    vwap_data = download_vwap_data(symbol, start_date, end_date)
    if vwap_data is not None:
        vwap_stock_data[symbol] = vwap_data
        close_data = vwap_data.get('Close')
        print(f"   ✓ VWAP {symbol}: {len(close_data)} puntos de datos (4s)")

print(f"\n✅ Datos descargados: RSI ({len(rsi_stock_data)} stocks), VWAP ({len(vwap_stock_data)} stocks)")

# Compare strategies for each symbol
all_comparisons = []

for symbol in STOCKS:
    if symbol in rsi_stock_data and symbol in vwap_stock_data:
        print(f"\n" + "="*80)
        print(f"🔍 ANALIZANDO {symbol}")
        print("="*80)
        
        comparison = compare_strategies_for_symbol(
            symbol, 
            rsi_stock_data[symbol], 
            vwap_stock_data[symbol]
        )
        
        all_comparisons.append(comparison)
        
        # Print individual results
        if comparison['winner'] != 'NONE':
            winner = comparison['winner']
            winner_return = comparison['winner_return']
            loser_return = comparison['loser_return']
            best_strategy = comparison['best_strategy']
            
            print(f"\n🏆 RESULTADO PARA {symbol}:")
            print(f"   Ganador: {winner}")
            print(f"   Mejor Return: {winner_return:.2%}")
            if loser_return is not None:
                print(f"   Diferencia: {winner_return - loser_return:.2%}")
            print(f"   Ganancia: ${best_strategy['profit']:,.2f}")
            print(f"   Trades: {best_strategy['num_trades']}")
            
            # Generate comparison plot
            print(f"\n📈 Generando gráfico comparativo para {symbol}...")
            plot_strategy_comparison(symbol, comparison, rsi_stock_data[symbol], vwap_stock_data[symbol])
        else:
            print(f"\n❌ No se encontraron estrategias válidas para {symbol}")
    else:
        print(f"\n⚠️  Datos insuficientes para {symbol}")

# Generate final comprehensive report
generate_final_report(all_comparisons) 