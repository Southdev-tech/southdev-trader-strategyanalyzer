import vectorbtpro as vbt
import datetime
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from data.alpaca_client import download_stock_data_both
from strategies.rsi_implementation import (
    test_rsi_strategy_on_stock,
    optimize_rsi_for_stock
)
from strategies.vwap_implementation import (
    test_vwap_strategy_on_stock,
    optimize_vwap_for_stock
)

warnings.filterwarnings('ignore')

STOCKS = ['EVLV', 'MSFT', 'GOOGL', 'AMZN', 'SPY']

TIMEFRAMES = {
    "1d": 1,
    "1w": 7,
    "1m": 22,
    "1y": 252,
    "5y": 1260
}

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=2)

vbt.AlpacaData.set_custom_settings(
    client_config=dict(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY")
    )
)

def get_timeframe_slice(df, days):
    if len(df) < days:
        return df
    return df.iloc[-days:]

def multi_timeframe_backtest(strategy_func, data, symbol, param_grid, is_vwap=False):
    results = []
    for params in param_grid:
        metrics_per_tf = {}
        for tf_name, tf_days in TIMEFRAMES.items():
            if is_vwap:
                df = data.get()
                sliced_df = df.iloc[-tf_days:] if len(df) > tf_days else df
                class Wrapper:
                    def __init__(self, df):
                        self.df = df
                    def get(self, col=None):
                        if col is None:
                            return self.df
                        return self.df[col]
                sliced_data = Wrapper(sliced_df)
            else:
                sliced_data = data.iloc[-tf_days:] if len(data) > tf_days else data
            try:
                res = strategy_func(sliced_data, symbol, **params)
            except Exception as e:
                res = None
            if res and 'portfolio' in res:
                stats = res['portfolio'].stats()
                metrics_per_tf[tf_name] = {
                    "total_return": stats.get('Total Return [%]', 0) / 100,
                    "sharpe_ratio": stats.get('Sharpe Ratio', 0),
                    "max_drawdown": stats.get('Max Drawdown [%]', 0) / 100,
                    "calmar_ratio": stats.get('Calmar Ratio', 0),
                    "win_rate": stats.get('Win Rate [%]', 0) / 100,
                    "sortino_ratio": stats.get('Sortino Ratio', 0)
                }
            else:
                metrics_per_tf[tf_name] = {
                    "total_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "calmar_ratio": 0,
                    "win_rate": 0,
                    "sortino_ratio": 0
                }
        avg_metrics = {k: np.mean([v[k] for v in metrics_per_tf.values()]) for k in ["total_return", "sharpe_ratio", "max_drawdown", "calmar_ratio", "win_rate", "sortino_ratio"]}
        results.append({"params": params, "metrics": metrics_per_tf, "avg_metrics": avg_metrics})
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
                    grid.append({"rsi_window": rsi_window, "entry_level": entry_level, "exit_level": exit_level})
    return grid

def get_vwap_param_grid(symbol):
    if symbol == 'SPY':
        entry_thresholds = np.arange(0.0005, 0.0025, 0.0003)
        exit_thresholds = np.arange(0.0002, 0.0018, 0.0003)
    else:
        entry_thresholds = np.arange(0.002, 0.014, 0.0005)
        exit_thresholds = np.arange(0.001, 0.009, 0.0005)
    grid = []
    for entry_threshold in entry_thresholds:
        for exit_threshold in exit_thresholds:
            if entry_threshold > exit_threshold:
                grid.append({"entry_threshold": entry_threshold, "exit_threshold": exit_threshold})
    return grid

def download_symbol_data(symbol, start_date, end_date):
    """Descarga datos para un símbolo específico - SÚPER OPTIMIZADO"""
    print(f"   🔄 Descargando {symbol}...")
    
    try:
        rsi_data, vwap_data = download_stock_data_both(symbol, start_date, end_date)
        return symbol, rsi_data, vwap_data
        
    except Exception as e:
        print(f"   ❌ Error descargando {symbol}: {e}")
        return symbol, None, None

def analyze_symbol_individual(symbol, rsi_data_dict, vwap_data_dict):
    """Analiza un símbolo específico en la Fase 1"""
    if symbol not in rsi_data_dict or symbol not in vwap_data_dict:
        return None
    
    print(f"   🔍 Analizando {symbol}...")
    
    try:
        best_rsi_result = optimize_rsi_for_stock(rsi_data_dict[symbol], symbol)
        
        best_vwap_result = optimize_vwap_for_stock(vwap_data_dict[symbol], symbol)
        
        rsi_return = best_rsi_result['total_return'] if best_rsi_result else 0
        vwap_return = best_vwap_result['total_return'] if best_vwap_result else 0
        winner = 'VWAP' if vwap_return > rsi_return else 'RSI'
        
        result = {
            'symbol': symbol,
            'rsi': best_rsi_result,
            'vwap': best_vwap_result,
            'winner': winner
        }
        
        print(f"   ✓ {symbol}: {winner} ganó ({rsi_return:.2%} RSI vs {vwap_return:.2%} VWAP)")
        return result
        
    except Exception as e:
        print(f"   ❌ Error analizando {symbol}: {e}")
        return None

def analyze_symbol_multiframe(symbol, rsi_data_dict, vwap_data_dict):
    """Analiza un símbolo específico en multi-timeframe"""
    if symbol not in rsi_data_dict or symbol not in vwap_data_dict:
        return None
    
    print(f"   🔍 Multi-timeframe {symbol}...")
    
    try:
        # RSI
        rsi_param_grid = get_rsi_param_grid()
        rsi_results = multi_timeframe_backtest(test_rsi_strategy_on_stock, rsi_data_dict[symbol], symbol, rsi_param_grid, is_vwap=False)
        best_rsi = max(rsi_results, key=lambda x: x['avg_metrics']['total_return'])
        
        # VWAP
        vwap_param_grid = get_vwap_param_grid(symbol)
        vwap_results = multi_timeframe_backtest(test_vwap_strategy_on_stock, vwap_data_dict[symbol], symbol, vwap_param_grid, is_vwap=True)
        best_vwap = max(vwap_results, key=lambda x: x['avg_metrics']['total_return'])
        
        # Comparación
        winner = 'RSI' if best_rsi['avg_metrics']['total_return'] > best_vwap['avg_metrics']['total_return'] else 'VWAP'
        
        result = {
            'symbol': symbol,
            'rsi': best_rsi,
            'vwap': best_vwap,
            'winner': winner
        }
        
        print(f"   ✓ Multi-timeframe {symbol}: {winner} ganó")
        return result
        
    except Exception as e:
        print(f"   ❌ Error multi-timeframe {symbol}: {e}")
        return None

print("="*100)
print("📥 Descargando datos históricos (5 años) - PARALELO")
print("="*100)

rsi_data = {}
vwap_data = {}

with ThreadPoolExecutor(max_workers=3) as executor:  # 3 workers para no saturar API
    future_to_symbol = {
        executor.submit(download_symbol_data, symbol, start_date, end_date): symbol 
        for symbol in STOCKS
    }
    
    for future in as_completed(future_to_symbol):
        symbol = future_to_symbol[future]
        try:
            symbol_result, rsi_result, vwap_result = future.result()
            if rsi_result is not None:
                rsi_data[symbol] = rsi_result
            if vwap_result is not None:
                vwap_data[symbol] = vwap_result
        except Exception as e:
            print(f"   ❌ Error descargando {symbol}: {e}")

print(f"\n✅ Descarga completada: RSI ({len(rsi_data)}), VWAP ({len(vwap_data)})")

print("\n" + "="*100)
print("🔍 FASE 1: OPTIMIZACIÓN INDIVIDUAL POR SÍMBOLO - PARALELO")
print("="*100)

individual_results = []

with ThreadPoolExecutor(max_workers=2) as executor:  # 2 workers para análisis
    future_to_symbol = {
        executor.submit(analyze_symbol_individual, symbol, rsi_data, vwap_data): symbol 
        for symbol in STOCKS if symbol in rsi_data and symbol in vwap_data
    }
    
    # Recoger resultados
    for future in as_completed(future_to_symbol):
        symbol = future_to_symbol[future]
        try:
            result = future.result()
            if result:
                individual_results.append(result)
        except Exception as e:
            print(f"   ❌ Error en análisis individual {symbol}: {e}")

print(f"\n{'='*100}")
print("📊 REPORTE FASE 1: OPTIMIZACIÓN INDIVIDUAL")
print("="*100)
print(f"{'Symbol':<8} {'Winner':<8} {'RSI Return':<12} {'VWAP Return':<12} {'RSI Sharpe':<12} {'VWAP Sharpe':<12}")
print("-"*70)

for res in individual_results:
    rsi_ret = res['rsi']['total_return'] if res['rsi'] else 0
    vwap_ret = res['vwap']['total_return'] if res['vwap'] else 0
    rsi_sharpe = res['rsi']['sharpe_ratio'] if res['rsi'] else 0
    vwap_sharpe = res['vwap']['sharpe_ratio'] if res['vwap'] else 0
    print(f"{res['symbol']:<8} {res['winner']:<8} {rsi_ret:<11.2%} {vwap_ret:<11.2%} {rsi_sharpe:<11.2f} {vwap_sharpe:<11.2f}")

print("\n" + "="*100)
print("🕒 FASE 2: ANÁLISIS MULTI-TIMEFRAME - PARALELO")
print("="*100)

all_results = []

with ThreadPoolExecutor(max_workers=2) as executor:  # 2 workers para multi-timeframe
    future_to_symbol = {
        executor.submit(analyze_symbol_multiframe, symbol, rsi_data, vwap_data): symbol 
        for symbol in STOCKS if symbol in rsi_data and symbol in vwap_data
    }
    
    # Recoger resultados
    for future in as_completed(future_to_symbol):
        symbol = future_to_symbol[future]
        try:
            result = future.result()
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"   ❌ Error en multi-timeframe {symbol}: {e}")

for res in all_results:
    print(f"\n🏆 RESULTADO MULTI-TIMEFRAME {res['symbol']}:")
    print(f"   Ganador: {res['winner']}")
    print(f"   RSI Avg Return: {res['rsi']['avg_metrics']['total_return']:.2%}")
    print(f"   VWAP Avg Return: {res['vwap']['avg_metrics']['total_return']:.2%}")
    print(f"   RSI Avg Sharpe: {res['rsi']['avg_metrics']['sharpe_ratio']:.2f}")
    print(f"   VWAP Avg Sharpe: {res['vwap']['avg_metrics']['sharpe_ratio']:.2f}")
    print(f"   RSI Avg Calmar: {res['rsi']['avg_metrics']['calmar_ratio']:.2f}")
    print(f"   VWAP Avg Calmar: {res['vwap']['avg_metrics']['calmar_ratio']:.2f}")
    print(f"   RSI Avg Sortino: {res['rsi']['avg_metrics']['sortino_ratio']:.2f}")
    print(f"   VWAP Avg Sortino: {res['vwap']['avg_metrics']['sortino_ratio']:.2f}")
    print(f"   RSI Avg Win Rate: {res['rsi']['avg_metrics']['win_rate']:.2%}")
    print(f"   VWAP Avg Win Rate: {res['vwap']['avg_metrics']['win_rate']:.2%}")
    print(f"   RSI Avg Max DD: {res['rsi']['avg_metrics']['max_drawdown']:.2%}")
    print(f"   VWAP Avg Max DD: {res['vwap']['avg_metrics']['max_drawdown']:.2%}")

print("\n" + "="*100)
print("🏆 REPORTE FINAL MULTI-TIMEFRAME")
print("="*100)
print(f"{'Symbol':<8} {'Winner':<8} {'RSI Ret':<10} {'VWAP Ret':<10} {'RSI Sharpe':<10} {'VWAP Sharpe':<10} {'RSI Calmar':<10} {'VWAP Calmar':<10} {'RSI Sortino':<10} {'VWAP Sortino':<10} {'RSI Win':<10} {'VWAP Win':<10}")
print("-"*140)
for res in all_results:
    print(f"{res['symbol']:<8} {res['winner']:<8} {res['rsi']['avg_metrics']['total_return']:<9.2%} {res['vwap']['avg_metrics']['total_return']:<9.2%} {res['rsi']['avg_metrics']['sharpe_ratio']:<10.2f} {res['vwap']['avg_metrics']['sharpe_ratio']:<10.2f} {res['rsi']['avg_metrics']['calmar_ratio']:<10.2f} {res['vwap']['avg_metrics']['calmar_ratio']:<10.2f} {res['rsi']['avg_metrics']['sortino_ratio']:<10.2f} {res['vwap']['avg_metrics']['sortino_ratio']:<10.2f} {res['rsi']['avg_metrics']['win_rate']:<9.2%} {res['vwap']['avg_metrics']['win_rate']:<9.2%}")
print("="*100)
print("🎯 ANÁLISIS MULTI-TIMEFRAME COMPLETADO")
print("="*100) 