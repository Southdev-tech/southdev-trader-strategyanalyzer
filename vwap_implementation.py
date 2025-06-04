import vectorbtpro as vbt
import datetime
import pandas as pd
import numpy as np
import talib
from numba import njit
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')
import os


STOCKS = ['EVLV', 'MSFT', 'GOOGL', 'AMZN', 'SPY']

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=1)
vbt.AlpacaData.set_custom_settings(
    client_config=dict(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY")
    )
)

def download_stock_data(symbol, start_date, end_date):
    try:
        data = vbt.AlpacaData.pull(
            symbol,
            start=start_date,  
            end=end_date,  
            timeframe="1m",
            adjustment="all",
            tz="US/Eastern",
            data_type="trade",
        )
        
        ohlcv_data = convert_trades_to_ohlcv(data, symbol)
        
        if ohlcv_data is not None and len(ohlcv_data.get('Close')) > 50:
            resampled_data = resample_to_4_seconds(ohlcv_data)
            return resampled_data
        else:
            print(f"Insufficient data for {symbol}")
            return None
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None

def convert_trades_to_ohlcv(trade_data, symbol):
    """
    Convert individual trade data to OHLCV format for VWAP calculation.
    Trade data structure: Exchange, Trade price, Trade size, Trade ID, Conditions, Tape
    """
    try:
        df = trade_data.get()
        
        if df is None or len(df) == 0:
            print(f"No trade data available for {symbol}")
            return None
        
        print(f"   📊 Converting {len(df)} trades to OHLCV format for {symbol}")
        print(f"   📋 Trade data columns: {list(df.columns)}")
        
        price_col = None
        volume_col = None
        
        for col in df.columns:
            if 'price' in col.lower() or col.lower() == 'price':
                price_col = col
                break
        
        for col in df.columns:
            if 'size' in col.lower() or 'volume' in col.lower():
                volume_col = col
                break
        
        if price_col is None:
            print(f"❌ Could not find price column in trade data for {symbol}")
            print(f"   Available columns: {list(df.columns)}")
            return None
            
        if volume_col is None:
            print(f"❌ Could not find volume/size column in trade data for {symbol}")
            print(f"   Available columns: {list(df.columns)}")
            return None
        
        print(f"   ✓ Using price column: '{price_col}', volume column: '{volume_col}'")
        
        df_trades = df[[price_col, volume_col]].copy()
        df_trades.columns = ['Price', 'Volume']
        
        ohlcv = df_trades.resample('4S').agg({
            'Price': ['first', 'max', 'min', 'last'],  # Open, High, Low, Close
            'Volume': 'sum'
        }).dropna()
        
        ohlcv.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        ohlcv = ohlcv.fillna(method='ffill').dropna()
        
        if len(ohlcv) == 0:
            print(f"❌ No OHLCV data generated for {symbol}")
            return None
        
        print(f"   ✅ Generated {len(ohlcv)} OHLCV bars from {len(df)} trades")
        
        class OHLCVData:
            def __init__(self, dataframe):
                self.df = dataframe
            
            def get(self, column=None):
                if column is None:
                    return self.df
                else:
                    return self.df[column]
        
        return OHLCVData(ohlcv)
        
    except Exception as e:
        print(f"Error converting trades to OHLCV for {symbol}: {e}")
        return None

def resample_to_4_seconds(data):
    """
    Since we already converted trades to 4-second OHLCV bars, 
    we can return the data as-is or do additional processing if needed.
    """
    try:
        df = data.get()
        
        print(f"   📊 Using {len(df)} pre-generated 4-second OHLCV bars")
        
        class ResampledData:
            def __init__(self, dataframe):
                self.df = dataframe
            
            def get(self, column=None):
                if column is None:
                    return self.df
                else:
                    return self.df[column]
        
        return ResampledData(df)
        
    except Exception as e:
        print(f"Error processing 4-second data: {e}")
        return data

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

def test_vwap_strategy_on_stock(data, symbol, entry_threshold=0.02, exit_threshold=0.015):
    try:
        close_price = data.get('Close')
        vwap = calculate_vwap(data)
        
        if vwap is None:
            return None
        
        price_deviation = (close_price - vwap) / vwap
        
        entries = price_deviation < -entry_threshold
        exits = price_deviation > exit_threshold
        
        if not entries.any() or not exits.any():
            return None
            
        pf = vbt.Portfolio.from_signals(
            close_price,
            entries,
            exits,
            init_cash=100000,
            sl_stop=0.05,
            tp_stop=0.03,
            accumulate=False,
            cash_sharing=True,
            call_seq='auto'
        )
        
        try:
            # Use VectorBT's built-in stats - show all stats in table format
            stats = pf.stats()
            
            # Extract values for optimization comparison
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
            # Fallback calculation
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
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'total_return': total_return,
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

def optimize_vwap_for_stock(data, symbol):
    print(f"\n🔍 Optimizando parámetros VWAP para {symbol} (barras de 4s desde trades)...")
    
    # Use smaller thresholds for 4-second analysis (higher frequency = smaller deviations)
    if symbol == 'SPY':
        # SPY with 4-second analysis - very small thresholds
        entry_thresholds = np.arange(0.0005, 0.003, 0.0005)  # Entry: 0.05%, 0.1%, 0.15%, 0.2%, 0.25%
        exit_thresholds = np.arange(0.0002, 0.002, 0.0002)   # Exit: 0.02%, 0.04%, 0.06%, 0.08%, 0.1%, 0.12%, 0.14%, 0.16%, 0.18%
    else:
        # Regular stocks with 4-second analysis - smaller thresholds than 1-minute
        entry_thresholds = np.arange(0.002, 0.015, 0.002)    # Entry: 0.2%, 0.4%, 0.6%, 0.8%, 1.0%, 1.2%, 1.4%
        exit_thresholds = np.arange(0.001, 0.01, 0.001)      # Exit: 0.1%, 0.2%, 0.3%, ..., 0.9%
    
    best_return = -np.inf
    best_params = None
    all_results = []
    
    total_combinations = len(entry_thresholds) * len(exit_thresholds)
    current_combination = 0
    
    print(f"   🔄 Probando {total_combinations} combinaciones con barras OHLCV de 4s desde trades...")
    
    for entry_threshold in entry_thresholds:
        for exit_threshold in exit_thresholds:
            current_combination += 1
            
            if entry_threshold <= exit_threshold:
                continue
                
            result = test_vwap_strategy_on_stock(data, symbol, entry_threshold, exit_threshold)
            
            if result:
                all_results.append(result)
                
                if result['total_return'] > best_return:
                    best_return = result['total_return']
                    best_params = result.copy()
            
            if current_combination % 10 == 0:
                print(f"   Progreso: {current_combination}/{total_combinations} combinaciones probadas...")
    
    print(f"   ✅ Completado: {len(all_results)} configuraciones válidas probadas")
    
    return best_params, all_results

def plot_best_strategy(data, symbol, best_params):
    try:
        close_price = data.get('Close')
        vwap = calculate_vwap(data)
        
        price_deviation = (close_price - vwap) / vwap
        entries = price_deviation < -best_params['entry_threshold']
        exits = price_deviation > best_params['exit_threshold']
        
        pf = vbt.Portfolio.from_signals(
            close_price,
            entries,
            exits,
            init_cash=100000,
            sl_stop=0.05,
            tp_stop=0.03,
            accumulate=False
        )
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} - Mejor Estrategia VWAP', 'VWAP vs Price', 'Price Deviation from VWAP', 'Portfolio Value'),
            row_heights=[0.4, 0.25, 0.2, 0.15]
        )
        
        fig.add_trace(
            go.Scatter(x=close_price.index, y=close_price.values, 
                      mode='lines', name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        entry_prices = close_price[entries]
        if len(entry_prices) > 0:
            fig.add_trace(
                go.Scatter(x=entry_prices.index, y=entry_prices.values,
                          mode='markers', name='Buy Signal', 
                          marker=dict(color='green', size=10, symbol='triangle-up')),
                row=1, col=1
            )
        
        exit_prices = close_price[exits]
        if len(exit_prices) > 0:
            fig.add_trace(
                go.Scatter(x=exit_prices.index, y=exit_prices.values,
                          mode='markers', name='Sell Signal',
                          marker=dict(color='red', size=10, symbol='triangle-down')),
                row=1, col=1
            )
        
        fig.add_trace(
            go.Scatter(x=close_price.index, y=close_price.values,
                      mode='lines', name='Price', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=vwap.index, y=vwap.values,
                      mode='lines', name='VWAP', line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=price_deviation.index, y=price_deviation.values * 100,
                      mode='lines', name='Price Deviation %', line=dict(color='purple')),
            row=3, col=1
        )
        
        fig.add_hline(y=-best_params['entry_threshold'] * 100, line_dash="dash", line_color="green", 
                     annotation_text=f"Entry ({-best_params['entry_threshold']*100:.1f}%)", row=3, col=1)
        fig.add_hline(y=best_params['exit_threshold'] * 100, line_dash="dash", line_color="red", 
                     annotation_text=f"Exit ({best_params['exit_threshold']*100:.1f}%)", row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
        
        portfolio_value = pf.value
        fig.add_trace(
            go.Scatter(x=portfolio_value.index, y=portfolio_value.values,
                      mode='lines', name='Portfolio Value', line=dict(color='darkgreen')),
            row=4, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} - Mejor Configuración VWAP (Entry={best_params["entry_threshold"]*100:.1f}%, Exit={best_params["exit_threshold"]*100:.1f}%)',
            height=1000,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Price/VWAP", row=2, col=1)
        fig.update_yaxes(title_text="Deviation %", row=3, col=1)
        fig.update_yaxes(title_text="Portfolio Value", row=4, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
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
        print(f"Configuración: Entry={result['entry_threshold']*100:.1f}%, Exit={result['exit_threshold']*100:.1f}%")
        print("-" * 80)
        
        try:
            # Get trade records (DataFrame)
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
print("OPTIMIZACIÓN DE PARÁMETROS VWAP - ANÁLISIS CON DATOS DE TRADES")
print("="*80)
print("📊 Descargando datos de trades individuales y convirtiendo a barras OHLCV de 4 segundos")
print("🤖 Simulando análisis de bot en tiempo real con datos de trades")
print("="*80)

stock_data = {}
for symbol in STOCKS:
    print(f"Descargando datos de trades para {symbol}...")
    data = download_stock_data(symbol, start_date, end_date)
    if data is not None:
        stock_data[symbol] = data
        close_data = data.get('Close')
        print(f"✓ {symbol}: {len(close_data)} barras OHLCV de 4s generadas desde trades")
    else:
        print(f"✗ {symbol}: No se pudieron obtener datos de trades")

print(f"\n✅ Datos de trades procesados exitosamente para {len(stock_data)} stocks")
print(f"🔄 Convertidos a barras OHLCV de 4 segundos para análisis VWAP")

optimization_results = []

for symbol, data in stock_data.items():
    print(f"\n" + "="*60)
    print(f"OPTIMIZANDO {symbol}")
    print("="*60)
    
    best_params, all_results = optimize_vwap_for_stock(data, symbol)
    
    if best_params:
        optimization_results.append(best_params)
        
        print(f"\n🏆 MEJOR CONFIGURACIÓN PARA {symbol}:")
        print(f"   Entry Threshold: {best_params['entry_threshold']*100:.1f}%")
        print(f"   Exit Threshold: {best_params['exit_threshold']*100:.1f}%")
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
            print(f"{'Rank':<5} {'Entry%':<8} {'Exit%':<7} {'Return':<10} {'Profit':<12} {'Trades':<8}")
            print("-" * 65)
            for i, result in enumerate(sorted_results, 1):
                print(f"{i:<5} {result['entry_threshold']*100:<7.1f} {result['exit_threshold']*100:<6.1f} {result['total_return']:<9.2%} ${result['profit']:<10,.0f} {result['num_trades']:<8}")
        
        print(f"\n📈 Generando gráfico para {symbol}...")
        plot_best_strategy(data, symbol, best_params)
        
        show_detailed_trades([best_params])
        
    else:
        print(f"❌ No se encontraron configuraciones válidas para {symbol}")

print(f"\n" + "="*80)
print("RESUMEN FINAL - MEJORES CONFIGURACIONES POR STOCK")
print("="*80)

if optimization_results:
    optimization_results.sort(key=lambda x: x['total_return'], reverse=True)
    
    print(f"{'Stock':<8} {'Entry%':<8} {'Exit%':<7} {'Return':<10} {'Profit':<12} {'Sharpe':<8} {'Trades':<8}")
    print("-" * 80)
    
    for result in optimization_results:
        print(f"{result['symbol']:<8} {result['entry_threshold']*100:<7.1f} {result['exit_threshold']*100:<6.1f} {result['total_return']:<9.2%} ${result['profit']:<10,.0f} {result['sharpe_ratio']:<7.2f} {result['num_trades']:<8}")
    
    best_overall = optimization_results[0]
    print(f"\n🥇 MEJOR STOCK GENERAL: {best_overall['symbol']}")
    print(f"   Configuración: Entry={best_overall['entry_threshold']*100:.1f}%, Exit={best_overall['exit_threshold']*100:.1f}%")
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
