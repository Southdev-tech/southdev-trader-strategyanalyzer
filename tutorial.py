import vectorbt as vbt
import datetime
import pandas as pd
import numpy as np
import talib
from numba import njit

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=1)

# btc_price = pd.read_csv('btc_price.csv')

# if btc_price.empty:
#     btc_price = vbt.YFData.download(['BTC-USD', 'ETH-USD'], missing_index='drop', interval='1m', start=start_date, end=end_date).get("Close")
#     btc_price.to_csv('btc_price.csv')
# else:
#     btc_price = pd.read_csv('btc_price.csv')
    
# btc_price["Datetime"] = pd.to_datetime(btc_price["Datetime"])
# btc_price.set_index("Datetime", inplace=True)

# btc_price = btc_price["BTC-USD"]

# RSI = vbt.IndicatorFactory.from_talib('RSI')

# print(btc_price)

# rsi = vbt.RSI.run(btc_price, window=14)

# print(rsi.rsi)

# entries = rsi.rsi_crossed_below(30)
# exits = rsi.rsi_crossed_above(70)

# pf = vbt.Portfolio.from_signals(btc_price, entries, exits)

# pf.plot().show()

# print(pf.stats())

# @njit
# def produce_signal(rsi, entry, exit):
#     trend = np.where(rsi > exit, -1, 0)
#     trend = np.where((rsi < entry), 1, trend)
#     return trend

# def custom_indicator(close, rsi_window=14, entry = 30, exit = 70): 
#     rsi = RSI.run(close, rsi_window).real.to_numpy()
#     return produce_signal(rsi, entry, exit)


# ind = vbt.IndicatorFactory(
#     class_name="Combination", 
#     short_name="comb", 
#     input_names=["close"], 
#     param_names=["rsi_window", "entry", "exit"], 
#     output_names=["value"]
#     ).from_apply_func(custom_indicator, rsi_window = 14, entry = 30, exit = 70)
 
# rsi_windows = np.arange(10,40,step=1, dtype=int)

# master_returns = []

# for window in rsi_windows:
#     res = ind.run(
#         btc_price,
#         rsi_window=window,
#         entry=np.arange(20, 35, 2, dtype=int),
#         exit=np.arange(65, 80, 2, dtype=int),
#         param_product=True
#     )
#     print(res.value)

#     entries = res.value == 1
#     exits = res.value == -1

#     pf = vbt.Portfolio.from_signals(btc_price, entries, exits)
#     master_returns.append(pf.total_return())

# print(master_returns)

# returns = pd.concat(master_returns)



# if 'symbol' in returns.index.names:
#     returns_grouped = returns.groupby(level=["comb_exit","comb_entry", "symbol"]).mean()
# else:
#     returns_grouped = returns.groupby(level=["comb_exit","comb_entry"]).mean()

# fig = returns.vbt.volume(
#     x_level = "comb_rsi_window",
#     y_level = "comb_entry",
#     z_level = "comb_exit",
#     slider_level = "symbol" if 'symbol' in returns.index.names else None
# )

# fig.show()

# btc_price = vbt.YFData.download(['BTC-USD', 'ETH-USD'], missing_index='drop', interval='1m', start=start_date, end=end_date).get("Close")

# fast_ma = vbt.MA.run(btc_price, window=50)
# slow_ma = vbt.MA.run(btc_price, window=200)

# entries = fast_ma.ma_crossed_above(slow_ma)
# exits = fast_ma.ma_crossed_below(slow_ma)

# pf = vbt.Portfolio.from_signals(btc_price, entries, exits)

# fig = btc_price.vbt.plot(trace_kwargs=dict(name="Price", line=dict(color="red")))
# fig = fast_ma.ma.vbt.plot(trace_kwargs=dict(name="Fast MA", line=dict(color="blue")), fig=fig)
# fig = slow_ma.ma.vbt.plot(trace_kwargs=dict(name="Slow MA", line=dict(color="green")), fig=fig)
# fig = entries.vbt.signals.plot_as_entry_markers(btc_price, fig=fig)
# fig = exits.vbt.signals.plot_as_exit_markers(btc_price, fig=fig)

# fig.show()


# -------

btc_price = vbt.YFData.download('BTC-USD', missing_index='drop', interval='1m', start=start_date, end=end_date).get("Close")

rsi = vbt.RSI.run(btc_price, window=21)

entries = rsi.rsi_crossed_above(30)
exits = rsi.rsi_crossed_below(70)

pf = vbt.Portfolio.from_signals(
    btc_price,
    entries=entries,
    exits=exits,
    short_entries=entries,
    short_exits=exits,
    upon_dir_conflict=vbt.Portfolio.enums.DirectionConflictMode.Short,
    upon_opposite_entry=vbt.Portfolio.enums.OppositeEntryMode.Close,
    )

pf.plot.show()

