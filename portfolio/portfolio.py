import vectorbtpro as vbt
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import exchange_calendars as ecals

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))


def test_rsi_strategy_on_stock(
    price_data,
    symbol,
    rsi_window=20,
    entry_level=30,
    exit_level=75,
    show_warnings=True,
    take_profit=0.03,
    stop_loss=0.05,
):
    try:
        rsi = vbt.RSI.run(price_data, window=rsi_window)
        calendar = ecals.get_calendar("XNYS")
        sessions = calendar.sessions_in_range(
            price_data.index.min().date(), price_data.index.max().date()
        )
        sessions_dates = pd.to_datetime(sessions).date
        index_dates = price_data.index.date
        holiday_mask = pd.Series(
            [d in sessions_dates for d in index_dates], index=price_data.index
        )
        time_mask = price_data.index.indexer_between_time("9:30", "16:00")
        time_bool_mask = pd.Series(False, index=price_data.index)
        if len(time_mask) > 0:
            time_bool_mask.iloc[time_mask] = True
        weekday_mask = price_data.index.weekday.isin([0, 1, 2, 3, 4])
        final_mask = time_bool_mask & weekday_mask & holiday_mask
        if final_mask.sum() == 0:
            if show_warnings:
                print(
                    f"WARNING: No valid trading times for {symbol}, using weekday+holiday filter"
                )
            final_mask = weekday_mask & holiday_mask

        entries = rsi.rsi_crossed_below(entry_level) & final_mask
        exits = rsi.rsi_crossed_above(exit_level) & final_mask
        if not entries.any() or not exits.any():
            return None

        INITIAL_CAPITAL = 20_000
        pf = vbt.Portfolio.from_signals(
            price_data,
            entries,
            exits,
            init_cash=INITIAL_CAPITAL,
            sl_stop=stop_loss,
            tp_stop=take_profit,
            accumulate=False,
        )

        stats = pf.stats()
        stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else {}
        total_return = stats_dict.get("Total Return [%]", 0) / 100
        sharpe_ratio = stats_dict.get("Sharpe Ratio", 0)
        max_drawdown = stats_dict.get("Max Drawdown [%]", 0) / 100
        num_trades = stats_dict.get("Total Trades", 0)
        win_rate = stats_dict.get("Win Rate [%]", 0) / 100

        return {
            "symbol": symbol,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": num_trades,
        }

    except Exception as e:
        print(f"Error testing RSI on {symbol}: {e}")
        return None


def optimize_portfolio(returns, weight_min=0.01, weight_max=0.3, solver="ECOS"):
    try:
        result = vbt.riskfolio_optimize(
            returns=returns,
            rm="MV",
            obj="Sharpe",
            rf=0,
            weight_min=weight_min,
            weight_max=weight_max,
            budget=1,
            solver=solver,
            display=False,
        )
        weights = pd.Series(result) if isinstance(result, dict) else result
        total_weight = weights.sum()
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Infeasible weights sum: {total_weight}")
        return weights

    except Exception as e:
        print("Optimization failed:", e)
        # En caso de fallo devolvemos ceros
        return pd.Series({col: 0.0 for col in returns.columns})


if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "SATL", "EVLV", "F", "TSLA", "NVDA"]
    start_date = "2020-01-01"
    end_date = "2025-06-01"

    # 1. Descarga precios y calcula retornos diarios
    data = vbt.YFData.pull(symbols, start=start_date, end=end_date)
    close = data.get("Close")
    daily_returns = close.vbt.pct_change().dropna()

    weights = optimize_portfolio(daily_returns, weight_min=0.01, weight_max=0.3)

    # 4. Imprime y grafica pesos
    print("\nPesos Óptimos:")
    for sym, w in weights.items():
        print(f"{sym}: {w*100:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.bar(weights.index, weights.values * 100)
    plt.title("Optimized Portfolio Weights")
    plt.xlabel("Symbol")
    plt.ylabel("Weight (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.pie(
        weights.values,
        labels=weights.index,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
    )
    plt.title("Portfolio Weight Distribution")
    plt.axis("equal")
    plt.show()
