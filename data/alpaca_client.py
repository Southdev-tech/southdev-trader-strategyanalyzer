import os

import vectorbtpro as vbt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

vbt.AlpacaData.set_custom_settings(
    client_config=dict(api_key=os.getenv("ALPACA_API_KEY"), secret_key=os.getenv("ALPACA_SECRET_KEY"))
)


def download_stock_data(symbol, start_date, end_date, strategy):
    """
    Unified function to download stock data for both RSI and VWAP strategies.
    Downloads trades once and converts to appropriate timeframe based on strategy.

    Args:
        symbol: Stock symbol to download
        start_date: Start date for data
        end_date: End date for data
        strategy: 'rsi' (1-minute bars) or 'vwap' (4-second bars)

    Returns:
        Processed data appropriate for the strategy
    """
    try:
        # Download trades once
        print(f"⏳ Downloading trades for {symbol} ({strategy})...")
        data = vbt.AlpacaData.pull(
            symbol,
            start=start_date,
            end=end_date,
            timeframe="1m",
            adjustment="all",
            tz="US/Eastern",
            data_type="trade",
        )

        if strategy == "rsi":
            processed_data = convert_trades_to_ohlcv_unified(data, symbol, timeframe="1T")
            if processed_data is not None and len(processed_data.get("Close")) > 50:
                return processed_data.get("Close")
            else:
                print(f"Insufficient data for {symbol} RSI")
                return None

        elif strategy == "vwap":
            processed_data = convert_trades_to_ohlcv_unified(data, symbol, timeframe="4S")
            if processed_data is not None and len(processed_data.get("Close")) > 50:
                return processed_data
            else:
                print(f"Insufficient data for {symbol} VWAP")
                return None
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'rsi' or 'vwap'")

    except Exception as e:
        print(f"Error downloading {symbol} for {strategy}: {e}")
        return None


def convert_trades_to_ohlcv_unified(trade_data, symbol, timeframe="4S"):
    """
    Unified function to convert individual trade data to OHLCV format.

    Args:
        trade_data: Raw trade data from Alpaca
        symbol: Stock symbol
        timeframe: '1T' for 1-minute, '4S' for 4-second

    Returns:
        OHLCVData object with get() method
    """
    try:
        df = trade_data.get()

        if df is None or len(df) == 0:
            print(f"No trade data available for {symbol}")
            return None

        timeframe_name = "1-minute" if timeframe == "1T" else "4-second"
        print(f"📊 Converting {len(df)} trades to {timeframe_name} OHLCV for {symbol}")
        print(f"📋 Trade data columns: {list(df.columns)}")

        price_col = None
        volume_col = None

        for col in df.columns:
            if "price" in col.lower() or col.lower() == "price":
                price_col = col
                break

        for col in df.columns:
            if "size" in col.lower() or "volume" in col.lower():
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

        print(f"✓ Using price column: '{price_col}', volume column: '{volume_col}'")

        # Prepare data for resampling
        df_trades = df[[price_col, volume_col]].copy()
        df_trades.columns = ["Price", "Volume"]

        ohlcv = (
            df_trades.resample(timeframe)
            .agg(
                {
                    # Open, High, Low, Close
                    "Price": ["first", "max", "min", "last"],
                    "Volume": "sum",
                }
            )
            .dropna()
        )

        ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]

        ohlcv = ohlcv.fillna(method="ffill").dropna()

        if len(ohlcv) == 0:
            print(f"❌ No OHLCV data generated for {symbol}")
            return None

        print(f"✅ Generated {len(ohlcv)} {timeframe_name} OHLCV bars from {len(df)} trades")

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
        print(f"Error converting trades to {timeframe} OHLCV for {symbol}: {e}")
        return None


def convert_trades_to_ohlcv_1min(trade_data, symbol):
    """
    Convert individual trade data to 1-minute OHLCV format for RSI calculation.
    """
    try:
        df = trade_data.get()

        if df is None or len(df) == 0:
            print(f"No trade data available for {symbol}")
            return None

        price_col = None
        volume_col = None

        for col in df.columns:
            if "price" in col.lower() or col.lower() == "price":
                price_col = col
                break

        for col in df.columns:
            if "size" in col.lower() or "volume" in col.lower():
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
        df_trades.columns = ["Price", "Volume"]

        ohlcv = (
            df_trades.resample("1T")
            .agg(
                {
                    # Open, High, Low, Close
                    "Price": ["first", "max", "min", "last"],
                    "Volume": "sum",
                }
            )
            .dropna()
        )

        ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]

        ohlcv = ohlcv.fillna(method="ffill").dropna()

        if len(ohlcv) == 0:
            print(f"❌ No OHLCV data generated for {symbol}")
            return None

        print(f"   ✅ Generated {len(ohlcv)} 1-minute OHLCV bars from {len(df)} trades")

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
        print(f"Error converting trades to 1-minute OHLCV for {symbol}: {e}")
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
            if "price" in col.lower() or col.lower() == "price":
                price_col = col
                break

        for col in df.columns:
            if "size" in col.lower() or "volume" in col.lower():
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
        df_trades.columns = ["Price", "Volume"]

        ohlcv = (
            df_trades.resample("4S")
            .agg(
                {
                    # Open, High, Low, Close
                    "Price": ["first", "max", "min", "last"],
                    "Volume": "sum",
                }
            )
            .dropna()
        )

        ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]

        ohlcv = ohlcv.fillna(method="ffill").dropna()

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


def download_stock_data_both(symbol, start_date, end_date):
    """
    Super optimized function: downloads trades once and converts to both RSI (1-minute) and VWAP (4-second) formats.

    Args:
        symbol: Stock symbol to download
        start_date: Start date for data
        end_date: End date for data

    Returns:
        tuple: (rsi_data, vwap_data) where rsi_data is Close prices and vwap_data is full OHLCV
    """
    try:
        # Download trades once
        print(f"⏳ Downloading trades for {symbol} (RSI + VWAP)...")
        data = vbt.AlpacaData.pull(
            symbol,
            start=start_date,
            end=end_date,
            timeframe="1m",
            adjustment="all",
            tz="US/Eastern",
            data_type="trade",
        )

        # Convert to both formats
        rsi_data = None
        vwap_data = None

        # RSI: 1-minute bars, only Close prices needed
        rsi_ohlcv = convert_trades_to_ohlcv_unified(data, symbol, timeframe="1T")
        if rsi_ohlcv is not None and len(rsi_ohlcv.get("Close")) > 50:
            rsi_data = rsi_ohlcv.get("Close")
            print(f"   ✓ RSI {symbol}: {len(rsi_data)} puntos (1-minute)")
        else:
            print(f"   ❌ Insufficient RSI data for {symbol}")

        # VWAP: 4-second bars, full OHLCV needed
        vwap_ohlcv = convert_trades_to_ohlcv_unified(data, symbol, timeframe="4S")
        if vwap_ohlcv is not None and len(vwap_ohlcv.get("Close")) > 50:
            vwap_data = vwap_ohlcv
            close_data = vwap_ohlcv.get("Close")
            print(f"   ✓ VWAP {symbol}: {len(close_data)} puntos (4-second)")
        else:
            print(f"   ❌ Insufficient VWAP data for {symbol}")

        return rsi_data, vwap_data

    except Exception as e:
        print(f"   ❌ Error downloading {symbol}: {e}")
        return None, None
