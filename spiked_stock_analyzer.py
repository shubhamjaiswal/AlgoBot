import json

with open("config.json") as f:
    cfg = json.load(f)

api_key = cfg["api_key"]
access_token = cfg["access_token"]
instrument_tokens = cfg["instrument_token"]

import datetime as dt
import pandas as pd
from kiteconnect import KiteConnect
import json
import time

with open("config.json") as f:
    cfg = json.load(f)
print(cfg)
# ---------------- USER CONFIG ----------------
API_KEY = cfg["api_key"]
ACCESS_TOKEN = cfg["access_token"]

# ---------------------------------------------
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)


def fetch_ohlc(instrument_token, interval="day", lookback=30):
    """Fetch last N days of historical data for given instrument token"""
    to_date = dt.date.today()
    from_date = to_date - dt.timedelta(days=lookback * 2)  # buffer for weekends/holidays
    data = kite.historical_data(
        instrument_token,
        from_date.strftime("%Y-%m-%d"),
        to_date.strftime("%Y-%m-%d"),
        interval
    )
    time.sleep(1.5)
    df = pd.DataFrame(data)
    if not df.empty:
        df.set_index("date", inplace=True)
    return df


def screen_movers():
    movers = []

    # Fetch all NSE instruments
    instruments = kite.instruments("NSE")
    nse_df = pd.DataFrame(instruments)

    # Filter for EQ stocks (ignore indices, bonds, etc.)
    nse_stocks = nse_df[nse_df['segment'] == 'NSE']
    nse_stocks = nse_stocks[nse_stocks['instrument_type'] == 'EQ']

    print(f"✅ Found {len(nse_stocks)} NSE stocks")

    for _, row in nse_stocks.iterrows():
        tradingsymbol = row['tradingsymbol']
        token = row['instrument_token']

        try:
            df = fetch_ohlc(token, interval="day", lookback=60)
            if len(df) < 2:
                continue
            last_close = df["close"].iloc[-1]
            prev_close = df["close"].iloc[-2]
            pct_change = (last_close - prev_close) / prev_close * 100
            breakout_condition = last_close > df["high"].rolling(30).max().iloc[-2]
            df["ATR14"] = (df["high"] - df["low"]).rolling(14).mean()
            atr_breakout = (last_close - prev_close) > 1.5 * last_close["ATR14"]
            # if pct_change >4.5:
            if (
                    pct_change > 5
                    and breakout_condition
                    and atr_breakout
            ):
                movers.append((tradingsymbol, round(pct_change, 2), last_close))
                print(f"{tradingsymbol}: {round(pct_change, 2)}% | Last Close = {last_close}")
        except Exception as e:
            print(f"⚠️ Error fetching {tradingsymbol}: {e}")

    return movers


def sma_volume_breakouts():
    movers = []

    # Fetch all NSE instruments
    instruments = kite.instruments("NSE")
    nse_df = pd.DataFrame(instruments)

    # Filter for EQ stocks
    nse_stocks = nse_df[(nse_df['segment'] == 'NSE') & (nse_df['instrument_type'] == 'EQ')]
    print(f"✅ Found {len(nse_stocks)} NSE stocks")

    for _, row in nse_stocks.iterrows():
        tradingsymbol = row['tradingsymbol']
        token = row['instrument_token']

        try:
            df = fetch_ohlc(token, interval="day", lookback=70)
            if len(df) < 51:  # Need at least 50 days for SMA
                continue

            # Calculate SMA and Avg Volume
            df["SMA50"] = df["close"].rolling(50).mean()
            df["AvgVol20"] = df["volume"].rolling(20).mean()

            last = df.iloc[-1]

            # Conditions
            close_above_sma50 = last["close"] > last["SMA50"]
            vol_breakout = last["volume"] > 1.5 * last["AvgVol20"]

            if close_above_sma50 and vol_breakout:
                movers.append({
                    "Symbol": tradingsymbol,
                    "Close": round(last["close"], 2),
                    "SMA50": round(last["SMA50"], 2),
                    "Volume": int(last["volume"]),
                    "AvgVol20": int(last["AvgVol20"])
                })
                print(
                    f"🔥 {tradingsymbol} | Close={last['close']} > SMA50={last['SMA50']} | Vol={last['volume']} > 1.5*{last['AvgVol20']}")

        except Exception as e:
            print(f"⚠️ Error fetching {tradingsymbol}: {e}")

    return pd.DataFrame(movers)

def sma_volume_breakouts_under_250():
    movers = []

    # Fetch all NSE instruments
    instruments = kite.instruments("NSE")
    nse_df = pd.DataFrame(instruments)

    # Filter for EQ stocks
    nse_stocks = nse_df[(nse_df['segment'] == 'NSE') & (nse_df['instrument_type'] == 'EQ')]
    print(f"✅ Found {len(nse_stocks)} NSE stocks")

    for _, row in nse_stocks.iterrows():
        tradingsymbol = row['tradingsymbol']
        token = row['instrument_token']

        try:
            df = fetch_ohlc(token, interval="day", lookback=100)
            if len(df) < 60:  # need at least 50 SMA + 20 volume avg
                continue

            # Indicators
            df["EMA20"] = df["close"].ewm(span=20).mean()
            df["SMA50"] = df["close"].rolling(50).mean()
            df["AvgVol20"] = df["volume"].rolling(20).mean()

            last = df.iloc[-1]
            prev = df.iloc[-2]

            # Price filter
            # if last["close"] < 300:
            #     continue

            # SMA50 crossover filter (yesterday below SMA, today above SMA)
            sma_cross = prev["close"] < prev["SMA50"] and last["close"] > last["SMA50"]

            # Volume breakout
            vol_condition = last["volume"] > 1.5 * last["AvgVol20"]

            # Breakout above recent highs
            breakout_condition = last["close"] > df["high"].rolling(30).max().iloc[-2]

            if sma_cross and vol_condition and breakout_condition:
                pct_change = (last["close"] - prev["close"]) / prev["close"] * 100
                movers.append({
                    "Symbol": tradingsymbol,
                    "Change%": round(pct_change, 2),
                    "Close": last["close"],
                    "Volume": last["volume"],
                    "AvgVol20": int(last["AvgVol20"]),
                    "SMA50": round(last["SMA50"], 2)
                })
                print(f"{tradingsymbol} ✅ Crossed SMA50 | Close={last['close']} | SMA50={last['SMA50']}")

        except Exception as e:
            print(f"⚠️ Error fetching {tradingsymbol}: {e}")

    return pd.DataFrame(movers)

def sma_crossover_4_16():
    movers = []

    # Fetch all NSE instruments
    instruments = kite.instruments("NSE")
    nse_df = pd.DataFrame(instruments)

    # Filter for EQ stocks
    nse_stocks = nse_df[(nse_df['segment'] == 'NSE') & (nse_df['instrument_type'] == 'EQ')]
    print(f"✅ Found {len(nse_stocks)} NSE stocks")

    for _, row in nse_stocks.iterrows():
        tradingsymbol = row['tradingsymbol']
        token = row['instrument_token']

        try:
            # Fetch last 40 days of data (enough for SMA16)
            df = fetch_ohlc(token, interval="day", lookback=50)

            if len(df) < 20:
                continue

            # Calculate SMAs
            df["SMA4"] = df["close"].rolling(4).mean()
            df["SMA16"] = df["close"].rolling(16).mean()

            last = df.iloc[0]
            prev = df.iloc[-1]
            # df["AvgVol20"] = df["volume"].rolling(40).mean()
            # Volume filter
            # vol_condition = last["volume"] > 1.3 * last["AvgVol20"]
            # Crossover condition: yesterday SMA4 < SMA16, today SMA4 > SMA16
            bullish_cross = prev["SMA4"] < prev["SMA16"] and last["SMA4"] > last["SMA16"]
            bearish_cross = prev["SMA4"] > prev["SMA16"] and last["SMA4"] < last["SMA16"]

            if bullish_cross:
                movers.append({
                    "Symbol": tradingsymbol,
                    "Close": round(last["close"], 2),
                    "SMA4": round(last["SMA4"], 2),
                    "SMA16": round(last["SMA16"], 2),
                    "Signal": "Bullish Cross"
                })
                print(f"📈 {tradingsymbol} | {movers[-1]['Signal']} | Close={last['close']}")

        except Exception as e:
            print(f"⚠️ Error fetching {tradingsymbol}: {e}")

    return pd.DataFrame(movers)

def potential_breakouts():
    movers = []

    # Fetch all NSE instruments
    instruments = kite.instruments("NSE")
    nse_df = pd.DataFrame(instruments)

    # Filter for EQ stocks
    nse_stocks = nse_df[(nse_df['segment'] == 'NSE') & (nse_df['instrument_type'] == 'EQ')]
    # nse_stocks = nse_stocks[~nse_stocks['tradingsymbol'].str.contains('-')]
    print(f"✅ Found {len(nse_stocks)} NSE stocks")

    for _, row in nse_stocks.iterrows():
        tradingsymbol = row['tradingsymbol']
        token = row['instrument_token']

        try:
            df = fetch_ohlc(token, interval="day", lookback=60)
            if len(df) < 22:  # need min 22 days for avg volume & EMA
                continue

            # Calculate indicators
            df["EMA20"] = df["close"].ewm(span=200).mean()
            df["AvgVol20"] = df["volume"].rolling(40).mean()

            last = df.iloc[-1]
            prev = df.iloc[-2]

            # % Change filter
            pct_change = (last["close"] - prev["close"]) / prev["close"] * 100

            # Volume filter
            vol_condition = last["volume"] > 1.3 * last["AvgVol20"]

            # Uptrend filter (last 2 bars higher & above EMA)
            uptrend_condition = (
                    last["close"] > last["EMA20"] and
                    last["close"] > prev["close"] and
                    last["close"] > last["open"] and  # green candle yesterday
                    prev["close"] > prev["open"]  # green candle day before
            )
            breakout_condition = last["close"] > df["high"].rolling(30).max().iloc[-2]
            if pct_change > 4 and vol_condition and uptrend_condition and breakout_condition:
                movers.append({
                    "Symbol": tradingsymbol,
                    "Change%": round(pct_change, 2),
                    "Close": last["close"],
                    "Volume": last["volume"],
                    "AvgVol20": int(last["AvgVol20"]),
                    "EMA20": round(last["EMA20"], 2)
                })
                print(f"{tradingsymbol} : Percent change : {pct_change} : Last close {last['close']}")

        except Exception as e:
            print(f"⚠️ Error fetching {tradingsymbol}: {e}")

    return pd.DataFrame(movers)

def potential_breakouts_with_ADX():
    movers = []

    instruments = kite.instruments("NSE")
    nse_df = pd.DataFrame(instruments)
    nse_stocks = nse_df[(nse_df['segment'] == 'NSE') & (nse_df['instrument_type'] == 'EQ')]
    print(f"✅ Found {len(nse_stocks)} NSE stocks")

    for _, row in nse_stocks.iterrows():
        tradingsymbol = row['tradingsymbol']
        token = row['instrument_token']

        try:
            df = fetch_ohlc(token, interval="day", lookback=90)
            if len(df) < 30:
                continue

            # Calculate Indicators
            df["EMA20"] = df["close"].ewm(span=20).mean()
            df["AvgVol20"] = df["volume"].rolling(20).mean()

            # ADX(14) Calculation
            df["TR"] = (df["high"] - df["low"]).combine((df["high"] - df["close"].shift()).abs(), max).combine(
                (df["low"] - df["close"].shift()).abs(), max)
            df["+DM"] = df["high"].diff()
            df["-DM"] = df["low"].diff().abs()
            df["+DM"] = df.apply(lambda x: x["+DM"] if x["+DM"] > x["-DM"] and x["+DM"] > 0 else 0, axis=1)
            df["-DM"] = df.apply(lambda x: x["-DM"] if x["-DM"] > x["+DM"] and x["-DM"] > 0 else 0, axis=1)

            df["+DI14"] = 100 * (df["+DM"].ewm(alpha=1/14).mean() / df["TR"].ewm(alpha=1/14).mean())
            df["-DI14"] = 100 * (df["-DM"].ewm(alpha=1/14).mean() / df["TR"].ewm(alpha=1/14).mean())

            df["DX"] = 100 * (df["+DI14"] - df["-DI14"]).abs() / (df["+DI14"] + df["-DI14"])
            df["ADX14"] = df["DX"].ewm(alpha=1/14).mean()

            last = df.iloc[-1]
            prev = df.iloc[-2]

            # % Change filter
            pct_change = (last["close"] - prev["close"]) / prev["close"] * 100

            # Volume filter
            vol_condition = last["volume"] > 1.3 * last["AvgVol20"]

            # Uptrend filter
            uptrend_condition = (
                last["close"] > last["EMA20"] and
                last["close"] > prev["close"] and
                last["close"] > last["open"] and
                prev["close"] > prev["open"]
            )

            # Breakout condition
            breakout_condition = last["close"] > df["high"].rolling(30).max().iloc[-2]

            # ✅ ADX crossover & trend strength condition (captures recent +DI cross)
            lookback = 3  # days to look back for crossover
            recent_cross = (
                    (df["+DI14"].shift(1) < df["-DI14"].shift(1)) &  # previously bearish
                    (df["+DI14"] > df["-DI14"])  # now bullish
            )
            recent_crossed = recent_cross.tail(lookback).any()

            adx_condition = (
                    recent_crossed and
                    last["+DI14"] > last["-DI14"] and  # still bullish trend
                    last["ADX14"] > 20 and  # trend strong
                    last["ADX14"] > df["ADX14"].iloc[-3]  # rising trend strength
            )

            # if pct_change > 4 and vol_condition and adx_condition:
            if adx_condition:
                movers.append({
                    "Symbol": tradingsymbol,
                    "Change%": round(pct_change, 2),
                    "Close": round(last["close"], 2),
                    "ADX14": round(last["ADX14"], 2),
                    "+DI14": round(last["+DI14"], 2),
                    "-DI14": round(last["-DI14"], 2),
                    "Volume": int(last["volume"]),
                    "AvgVol20": int(last["AvgVol20"]),
                    "EMA20": round(last["EMA20"], 2)
                })
                print(f"🔥 {tradingsymbol} | ADX={round(last['ADX14'],1)} | +DI={round(last['+DI14'],1)} > -DI={round(last['-DI14'],1)}")

        except Exception as e:
            print(f"⚠️ Error fetching {tradingsymbol}: {e}")

    return pd.DataFrame(movers)


if __name__ == "__main__":
    # movers = screen_movers()
    # Fetch all NSE instruments
    # sma_volume_breakouts_under_250()
    # potential_breakouts()
    potential_breakouts_with_ADX()
    # sma_crossover_4_16()
    # print("\n📊 Stocks with >4.5% upside yesterday:")
    # for sym, pct, price in movers:
    #     print(f"{sym}: {pct}% | Last Close = {price}")
