
import backtrader as bt
from datetime import datetime, time
import os, csv
import datetime as dt
import math
# import time
import pytz
import backtrader as bt
import math
import logging
import requests
from indicators import HybridChopFilter, VIDYA, Supertrend, VWAP, SessionVWAP, ChoppinessIndex, FractalChaosOsc, \
    MADiv_Stoch_Supertrend_Vol, DirectionalChoppiness

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
"""
Backtrader breakout + retracement strategy (3-min feed recommended)

Logic:
1. Lookback N bars -> compute highest high over last N bars (excluding current bar).
2. Breakout detected when current close > highest_high AND (optional) volume confirms:
     volume > vol_multiplier * SMA(volume, N)
3. When breakout occurs record:
     wave_high = close_of_breakout_bar
     wave_low  = lowest low over the previous N bars (or a defined start)
   wave_range = wave_high - wave_low
4. Calculate retracement price:
     retrace_price = wave_high - retrace_pct * wave_range
5. Wait for price to retrace to the retrace_price. Once reached, enter long.
6. Set stop loss:
     stop = wave_low - atr_multiplier * ATR(14) (small buffer)
   Set target: target = entry + rr * (entry - stop)  (R:R = rr:1)
7. Position sizing: risk_per_trade_pct of portfolio equity.
"""

import backtrader as bt
import math

"""
Breakout + Retracement Strategy (3-min feed)
Exit condition: exit long if previous bar's low is broken.
R:R still used for target.
"""


class BreakoutRetraceStrategy(bt.Strategy):
    params = dict(
        n=7,                    # lookback bars for breakout
        vol_multiplier=1.5,      # breakout volume confirmation factor
        use_volume=True,         # use volume filter
        retrace_pct=0.15,        # 15% retrace level
        risk_per_trade=0.01,     # 1% of account per trade
        rr=2.5,                  # risk:reward = 1:2
        atr_period=14,           # ATR for minor padding
        atr_multiplier=0.5,      # optional stop padding
        verbose=False,
        reject_if_too_small=True,
    )

    def __init__(self):
        self.order = None
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavol = self.datas[0].volume

        self.vol_sma = bt.indicators.SimpleMovingAverage(self.datavol, period=self.p.n)
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)

        # state vars
        self.in_breakout_wait = False
        self.wave_high = None
        self.wave_low = None
        self.retrace_price = None
        self.wave_range = None

        self.entry_price = None
        self.stop_price = None
        self.target_price = None

    def log(self, txt, dt=None):
        if self.p.verbose:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.isoformat()}  {txt}')

    def next(self):
        if len(self) < max(self.p.n, self.p.atr_period) + 2:
            return
        if self.order:
            return

        close = self.dataclose[0]
        volume = self.datavol[0]

        # --- Manage open position ---
        if self.position:
            prev_low = self.datalow[-1]
            # early exit: if current price breaks previous low, exit immediately
            if close < prev_low:
                self.log(f"Exit triggered: close {close:.2f} < prev low {prev_low:.2f}")
                self.close()  # close market order
                self.entry_price = None
                self.stop_price = None
                self.target_price = None
                return

            # also optional: if target reached, exit
            if self.target_price and close >= self.target_price:
                self.log(f"Target hit at {close:.2f}")
                self.close()
                self.entry_price = None
                self.stop_price = None
                self.target_price = None
            return

        # --- Detect breakout ---
        hh = max([self.datahigh[-i] for i in range(1, self.p.n + 1)])
        ll = min([self.datalow[-i] for i in range(1, self.p.n + 1)])

        breakout = close > hh
        vol_ok = True
        if self.p.use_volume:
            vol_ok = volume > self.vol_sma[0] * self.p.vol_multiplier

        if breakout and vol_ok:
            self.wave_high = close
            self.wave_low = ll
            self.wave_range = max(1e-6, (self.wave_high - self.wave_low))
            self.retrace_price = self.wave_high - self.p.retrace_pct * self.wave_range
            self.in_breakout_wait = True
            self.breakout_bar_idx = len(self)
            self.log(f"Breakout detected: close={close:.2f}, retrace={self.retrace_price:.2f}")
            return

        # --- Wait for retrace ---
        if self.in_breakout_wait:
            max_wait = int(self.p.n * 3)
            if len(self) - self.breakout_bar_idx > max_wait:
                self.log("Abandoning breakout (timeout)")
                self._reset_wave_state()
                return

            low = self.datalow[0]
            if low <= self.retrace_price:
                atr_val = float(self.atr[0]) if self.atr[0] else 0.0
                stop = self.wave_low - self.p.atr_multiplier * atr_val
                entry = close
                risk_per_share = entry - stop
                if risk_per_share <= 0:
                    self._reset_wave_state()
                    return

                cash = self.broker.getvalue()
                risk_amount = cash * self.p.risk_per_trade
                size = math.floor(risk_amount / risk_per_share)
                if size <= 0 and self.p.reject_if_too_small:
                    self._reset_wave_state()
                    return

                target = entry + self.p.rr * risk_per_share

                self.entry_price = entry
                self.stop_price = stop
                self.target_price = target

                self.log(f"Entering LONG: entry={entry:.2f}, stop={stop:.2f}, target={target:.2f}, size={size}")
                self.buy(size=size)
                self.in_breakout_wait = False
                return

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED: {order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED: {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order canceled/rejected")
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"Trade closed: PnL={trade.pnl:.2f}")

    def _reset_wave_state(self):
        self.in_breakout_wait = False
        self.wave_high = None
        self.wave_low = None
        self.wave_range = None
        self.retrace_price = None
        self.breakout_bar_idx = None

class Sma2Sma7Crossover(bt.Strategy):
    def __init__(self):
        self.sma2 = bt.ind.SMA(self.data.close, period=2)
        self.sma7 = bt.ind.SMA(self.data.close, period=7)
        self.crossover = bt.ind.CrossOver(self.sma2, self.sma7)
        self.order = None

    def next(self):
        if self.order:
            return

        # Time from data file (1-minute bar timestamp)
        bar_time = self.data.datetime.datetime(0)
        close_price = self.data.close[0]

        if len(self.data) < 8:
            return

        position = self.broker.getposition(self.data)

        if position.size == 0:
            if self.crossover > 0:
                self.order = self.buy()
                print(f"[{bar_time}] BUY triggered @ {close_price:.2f}")
        elif position.size > 0:
            if self.crossover < 0:
                self.order = self.sell()
                print(f"[{bar_time}] SELL triggered @ {close_price:.2f}")

    def notify_order(self, order):
        if order is None:
            self.order = None
            return

        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


## WORKING STRAT ##
class Sma2Sma7CrossoverBarCheck(bt.Strategy):
    params = dict(
        sma3=3,
        sma7=7,
        lookback=5  # Number of bars to look back for highest close
    )

    def __init__(self):
        self.sma3 = bt.ind.SMA(self.data.close, period=3)
        self.sma7 = bt.ind.SMA(self.data.close, period=7)
        self.crossover = bt.ind.CrossOver(self.sma3, self.sma7)
        self.order = None

    def next(self):
        if self.order:
            return

        bar_time = self.data.datetime.datetime(0)
        close_price = self.data.close[0]

        if len(self.data) < max(5, self.p.lookback + 1):
            return

        position = self.broker.getposition(self.data)

        # Check if current close is higher than the highest closing of the last N bars
        highest_recent_close = max(self.data.close[-i] for i in range(1, self.p.lookback + 1))
        is_close_breakout = close_price > highest_recent_close

        if position.size == 0:
            if self.crossover > 0 and is_close_breakout:
                self.order = self.buy()
                print(f"[{bar_time}] BUY triggered @ {close_price:.2f} | Close > Last {self.p.lookback} bars")
        elif position.size > 0:
            if self.crossover < 0:
                self.order = self.sell()
                print(f"[{bar_time}] SELL triggered @ {close_price:.2f}")

    def notify_order(self, order):
        if order is None:
            self.order = None
            return

        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


import backtrader as bt
from datetime import datetime, time
import os, csv
import datetime as dt
import math
# import time
import pytz
import backtrader as bt
import math
import logging
import requests
from indicators import HybridChopFilter, VIDYA, Supertrend, VWAP, SessionVWAP, ChoppinessIndex, FractalChaosOsc, \
    MADiv_Stoch_Supertrend_Vol, DirectionalChoppiness

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
"""
Backtrader breakout + retracement strategy (3-min feed recommended)

Logic:
1. Lookback N bars -> compute highest high over last N bars (excluding current bar).
2. Breakout detected when current close > highest_high AND (optional) volume confirms:
     volume > vol_multiplier * SMA(volume, N)
3. When breakout occurs record:
     wave_high = close_of_breakout_bar
     wave_low  = lowest low over the previous N bars (or a defined start)
   wave_range = wave_high - wave_low
4. Calculate retracement price:
     retrace_price = wave_high - retrace_pct * wave_range
5. Wait for price to retrace to the retrace_price. Once reached, enter long.
6. Set stop loss:
     stop = wave_low - atr_multiplier * ATR(14) (small buffer)
   Set target: target = entry + rr * (entry - stop)  (R:R = rr:1)
7. Position sizing: risk_per_trade_pct of portfolio equity.
"""

import backtrader as bt
import math

"""
Breakout + Retracement Strategy (3-min feed)
Exit condition: exit long if previous bar's low is broken.
R:R still used for target.
"""


class BreakoutRetraceStrategy(bt.Strategy):
    params = dict(
        n=7,                    # lookback bars for breakout
        vol_multiplier=1.5,      # breakout volume confirmation factor
        use_volume=True,         # use volume filter
        retrace_pct=0.15,        # 15% retrace level
        risk_per_trade=0.01,     # 1% of account per trade
        rr=2.5,                  # risk:reward = 1:2
        atr_period=14,           # ATR for minor padding
        atr_multiplier=0.5,      # optional stop padding
        verbose=False,
        reject_if_too_small=True,
    )

    def __init__(self):
        self.order = None
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavol = self.datas[0].volume

        self.vol_sma = bt.indicators.SimpleMovingAverage(self.datavol, period=self.p.n)
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)

        # state vars
        self.in_breakout_wait = False
        self.wave_high = None
        self.wave_low = None
        self.retrace_price = None
        self.wave_range = None

        self.entry_price = None
        self.stop_price = None
        self.target_price = None

    def log(self, txt, dt=None):
        if self.p.verbose:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.isoformat()}  {txt}')

    def next(self):
        if len(self) < max(self.p.n, self.p.atr_period) + 2:
            return
        if self.order:
            return

        close = self.dataclose[0]
        volume = self.datavol[0]

        # --- Manage open position ---
        if self.position:
            prev_low = self.datalow[-1]
            # early exit: if current price breaks previous low, exit immediately
            if close < prev_low:
                self.log(f"Exit triggered: close {close:.2f} < prev low {prev_low:.2f}")
                self.close()  # close market order
                self.entry_price = None
                self.stop_price = None
                self.target_price = None
                return

            # also optional: if target reached, exit
            if self.target_price and close >= self.target_price:
                self.log(f"Target hit at {close:.2f}")
                self.close()
                self.entry_price = None
                self.stop_price = None
                self.target_price = None
            return

        # --- Detect breakout ---
        hh = max([self.datahigh[-i] for i in range(1, self.p.n + 1)])
        ll = min([self.datalow[-i] for i in range(1, self.p.n + 1)])

        breakout = close > hh
        vol_ok = True
        if self.p.use_volume:
            vol_ok = volume > self.vol_sma[0] * self.p.vol_multiplier

        if breakout and vol_ok:
            self.wave_high = close
            self.wave_low = ll
            self.wave_range = max(1e-6, (self.wave_high - self.wave_low))
            self.retrace_price = self.wave_high - self.p.retrace_pct * self.wave_range
            self.in_breakout_wait = True
            self.breakout_bar_idx = len(self)
            self.log(f"Breakout detected: close={close:.2f}, retrace={self.retrace_price:.2f}")
            return

        # --- Wait for retrace ---
        if self.in_breakout_wait:
            max_wait = int(self.p.n * 3)
            if len(self) - self.breakout_bar_idx > max_wait:
                self.log("Abandoning breakout (timeout)")
                self._reset_wave_state()
                return

            low = self.datalow[0]
            if low <= self.retrace_price:
                atr_val = float(self.atr[0]) if self.atr[0] else 0.0
                stop = self.wave_low - self.p.atr_multiplier * atr_val
                entry = close
                risk_per_share = entry - stop
                if risk_per_share <= 0:
                    self._reset_wave_state()
                    return

                cash = self.broker.getvalue()
                risk_amount = cash * self.p.risk_per_trade
                size = math.floor(risk_amount / risk_per_share)
                if size <= 0 and self.p.reject_if_too_small:
                    self._reset_wave_state()
                    return

                target = entry + self.p.rr * risk_per_share

                self.entry_price = entry
                self.stop_price = stop
                self.target_price = target

                self.log(f"Entering LONG: entry={entry:.2f}, stop={stop:.2f}, target={target:.2f}, size={size}")
                self.buy(size=size)
                self.in_breakout_wait = False
                return

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED: {order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED: {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order canceled/rejected")
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"Trade closed: PnL={trade.pnl:.2f}")

    def _reset_wave_state(self):
        self.in_breakout_wait = False
        self.wave_high = None
        self.wave_low = None
        self.wave_range = None
        self.retrace_price = None
        self.breakout_bar_idx = None

class Sma2Sma7Crossover(bt.Strategy):
    def __init__(self):
        self.sma2 = bt.ind.SMA(self.data.close, period=2)
        self.sma7 = bt.ind.SMA(self.data.close, period=7)
        self.crossover = bt.ind.CrossOver(self.sma2, self.sma7)
        self.order = None

    def next(self):
        if self.order:
            return

        # Time from data file (1-minute bar timestamp)
        bar_time = self.data.datetime.datetime(0)
        close_price = self.data.close[0]

        if len(self.data) < 8:
            return

        position = self.broker.getposition(self.data)

        if position.size == 0:
            if self.crossover > 0:
                self.order = self.buy()
                print(f"[{bar_time}] BUY triggered @ {close_price:.2f}")
        elif position.size > 0:
            if self.crossover < 0:
                self.order = self.sell()
                print(f"[{bar_time}] SELL triggered @ {close_price:.2f}")

    def notify_order(self, order):
        if order is None:
            self.order = None
            return

        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


## WORKING STRAT ##
class Sma2Sma7CrossoverBarCheck(bt.Strategy):
    params = dict(
        sma3=3,
        sma7=7,
        lookback=5  # Number of bars to look back for highest close
    )

    def __init__(self):
        self.sma3 = bt.ind.SMA(self.data.close, period=3)
        self.sma7 = bt.ind.SMA(self.data.close, period=7)
        self.crossover = bt.ind.CrossOver(self.sma3, self.sma7)
        self.order = None

    def next(self):
        if self.order:
            return

        bar_time = self.data.datetime.datetime(0)
        close_price = self.data.close[0]

        if len(self.data) < max(5, self.p.lookback + 1):
            return

        position = self.broker.getposition(self.data)

        # Check if current close is higher than the highest closing of the last N bars
        highest_recent_close = max(self.data.close[-i] for i in range(1, self.p.lookback + 1))
        is_close_breakout = close_price > highest_recent_close

        if position.size == 0:
            if self.crossover > 0 and is_close_breakout:
                self.order = self.buy()
                print(f"[{bar_time}] BUY triggered @ {close_price:.2f} | Close > Last {self.p.lookback} bars")
        elif position.size > 0:
            if self.crossover < 0:
                self.order = self.sell()
                print(f"[{bar_time}] SELL triggered @ {close_price:.2f}")

    def notify_order(self, order):
        if order is None:
            self.order = None
            return

        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

class ORB3MinStrategy(bt.Strategy):
    params = dict(
        orb_wait_bars=5,            # 5 bars × 3 min = first 15 minutes
        log_file='orb3m_log.csv',
        size=1,
        market_close=time(15, 15),  # EOD exit time
    )

    def __init__(self):
        # track order / entry / range per data
        self.orders = {d: None for d in self.datas}
        self.entry_price = {d: None for d in self.datas}
        self.orb_high = {d: None for d in self.datas}
        self.orb_low = {d: None for d in self.datas}
        self.range_built = {d: False for d in self.datas}
        self.barcount = {d: 0 for d in self.datas}
        # if you want: trackers for logging
        if self.p.log_file:
            import os, csv
            if not os.path.exists(self.p.log_file):
                with open(self.p.log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Symbol', 'EntryTime', 'EntryPrice', 'ExitTime', 'ExitPrice', 'PnL'])

    def _get_dt_ist(self, data):
        dt = data.datetime.datetime(0)
        IST = pytz.timezone('Asia/Kolkata')
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        return dt.astimezone(IST)

    def next(self):
        for data in self.datas:
            # skip if there's active order
            if self.orders[data]:
                continue

            self.barcount[data] += 1
            close = data.close[0]
            high = data.high[0]
            low = data.low[0]
            pos = self.getposition(data).size
            dt_ist = self._get_dt_ist(data)
            t = dt_ist.time()

            # 1. Build range for first orb_wait_bars
            if self.barcount[data] <= self.p.orb_wait_bars:
                if self.orb_high[data] is None or high > self.orb_high[data]:
                    self.orb_high[data] = high
                if self.orb_low[data] is None or low < self.orb_low[data]:
                    self.orb_low[data] = low
                # do not trade yet
                continue

            # 2. After the range is built (i.e. barcount > wait bars), mark range built
            if not self.range_built[data]:
                # Only mark when both high & low exist
                if self.orb_high[data] is not None and self.orb_low[data] is not None:
                    self.range_built[data] = True
                    # optional logging
                    # print(f"[{dt_ist}] {data._name} ORB built: High={self.orb_high[data]}, Low={self.orb_low[data]}")
                else:
                    # If for some reason range is still incomplete (unlikely), skip
                    continue

            # 3. Entry logic (only if no position)
            if pos == 0 and self.range_built[data]:
                # if close of this candle > orb_high => long
                if close > self.orb_high[data]:
                    self.orders[data] = self.buy(data=data, size=self.p.size)
                    self.entry_price[data] = close
                    # print(f"[{dt_ist}] {data._name} ENTRY LONG @ {close:.2f}")
                # else optionally, you might also do short side if you want (close < low)
                # but per your description you only described breakout above then exit on below
                # so not doing short entry here.

            # 4. Exit logic (if in position)
            elif pos > 0:
                # exit when candle closes below orb_low
                if close < self.orb_low[data]:
                    self.orders[data] = self.sell(data=data, size=pos)
                    # print(f"[{dt_ist}] {data._name} EXIT @ {close:.2f}")

            # 5. End-of-day exit
            if t >= self.p.market_close and pos > 0:
                self.orders[data] = self.close(data=data)
                # print(f"[{dt_ist}] {data._name} EOD EXIT @ {close:.2f}")

    def notify_order(self, order):
        if order.status not in [order.Completed, order.Canceled, order.Rejected]:
            return

        data = order.data
        dt = self._get_dt_ist(data)
        name = data._name
        price = order.executed.price

        if order.isbuy():
            # bought
            # print(f"+++ {name} BUY EXECUTED @ {price:.2f}")
            pass
        elif order.issell():
            entry = self.entry_price.get(data, None)
            pnl = None
            if entry is not None:
                pnl = round(price - entry, 4)
            # print(f"+++ {name} SELL EXECUTED @ {price:.2f} | PnL: {pnl}")

            # Log trade
            if self.p.log_file:
                import csv
                with open(self.p.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, None, entry, dt, price, pnl])

        # clear order on any finish
        self.orders[data] = None

import backtrader as bt
import pandas as pd
import joblib
import numpy as np
from datetime import time


# Train ML separately, save as model.pkl using sklearn
# Features must match training phase!

# class SmaMLFilter(bt.Strategy):
#     params = dict(
#         sma3=4,
#         sma19=16,
#         lookback=1,
#         log_file='trades_log_ml.csv',
#         model_file='model.pkl',  # trained ML model
#         prob_threshold=0.6,
#         size=1,
#         market_close=time(15, 15),
#     )
#
#     def __init__(self):
#         self.orders = {}
#         self.sma3 = {}
#         self.sma19 = {}
#         self.crossovers = {}
#         self.choppiness = {}
#         self.crossover_flags = {}
#
#         # Load ML model
#         self.model = joblib.load(self.p.model_file)
#
#         for data in self.datas:
#             self.sma3[data] = bt.ind.SMA(data.close, period=self.p.sma3)
#             self.sma19[data] = bt.ind.SMA(data.close, period=self.p.sma19)
#             self.crossovers[data] = bt.ind.CrossOver(self.sma3[data], self.sma19[data])
#             self.choppiness[data] = ChoppinessIndex(data)
#             self.orders[data] = None
#             self.crossover_flags[data] = False
#
#     def _make_features(self, data):
#         """Extract features for ML model at current bar"""
#         sma3_val = self.sma3[data][0]
#         sma19_val = self.sma19[data][0]
#         chop_val = self.choppiness[data][0]
#         price = data.close[0]
#         ret_5 = (price - data.close[-5]) / data.close[-5] if len(data) > 5 else 0
#         hour = data.datetime.datetime(0).hour
#
#         return pd.DataFrame([{
#             "sma_diff": sma3_val - sma19_val,
#             "choppiness": chop_val,
#             "price": price,
#             "ret_5": ret_5,
#             "hour": hour,
#         }])
#
#     def next(self):
#         for data in self.datas:
#             if self.orders[data]:
#                 continue
#
#             crossover = self.crossovers[data][0]
#             position = self.broker.getposition(data)
#
#             # Update crossover flag
#             if crossover > 0:
#                 self.crossover_flags[data] = True
#             elif self.sma3[data][0] < self.sma19[data][0]:
#                 self.crossover_flags[data] = False
#
#             # Entry
#             if position.size == 0 and self.crossover_flags[data] and self.choppiness[data][0] < 50:
#                 # Build features for ML
#                 # X = self._make_features(data)
#                 # prob = self.model.predict_proba(X)[0][1]  # probability of profitable trade
#                 features = [[buy_hour, buy_minute, buy_price, sell_price, pnl]]
#                 prob = self.model.predict(features)[0]
#                 print(f"{data._name} → ML prob = {prob:.2f}")
#
#                 if prob >= self.p.prob_threshold:
#                     self.orders[data] = self.buy(data=data, size=self.p.size)
#                     print(f"BUY {data._name} @ {data.close[0]:.2f} (ML confirmed)")
#                 else:
#                     print(f"Trade skipped by ML filter ({prob:.2f})")
#
#             # Exit
#             elif position.size > 0 and self.sma3[data][0] < self.sma19[data][0]:
#                 self.orders[data] = self.sell(data=data, size=self.p.size)
#                 self.crossover_flags[data] = False
#                 print(f"SELL {data._name} @ {data.close[0]:.2f}")


class ADXCrossoverStrategyPEAKExits(bt.Strategy):
    params = dict(
        adx_period=14,
        adx_smooth=8,
        adx_threshold=21,

        log_file='adx_trades_log.csv',
        size=1,
        market_close=time(15, 15),  # cutoff for squaring off
    )

    def __init__(self):
        self.orders = dict()
        self.buy_prices = dict()
        self.buy_times = dict()
        self.plus_di = dict()
        self.minus_di = dict()
        self.adx = dict()
        self.win_count = 0
        self.loss_count = 0
        self.trades = 0
        self.total_points = 0
        self.crossovers = dict()
        self.atr = dict()
        self.entry_orders = dict()  # track entry order IDs
        self.active_positions = dict()  # map data -> entry_order_id
        self.last_adx_peak = {}
        self.di_fall_count = {}

        for data in self.datas:
            # ADX setup
            self.adx_threshold = self.p.adx_threshold
            self.plus_di[data] = bt.ind.PlusDI(data, period=self.p.adx_period)
            self.minus_di[data] = bt.ind.MinusDI(data, period=self.p.adx_period)
            self.adx[data] = bt.ind.AverageDirectionalMovementIndex(data, period=self.p.adx_period)
            self.crossovers[data] = bt.ind.CrossOver(self.plus_di[data], self.minus_di[data])
            self.orders[data] = None
            self.buy_prices[data] = None
            self.buy_times[data] = None
            self.last_adx_peak[data] = 0
            self.di_fall_count[data] = 0

        # Init log file
        if not os.path.exists(self.p.log_file):
            with open(self.p.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Symbol', 'Buy Time', 'Buy Price', 'Sell Time', 'Sell Price', 'PnL'])

    def _get_ist_dt_from_feed(self):
        """Return the datetime of the first data feed converted to IST."""
        dt = self.datas[0].datetime.datetime(0)
        IST = pytz.timezone('Asia/Kolkata')

        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
            dt = dt.astimezone(IST)
        elif dt.tzinfo != IST:
            dt = dt.astimezone(IST)
        return dt

    def next(self):
        # if not self.data.live:  # ignore all backfill bars
        #     return

        bar_dt_ist = self._get_ist_dt_from_feed()
        bar_hour = bar_dt_ist.hour
        bar_minute = bar_dt_ist.minute

        # 🛑 Square-off check after cutoff
        if (bar_hour > 15) or (bar_hour == 15 and bar_minute >= 15):
            for data in self.datas:
                pos = self.broker.getposition(data)
                if pos.size != 0:
                    logger.info(f"[{bar_dt_ist}] {data.tradingsymbol} > EOD square-off (IST)")
                    self.orders[data] = self.close(data=data)
                    print(f"{data.tradingsymbol} EOD SELL triggered @ {data.close[0]:.2f}")
            return

        # ✅ Trading logic
        for data in self.datas:
            if self.orders[data]:
                continue

            pos = self.broker.getposition(data)
            close_price = data.close[0]
            crossover = self.crossovers[data][0]
            adx_val = self.adx[data][0]
            plus_di = self.plus_di[data][0]
            minus_di = self.minus_di[data][0]
            # print(f"{data.tradingsymbol} Checking for entry and exit")
            # Entry condition: +DI cross above -DI
            if pos.size == 0:
                if crossover > 0 and self.adx[data][0] > self.adx_threshold:
                    print(f"Entry condition satisfied for {data.tradingsymbol}")
                    self.orders[data] = self.buy(data=data, size=self.p.size)
                    self.last_adx_peak[data] = adx_val  # reset peak at entry
                    self.di_fall_count[data] = 0
                    print(f"BUY {data.tradingsymbol} @ {close_price:.2f} | "
                          f"+DI {self.plus_di[data][0]:.2f} > -DI {self.minus_di[data][0]:.2f} "
                          f"(ADX={self.adx[data][0]:.2f})")
                    entry_price = data.close[0]


            # Exit condition: -DI cross above +DI
            elif pos.size > 0:
                # Track ADX peak
                if adx_val > self.last_adx_peak[data]:
                    self.last_adx_peak[data] = adx_val
                    self.di_fall_count[data] = 0
                elif plus_di < self.plus_di[data][-1]:  # +DI is weakening
                    self.di_fall_count[data] += 1
                print(f"OPEN POSITION CURRENT STATE : {data.tradingsymbol} @ CLOSE {close_price:.2f} | "
                      f"+DI {self.plus_di[data][0]:.2f} :: -DI {self.minus_di[data][0]:.2f} ::"
                      f"(ADX={self.adx[data][0]:.2f})")

                # Exit Rule #3: ADX turned down & +DI falling consistently
                if (adx_val < self.last_adx_peak[data] and self.di_fall_count[data] >= 2):
                    self.orders[data] = self.sell(data=data, size=self.p.size)
                    print(f"SELL {data._name} @ {close_price:.2f} | "
                          f"ADX peaked {self.last_adx_peak[data]:.2f} → {adx_val:.2f}, "
                          f"+DI falling for {self.di_fall_count[data]} bars")

                # Backup exit: EMA trail
                elif crossover < 0:  # close_price < self.ema[data][0]:
                    self.orders[data] = self.sell(data=data, size=self.p.size)
                    print(f"CROSSOVER SELL {data.tradingsymbol} @ {close_price:.2f} | "
                          f"-DI {self.minus_di[data][0]:.2f} > +DI {self.plus_di[data][0]:.2f} "
                          f"(ADX={self.adx[data][0]:.2f})")
                    # print(f"SELL {data._name} @ {close_price:.2f} | "
                    #       f"Close < EMA({self.p.ema_period}) {self.ema[data][0]:.2f}")
                # if crossover < 0:
                #     print(f"Exit condition satisfied for {data.tradingsymbol}")
                #     self.orders[data] = self.sell(data=data, size=self.p.size)
                #     print(f"SELL {data.tradingsymbol} @ {close_price:.2f} | "
                #           f"-DI {self.minus_di[data][0]:.2f} > +DI {self.plus_di[data][0]:.2f} "
                #           f"(ADX={self.adx[data][0]:.2f})")

    def notify_trade(self, trade):
        if trade.isclosed:  # only when trade closes
            self.trades += 1
            self.total_points += trade.pnlcomm
            if trade.pnlcomm > 0:
                self.win_count += 1
            else:
                self.loss_count += 1

    def notify_order(self, order):
        if order is None or order.status not in [order.Completed, order.Canceled, order.Rejected]:
            return

        data = order.data
        dt = data.datetime.datetime(0)
        name = data._name
        price = order.executed.price
        print("---")
        if order.isbuy():
            self.buy_prices[data] = price
            self.buy_times[data] = dt
            print(f"+++ {name} {data.tradingsymbol} BUY EXECUTED @ {price:.2f}")
        elif order.issell():
            buy_price = self.buy_prices.get(data, None)
            buy_time = self.buy_times.get(data, "NA")
            pnl = round(price - buy_price, 2) if buy_price is not None else "NA"
            print(f"+++ {name} {data.tradingsymbol} SELL EXECUTED @ {price:.2f} | PnL: {pnl}")

            # Log trade
            with open(self.p.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, buy_time, buy_price, dt, price, pnl])

            self.buy_prices[data] = None
            self.buy_times[data] = None

        self.orders[data] = None

class StochasticDivergenceStrategy(bt.Strategy):
    params = dict(
        stoch_period=14,
        stoch_dperiod=3,
        stoch_kperiod=3,
        stop_loss=0.02,  # 2% SL
        take_profit=0.04  # 4% TP
    )

    def __init__(self):
        # Dictionaries for each data feed
        self.stoch = {}
        self.last_price_high = {}
        self.last_price_low = {}
        self.last_stoch_high = {}
        self.last_stoch_low = {}
        self.sl_price = {}
        self.tp_price = {}
        self.orders = {}

        for data in self.datas:
            self.stoch[data] = bt.indicators.Stochastic(
                data,
                period=self.p.stoch_period,
                period_dfast=self.p.stoch_kperiod,
                period_dslow=self.p.stoch_dperiod
            )
            self.last_price_high[data] = None
            self.last_price_low[data] = None
            self.last_stoch_high[data] = None
            self.last_stoch_low[data] = None
            self.sl_price[data] = None
            self.tp_price[data] = None
            self.orders[data] = None

    def next(self):
        for data in self.datas:
            if self.orders[data]:
                continue  # wait for order completion

            price = data.close[0]
            stoch_val = self.stoch[data].percK[0]

            # --- Detect bullish divergence ---
            if self.last_price_low[data] and self.last_stoch_low[data]:
                if price < self.last_price_low[data] and stoch_val > self.last_stoch_low[data]:
                    self.orders[data] = self.buy(data=data, size=1)
                    self.sl_price[data] = price * (1 - self.p.stop_loss)
                    self.tp_price[data] = price * (1 + self.p.take_profit)
                    print(f"[{data._name}] BUY @ {price:.2f} | Bullish divergence")

            # --- Detect bearish divergence ---
            if self.last_price_high[data] and self.last_stoch_high[data]:
                if price > self.last_price_high[data] and stoch_val < self.last_stoch_high[data]:
                    self.orders[data] = self.sell(data=data, size=1)
                    self.sl_price[data] = price * (1 + self.p.stop_loss)
                    self.tp_price[data] = price * (1 - self.p.take_profit)
                    print(f"[{data._name}] SELL @ {price:.2f} | Bearish divergence")

            # Update reference points
            if stoch_val < 20:  # oversold = potential low
                self.last_price_low[data] = price
                self.last_stoch_low[data] = stoch_val

            if stoch_val > 80:  # overbought = potential high
                self.last_price_high[data] = price
                self.last_stoch_high[data] = stoch_val

            # Manage exits
            pos = self.getposition(data)
            if pos.size > 0:  # Long:  # Long
                if price <= self.sl_price[data] or price >= self.tp_price[data]:
                    self.close(data=data)
                    print(f"[{data._name}] EXIT LONG @ {price:.2f}")
                    self.orders[data] = None

            elif pos.size < 0:  # Short
                if price >= self.sl_price[data] or price <= self.tp_price[data]:
                    self.close(data=data)
                    print(f"[{data._name}] EXIT SHORT @ {price:.2f}")
                    self.orders[data] = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"+++ [{order.data._name}] BUY EXECUTED @ {order.executed.price:.2f}")
            elif order.issell():
                print(f"+++ [{order.data._name}] SELL EXECUTED @ {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"--- [{order.data._name}] Order failed")
        self.orders[order.data] = None

class ADXCrossoverStrategy(bt.Strategy):
    params = dict(
        adx_period=14,
        adx_smooth=8,
        adx_threshold=10,

        log_file='adx_trades_log.csv',
        size=1,
        market_close=time(15, 15),  # cutoff for squaring off
        max_loss_streak=6
    )

    def __init__(self):
        self.orders = dict()
        self.buy_prices = dict()
        self.buy_times = dict()
        self.plus_di = dict()
        self.minus_di = dict()
        self.adx = dict()
        self.win_count = 0
        self.loss_count = 0
        self.trades = 0
        self.total_points = 0
        self.crossovers = dict()
        self.atr = dict()
        self.entry_orders = dict()  # track entry order IDs
        self.active_positions = dict()  # map data -> entry_order_id
        self.crossover_flags = dict()
        self.consecutive_losses = 0  # 🔴 track losing streak
        self.trading_disabled = False  # 🔒 block new trades if streak breached
        self.choppiness = dict()

        for data in self.datas:
            # ADX setup
            self.adx_threshold = self.p.adx_threshold
            self.plus_di[data] = bt.ind.PlusDI(data, period=self.p.adx_period)
            self.minus_di[data] = bt.ind.MinusDI(data, period=self.p.adx_period)
            self.adx[data] = bt.ind.AverageDirectionalMovementIndex(data, period=self.p.adx_period)
            self.crossovers[data] = bt.ind.CrossOver(self.plus_di[data], self.minus_di[data])
            self.orders[data] = None
            self.buy_prices[data] = None
            self.buy_times[data] = None
            self.atr[data] = bt.ind.ATR(data, period=self.p.adx_period)
            self.choppiness[data] = ChoppinessIndex(data)

        # Init log file
        if not os.path.exists(self.p.log_file):
            with open(self.p.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Symbol', 'Buy Time', 'Buy Price', 'Sell Time', 'Sell Price', 'PnL'])

    def _get_ist_dt_from_feed(self):
        """Return the datetime of the first data feed converted to IST."""
        dt = self.datas[0].datetime.datetime(0)
        IST = pytz.timezone('Asia/Kolkata')

        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
            dt = dt.astimezone(IST)
        elif dt.tzinfo != IST:
            dt = dt.astimezone(IST)
        return dt

    def next(self):
        # if not self.data.live:  # ignore all backfill bars
        #     return

        bar_dt_ist = self._get_ist_dt_from_feed()
        bar_hour = bar_dt_ist.hour
        bar_minute = bar_dt_ist.minute

        # 🛑 Square-off check after cutoff
        if (bar_hour > 15) or (bar_hour == 15 and bar_minute >= 15):
            for data in self.datas:
                pos = self.broker.getposition(data)
                if pos.size != 0:
                    logger.info(f"[{bar_dt_ist}] {data.tradingsymbol} > EOD square-off (IST)")
                    self.orders[data] = self.close(data=data)
                    print(f"{data.tradingsymbol} EOD SELL triggered @ {data.close[0]:.2f}")
            return

        # ✅ Trading logic
        for data in self.datas:
            if self.orders[data]:
                continue

            pos = self.broker.getposition(data)
            close_price = data.close[0]
            crossover = self.crossovers[data][0]
            # choppiness = self.choppiness[data][0]
            # Entry condition: +DI cross above -DI
            if pos.size == 0:
                print(f"{data.tradingsymbol} : No open pos")
                print(f"------ {data.tradingsymbol} @ {close_price:.2f} | "
                      f"+DI {self.plus_di[data][0]:.2f} > -DI {self.minus_di[data][0]:.2f} "
                      f"(ADX={self.adx[data][0]:.2f}) ------")
                if crossover > 0 and self.adx[data][0] > self.adx_threshold:
                    print(f"{data.tradingsymbol} : Entry condition satisfied ")
                    self.orders[data] = self.buy(data=data, size=self.p.size)
                    print(f"+++++ BUY {data.tradingsymbol} @ {close_price:.2f} | "
                          f"+DI {self.plus_di[data][0]:.2f} > -DI {self.minus_di[data][0]:.2f} "
                          f"(ADX={self.adx[data][0]:.2f}) ------")
                    entry_price = data.close[0]


            # Exit condition: -DI cross above +DI
            elif pos.size > 0:
                print(f"CURRENT OPEN POSITION : {data.tradingsymbol} @ CLOSE {close_price:.2f} | "
                      f"+DI {self.plus_di[data][0]:.2f} :: -DI {self.minus_di[data][0]:.2f} ::"
                      f"(ADX={self.adx[data][0]:.2f})")
                diff = self.plus_di[data][0] - self.minus_di[data][0]
                if crossover < 0:
                    print(f"{data.tradingsymbol} : Exit condition satisfied ")
                    self.orders[data] = self.sell(data=data, size=self.p.size)
                    print(f"------ SELL {data.tradingsymbol} @ {close_price:.2f} | "
                          f"-DI {self.minus_di[data][0]:.2f} > +DI {self.plus_di[data][0]:.2f} "
                          f"(ADX={self.adx[data][0]:.2f}) ------")
                # elif diff>(self.atr[data][0] / data.close[0]) * 100: # new condition for DI gap widened
                # elif self.plus_di[data][0] > 28:
                #     print(f"{data.tradingsymbol} : Exit condition satisfied ")
                #     self.orders[data] = self.sell(data=data, size=self.p.size)
                #     print(f"------ DI DIFF - SELL {data.tradingsymbol} @ {close_price:.2f} | "
                #           f"-DI {self.minus_di[data][0]:.2f} > +DI {self.plus_di[data][0]:.2f} "
                #           f"(ADX={self.adx[data][0]:.2f}) ------")

    def notify_trade(self, trade):
        if trade.isclosed:  # only when trade closes
            self.trades += 1
            self.total_points += trade.pnlcomm
            if trade.pnlcomm > 0:
                self.win_count += 1
                self.consecutive_losses = 0  # reset on win
            else:
                self.loss_count += 1
                self.consecutive_losses += 1  # increment on loss

                if self.consecutive_losses > self.p.max_loss_streak:
                    self.trading_disabled = True
                    print(f"🚨 Trading disabled due to {self.consecutive_losses} consecutive losses")

    def notify_order(self, order):
        if order is None or order.status not in [order.Completed, order.Canceled, order.Rejected]:
            return

        data = order.data
        dt = data.datetime.datetime(0)
        name = data._name
        price = order.executed.price
        print("---")
        if order.isbuy():
            self.buy_prices[data] = price
            self.buy_times[data] = dt
            print(f"+++ {name} {data.tradingsymbol} BUY EXECUTED @ {price:.2f}")
        elif order.issell():
            buy_price = self.buy_prices.get(data, None)
            buy_time = self.buy_times.get(data, "NA")
            pnl = round(price - buy_price, 2) if buy_price is not None else "NA"
            print(f"+++ {name} {data.tradingsymbol} SELL EXECUTED @ {price:.2f} | PnL: {pnl}")

            # Log trade
            with open(self.p.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, buy_time, buy_price, dt, price, pnl])

            self.buy_prices[data] = None
            self.buy_times[data] = None

        self.orders[data] = None


class ATRStrategy(bt.Strategy):
    params = dict(
        adx_period=8,
        atr_period=6,
        risk_atr=1.5,  # stop = 1.5 ATR below/above entry
        reward_atr=1.6,  # tp   = 3.0 ATR above/below entry
        size=1,
    )

    def __init__(self):
        # Keep separate indicators per instrument
        self.adx = {}
        self.di_plus = {}
        self.di_minus = {}
        self.atr = {}
        self.orders = {}
        self.sl = {}
        self.tp = {}
        self.crossovers = {}

        for data in self.datas:
            self.adx[data] = bt.ind.ADX(data, period=self.p.adx_period)
            self.di_plus[data] = bt.ind.PlusDI(data, period=self.p.adx_period)
            self.di_minus[data] = bt.ind.MinusDI(data, period=self.p.adx_period)
            self.atr[data] = bt.ind.ATR(data, period=self.p.atr_period)
            self.adx[data] = bt.ind.AverageDirectionalMovementIndex(data, period=self.p.adx_period)
            self.crossovers[data] = bt.ind.CrossOver(self.di_plus[data], self.di_minus[data])
            self.orders[data] = None
            self.sl[data] = None
            self.tp[data] = None

    def next(self):
        for data in self.datas:
            pos = self.getposition(data)
            crossover = self.crossovers[data][0]
            # ENTRY: +DI > -DI and ADX > 20
            if not pos and crossover > 0 and self.adx[data][0] > 14:
                entry_price = data.close[0]
                self.orders[data] = self.buy(data=data, size=self.p.size)

                # set SL/TP
                self.sl[data] = entry_price - self.p.risk_atr * self.atr[data][0]
                self.tp[data] = entry_price + self.p.reward_atr * self.atr[data][0]
                self.log(f"BUY {data._name} at {entry_price}, SL {self.sl[data]}, TP {self.tp[data]}")

            elif pos:
                price = data.close[0]

                # Stop Loss
                if self.sl[data] and price <= self.sl[data]:
                    self.close(data=data)
                    self.sl[data], self.tp[data] = None, None
                    self.log(f"Stop Loss hit {data._name} at {price}")

                # Take Profit
                elif self.tp[data] and price >= self.tp[data]:
                    self.close(data=data)
                    self.sl[data], self.tp[data] = None, None
                    self.log(f"Take Profit hit {data._name} at {price}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED {order.data._name} at {order.executed.price}")
            elif order.issell():
                self.log(f"SELL EXECUTED {order.data._name} at {order.executed.price}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order failed for {order.data._name}: {order.getstatusname()}")

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"{dt} - {txt}")

class Sma2Sma7CrossoverBarCheckMultiTP(bt.Strategy):
    params = dict(
        sma3=3,
        sma7=8,
        sma19=18,
        lookback=1,
        log_file='trades_log.csv',
        size=1,
        trail_percent=0.04,  # <-- NEW: trailing stop % (1% by default)
        market_close=time(15, 15),
    )

    def __init__(self):
        self.orders = dict()
        self.buy_prices = dict()
        self.buy_times = dict()
        self.highest_prices = dict()  # track highest since entry
        self.sma3 = dict()
        self.sma7 = dict()
        self.sma19 = dict()
        self.crossovers = dict()
        self.crossover_flags = dict()

        for data in self.datas:
            self.sma3[data] = bt.ind.SMA(data.close, period=self.p.sma3)
            self.sma7[data] = bt.ind.SMA(data.close, period=self.p.sma7)
            self.sma19[data] = bt.ind.SMA(data.close, period=self.p.sma19)
            self.crossovers[data] = bt.ind.CrossOver(self.sma3[data], self.sma7[data])
            self.orders[data] = None
            self.buy_prices[data] = None
            self.buy_times[data] = None
            self.highest_prices[data] = None
            self.crossover_flags[data] = False

        # Init log file
        if not os.path.exists(self.p.log_file):
            with open(self.p.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Symbol', 'Buy Time', 'Buy Price', 'Sell Time', 'Sell Price', 'PnL'])

    def _get_ist_dt_from_feed(self):
        """Return the datetime of the first data feed converted to IST."""

        dt = self.datas[0].datetime.datetime(0)
        IST = pytz.timezone('Asia/Kolkata')

        if dt.tzinfo is None:
            # CSV has no tz → assume UTC
            dt = pytz.utc.localize(dt)
            dt = dt.astimezone(IST)
        elif dt.tzinfo != IST:
            # If tz-aware but not IST → convert
            dt = dt.astimezone(IST)
        # If already IST → do nothing
        return dt

    def next(self):
        bar_dt_ist = self._get_ist_dt_from_feed()
        bar_time = self.datas[0].datetime.datetime(0)

        # Market cutoff square-off
        if (bar_dt_ist.hour > 15) or (bar_dt_ist.hour == 15 and bar_dt_ist.minute >= 15):
            for data in self.datas:
                pos = self.broker.getposition(data)
                if pos.size > 0:
                    self.orders[data] = self.sell(data=data, size=pos.size)
                    print(f"{data._name} >>> EOD Square-off @ {data.close[0]:.2f}")
            return

        # Main trading loop
        for data in self.datas:
            position = self.broker.getposition(data)
            close_price = data.close[0]
            crossover = self.crossovers[data][0]

            # Track crossover state
            if crossover > 0 or (self.sma3[data][0] > self.sma7[data][0]):
                self.crossover_flags[data] = True
            elif crossover < 0 or (self.sma3[data][0] < self.sma7[data][0]):
                self.crossover_flags[data] = False

            # ENTRY
            if position.size == 0 and self.crossover_flags[data]:
                if self.sma7[data][0] > self.sma19[data][0]:
                    self.orders[data] = self.buy(data=data, size=self.p.size)
                    self.crossover_flags[data] = False
                    self.highest_prices[data] = close_price
                    print(f"[{bar_time}] {data._name} BUY @ {close_price:.2f}")

            # EXIT
            elif position.size > 0:
                # update highest price since entry
                self.highest_prices[data] = max(self.highest_prices[data] or close_price, close_price)

                # percent-based trailing stop
                trail_stop = self.highest_prices[data] * (1 - self.p.trail_percent)

                # exit if price drops below trail stop OR fallback SMA cross
                if close_price < trail_stop or (crossover < 0):
                    self.orders[data] = self.sell(data=data, size=position.size)
                    self.crossover_flags[data] = False
                    print(f"[{bar_time}] {data._name} SELL @ {close_price:.2f} | TrailStop={trail_stop:.2f}")
                    self.highest_prices[data] = None

    def notify_order(self, order):
        if order is None or order.status not in [order.Completed, order.Canceled, order.Rejected]:
            return

        data = order.data
        dt = data.datetime.datetime(0)
        price = order.executed.price

        if order.isbuy():
            self.buy_prices[data] = price
            self.buy_times[data] = dt
            print(f"+++ {data._name} BUY EXECUTED @ {price:.2f}")
        elif order.issell():
            buy_price = self.buy_prices.get(data, None)
            buy_time = self.buy_times.get(data, "NA")
            pnl = round(price - buy_price, 2) if buy_price is not None else "NA"
            print(f"+++ {data._name} SELL EXECUTED @ {price:.2f} | PnL: {pnl}")

            with open(self.p.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([data._name, buy_time, buy_price, dt, price, pnl])

            self.buy_prices[data] = None
            self.buy_times[data] = None
            self.highest_prices[data] = None

        self.orders[data] = None

class Sma2Sma7CrossoverBarCheckMultiTakeProfit(bt.Strategy):
    params = dict(
        sma3=3,
        sma7=8,
        sma19=18,  # Added for trend filter
        lookback=1,
        log_file='trades_log.csv',
        size=1,
        market_close=time(15, 15),  # cuttoff param
    )

    def __init__(self):
        self.orders = dict()
        self.buy_prices = dict()
        self.sma3 = dict()
        self.sma7 = dict()
        self.sma19 = dict()
        self.crossovers = dict()
        self.crossover_flags = dict()
        self.buy_times = dict()
        self.size = 1
        self.profit_points_to_take = 10

        for data in self.datas:
            self.sma3[data] = bt.ind.SMA(data.close, period=self.p.sma3)
            self.sma7[data] = bt.ind.SMA(data.close, period=self.p.sma7)
            self.sma19[data] = bt.ind.SMA(data.close, period=self.p.sma19)
            self.crossovers[data] = bt.ind.CrossOver(self.sma3[data], self.sma7[data])
            self.orders[data] = None
            self.buy_prices[data] = None
            self.buy_times[data] = None
            # self.crossover_flags[data] = False
            self.market_cutoff = True
            self.entry_prices = {data: None for data in self.datas}  # Track entry prices
            self.in_long = {data: False for data in self.datas}  # Track if currently in long
            self.crossover_flags = {data: False for data in self.datas}

        # Init log file
        if not os.path.exists(self.p.log_file):
            with open(self.p.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Symbol', 'Buy Time', 'Buy Price', 'Sell Time', 'Sell Price', 'PnL'])

    def _get_ist_dt_from_feed(self):
        """Return the datetime of the first data feed converted to IST."""

        dt = self.datas[0].datetime.datetime(0)
        IST = pytz.timezone('Asia/Kolkata')

        if dt.tzinfo is None:
            # CSV has no tz → assume UTC
            dt = pytz.utc.localize(dt)
            dt = dt.astimezone(IST)
        elif dt.tzinfo != IST:
            # If tz-aware but not IST → convert
            dt = dt.astimezone(IST)
        # If already IST → do nothing
        return dt

    def next(self):
        bar_dt_ist = self._get_ist_dt_from_feed()
        bar_hour = bar_dt_ist.hour
        bar_minute = bar_dt_ist.minute

        # --------------------
        # End-of-day square-off
        # --------------------
        if self.market_cutoff:
            if (bar_hour > 15) or (bar_hour == 15 and bar_minute >= 15):
                for data in self.datas:
                    pos = self.broker.getposition(data)
                    if pos.size != 0:
                        print(f"[{bar_dt_ist}] > Triggering end-of-day square-off (IST)")
                        self.orders[data] = self.sell(data=data)
                        self.crossover_flags[data] = False
                        self.in_long[data] = False
                        print(f"{data.tradingsymbol} SELL triggered @ {data.close[0]:.2f}")
                return

        # --------------------
        # Main loop over data feeds
        # --------------------
        for data in self.datas:
            if self.orders[data]:  # Skip if order already pending
                continue

            close_price = data.close[0]
            bar_time = data.datetime.datetime(0)
            crossover = self.crossovers[data][0]
            position = self.broker.getposition(data)

            # --------------------
            # Crossover flag logic
            # --------------------
            # Only set flag for BUY if we are not already in a long trade
            if crossover > 0 and not self.in_long[data]:
                self.crossover_flags[data] = True
            elif crossover < 0:
                self.crossover_flags[data] = False

            # --------------------
            # ENTRY
            # --------------------
            if position.size == 0:
                if self.crossover_flags[data]:
                    # SMA confirmation
                    if self.sma3[data][0] > self.sma19[data][0] and self.sma7[data][0] > self.sma19[data][0]:
                        self.orders[data] = self.buy(data=data, size=self.size)
                        self.entry_prices[data] = close_price  # Track entry price
                        self.crossover_flags[data] = False  # Reset until new crossover
                        self.in_long[data] = True  # Mark as in a trade
                        print(f"[{bar_time}] {data.tradingsymbol} BUY triggered @ {close_price:.2f}")

            # --------------------
            # EXIT
            # --------------------
            elif position.size > 0:
                entry_price = self.entry_prices.get(data, None)

                # Condition 1: Opposite crossover
                if crossover < 0:
                    self.orders[data] = self.sell(data=data, size=self.size)
                    self.crossover_flags[data] = False
                    self.in_long[data] = False
                    print(f"[{bar_time}] {data.tradingsymbol} SELL (crossover) @ {close_price:.2f}")

                # Condition 2: 10-point profit target
                elif entry_price is not None and close_price >= entry_price + 12:
                    self.orders[data] = self.sell(data=data, size=self.size)
                    self.crossover_flags[data] = False
                    self.in_long[data] = False
                    print(
                        f"[{bar_time}] {data.tradingsymbol} SELL (target hit {self.profit_points_to_take}) @ {close_price:.2f}")

                # ✅ Condition 3: 5-point stop loss
                elif entry_price is not None and close_price <= entry_price - 10:
                    self.orders[data] = self.sell(data=data, size=self.size)
                    self.crossover_flags[data] = False
                    self.in_long[data] = False
                    print(f"[{bar_time}] {data.tradingsymbol} SELL (stop loss) @ {close_price:.2f}")

    def notify_order(self, order):
        if order is None or order.status not in [order.Completed, order.Canceled, order.Rejected]:
            return

        data = order.data
        dt = data.datetime.datetime(0)
        name = data._name
        price = order.executed.price

        if order.isbuy():
            self.buy_prices[data] = price
            self.buy_times[data] = dt
            print(f"+++ {name} BUY EXECUTED @ {price:.2f}")
        elif order.issell():
            buy_price = self.buy_prices.get(data, None)
            buy_time = self.buy_times.get(data, "NA")
            pnl = round(price - buy_price, 2) if buy_price is not None else "NA"
            print(f"+++ {name} SELL EXECUTED @ {price:.2f} | PnL: {pnl}")

            # Log trade
            with open(self.p.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, buy_time, buy_price, dt, price, pnl])

            self.buy_prices[data] = None
            self.buy_times[data] = None

        self.orders[data] = None

class SupertrendPullback(bt.Strategy):
    params = dict(period=7, multiplier=1.7)

    def __init__(self):
        self.st = Supertrend(period=self.p.period, multiplier=self.p.multiplier)
        self.buy_price = None

    def next(self):
        print(self.data.close[0])
        print(self.st.lowerband[0])
        print(self.st.upperband[0])
        print(self.st.trend[0])
        if not self.position:
            print(self.data.close[0])
            print(self.st.lowerband[0])
            print(self.st.upperband[0])
            print(self.st.trend[0])
            # Enter if price touches lowerband in uptrend
            if self.data.close[0] <= self.st.lowerband[0] and self.st.trend[0] == 1:
                self.buy_price = self.data.close[0]
                self.buy(self.data)
                print(f"BUY at {self.buy_price}")
        else:
            # Exit if price touches upperband in downtrend
            if self.data.close[0] >= self.st.upperband[0] and self.st.trend[0] == -1:
                print(f"SELL at {self.data.close[0]}")
                self.sell(self.data)
                self.buy_price = None

class PrintSMAStrategy(bt.Strategy):
    def __init__(self):
        self.last_print_time = None
        self.sma = bt.ind.SMA(self.data.close, period=self.p.sma_period)

    params = (
        ('sma_period', 1),
    )

    def next(self):
        print(
            f"{self.data.datetime.datetime(0)} | Close: {self.data.close[0]} | SMA({self.p.sma_period}): {self.sma[0]}")

import backtrader as bt
import math


class SuperTrend(bt.Indicator):
    lines = ('supertrend', 'direction')
    params = (('period', 10), ('multiplier', 3))

    plotlines = dict(
        supertrend=dict(color='green', linewidth=2),
        direction=dict(_plotskip=True),
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.period)
        self.addminperiod(self.p.period + 1)

        # store bands
        self.final_upper = [math.nan]
        self.final_lower = [math.nan]

    def next(self):
        hl2 = (self.data.high[0] + self.data.low[0]) / 2
        upperband = hl2 + self.p.multiplier * self.atr[0]
        lowerband = hl2 - self.p.multiplier * self.atr[0]

        # previous values
        prev_close = self.data.close[-1] if len(self) > 1 else self.data.close[0]
        prev_st = self.lines.supertrend[-1] if len(self) > 1 else math.nan
        prev_final_upper = self.final_upper[-1] if len(self.final_upper) > 0 else math.nan
        prev_final_lower = self.final_lower[-1] if len(self.final_lower) > 0 else math.nan

        # Final upperband
        if math.isnan(prev_final_upper):
            final_upper = upperband
        elif (upperband < prev_final_upper) or (prev_close > prev_final_upper):
            final_upper = upperband
        else:
            final_upper = prev_final_upper

        # Final lowerband
        if math.isnan(prev_final_lower):
            final_lower = lowerband
        elif (lowerband > prev_final_lower) or (prev_close < prev_final_lower):
            final_lower = lowerband
        else:
            final_lower = prev_final_lower

        # Supertrend direction
        if math.isnan(prev_st):
            # init
            st = final_lower if self.data.close[0] >= final_lower else final_upper
            direction = 1 if self.data.close[0] >= final_lower else -1
        elif prev_st == prev_final_upper:
            if self.data.close[0] <= final_upper:
                st = final_upper
                direction = -1
            else:
                st = final_lower
                direction = 1
        elif prev_st == prev_final_lower:
            if self.data.close[0] >= final_lower:
                st = final_lower
                direction = 1
            else:
                st = final_upper
                direction = -1
        else:
            st = final_lower if self.data.close[0] >= final_lower else final_upper
            direction = 1 if st == final_lower else -1

        # store results
        self.final_upper.append(final_upper)
        self.final_lower.append(final_lower)
        self.lines.supertrend[0] = st
        self.lines.direction[0] = direction


class SuperTrendBLOGIND(bt.Indicator):
    """
    SuperTrend Algorithm :

        BASIC UPPERBAND = (high + low) / 2 + Multiplier * ATR
        BASIC lowERBAND = (high + low) / 2 - Multiplier * ATR

        FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous close > Previous FINAL UPPERBAND))
                            THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
        FINAL lowERBAND = IF( (Current BASIC lowERBAND > Previous FINAL lowERBAND) or (Previous close < Previous FINAL lowERBAND))
                            THEN (Current BASIC lowERBAND) ELSE Previous FINAL lowERBAND)
        SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current close <= Current FINAL UPPERBAND)) THEN
                        Current FINAL UPPERBAND
                    ELSE
                        IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current close > Current FINAL UPPERBAND)) THEN
                            Current FINAL lowERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL lowERBAND) and (Current close >= Current FINAL lowERBAND)) THEN
                                Current FINAL lowERBAND
                            ELSE
                                IF((Previous SUPERTREND = Previous FINAL lowERBAND) and (Current close < Current FINAL lowERBAND)) THEN
                                    Current FINAL UPPERBAND

    """

    lines = ('super_trend',)
    params = (('period', 8),
              ('multiplier', 2),
              )
    plotlines = dict(
        super_trend=dict(
            _name='ST',
            color='blue',
            alpha=1
        ))

    plotinfo = dict(subplot=False)

    def __init__(self):
        self.st = [0]
        self.finalupband = [0]
        self.finallowband = [0]
        self.addminperiod(self.p.period)
        atr = bt.ind.ATR(self.data, period=self.p.period)
        self.upperband = (self.data.high + self.data.low) / 2 + self.p.multiplier * atr
        self.lowerband = (self.data.high + self.data.low) / 2 - self.p.multiplier * atr

    def next(self):

        pre_upband = self.finalupband[0]
        pre_lowband = self.finallowband[0]

        if self.upperband[0] < self.finalupband[-1] or self.data.close[-1] > self.finalupband[-1]:
            self.finalupband[0] = self.upperband[0]

        else:
            self.finalupband[0] = self.finalupband[-1]

        if self.lowerband[0] > self.finallowband[-1] or self.data.close[-1] < self.finallowband[-1]:

            self.finallowband[0] = self.lowerband[0]

        else:
            self.finallowband[0] = self.finallowband[-1]

        if self.data.close[0] <= self.finalupband[0] and ((self.st[-1] == pre_upband)):

            self.st[0] = self.finalupband[0]
            self.lines.super_trend[0] = self.finalupband[0]

        elif (self.st[-1] == pre_upband) and (self.data.close[0] > self.finalupband[0]):

            self.st[0] = self.finallowband[0]
            self.lines.super_trend[0] = self.finallowband[0]

        elif (self.st[-1] == pre_lowband) and (self.data.close[0] >= self.finallowband[0]):

            self.st[0] = self.finallowband[0]
            self.lines.super_trend[0] = self.finallowband[0]

        elif (self.st[-1] == pre_lowband) and (self.data.close[0] < self.finallowband[0]):

            self.st[0] = self.finalupband[0]
            self.lines.super_trend[0] = self.st[0]


# --------------------------
# Strategy
# --------------------------
class SupertrendPullbackStrategy(bt.Strategy):
    params = dict(
        log_file="supertrend_trades_log.csv",
        size=1,
        market_close=time(15, 15),
    )

    def __init__(self):
        self.orders = {}
        self.buy_prices = {}
        self.buy_times = {}
        self.in_position = {}  # ✅ Track open/close manually

        self.sma8 = {}
        self.supertrend = {}
        self.supertrend1 = {}
        self.ema30 = {}

        for data in self.datas:
            print("----")
            print(data)
            # calculate SuperTrend(14,3)

            # self.supertrend[data] = compute_supertrend(data, lookback=8, multiplier=2)
            self.supertrend[data] = SuperTrend(data, period=8, multiplier=2)
            self.supertrend1[data] = SuperTrend(data, period=7, multiplier=3.6)
            # self.supertrend[data] = SuperTrendBLOGIND(data)
            # self.supertrend[data] = ta.supertrend(data["High"], data["Low"], data["Close"], length=10, multiplier=3)
            self.ema30[data] = bt.ind.EMA((data.open + data.high + data.low + data.close) / 4, period=30)
            self.sma8[data] = bt.ind.SMA(data.close, period=8)

            self.orders[data] = None
            self.buy_prices[data] = None
            self.buy_times[data] = None
            self.in_position[data] = False  # initially not in trade

    def _get_ist_dt_from_feed(self):
        """Return the datetime of the *first* data feed converted to IST.

        Handles both cases:
         - Backtrader returned naive datetime representing UTC (common),
         - or Backtrader returned tz-aware datetime (less common).
        """
        dt = self.datas[0].datetime.datetime(0)

        IST = pytz.timezone('Asia/Kolkata')

        # If naive -> assume it is UTC and localize; otherwise assume it's tz-aware already.
        if dt.tzinfo is None:
            # Common case: Backtrader gives naive but CSV had tz info converted to UTC
            dt = pytz.utc.localize(dt)
        # Convert to IST
        return dt.astimezone(IST)

    def next(self):
        bar_dt_ist = self._get_ist_dt_from_feed()

        for data in self.datas:
            if self.orders[data]:  # pending order, wait
                continue

            st_dir = self.supertrend[data].direction[0]

            close_price = data.close[0]
            sma_val = self.sma8[data][0]
            # print(f"SP {data._name} {bar_dt_ist} : {self.supertrend[data].supertrend[0]}")
            # ✅ Entry condition
            # if st_dir > 0:
            #     print(f"UP SUPERTREND {data._name} {self.supertrend[data].supertrend[0]} at {bar_dt_ist} {st_dir} at {close_price} closing")
            # else:
            #     print(f"DOWN SUPERTREND {data._name} {self.supertrend[data].supertrend[0]}  at {bar_dt_ist} {st_dir} at {close_price} closing")
            if not self.in_position[data] and st_dir > 0:
                print(f"UP SUPERTREND {data._name} at {bar_dt_ist} {st_dir} at {close_price} closing")
                print(f"SMA * {sma_val} Close {data.close[0]} Open {data.open[0]} LOW {data.low[0]}")
                if data.close[0] < sma_val:  # or data.low[0] < sma_val or data.open[0] < sma_val:
                    # print(f"++++ SUPERTREND SATISFIED ENTRY {bar_dt_ist} {st_dir} at {close_price} closing")
                    # print(f"SMA * {sma_val} Close {data.close[0]} Open {data.open[0]} LOW {data.low[0]}")
                    self.orders[data] = self.buy(data=data, size=self.p.size)
                    self.buy_prices[data] = close_price
                    self.buy_times[data] = bar_dt_ist
                    print(f"+++++ [{bar_dt_ist}] BUY {data._name} @ {close_price:.2f}")
                    self.in_position[data] = True

            # ✅ Exit condition
            # elif self.in_position[data] and close_price < self.supertrend[data].supertrend[0]:
            elif self.in_position[data] and data.close[0] > self.supertrend1[data].supertrend[0]:  # or\
                # data.close[0] > self.supertrend1[data].supertrend[0] or data.open[0] > self.supertrend1[data].supertrend[0]:
                self.orders[data] = self.sell(data=data, size=self.p.size)
                print(
                    f"----- TARGET HIT [{bar_dt_ist}] SELL {data._name} @ {close_price:.2f} {self.supertrend1[data].supertrend[0]}")
                self.in_position[data] = False
            # Stop loss
            elif self.in_position[data] and data.close[0] < self.supertrend[data].supertrend[0]:  # or\
                # data.close[0] > self.supertrend1[data].supertrend[0] or data.open[0] > self.supertrend1[data].supertrend[0]:
                self.orders[data] = self.sell(data=data, size=self.p.size)
                print(
                    f"----- STOP LOSS [{bar_dt_ist}] SELL {data._name} @ {close_price:.2f} {self.supertrend[data].supertrend[0]}")
                self.in_position[data] = False

    def notify_order(self, order):
        if order is None:
            # Zerodha broker is sometimes calling notify_order(None)
            return

        if order is not None:
            if order.status not in [order.Completed]:
                return

        data = order.data
        dt = data.datetime.datetime(0)
        name = data._name
        price = order.executed.price

        if order.isbuy():
            self.in_position[data] = True  # ✅ Mark as in position
            print(f"+++ {name} BUY EXECUTED @ {price:.2f}")
        elif order.issell():
            self.in_position[data] = False  # ✅ Mark as flat
            buy_price = self.buy_prices.get(data, None)
            pnl = round(price - buy_price, 2) if buy_price else "NA"
            print(f"+++ {name} SELL EXECUTED @ {price:.2f} | PnL: {pnl}")

            with open(self.p.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, self.buy_times.get(data), buy_price, dt, price, pnl])

        self.orders[data] = None


class IntradayPullbackOptions(bt.Strategy):
    params = dict(
        trend_ema_period=200,
        pullback_ema_period=20,
        size=1,
        market_close=time(15, 15),
    )

    def __init__(self):
        # Data mapping
        self.data_index = self.datas[0]  # underlying index (spot/fut)
        self.data_ce = self.datas[1]  # Call option
        self.data_pe = self.datas[2]  # Put option

        # Trend filter on 30m resampled index data
        self.data_index30 = self.datas[3]
        self.trend_ema = bt.ind.EMA(self.data_index30.close, period=self.p.trend_ema_period)

        # Pullback filter on 5m index data
        self.pullback_ema = bt.ind.EMA(self.data_index.close, period=self.p.pullback_ema_period)

        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.active_data = None  # CE or PE currently traded

    def next(self):
        dt = self.data_index.datetime.datetime(0)

        # Square-off before close
        if dt.time() >= self.p.market_close:
            if self.position:
                self.close(data=self.active_data)
                print(f"{dt} Square-off {self.active_data._name} @ {self.active_data.close[0]:.2f}")
            return

        if self.order:
            return

        pos = self.getposition(self.data_ce).size or self.getposition(self.data_pe).size

        # ---------------------------
        # Entry Logic
        # ---------------------------
        if not pos:
            # Trend check
            uptrend = self.data_index.close[0] > self.trend_ema[0]
            downtrend = self.data_index.close[0] < self.trend_ema[0]

            # Pullback check
            price_above = self.data_index.close[0] > self.pullback_ema[0]
            price_below = self.data_index.close[0] < self.pullback_ema[0]

            if uptrend and price_above:
                self.active_data = self.data_ce
                self.entry_price = self.active_data.close[0]
                self.stop_price = min(self.active_data.low.get(size=3))
                self.order = self.buy(data=self.active_data, size=self.p.size)
                print(f"{dt} BUY CE {self.active_data._name} @ {self.entry_price:.2f}, SL={self.stop_price:.2f}")

            elif downtrend and price_below:
                self.active_data = self.data_pe
                self.entry_price = self.active_data.close[0]
                self.stop_price = min(self.active_data.low.get(size=3))
                self.order = self.buy(data=self.active_data, size=self.p.size)
                print(f"{dt} BUY PE {self.active_data._name} @ {self.entry_price:.2f}, SL={self.stop_price:.2f}")

        # ---------------------------
        # Exit Logic
        # ---------------------------
        else:
            pos = self.getposition(self.active_data)
            if pos.size > 0:
                rr_target = self.entry_price + (self.entry_price - self.stop_price)

                if self.active_data.close[0] >= rr_target:
                    self.close(data=self.active_data)
                    print(f"{dt} Target hit {self.active_data._name} @ {self.active_data.close[0]:.2f}")

                elif self.active_data.close[0] < self.stop_price:
                    self.close(data=self.active_data)
                    print(f"{dt} Stoploss hit {self.active_data._name} @ {self.active_data.close[0]:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

class Sma3_18_Cross(bt.Strategy):
    params = dict(
        sma3=4,
        sma19=16,  # Added for trend filter
        lookback=1,
        log_file='trades_log.csv',
        size=1,
        market_close=time(15, 15),  # cuttoff param
        choppiness_threshold = 50
    )

    def __init__(self):
        self.orders = dict()
        self.buy_prices = dict()
        self.sma3 = dict()
        self.sma19 = dict()
        self.crossovers = dict()
        self.crossover_flags = dict()
        self.buy_times = dict()
        self.fco = dict()
        self.size = 1
        self.choppiness = dict()
        self.choppiness_threshold = dict()

        for data in self.datas:
            self.sma3[data] = bt.ind.SMA(data.close, period=self.p.sma3)
            self.sma19[data] = bt.ind.SMA(data.close, period=self.p.sma19)
            self.choppiness[data] = ChoppinessIndex(data)
            self.crossovers[data] = bt.ind.CrossOver(self.sma3[data], self.sma19[data])
            self.orders[data] = None
            self.buy_prices[data] = None
            self.buy_times[data] = None
            self.crossover_flags[data] = False
            self.market_cutoff = True  # MAKE THIS FALSE AS IT IS MALFUNCTIONING WITH LIVE FEED
            self.choppiness_threshold[data] = 50

        # Init log file
        if not os.path.exists(self.p.log_file):
            with open(self.p.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Symbol', 'Buy Time', 'Buy Price', 'Sell Time', 'Sell Price', 'PnL'])

    def _get_ist_dt_from_feed(self):
        """Return the datetime of the first data feed converted to IST."""

        dt = self.datas[0].datetime.datetime(0)
        IST = pytz.timezone('Asia/Kolkata')

        if dt.tzinfo is None:
            # CSV has no tz → assume UTC
            dt = pytz.utc.localize(dt)
            dt = dt.astimezone(IST)
        elif dt.tzinfo != IST:
            # If tz-aware but not IST → convert
            dt = dt.astimezone(IST)
        # If already IST → do nothing
        return dt

    def _get_current_ist(self):
        """Return current system time in IST (Asia/Kolkata)."""
        IST = pytz.timezone("Asia/Kolkata")
        return datetime.now(IST)

    def next(self):
        print("Starting checks")
        # if not self.data.live:  # skip all backfill bars
        #     print("Filling in the bars, refrain from entering a trade until live feed appears")
        #     return
        # for non live feed
        bar_dt_ist = self._get_ist_dt_from_feed()
        bar_hour = bar_dt_ist.hour
        bar_minute = bar_dt_ist.minute
        now_ist = self._get_current_ist()

        # close for live feed
        # if now_ist.time() >= self.p.market_close:
        #     print("Current time greater than 3:15 PM, trying to close all positions !!")
        #     for data in self.datas:
        #         pos = self.broker.getposition(data)
        #         print(f"Open position : {pos}")
        #         if pos.size != 0:
        #             print(f"[{now_ist}] > Auto square-off triggered for {data._name}")
        #             self.orders[data] = self.sell(data=data, size=self.size)  # safer than self.sell, works for long/short
        #             self.crossover_flags[data] = False
        #             print(f"{data.tradingsymbol} SELL triggered @ {data.close[0]:.2f}")
        #             print(f"[{bar_dt_ist}] {data.tradingsymbol} - issued square-off close (size={pos.size})")
        #     return  # stop further entries after cutoff

        # for non live feed
        if self.market_cutoff:
            if (bar_hour > 15) or (bar_hour == 15 and bar_minute >= 15):
                for data in self.datas:
                    pos = self.broker.getposition(data)
                    if pos.size != 0:
                        logger.info(f"[{bar_dt_ist}] > Triggering end-of-day square-off (IST)")
                        # Use close() which is safe for both long/short
                        self.orders[data] = self.sell(data=data)
                        self.crossover_flags[data] = False
                        print(f"{data.tradingsymbol} SELL triggered @ {data.close[0]:.2f}")
                        print(f"[{bar_dt_ist}] {data.tradingsymbol} - issued square-off close (size={pos.size})")

                # No more entries beyond 15:15 IST
                return

        # ✅ Continue normal trading if before cutoff
        for data in self.datas:
            if self.orders[data]:
                continue
            print("1")
            close_price = data.close[0]

            bar_time = data.datetime.datetime(0)
            crossover = self.crossovers[data][0]
            choppiness = self.choppiness[data][0]
            position = self.broker.getposition(data)

            # Record crossover state after BUG FIX - because crossover downside may happen before we enter trade
            if crossover > 0  or (self.sma3[data][0] > self.sma19[data][0] and self.sma3[data][-1] <= self.sma19[data][-1]):
                self.crossover_flags[data] = True
                print(f"{data.tradingsymbol} CROSSOVER TRUE FLAG : {crossover} AND ")
                print(f"{data.tradingsymbol} SMA3 : {self.sma3[data][0]}")
                print(f"{data.tradingsymbol} SMA18 : {self.sma19[data][0]}")
            elif (self.sma3[data][0] < self.sma19[data][0]):
                # Reset this to false when reverse crossover happens before BUYING
                print("Resetting SMA crossover flag")
                print(f"{data.tradingsymbol} SMA3 : {self.sma3[data][0]}")
                print(f"{data.tradingsymbol} SMA18 : {self.sma19[data][0]}")
                self.crossover_flags[data] = False

            print("2")
            # Entry condition
            if position.size == 0:
                print("3")
                print(f"{data.tradingsymbol} - Choppiness : {self.choppiness[data][0]}")
                print(f"{data.tradingsymbol} SMA3 : {self.sma3[data][0]} ")
                print(f"{data.tradingsymbol} SMA18 : {self.sma19[data][0]} ")
                print(f"crossover {self.crossover_flags[data]}")
                if self.crossover_flags[data] and choppiness < 50:
                    print("entering trade")
                    print(f"{data.tradingsymbol} SMA3 : {self.sma3[data][0]}")
                    print(f"{data.tradingsymbol} SMA18 : {self.sma19[data][0]}")
                    # ✅ New condition: crossover should occur above SMA20
                    # if self.sma3[data][0] > self.sma19[data][0]:# and self.sma7[data][0] > self.sma19[data][0]:
                    #     # and self.plus_di[data][0] > self.adx[data][0]:
                    #     print("---------CONFIRMED BUY AFTER 3rd SMA cross with these values---------")
                    #     print(f"{data.tradingsymbol} SMA3 : {self.sma3[data][0]}")
                    #     print(f"{data.tradingsymbol} SMA18 : {self.sma19[data][0]}")

                    self.orders[data] = self.buy(data=data, size=self.size)
                    # self.crossover_flags[data] = False  # Reset after entry
                    print(f"[{bar_time}] {data.tradingsymbol} BUY triggered @ {close_price:.2f}")
            # Exit condition
            elif position.size > 0:
                print(f"Currently a position is OPEN for {data.tradingsymbol} :::: CLOSE {close_price:.2f} SMA3 {self.sma3[data][0]} SMA18 {self.sma19[data][0]} ")
                print(f"{data.tradingsymbol} : SMA3 : {self.sma3[data][0]} ")
                print(f"{data.tradingsymbol} : SMA18 : {self.sma19[data][0]} ")
                if self.sma3[data][0] < self.sma19[data][0]:#crossover < 0: #not self.crossover_flags[data]:
                    print(f"CONFIRMED SELL {data.tradingsymbol} -")
                    print(f"SMA3 : {self.sma3[data][0]} FOR {data.tradingsymbol}")
                    print(f"SMA18 : {self.sma19[data][0]} FOR {data.tradingsymbol}")
                    self.orders[data] = self.sell(data=data, size=self.size)
                    self.crossover_flags[data] = False
                    print(f"[{bar_time}] {data.tradingsymbol} SELL triggered @ {close_price:.2f}")

    def notify_order(self, order):
        if order is None or order.status not in [order.Completed, order.Canceled, order.Rejected]:
            return

        data = order.data
        dt = data.datetime.datetime(0)
        name = data._name
        price = order.executed.price

        if order.isbuy():
            self.buy_prices[data] = price
            self.buy_times[data] = dt
            print(f"+++ {name} BUY EXECUTED @ {price:.2f}")
        elif order.issell():
            buy_price = self.buy_prices.get(data, None)
            buy_time = self.buy_times.get(data, "NA")
            pnl = round(price - buy_price, 2) if buy_price is not None else "NA"
            print(f"+++ {name} SELL EXECUTED @ {price:.2f} | PnL: {pnl}")

            # Log trade
            with open(self.p.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, buy_time, buy_price, dt, price, pnl])

            self.buy_prices[data] = None
            self.buy_times[data] = None

        self.orders[data] = None