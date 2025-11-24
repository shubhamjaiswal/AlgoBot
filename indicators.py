import math
import backtrader as bt
import pandas as pd
from finta import TA
import talib
import numpy as np
class HybridChopFilter(bt.Indicator):
    lines = ('allow_trade',)
    params = dict(period=14)

    def __init__(self):
        self.ci = ChoppinessIndex(self.data, period=self.p.period)
        self.adx = bt.ind.ADX(self.data, period=self.p.period)
        self.atr = bt.ind.ATR(self.data, period=self.p.period)
        self.atr_sma = bt.ind.SMA(self.atr, period=self.p.period)
        self.vol_sma = bt.ind.SMA(self.data.volume, period=8)

    def next(self):
        ci_val = self.ci[0]
        adx_val = self.adx[0]
        atr_val, atr_avg = self.atr[0], self.atr_sma[0]
        vol, vol_avg = self.data.volume[0], self.vol_sma[0]

        allow = True

        if ci_val > 50:  # market appears choppy
            if atr_val < atr_avg:
                allow = False  # block trades in sideways regime
            elif vol > 1.5 * vol_avg or adx_val > 25:
                allow = True  # breakout or strong trend, ignore CI

        self.lines.allow_trade[0] = 1 if allow else 0

class DirectionalChoppiness(bt.Indicator):
    lines = ('dchop',)
    params = dict(period=14)

    def __init__(self):
        atr = bt.ind.ATR(self.data, period=1)
        self.atr_sum = bt.ind.SumN(atr, period=self.p.period)
        self.ret_sum = bt.ind.SumN(self.data.close - self.data.open, period=self.p.period)

    def next(self):
        high_max = max(self.data.high.get(size=self.p.period))
        low_min = min(self.data.low.get(size=self.p.period))
        tr_sum = self.atr_sum[0]

        if high_max != low_min and tr_sum != 0:
            chop = 100 * math.log10(tr_sum / (high_max - low_min)) / math.log10(self.p.period)
            trend_bias = math.tanh(self.ret_sum[0] / tr_sum) * 50
            self.lines.dchop[0] = chop - trend_bias  # Lower when strong directional move
        else:
            self.lines.dchop[0] = 50

class StochRSI(bt.Indicator):
    '''
    Custom Stochastic RSI Indicator
    --------------------------------
    Formula:
        RSI = Relative Strength Index
        StochRSI = (RSI - min(RSI)) / (max(RSI) - min(RSI))

    Parameters:
        period (int): RSI period (default=14)
        stoch_period (int): Lookback for StochRSI (default=14)
        smoothK (int): Smoothing for %K (default=3)
        smoothD (int): Smoothing for %D (default=3)
        src (str): Source type ('close', 'open', 'hl2', 'ohlc4', etc.)
    '''
    lines = ('stochrsi', 'k', 'd')
    params = dict(
        period=14,
        stoch_period=14,
        smoothK=3,
        smoothD=3,
        src='close'  # 'close', 'open', 'hl2', 'ohlc4', etc.
    )

    def __init__(self):
        # Select price source
        if self.p.src == 'close':
            price = self.data.close
        elif self.p.src == 'open':
            price = self.data.open
        elif self.p.src == 'hl2':
            price = (self.data.high + self.data.low) / 2.0
        elif self.p.src == 'ohlc4':
            price = (self.data.open + self.data.high + self.data.low + self.data.close) / 4.0
        else:
            raise ValueError(f"Unsupported src type: {self.p.src}")

        # --- RSI Calculation ---
        rsi = bt.ind.RSI(price, period=self.p.period)

        # --- Stochastic RSI Calculation ---
        rsi_low = bt.ind.Lowest(rsi, period=self.p.stoch_period)
        rsi_high = bt.ind.Highest(rsi, period=self.p.stoch_period)
        stochrsi = (rsi - rsi_low) / (rsi_high - rsi_low)

        # Smooth %K and %D
        k = bt.ind.SMA(stochrsi, period=self.p.smoothK)
        d = bt.ind.SMA(k, period=self.p.smoothD)

        self.lines.stochrsi = stochrsi
        self.lines.k = k
        self.lines.d = d
class ChoppinessIndex(bt.Indicator):
    lines = ('chop',)
    params = (
        ('period', 13),
        ('data', None),
    )
    plotinfo = dict(subplot=True)
    plotlines = dict(
        chop=dict(color='orange', linewidth=1.5)
    )

    def __init__(self):
        atr = bt.ind.ATR(self.data, period=1, movav=bt.ind.SmoothedMovingAverage)
        self.atr_sum = bt.indicators.SumN(atr, period=self.p.period)

    def next(self):
        high_max = max(self.data.high.get(size=self.p.period))
        low_min = min(self.data.low.get(size=self.p.period))
        tr_sum = self.atr_sum[0]

        if high_max != low_min and tr_sum != 0:
            ci = 100 * math.log10(tr_sum / (high_max - low_min)) / math.log10(self.p.period)
            self.lines.chop[0] = ci
        else:
            self.lines.chop[0] = 50  # Neutral if invalid calc
class MADiv_Stoch_Supertrend_Vol(bt.Indicator):
    """
    Composite indicator with Volatility Filter:
      - Moving Average Divergence (EMA fast - slow)
      - Stochastic Oscillator (%K, %D)
      - ATR-based Supertrend-like filter
      - Volatility filter (ATR ratio + optional Choppiness Index)

    Outputs:
      - signal: 1 (buy), -1 (sell), 0 (no trade)
      - trend, macddiv, stochk, stochd, upper, lower
    """
    lines = ('signal', 'trend', 'macddiv', 'stochk', 'stochd', 'upper', 'lower', 'vol_ok',)
    params = (
        ('ma_fast', 8),
        ('ma_slow', 21),
        ('stoch_k_period', 14),
        ('stoch_d_period', 3),
        ('stoch_k_buy', 20),
        ('stoch_k_sell', 80),
        ('atr_period', 10),
        ('atr_mult', 3.0),
        ('atr_vol_ratio', 0.005),   # ATR must be > 0.5% of close to allow trades
        ('use_ci', True),
        ('ci_period', 14),
        ('ci_thresh', 55),          # CI above this => too choppy, block trades
        ('require_all', True),
    )

    def __init__(self):
        # --- core indicators ---
        self.mad = MADivergence(self.data.close, fast=self.p.ma_fast, slow=self.p.ma_slow)
        self.stoch = bt.ind.Stochastic(self.data,
                                       period=self.p.stoch_k_period,
                                       period_d=self.p.stoch_d_period,
                                       safediv=True)
        self.super = ATRSupertrend(self.data, period=self.p.atr_period, mult=self.p.atr_mult)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)

        # Choppiness Index (if enabled)
        if self.p.use_ci:
            tr_sum = bt.ind.SumN(bt.ind.TrueRange(self.data), period=self.p.ci_period)
            high_max = bt.ind.Highest(self.data.high, period=self.p.ci_period)
            low_min = bt.ind.Lowest(self.data.low, period=self.p.ci_period)
            ci = 100 * bt.ind.LogN(tr_sum / (high_max - low_min), 10) / bt.ind.LogN(self.p.ci_period, 10)
            self.ci = ci
        else:
            self.ci = None

        # wire through
        self.lines.macddiv = self.mad.lines.mad
        self.lines.stochk = self.stoch.percK
        self.lines.stochd = self.stoch.percD
        self.lines.trend = self.super.lines.trend
        self.lines.upper = self.super.lines.upper
        self.lines.lower = self.super.lines.lower

    def next(self):
        sig = 0
        macddiv = self.lines.macddiv[0]
        stochk = self.lines.stochk[0]
        stochd = self.lines.stochd[0]
        trend = int(self.lines.trend[0])
        atr_val = self.atr[0]
        close = self.data.close[0]

        # --- Volatility Filter ---
        vol_ok = True
        # ATR vs Close ratio
        if atr_val / close < self.p.atr_vol_ratio:
            vol_ok = False
        # Choppiness check
        if self.p.use_ci and self.ci[0] > self.p.ci_thresh:
            vol_ok = False
        self.lines.vol_ok[0] = 1 if vol_ok else 0

        # --- Signal Logic ---
        mad_prev = self.lines.macddiv[-1]
        macd_bull_cross = (mad_prev <= 0) and (macddiv > 0)
        macd_bear_cross = (mad_prev >= 0) and (macddiv < 0)

        stoch_prev_k, stoch_prev_d = self.lines.stochk[-1], self.lines.stochd[-1]
        stoch_k_cross_up = (stoch_prev_k <= stoch_prev_d) and (stochk > stochd)
        stoch_k_cross_down = (stoch_prev_k >= stoch_prev_d) and (stochk < stochd)
        stoch_oversold = (stochk <= self.p.stoch_k_buy)
        stoch_overbought = (stochk >= self.p.stoch_k_sell)

        buy_cond = macd_bull_cross and (stoch_k_cross_up or stoch_oversold)
        sell_cond = macd_bear_cross and (stoch_k_cross_down or stoch_overbought)

        if self.p.require_all:
            buy_cond = buy_cond and (trend == 1)
            sell_cond = sell_cond and (trend == -1)

        # apply volatility filter
        if vol_ok:
            if buy_cond:
                sig = 1
            elif sell_cond:
                sig = -1

        self.lines.signal[0] = sig

class FractalChaosOsc(bt.Indicator):
    """
    Fractal Chaos Oscillator (Kite-style)
    Outputs: -1 (bearish), 0 (neutral), 1 (bullish)
    """
    lines = ('fco',)
    params = dict(
        period=3  # lookback for fractals
    )

    def __init__(self):
        # Left/right bars to confirm fractals
        self.addminperiod(self.p.period)

    def next(self):
        i = self.p.period // 2

        if len(self.data) < self.p.period:
            self.lines.fco[0] = 0
            return

        # Define fractals
        is_high = self.data.high[-i] == max(self.data.high.get(size=self.p.period))
        is_low = self.data.low[-i] == min(self.data.low.get(size=self.p.period))

        # Directional check
        if is_high and self.data.close[0] > self.data.close[-i]:
            self.lines.fco[0] = 1   # bullish
        elif is_low and self.data.close[0] < self.data.close[-i]:
            self.lines.fco[0] = -1  # bearish
        else:
            self.lines.fco[0] = 0   # neutral
class ChandeMomentumOscillator(bt.Indicator):
    """Chande Momentum Oscillator (CMO)"""
    lines = ('cmo',)
    params = (('period', 14),)

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        period = self.p.period
        up, down = 0.0, 0.0

        for i in range(1, period + 1):
            diff = self.data[-i + 1] - self.data[-i]
            if diff > 0:
                up += diff
            else:
                down -= diff

        total = up + down
        self.lines.cmo[0] = 100.0 * (up - down) / total if total != 0 else 0.0


class VIDYA(bt.Indicator):
    """
    Variable Index Dynamic Average (VIDYA)
    """
    lines = ('vidya',)
    params = (
        ('data', None),
        ('period', 9),       # smoothing length
        ('select', True),    # True = CMO-based, False = StdDev
    )

    def __init__(self):
        self.addminperiod(self.p.period + 1)
        if self.p.select:
            self.cmo = ChandeMomentumOscillator(self.data.close,
                                                period=self.p.period)

    def next(self):
        period = self.p.period
        alpha = 2 / (period + 1)

        # --- Factor k ---
        if self.p.select:
            k = abs(self.cmo[0]) / 100.0 if not np.isnan(self.cmo[0]) else 0.0
        else:
            if len(self) >= period:
                closes = [self.data.close[-i] for i in range(period)]
                k = np.nan_to_num(np.std(closes))
            else:
                k = 0.0

        # --- Seeding ---
        if len(self) <= period:
            self.lines.vidya[0] = self.data.close[0]
            return  # seed, stop here

        prev = self.lines.vidya[-1]
        if np.isnan(prev):
            prev = self.data.close[-1]

        self.lines.vidya[0] = (
            alpha * k * self.data.close[0]
            + (1 - alpha * k) * prev
        )

        # --- Debugging output ---
        # print(f"[VIDYA DEBUG] len={len(self)} close={self.data.close[0]:.2f} "
        #       f"alpha={alpha:.4f} k={k:.4f} prev={prev:.2f} "
        #       f"vidya={self.lines.vidya[0]:.2f}")


class WMA_TALIB(bt.Indicator):
    lines = ('wma',)
    params = (('period', 14),)

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        # take last period closes and feed into talib
        closes = np.array(self.data.get(size=self.p.period), dtype=float)
        self.lines.wma[0] = talib.WMA(closes, timeperiod=self.p.period)[-1]


class VIDYA_old(bt.Indicator):
    """
    Variable Index Dynamic Average (VIDYA)
    Based on Tushar Chande’s 1995 formula.

    alpha = |CMO(period)| / 100 * (2 / (ema_period + 1))
    vidya[t] = vidya[t-1] + alpha * (price[t] - vidya[t-1])

    Params
    ------
    period : int
        Lookback for CMO (default: 14)
    ema_period : int
        Base EMA period (default: 14)
    alpha_min : float
        Optional minimum alpha (default: 0.0)
    """

    lines = ('vidya',)
    params = dict(
        period=14,
        ema_period=14,
        alpha_min=0.0,
    )

    def __init__(self):
        self.addminperiod(max(self.p.period, self.p.ema_period))
        self._price = self.data.close

        # rolling sums for CMO
        self._up = []
        self._dn = []
        self._sum_up = 0.0
        self._sum_dn = 0.0

        self._k = 2.0 / (self.p.ema_period + 1.0)
        self._seeded = False

    def next(self):
        # price change
        if len(self) == 1:
            delta = 0.0
        else:
            delta = float(self._price[0] - self._price[-1])

        up = max(delta, 0.0)
        dn = max(-delta, 0.0)

        # update rolling sums
        self._up.append(up)
        self._dn.append(dn)
        self._sum_up += up
        self._sum_dn += dn

        if len(self._up) > self.p.period:
            self._sum_up -= self._up.pop(0)
            self._sum_dn -= self._dn.pop(0)

        # compute CMO
        denom = self._sum_up + self._sum_dn
        cmo = 0.0 if denom == 0 else 100.0 * (self._sum_up - self._sum_dn) / denom
        alpha = max(self.p.alpha_min, abs(cmo) / 100.0 * self._k)

        # seed with SMA on first full period
        if not self._seeded and len(self) >= self.p.ema_period:
            sma = sum(self._price.get(size=self.p.ema_period)) / self.p.ema_period
            self.lines.vidya[0] = sma
            self._seeded = True
            return

        # recursive update
        if self._seeded:
            prev = float(self.lines.vidya[-1])
            price_now = float(self._price[0])
            self.lines.vidya[0] = prev + alpha * (price_now - prev)


class SessionVWAP(bt.Indicator):
    lines = ('vwap',)

    def __init__(self):
        self.cum_pv = 0.0
        self.cum_vol = 0.0
        self.current_date = None

    def next(self):
        dt = bt.num2date(self.data.datetime[0])
        price = self.data.close[0]
        vol = self.data.volume[0]

        # Reset at new session
        if self.current_date != dt.date():
            self.cum_pv = 0.0
            self.cum_vol = 0.0
            self.current_date = dt.date()

        self.cum_pv += price * vol
        self.cum_vol += vol

        if self.cum_vol > 0:
            self.lines.vwap[0] = self.cum_pv / self.cum_vol
        else:
            self.lines.vwap[0] = price  # fallback if no volume yet


class VWAP(bt.Indicator):
    lines = ('vwap',)
    params = (('period', None),)  # Session VWAP, so no fixed period

    def __init__(self):
        price_volume = self.data.close * self.data.volume
        cum_pv = bt.indicators.SumN(price_volume, period=self.p.period or len(self.data))
        cum_vol = bt.indicators.SumN(self.data.volume, period=self.p.period or len(self.data))
        self.lines.vwap = cum_pv / cum_vol


class Supertrend(bt.Indicator):
    lines = ('supertrend', 'upperband', 'lowerband', 'trend')
    params = (('period', 10), ('multiplier', 3.0))

    def __init__(self):
        atr = bt.ind.ATR(self.data, period=self.p.period)

        hl2 = (self.data.high + self.data.low) / 2
        upperband = hl2 + (self.p.multiplier * atr)
        lowerband = hl2 - (self.p.multiplier * atr)

        self.l.upperband = upperband
        self.l.lowerband = lowerband

        self.l.supertrend = bt.If(self.data.close > upperband, lowerband, upperband)
        self.l.trend = bt.If(self.data.close > upperband, 1, -1)

# class VIDYA(bt.Indicator):
#     lines = ('vidya',)
#     params = (('period', 14),)
#
#     def __init__(self):
#         self.addminperiod(self.p.period)
#         self.sma = bt.ind.SMA(self.data, period=self.p.period)  # for warm-up
#         self.roc = bt.ind.ROC(self.data, period=1)  # 1-bar rate of change
#
#     def next(self):
#         # Calculate CMO (Chande Momentum Oscillator) for smoothing factor
#         up_sum = down_sum = 0.0
#         for i in range(1, self.p.period + 1):
#             diff = self.data[-i+1] - self.data[-i]
#             if diff > 0:
#                 up_sum += diff
#             else:
#                 down_sum -= diff
#         cmo = 0
#         if up_sum + down_sum != 0:
#             cmo = (up_sum - down_sum) / (up_sum + down_sum)
#
#         alpha = abs(cmo) * 2.0 / self.p.period
#
#         if math.isnan(self.lines.vidya[-1]):
#             self.lines.vidya[0] = self.sma[0]  # seed with SMA
#         else:
#             self.lines.vidya[0] = self.lines.vidya[-1] + alpha * (self.data[0] - self.lines.vidya[-1])
