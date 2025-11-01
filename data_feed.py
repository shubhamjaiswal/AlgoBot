import threading
import pytz
import backtrader as bt
from datetime import datetime, timedelta, time

# Working live data
class LiveKiteDataMultiTimeframe(bt.feeds.DataBase):
    params = (
        ('symbol', ''),
        ('instrument_token', 0),
        ('broker', None),
        ('timeframe', bt.TimeFrame.Minutes),
        ('compression', 1),  # 1min, 3min, 5min, etc.
    )

    def __init__(self):
        self.symbol = self.p.symbol
        self.instrument_token = self.p.instrument_token
        self.broker = self.p.broker
        self._tick_buffer = []
        self._last_bar_time = datetime.now().replace(second=0, microsecond=0)
        super().__init__()

    def start(self):
        super().start()

    def _load(self):
        now = datetime.now()
        price = self.broker._tickers.get(self.instrument_token)
        # print("bef")
        if price is None:
            return None
        # print("af")

        # Add the tick to buffer
        self._tick_buffer.append((now, price))

        # Check if we should emit a bar
        if now >= self._last_bar_time + timedelta(minutes=self.p.compression):
            bar_start = self._last_bar_time
            bar_end = self._last_bar_time + timedelta(minutes=self.p.compression)
            self._last_bar_time = bar_end

            # Filter ticks for this bar
            ticks = [p for t, p in self._tick_buffer if bar_start <= t < bar_end]
            self._tick_buffer = [(t, p) for t, p in self._tick_buffer if t >= bar_end]

            if not ticks:
                return None

            o = ticks[0]
            h = max(ticks)
            l = min(ticks)
            c = ticks[-1]
            v = len(ticks)

            self.lines.datetime[0] = bt.date2num(bar_end)
            self.lines.open[0] = o
            self.lines.high[0] = h
            self.lines.low[0] = l
            self.lines.close[0] = c
            self.lines.volume[0] = v
            print("emitting")
            return True

        return None





class LiveKiteDataMultiTimeframeSyncedPrefill(bt.feeds.DataBase):
    params = (
        ('symbol', ''),
        ('instrument_token', 0),
        ('broker', None),
        ('kite', None),
        ('timeframe', bt.TimeFrame.Minutes),
        ('compression', 1),
        ('backfill_days', 1),
    )

    def __init__(self):
        self.symbol = self.p.symbol
        self.instrument_token = self.p.instrument_token
        self.broker = self.p.broker
        self.kite = self.p.kite
        self._tick_buffer = []
        self._last_bar_time = None
        self._backfill_data = []
        self.live = False        # <--- add this flag
        super().__init__()
    def _align_to_market_open(self):
        """Aligns bar start to Kite candle times (from 9:15)."""
        MARKET_OPEN = time(9, 15)
        now = datetime.now()
        market_open_dt = datetime.combine(now.date(), MARKET_OPEN)

        if now < market_open_dt:
            # Before market open — wait until 9:15
            self._last_bar_time = market_open_dt
        else:
            # Align to nearest multiple of compression since market open
            minutes_since_open = int((now - market_open_dt).total_seconds() // 60)
            aligned_minutes = (minutes_since_open // self.p.compression) * self.p.compression
            self._last_bar_time = market_open_dt + timedelta(minutes=aligned_minutes)

    def start(self):
        super().start()
        self._align_to_market_open()
        self._backfill_history()
        # don’t set live yet; it will be set after backfill ends

    def _backfill_history(self):
        today = datetime.now().date()
        # from_time = datetime.combine(today, datetime.min.time())
        from_time = datetime.combine(today, time(9, 15))
        to_time = datetime.now()

        records = self.p.kite.historical_data(
            self.p.instrument_token,
            from_time,
            to_time,
            "3minute"
        )

        for row in records:
            dt = row["date"]
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            # dt = dt + timedelta(minutes=self.p.compression)
            self._backfill_data.append([
                bt.date2num(dt),
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
            ])
    def _load(self):
        # Process backfill first
        if self._backfill_data:
            bar = self._backfill_data.pop(0)
            self.lines.datetime[0] = bar[0]
            self.lines.open[0] = bar[1]
            self.lines.high[0] = bar[2]
            self.lines.low[0] = bar[3]
            self.lines.close[0] = bar[4]
            self.lines.volume[0] = bar[5]
            self.lines.openinterest[0] = 0
            print(f"[PREFILL] {bt.num2date(self.lines.datetime[0])} "
                  f"O={self.lines.open[0]} H={self.lines.high[0]} "
                  f"L={self.lines.low[0]} C={self.lines.close[0]} V={self.lines.volume[0]}")
            if not self._backfill_data:   # backfill completed
                self.live = True          # <--- now live mode
                print(f"✅ Backfill done for {self.symbol}. Switching to live mode.")
            return True

        # if not self._backfill_data:  # backfill just ended
        #     self.live = True
        #     print("✅ Backfill done. Switching to live mode.")
        #
        #     # Call reset only in live feeds
        #     if hasattr(self, "_owner"):  # Backtrader cerebro owns the feed
        #         for strat in self._owner._strategies:
        #             if hasattr(strat, "reset_after_backfill"):
        #                 strat.reset_after_backfill()

        # Ignore ticks until we are live
        if not self.live:
            return None

        # ---- Live processing ----
        now = datetime.now()
        price = self.broker._tickers.get(self.instrument_token)
        if price is None:
            return None

        self._tick_buffer.append((now, price))

        if self._last_bar_time is None:
            self._last_bar_time = now.replace(second=0, microsecond=0)

        if now >= self._last_bar_time + timedelta(minutes=self.p.compression):
            bar_start = self._last_bar_time
            bar_end = bar_start + timedelta(minutes=self.p.compression)
            self._last_bar_time = bar_end

            ticks = [p for t, p in self._tick_buffer if bar_start <= t < bar_end]
            self._tick_buffer = [(t, p) for t, p in self._tick_buffer if t >= bar_end]

            if not ticks:
                return None

            o = ticks[0]
            h = max(ticks)
            l = min(ticks)
            c = ticks[-1]
            v = len(ticks)

            self.lines.datetime[0] = bt.date2num(bar_end)
            self.lines.open[0] = o
            self.lines.high[0] = h
            self.lines.low[0] = l
            self.lines.close[0] = c
            self.lines.volume[0] = v
            self.lines.openinterest[0] = 0

            print(f"{self.symbol} : Emitting Live bar [{bar_end.strftime('%H:%M:%S')}] Live bar O:{o} H:{h} L:{l} C:{c} V:{v}")
            return True

        return None
class LiveKiteDataMultiTimeframeSynced(bt.feeds.DataBase):
    params = (
        ('symbol', ''),
        ('instrument_token', 0),
        ('broker', None),
        ('timeframe', bt.TimeFrame.Minutes),
        ('compression', 1),  # 1min, 3min, 5min, etc.
    )

    def __init__(self):
        self.symbol = self.p.symbol
        self.instrument_token = self.p.instrument_token
        self.broker = self.p.broker
        self._tick_buffer = []
        self._last_bar_time = None  # will be set in start()
        super().__init__()

    def start(self):
        super().start()
        self._align_to_market_open()

    def _align_to_market_open(self):
        """Aligns bar start to Kite candle times (from 9:15)."""
        MARKET_OPEN = time(9, 15)
        now = datetime.now()
        market_open_dt = datetime.combine(now.date(), MARKET_OPEN)

        if now < market_open_dt:
            # Before market open — wait until 9:15
            self._last_bar_time = market_open_dt
        else:
            # Align to nearest multiple of compression since market open
            minutes_since_open = int((now - market_open_dt).total_seconds() // 60)
            aligned_minutes = (minutes_since_open // self.p.compression) * self.p.compression
            self._last_bar_time = market_open_dt + timedelta(minutes=aligned_minutes)

    def _load(self):
        now = datetime.now()
        price = self.broker._tickers.get(self.instrument_token)

        if price is None:
            return None

        # Add tick to buffer
        self._tick_buffer.append((now, price))

        # Check if it's time to emit a bar
        if now >= self._last_bar_time + timedelta(minutes=self.p.compression):
            bar_start = self._last_bar_time
            bar_end = bar_start + timedelta(minutes=self.p.compression)
            self._last_bar_time = bar_end

            # Ticks belonging to this bar
            ticks = [p for t, p in self._tick_buffer if bar_start <= t < bar_end]
            self._tick_buffer = [(t, p) for t, p in self._tick_buffer if t >= bar_end]

            if not ticks:
                return None

            o = ticks[0]
            h = max(ticks)
            l = min(ticks)
            c = ticks[-1]
            v = len(ticks)  # Using tick count as volume proxy

            self.lines.datetime[0] = bt.date2num(bar_end)
            self.lines.open[0] = o
            self.lines.high[0] = h
            self.lines.low[0] = l
            self.lines.close[0] = c
            self.lines.volume[0] = v

            # Debug
            print(f"[{bar_end.strftime('%H:%M:%S')}] Emitting bar O:{o} H:{h} L:{l} C:{c} V:{v}")

            return True

        return None

class LiveKiteDataMultiTimeframeSyncWithKite(bt.feeds.DataBase):
    params = (
        ('symbol', ''),
        ('instrument_token', 0),
        ('broker', None),
        ('timeframe', bt.TimeFrame.Minutes),
        ('compression', 1),  # 1min, 3min, 5min, etc.
    )

    def __init__(self):
        self.symbol = self.p.symbol
        self.instrument_token = self.p.instrument_token
        self.broker = self.p.broker
        self._tick_buffer = []

        # Get IST time
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)
        market_open = datetime.combine(now.date(), time(9, 15), tzinfo=ist)

        if now < market_open:
            # Before market open → set last bar to one bar before open
            self._last_bar_time = market_open - timedelta(minutes=self.p.compression)
        else:
            # After open → align to last completed compression bar
            minutes_since_open = int((now - market_open).total_seconds() // 60)
            last_bar_offset = (minutes_since_open // self.p.compression) * self.p.compression
            self._last_bar_time = market_open + timedelta(minutes=last_bar_offset)

        super().__init__()

    def _load(self):
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)

        price = self.broker._tickers.get(self.instrument_token)
        if price is None:
            return None

        # Store tick
        self._tick_buffer.append((now, price))

        # If time for a new bar
        if now >= self._last_bar_time + timedelta(minutes=self.p.compression):
            bar_start = self._last_bar_time
            bar_end = self._last_bar_time + timedelta(minutes=self.p.compression)
            self._last_bar_time = bar_end

            # Filter ticks for the current bar period
            ticks = [p for t, p in self._tick_buffer if bar_start <= t < bar_end]
            self._tick_buffer = [(t, p) for t, p in self._tick_buffer if t >= bar_end]

            if not ticks:
                return None

            o = ticks[0]
            h = max(ticks)
            l = min(ticks)
            c = ticks[-1]
            v = len(ticks)

            self.lines.datetime[0] = bt.date2num(bar_end)
            self.lines.open[0] = o
            self.lines.high[0] = h
            self.lines.low[0] = l
            self.lines.close[0] = c
            self.lines.volume[0] = v

            print(f"[{datetime.now(ist).strftime('%H:%M:%S')}] Emitting bar: {bar_start.time()} → {bar_end.time()}")
            return True

        return None


from kiteconnect import KiteConnect, KiteTicker


class HybridKiteDataFinal(bt.feeds.DataBase):
    params = (
        ('symbol', ''),
        ('instrument_token', 0),
        ('kite', None),
        ('timeframe', bt.TimeFrame.Minutes),
        ('compression', 3),
        ('backfill_minutes', 60),  # minimal history
    )

    def __init__(self):
        self.data_queue = []
        self.streaming = False


    def start(self):
        super().start()

        # 1️⃣ Minimal backfill
        if self.p.kite:
            end = datetime.now()
            start = end - timedelta(minutes=self.p.backfill_minutes)
            df = self.p.kite.historical_data(
                self.p.instrument_token, start, end, "minute"
            )
            for row in df:
                dt = row['date']
                if dt.tzinfo is None:
                    dt = pytz.timezone('Asia/Kolkata').localize(dt)
                self.data_queue.append((dt, row['open'], row['high'], row['low'], row['close'], row['volume']))

        # 2️⃣ Start streaming after history load
        self._start_streaming()

    def _start_streaming(self):
        if not self.p.kws:
            # Create KiteTicker instance from your API_KEY and access token
            kws = KiteTicker(self.p.api_key, self.p.access_token)
        # kws = KiteTicker(api_key=self.p.kite.api_key, access_token=self.p.kite.access_token)
        kws.on_ticks = self.on_ticks
        kws.subscribe([self.p.instrument_token])
        kws.set_mode(kws.MODE_FULL, [self.p.instrument_token])
        threading.Thread(target=kws.connect, daemon=True).start()
        self.streaming = True

    def on_ticks(self, ws, ticks):
        # Example: assume you’re aggregating ticks into 1-min bars externally
        for tick in ticks:
            dt = datetime.fromtimestamp(tick['timestamp'], pytz.timezone('Asia/Kolkata'))
            self.data_queue.append((dt, tick['ohlc']['open'], tick['ohlc']['high'],
                                    tick['ohlc']['low'], tick['ohlc']['close'], tick['volume']))

    def _load(self):
        if not self.data_queue:
            return None  # no data yet

        dt, o, h, l, c, v = self.data_queue.pop(0)
        self.lines.datetime[0] = bt.date2num(dt)
        self.lines.open[0] = o
        self.lines.high[0] = h
        self.lines.low[0] = l
        self.lines.close[0] = c
        self.lines.volume[0] = v
        self.lines.openinterest[0] = 0
        return True


class LiveKiteDataMultiTimeframeSyncWithKiteFinal(bt.feeds.DataBase):
    params = (
        ('symbol', ''),
        ('instrument_token', 0),
        ('broker', None),
        ('kite', None),  # KiteConnect instance for backfill
        ('timeframe', bt.TimeFrame.Minutes),
        ('compression', 1),
    )

    def __init__(self):
        self.symbol = self.p.symbol
        self.instrument_token = self.p.instrument_token
        self.broker = self.p.broker
        self.kite = self.p.kite
        self._tick_buffer = []
        self._backfill_done = False
        from datetime import time
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)
        market_open = datetime.combine(now.date(), time(9, 15), tzinfo=ist)

        # Align _last_bar_time to compression
        if now < market_open:
            self._last_bar_time = market_open - timedelta(minutes=self.p.compression)
        else:
            minutes_since_open = int((now - market_open).total_seconds() // 60)
            last_bar_offset = (minutes_since_open // self.p.compression) * self.p.compression
            self._last_bar_time = market_open + timedelta(minutes=last_bar_offset)

        super().__init__()

    def _do_backfill(self):
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)
        market_open = datetime.combine(now.date(), time(9, 15), tzinfo=ist)

        # Backfill only if market is open and we are late
        if now <= market_open or self._backfill_done:
            return

        from_dt = market_open
        to_dt = self._last_bar_time

        interval_map = {
            1: "minute",
            3: "3minute",
            5: "5minute",
            15: "15minute"
        }
        interval = interval_map.get(self.p.compression, "minute")

        print(f"[Backfill] Fetching {interval} data from {from_dt} to {to_dt}")

        hist = self.kite.historical_data(
            self.instrument_token,
            from_dt,
            to_dt,
            interval,
            continuous=False
        )

        for bar in hist:
            dt_ist = pytz.timezone("Asia/Kolkata").localize(
                datetime.strptime(bar['date'].strftime('%Y-%m-%d %H:%M:%S'),
                                  '%Y-%m-%d %H:%M:%S')
            )
            self.lines.datetime[0] = bt.date2num(dt_ist)
            self.lines.open[0] = bar['open']
            self.lines.high[0] = bar['high']
            self.lines.low[0] = bar['low']
            self.lines.close[0] = bar['close']
            self.lines.volume[0] = bar['volume']
            yield True  # Push to Backtrader

        self._backfill_done = True
        print("[Backfill] Completed.")

    def _load(self):
        # First, run backfill
        if not self._backfill_done:
            try:
                return next(self._do_backfill())
            except StopIteration:
                pass

        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)

        price = self.broker._tickers.get(self.instrument_token)
        if price is None:
            return None

        # Store tick
        self._tick_buffer.append((now, price))

        # If time for a new bar
        if now >= self._last_bar_time + timedelta(minutes=self.p.compression):
            bar_start = self._last_bar_time
            bar_end = self._last_bar_time + timedelta(minutes=self.p.compression)
            self._last_bar_time = bar_end

            ticks = [p for t, p in self._tick_buffer if bar_start <= t < bar_end]
            self._tick_buffer = [(t, p) for t, p in self._tick_buffer if t >= bar_end]

            if not ticks:
                return None

            o = ticks[0]
            h = max(ticks)
            l = min(ticks)
            c = ticks[-1]
            v = len(ticks)

            self.lines.datetime[0] = bt.date2num(bar_end)
            self.lines.open[0] = o
            self.lines.high[0] = h
            self.lines.low[0] = l
            self.lines.close[0] = c
            self.lines.volume[0] = v

            print(f"[{datetime.now(ist).strftime('%H:%M:%S')}] Live bar: {bar_start.time()} → {bar_end.time()}")
            return True

        return None


class LiveKiteDataMultiTimeframeBackfill(bt.feeds.DataBase):
    params = (
        ('symbol', ''),
        ('instrument_token', 0),
        ('broker', None),
        ('kite', None),  # KiteConnect instance
        ('timeframe', bt.TimeFrame.Minutes),
        ('compression', 1),  # 1min, 3min, 5min, etc.
        ('session_start', datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)),
        ('backfill', True),  # Whether to fetch historical bars before going live
    )

    def __init__(self):
        self.symbol = self.p.symbol
        self.instrument_token = self.p.instrument_token
        self.broker = self.p.broker
        self.kite = self.p.kite
        self._tick_buffer = []
        self._last_bar_time = None
        self._backfilled = False
        self._hist_data = []
        super().__init__()

    def start(self):
        super().start()
        import pytz
        if self.p.backfill and self.kite:
            print("--------------------")
            now = datetime.now(pytz.timezone("Asia/Kolkata"))
            market_start = now.replace(hour=self.p.session_start.hour,
                                       minute=self.p.session_start.minute,
                                       second=0, microsecond=0)

            # Only backfill if we start after market start
            if now > market_start:
                from_dt = market_start
                to_dt = now - timedelta(minutes=self.p.compression)

                try:
                    hist = self.kite.historical_data(
                        instrument_token=self.instrument_token,
                        from_date=from_dt,
                        to_date=to_dt,
                        interval=f"{self.p.compression}minute"
                    )

                    for bar in hist:
                        dt_obj = bar['date'].replace(tzinfo=None)  # remove timezone
                        self._hist_data.append((
                            dt_obj, bar['open'], bar['high'],
                            bar['low'], bar['close'], bar['volume']
                        ))

                    self._hist_data_iter = iter(self._hist_data)
                    print(f"[BACKFILL] Loaded {len(self._hist_data)} bars for {self.symbol}")

                except Exception as e:
                    print(f"[BACKFILL ERROR] {e}")

    def _load(self):
        # Step 1: Serve historical bars first
        if not self._backfilled and self._hist_data:
            try:
                dt_obj, o, h, l, c, v = next(self._hist_data_iter)
                self.lines.datetime[0] = bt.date2num(dt_obj)
                self.lines.open[0] = o
                self.lines.high[0] = h
                self.lines.low[0] = l
                self.lines.close[0] = c
                self.lines.volume[0] = v
                return True
            except StopIteration:
                self._backfilled = True
                self._last_bar_time = datetime.now().replace(second=0, microsecond=0)
                print(f"[BACKFILL] Completed for {self.symbol}")

        # Step 2: Build live bars from ticks
        if self._backfilled:
            print("liveee.. feeed")
            now = datetime.now()
            price = self.broker._tickers.get(self.instrument_token)

            if price is None:
                return None

            self._tick_buffer.append((now, price))

            if now >= self._last_bar_time + timedelta(minutes=self.p.compression):
                bar_start = self._last_bar_time
                bar_end = bar_start + timedelta(minutes=self.p.compression)
                self._last_bar_time = bar_end

                ticks = [p for t, p in self._tick_buffer if bar_start <= t < bar_end]
                self._tick_buffer = [(t, p) for t, p in self._tick_buffer if t >= bar_end]

                if not ticks:
                    return None

                o = ticks[0]
                h = max(ticks)
                l = min(ticks)
                c = ticks[-1]
                v = len(ticks)

                self.lines.datetime[0] = bt.date2num(bar_end)
                self.lines.open[0] = o
                self.lines.high[0] = h
                self.lines.low[0] = l
                self.lines.close[0] = c
                self.lines.volume[0] = v
                return True

        return None
