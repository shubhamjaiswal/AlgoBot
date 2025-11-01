import backtrader as bt
import logging
import threading
import time
import csv
import math
from kiteconnect import KiteConnect, KiteTicker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler('strategy_trades.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class SimplePosition:
    def __init__(self, size=0, price=0.0):
        self.size = size
        self.price = price

class KiteBroker(bt.BrokerBase):
    params = (
        ('api_key', 'm1yzl03jrngy1upu'),
        ('access_token', '890tCW3N73Kz0268spSrAaHZTdrCdoDq'),
        ('instruments', [11383810,11393282]),  # Pass list of instrument tokens here
        ('debug', False),
        ('live', False),
    )

    def __init__(self):
        super(KiteBroker, self).__init__()
        self.kite = KiteConnect(api_key=self.p.api_key)
        self.kite.set_access_token(self.p.access_token)

        self.cash = 1000000  # Starting cash for simulation
        self._starting_cash = self.cash
        self.positions = {}  # data -> (size, avg_price)
        self.trade_log = []
        self.trade_details = []
        self.last_buy = {}
        self._tickers = {}
        self.cumulative_profit = []

        self.win_count = 0
        self.loss_count = 0
        self.max_drawdown = 0
        self.equity_curve = [self.cash]

        self._start_price_thread()

    def setcash(self, cash):
        self.cash = cash
        self._starting_cash = cash

    def getstartingcash(self):
        return getattr(self, '_starting_cash', self.getcash())

    @property
    def startingcash(self):
        return self.getstartingcash()

    def getcash(self):
        return self.cash

    def getvalue(self):
        return self.getcash()

    def get_instrument_token_map(self):
        token_map = {}
        # Fetch all instruments from Zerodha (exchange-wise or all)
        instruments = self.kite.instruments()

        # Create lookup dictionary: {instrument_token: tradingsymbol}
        token_to_symbol = {inst["instrument_token"]: inst["tradingsymbol"] for inst in instruments}

        return token_to_symbol

    def getposition(self, data):
        size, price = self.positions.get(data, (0, 0.0))
        return SimplePosition(size=size, price=price)

    def buy(self, owner, data=None, size=None, price=None, **kwargs):
        bar_time = data.datetime.datetime(0)
        self._update_position(data, size, price)
        self.last_buy[data] = (bar_time, price, size)
        tradingsymbol = getattr(data, "tradingsymbol", None)
        if not tradingsymbol:
            logger.error("No tradingsymbol attribute set for data feed.")
            return None
        self._place_order(
            tradingsymbol=tradingsymbol,
            transaction_type=self.kite.TRANSACTION_TYPE_BUY,
            quantity=size
        )

        owner.notify_order(None)
        return None

    def sell(self, owner, data=None, size=None, price=None, **kwargs):
        bar_time = data.datetime.datetime(0)
        pnl = self._update_position(data, -size, price, log_trade=True, timestamp=bar_time, symbol=data._dataname)

        if pnl > 0:
            self.win_count += 1
        elif pnl < 0:
            self.loss_count += 1

        buy_time, buy_price, buy_size = self.last_buy.get(data, ("NA", 0.0, 0))
        if isinstance(buy_time, str):
            duration = "NA"
        else:
            duration = (bar_time - buy_time).total_seconds() / 60
        cumulative_pnl = sum(self.trade_log)
        self.equity_curve.append(self._starting_cash + cumulative_pnl)

        self.trade_details.append([
            buy_time, bar_time, data._dataname,
            buy_size, buy_price, size, price,
            pnl, round(duration, 2), round(cumulative_pnl, 2), self.win_count, self.loss_count
        ])
        tradingsymbol = getattr(data, "tradingsymbol", None)
        if not tradingsymbol:
            logger.error("No tradingsymbol attribute set for data feed.")
            return None

        self._place_order(
            tradingsymbol=tradingsymbol,
            transaction_type=self.kite.TRANSACTION_TYPE_SELL,
            quantity=size
        )

        owner.notify_order(None)
        return None

    def safe_place_order(kite, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                order_id = kite.place_order(**kwargs)
                logging.info(f"Order placed successfully: {order_id}")
                return order_id
            except Exception as e:
                err = str(e)
                logging.error(f"Attempt {attempt + 1}/{max_retries} failed: {err}")

                # Retry if it's a timeout or transient error
                if "Read timed out" in err or "Connection aborted" in err:
                    time.sleep(3)  # small delay before retry
                    continue
                else:
                    # For other errors, don't retry (e.g. insufficient funds)
                    raise

        logging.error("Order placement failed after retries.")
        return None
    def _place_order(self, tradingsymbol, transaction_type, quantity):
        if not self.p.live:
            # logger.info(f"[SIMULATED] {transaction_type} {quantity} of {tradingsymbol}")
            return None

        try:
            print(tradingsymbol)
            # order_id = self.kite.place_order(
            #     variety=self.kite.VARIETY_REGULAR,
            #     exchange="NFO",  # or "NFO"
            #     tradingsymbol=tradingsymbol,
            #     transaction_type=transaction_type,
            #     quantity=quantity,
            #     order_type=self.kite.ORDER_TYPE_MARKET,
            #     product=self.kite.PRODUCT_MIS,
            #     validity=self.kite.VALIDITY_DAY
            # )
            # print(f"[LIVE] Order Placed: {order_id} -> {transaction_type} {quantity} {tradingsymbol}")
            order_id = 1
            return order_id
        except Exception as e:
            logger.error(f"[LIVE] Order Failed: {transaction_type} {quantity} {tradingsymbol}: {e}")
            return None

    def _update_position(self, data, size_change, trade_price, log_trade=False, timestamp=None, symbol=None):
        current_size, avg_price = self.positions.get(data, (0, 0.0))
        if trade_price is None:
            trade_price = data.close[0]
        new_size = current_size + size_change

        pnl = 0.0
        if log_trade and current_size > 0 and new_size == 0:
            total_points = (trade_price - avg_price)
            pnl = total_points * current_size
            self.trade_log.append(pnl)
        if new_size == 0:
            new_price = 0.0
        elif current_size == 0:
            new_price = trade_price
        else:
            new_price = ((current_size * avg_price) + (size_change * trade_price)) / new_size

        self.positions[data] = (new_size, new_price)
        return pnl

    def stop(self):
        cumulative_pnl = sum(self.trade_log)
        peak = self.equity_curve[0]
        for equity in self.equity_curve:
            peak = max(peak, equity)
            dd = peak - equity
            self.max_drawdown = max(self.max_drawdown, dd)

        with open('trade_log.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Buy Time', 'Sell Time', 'Symbol', 'Buy Size', 'Buy Price',
                'Sell Size', 'Sell Price', 'PnL', 'Holding Duration (min)',
                'Cumulative PnL', 'Wins', 'Losses'
            ])
            writer.writerows(self.trade_details)

            # Write summary as final row
            writer.writerow([])  # Blank row
            writer.writerow(['TOTAL PnL', round(cumulative_pnl, 2)])
            writer.writerow(['Max Drawdown', round(self.max_drawdown, 2)])
            writer.writerow(['Winning Trades', self.win_count])
            writer.writerow(['Losing Trades', self.loss_count])

        logger.info(f"Cumulative PnL: {cumulative_pnl:.2f}")
        logger.info(f"Winning Trades: {self.win_count} | Losing Trades: {self.loss_count}")
        logger.info(f"Max Drawdown: {self.max_drawdown:.2f}")

    def get_notification(self):
        pass

    def get_cumulative_profit(self):
        return sum(self.trade_log)

    def _start_price_thread(self):
        def ticker_thread():
            kws = KiteTicker(self.p.api_key, self.p.access_token)

            def on_ticks(ws, ticks):
                for tick in ticks:
                    self._tickers[tick['instrument_token']] = tick['last_price']

            def on_connect(ws, response):
                ws.subscribe(self.p.instruments)
                logger.info(f"Subscribed to instruments ::: {self.p.instruments}")

            def on_close(ws, code, reason):
                logger.warning("WebSocket closed: %s - %s", code, reason)
                time.sleep(5)
                ticker_thread()

            kws.on_ticks = on_ticks
            kws.on_connect = on_connect
            kws.on_close = on_close
            self.kws = kws

            try:
                kws.connect(threaded=True)
            except Exception as e:
                logger.error("Ticker connect failed: %s", e)

        t = threading.Thread(target=ticker_thread)
        t.daemon = True
        t.start()

    # 1. Get available funds
    def get_available_funds(self):
        try:
            margins = self.kite.margins("equity")

            # Direct net available margin (recommended by Zerodha docs)
            return float(margins.get("net", 0.0))

        except Exception as e:
            print(f"Error fetching funds: {e}")
            return 0.0

    # 2. Calculate margin required for instrument
    def calculate_margin(self, tradingsymbol, exchange="NSE", transaction_type="BUY", product="NRML", quantity=1):
        try:
            order_margin = self.kite.order_margins(
                [{
                    "exchange": exchange,
                    "tradingsymbol": tradingsymbol,
                    "transaction_type": transaction_type,
                    "variety": "regular",
                    "product": product,
                    "order_type": "MARKET",
                    "quantity": quantity,
                }]
            )
            return order_margin[0]['total']
        except Exception as e:
            print(f"Error calculating margin: {e}")
            return None

    # 3. Calculate maximum quantity that can be bought
    def get_max_quantity(self, tradingsymbol, exchange="NSE", product="NRML", lot_size=1):
        try:
            funds = self.get_available_funds()
            ltp = self.kite.ltp(f"{exchange}:{tradingsymbol}")[f"{exchange}:{tradingsymbol}"]["last_price"]

            # Get margin required for 1 lot
            margin_per_lot = self.calculate_margin(tradingsymbol, exchange, "BUY", product, lot_size)
            if not margin_per_lot or margin_per_lot == 0:
                return 0

            max_lots = math.floor(funds / margin_per_lot)
            max_qty = max_lots * lot_size
            return max_qty
        except Exception as e:
            print(f"Error calculating max quantity: {e}")
            return 0

    def get_instrument_details(self, identifier, exchange="NFO"):
        """Fetch instrument details by token or tradingsymbol on demand"""
        try:
            instruments = self.kite.instruments(exchange)

            if isinstance(identifier, int):  # instrument_token
                return next((i for i in instruments if i["instrument_token"] == identifier), None)
            else:  # tradingsymbol
                return next((i for i in instruments if i["tradingsymbol"] == identifier), None)

        except Exception as e:
            print(f"Error fetching instrument details: {e}")
            return None

    def get_tradingsymbol_by_token(self, instrument_token, exchange="NFO"):
        try:
            inst = next(
                (i for i in self.instruments if
                 i["instrument_token"] == instrument_token and i["exchange"] == exchange),
                None
            )
            return inst["tradingsymbol"] if inst else None
        except Exception as e:
            print(f"Error fetching tradingsymbol: {e}")
            return None

    def get_instrument_details(self, identifier, exchange="NFO"):
        """Find instrument by tradingsymbol or token"""
        try:
            if isinstance(identifier, int):  # instrument_token passed
                return next(
                    (i for i in self.instruments if i["instrument_token"] == identifier and i["exchange"] == exchange),
                    None)
            else:  # tradingsymbol passed
                return next(
                    (i for i in self.instruments if i["tradingsymbol"] == identifier and i["exchange"] == exchange),
                    None)
        except Exception as e:
            print(f"Error fetching instrument details: {e}")
            return None

    def get_max_buy_qty_for_options(self, identifier, exchange="NFO", product="NRML"):
        """Calculate max buyable quantity for a given NFO instrument (options supported)"""
        try:
            funds = self.get_available_funds()
            inst = self.get_instrument_details(identifier, exchange)

            if not inst:
                print(f"Instrument {identifier} not found in {exchange}")
                return 0

            tradingsymbol = inst["tradingsymbol"]
            lot_size = inst["lot_size"]

            margin_info = self.kite.order_margins([{
                "exchange": exchange,
                "tradingsymbol": tradingsymbol,
                "transaction_type": "BUY",
                "variety": "regular",
                "product": product,
                "order_type": "MARKET",
                "quantity": lot_size
            }])

            margin_per_lot = margin_info[0]["total"] if margin_info else 0

            if margin_per_lot <= 0:
                return 0

            max_lots = math.floor(funds / margin_per_lot)
            return max_lots * lot_size

        except Exception as e:
            print(f"Error calculating max buy quantity: {e}")
            return 0

