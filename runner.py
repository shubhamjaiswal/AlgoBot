import backtrader as bt
import json
from strategies import Sma3_18_Cross
from broker import KiteBroker
from data_feed import LiveKiteDataMultiTimeframeSyncedPrefill
import time

# Control live vs simulated run - SET THIS TO TRUE OR FALSE
run_live = True

# Load config
with open("config.json") as f:
    cfg = json.load(f)
print(cfg)

# cerebro = bt.Cerebro(maxcpus=1)
cerebro = bt.Cerebro()
broker = broker = KiteBroker(
    api_key=cfg["api_key"],
    access_token=cfg["access_token"],
    instruments=cfg["instrument_token"],  # Your instrument tokens
    debug=False,
    live=run_live
)
cerebro.broker = broker

# # Replace with your instrument token for live trading
instrument_token_CE = cfg["instrument_token"][0]  # Example for 24500 CE
instrument_token_PE = cfg["instrument_token"][1]  # Example for 24700 PE
all_maps = broker.get_instrument_token_map()
print(f"Starting trades with : {all_maps[instrument_token_CE]} and {all_maps[instrument_token_PE]}")
print(f"Available funds : {broker.get_available_funds()}")
data = LiveKiteDataMultiTimeframeSyncedPrefill(
    symbol=all_maps[instrument_token_CE],
    instrument_token=instrument_token_CE,
    broker=broker,
    kite=broker.kite,
    timeframe=bt.TimeFrame.Minutes,
    compression=3
)
data.tradingsymbol = all_maps[instrument_token_CE]
data1 = LiveKiteDataMultiTimeframeSyncedPrefill(
    symbol=all_maps[instrument_token_PE],
    instrument_token=instrument_token_PE,  # 24500 PE
    broker=broker,
    kite=broker.kite,
    timeframe=bt.TimeFrame.Minutes,
    compression=3,
)
data1.tradingsymbol = all_maps[instrument_token_PE]
cerebro.adddata(data)
cerebro.adddata(data1)
cerebro.addstrategy(Sma3_18_Cross)
# Don't let Backtrader exit — keep running
cerebro.run(runonce=False, stdstats=False, live=True)
# Keep the script alive
while True:
    time.sleep(1)
