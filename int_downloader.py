import datetime
import pandas as pd
from kiteconnect import KiteConnect
from datetime import date
import os
import json

with open("config.json") as f:
    cfg = json.load(f)
print(cfg)

api_key = cfg["api_key"]
# api_secret = cfg["api_secret"]
access_token = cfg["access_token"]
def fetch_and_save_intraday_5min_data_per_day(kite: KiteConnect, instrument_token: int, start_date: datetime.date, end_date: datetime.date, output_dir: str):
    """
    Fetch 5-minute intraday data for a range of days and save each day's data as a separate CSV.

    Args:
        kite (KiteConnect): Authenticated KiteConnect object
        instrument_token (int): Instrument token for the instrument (e.g., NIFTY 25400 CE)
        start_date (datetime.date): Start date of data
        end_date (datetime.date): End date of data
        output_dir (str): Directory to save daily CSVs
    """
    os.makedirs(output_dir, exist_ok=True)

    current_date = start_date
    while current_date <= end_date:
        from_dt = datetime.datetime.combine(current_date, datetime.time(9, 15))
        to_dt = datetime.datetime.combine(current_date, datetime.time(15, 30))
        output_file = os.path.join(output_dir, f"{instrument_token}_{current_date.strftime('%Y%m%d')}.csv")

        try:
            data = kite.historical_data(
                instrument_token=instrument_token,
                interval='3minute',
                from_date=from_dt,
                to_date=to_dt,
                continuous=False,
                oi=True
            )

            if not data:
                print(f"No data for {current_date}")
                current_date += datetime.timedelta(days=1)
                continue

            with open(output_file, "w") as f:
                f.write("date,open,high,low,close,adjusted_close,volume\n")
                for row in data:
                    f.write(f"{row['date']},{row['open']},{row['high']},{row['low']},{row['close']},{row['close']},{row['volume']}\n")

            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"Error on {current_date}: {e}")

        current_date += datetime.timedelta(days=1)
def fetch_and_save_intraday_5min_data(kite: KiteConnect, instrument_token: int, output_file: str):
    """
    Fetch 5-minute intraday data for today and save to CSV.

    Args:
        kite (KiteConnect): Authenticated KiteConnect object
        instrument_token (int): Instrument token for symbol (e.g., NIFTY 25400 CE)
        output_file (str): Path to save CSV (e.g., 'nifty_25400_ce.csv')
    """
    from_date = datetime.datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    to_date = datetime.datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)

    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            interval='3minute',
            from_date=from_date,
            to_date=to_date,
            continuous=False,
            oi=True
        )

        if not data:
            print("No data received.")
            return

        with open(output_file, "w") as f:
            f.write("date,open,high,low,close,adjusted_close,volume\n")
            for row in data:
                f.write(
                    f"{row['date']},{row['open']},{row['high']},{row['low']},{row['close']},{row['close']},{row['volume']}\n")

        print(f"Data saved to {output_file}")

    except Exception as e:
        print(f"Error fetching data: {e}")


kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)
nifty_50 =[6401,40193,60417,81153,119553,134657,140033,177665,225537,232961,315393,341249,345089,348929,356865,408065,424961,492033,519937,633601,738561,779521,794369,857857,884737,895745,897537,969473,270529,346049,510401,850625,714625,800641,815745,939649,952193,953217,977281,1089,465729,834113,861249,267265,268801,598529,774913,215745,582849]
# fetch_and_save_intraday_5min_data(kite, instrument_token=12670722, output_file="25100_PE_15min.csv")

nifty_50 = [12205058, 12212994]# [13596674, 13598466]

for i in nifty_50:
    fetch_and_save_intraday_5min_data_per_day(
        kite=kite,
        instrument_token=i, #12105730 25200 PE #12102402 CE 25000,  # Replace with your instrument token
        start_date=date(2025, 10, 30),
        end_date=date(2025, 10,30),
        output_dir="data\\nifty\\nifty_29oct_3min"
    )