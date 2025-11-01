from kiteconnect import KiteConnect
import pandas as pd
api_key = "m1yzl03jrngy1upu"
api_secret = "x0vtoav40v1nugo5fx5aoc5durt2pbjq"
request_token = "rDNHj6qoLYZ1ZdkMIOWFpvI0e2aVOb6s"  # from redirect URL
# #
kite = KiteConnect(api_key=api_key)
# data = kite.generate_session(request_token, api_secret=api_secret)
# access_token = data["access_token"]
# print("ACCESS TOKEN:", access_token)

# https://kite.trade/connect/login?v=3&api_key=m1yzl03jrngy1upu

# Download all instruments (takes a few seconds)
# instruments = kite.instruments("NSE")  # Or just kite.instruments() for all exchanges
# df = pd.DataFrame(instruments)
# nifty_50_symbols = [
#     'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'LT',
#     'SBIN', 'AXISBANK', 'BHARTIARTL', 'KOTAKBANK', 'ASIANPAINT', 'BAJFINANCE',
#     'HCLTECH', 'MARUTI', 'WIPRO', 'SUNPHARMA', 'NTPC', 'TITAN', 'ULTRACEMCO',
#     'BAJAJFINSV', 'TECHM', 'NESTLEIND', 'POWERGRID', 'JSWSTEEL', 'TATASTEEL',
#     'GRASIM', 'HDFCLIFE', 'CIPLA', 'ONGC', 'ADANIENT', 'ADANIPORTS', 'M&M',
#     'COALINDIA', 'BRITANNIA', 'BPCL', 'DIVISLAB', 'HEROMOTOCO', 'DRREDDY', 'BAJAJ-AUTO',
#     'EICHERMOT', 'SBILIFE', 'HINDALCO', 'INDUSINDBK', 'APOLLOHOSP', 'ICICIPRULI',
#     'TATAMOTORS', 'SHREECEM'
# ]
#
# nifty_50_df = df[df['tradingsymbol'].isin(nifty_50_symbols)].reset_index(drop=True)
# View the list
# print(nifty_50_df[['tradingsymbol', 'instrument_token']])

# Optional: Save to CSV
# nifty_50_df[['tradingsymbol', 'instrument_token']].to_csv("nifty50_tokens.csv", index=False)

# # Download instrument dump
instruments = kite.instruments()
df = pd.DataFrame(instruments)
# #
# Filter for 25300 CE of current expiry
# df = df[
#     (df['name'] == 'NIFTY') &
#     (df['strike'] == 25200) &
#     (df['instrument_type'] == 'CE')
# ]
# print(df[['instrument_token', 'tradingsymbol', 'expiry']])
# df1 = df[
#     (df['name'] == 'NIFTY') &
#     (df['strike'] == 25900) &
#     (df['instrument_type'] == 'CE')
# ]

# # [12085506, 12097794]
# print(df1[['instrument_token', 'tradingsymbol', 'expiry']])

df1 = df[
    (df['name'] == 'NIFTY') &
    (df['strike'] == 26200) &
    (df['instrument_type'] == 'PE')
]
print(df1[['instrument_token', 'tradingsymbol', 'expiry']])