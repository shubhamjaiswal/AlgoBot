from kiteconnect import KiteConnect
import pandas as pd

api_key = "m1yzl03jrngy1upu"
api_secret = "x0vtoav40v1nugo5fx5aoc5durt2pbjq"
request_token = "uUJAg9PLFOWE6C5sQozDbhObe1kJwqMw"

kite = KiteConnect(api_key=api_key)
data = kite.generate_session(request_token, api_secret=api_secret)
access_token = data["access_token"]
kite.set_access_token(access_token)

print("ACCESS TOKEN:", access_token)

# ---- Download all instruments ----6u
# instruments = kite.instruments()
# df = pd.DataFrame(instruments)
#
# # ---- Filter NIFTY 25200 PE of nearest expiry ----
# nifty_pe = df[
#     (df['name'] == 'NIFTY') &
#     (df['strike'] == 25200) &
#     (df['instrument_type'] == 'PE')
# ]
#
# # Nearest expiry
# nifty_pe = nifty_pe[nifty_pe['expiry'] == nifty_pe['expiry'].min()]
#
# print(nifty_pe[['instrument_token', 'tradingsymbol', 'expiry']])
#
# # ---- Filter and Save Nifty 50 Tokens ----
# nifty_50_symbols = [...]
# nifty_50_df = df[df['tradingsymbol'].isin(nifty_50_symbols)].reset_index(drop=True)
# nifty_50_df[['tradingsymbol', 'instrument_token']].to_csv("nifty50_tokens.csv", index=False)