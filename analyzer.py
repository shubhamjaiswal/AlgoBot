from strategies import  BreakoutRetraceStrategy, ORB3MinStrategy, SupertrendPullbackStrategy,Sma3_18_Cross, StochasticDivergenceStrategy,ATRStrategy, ADXCrossoverStrategyPEAKExits, ADXCrossoverStrategy
import pandas as pd
import backtrader as bt
import os
import numpy as np
from tabulate import tabulate
# -------------------------------
# Analyzer for PnL and Win/Loss
# -------------------------------
class TradeStats(bt.Analyzer):
    def __init__(self):
        self.trades = []

    def notify_trade(self, trade):
        if trade.isclosed:
            # avoid division by zero
            if trade.size != 0:
                points = trade.pnl / abs(trade.size)
            else:
                points = 0.0

            self.trades.append({
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'points': points,
            })

    def get_analysis(self):
        df = pd.DataFrame(self.trades)
        if df.empty:
            return dict(total_trades=0, wins=0, losses=0, net_pnl=0.0, total_points=0.0)

        wins = (df['pnl'] > 0).sum()
        losses = (df['pnl'] <= 0).sum()
        total_points = df['points'].sum()

        return dict(
            total_trades=len(df),
            wins=wins,
            losses=losses,
            net_pnl=round(df['pnl'].sum(), 2),
            total_points=round(total_points, 2)
        )


# -------------------------------
# CSV Loader Helper
# -------------------------------
def load_csv(path, compression=3):
    return bt.feeds.GenericCSVData(
        dataname=path,
        dtformat='%Y-%m-%d %H:%M:%S%z',
        datetime=0,
        open=1, high=2, low=3, close=4, volume=6,
        openinterest=-1,
        timeframe=bt.TimeFrame.Minutes,
        compression=compression,
    )

def consolidate_results(input_file, output_file="sme_nifty_summary.csv"):
    # Input CSV file path
    # input_file = "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\stoch_nifty.csv"  # replace with your file name

    # Read CSV
    df = pd.read_csv(input_file)

    # Ensure numeric columns are properly converted
    df["net_pnl"] = pd.to_numeric(df["net_pnl"], errors="coerce")
    df["wins"] = pd.to_numeric(df["wins"], errors="coerce")
    df["losses"] = pd.to_numeric(df["losses"], errors="coerce")
    df["total_trades"] = pd.to_numeric(df["total_trades"], errors="coerce")

    # Group by SME3/SME19 combo
    grouped = (
        df.groupby(["SME 3", "SME 19"])
        .agg(
            total_pnl=("net_pnl", "sum"),
            avg_pnl=("net_pnl", "mean"),
            total_trades=("total_trades", "sum"),
            total_wins=("wins", "sum"),
            total_losses=("losses", "sum"),
            files=("CE_file", "nunique"),  # number of unique test files
        )
        .reset_index()
    )

    # Calculate winrate
    grouped["win_rate_%"] = (grouped["total_wins"] / (grouped["total_wins"] + grouped["total_losses"])) * 100

    # Sort by total_pnl descending
    grouped = grouped.sort_values("total_pnl", ascending=False)

    # Save summary
    grouped.to_csv(output_file, index=False)

    # Display top results
    print("\n=== P&L Summary by (SME3, SME19) ===")
    print(grouped.head(20))
    print(f"\nSaved summary to: {output_file}")


# -------------------------------
# Multi-run Manager
# -------------------------------
def run_all(data_pairs, strategies, optimize=False):
    all_results = []

    for ce_path, pe_path in data_pairs:
        for strat in strategies:
            cerebro = bt.Cerebro()
            ce_data, pe_data = load_csv(ce_path), load_csv(pe_path)
            ce_data.tradingsymbol = "CE"
            pe_data.tradingsymbol = "PE"
            cerebro.adddata(ce_data, name="CE")
            cerebro.adddata(pe_data, name="PE")

            if optimize and strat.__name__ == "ADXCrossoverStrategy":
                cerebro.optstrategy(
                    strat,
                    adx_period=range(8, 15, 1),   # test 6, 8, 10, 12, 14
                    adx_threshold=range(14, 22, 1),# test 6, 8, 10, 12, 14
                )
            elif optimize and strat.__name__ == "Sma2Sma7CrossoverBarCheckMulti":
                cerebro.optstrategy(
                    strat,
                    sma3=range(3, 8, 1),   # test 6, 8, 10, 12, 14
                    sma7=range(9, 15, 1),# test 6, 8, 10, 12, 14
                    sma19 = range(16, 20, 1),  # test 6, 8, 10, 12, 14
                )
            elif optimize and strat.__name__ == "Sma3_18_Cross":
                cerebro.optstrategy(
                    strat,
                    sma3=range(3, 8, 1),   # test 6, 8, 10, 12, 14
                    sma19 = range(9, 18, 1),  # test 6, 8, 10, 12, 14
                    # trail_trigger_pct= np.arange(0.1, 0.3, 0.04),
                    choppiness_threshold = range(50,51,1)
                )
            elif optimize and strat.__name__ == "BreakoutRetraceStrategy":

                cerebro.optstrategy(
                    strat,
                    n=range(7, 12, 1),  # integer okay
                    vol_multiplier=np.arange(1.0, 1.6, 0.1),  # float range
                    retrace_pct=np.arange(0.15, 0.31, 0.05),  # float range
                    rr=np.arange(2.0, 2.6, 0.1),  # float range
                )
            else:
                cerebro.addstrategy(strat)

            cerebro.addanalyzer(TradeStats, _name='trade_stats')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            # cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            # cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
            results = cerebro.run()
            # stats = results[0].analyzers.trade_stats.get_analysis()
            # print(f":::: Trades found for {strat.__name__}:::::", stats)
            # handle optimization differently
            if optimize and strat.__name__ == "ADXCrossoverStrategy":
                print("1")
                for res in results:   # each res is a list of strategy instances
                    print("2")

                    stats = res[0].analyzers.trade_stats.get_analysis()
                    dd = res[0].analyzers.drawdown.get_analysis()
                    all_results.append({
                        'CE_file': os.path.basename(ce_path),
                        'PE_file': os.path.basename(pe_path),
                        'strategy': strat.__name__,
                        'adx_period': res[0].params.adx_period,
                        'adx_threshold': res[0].params.adx_threshold,
                        'max_drawdown': round(dd.max.drawdown, 2) if dd else None,
                        **stats
                    })
            elif optimize and strat.__name__ == "Sma3_18_Cross":
                for res in results:   # each res is a list of strategy instances
                    stats = res[0].analyzers.trade_stats.get_analysis()
                    dd = res[0].analyzers.drawdown.get_analysis()
                    all_results.append({
                        'CE_file': os.path.basename(ce_path),
                        'PE_file': os.path.basename(pe_path),
                        'strategy': strat.__name__,
                        'SME 3': res[0].params.sma3,
                        'SME 19': res[0].params.sma19,
                        'CHoppiness': res[0].params.choppiness_threshold,
                        # 'trail_trigger_pct' : res[0].params.trail_trigger_pct,
                        **stats
                    })
            elif optimize and strat.__name__ == "Sma2Sma7CrossoverBarCheckMulti":
                for res in results:  # each res is a list of strategy instances
                    stats = res[0].analyzers.trade_stats.get_analysis()
                    dd = res[0].analyzers.drawdown.get_analysis()
                    all_results.append({
                        'CE_file': os.path.basename(ce_path),
                        'PE_file': os.path.basename(pe_path),
                        'strategy': strat.__name__,
                        'SME 3': res[0].params.sma3,
                        'SME 7': res[0].params.sma7,
                        'SME 19': res[0].params.sma19,
                        **stats
                    })
            elif optimize and strat.__name__ == "BreakoutRetraceStrategy":
                for res in results:  # each res is a list of strategy instances
                    stats = res[0].analyzers.trade_stats.get_analysis()
                    dd = res[0].analyzers.drawdown.get_analysis()
                    all_results.append({
                        'CE_file': os.path.basename(ce_path),
                        'PE_file': os.path.basename(pe_path),
                        'strategy': strat.__name__,
                        'n': res[0].params.n,
                        'vol_multiplier': res[0].params.vol_multiplier,
                        'retrace_pct': res[0].params.retrace_pct,
                        'rr': res[0].params.rr,
                        **stats
                    })
                # stats = results[0].analyzers.trade_stats.get_analysis()
                # all_results.append({
                #     'CE_file': os.path.basename(ce_path),
                #     'PE_file': os.path.basename(pe_path),
                #     'strategy': strat.__name__,
                #     **stats
                # })
            else:
                # 🔹 Non-optimized single run
                strat_instance = results[0]
                stats = strat_instance.analyzers.trade_stats.get_analysis()
                dd = strat_instance.analyzers.drawdown.get_analysis()
                all_results.append({
                    'CE_file': os.path.basename(ce_path),
                    'PE_file': os.path.basename(pe_path),
                    'strategy': strat.__name__,
                    'max_drawdown': round(dd.max.drawdown, 2) if dd else None,
                    **stats
                })

    return pd.DataFrame(all_results)

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Define CE/PE file pairs you want to test
    data_pairs = [
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24900PE_old\\14280450_20250730.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24600CE_old\\14283778_20250730.csv"),
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24600CE_29\\14280450_20250729.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24900PE_29\\14283778_20250729.csv"),
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24900PE_3107\\14283778_20250731.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24600CE_3107\\14280450_20250731.csv"),
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24800PE_18\\10203906_20250801.csv",
#     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24600CE_18\\10202626_20250801.csv"),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24700CE_48\\10202626_20250804.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24800PE_48\\10203906_20250804.csv"
#
#         ),
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24800PE_58\\10203906_20250805.csv",
#     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24600CE_58\\10200322_20250805.csv"),
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24800PE_68\\10203906_20250806.csv",
#     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24500CE_68\\10196738_20250806.csv"),
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24600PE_78\\10200578_20250807.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24400CE_78\\10189826_20250807.csv"), #24400 CE
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24500PE_88\\11385090_20250808.csv", # 24500 PE
#     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24400CE_88\\11383810_20250808.csv"),
#
#     ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24300CE_118\\11382274_20250811.csv",
#     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_24500PE_118\\11385090_20250811.csv"),
#
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_12_08\\11384834_20250812.csv",
#         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_12_08\\11391746_20250812.csv"),
#
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_12_08\\11383810_20250813.csv",
# "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_12_08\\11391746_20250813.csv"),
#
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_14_08\\11383810_20250814.csv",
#         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_14_08\\11393282_20250814.csv"),
#
#
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_18_08_3min\\12085506_20250818.csv",
#         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_18_08_3min\\12091394_20250818.csv"),
#
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_19_08_3min\\12085506_20250819.csv",
#         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_19_08_3min\\12091394_20250819.csv"),
#
#
#
#         (   "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_20_08_3min\\12085506_20250820.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_20_08_3min\\12096770_20250820.csv"),
# #
#
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_20_08_3min\\12091138_20250821.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_20_08_3min\\12097794_20250821.csv"),
#
#
# #
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_22_08_3min\\18425346_20250822.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_22_08_3min\\18429186_20250822.csv"
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_25_08_3min\\18423298_20250825.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_25_08_3min\\18429186_20250825.csv"
#
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_26_08_3min\\18422274_20250826.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_26_08_3min\\18424578_20250826.csv"
#
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_28_08_3min\\18420226_20250828.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_28_08_3min\\18422530_20250828.csv"
#         ),
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_29_08_3min\\12524802_20250829.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_29_08_3min\\12527618_20250829.csv"),
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_1_09_3min\\12524802_20250901.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_1_09_3min\\12528642_20250901.csv"),
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_2_09_3min\\12525826_20250902.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_2_09_3min\\12528642_20250902.csv"),
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_3_09_3min\\10384898_20250903.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_3_09_3min\\10387970_20250903.csv"),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_4_09_3min\\10388738_20250904.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_4_09_3min\\10391554_20250904.csv"
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_5_09_3min\\10387714_20250905.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_5_09_3min\\10390018_20250905.csv"
#         ),
#         (
#            "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_8_09_3min\\10385922_20250908.csv",
#            "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_8_09_3min\\10388994_20250908.csv"
#
#         ),
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_9_09_3min\\10387714_20250909.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_9_09_3min\\10390018_20250909.csv"
#         ),
#         (# 10 sep
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_9_09_5min\\11427330_20250910.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_9_09_5min\\11439874_20250910.csv"
#
#         ),
#         (  # 11 sep
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_12_3min\\11430914_20250912.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_12_3min\\11453698_20250912.csv"
#
#         ),
#         (#12 sep
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_11_3min\\11427330_20250911.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_11_3min\\11450114_20250911.csv"),
#         # 16
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_16_3min\\11435010_20250916.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_16_3min\\11453698_20250916.csv"),
#
#         # 17 sep
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_17_3min\\12225794_20250917.csv",
#         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_17_3min\\12228098_20250917.csv"),
#
#         # 18 Sept
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_18_3min\\12226818_20250918.csv",
#         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_18_3min\\12231170_20250918.csv"),
#
#         # 19 nifty
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_19_3min\\12226818_20250919.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_19_3min\\12231170_20250919.csv"),
#
#         # 22 sept
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_22_3min\\12224770_20250922.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_22_3min\\12227074_20250922.csv"),
#
#         # nifty 23 sept
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_23_3min\\12224770_20250923.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_23_3min\\12227074_20250923.csv"),
#
#         #24
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_24_3min\\16561666_20250924.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_24_3min\\15484930_20250924.csv"),
#
#         # 25 sep
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_25_3min\\16561666_20250925.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_25_3min\\15484930_20250925.csv"),
#
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_26_3min\\15476994_20250926.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_26_3min\\15480322_20250926.csv"),
#
#         (
#             # 29 sep nifty
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_29_3min\\15475970_20250929.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_9_29_3min\\15479298_20250929.csv"
#         ),
#         (
#         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_1_3min\\9819394_20251001.csv",
#         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_1_3min\\9822722_20251001.csv"
#         ),
#         (
#         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_3_3min\\9820418_20251003.csv",
#         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_3_3min\\9825026_20251003.csv"
#
#         ),
#         ("C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_6_3min\\9822466_20251006.csv",
#          "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_6_3min\\9827586_20251006.csv"),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_7_3min\\9827074_20251007.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_7_3min\\9830402_20251007.csv"
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_8_3min\\10924290_20251008.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_8_3min\\10929666_20251008.csv"
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_9_3min\\10924290_20251009.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_9_3min\\10928642_20251009.csv"
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_10_3min\\10925826_20251010.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_10_3min\\10929666_20251010.csv"
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_13_3min\\10925826_20251013.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_13_3min\\10929666_20251013.csv"
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_14_3min\\10927874_20251014.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_14_3min\\10932226_20251014.csv"
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_15_3min\\11585538_20251015.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_15_3min\\11587842_20251015.csv"
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_16_3min\\11586562_20251016.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_16_3min\\11588866_20251016.csv"
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_16_3min\\11587586_20251017.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_16_3min\\11591426_20251017.csv"
#         ),
#         (
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_20_3min\\11591170_20251020.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_20_3min\\11593474_20251020.csv"
#         ),
# #         # ( #PURE ITM repeat on 10th . better results
# #         #     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_20_3min\\11591170_20251020.csv",
# #         #     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_20_3min\\11594498_20251020.csv"
# #         # )
#         ( # 23 oct
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_23_3min\\15097346_20251023.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_23_3min\\15112706_20251023.csv"
#         ),
#         (  # 24 oct
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_24_3min\\15092738_20251024.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_24_3min\\15109890_20251024.csv"
#         ),
#         (  # 27 oct
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_27_3min\\15092738_20251027.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_27_3min\\15109890_20251027.csv",
#
#         ),
#         # (  # 27 oct - closer
#         #     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\alt_nifty_10_27_3min\\15094530_20251027.csv",
#         #     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\alt_nifty_10_27_3min\\15105282_20251027.csv",
#         #
#         # ),
#         (  # 28 oct
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_28_3min\\15094530_20251028.csv",
#             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_28_3min\\15109890_20251028.csv",
#
#         ),
        (  # 29 oct
            "data\\nifty\\nifty_29oct_3min\\12205058_20251030.csv",
            "data\\nifty\\nifty_29oct_3min\\12212994_20251030.csv",

        ),



    ]
    #
    # data_pairs = [
    #     (  # 1 Oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_1_bnifty_3min\\13572866_20251001.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_1_bnifty_3min\\13587202_20251001.csv"
    #     ),
    #     (  # 3 Oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_3_bnifty_3min\\13586946_20251003.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_3_bnifty_3min\\13589762_20251003.csv"
    #     ),
    #     (  # 6 Oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_6_bnifty_3min\\13591042_20251006.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_6_bnifty_3min\\13593858_20251006.csv"
    #     ),
    #     (  # 7 Oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_7_bnifty_3min\\13592578_20251007.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_7_bnifty_3min\\13595906_20251007.csv"
    #     ),
    #     (  # 8 Oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_10_bnifty_3min\\13592578_20251008.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_10_bnifty_3min\\13595906_20251008.csv"
    #     ),
    #     (  # 9 Oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_10_bnifty_3min\\13592066_20251009.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_10_bnifty_3min\\13594370_20251009.csv"
    #     ),
    #     (  # 10 Oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_10_bnifty_3min\\13592578_20251010.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_10_bnifty_3min\\13595906_20251010.csv"
    #     ),
    #     (  # 13 Oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_13_bnifty_3min\\13593602_20251013.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\oct_13_bnifty_3min\\13596930_20251013.csv"
    #     ),
    #     ( #14 oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_14_3min\\13597186_20251014.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_14_3min\\13598466_20251014.csv"
    #     ),
    #     ( # 15 oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_15_3min\\13596674_20251015.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_15_3min\\13598466_20251015.csv"
    #     ),
    #     (  # 16 oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_16_3min\\13600258_20251016.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_16_3min\\13604610_20251016.csv"
    #     ),
    #     (  # 17 Oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_17_3min\\13600258_20251017.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_17_3min\\13605122_20251017.csv"
    #     ),
    # ( #20 Oct
    #     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_20_3min\\13604354_20251020.csv",
    #     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_20_3min\\13613570_20251020.csv"
    # ),
    # (  # 23 Oct
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_23_3min\\13607682_20251023.csv",
    #         "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_23_3min\\13617666_20251023.csv"
    #     ),
    #     (  # 24 Oct
    #                 "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_24_3min\\13605890_20251024.csv",
    #                 "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_24_3min\\13613570_20251024.csv"
    #             ),
    #     (  # 27 Oct
    #     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_27_3min\\13604354_20251027.csv",
    #     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\nifty_10_27_3min\\13611010_20251027.csv"
    # ),
    #     (  # 28 Oct
    #             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_28_3min\\15094530_20251028.csv",
    #             "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_10_28_3min\\13605378_20251028.csv"
    #         ),
    #     (  # 28 Oct
    #                     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_29oct_3min\\12815618_20251029.csv",
    #                     "C:\\Users\\Shubham_Jaiswal1\\PycharmProjects\\backtrader\\Bot\\utils\\bnifty_29oct_3min\\12820994_20251029.csv"
    #                 ),

    #
    # ]
    # Add multiple strategies here
    optimize_flag = False
    result_file = "stoch_nifty_single.csv"
    strategies = [Sma3_18_Cross]#]ADXCrossoverStrategyPEAKExits#Sma2Sma7CrossoverBarCheckMulti,]#]#ATRStrategy]#]#,ADXCrossoverStrategyPEAKExits]# , Sma2Sma7CrossoverBarCheckMulti, ], Sma2Sma7CrossoverBarCheckMulti, SmaAtrStrategy]  # add Sma2Sma7CrossoverBarCheckMulti etc.

    df_results = run_all(data_pairs, strategies, optimize=optimize_flag)
    print(df_results)
    # Pretty print table in console
    print(tabulate(df_results, headers="keys", tablefmt="pretty", showindex=True))
    df_results.to_csv(result_file, index=False)
    if optimize_flag:
        consolidate_results(input_file=result_file, output_file="stoch_nifty_all_2910.csv")
    # 🔹 Dump top 4 by net profit
    # if not df_results.empty and "net_pnl" in df_results.columns:
    #     df_top = df_results.sort_values(by="net_pnl", ascending=False).head(8)
    #     print("\n=== Top Strategies by Net Profit ===")
    #     print(tabulate(df_top, headers="keys", tablefmt="pretty", showindex=False))
    #     df_top.to_csv("backtest_top8_20_08.csv", index=False)
    #
    #     df_top = df_results.sort_values(by="net_pnl", ascending=True).head(8)
    #     print("\n=== Worst Strategies by Net Profit ===")
    #     print(tabulate(df_top, headers="keys", tablefmt="pretty", showindex=False))
    #     df_top.to_csv("backtest_top8_20_08.csv", index=False)
    # else:
    #     print("\n⚠️ No results found or net_pnl column missing.")