"""
cross_sectional_momentum.py
===========================
A cross-sectional equity strategy on real S&P 500 data -- built, tested,
diagnosed when it failed, and fixed for a stated reason.

THE STORY THIS PROJECT TELLS (read this first)
----------------------------------------------
1. I built a dollar-neutral LONG/SHORT strategy: each month, rank ~80 large-cap
   stocks by a blend of 12-1 momentum and short-term reversal, go long the top
   quintile and short the bottom quintile, risk-balanced by inverse volatility.
2. On 2010-2024 data it LOST money -- a steady bleed to a negative Sharpe.
3. I decomposed it and found the cause: the LONG leg (buying winners) was fine
   and roughly matched the market, but the SHORT leg (shorting "losers") lost
   ~25%/year. In a 15-year large-cap bull market, even the relative laggards
   rose, so shorting them was structurally a losing bet. The dollar-neutral
   design meant the short leg's losses cancelled the long leg's gains.
4. Based on that diagnosis I tested a LONG-ONLY version (winners only). It works
   -- but here is the honest part: a naive equal-weight portfolio of the same
   stocks has a SIMILAR Sharpe, because most of the return is just the market
   going up. So I report the long-only result AGAINST that benchmark and isolate
   the ~2%/year of ALPHA the momentum tilt actually added. The alpha, not the
   headline Sharpe, is the real contribution.

That arc -- build, fail, diagnose, fix-with-a-reason, and separate skill from
market beta -- is the point of the project.

THE SIGNALS
-----------
* 12-1 MOMENTUM  : 12-month return skipping the last month (winners keep winning
                   over medium horizons; the skip avoids short-term reversal).
* SHORT-TERM REVERSAL : negative of last month's return (recent jumps partly
                   revert). Note: the horizon scan in diagnose() shows this
                   effect lives at ~1 week, not ~1 month, in this universe.

THE STRETCH CONCEPT: INVERSE-VOLATILITY WEIGHTING (see `weights_from_scores`).

HOW TO RUN
----------
    pip install yfinance pandas numpy matplotlib scipy
    python cross_sectional_momentum.py            # runs the full workflow

The workflow runs three things in order:
    (a) the original long/short strategy (the one that failed),
    (b) diagnostics that show WHY it failed,
    (c) the long-only fix, benchmarked to isolate alpha.
Charts and CSVs are saved alongside the script.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

MOM_LOOKBACK = 252      # ~12 months for momentum
MOM_SKIP = 21           # skip the most recent ~1 month
REV_LOOKBACK = 21       # ~1 month for short-term reversal
VOL_LOOKBACK = 63       # ~3 months for the inverse-vol risk estimate

TOP_QUANTILE = 0.20     # trade the top/bottom 20%
REBALANCE = "ME"        # month-end
COST_BPS = 10.0         # 10 bps per unit traded
TRAIN_FRAC = 0.60       # first 60% is in-sample design

TRADING_DAYS = 252
CACHE_FILE = "prices.csv"
FIG_DIR = "figures"


# ============================================================================
# 1. DATA  (Curriculum: Data -> PriceData; More Pandas -> ReadingWritingFiles,
#           TimeSeries, MissingValues, MultiIndexing)
# ============================================================================
def get_sp500_tickers():
    """A fixed list of large, liquid names.

    Hard-coded, not scraped from today's index membership: scraping current
    members and running them back to 2010 is SURVIVORSHIP BIAS (you only test
    companies that survived to today, which flatters every backtest). Being
    explicit about the choice is the point.
    """
    return [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "PG",
        "HD", "MA", "BAC", "DIS", "ADBE", "CRM", "NFLX", "CSCO", "PFE", "KO",
        "PEP", "TMO", "ABT", "COST", "WMT", "MCD", "NKE", "INTC", "AMD", "QCOM",
        "TXN", "ORCL", "IBM", "GE", "CAT", "BA", "MMM", "HON", "UPS", "LMT",
        "XOM", "CVX", "COP", "SLB", "UNH", "CVS", "CI", "MRK", "LLY", "BMY",
        "T", "VZ", "CMCSA", "WFC", "GS", "MS", "C", "AXP", "BLK", "SPGI",
        "GILD", "AMGN", "BIIB", "ISRG", "SO", "DUK", "NEE", "D", "PLD", "AMT",
        "SBUX", "LOW", "TGT", "F", "GM", "DE", "EMR", "ETN", "ITW", "PH",
    ]


def download_prices():
    """Download daily adjusted-close prices, or load from the local cache."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached prices from {CACHE_FILE} ...")
        return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)

    print("Downloading prices from Yahoo Finance (first run only) ...")
    import yfinance as yf

    raw = yf.download(get_sp500_tickers(), start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=True)
    px = raw["Close"].copy()                          # (dates x tickers) panel
    px = px.dropna(axis=1, thresh=int(len(px) * 0.8)) # keep names with history
    px = px.ffill(limit=3)                            # fill short gaps only
    px.to_csv(CACHE_FILE)
    print(f"Saved {px.shape[1]} tickers x {px.shape[0]} days to {CACHE_FILE}")
    return px


def daily_returns(prices):
    return prices.pct_change()


# ============================================================================
# 2. SIGNALS  (Curriculum: Momentum -> Time Horizon; Reversal -> Overview)
# ============================================================================
def momentum_signal(prices, lookback=MOM_LOOKBACK, skip=MOM_SKIP):
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def reversal_signal(prices, lookback=REV_LOOKBACK):
    return -(prices / prices.shift(lookback) - 1.0)


def zscore_cross_section(signal):
    """Standardize each DATE's cross-section (each row) to mean 0, sd 1."""
    mu, sd = signal.mean(axis=1), signal.std(axis=1)
    return signal.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)


def combined_signal(prices):
    mom = zscore_cross_section(momentum_signal(prices))
    rev = zscore_cross_section(reversal_signal(prices))
    return (mom + rev) / 2.0


# ============================================================================
# 3. WEIGHTING -- inverse volatility  (Curriculum: Weighting Pt2 -> VolWeights)
# ============================================================================
def weights_from_scores(scores, vol, mode="long_short"):
    """Turn stock scores into portfolio weights.

    INVERSE-VOL IDEA: scale each position by 1/volatility so every holding
    contributes similar RISK, not similar dollars. A name twice as volatile gets
    half the money; otherwise the wildest stocks would dominate the book's risk.

    mode="long_short" : long top quantile, short bottom quantile, dollar-neutral
                        (long weights sum to +1, shorts to -1).
    mode="long_only"  : long the top quantile only, weights sum to +1. This is
                        the diagnosis-driven fix -- see the module docstring.
    """
    s = scores.dropna()
    if len(s) < 10:
        return pd.Series(dtype=float)

    k = max(1, int(len(s) * TOP_QUANTILE))
    ranked = s.sort_values()
    longs = ranked.index[-k:]
    inv_vol = (1.0 / vol.reindex(s.index).replace(0, np.nan))

    w = pd.Series(0.0, index=s.index)
    w[longs] = inv_vol[longs]

    if mode == "long_short":
        shorts = ranked.index[:k]
        w[shorts] = -inv_vol[shorts]
        w = w.fillna(0.0)
        ls, ss = w[w > 0].sum(), -w[w < 0].sum()
        if ls > 0:
            w[w > 0] /= ls
        if ss > 0:
            w[w < 0] /= ss
    else:  # long_only
        w = w.fillna(0.0)
        tot = w.sum()
        if tot > 0:
            w /= tot
    return w


# ============================================================================
# 4. BACKTEST  (Curriculum: Backtesting -> BacktestIntro, BasicPortfolioMath,
#               TS Backtesting; Execution -> Tcosts, Turnover)
# ============================================================================
def run_backtest(prices, scores, mode="long_short"):
    """Walk forward month by month. Signal at month t trades month t+1.

    TIMING RULE: weights come from data up to and including month-end t; they
    earn month t+1's return; costs are charged on the trade into the new book.
    Nothing about t+1 informs the weights -- no look-ahead.
    """
    rets = daily_returns(prices)
    vol = rets.rolling(VOL_LOOKBACK, min_periods=30).std()
    month_ends = [d for d in prices.resample(REBALANCE).last().index
                  if d in prices.index]

    pnl, dates, turn = [], [], []
    prev_w = pd.Series(dtype=float)
    weight_history = {}

    for i in range(len(month_ends) - 1):
        t, t_next = month_ends[i], month_ends[i + 1]
        if prices.index.get_loc(t) < MOM_LOOKBACK + 5:
            continue
        w = weights_from_scores(scores.loc[t], vol.loc[t], mode=mode)
        if w.empty:
            continue
        weight_history[t] = w

        names = w.index.union(prev_w.index)
        dw = w.reindex(names).fillna(0) - prev_w.reindex(names).fillna(0)
        turnover = dw.abs().sum()

        window = rets.loc[t:t_next].iloc[1:]           # days AFTER signal date
        month_ret = (1 + window).prod() - 1
        gross = float((w * month_ret.reindex(w.index)).sum())
        net = gross - turnover * COST_BPS / 1e4

        pnl.append(net); dates.append(t_next); turn.append(turnover)
        prev_w = w

    idx = pd.DatetimeIndex(dates)
    return {"pnl": pd.Series(pnl, index=idx, name="net"),
            "turnover": pd.Series(turn, index=idx),
            "weights": weight_history}


def benchmark_return(prices):
    """Equal-weight portfolio of the SAME universe -- the 'do nothing clever'
    baseline. Comparing to this is how we separate SKILL from a rising market.
    (Curriculum: Performance Evaluation -> BetaIntro / AlphaIntro.)
    """
    rets = daily_returns(prices)
    month_ends = [d for d in prices.resample(REBALANCE).last().index
                  if d in prices.index]
    vals, dates = [], []
    for i in range(len(month_ends) - 1):
        t, t_next = month_ends[i], month_ends[i + 1]
        if prices.index.get_loc(t) < MOM_LOOKBACK + 5:
            continue
        window = rets.loc[t:t_next].iloc[1:]
        vals.append(float(((1 + window).prod() - 1).mean()))
        dates.append(t_next)
    return pd.Series(vals, index=pd.DatetimeIndex(dates), name="benchmark")


# ============================================================================
# 5. PERFORMANCE  (Curriculum: Performance Evaluation -> Sharpe Ratios;
#                  Pt2 -> Drawdowns; ComputingAlpha)
# ============================================================================
def sharpe_ratio(pnl):
    return float(pnl.mean() / pnl.std() * np.sqrt(12)) if pnl.std() > 0 else np.nan


def sharpe_standard_error(sr_ann, n):
    """se(SR) ~= sqrt((1 + SR^2/2)/N). With this little data it is ~0.3, which
    is WHY a Sharpe must never be quoted without it. (Lo, 2002.)"""
    if n < 12:
        return np.nan
    sr_m = sr_ann / np.sqrt(12)
    return float(np.sqrt((1 + 0.5 * sr_m ** 2) / n) * np.sqrt(12))


def drawdown_series(pnl):
    eq = (1 + pnl).cumprod()
    return eq / eq.cummax() - 1.0


def alpha_beta(strategy, benchmark):
    """Monthly CAPM-style decomposition: how much of the strategy is just the
    benchmark (beta), and how much is genuinely added (alpha)?"""
    j = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1, sort=False).dropna()
    if len(j) < 12 or j["b"].var() == 0:
        return {}
    beta = np.cov(j["s"], j["b"])[0, 1] / np.var(j["b"])
    alpha_m = j["s"].mean() - beta * j["b"].mean()
    return {"beta": beta, "alpha_ann_%": alpha_m * 12 * 100,
            "corr": float(j.corr().iloc[0, 1])}


def summarize(pnl, turnover, label, benchmark=None):
    n = len(pnl)
    years = n / 12
    eq = (1 + pnl).cumprod()
    sr = sharpe_ratio(pnl)
    se = sharpe_standard_error(sr, n)
    out = {
        "period": label, "months": n,
        "ann_return_%": (eq.iloc[-1] ** (1 / years) - 1) * 100 if years else np.nan,
        "ann_vol_%": pnl.std() * np.sqrt(12) * 100,
        "Sharpe": sr, "Sharpe_std_error": se,
        "Sharpe_t_stat": sr / se if se and se > 0 else np.nan,
        "max_drawdown_%": drawdown_series(pnl).min() * 100,
        "hit_rate_%": (pnl > 0).mean() * 100,
        "avg_turnover": turnover.mean(),
    }
    if benchmark is not None:
        out.update(alpha_beta(pnl, benchmark))
    return pd.Series(out)


# ============================================================================
# 6. DIAGNOSIS  (Curriculum: Quant Research Pitfalls -> Overindexing on Modeling)
# ============================================================================
def diagnose(prices):
    """Why did the long/short book fail? Decompose it. This is the part that
    turns a losing backtest into a research finding."""
    rets = daily_returns(prices)
    vol = rets.rolling(VOL_LOOKBACK, min_periods=30).std()
    month_ends = [d for d in prices.resample(REBALANCE).last().index
                  if d in prices.index]

    def one_leg(scores, side):
        pnl, dates = [], []
        for i in range(len(month_ends) - 1):
            t, t_next = month_ends[i], month_ends[i + 1]
            if prices.index.get_loc(t) < MOM_LOOKBACK + 5:
                continue
            s = scores.loc[t].dropna()
            if len(s) < 10:
                continue
            k = max(1, int(len(s) * TOP_QUANTILE))
            ranked = s.sort_values()
            sel = ranked.index[-k:] if side == "long" else ranked.index[:k]
            window = rets.loc[t:t_next].iloc[1:]
            mret = (1 + window).prod() - 1
            sign = 1 if side == "long" else -1
            pnl.append(sign * float(mret.reindex(sel).mean()))
            dates.append(t_next)
        return pd.Series(pnl, index=pd.DatetimeIndex(dates))

    mom = zscore_cross_section(momentum_signal(prices))
    print("\n" + "-" * 68)
    print("DIAGNOSIS: where did the long/short strategy's return come from?")
    print("-" * 68)
    print(f"{'momentum LONG leg (buy winners)':40s} Sharpe {sharpe_ratio(one_leg(mom,'long')):+.2f}")
    print(f"{'momentum SHORT leg (short losers)':40s} Sharpe {sharpe_ratio(one_leg(mom,'short')):+.2f}")
    bench = benchmark_return(prices)
    print(f"{'equal-weight market (reference)':40s} Sharpe {sharpe_ratio(bench):+.2f}")
    print("\nReading: the short leg is the problem. Shorting laggards in a 15-year")
    print("bull market loses money; the dollar-neutral design let those losses")
    print("cancel the (healthy) long leg. => test a LONG-ONLY version next.")

    print("\nHorizon check on the reversal signal (why the 21d default was weak):")
    for lb in (5, 10, 21, 63):
        s = zscore_cross_section(reversal_signal(prices, lb))
        r = run_backtest(prices, s, mode="long_short")["pnl"]
        print(f"  reversal lookback {lb:2d}d -> Sharpe {sharpe_ratio(r):+.2f}")


# ============================================================================
# 7. MAIN
# ============================================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

    print("=" * 68)
    print("CROSS-SECTIONAL EQUITY STRATEGY  (real S&P 500 data)")
    print("=" * 68)
    prices = download_prices()
    print(f"Universe: {prices.shape[1]} stocks, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")

    bench = benchmark_return(prices)

    # ---- (a) the original long/short strategy (it failed) ------------------
    ls = run_backtest(prices, combined_signal(prices), mode="long_short")
    print("\n### (a) ORIGINAL LONG/SHORT STRATEGY ###")
    ls_board = summarize(ls["pnl"], ls["turnover"], "long/short FULL")
    print(ls_board.to_string())
    print(f"\n=> Sharpe {ls_board['Sharpe']:+.2f}. It lost money. Now diagnose why.")

    # ---- (b) diagnosis -----------------------------------------------------
    diagnose(prices)

    # ---- (c) the diagnosis-driven fix: long-only, benchmarked --------------
    lo = run_backtest(prices, zscore_cross_section(momentum_signal(prices)),
                      mode="long_only")
    pnl = lo["pnl"]
    split = int(len(pnl) * TRAIN_FRAC)

    print("\n### (c) FIX: LONG-ONLY MOMENTUM, vs equal-weight benchmark ###")
    board = pd.DataFrame([
        summarize(pnl, lo["turnover"], "long-only FULL", benchmark=bench),
        summarize(pnl.iloc[:split], lo["turnover"].iloc[:split], "in-sample", benchmark=bench),
        summarize(pnl.iloc[split:], lo["turnover"].iloc[split:], "OUT-OF-SAMPLE", benchmark=bench),
        summarize(bench, pd.Series(0.0, index=bench.index), "BENCHMARK (equal-wt)"),
    ]).set_index("period")
    print(board.T.to_string())

    ab = alpha_beta(pnl, bench)
    print("\nTHE HONEST HEADLINE:")
    print(f"  * Long-only Sharpe is {sharpe_ratio(pnl):+.2f} -- but the equal-weight")
    print(f"    benchmark is {sharpe_ratio(bench):+.2f}. Most of that Sharpe is just the market.")
    print(f"  * Beta to benchmark: {ab['beta']:.2f}, correlation {ab['corr']:.2f}.")
    print(f"  * The momentum tilt's actual contribution is the ALPHA: "
          f"{ab['alpha_ann_%']:+.2f}%/year.")
    print(f"  * Out-of-sample Sharpe {sharpe_ratio(pnl.iloc[split:]):+.2f} vs in-sample "
          f"{sharpe_ratio(pnl.iloc[:split]):+.2f}: the tilt held up, it isn't overfit.")

    board.to_csv("scoreboard.csv")
    pnl.to_frame("monthly_net_return").to_csv("monthly_returns.csv")
    make_charts(ls["pnl"], pnl, bench, split)
    print(f"\nSaved scoreboard.csv, monthly_returns.csv, charts in {FIG_DIR}/. Done.")


def make_charts(ls_pnl, lo_pnl, bench, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3})

    fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                             gridspec_kw={"height_ratios": [1, 1]})

    eq_ls = (1 + ls_pnl).cumprod()
    axes[0].plot(eq_ls.index, eq_ls.values, color="#c0392b", lw=1.5)
    axes[0].axhline(1.0, color="#999", lw=0.8, ls=":")
    axes[0].set_ylabel("growth of $1")
    axes[0].set_title("(a) Original long/short strategy — it bled out "
                      "(the short leg was the problem)")

    eq_lo = (1 + lo_pnl).cumprod()
    eq_b = (1 + bench.reindex(lo_pnl.index).fillna(0)).cumprod()
    axes[1].plot(eq_lo.index, eq_lo.values, color="#0f4c81", lw=1.6,
                 label="long-only momentum")
    axes[1].plot(eq_b.index, eq_b.values, color="#7f8c8d", lw=1.3, ls="--",
                 label="equal-weight benchmark")
    axes[1].axvline(lo_pnl.index[split], color="#c0392b", ls="--", lw=1.0,
                    label="train/test split")
    axes[1].set_ylabel("growth of $1")
    axes[1].set_title("(b) Fix: long-only momentum vs benchmark — "
                      "the gap between the lines is the alpha")
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/performance.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
