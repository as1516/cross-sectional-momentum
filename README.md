# Cross-Sectional Momentum — Build, Failure, Diagnosis, and Fix

A long/short equity strategy on **real S&P 500 data (2010–2024)** that **lost
money** — and the research process of finding out *why*, fixing it for a stated
reason, and separating genuine skill from a rising market.

This project is deliberately not a "look at my great Sharpe ratio" backtest. It's
the realistic version: most strategies don't work, and the value is in diagnosing
them honestly. The final result is a modest, real, out-of-sample alpha — reported
next to the benchmark that most of the raw return actually came from.

---

## The story, in four steps

1. **Built** a dollar-neutral long/short strategy: each month rank ~80 large-cap
   stocks by a blend of 12-1 momentum and short-term reversal, go long the top
   20% and short the bottom 20%, risk-balanced by inverse volatility.
2. **It failed.** Full-sample Sharpe **−0.32**; $1 bled to ~$0.54 over 15 years.
3. **Diagnosed why.** Decomposing the book:
   - momentum **long** leg (buy winners): Sharpe **+1.39** — healthy
   - momentum **short** leg (short losers): Sharpe **−1.10** — the culprit
   - equal-weight market: Sharpe **+1.37**

   In a 15-year large-cap bull market, even the relative laggards rose, so
   shorting them was a structural loser. The dollar-neutral design let the short
   leg's losses cancel the long leg's gains. (A separate horizon check also
   showed the reversal signal only works at ~1 week, not the ~1 month default.)
4. **Fixed it for that reason.** Based on the diagnosis, tested a **long-only**
   winners portfolio. Sharpe **+1.31**, out-of-sample **+1.16** vs in-sample
   **+1.45** — it held up, so it isn't overfit.

## The honest headline (this is the important part)

The long-only Sharpe of +1.31 looks great — but an **equal-weight portfolio of
the same stocks** scores **+1.37**. So *most of that Sharpe is just the market
going up*, not skill. Measured against that benchmark:

- **Beta to benchmark: 0.89**, correlation 0.86 — the strategy is mostly market.
- **Alpha: ~+2.1%/year** — the momentum tilt's genuine contribution.

**The alpha, not the headline Sharpe, is the real result.** Separating the two is
the single most important lesson here: any long-only equity strategy in a bull
market looks brilliant on raw Sharpe, and the job is to isolate what you added on
top.

### One caveat I won't hide

The +2.1% full-sample alpha is uneven: it's ~0% in the first (in-sample) half and
~+2.7% in the second (out-of-sample) half. So the tilt's edge shows up mostly in
the later period. That's worth stating plainly rather than rounding away — with
~10 years of monthly data and a Sharpe standard error near 0.3, none of these
numbers are precise, and the honest read is "a small, positive, but noisy alpha."

## Run it

```bash
pip install -r requirements.txt
python cross_sectional_momentum.py
```

First run downloads ~15 years of prices (a few minutes) and caches to
`prices.csv`; later runs are fast. The script prints the full workflow —
(a) the failed long/short, (b) the diagnosis, (c) the benchmarked fix — and saves
`scoreboard.csv`, `monthly_returns.csv`, and `figures/performance.png`.

## How each part maps to the quant curriculum

| Part of the code | Course topics |
|---|---|
| `download_prices`, caching, missing-value handling | Data → PriceData; More Pandas → ReadingWritingFiles, MissingValues, MultiIndexing, TimeSeries |
| `momentum_signal`, `reversal_signal`, `zscore_cross_section` | Momentum → Time Horizon; Reversal → Overview; Pandas → DataAnalysis |
| `weights_from_scores` (inverse-vol) | Weighting Pt2 → VolWeights (the stretch concept) |
| `run_backtest` timing & PnL; long/short vs long-only | Backtesting → BacktestIntro, BasicPortfolioMath, TS Backtesting, StrategyTypes |
| turnover & `COST_BPS` | Execution → Tcosts, Turnover |
| `sharpe_ratio`, `sharpe_standard_error`, `drawdown_series` | Performance Evaluation → Sharpe Ratios; Pt2 → Drawdowns |
| `benchmark_return`, `alpha_beta` | Performance Evaluation → BetaIntro, AlphaIntro, ComputingAlpha |
| `diagnose`, train/test split | Quant Research Pitfalls → Overfitting, Overindexing on Modeling |

## What I deliberately left out (my roadmap)

Focused first version, not a kitchen sink. Next steps I understand but chose not
to include yet:
- **Risk-factor / sector-beta neutralization** of the signal (the long-only book
  is ~0.9 beta to the market — a neutralized version would isolate the alpha
  directly instead of by regression after the fact).
- **A larger universe** (hundreds of names) — momentum's cross-sectional edge is
  stronger with more breadth than 80 correlated large-caps provide.
- **Portfolio optimization** and a **trade-size-aware cost model**.

## Honest limitations

- **Survivorship bias:** fixed ticker list of large names that mostly survived.
- **Small, correlated universe:** ~80 mega-caps, so the strategy is dominated by
  market beta.
- **Noisy estimates:** ~10 years of monthly data → Sharpe standard error ≈ 0.3,
  which is exactly why every Sharpe here is reported with that error.

## Interview-defense cheat sheet

- *Your strategy lost money — why show it?* Because the diagnosis is the result:
  the short leg (shorting laggards in a bull market) was structurally broken, and
  finding that is real research.
- *Your long-only Sharpe is 1.3 — is that skill?* Mostly no. Beta to an
  equal-weight benchmark is 0.89; the actual skill is ~2%/year of alpha.
- *Is the alpha robust?* Partially — it's ~0% in-sample and ~+2.7% out-of-sample,
  and with this much data the standard error is large. Small and positive, but
  noisy.
- *Why inverse-vol weighting?* So each position contributes similar risk;
  equal dollars would let the most volatile names dominate.
- *Biggest weakness / next step?* No sector/beta neutralization yet, and the
  universe is too small and market-correlated — both on the roadmap.
