# Cross-Sectional Momentum Strategy

A long/short equity momentum strategy backtested on S&P 500 constituents from
2010–2024. The initial long/short construction was unprofitable; this repo walks
through the backtest, a decomposition that identifies the cause, and a revised
long-only version evaluated against an equal-weight benchmark.

## Summary of results

| Strategy | Sharpe | Ann. return | Max DD | Beta to benchmark | Alpha |
|---|---|---|---|---|---|
| Long/short (initial) | −0.32 | −6.1% | −49% | — | — |
| Long-only (revised) | 1.31 | 23.4% | −19% | 0.89 | +2.1%/yr |
| Equal-weight benchmark | 1.37 | 24.0% | −21% | — | — |

The long-only Sharpe of 1.31 is below the equal-weight benchmark's 1.37, so the
strategy's raw performance is largely market exposure. Regressed against the
benchmark, the momentum tilt contributes roughly 2% of annualized alpha at a beta
of 0.89. This alpha is uneven across the sample: near zero in the first 60% of
the period and about +2.7%/yr in the held-out remainder. Given ~10 years of
monthly observations, the Sharpe standard error is approximately 0.3, so these
estimates carry meaningful uncertainty.

## Method

Each month, ~80 large-cap stocks are ranked by a combined signal:
- **12-1 momentum:** trailing 12-month return, skipping the most recent month.
- **Short-term reversal:** negative of the trailing 1-month return.

Both signals are cross-sectionally z-scored and averaged. Positions are sized by
inverse volatility so each holding contributes comparable risk rather than
comparable notional. The book rebalances monthly with a 10 bps cost charged on
turnover. Signals computed at month *t* are traded in month *t+1* (no
look-ahead), and the sample is split 60/40 into in-sample and held-out periods.

## Why the long/short version failed

Decomposing the momentum book by leg:

| Leg | Sharpe |
|---|---|
| Long (buy winners) | +1.39 |
| Short (short losers) | −1.10 |
| Equal-weight market | +1.37 |

The long leg tracked the market; the short leg lost ~25%/yr. Over a 15-year
large-cap bull market, the bottom-ranked names still appreciated, so shorting
them was a persistent drag, and the dollar-neutral construction let those losses
offset the long leg's gains. A separate horizon scan showed the reversal signal
was only additive at a ~1-week lookback, not the ~1-month default used in the
combined signal.

The revised strategy drops the short leg and holds the top-quantile winners
long-only.

## Running it

```bash
pip install -r requirements.txt
python cross_sectional_momentum.py
```

The first run downloads daily prices via `yfinance` and caches them to
`prices.csv`; subsequent runs load from cache. Output: console summary,
`scoreboard.csv`, `monthly_returns.csv`, and `figures/performance.png`.

## Code structure

| Function | Purpose |
|---|---|
| `download_prices` | Fetch/cache the adjusted-price panel; handle missing data |
| `momentum_signal`, `reversal_signal`, `zscore_cross_section` | Construct and standardize signals |
| `weights_from_scores` | Inverse-volatility position sizing (long/short or long-only) |
| `run_backtest` | Walk-forward monthly backtest with turnover and costs |
| `benchmark_return`, `alpha_beta` | Equal-weight benchmark and alpha/beta decomposition |
| `sharpe_ratio`, `sharpe_standard_error`, `drawdown_series` | Performance metrics |
| `diagnose` | Leg decomposition and reversal-horizon scan |

## Limitations

- **Survivorship bias:** the ticker list is a fixed set of large-cap names that
  largely survived the sample; historical index membership is not reconstructed.
- **Universe size:** ~80 correlated mega-caps, so the book is dominated by market
  beta and understates the breadth cross-sectional momentum benefits from.
- **Estimation noise:** with monthly data over ~10 years, the Sharpe standard
  error is ~0.3; point estimates should be read with that in mind.

## Possible extensions

- Sector- and beta-neutralize the signal so alpha is isolated directly rather
  than via post-hoc regression.
- Expand to a larger universe (several hundred names).
- Add mean-variance portfolio construction and a trade-size-aware cost model.
