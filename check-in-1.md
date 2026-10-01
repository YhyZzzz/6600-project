# Project Check-In 1 — Problem Framing + Data

## 1. Problem framing and semester scope

**Task.** At the close of a completed SPY 15-minute bar, predict the next within-session close-to-close log return using information available at that moment:

`y[t] = log(close[t+1] / close[t])`.

Both bars must belong to the same regular trading session and be exactly 15 minutes apart. A bar timestamp marks its start; the input becomes available only after its interval completes. The core question is whether adding causal GARCH(1,1) volatility features improves an LSTM relative to the same LSTM using market features alone. A variance forecast represents uncertainty, not the direction of a price move.

**Motivation.** This project connects econometric volatility modeling with neural sequence learning. It extends the earlier GARCH and machine-learning stock-prediction project described in the supplied README to SPY, a longer history, and a stricter chronological evaluation. We want to learn whether modeling changing volatility helps a neural forecast rather than relying on price-level fit. Earlier thesis results are not evidence of an advantage on this dataset.

**Scope.** One instrument, one primary horizon, and two small neural models: OHLCV-feature LSTM and GARCH-feature LSTM. Compare them with zero-return persistence and ridge regression. EGARCH/GJR-GARCH, attention, longer horizons, and trading simulations are optional extensions, not requirements for the core semester deliverable.

**Success criteria.** Deliver a reproducible, leakage-controlled comparison. Evidence supporting the hypothesis would be lower held-out log-return RMSE for the hybrid than the market-only LSTM and zero-return baseline, with consistent results across three seeds and a paired trading-day block-bootstrap confidence interval for the hybrid-minus-LSTM loss difference. Report effect size and uncertainty, including a result showing no improvement. No arbitrary accuracy target or profitability claim is used.

## 2. Accessible dataset and documentation

The supplied Parquet snapshot is locally readable and its structure matches the supplied Alpaca acquisition script. Actual data coverage, rather than the filename, defines this check-in.

| Item | Observed value |
| --- | --- |
| Instrument / interval | SPY / 15-minute bars, per supplied acquisition documentation |
| Raw rows / columns | 169,241 / 9 |
| File size | 7,840,700 bytes (about 7.48 MiB) |
| First timestamp, New York | 2016-01-04 04:00 |
| Last timestamp, New York | 2026-09-30 19:45 |
| Core-session rows | 69,973 |
| Observed core sessions | 2,701 |
| 2026 coverage | Partial year, through September 30 |
| SHA-256 | `bda19e9f8a784ee50d20fe7dc50826d5fce0fddea97b205257a569312aa42b67` |

Fields are `ts_utc`, `ts_et`, `open`, `high`, `low`, `close`, `volume`, `trade_count`, and `vwap`. Price fields, volume, and trade count are stored as float64; timestamps are timezone aware. UTC and New York timestamps describe the same instants. The primary model will use derived OHLCV features; trade count and VWAP are retained for auditing.

**Provenance caveat.** The script defaults to SIP and adjustment `all`, but the actual invocation, acquisition time, account entitlement, and provider response metadata are absent. These settings cannot be proven from the table alone. Confirm them from the original download log or reacquire with explicit arguments and record a manifest before modeling. Do not silently substitute IEX for SIP.

**Access and usage.** [DATA_ACCESS.md](DATA_ACCESS.md) documents the local snapshot path, downloader command, API credentials, reference links, and instructor access procedure. Alpaca data is provider-controlled market data; no open-data redistribution license was supplied. Keep raw data and secrets out of public Git. Each reader can acquire data with their own authorized account; provide the exact snapshot privately to the instructional team only if the applicable agreement permits it. Confirm access before submission. The notebook and aggregated audit results are available without credentials.

## 3. Completed audit and EDA

The [executed notebook](notebooks/01_data_audit.ipynb) shows raw head/tail samples, one core-session sample per year, field summaries, calendar coverage, eligible target counts, return summaries, and figures. Generated CSVs and the manifest are in `reports/`.

### Integrity and session coverage

| Check | Actual result |
| --- | --- |
| Null cells / duplicate timestamps | 0 / 0 |
| Nonfinite numeric values / nonpositive prices | 0 / 0 |
| Invalid OHLC ranges or negative volume/trade count | 0 rows |
| Zero-volume bars | 0 |
| Original timestamps sorted / timezone agreement | Yes / all rows agree |
| Bars outside calendar core sessions | 99,268 (58.65% of raw rows) |
| Expected / observed core-session bars | 69,974 / 69,973 |
| Missing core-session bars | 1, listed in `reports/missing_bars.csv` |
| Expected / observed sessions | 2,701 / 2,701 |
| Early-close sessions | 21 |

The audit uses the NYSE calendar as a US-equity regular-session proxy, including holidays and early closes. Only expected 15-minute bar starts between session open and close are retained. No absent bar is fabricated or filled. This is a calendar consistency check; it does not independently authenticate the provider, exchange-wide completeness, or trading halts.

### Key distributions: training period only (2016–2022)

There are 43,893 eligible training-period next-bar returns before sequence and GARCH warm-up. All values below are computed from the supplied snapshot.

| Statistic | Log return, expressed as percent |
| --- | ---: |
| Mean | 0.000546% |
| Standard deviation | 0.168708% |
| 1st / 99th percentile | −0.487950% / 0.458548% |
| Minimum / maximum | −2.786200% / 3.806359% |
| Excess kurtosis (dimensionless) | 31.0271 |

Negative returns: 20,585 (46.90%); exact zeros: 873 (1.99%); positive returns: 22,435 (51.11%). The task is regression; these counts contextualize secondary directional accuracy and are not training labels for a separate classifier.

![EDA dashboard](reports/figures/eda_dashboard.png)

The price level changes substantially over the training years, so price-level fit alone would be misleading. The return histogram has heavy tails. Session variability changes over time, and median volume is higher near the open and close than midday. These patterns motivate returns, trailing volatility, and time-of-day features.

![Return dependence](reports/figures/train_acf.png)

At lag one, training return correlation is approximately 0.00865, while squared-return correlation is approximately 0.21455. Pairs are aligned by timestamp within sessions, excluding overnight gaps. This motivates testing variance features; it does not prove directional predictability or establish that GARCH is the best model. Intraday seasonality also contributes to dependence and must be considered.

### Planned data features: log returns and GARCH-family statistics

We plan to augment market features with statistics derived from GARCH-family models to test whether explicit volatility information improves neural forecasting accuracy. Because these models describe the conditional variance of return innovations, their outputs characterize the magnitude and persistence of price fluctuations rather than directly predicting whether prices will rise or fall.

Raw financial price levels are often nonstationary. We therefore first apply a first difference to log closing prices:

$$r_t = \log P_t - \log P_{t-1} = \log(P_t/P_{t-1}).$$

Here, $r_t$ is an observed input return; the next eligible return $r_{t+1}$ is the prediction target. Compute $r_t$ only when the two bars are consecutive within the same session. Log differencing puts changes on a relative scale and can improve stationarity, but does not guarantee it. The present EDA motivates this transformation; it does not constitute a formal stationarity test. Check training-period stationarity, residual dependence, and conditional heteroskedasticity before selecting the mean and volatility specifications. ADF/KPSS and ARCH-effect diagnostics are planned, not completed results.

The reason for this transformation is not that every time series must be white noise. GARCH explicitly allows predictable variation in conditional variance. With a suitable conditional mean model, write:

$$r_t = \mu_t + \epsilon_t, \qquad \epsilon_t = \sigma_t z_t,$$

where $z_t$ is the standardized innovation, typically assumed independent with mean zero and unit variance. A starting constant-mean GARCH(1,1) specification is:

$$\sigma_{t+1\mid t}^{2} = \omega + \alpha\epsilon_t^2 + \beta\sigma_t^2.$$

If residual mean dependence remains, evaluate a parsimonious autoregressive mean specification on validation. The variance forecast uses information available through $t$; it does not use the realized next return. Even when return autocorrelation is weak, squared returns can remain dependent, which is precisely why a conditional-variance model can be useful.

| Candidate extra feature at forecast origin $t$ | Meaning | Priority |
| --- | --- | --- |
| $\hat\sigma^2_{t+1\mid t}$ | One-step conditional variance forecast | Core GARCH-LSTM feature |
| $\hat z_t=\hat\epsilon_t/\hat\sigma_t$ | Current shock relative to estimated volatility | Core GARCH-LSTM feature |
| $\hat\sigma_{t+1\mid t}$ | Conditional volatility in return units | Alternative to variance; select on validation |
| Rolling fitted coefficients | Volatility persistence and shock-response characteristics | Optional feature ablation |
| EGARCH / GJR-GARCH outputs | Potential asymmetric responses to positive and negative shocks | Optional family comparison |

Do not interpret coefficients from different GARCH families as interchangeable. For example, $\alpha+\beta$ is a standard GARCH(1,1) persistence measure under the usual specification, not a universal formula for EGARCH or GJR-GARCH. Variance and its square root contain the same ordering information; adding both does not inherently add independent information.

All parameters and generated features must use only the past at their own forecast origins. Fit feature scaling on training observations only. These features are planned for Check-In 2 and have not yet been generated in this check-in. Improved accuracy is a hypothesis to test against the market-only LSTM, not an established outcome. See the [arch volatility-modeling documentation](https://arch.readthedocs.io/en/stable/univariate/univariate_volatility_modeling.html) for mean, volatility, and innovation specifications.

### Artifacts, bias, and failure modes

- Extended-hours liquidity differs from regular sessions; the 99,268 non-core bars are excluded from the main experiment.
- Overnight and missing-bar transitions must not become 15-minute targets. The notebook implements exact time and session checks.
- Large returns may be genuine stress events or data errors. Inspect flagged observations against source data before excluding them; do not remove inconvenient extremes automatically.
- Retrospective corporate-action adjustment can differ from prices known at a historical forecast origin. Confirm the adjustment setting and assess raw versus adjusted sensitivity later.
- Only SPY is studied; results cannot establish generalization to individual stocks or other assets. Changing market conditions may limit stability across time.
- Random splits, full-history scaling, centered rolling features, and backfilled GARCH estimates would leak future information. Avoid all four.
- Calendar/sample counts are audited across all years, but outcome distributions and model selection are restricted to training/validation. No held-out performance was calculated for this check-in.

## 4. Evaluation plan

| Partition | Period | Eligible one-step targets now |
| --- | --- | ---: |
| Training | 2016–2022 | 43,893 |
| Validation | 2023–2024 | 12,490 |
| Final test | 2025–2026-09-30 | 10,889 |

These are candidate target counts, not final sequence counts. Sequence length, missing-input handling, and causal GARCH warm-up will reduce them. Assign samples by target timestamp; exclude target horizons crossing a split boundary. Historical context from an earlier partition may be used, while all transforms and parameter selection remain past-only. No shuffling across partitions.

**Metrics.** Primary: RMSE of log returns. Secondary: MAE and out-of-sample R² relative to zero-return persistence, `1 - sum((y-yhat)^2) / sum(y^2)`. A zero prediction implies the current close is the predicted next close. Reconstructed-price errors are secondary and will not substitute for return comparison. For directional accuracy, score only nonzero true returns using three-valued sign; an exactly zero prediction is incorrect for a nonzero target. Report excluded zero-target counts and a training-majority direction baseline.

Choose lookback, architecture, features, and stopping epoch using validation only. Freeze the design before evaluating test outcomes. The core evaluation uses models trained on 2016–2022 with validation-based early stopping; fit scalers only on training. A later, optional walk-forward extension can refit on all pre-2025 data and retrain quarterly using only labels already observed, without retuning on test results. Report it separately from the frozen experiment.

Compare models on identical eligible forecast origins and three fixed seeds (42, 123, 2026). Report paired trading-day bootstrap uncertainty, annual errors, and regimes defined by past trailing volatility using training-fitted thresholds. No trading profitability is claimed.

## 5. Initial neural direction for Check-In 2

Start with one LSTM layer, 32 hidden units, a linear one-unit regression head, dropout 0.1, Adam with initial learning rate 0.001, batch size 128, at most 50 epochs, and validation early stopping with patience 5. These are initial settings, not trained results. Try lookbacks of 26 and 52 bars on validation. Use lagged within-session returns, relative high-low range, close-open change, log-volume, time-of-day encoding, and boundary/missingness indicators. Input histories may cross earlier sessions but must explicitly mark boundaries; do not encode overnight returns as intraday returns.

For the hybrid, add one-step conditional variance and standardized residuals from GARCH(1,1) with Student-t innovations. Begin with a trailing 100-session fit, refit before each session using only earlier sessions, and update the variance state as each bar completes. Generate historical features at their own forecast origins. Log convergence failures; use a predeclared trailing-variance fallback from earlier observations, never selectively drop difficult days. GARCH units must match neural input scaling. Give both LSTMs the same tuning budget.

Check-In 2 deliverables: zero-return and ridge baselines; an executed market-only LSTM run; a small causal GARCH-feature experiment; validation metrics and diagnostics. Final work adds matched ablations, seed variation, and held-out evaluation. No baseline or neural training was required or completed at Check-In 1.

## 6. Async progress video

Use [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) to record approximately five minutes showing this report, the actual executed notebook, raw samples, audit tables, and both figures. **Video recording/upload is still pending.** Add the link to README or upload to Canvas. Do not present this script as a completed video.

## References and AI assistance

- [Alpaca historical bars API](https://docs.alpaca.markets/us/reference/stockbars): interval, feed, and adjustment parameters.
- [Alpaca market-data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq): SIP/IEX distinction and historical access restrictions.
- [Alpaca disclosure and agreement library](https://alpaca.markets/disclosures): review the applicable market-data agreement before sharing data.
- Supplied `fetch_spy.py`, supplied README, and the supplied frozen Parquet snapshot.

ChatGPT/Codex assisted with code, execution, audit interpretation, documentation, and the video script. The author should verify the files, confirm provenance/access, and disclose assistance as required by the course.
