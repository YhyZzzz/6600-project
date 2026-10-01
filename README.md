# SPY Intraday Forecasting with GARCH and Deep Learning

A DSAN 6600 project investigating whether volatility information from GARCH-family models improves deep learning forecasts of SPY using **15-minute OHLCV data over 2016–2026**.

**Status:** Research design and data acquisition scaffold. The repository currently contains an Alpaca download script; preprocessing, models, evaluation, and results are planned. No predictive or trading performance is claimed.

## Motivation and connection to the original project

This project extends my graduation thesis, *Stock Price Prediction Based on GARCH and Machine Learning Models*. That study used CSI 500 five-minute observations from April 2021 to April 2023 and compared an LSTM with hybrid models incorporating rolling GARCH, EGARCH, and TGARCH parameter estimates.

The new project retains the combination of econometric volatility modeling and neural sequence learning while changing the asset, interval, and evaluation design:

| Dimension | Graduation project | New project |
| --- | --- | --- |
| Asset | CSI 500 index | SPY, an ETF tracking the S&P 500 |
| Sampling | 5-minute bars | 15-minute bars |
| Period | 2021–2023 | 2016–2026, subject to observed coverage |
| Prediction | Closing prices; additional multi-step experiments | Future log returns, with reconstructed closing prices |
| Hybrid inputs | Rolling GARCH-family parameter estimates | Causal volatility forecasts, residuals, and parameter estimates |
| Evaluation | Chronological train/test comparison | Validation, a held-out test, and walk-forward evaluation |
| Main extension | Multiple volatility models combined with LSTM | Leakage controls, market-regime analysis, and feature ablations |

The thesis motivates the hypothesis; its findings are not evidence that the same advantage will hold for SPY.

## Research questions

1. Do GARCH-derived features improve next-bar return forecasts compared with an OHLCV-only LSTM and simpler baselines?
2. Do asymmetric volatility models add value beyond a standard GARCH model?
3. Are improvements consistent across years, volatility regimes, and forecast horizons?
4. Does an attention-enhanced LSTM improve performance enough to justify its additional complexity?

The primary comparison is **LSTM versus GARCH-feature LSTM on next-bar log-return RMSE**. Multi-horizon forecasts and attention are extensions after the core comparison is reproducible.

## Data specification

| Item | Planned specification |
| --- | --- |
| Instrument | SPY |
| Requested date range | January 1, 2016–December 31, 2026 |
| Frequency | 15 minutes |
| Model inputs | Open, high, low, close, volume, and features derived from them |
| Source | Alpaca historical stock bars through the existing downloader |
| Feed | SIP for the main experiment; record the feed in every dataset manifest |
| Session | Regular trading hours, 09:30–16:00 America/New_York, respecting holidays and early closes |
| Storage | Parquet, with UTC and America/New_York timestamps |
| Adjustment | Explicitly recorded; the existing script defaults to `all` |

**2026 is a partial year until it finishes.** A filename ending in `2026` does not establish full-year coverage. Freeze the actual last completed bar, retrieval time, feed, adjustment mode, row count, and file checksum for each experiment. Verify coverage back to 2016 before training; access and available history must be checked with the chosen account.

The downloader saves these columns:

| Column | Meaning |
| --- | --- |
| `ts_utc` | Provider bar timestamp in UTC |
| `ts_et` | The same timestamp in America/New_York |
| `open`, `high`, `low`, `close` | Bar prices |
| `volume` | Bar trading volume |
| `trade_count`, `vwap` | Additional provider fields retained for auditing; excluded from the core OHLCV experiment |

Alpaca labels bars by the start of the interval. A bar starting at 09:30 becomes an input only after it closes at 09:45. See the [Alpaca bar timestamp explanation](https://alpaca.markets/learn/stock-minute-bars).

SIP and IEX are different feeds and must not be silently mixed. Consult the [Alpaca market data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq) and [historical bars API](https://docs.alpaca.markets/us/reference/stockbars) for feed, access, and adjustment behavior.

### Planned data-quality checks

- Sort and deduplicate timestamps, then validate against an exchange calendar. A full regular session has 26 fifteen-minute bars; early-close sessions have fewer.
- Retain only completed regular-session bars for the main experiment. The current downloader does **not** apply this filter.
- Check positive prices, nonnegative volume, and `low <= open, close <= high`.
- Report missing bars, zero-volume bars, duplicate conflicts, and anomalous prices by year. Do not fabricate bars or forward-fill prices across missing intervals.
- Keep overnight gaps separate from within-session 15-minute returns. Exclude targets crossing a session boundary or a missing bar.
- Apply one consistent adjustment policy to all OHLC fields. Audit corporate actions and compare raw versus adjusted returns; record the limitations of retrospectively adjusted data.
- Estimate intraday seasonal patterns using training history only. Preserve raw downloads separately from cleaned datasets.

## Forecasting tasks

At the close of bar `t`, use only information available through that close. The primary target is the next within-session close-to-close log return:

```text
r[t]        = log(C[t] / C[t-1])
y[t, h]     = log(C[t+h] / C[t])
C_hat[t+h]  = C[t] * exp(y_hat[t, h])
```

Use `h = 1` (15 minutes) as the primary horizon. Add `h = 2` and `h = 4` (30 and 60 minutes) as secondary experiments using direct horizon-specific predictions. Retain a sample only when every required target bar is consecutive and within the same trading session. Comparisons across horizons should also report results on a common eligible sample.

Return prediction avoids using price-level fit alone as evidence of predictability. Reconstructed price errors remain useful for comparison with the original thesis. A GARCH variance forecast is a feature describing uncertainty, not a directional return forecast.

## Proposed methodology

### 1. Exploratory analysis and features

Inspect return distributions, autocorrelation, squared-return autocorrelation, ARCH effects, and time-of-day patterns. Construct lagged returns, high–low ranges, close–open changes, rolling return statistics, and log-volume changes. Use trailing windows only and fit scaling or seasonal normalization on the training period.

Start with sequence lengths of 26, 52, and 130 observed regular-session bars. Historical input sequences may span earlier sessions, but include session-boundary indicators and do not treat an overnight interval as a 15-minute return. Choose the lookback on validation data.

### 2. Causal volatility features

Fit GARCH(1,1), EGARCH(1,1), and an explicitly specified asymmetric threshold model. Use GJR-GARCH(1,1) for the initial threshold experiment; document its variance equation because the label TGARCH can refer to different parameterizations.

Start with Student-t innovations and a trailing 100-session estimation window. Refit before each session using only completed earlier sessions; update the variance state as new bars close. Log convergence failures and use a predefined past-only fallback rather than dropping difficult periods selectively.

Candidate hybrid inputs include one-step conditional variance forecasts, standardized residuals, and rolling fitted coefficients. Generate every historical feature at its own forecast origin: fitting once on the entire training period and backfilling features would leak later information into earlier samples. Keep scaling units consistent if returns are multiplied by 100 for estimation.

### 3. Model comparisons

| Model | Purpose |
| --- | --- |
| Zero-return / last-close persistence | Essential forecast baseline |
| Ridge regression on lagged features | Linear baseline |
| Gradient-boosted trees on the same causal features | Non-neural nonlinear baseline |
| OHLCV-feature LSTM | Main neural baseline |
| GARCH-LSTM | Add standard GARCH features |
| Multi-GARCH-LSTM | Add GARCH, EGARCH, and GJR-GARCH features |
| Attention-enhanced hybrid LSTM | Optional architectural extension |

Keep data splits, target timestamps, and tuning budgets comparable. Start with a small one- or two-layer LSTM, validation-based early stopping, and at least three random seeds. Select architecture, optimizer settings, and regularization on validation data only.

Ablate volatility forecasts versus coefficient features, individual GARCH families, and all GARCH features removed. This distinguishes the value of volatility information from simply increasing the input dimension.

## Evaluation protocol

### Chronological split

| Partition | Period | Use |
| --- | --- | --- |
| Training | 2016–2022 | Fit features and models |
| Validation | 2023–2024 | Select hyperparameters and experiment design |
| Final test | 2025–last completed bar available in 2026 | Locked out-of-sample evaluation |

Allow a warm-up period for rolling features. Assign samples by target timestamps and purge any training or validation sample whose target horizon crosses a partition boundary. Earlier observations may provide input context for later forecasts; later observations must never influence earlier fitted transforms or targets.

After selecting the design, refit on data through 2024 and evaluate with a predeclared quarterly expanding-window retraining schedule. Within the test period, a refit may use only labels already observed at that date; do not retune from test performance. Also report a frozen-model comparison to separate adaptation from initial model quality. Daily GARCH refits follow the same past-only rule.

### Metrics and reporting

- **Primary:** RMSE of next-bar log returns; also report MAE and out-of-sample R-squared relative to zero-return predictions.
- **Secondary:** Directional accuracy, reconstructed-price MAE/RMSE, and errors by horizon, year, and time of day. Define treatment of zero returns before scoring direction.
- **Robustness:** Report variation across seeds and confidence intervals for paired loss differences using a trading-day block bootstrap.
- **Regime analysis:** Compare low/high volatility groups defined by trailing volatility and thresholds fitted on training data, rather than future realized outcomes.
- **Optional volatility task:** Compare next-hour variance forecasts against the sum of the next four squared 15-minute returns using QLIKE and variance MSE; keep this separate from return forecasting.

An optional trading simulation must execute no earlier than the next bar open and specify holding periods, turnover, fees, and slippage. Forecast accuracy alone does not establish profitability. OHLCV bars cannot directly establish executable bid–ask spreads or market impact.

## Getting started: existing downloader

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set your Alpaca credentials locally in `.env`:

```dotenv
APCA_API_KEY_ID=your_api_key
APCA_API_SECRET_KEY=your_api_secret
```

Download the requested range:

```bash
python scripts/fetch_spy.py \
  --symbol SPY \
  --start 2016-01-01 \
  --end 2026-12-31 \
  --minutes 15 \
  --feed sip \
  --adjustment all \
  --out data/spy_15min_2016_2026.parquet
```

The script interprets dates in America/New_York, advances the requested end date by one day, and caps the query end at the current time minus 16 minutes. It requests data in yearly chunks, deduplicates timestamps, and writes Parquet. It does not guarantee complete coverage, remove extended-hours bars, validate exchange sessions, or provide a modeling pipeline. Inspect the printed timestamp range and audit the saved data before use. If access fails, check account entitlements rather than silently switching feeds.

The current requirements cover acquisition only: `alpaca-py`, `pandas`, `pyarrow`, and `python-dotenv`. Modeling dependencies and pinned versions will be added with the implementation. No training or evaluation command is available yet.

## Repository and planned deliverables

Current tracked project files:

```text
6600-project/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── scripts/
    └── fetch_spy.py
```

Planned additions include preprocessing and feature modules, model configurations, training/evaluation scripts, exploratory notebooks, and reports. Preserve dataset manifests, split definitions, fitted preprocessing objects, random seeds, dependency versions, predictions, and checkpoints so each reported result can be reproduced.

### Milestones

1. **Data audit:** Acquire and freeze available 2016–2026 bars; document coverage and session cleaning.
2. **Baselines:** Build causal targets/features and evaluate persistence, ridge, and boosted trees.
3. **Neural baseline:** Implement and tune the OHLCV-feature LSTM.
4. **Hybrid models:** Generate rolling GARCH features and run controlled ablations.
5. **Final evaluation:** Freeze the design, run the held-out walk-forward experiment, and analyze uncertainty and regimes.
6. **Delivery:** Publish reproducible code, dataset metadata, comparison tables, diagnostic plots, and a final project report. Add attention or a trading simulation only after the core experiments are complete.

Success means a reproducible answer to whether volatility features improve out-of-sample forecasts, including a negative result. Keep credentials and downloaded market data out of Git; the existing `.gitignore` excludes `.env` and `data/`.
