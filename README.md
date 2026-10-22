# Bayesian Stock Picker

Price-only features (momentum, trend, mean reversion, volatility, correlation)
feed a Bayesian neural network that predicts the full distribution of each
stock's return over 1M, 3M, 6M, 1Y, 2Y and 3Y. You rank stocks by uncertainty
(SD, probability of loss, CVaR, epistemic SD) instead of by a single point
forecast, and you can filter by region and sector.

## Run

```bash
./run.sh                      # creates .venv, installs deps, opens the app in your browser
# or manually
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

In the sidebar, choose **Yahoo Finance (real)**, open **Run pipeline** and press
**Download · features · train**. On a laptop the run takes about 1–3 minutes,
most of it the download. **Synthetic (offline demo)** works without internet;
it is a simulated market with planted momentum, volatility regimes and mean
reversion, so you can check that the model finds structure that is really there.

Headless / scripted:

```bash
python -m stockpicker.pipeline --source yahoo --start 2005-01-01 --epochs 40
```

## Layout

| File | What it does |
|---|---|
| `universe.csv` | 153 tickers with sector / region / country. Add rows to extend it (Yahoo symbols). |
| `stockpicker/data.py` | yfinance download, business-day alignment, cache in `data/`, synthetic market |
| `stockpicker/features.py` | 27 features in 6 groups, forward log-return targets, rank-IC analysis |
| `stockpicker/model.py` | Bayes-by-Backprop BNN, purged walk-forward split, evaluation, selection stats |
| `stockpicker/pipeline.py` | prices → features → eval model → final model → forecasts |
| `app.py` | Streamlit UI with 4 tabs: Data, Features, Model diagnostics, Forecast & selection |

## Model

* **Inputs:** 27 features, robust-scaled (median / IQR, clipped at ±6).
* **Network:** 2 hidden layers (64, 32, SiLU). Every weight has a Gaussian
  posterior N(μ, σ²) with a N(0, 1) prior, trained by variational inference
  (Bayes-by-Backprop, local reparameterisation, KL warm-up).
* **Output:** for each of the 6 horizons, a mean and a log-variance of the
  forward log return (heteroscedastic), so the network can say that some stocks
  are noisier than others.
* **Predictive:** 200 weight samples, each giving N(μ_s, σ_s²). Their mixture is
  the forecast.
  * Epistemic variance = Var_s(μ_s): the model is unsure. This shrinks with more data.
  * Aleatoric variance = E_s(σ_s²): the noise the model expects to remain.
* **KL weighting:** weekly samples overlap heavily, so the KL term is divided by
  an effective sample size of n/4 rather than n. The **KL scale** slider widens or
  narrows the weight posterior.
* **Evaluation:** a purged walk-forward split for each horizon. The test set is
  the last 20 % of dates with a realised target. Training stops one horizon
  before the test set starts, so no future return leaks in. Results are compared
  with a naive baseline: average drift plus trailing 1-year volatility × √h.

## Synthetic-market check (40 epochs, out-of-sample)

| Horizon | NLL BNN | NLL baseline | 90 % interval hit-rate | Rank IC | Realised SD, decile 1 / decile 10 of predicted SD |
|---|---|---|---|---|---|
| 1M | −0.877 | −0.842 | 89.7 % | +0.016 | 8.3 % / 13.7 % |
| 3M | −0.302 | −0.271 | 89.3 % | +0.015 | 14.1 % / 23.0 % |
| 6M | 0.060 | 0.102 | 88.5 % | +0.037 | 20.7 % / 34.0 % |
| 1Y | 0.404 | 0.462 | 89.9 % | +0.061 | 29.0 % / 50.0 % |
| 2Y | 0.747 | 0.848 | 91.1 % | −0.004 | 36.8 % / 72.5 % |
| 3Y | 0.991 | 1.023 | 90.5 % | −0.027 | 44.2 % / 81.2 % |

Lower NLL is better. The BNN beats the baseline at all 6 horizons. The decile
with the lowest predicted SD realised 0.51–0.61× the dispersion of the decile
with the highest. Real-market numbers will be weaker; check the
**Model diagnostics** tab after training on Yahoo data.

## Caveats

* Returns are in local currency, with no FX conversion.
* The universe is today's large caps, so the history has survivorship bias:
  companies that failed are missing. This flatters long-horizon returns.
* 2Y and 3Y targets overlap heavily, so they rest on few independent observations.
* This is a research tool, not investment advice.
