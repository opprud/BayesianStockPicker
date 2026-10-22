"""Price data: ticker universe, Yahoo Finance download with on-disk cache, and a
synthetic market generator (offline demo / testing).

Prices are adjusted close in each stock's local currency, aligned on a common
business-day calendar (forward-filled across local holidays, max 5 days).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UNIVERSE_CSV = ROOT / "universe.csv"


def load_universe() -> pd.DataFrame:
    return pd.read_csv(UNIVERSE_CSV, dtype=str).set_index("ticker")


def cache_path(source: str) -> Path:
    return DATA_DIR / f"prices_{source}.pkl"


def _align(px: pd.DataFrame, min_years: float = 3.0) -> pd.DataFrame:
    px = px.sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[~px.index.duplicated(keep="last")]
    bdays = pd.bdate_range(px.index.min(), px.index.max())
    px = px.reindex(bdays).ffill(limit=5)
    # drop non-positive / absurd values and tickers with too little history
    px = px.where(px > 0)
    keep = px.notna().sum() >= int(min_years * 252)
    return px.loc[:, keep]


def download_yahoo(tickers: list[str], start: str = "2005-01-01") -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                      threads=True, group_by="column")
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"]
    else:  # single ticker
        px = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return _align(px.dropna(how="all"))


def synthetic_prices(universe: pd.DataFrame, start: str = "2005-01-01",
                     end: str | None = None, seed: int = 7) -> pd.DataFrame:
    """Factor-model market with regime-switching volatility, slowly drifting
    expected returns (=> momentum/trend signal), short-term mean reversion and
    jumps.  Used for offline demos and to test that the pipeline can recover
    planted structure."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end or pd.Timestamp.today().normalize())
    T, tick = len(dates), list(universe.index)
    N = len(tick)
    regions = universe["region"].values
    sectors = universe["sector"].values

    # global volatility regime: 2-state Markov chain (calm / stormy)
    p_stay = np.array([0.995, 0.98])
    state = np.zeros(T, dtype=int)
    for t in range(1, T):
        s = state[t - 1]
        state[t] = s if rng.random() < p_stay[s] else 1 - s
    regime_mult = np.where(state == 0, 0.8, 1.9)

    def factor(n_f, vol_ann):
        return rng.standard_normal((T, n_f)) * vol_ann / np.sqrt(252) * regime_mult[:, None]

    g = factor(1, 0.14)[:, 0]
    ureg, usec = np.unique(regions), np.unique(sectors)
    fr, fs = factor(len(ureg), 0.08), factor(len(usec), 0.10)
    ri = np.searchsorted(ureg, regions)
    si = np.searchsorted(usec, sectors)

    beta = rng.uniform(0.6, 1.4, N)
    idio_vol = np.exp(rng.normal(np.log(0.22), 0.35, N)) / np.sqrt(252)
    # slowly varying expected return per stock: AR(1), half-life ~ 1 year
    phi = 0.5 ** (1 / 252)
    mu_sd = 0.10 / 252  # 10%/yr cross-sectional dispersion of drift
    mu = np.zeros((T, N))
    mu[0] = rng.normal(0, mu_sd, N)
    eps_mu = rng.standard_normal((T, N)) * mu_sd * np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        mu[t] = phi * mu[t - 1] + eps_mu[t]
    mu += 0.07 / 252  # equity premium

    idio = rng.standard_normal((T, N)) * idio_vol * regime_mult[:, None] ** 0.7
    # short-term mean reversion: negative autocorrelation in idiosyncratic part
    idio[1:] -= 0.06 * idio[:-1]
    jumps = (rng.random((T, N)) < 0.002) * rng.normal(-0.02, 0.08, (T, N))

    r = (mu + beta * g[:, None] + fr[:, ri] + fs[:, si] + idio + jumps)
    r -= 0.5 * (idio_vol ** 2)  # rough Ito correction on idio part
    logp = np.cumsum(r, axis=0) + np.log(rng.uniform(20, 300, N))
    px = pd.DataFrame(np.exp(logp), index=dates, columns=tick)

    # staggered listing dates for realism
    for j, t0 in enumerate(rng.integers(0, int(T * 0.35), N)):
        if rng.random() < 0.3:
            px.iloc[:t0, j] = np.nan
    return _align(px)


def get_prices(source: str = "synthetic", start: str = "2005-01-01",
               refresh: bool = False, tickers: list[str] | None = None) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    path = cache_path(source)
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    uni = load_universe()
    if tickers is not None:
        uni = uni.loc[[t for t in tickers if t in uni.index]]
    if source == "yahoo":
        px = download_yahoo(list(uni.index), start)
    elif source == "synthetic":
        px = synthetic_prices(uni, start)
    else:
        raise ValueError(source)
    px.to_pickle(path)
    return px
