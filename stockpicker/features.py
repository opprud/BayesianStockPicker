"""Price-only features in five families (momentum, trend, mean reversion,
volatility, correlation) plus market-context features, and forward log-return
targets at several horizons.

All features at date t use prices up to and including t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "3Y": 756}

FEATURE_GROUPS = {
    "Momentum": ["mom_12_1", "mom_6_1", "mom_3m", "ret_1m"],
    "Trend": ["px_sma200", "sma50_sma200", "trend_t_126", "up_frac_63"],
    "Mean reversion": ["z_20", "rsi_14", "dist_52w_high", "dist_52w_low"],
    "Volatility": ["vol_21", "vol_63", "vol_252", "vol_ratio", "downvol_63",
                   "maxdd_252", "vol_of_vol"],
    "Correlation": ["corr_mkt_252", "beta_252", "idio_vol_252", "corr_change",
                    "corr_global_252"],
    "Market context": ["mkt_ret_252", "mkt_vol_63", "mkt_dd_252"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
FEATURE_GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

DESCRIPTIONS = {
    "mom_12_1": "Log return from t-12m to t-1m (classic momentum, skips last month)",
    "mom_6_1": "Log return from t-6m to t-1m",
    "mom_3m": "Log return over last 3 months",
    "ret_1m": "Log return over last month (short-term reversal)",
    "px_sma200": "log(price / 200-day SMA)",
    "sma50_sma200": "log(SMA50 / SMA200) — golden/death cross",
    "trend_t_126": "t-statistic of slope of log price over 6 months",
    "up_frac_63": "Fraction of up days over 3 months",
    "z_20": "(price - SMA20) / std20 — Bollinger z-score",
    "rsi_14": "RSI(14), scaled to [-0.5, 0.5]",
    "dist_52w_high": "log(price / 52-week high)",
    "dist_52w_low": "log(price / 52-week low)",
    "vol_21": "Annualised realised volatility, 1 month",
    "vol_63": "Annualised realised volatility, 3 months",
    "vol_252": "Annualised realised volatility, 1 year",
    "vol_ratio": "log(vol_21 / vol_252) — volatility regime shift",
    "downvol_63": "Annualised downside volatility, 3 months",
    "maxdd_252": "Max drawdown within the last year (approx., log)",
    "vol_of_vol": "Std of 1m volatility over 6 months / mean",
    "corr_mkt_252": "1y correlation with regional equal-weight market",
    "beta_252": "1y beta to regional market",
    "idio_vol_252": "Annualised idiosyncratic volatility vs regional market",
    "corr_change": "3m corr minus 1y corr with regional market",
    "corr_global_252": "1y correlation with global equal-weight market",
    "mkt_ret_252": "Regional market 1y log return",
    "mkt_vol_63": "Regional market 3m annualised volatility",
    "mkt_dd_252": "Regional market log distance from 1y high",
}


def _rolling_trend_t(logp: pd.DataFrame, n: int) -> pd.DataFrame:
    """t-stat of OLS slope of log price on time over a rolling window."""
    pos = np.arange(len(logp), dtype=float)[:, None]
    y = logp.values
    yv = np.where(np.isnan(y), 0.0, y)
    valid = (~np.isnan(y)).astype(float)
    cs = lambda a: np.cumsum(np.vstack([np.zeros((1, a.shape[1])), a]), axis=0)
    S_y, S_py, S_yy, S_n = cs(yv), cs(pos * yv), cs(yv * yv), cs(valid)
    out = np.full_like(y, np.nan)
    i = np.arange(n - 1, len(y))
    w = lambda S: S[i + 1] - S[i + 1 - n]
    sy, spy, syy, cnt = w(S_y), w(S_py), w(S_yy), w(S_n)
    start = (i - n + 1)[:, None].astype(float)
    sty = spy - start * sy                     # sum of local_t * y
    mt, my = (n - 1) / 2, sy / n
    cov = sty / n - mt * my
    var_t = (n * n - 1) / 12
    var_y = np.maximum(syy / n - my * my, 1e-12)
    r = np.clip(cov / np.sqrt(var_t * var_y), -0.9999, 0.9999)
    t = r * np.sqrt((n - 2) / (1 - r * r))
    t[cnt < n] = np.nan
    out[i] = t
    return pd.DataFrame(out, index=logp.index, columns=logp.columns)


def _rolling_corr_beta(r: pd.DataFrame, m: pd.Series, n: int):
    mp = min(n, max(20, int(0.8 * n)))
    mean_r = r.rolling(n, min_periods=mp).mean()
    mean_m = m.rolling(n, min_periods=mp).mean()
    cov = r.mul(m, axis=0).rolling(n, min_periods=mp).mean().sub(mean_r.mul(mean_m, axis=0))
    var_m = m.rolling(n, min_periods=mp).var(ddof=0)
    var_r = r.rolling(n, min_periods=mp).var(ddof=0)
    beta = cov.div(var_m, axis=0)
    corr = cov.div(np.sqrt(var_r.mul(var_m, axis=0)))
    idio = np.sqrt((var_r - beta.pow(2).mul(var_m, axis=0)).clip(lower=0) * 252)
    return corr, beta, idio


def compute_feature_panels(px: pd.DataFrame, universe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Wide (date x ticker) panels for every feature."""
    logp = np.log(px)
    r = logp.diff()
    A = np.sqrt(252)
    F: dict[str, pd.DataFrame] = {}

    # --- Momentum
    F["mom_12_1"] = logp.shift(21) - logp.shift(252)
    F["mom_6_1"] = logp.shift(21) - logp.shift(126)
    F["mom_3m"] = logp - logp.shift(63)
    F["ret_1m"] = logp - logp.shift(21)

    # --- Trend
    sma = lambda n: px.rolling(n, min_periods=int(n * 0.9)).mean()
    s20, s50, s200 = sma(20), sma(50), sma(200)
    F["px_sma200"] = np.log(px / s200)
    F["sma50_sma200"] = np.log(s50 / s200)
    F["trend_t_126"] = _rolling_trend_t(logp, 126).clip(-30, 30)
    F["up_frac_63"] = (r > 0).astype(float).where(r.notna()).rolling(63, min_periods=50).mean() - 0.5

    # --- Mean reversion
    F["z_20"] = ((px - s20) / px.rolling(20, min_periods=18).std()).clip(-5, 5)
    up = r.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    dn = (-r.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    F["rsi_14"] = up / (up + dn + 1e-12) - 0.5
    F["dist_52w_high"] = logp - logp.rolling(252, min_periods=200).max()
    F["dist_52w_low"] = logp - logp.rolling(252, min_periods=200).min()

    # --- Volatility
    v21 = r.rolling(21, min_periods=18).std() * A
    F["vol_21"] = v21
    F["vol_63"] = r.rolling(63, min_periods=50).std() * A
    F["vol_252"] = r.rolling(252, min_periods=200).std() * A
    F["vol_ratio"] = np.log((v21 + 1e-4) / (F["vol_252"] + 1e-4))
    F["downvol_63"] = np.sqrt((r.clip(upper=0) ** 2).rolling(63, min_periods=50).mean()) * A
    dd = logp - logp.rolling(252, min_periods=60).max()
    F["maxdd_252"] = dd.rolling(252, min_periods=200).min()
    F["vol_of_vol"] = v21.rolling(126, min_periods=100).std() / v21.rolling(126, min_periods=100).mean()

    # --- Correlation (vs regional and global equal-weight markets)
    reg = universe.reindex(px.columns)["region"].fillna("Other")
    glob = r.mean(axis=1)
    corr, beta, idio, corr63, mret, mvol, mdd = ({} for _ in range(7))
    for region, cols in reg.groupby(reg).groups.items():
        cols = list(cols)
        m = r[cols].mean(axis=1) if len(cols) >= 3 else glob
        c, b, iv = _rolling_corr_beta(r[cols], m, 252)
        c63, _, _ = _rolling_corr_beta(r[cols], m, 63)
        mlog = m.cumsum()
        for col in cols:
            corr[col], beta[col], idio[col], corr63[col] = c[col], b[col], iv[col], c63[col]
            mret[col] = mlog - mlog.shift(252)
            mvol[col] = m.rolling(63).std() * A
            mdd[col] = mlog - mlog.rolling(252, min_periods=60).max()
    to_df = lambda d: pd.DataFrame(d)[px.columns]
    F["corr_mkt_252"] = to_df(corr)
    F["beta_252"] = to_df(beta).clip(-3, 5)
    F["idio_vol_252"] = to_df(idio)
    F["corr_change"] = to_df(corr63) - F["corr_mkt_252"]
    F["corr_global_252"], _, _ = _rolling_corr_beta(r, glob, 252)
    F["mkt_ret_252"], F["mkt_vol_63"], F["mkt_dd_252"] = to_df(mret), to_df(mvol), to_df(mdd)
    return F


def forward_returns(px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    logp = np.log(px)
    return {h: logp.shift(-d) - logp for h, d in HORIZONS.items()}


def build_panel(px: pd.DataFrame, universe: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    """Long panel (date, ticker) of features + forward-return targets, sampled
    every `step` business days. Rows with any missing feature are dropped;
    targets may be NaN (not yet realised)."""
    F = compute_feature_panels(px, universe)
    Y = forward_returns(px)
    dates = px.index[::-1][::step][::-1]  # anchor sampling on the latest date
    cols = {}
    for k, df in {**F, **{f"y_{h}": v for h, v in Y.items()}}.items():
        cols[k] = df.loc[dates].stack(future_stack=True)
    panel = pd.DataFrame(cols)
    panel.index.names = ["date", "ticker"]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    meta = universe.reindex(panel.index.get_level_values("ticker"))
    for c in ["sector", "region", "country", "name"]:
        panel[c] = meta[c].values
    return panel


def latest_features(px: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    F = compute_feature_panels(px, universe)
    last = px.index[-1]
    df = pd.DataFrame({k: v.loc[last] for k, v in F.items()})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df.index.name = "ticker"
    return df.join(universe[["name", "sector", "region", "country"]], how="left")


# ---------------------------------------------------------------- analysis

def rank_ic(panel: pd.DataFrame, feature: str, target: str) -> pd.Series:
    """Per-date Spearman correlation between a feature and a forward return."""
    d = panel[[feature, target]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    lvl = d.index.get_level_values("date")
    rk = d.groupby(level="date").rank()
    dm = rk - rk.groupby(lvl).transform("mean")
    g = pd.DataFrame(dict(xy=dm[feature] * dm[target], xx=dm[feature] ** 2,
                          yy=dm[target] ** 2, n=1.0)).groupby(lvl).sum()
    ic = g.xy / np.sqrt(g.xx * g.yy).replace(0, np.nan)
    return ic[g.n >= 10].dropna()


def ic_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean rank IC and its t-stat (dates sub-sampled to non-overlap) for every
    feature x horizon."""
    rows = []
    for f in FEATURES:
        for h, d in HORIZONS.items():
            ic = rank_ic(panel, f, f"y_{h}")
            if ic.empty:
                continue
            step = max(1, d // 5)  # panel sampled weekly; thin to non-overlap
            ic_no = ic.iloc[::step]
            t = ic_no.mean() / (ic_no.std(ddof=1) / np.sqrt(len(ic_no))) if len(ic_no) > 2 else np.nan
            rows.append(dict(feature=f, group=FEATURE_GROUP_OF[f], horizon=h,
                             ic=ic.mean(), t_stat=t, n_periods=len(ic_no)))
    return pd.DataFrame(rows)


def quintile_returns(panel: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    d = panel[[feature, target]].dropna()
    q = d.groupby(level="date")[feature].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1 if len(x) >= 10 else np.nan)
    d = d.assign(q=q).dropna()
    return d.groupby("q")[target].agg(mean="mean", std="std", n="size")
