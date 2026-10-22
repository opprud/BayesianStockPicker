"""Stock picker — price features + Bayesian neural network, in the browser.

    streamlit run app.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from stockpicker.data import DATA_DIR, cache_path, load_universe
from stockpicker.features import (DESCRIPTIONS, FEATURE_GROUP_OF, FEATURES, HORIZONS,
                                  ic_table, quintile_returns, rank_ic)
from stockpicker.model import MODEL_DIR, TrainConfig
from stockpicker import pipeline

st.set_page_config(page_title="Bayesian Stock Picker", layout="wide")

# ------------------------------------------------------------------ styling
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REGIONS = ["North America", "Europe", "Nordic", "Asia-Pacific", "Emerging"]
REGION_COLOR = dict(zip(REGIONS, SERIES))
REGION_SYMBOL = dict(zip(REGIONS, ["circle", "square", "diamond", "triangle-up", "x"]))
BLUE, ORANGE, GRAY = SERIES[0], SERIES[1], "#8a8985"
DIVERGING = [[0, "#b8322f"], [0.25, "#e88f86"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1, "#1c5cab"]]
H_LIST = list(HORIZONS)


def style(fig, h=380, legend=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=78 if legend else 48, b=10),
                      title=dict(y=0.98, yanchor="top", x=0, xanchor="left", font=dict(size=15)),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0) if legend else None,
                      showlegend=legend, hovermode="closest")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ------------------------------------------------------------------ cached loaders
def _mtime(p):
    return p.stat().st_mtime if p.exists() else 0


@st.cache_data(show_spinner=False)
def load_px(source, mt):
    return pd.read_pickle(cache_path(source))


@st.cache_data(show_spinner=False)
def load_panel(source, mt):
    return pd.read_pickle(pipeline.panel_path(source))


@st.cache_data(show_spinner="Computing rank ICs for all features × horizons…")
def load_ic(source, mt):
    return ic_table(load_panel(source, mt))


@st.cache_data(show_spinner=False)
def load_metrics(source, mt):
    p = MODEL_DIR / f"metrics_{source}.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_eval_meta(source, mt):
    p = MODEL_DIR / f"bnn_eval_{source}.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_forecast(source, mt):
    p = DATA_DIR / f"forecast_{source}.pkl"
    return pd.read_pickle(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def rolling_ic(source, mt, feature, target):
    ic = rank_ic(load_panel(source, mt), feature, target)
    return ic, ic.rolling(52, min_periods=26).mean()


# ------------------------------------------------------------------ sidebar
uni = load_universe()
with st.sidebar:
    st.title("Bayesian Stock Picker")
    src_label = st.radio("Price data", ["Yahoo Finance (real)", "Synthetic (offline demo)"],
                         index=0 if cache_path("yahoo").exists() or not cache_path("synthetic").exists() else 1)
    source = "yahoo" if src_label.startswith("Yahoo") else "synthetic"

    with st.expander("Run pipeline", expanded=not (DATA_DIR / f"forecast_{source}.pkl").exists()):
        start = st.date_input("History from", pd.Timestamp("2005-01-01"))
        refresh = st.checkbox("Re-download prices", value=False)
        epochs = st.slider("Epochs", 10, 100, 40, 5)
        hidden = st.selectbox("Hidden layers", ["64,32", "128,64", "32,16", "128,64,32"])
        kl_scale = st.select_slider("KL scale (posterior width)", [0.25, 0.5, 1.0, 2.0, 4.0], 1.0,
                                    help=">1 makes the network more humble (wider epistemic uncertainty)")
        if st.button("Download · features · train", type="primary", width="stretch"):
            cfg = TrainConfig(epochs=epochs, kl_scale=kl_scale,
                              hidden=tuple(int(x) for x in hidden.split(",")))
            with st.status("Running pipeline…", expanded=True) as stat:
                lines = []
                box = st.empty()

                def log(msg):
                    lines.append(msg)
                    box.code("\n".join(lines[-14:]))
                try:
                    pipeline.run(source, str(start), refresh, cfg, log=log)
                    stat.update(label="Pipeline finished", state="complete")
                except Exception as e:  # surface download / training errors
                    stat.update(label=f"Failed: {e}", state="error")
                    st.exception(e)
            st.cache_data.clear()

    st.subheader("Universe filter")
    regions = st.multiselect("Region", REGIONS, default=REGIONS)
    sectors_all = sorted(uni.sector.unique())
    sectors = st.multiselect("Sector", sectors_all, default=sectors_all)
    st.caption(f"{len(uni)} tickers in universe.csv — edit it to add your own.")

px_path = cache_path(source)
if not px_path.exists() or not pipeline.panel_path(source).exists():
    st.info("No data yet for this source. Open **Run pipeline** in the sidebar and press the button. "
            "Yahoo needs internet; the synthetic market works offline.")
    st.stop()

mt_px, mt_pan = _mtime(px_path), _mtime(pipeline.panel_path(source))
mt_mod = _mtime(MODEL_DIR / f"metrics_{source}.json")
mt_fc = _mtime(DATA_DIR / f"forecast_{source}.pkl")
px = load_px(source, mt_px)
panel = load_panel(source, mt_pan)
in_filter = uni[uni.region.isin(regions) & uni.sector.isin(sectors)].index
sel_tickers = [t for t in px.columns if t in in_filter]

if source == "synthetic":
    st.warning("Showing the **synthetic** market (simulated prices with planted momentum, "
               "volatility regimes and mean reversion). Switch to Yahoo for real stocks.", icon="🧪")

tab_data, tab_feat, tab_model, tab_sel = st.tabs(
    ["Data", "Features", "Model diagnostics", "Forecast & selection"])

# ================================================================== DATA
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers with data", f"{px.shape[1]}")
    c2.metric("In current filter", f"{len(sel_tickers)}")
    c3.metric("History", f"{px.index[0]:%Y-%m-%d} → {px.index[-1]:%Y-%m-%d}")
    c4.metric("Panel rows (weekly samples)", f"{len(panel):,}")

    cnt = uni.loc[uni.index.isin(px.columns)].groupby(["region", "sector"]).size().unstack(fill_value=0)
    fig = go.Figure(go.Heatmap(z=cnt.values, x=cnt.columns, y=cnt.index, colorscale=[[0, "#f0efec"], [1, "#1c5cab"]],
                               text=cnt.values, texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{z}<extra></extra>",
                               showscale=False, xgap=2, ygap=2))
    fig.update_layout(title="Tickers per region × sector")
    show(style(fig, 300, legend=False))

    pick = st.multiselect("Compare price history (indexed to 100)", sel_tickers, default=sel_tickers[:3],
                          max_selections=8)
    if pick:
        fig = go.Figure()
        for i, t in enumerate(pick):
            s = px[t].dropna()
            fig.add_trace(go.Scatter(x=s.index, y=100 * s / s.iloc[0], name=f"{t} · {uni.loc[t, 'name']}",
                                     line=dict(width=2, color=SERIES[i % 8])))
        fig.update_yaxes(type="log", title="Index (log scale)")
        fig.update_layout(hovermode="x unified")
        show(style(fig, 420))
    missing = sorted(set(uni.index) - set(px.columns))
    if missing:
        st.caption(f"{len(missing)} tickers had no / too little data: {', '.join(missing)}")

# ================================================================== FEATURES
with tab_feat:
    st.markdown("Five families of price features, as on the slide: **momentum, trend, mean reversion, "
                "volatility, correlation** (+ market context). Predictive power is measured as the "
                "**rank IC**: Spearman correlation, across stocks on each date, between the feature and the "
                "subsequent return; averaged over dates.")
    ic = load_ic(source, mt_pan)
    piv = ic.pivot(index="feature", columns="horizon", values="ic").reindex(index=FEATURES, columns=H_LIST)
    tpiv = ic.pivot(index="feature", columns="horizon", values="t_stat").reindex(index=FEATURES, columns=H_LIST)
    ylab = [f"{FEATURE_GROUP_OF[f]} · {f}" for f in FEATURES]
    lim = float(np.nanmax(np.abs(piv.values)))
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=H_LIST, y=ylab, zmin=-lim, zmax=lim, colorscale=DIVERGING, xgap=2, ygap=2,
        customdata=tpiv.values, text=np.round(piv.values, 3), texttemplate="%{text}",
        hovertemplate="%{y}<br>horizon %{x}<br>IC %{z:.3f}<br>t-stat (non-overlapping) %{customdata:.1f}<extra></extra>",
        colorbar=dict(title="IC", thickness=10)))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Mean rank IC: feature × horizon (blue = higher feature → higher future return)")
    show(style(fig, 760, legend=False))
    st.caption("|t| > 2 is a rough significance threshold. t-stats thin samples to non-overlapping periods; "
               "long horizons have few independent observations.")

    st.divider()
    c1, c2 = st.columns([1, 1])
    feat = c1.selectbox("Feature", FEATURES, index=0, format_func=lambda f: f"{f} — {DESCRIPTIONS[f]}")
    hz = c2.selectbox("Horizon", H_LIST, index=3)
    c1, c2 = st.columns(2)
    q = quintile_returns(panel, feat, f"y_{hz}")
    fig = go.Figure(go.Bar(x=[f"Q{int(i)}" for i in q.index], y=q["mean"], marker_color=BLUE,
                           customdata=np.c_[q["std"], q["n"]],
                           hovertemplate="%{x}: mean log return %{y:.2%}<br>sd %{customdata[0]:.2%}<br>n=%{customdata[1]}<extra></extra>"))
    fig.update_layout(title=f"Mean {hz} forward log return by {feat} quintile (Q1 low → Q5 high)")
    fig.update_yaxes(tickformat=".1%")
    with c1:
        show(style(fig, 340, legend=False))
    ic_s, ic_roll = rolling_ic(source, mt_pan, feat, f"y_{hz}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ic_s.index, y=ic_s, name="weekly IC", line=dict(width=1, color="rgba(42,120,214,0.25)")))
    fig.add_trace(go.Scatter(x=ic_roll.index, y=ic_roll, name="1-year average", line=dict(width=2, color=BLUE)))
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    fig.update_layout(title=f"IC of {feat} over time — 'they come and go'", hovermode="x unified")
    with c2:
        show(style(fig, 340))

    c1, c2 = st.columns(2)
    last = panel.index.get_level_values("date").max()
    snap = panel.xs(last, level="date")
    snap = snap[snap.index.isin(sel_tickers)]
    fig = go.Figure()
    for r in REGIONS:
        v = snap.loc[snap.region == r, feat]
        if len(v):
            fig.add_trace(go.Box(y=v, name=r, marker_color=REGION_COLOR[r], boxpoints="all", jitter=0.4,
                                 pointpos=0, text=v.index, hovertemplate="%{text}: %{y:.3f}<extra></extra>"))
    fig.update_layout(title=f"{feat} today by region ({last:%Y-%m-%d})")
    with c1:
        show(style(fig, 360, legend=False))
    samp = panel[FEATURES].sample(min(20000, len(panel)), random_state=0)
    cm = samp.rank().corr()
    fig = go.Figure(go.Heatmap(z=cm.values, x=FEATURES, y=FEATURES, zmin=-1, zmax=1, colorscale=DIVERGING,
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>", colorbar=dict(thickness=10)))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Feature–feature rank correlation (redundancy)")
    with c2:
        show(style(fig, 360, legend=False))

    st.divider()
    tk = st.selectbox("Single-stock feature view", sel_tickers or list(px.columns),
                      format_func=lambda t: f"{t} · {uni.loc[t, 'name']} ({uni.loc[t, 'region']}, {uni.loc[t, 'sector']})")
    reps = {"Momentum": "mom_12_1", "Trend": "px_sma200", "Mean reversion": "z_20",
            "Volatility": "vol_63", "Correlation": "corr_mkt_252"}
    d = panel.xs(tk, level="ticker")
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=["Price (log)"] + [f"{g}: {f}" for g, f in reps.items()])
    s = px[tk].dropna()
    fig.add_trace(go.Scatter(x=s.index, y=s, line=dict(width=1.5, color=BLUE), name="price"), 1, 1)
    fig.update_yaxes(type="log", row=1, col=1)
    for i, (g, f) in enumerate(reps.items(), start=2):
        fig.add_trace(go.Scatter(x=d.index, y=d[f], line=dict(width=1.5, color=BLUE), name=f), i, 1)
        fig.add_hline(y=0, line_color=GRAY, line_width=0.5, row=i, col=1)
    fig.update_layout(hovermode="x unified")
    show(style(fig, 900, legend=False))

# ================================================================== MODEL
with tab_model:
    metrics = load_metrics(source, mt_mod)
    meta = load_eval_meta(source, mt_mod)
    if not metrics:
        st.info("Train the model from the sidebar first.")
    else:
        st.markdown(
            "The network is evaluated on a **purged walk-forward split**: for every horizon the last 20 % of dates "
            "with a realised return is held out, and training data stops one horizon before that, so no future "
            "return leaks into training. The **baseline** is a naive forecast: average drift and trailing 1-year "
            "volatility × √horizon.")
        hs = [h for h in H_LIST if h in metrics]
        rows = []
        for h in hs:
            m = metrics[h]
            cov = dict(zip(m["levels"], m["coverage"]))
            bcov = dict(zip(m["levels"], m["coverage_baseline"]))
            sp = metrics["_split"][h]
            rows.append({"horizon": h, "train ≤": sp["train_end"], "test": f"{sp['test_start']} → {sp['test_end']}",
                         "n test": m["n"], "NLL BNN": m["nll"], "NLL baseline": m["nll_baseline"],
                         "90% interval hit-rate": cov[0.9], "baseline 90%": bcov[0.9],
                         "50% interval hit-rate": cov[0.5],
                         "rank IC (mean)": m["rank_ic"], "corr(pred sd, |error|)": m["sd_vs_abs_err_spearman"],
                         "epistemic share of var": m["epistemic_share"]})
        mt = pd.DataFrame(rows).set_index("horizon")
        st.dataframe(mt.style.format({"NLL BNN": "{:.3f}", "NLL baseline": "{:.3f}", "90% interval hit-rate": "{:.1%}",
                                      "baseline 90%": "{:.1%}", "50% interval hit-rate": "{:.1%}",
                                      "rank IC (mean)": "{:+.3f}", "corr(pred sd, |error|)": "{:+.3f}",
                                      "epistemic share of var": "{:.1%}"}), width="stretch")
        st.caption("Lower NLL is better. A calibrated model hits its 90 % interval 90 % of the time. "
                   "Rank IC > 0 means the predicted mean orders stocks correctly; corr(pred sd, |error|) > 0 means "
                   "the predicted uncertainty knows when it will be wrong.")

        hsel = st.radio("Horizon", hs, horizontal=True, index=min(3, len(hs) - 1), key="mh")
        m = metrics[hsel]
        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot", width=1),
                                 name="perfect"))
        fig.add_trace(go.Scatter(x=m["levels"], y=m["coverage"], name="BNN", mode="lines+markers",
                                 line=dict(color=BLUE, width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=m["levels"], y=m["coverage_baseline"], name="baseline", mode="lines+markers",
                                 line=dict(color=ORANGE, width=2), marker=dict(size=8, symbol="square")))
        fig.update_layout(title=f"Calibration ({hsel}): nominal vs realised interval coverage",
                          xaxis_title="nominal central interval", yaxis_title="share of outcomes inside")
        fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
        with c1:
            show(style(fig, 380))
        dec = pd.DataFrame(m["deciles"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dec.dec, y=dec.realised_sd, name="realised sd of returns", marker_color=BLUE))
        fig.add_trace(go.Scatter(x=dec.dec, y=dec.pred_sd, name="predicted sd", mode="lines+markers",
                                 line=dict(color=ORANGE, width=2), marker=dict(size=8)))
        fig.update_layout(title=f"Does 'low predicted variance' pick safer stocks? ({hsel})",
                          xaxis_title="predicted-sd decile within each date (1 = most certain)", yaxis_title="log-return sd")
        with c2:
            show(style(fig, 380))
        k1, k2, k3 = st.columns(3)
        k1.metric("Realised sd, decile 1 (most certain)", pct(dec.realised_sd.iloc[0]))
        k2.metric("Realised sd, decile 10 (least certain)", pct(dec.realised_sd.iloc[-1]))
        k3.metric("Mean return D1 vs D10", f"{pct(dec.mean_ret.iloc[0])} vs {pct(dec.mean_ret.iloc[-1])}")

        c1, c2 = st.columns(2)
        bins = np.linspace(0.025, 0.975, 20)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bins, y=np.array(m["pit_hist"]) / m["n"] * 20, name="BNN", marker_color=BLUE))
        fig.add_trace(go.Bar(x=bins, y=np.array(m["pit_hist_baseline"]) / m["n"] * 20, name="baseline",
                             marker_color=ORANGE))
        fig.add_hline(y=1, line_color=GRAY, line_dash="dot")
        fig.update_layout(title="PIT histogram (flat = calibrated; ∪ = too narrow; ∩ = too wide)",
                          barmode="group", bargap=0.15, bargroupgap=0.05, xaxis_title="predictive CDF at outcome",
                          yaxis_title="density")
        with c1:
            show(style(fig, 340))
        if meta:
            hist = pd.DataFrame(meta["history"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.epoch, y=hist.nll, name="data NLL (standardised)",
                                     line=dict(color=BLUE, width=2)))
            fig.add_trace(go.Scatter(x=hist.epoch, y=hist.kl, name="KL / n_eff", line=dict(color=ORANGE, width=2)))
            fig.update_layout(title="Training curve (evaluation model)", xaxis_title="epoch")
            with c2:
                show(style(fig, 340))

# ================================================================== SELECTION
with tab_sel:
    fc = load_forecast(source, mt_fc)
    if fc is None:
        st.info("Train the model from the sidebar first.")
        st.stop()
    tickers = fc["tickers"]
    st.markdown(f"Forecasts as of **{pd.Timestamp(fc['asof']):%Y-%m-%d}** from the final model (trained on all "
                "realised data). Every number comes from the full predictive distribution — weight uncertainty "
                "(epistemic) × noise the model expects (aleatoric) — not a single point forecast. Returns are in "
                "each stock's local currency.")
    CRIT = {
        "Lowest total uncertainty (sd)": ("sd_return", True),
        "Lowest probability of loss": ("p_loss", True),
        "Best 5% CVaR (expected tail loss)": ("cvar_5", False),
        "Best return per unit risk": ("ret_per_risk", False),
        "Highest expected return": ("exp_return", False),
        "Lowest model (epistemic) uncertainty": ("epistemic_sd", True),
    }
    c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
    hz = c1.selectbox("Horizon", H_LIST, index=3, key="sh")
    crit = c2.selectbox("Rank by", list(CRIT), index=0)
    min_ret = c3.number_input("Min. expected return (%)", value=0.0, step=1.0) / 100
    topn = c4.slider("Show top", 5, 40, 15)
    T = fc["tables"][hz]
    T = T[T.index.isin(sel_tickers)]
    col, asc = CRIT[crit]
    ranked = T[T.exp_return >= min_ret].sort_values(col, ascending=asc)
    top = ranked.head(topn)

    show_cols = {"name": "Name", "region": "Region", "sector": "Sector", "exp_return": "E[return]",
                 "median_return": "Median", "sd_return": "SD", "p_loss": "P(loss)", "var_5": "5% VaR",
                 "cvar_5": "5% CVaR", "q95": "95% upside", "epistemic_sd": "Epistemic sd (log)",
                 "aleatoric_sd": "Aleatoric sd (log)", "ret_per_risk": "Return / SD"}
    pfmt = {v: "{:.1%}" for k, v in show_cols.items() if k not in ("name", "region", "sector", "ret_per_risk")}
    pfmt["Return / SD"] = "{:.2f}"
    st.dataframe(top[list(show_cols)].rename(columns=show_cols).style.format(pfmt), width="stretch",
                 height=min(560, 38 + 35 * len(top)))

    # --- risk / return map with Pareto frontier
    c1, c2 = st.columns([3, 2])
    fig = go.Figure()
    for r in REGIONS:
        d = T[T.region == r]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d.sd_return, y=d.exp_return, mode="markers", name=r,
            marker=dict(size=10, color=REGION_COLOR[r], symbol=REGION_SYMBOL[r],
                        line=dict(width=1.5, color="rgba(255,255,255,0.9)")),
            customdata=np.c_[d.index, d.name, d.sector, d.p_loss],
            hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>%{customdata[2]}<br>"
                          "E[r] %{y:.1%} · SD %{x:.1%}<br>P(loss) %{customdata[3]:.0%}<extra></extra>"))
    s = T.sort_values("sd_return")
    front = s[s.exp_return >= s.exp_return.cummax()]
    fig.add_trace(go.Scatter(x=front.sd_return, y=front.exp_return, mode="lines", name="efficient frontier",
                             line=dict(color=GRAY, width=2, dash="dot"), hoverinfo="skip"))
    for t in top.index[:5]:
        fig.add_annotation(x=T.loc[t, "sd_return"], y=T.loc[t, "exp_return"], text=t, showarrow=False,
                           yshift=12, font=dict(size=11))
    fig.update_layout(title=f"Risk vs expected return — {hz} horizon", xaxis_title="SD of return",
                      yaxis_title="expected return")
    fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
    with c1:
        show(style(fig, 480))

    grp_by = c2.radio("Group by", ["region", "sector"], horizontal=True)
    G = T.groupby(grp_by).agg(exp_return=("exp_return", "median"), sd=("sd_return", "median"),
                              p_loss=("p_loss", "median"), n=("name", "size")).sort_values("sd")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=G.index, x=G.sd, orientation="h", name="median SD", marker_color=BLUE,
                         customdata=np.c_[G.exp_return, G.p_loss, G.n],
                         hovertemplate="%{y}<br>median SD %{x:.1%}<br>median E[r] %{customdata[0]:.1%}"
                                       "<br>median P(loss) %{customdata[1]:.0%}<br>n=%{customdata[2]}<extra></extra>"))
    fig.add_trace(go.Bar(y=G.index, x=G.exp_return, orientation="h", name="median E[return]", marker_color=ORANGE))
    fig.update_layout(title=f"By {grp_by}: median uncertainty vs return ({hz})", barmode="group",
                      bargap=0.25, bargroupgap=0.08)
    fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(autorange="reversed")
    with c2:
        show(style(fig, 440))

    # --- per-stock uncertainty views
    st.divider()
    default = list(top.index[:3])
    picks = st.multiselect("Compare stocks (max 4)", list(T.index), default=default, max_selections=4,
                           format_func=lambda t: f"{t} · {T.loc[t, 'name']}")
    if picks:
        D = fc["draws"]                               # (draws, stocks, horizons) log returns
        idx = [tickers.index(t) for t in picks]
        days = ["now"] + H_LIST
        c1, c2 = st.columns(2)
        fig = go.Figure()
        for k, (t, i) in enumerate(zip(picks, idx)):
            c = SERIES[k]
            V = 100 * np.exp(D[:, i, :])
            q = {p: np.r_[100, np.quantile(V, p, axis=0)] for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
            rgb = tuple(int(c[j:j + 2], 16) for j in (1, 3, 5))
            fig.add_trace(go.Scatter(x=days + days[::-1], y=np.r_[q[0.95], q[0.05][::-1]], fill="toself", mode="lines",
                                     fillcolor=f"rgba{rgb + (0.10,)}", line=dict(width=0), hoverinfo="skip",
                                     showlegend=False, legendgroup=t))
            fig.add_trace(go.Scatter(x=days + days[::-1], y=np.r_[q[0.75], q[0.25][::-1]], fill="toself", mode="lines",
                                     fillcolor=f"rgba{rgb + (0.22,)}", line=dict(width=0), hoverinfo="skip",
                                     showlegend=False, legendgroup=t))
            fig.add_trace(go.Scatter(x=days, y=q[0.5], name=t, line=dict(color=c, width=2), legendgroup=t,
                                     mode="lines+markers", marker=dict(size=8),
                                     customdata=np.c_[q[0.05], q[0.95]],
                                     hovertemplate=f"{t} %{{x}}: median %{{y:.0f}}"
                                                   "<br>90% band %{customdata[0]:.0f}–%{customdata[1]:.0f}<extra></extra>"))
        fig.add_hline(y=100, line_color=GRAY, line_width=1)
        fig.update_layout(title="Fan chart: value of 100 invested (median, 50% and 90% bands)",
                          xaxis_title="horizon (not to time scale)", yaxis_title="value")
        with c1:
            show(style(fig, 440))

        j = H_LIST.index(hz)
        fig = go.Figure()
        for k, (t, i) in enumerate(zip(picks, idx)):
            R = np.expm1(D[:, i, j])
            fig.add_trace(go.Violin(x=R, name=t, line_color=SERIES[k], orientation="h", side="positive",
                                    meanline_visible=True, points=False, width=1.8, spanmode="hard"))
        fig.add_vline(x=0, line_color=GRAY, line_width=1)
        fig.update_layout(title=f"Predictive distribution of {hz} return", xaxis_title="return", violingap=0.1)
        fig.update_xaxes(tickformat=".0%")
        with c2:
            show(style(fig, 440, legend=False))

        c1, c2 = st.columns(2)
        fig = go.Figure()
        for k, t in enumerate(picks):
            fig.add_trace(go.Scatter(x=H_LIST, y=[fc["tables"][h].loc[t, "p_loss"] for h in H_LIST], name=t,
                                     mode="lines+markers", line=dict(color=SERIES[k], width=2), marker=dict(size=8)))
        fig.update_layout(title="Probability of loss by horizon", yaxis_title="P(return < 0)")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        with c1:
            show(style(fig, 360))
        fig = go.Figure()
        epi = [fc["tables"][hz].loc[t, "epistemic_sd"] ** 2 for t in picks]
        ale = [fc["tables"][hz].loc[t, "aleatoric_sd"] ** 2 for t in picks]
        fig.add_trace(go.Bar(x=picks, y=ale, name="aleatoric (market noise)", marker_color=BLUE))
        fig.add_trace(go.Bar(x=picks, y=epi, name="epistemic (model doesn't know)", marker_color=ORANGE))
        fig.update_layout(title=f"Where the uncertainty comes from ({hz}, log-return variance)", barmode="stack",
                          bargap=0.4)
        with c2:
            show(style(fig, 360))

    with st.expander("Full table (all filtered stocks, this horizon)"):
        st.dataframe(ranked[list(show_cols)].rename(columns=show_cols).style.format(pfmt),
                     width="stretch")
        st.download_button("Download CSV", ranked.to_csv().encode(), f"forecast_{source}_{hz}.csv", "text/csv")

st.sidebar.caption("Research tool, not investment advice. Past price patterns fade; check the "
                   "Model diagnostics tab before trusting any ranking.")
