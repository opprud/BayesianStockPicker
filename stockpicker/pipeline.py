"""End-to-end pipeline: prices -> features -> BNN (eval + final) -> forecasts.

    python -m stockpicker.pipeline --source yahoo --start 2005-01-01
    python -m stockpicker.pipeline --source synthetic        # offline demo
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from .data import DATA_DIR, get_prices, load_universe
from .features import build_panel, latest_features
from .model import MODEL_DIR, StockBNN, TrainConfig, selection_table


def panel_path(source):
    return DATA_DIR / f"panel_{source}.pkl"


def get_panel(source, px=None, refresh=False):
    p = panel_path(source)
    if p.exists() and not refresh:
        return pd.read_pickle(p)
    px = get_prices(source) if px is None else px
    panel = build_panel(px, load_universe())
    panel.to_pickle(p)
    return panel


def run(source="synthetic", start="2005-01-01", refresh=False, cfg: TrainConfig | None = None,
        log=print):
    t0 = time.time()
    cfg = cfg or TrainConfig()
    uni = load_universe()
    log(f"[1/5] prices ({source})")
    px = get_prices(source, start, refresh=refresh)
    log(f"      {px.shape[1]} tickers, {px.index[0].date()} .. {px.index[-1].date()}")
    log("[2/5] features")
    panel = get_panel(source, px, refresh=True)
    log(f"      panel {panel.shape}")

    def ep_log(ep, n, h):
        if ep == 1 or ep % 5 == 0 or ep == n:
            log(f"      epoch {ep}/{n}  nll={h['nll']:.4f}  kl/n={h['kl']:.4f}")

    log("[3/5] train evaluation model (purged walk-forward split)")
    ev = StockBNN(cfg).fit(panel, log=ep_log)
    metrics = ev.evaluate(panel)
    metrics["_split"] = ev.split_info
    ev.save(f"bnn_eval_{source}")
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / f"metrics_{source}.json").write_text(json.dumps(metrics, indent=1, default=float))
    for h, m in metrics.items():
        if h.startswith("_"):
            continue
        log(f"      {h:>3}: NLL {m['nll']:.3f} (baseline {m['nll_baseline']:.3f})  "
            f"cov90 {dict(zip(m['levels'], m['coverage']))[0.9]:.2f}  rankIC {m['rank_ic']:+.3f}")

    log("[4/5] train final model on all realised data")
    fin = StockBNN(cfg).fit(panel, log=ep_log, final=True)
    fin.save(f"bnn_final_{source}")

    log("[5/5] forecasts for latest date")
    lf = latest_features(px, uni)
    pred = fin.predict(lf)
    tabs = selection_table(pred, lf[["name", "sector", "region", "country"]])
    pd.to_pickle(dict(asof=px.index[-1], tables=tabs, last_price=px.iloc[-1],
                      draws=pred["draws"].astype("float32"), tickers=list(lf.index)),
                 DATA_DIR / f"forecast_{source}.pkl")
    log(f"done in {time.time() - t0:.0f}s")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "yahoo"])
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--kl-scale", type=float, default=1.0)
    a = ap.parse_args()
    run(a.source, a.start, a.refresh, TrainConfig(epochs=a.epochs, kl_scale=a.kl_scale))
