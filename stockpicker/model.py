"""Bayesian neural network (Bayes-by-Backprop, mean-field Gaussian weights)
with a heteroscedastic Gaussian head per horizon.

For every horizon h the network outputs mean mu_h(x) and log-variance
log s_h^2(x) of the forward log return.  Uncertainty splits into
  * epistemic  = Var_w[mu_h(x; w)]   (model doesn't know — shrinks with data)
  * aleatoric  = E_w[s_h^2(x; w)]    (inherent noise the model expects)
and the predictive distribution is the mixture over weight samples.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .features import FEATURES, HORIZONS

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
H_NAMES = list(HORIZONS)


class BayesLinear(nn.Module):
    def __init__(self, n_in, n_out, prior_sigma=1.0, rho_init=-5.0):
        super().__init__()
        self.w_mu = nn.Parameter(torch.empty(n_out, n_in).normal_(0, 1 / math.sqrt(n_in)))
        self.w_rho = nn.Parameter(torch.full((n_out, n_in), rho_init))
        self.b_mu = nn.Parameter(torch.zeros(n_out))
        self.b_rho = nn.Parameter(torch.full((n_out,), rho_init))
        self.prior_sigma = prior_sigma

    def forward(self, x, mode="local"):
        w_s, b_s = Fnn.softplus(self.w_rho), Fnn.softplus(self.b_rho)
        if mode == "mean":
            return Fnn.linear(x, self.w_mu, self.b_mu)
        if mode == "sample":  # one weight draw shared by the whole batch
            w = self.w_mu + w_s * torch.randn_like(w_s)
            b = self.b_mu + b_s * torch.randn_like(b_s)
            return Fnn.linear(x, w, b)
        # local reparameterisation trick (lower-variance gradients in training)
        m = Fnn.linear(x, self.w_mu, self.b_mu)
        v = Fnn.linear(x * x, w_s * w_s, b_s * b_s)
        return m + torch.sqrt(v + 1e-12) * torch.randn_like(m)

    def kl(self):
        ps = self.prior_sigma
        tot = 0.0
        for mu, rho in ((self.w_mu, self.w_rho), (self.b_mu, self.b_rho)):
            s = Fnn.softplus(rho)
            tot = tot + (torch.log(ps / s) + (s * s + mu * mu) / (2 * ps * ps) - 0.5).sum()
        return tot


class BNN(nn.Module):
    def __init__(self, n_in, n_out, hidden=(64, 32), prior_sigma=1.0):
        super().__init__()
        dims = [n_in, *hidden]
        self.hidden = nn.ModuleList(BayesLinear(a, b, prior_sigma) for a, b in zip(dims[:-1], dims[1:]))
        self.head = BayesLinear(dims[-1], 2 * n_out, prior_sigma)
        self.n_out = n_out

    def forward(self, x, mode="local"):
        for layer in self.hidden:
            x = Fnn.silu(layer(x, mode))
        out = self.head(x, mode)
        mu, logvar = out[:, :self.n_out], out[:, self.n_out:]
        return mu, logvar.clamp(-8, 4)

    def kl(self):
        return sum(m.kl() for m in self.modules() if isinstance(m, BayesLinear))


@dataclass
class TrainConfig:
    hidden: tuple = (64, 32)
    prior_sigma: float = 1.0
    epochs: int = 40
    batch_size: int = 1024
    lr: float = 2e-3
    kl_scale: float = 1.0      # multiply KL term; >1 = wider (more humble) posterior
    test_frac: float = 0.2     # last fraction of dates per horizon held out
    seed: int = 0
    n_samples: int = 200       # weight samples at prediction time


@dataclass
class Scaler:
    x_med: list = field(default_factory=list)
    x_iqr: list = field(default_factory=list)
    y_mean: list = field(default_factory=list)
    y_std: list = field(default_factory=list)

    def fx(self, X):
        return np.clip((X - np.array(self.x_med)) / np.array(self.x_iqr), -6, 6).astype(np.float32)


def purged_split(panel: pd.DataFrame, test_frac: float):
    """Per-horizon walk-forward split.  For horizon h with realised targets on
    dates [d0, d1]: test = last `test_frac` of those dates; train = dates whose
    target window ends before the test period starts (purge gap = h)."""
    dates = panel.index.get_level_values("date")
    udates = np.sort(dates.unique())
    tr = np.zeros((len(panel), len(H_NAMES)), bool)
    te = np.zeros_like(tr)
    info = {}
    for j, (h, d) in enumerate(HORIZONS.items()):
        y = panel[f"y_{h}"].values
        ok = ~np.isnan(y)
        if ok.sum() == 0:
            continue
        hd = np.sort(np.unique(dates[ok]))
        t0 = hd[int(len(hd) * (1 - test_frac))]
        purge_idx = max(0, np.searchsorted(udates, t0) - int(math.ceil(d / 5)))
        train_end = udates[purge_idx]
        tr[:, j] = ok & (dates < train_end)
        te[:, j] = ok & (dates >= t0)
        info[h] = dict(train_end=str(pd.Timestamp(train_end).date()),
                       test_start=str(pd.Timestamp(t0).date()),
                       test_end=str(pd.Timestamp(hd[-1]).date()),
                       n_train=int(tr[:, j].sum()), n_test=int(te[:, j].sum()))
    return tr, te, info


def _fit(X, Y, M, cfg: TrainConfig, n_eff: float, log=None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    net = BNN(X.shape[1], Y.shape[1], cfg.hidden, cfg.prior_sigma)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    Xt, Yt, Mt = map(torch.from_numpy, (X, np.nan_to_num(Y).astype(np.float32), M.astype(np.float32)))
    n = len(Xt)
    hist = []
    for ep in range(cfg.epochs):
        perm = torch.randperm(n)
        tot_nll, tot_kl, nb = 0.0, 0.0, 0
        warm = min(1.0, (ep + 1) / max(1, cfg.epochs // 4))  # KL warm-up
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            mu, lv = net(Xt[idx])
            m = Mt[idx]
            nll = 0.5 * (lv + (Yt[idx] - mu) ** 2 * torch.exp(-lv) + math.log(2 * math.pi))
            nll = (nll * m).sum() / m.sum().clamp(min=1)
            kl = net.kl() / n_eff
            loss = nll + warm * cfg.kl_scale * kl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot_nll += nll.item(); tot_kl += kl.item(); nb += 1
        sched.step()
        hist.append(dict(epoch=ep + 1, nll=tot_nll / nb, kl=tot_kl / nb))
        if log:
            log(ep + 1, cfg.epochs, hist[-1])
    return net, hist


@torch.no_grad()
def predict_samples(net: BNN, Xs: np.ndarray, n_samples: int, scaler: Scaler, seed=123):
    """Returns (mu_s, sig_s) arrays of shape (S, N, H) in log-return units."""
    torch.manual_seed(seed)
    net.eval()
    X = torch.from_numpy(Xs)
    mus, sigs = [], []
    ym, ys = np.array(scaler.y_mean), np.array(scaler.y_std)
    for _ in range(n_samples):
        mu, lv = net(X, mode="sample")
        mus.append(mu.numpy() * ys + ym)
        sigs.append(np.exp(0.5 * lv.numpy()) * ys)
    return np.stack(mus), np.stack(sigs)


def summarize_predictive(mu_s, sig_s, n_draw=4, seed=0):
    """Moment summary + Monte-Carlo draws of the log-return predictive."""
    rng = np.random.default_rng(seed)
    S, N, H = mu_s.shape
    eps = rng.standard_normal((n_draw, S, N, H))
    draws = (mu_s[None] + sig_s[None] * eps).reshape(n_draw * S, N, H)
    out = dict(
        mu=mu_s.mean(0),
        epi_sd=mu_s.std(0),
        ale_sd=np.sqrt((sig_s ** 2).mean(0)),
    )
    out["tot_sd"] = np.sqrt(out["epi_sd"] ** 2 + out["ale_sd"] ** 2)
    out["draws"] = draws  # log returns
    return out


class StockBNN:
    """Train / evaluate / persist wrapper."""

    def __init__(self, cfg: TrainConfig | None = None):
        self.cfg = cfg or TrainConfig()
        self.net: BNN | None = None
        self.scaler = Scaler()
        self.history: list = []
        self.split_info: dict = {}
        self.features = list(FEATURES)

    # ---------------------------------------------------------- training
    def fit(self, panel: pd.DataFrame, log=None, final=False):
        """final=False: train on train split only (for honest evaluation).
        final=True: train on every realised target (for live forecasting)."""
        cfg = self.cfg
        tr, te, self.split_info = purged_split(panel, cfg.test_frac)
        X = panel[self.features].values.astype(np.float64)
        Y = panel[[f"y_{h}" for h in H_NAMES]].values
        M = (~np.isnan(Y)) if final else tr
        fitrows = M.any(1)
        med = np.nanmedian(X[fitrows], 0)
        iqr = np.nanpercentile(X[fitrows], 75, 0) - np.nanpercentile(X[fitrows], 25, 0)
        ym = np.array([np.nanmean(np.where(M[:, j], Y[:, j], np.nan)) for j in range(len(H_NAMES))])
        ysd = np.array([np.nanstd(np.where(M[:, j], Y[:, j], np.nan)) for j in range(len(H_NAMES))])
        self.scaler = Scaler(med.tolist(), np.maximum(iqr, 1e-6).tolist(), ym.tolist(), ysd.tolist())
        Xs = self.scaler.fx(X[fitrows])
        Ys = ((Y - ym) / ysd)[fitrows].astype(np.float32)
        # effective sample size: overlapping weekly samples of the same stock are
        # highly dependent, so the "data size" used to weigh the KL is reduced.
        n_eff = max(1000.0, fitrows.sum() / 4.0)
        self.net, self.history = _fit(Xs, Ys, M[fitrows], cfg, n_eff, log)
        self._tr, self._te = tr, te
        return self

    # ---------------------------------------------------------- inference
    def predict(self, X_df: pd.DataFrame, n_samples=None):
        Xs = self.scaler.fx(X_df[self.features].values.astype(np.float64))
        mu_s, sig_s = predict_samples(self.net, Xs, n_samples or self.cfg.n_samples, self.scaler)
        return summarize_predictive(mu_s, sig_s)

    # ---------------------------------------------------------- evaluation
    def evaluate(self, panel: pd.DataFrame, n_samples=100) -> dict:
        """Out-of-sample metrics on the purged test split vs a naive baseline
        (training-mean drift, trailing 1y volatility scaled by sqrt(h)).
        Calibration uses the exact mixture CDF (PIT) of the predictive."""
        from scipy.special import logsumexp
        from scipy.stats import norm, spearmanr

        te = self._te
        rows = te.any(1)
        sub = panel[rows]
        Xs = self.scaler.fx(sub[self.features].values.astype(np.float64))
        mu_s, sig_s = predict_samples(self.net, Xs, n_samples, self.scaler)
        levels = np.round(np.arange(0.05, 1.0, 0.05), 2)
        res = {}
        for j, (h, d) in enumerate(HORIZONS.items()):
            m = te[rows, j]
            if m.sum() < 50:
                continue
            y = sub[f"y_{h}"].values[m]
            ms, ss = mu_s[:, m, j].astype(np.float64), sig_s[:, m, j].astype(np.float64)
            mu = ms.mean(0)
            epi2, ale2 = ms.var(0), (ss ** 2).mean(0)
            sd = np.sqrt(epi2 + ale2)
            nll = -(logsumexp(norm.logpdf(y, ms, ss), axis=0) - np.log(n_samples)).mean()
            pit = norm.cdf((y - ms) / ss).mean(0)
            b_mu = np.full_like(y, self.scaler.y_mean[j])
            b_sd = np.maximum(sub["vol_252"].values[m] * np.sqrt(d / 252), 1e-3)
            b_nll = -norm.logpdf(y, b_mu, b_sd).mean()
            b_pit = norm.cdf((y - b_mu) / b_sd)
            cov = [float((np.abs(pit - 0.5) <= L / 2).mean()) for L in levels]
            bcov = [float((np.abs(b_pit - 0.5) <= L / 2).mean()) for L in levels]
            dates = sub.index.get_level_values("date")[m]
            dfm = pd.DataFrame(dict(date=dates, y=y, mu=mu, sd=sd, err=np.abs(y - mu)))
            ic = dfm.groupby("date").apply(
                lambda g: spearmanr(g.mu, g.y)[0] if len(g) >= 10 else np.nan).mean()
            # does a low predicted sd => low realised dispersion?  (per-date deciles)
            dfm["dec"] = dfm.groupby("date")["sd"].transform(
                lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1 if len(s) >= 10 else np.nan)
            dec = dfm.dropna().groupby("dec").agg(
                pred_sd=("sd", "mean"), realised_sd=("y", "std"),
                mean_ret=("y", "mean"), abs_err=("err", "mean"))
            res[h] = dict(
                n=int(m.sum()), nll=float(nll), nll_baseline=float(b_nll),
                levels=levels.tolist(), coverage=cov, coverage_baseline=bcov,
                pit_hist=np.histogram(pit, 20, (0, 1))[0].tolist(),
                pit_hist_baseline=np.histogram(b_pit, 20, (0, 1))[0].tolist(),
                rank_ic=float(ic),
                sd_vs_abs_err_spearman=float(spearmanr(sd, np.abs(y - mu))[0]),
                mean_pred_sd=float(sd.mean()), realised_sd=float(y.std()),
                epistemic_share=float((epi2 / sd ** 2).mean()),
                deciles=dec.reset_index().to_dict("list"),
            )
        return res

    # ---------------------------------------------------------- persistence
    def save(self, name="bnn"):
        MODEL_DIR.mkdir(exist_ok=True)
        torch.save(self.net.state_dict(), MODEL_DIR / f"{name}.pt")
        cfg = asdict(self.cfg); cfg["hidden"] = list(cfg["hidden"])
        meta = dict(cfg=cfg, scaler=asdict(self.scaler), history=self.history,
                    split_info=self.split_info, features=self.features)
        (MODEL_DIR / f"{name}.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, name="bnn"):
        meta = json.loads((MODEL_DIR / f"{name}.json").read_text())
        cfg = TrainConfig(**{**meta["cfg"], "hidden": tuple(meta["cfg"]["hidden"])})
        obj = cls(cfg)
        obj.scaler = Scaler(**meta["scaler"])
        obj.history, obj.split_info, obj.features = meta["history"], meta["split_info"], meta["features"]
        obj.net = BNN(len(obj.features), len(H_NAMES), cfg.hidden, cfg.prior_sigma)
        obj.net.load_state_dict(torch.load(MODEL_DIR / f"{name}.pt"))
        return obj

    @staticmethod
    def exists(name="bnn"):
        return (MODEL_DIR / f"{name}.pt").exists()


# ---------------------------------------------------------------- selection

def selection_table(pred: dict, meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-horizon table of risk/return statistics in simple-return space."""
    out = {}
    D = pred["draws"]
    for j, h in enumerate(H_NAMES):
        R = np.expm1(D[:, :, j])                       # simple returns
        q05 = np.quantile(R, 0.05, axis=0)
        cvar = np.array([R[R[:, i] <= q05[i], i].mean() for i in range(R.shape[1])])
        years = HORIZONS[h] / 252
        df = pd.DataFrame(dict(
            exp_return=R.mean(0), median_return=np.median(R, 0), sd_return=R.std(0),
            p_loss=(R < 0).mean(0), var_5=q05, cvar_5=cvar,
            q95=np.quantile(R, 0.95, 0),
            mu_log=pred["mu"][:, j], sd_log=pred["tot_sd"][:, j],
            epistemic_sd=pred["epi_sd"][:, j], aleatoric_sd=pred["ale_sd"][:, j],
        ), index=meta.index)
        df["ret_per_risk"] = df.exp_return / df.sd_return
        df["ann_exp_return"] = (1 + df.exp_return).clip(lower=1e-6) ** (1 / years) - 1
        df["ann_sd_log"] = df.sd_log / np.sqrt(years)
        out[h] = meta.join(df)
    return out
