"""
Valuation engine: DCF, scenarios, Monte Carlo and sensitivity.

Pure NumPy, no Streamlit imports, so it can be unit-tested on its own.

One vectorised code path serves everything: pass scalars for a single DCF,
or arrays (one entry per simulation) for Monte Carlo. Parameters broadcast
against a (simulations x years) grid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

TV_METHODS = ("gordon", "exit", "blend")


@dataclass(frozen=True)
class Assumptions:
    rev0: float              # base-year revenue (or base FCF in fallback mode)
    growth_start: float      # year-1 revenue growth; fades linearly to terminal_g
    margin_start: float      # current EBIT margin
    margin_target: float     # EBIT margin reached after `margin_years`
    margin_years: int
    tax: float               # tax rate on EBIT
    da_pct: float            # D&A as % of revenue
    capex_pct: float         # capex as % of revenue
    nwc_pct: float           # investment in NWC as % of incremental revenue
    wacc: float
    terminal_g: float
    exit_multiple: float     # EV / EBITDA applied to final-year EBITDA
    years: int
    net_debt: float
    shares: float
    tv_method: str = "gordon"  # gordon | exit | blend


def _c(x):
    """Parameter -> column vector (n, 1)."""
    return np.atleast_1d(np.asarray(x, dtype=float))[:, None]


def run_dcf(a: Assumptions) -> dict:
    """Run the DCF. Returns 2-D arrays (n, years) and 1-D arrays (n,)."""
    T = int(a.years)
    t = np.arange(1, T + 1, dtype=float)[None, :]

    g0, tg, w = _c(a.growth_start), _c(a.terminal_g), _c(a.wacc)
    m0, m1 = _c(a.margin_start), _c(a.margin_target)
    tax, da_p, cx_p, nwc, mult = (_c(a.tax), _c(a.da_pct), _c(a.capex_pct),
                                  _c(a.nwc_pct), _c(a.exit_multiple))

    growth = g0 + (tg - g0) * (t - 1) / max(T - 1, 1)          # fades to terminal g
    k = max(min(int(a.margin_years), T), 1)
    margin = m0 + (m1 - m0) * np.minimum(t / k, 1.0)           # linear ramp to target

    rev = float(a.rev0) * np.cumprod(1.0 + growth, axis=1)
    prev = np.concatenate([np.full((rev.shape[0], 1), float(a.rev0)), rev[:, :-1]], axis=1)

    ebit = rev * margin
    da = rev * da_p
    capex = rev * cx_p
    d_nwc = nwc * (rev - prev)
    fcff = ebit * (1.0 - tax) + da - capex - d_nwc
    ebitda = ebit + da

    disc = (1.0 + w) ** t
    pv = fcff / disc

    f_T, e_T = fcff[:, -1:], ebitda[:, -1:]
    valid = w > tg
    den = np.where(valid, w - tg, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        tv_g = f_T * (1.0 + tg) / den                           # Gordon growth
        tv_x = e_T * mult                                       # exit multiple
        implied_g = (tv_x * w - f_T) / (tv_x + f_T)             # g implied by exit TV
        implied_mult = tv_g / e_T                               # multiple implied by Gordon TV

    if a.tv_method == "gordon":
        tv = tv_g
    elif a.tv_method == "exit":
        tv = tv_x
    else:
        tv = (tv_g + tv_x) / 2.0

    pv_tv = tv / disc[:, -1:]
    ev = pv.sum(axis=1, keepdims=True) + pv_tv
    equity = ev - a.net_debt
    per_share = equity / a.shares

    one = lambda x: x[:, 0]  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        tv_share = pv_tv / ev
    return {
        "years": T,
        "growth": growth, "margin": margin, "rev": rev, "ebit": ebit, "da": da,
        "capex": capex, "d_nwc": d_nwc, "fcff": fcff, "ebitda": ebitda,
        "disc": disc, "pv": pv,
        "tv_gordon": one(tv_g), "tv_exit": one(tv_x), "tv": one(tv), "pv_tv": one(pv_tv),
        "ev": one(ev), "equity": one(equity), "per_share": one(per_share),
        "implied_g": one(implied_g), "implied_mult": one(implied_mult),
        "tv_share": one(tv_share), "pv_sum": pv.sum(axis=1),
    }


def point_value(a: Assumptions) -> float:
    """Intrinsic value per share for scalar assumptions."""
    return float(run_dcf(a)["per_share"][0])


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------
def apply_scenario(a: Assumptions, d_growth=0.0, d_margin=0.0, d_wacc=0.0,
                   d_tg=0.0, d_mult=0.0) -> Assumptions:
    """Shift the key drivers (all deltas in decimals, multiple delta in turns)."""
    return replace(
        a,
        growth_start=a.growth_start + d_growth,
        margin_target=a.margin_target + d_margin,
        wacc=a.wacc + d_wacc,
        terminal_g=max(a.terminal_g + d_tg, 0.0),
        exit_multiple=max(a.exit_multiple + d_mult, 1.0),
    )


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def monte_carlo(a: Assumptions, n: int, seed: int, sd: dict) -> np.ndarray:
    """
    Simulate intrinsic value per share. Each key input is drawn independently
    from a normal distribution centred on the base case, then clipped to
    sensible bounds. `sd` keys: growth, margin, wacc, tg (decimals), mult_rel (fraction).
    """
    rng = np.random.default_rng(seed)
    g = np.clip(rng.normal(a.growth_start, sd["growth"], n), -0.20, 0.60)
    m = np.clip(rng.normal(a.margin_target, sd["margin"], n), -0.10, max(0.70, a.margin_target))
    w = np.clip(rng.normal(a.wacc, sd["wacc"], n), 0.04, 0.25)
    tg = np.clip(rng.normal(a.terminal_g, sd["tg"], n), 0.0, 0.05)
    tg = np.minimum(tg, w - 0.01)                       # keep WACC > g
    mult = np.maximum(rng.normal(a.exit_multiple, a.exit_multiple * sd["mult_rel"], n), 1.0)
    res = run_dcf(replace(a, growth_start=g, margin_target=m, wacc=w,
                          terminal_g=tg, exit_multiple=mult))
    out = res["per_share"]
    return out[np.isfinite(out)]


def mc_summary(samples: np.ndarray, price: float, band: float) -> dict:
    pct = {p: float(np.percentile(samples, p)) for p in (5, 10, 25, 50, 75, 90, 95)}
    return {
        "mean": float(samples.mean()), "std": float(samples.std()), "pct": pct,
        "p_above": float((samples > price).mean()),
        "p_above_band": float((samples > price * (1 + band)).mean()),
        "p_below_band": float((samples < price * (1 - band)).mean()),
        "n": int(samples.size),
    }


# --------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------
def sensitivity(a: Assumptions, axis: str = "tg", n: int = 5):
    """Grid of value/share: WACC (rows) vs terminal growth or exit multiple (cols)."""
    steps = np.linspace(-(n // 2), n // 2, n)
    waccs = a.wacc + steps * 0.01
    if axis == "tg":
        cols = np.maximum(a.terminal_g + steps * 0.005, 0.0)
    else:
        cols = np.maximum(a.exit_multiple + steps * 1.0, 1.0)
    grid = np.full((n, n), np.nan)
    for i, w in enumerate(waccs):
        for j, c in enumerate(cols):
            kw = {"terminal_g": float(c)} if axis == "tg" else {"exit_multiple": float(c)}
            grid[i, j] = point_value(replace(a, wacc=float(w), **kw))
    return waccs, cols, grid


# --------------------------------------------------------------------------
# WACC and formatting helpers
# --------------------------------------------------------------------------
def compute_wacc(rf, beta, erp, cost_debt, tax, mcap, debt):
    ke = rf + beta * erp
    kd = cost_debt * (1 - tax)
    total = mcap + debt
    we, wd = (mcap / total, debt / total) if total > 0 else (1.0, 0.0)
    return we * ke + wd * kd, ke, kd, we, wd


def fmt_big(x, cur=""):
    if x is None or not np.isfinite(x):
        return "n/a"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= div:
            return f"{cur}{x / div:,.2f}{unit}"
    return f"{cur}{x:,.0f}"
