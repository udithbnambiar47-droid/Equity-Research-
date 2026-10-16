"""Market-data access (Yahoo Finance via yfinance) and default-assumption derivation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# A few starter peer sets. Anything else: the user types their own peers.
DEFAULT_PEERS = {
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "ORCL", "ADBE"],
    "TCS.NS": ["INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "INFY.NS": ["TCS.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "HDFCBANK.NS": ["ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "ICICIBANK.NS": ["HDFCBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
}


def _series(df, *names) -> pd.Series:
    """First matching row of a yfinance statement as a year-indexed series (oldest first)."""
    if df is None or getattr(df, "empty", True):
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce").dropna().astype(float)
            s.index = pd.to_datetime(s.index).year
            s = s[~s.index.duplicated(keep="first")]
            return s.sort_index()
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def load_company(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    return {
        "info": t.info or {},
        "cashflow": t.cashflow,
        "income": t.income_stmt,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_multiples(tickers: tuple) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        try:
            i = yf.Ticker(tk).info or {}
        except Exception:
            i = {}
        rows.append({
            "Ticker": tk,
            "Name": i.get("shortName") or tk,
            "Market cap": i.get("marketCap"),
            "P/E (TTM)": i.get("trailingPE"),
            "P/E (Fwd)": i.get("forwardPE"),
            "EV/EBITDA": i.get("enterpriseToEbitda"),
            "EV/Revenue": i.get("enterpriseToRevenue"),
        })
    df = pd.DataFrame(rows)
    for c in ("Market cap", "P/E (TTM)", "P/E (Fwd)", "EV/EBITDA", "EV/Revenue"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


@dataclass
class Defaults:
    has_operating: bool
    rev0: float = np.nan
    growth: float = 0.08
    margin_now: float = np.nan
    margin_target: float = np.nan
    da_pct: float = 0.04
    capex_pct: float = 0.05
    nwc_pct: float = 0.05
    tax: float = 0.21
    ebitda_ttm: float = np.nan
    eps: float = np.nan
    fcf_hist: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rev_hist: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    margin_hist: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fcf_year: int = 0


def _tail_mean(s: pd.Series, n: int = 3, default: float = np.nan) -> float:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.tail(n).mean()) if len(s) else default


def _fcf_history(cf) -> pd.Series:
    s = _series(cf, "Free Cash Flow")
    if s.empty:
        ocf = _series(cf, "Operating Cash Flow", "Total Cash From Operating Activities")
        capex = _series(cf, "Capital Expenditure")
        if ocf.empty or capex.empty:
            return pd.Series(dtype=float)
        s = (ocf + capex).dropna()  # capex is negative in yfinance
    return s


def derive(company: dict) -> Defaults:
    """Derive sensible starting assumptions from reported history."""
    info, cf, inc = company["info"], company["cashflow"], company["income"]

    rev = _series(inc, "Total Revenue", "Operating Revenue")
    ebit = _series(inc, "EBIT", "Operating Income")
    da = _series(cf, "Depreciation And Amortization", "Depreciation Amortization Depletion", "Depreciation")
    capex = _series(cf, "Capital Expenditure")
    d_wc = _series(cf, "Change In Working Capital")
    tax_rate = _series(inc, "Tax Rate For Calcs")
    fcf = _fcf_history(cf)

    margin = (ebit / rev).dropna() if len(rev) and len(ebit) else pd.Series(dtype=float)
    has_op = bool(len(rev) and len(margin) and rev.iloc[-1] > 0)

    d = Defaults(has_operating=has_op, fcf_hist=fcf, rev_hist=rev, margin_hist=margin)
    d.fcf_year = int(fcf.index[-1]) if len(fcf) else 0
    d.eps = info.get("trailingEps", np.nan) or np.nan

    if has_op:
        d.rev0 = float(rev.iloc[-1])
        pos = rev[rev > 0]
        if len(pos) >= 2 and pos.index[-1] > pos.index[0]:
            n = pos.index[-1] - pos.index[0]
            d.growth = float(np.clip((pos.iloc[-1] / pos.iloc[0]) ** (1 / n) - 1, -0.05, 0.30))
        d.margin_now = float(margin.iloc[-1])
        d.margin_target = float(np.clip(_tail_mean(margin, 3, d.margin_now), -0.10, 0.60))
        d.da_pct = float(np.clip(_tail_mean(da / rev, 3, 0.04), 0.0, 0.30))
        d.capex_pct = float(np.clip(_tail_mean(capex.abs() / rev, 3, 0.05), 0.0, 0.40))
        d.tax = float(np.clip(_tail_mean(tax_rate, 3, 0.21), 0.05, 0.40))
        nwc = (-d_wc / rev.diff()).replace([np.inf, -np.inf], np.nan).dropna()
        d.nwc_pct = float(np.clip(nwc.median(), 0.0, 0.30)) if len(nwc) else 0.05

    ebitda = info.get("ebitda")
    if not ebitda and has_op and len(da):
        ebitda = float(ebit.iloc[-1] + da.iloc[-1])
    d.ebitda_ttm = float(ebitda) if ebitda else np.nan
    return d
