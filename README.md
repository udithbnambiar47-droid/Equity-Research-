# Equity Research App: DCF, Scenarios, Monte Carlo & Relative Valuation

A Streamlit app that values a listed stock with a revenue and margin driven **DCF**, cross-checks it with an **exit multiple**, stress-tests it with **bull / base / bear scenarios** and a **Monte Carlo simulation**, compares it with **peer P/E and EV/EBITDA**, and exports the result to **Excel and PDF**.

> **Disclaimer:** Educational use only. Not investment advice. Output depends heavily on the assumptions you choose.

---

## Features

| Area | What it does |
|---|---|
| **Revenue & margin forecast** | Projects revenue (growth fades to terminal rate), EBIT margin (ramps to a target), D&A, capex and working capital, then derives free cash flow to the firm (FCFF). Defaults come from reported history. |
| **Terminal value cross-check** | Gordon growth and exit EV/EBITDA are both computed. Choose one or blend them. Each method shows the assumption the other implies (implied exit multiple, implied perpetual growth) with warnings when they disagree. |
| **Scenarios** | Editable bear / base / bull shifts to growth, margin, WACC, terminal growth and exit multiple, with probability weights and a probability-weighted value. |
| **Monte Carlo** | Thousands of simulations on growth, margin, WACC, terminal growth and exit multiple. Shows the distribution, percentiles and the probability that value exceeds price. |
| **Relative valuation** | Pulls peer multiples (P/E, forward P/E, EV/EBITDA, EV/Revenue), computes P25 / median / P75 and the value per share they imply. |
| **Football field** | One chart combining DCF range, Monte Carlo P10-P90, peer ranges and the 52-week range against the current price. |
| **Export** | One-click Excel workbook (one sheet per table) and a PDF report with charts. |
| **Fallback mode** | If income-statement data is missing, switch to a plain FCF-growth DCF. |

---

## Project Structure

```
.
├── app.py             # Streamlit UI and report assembly
├── engine.py          # Vectorised DCF, scenarios, Monte Carlo, sensitivity (no Streamlit)
├── market_data.py     # Yahoo Finance access and default-assumption derivation
├── export.py          # Number formatting, Excel and PDF builders
├── requirements.txt
└── README.md
```

`engine.py` has no Streamlit dependency, so it can be unit-tested or reused on its own.

---

## Getting Started

Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Deploy on Streamlit Community Cloud

1. Push all files in this folder to a GitHub repository.
2. At [share.streamlit.io](https://share.streamlit.io) click **New app**, pick the repo and set the main file to `app.py`.
3. Click **Deploy**.

---

## Usage

1. Enter a **ticker** (`AAPL`, `TCS.NS`, `RELIANCE.BO`, `VOD.L`). Defaults load from reported financials.
2. Review **Operating assumptions**: base revenue, year-1 growth, target EBIT margin and ramp period, D&A, capex, NWC.
3. Set the **terminal value method**, terminal growth, exit multiple and WACC inputs in the sidebar.
4. Add **peer tickers** (a few starter peer sets are pre-filled for some large caps).
5. Read the verdict, then explore the tabs: Valuation, Forecast, Scenarios, Monte Carlo, Relative valuation, Sensitivity.
6. In **Export**, click *Generate* then download the Excel or PDF file.

---

## Methodology

**Forecast**

```
Revenue_t   = Revenue_(t-1) x (1 + g_t)          g_t fades linearly from year-1 growth to terminal growth
EBIT margin = ramps linearly from current to target over N years
FCFF_t      = EBIT x (1 - tax) + D&A - Capex - NWC% x (Revenue_t - Revenue_(t-1))
EBITDA_t    = EBIT + D&A
```

**Discount rate**

```
Ke   = Rf + Beta x ERP
Kd   = pre-tax Kd x (1 - tax)
WACC = E/(D+E) x Ke + D/(D+E) x Kd
```

**Terminal value**

```
Gordon:   TV = FCFF_N x (1 + g) / (WACC - g)
Exit:     TV = EBITDA_N x exit multiple
Blend:    average of the two

Implied growth (from exit TV):    g = (TV x WACC - FCFF_N) / (TV + FCFF_N)
Implied multiple (from Gordon TV): TV / EBITDA_N
```

**Value per share**

```
EV = sum of PV(FCFF) + PV(TV);  Equity = EV - Net debt;  Value/share = Equity / Shares
```

**Verdict:** upside above +band is *undervalued*, below -band is *overvalued*, otherwise *fairly valued* (default band ±15%).

**Monte Carlo:** each key input is drawn independently from a normal distribution centred on the base case (standard deviations are set in the sidebar), clipped to sensible bounds, with terminal growth always kept below WACC.

**Relative valuation:** P/E value = peer multiple x company TTM EPS. EV/EBITDA value = (peer multiple x company EBITDA - net debt) / shares. Negative or missing peer multiples are ignored.

---

## Default Assumptions

| Input | Default |
|---|---|
| Risk-free rate | 4.2% (US-style; use roughly 7% for India) |
| Equity risk premium | 5.0% |
| Pre-tax cost of debt | 5.5% |
| Tax rate | 3-year average effective rate from reported data |
| Terminal growth | 2.5% |
| Target EBIT margin | 3-year average margin |
| Exit multiple | Company's own EV/EBITDA, capped 6-25x |
| Scenarios | Bear -3pp growth, -2pp margin, +1pp WACC, -0.5pp g, -2x multiple (bull mirrors it); 25 / 50 / 25 weights |
| Monte Carlo | 5,000 runs; sd 2pp growth, 1.5pp margin, 0.75pp WACC, 0.4pp g, 10% multiple |

---

## Limitations

- **Terminal value dominates** most DCFs. Use the cross-check and sensitivity tab.
- **Not for every business.** Banks, insurers and early-stage firms need different models (for example excess returns or dividend discount).
- **Monte Carlo inputs are independent.** Real drivers are correlated, so the spread is indicative.
- **Peer sets are user-defined.** Peer quality drives the relative valuation; Yahoo Finance multiples can be stale or missing.
- **Currency mismatches** (for example ADRs) are flagged but not converted.
- **Excel export contains values, not live formulas.**
- **Data quality.** Yahoo Finance data may be delayed, incomplete or rate-limited.

---

## Tech Stack

[Streamlit](https://streamlit.io), [yfinance](https://github.com/ranaroussi/yfinance), [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org), [Plotly](https://plotly.com/python/), [openpyxl](https://openpyxl.readthedocs.io), [ReportLab](https://www.reportlab.com), [Matplotlib](https://matplotlib.org)

---

## Possible Next Steps

- Correlated Monte Carlo inputs
- Sector-specific models (banks, REITs)
- Automatic peer discovery
- Live-formula Excel model
- Historical multiple bands for the company itself
