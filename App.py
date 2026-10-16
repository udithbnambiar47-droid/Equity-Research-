"""
Equity Research App: DCF, scenarios, Monte Carlo and relative valuation.

Run locally:  streamlit run app.py
Educational use only. Not investment advice.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as eng
import export as ex
import market_data as md

st.set_page_config(page_title="DCF Equity Research", page_icon="📈", layout="wide")

NAVY, BLUE, RED, GREY = "#1F3A5F", "#3B6FB6", "#C0392B", "#8A94A6"

st.title("📈 Equity Research: DCF, Scenarios & Relative Valuation")
st.caption("Revenue and margin driven DCF with exit-multiple cross-check, bull/base/bear scenarios, "
           "Monte Carlo and peer multiples. Educational use only, not investment advice.")

# ==========================================================================
# Sidebar: stock, forecast, discount rate, simulation, peers
# ==========================================================================
with st.sidebar:
    st.header("1. Stock")
    ticker = st.text_input("Ticker symbol", value="AAPL",
                           help="US: AAPL | NSE: TCS.NS | BSE: TCS.BO | London: VOD.L").strip().upper()
    if not ticker:
        st.stop()
    K = lambda s: f"{ticker}:{s}"  # noqa: E731  widget keys reset when the ticker changes

    with st.spinner(f"Fetching {ticker}..."):
        try:
            company = md.load_company(ticker)
        except Exception as e:  # network / rate limit
            st.error(f"Could not fetch data: {e}")
            st.stop()

    info = company["info"]
    d = md.derive(company)
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    shares = info.get("sharesOutstanding")
    if not price or not shares:
        st.error("Price or share count missing for this ticker. Check the symbol / exchange suffix.")
        st.stop()

    model_choice = st.radio("Forecast model", ["Revenue & margin driven", "FCF growth (fallback)"],
                            help="Revenue & margin is the default. The fallback projects reported free "
                                 "cash flow directly, for companies with missing income-statement data.")
    fallback = model_choice.startswith("FCF")
    if not fallback and not d.has_operating:
        st.warning("Revenue / EBIT history unavailable, using the FCF-growth fallback.")
        fallback = True
    if fallback and d.fcf_hist.empty:
        st.error("No usable financial history for this ticker.")
        st.stop()

    st.header("2. Forecast")
    years = st.slider("Projection years", 3, 15, 10)
    tv_label = st.selectbox(
        "Terminal value method",
        ["Gordon growth", "Exit multiple (EV/EBITDA)", "Blend (average of both)"],
        disabled=fallback,
        help="The other method is always shown as a cross-check.")
    tv_method = "gordon" if fallback else {"Gordon": "gordon", "Exit": "exit", "Blend": "blend"}[tv_label.split()[0]]
    terminal_g = st.slider("Terminal growth (%)", 0.0, 6.0, 2.5, 0.1) / 100
    own_mult = info.get("enterpriseToEbitda")
    mult_default = float(np.clip(own_mult, 6, 25)) if isinstance(own_mult, (int, float)) and own_mult > 0 else 12.0
    exit_mult = st.number_input("Exit EV/EBITDA multiple (x)", 1.0, 60.0, round(mult_default, 1), 0.5,
                                key=K("mult"), disabled=fallback,
                                help="Defaults to the company's current multiple, capped 6-25x.")

    st.header("3. Discount rate")
    wacc_mode = st.radio("WACC", ["Calculate (CAPM)", "Manual"], horizontal=True)
    rf = st.number_input("Risk-free rate (%)", 0.0, 15.0, 4.2, 0.1) / 100
    erp = st.number_input("Equity risk premium (%)", 0.0, 12.0, 5.0, 0.1) / 100
    cost_debt = st.number_input("Pre-tax cost of debt (%)", 0.0, 20.0, 5.5, 0.1) / 100
    tax = st.number_input("Tax rate (%)", 0.0, 50.0, round(d.tax * 100, 1), 0.5, key=K("tax")) / 100
    manual_wacc = st.slider("Manual WACC (%)", 4.0, 20.0, 9.0, 0.1) / 100

    st.header("4. Decision rule")
    band = st.slider("Margin of safety band (±%)", 5, 40, 15,
                     help="Inside this band the stock is 'fairly valued'.") / 100

    st.header("5. Monte Carlo")
    n_sims = st.select_slider("Simulations", [1000, 2000, 5000, 10000, 20000], value=5000)
    seed = st.number_input("Random seed", 0, 99999, 42, 1)
    with st.expander("Input uncertainty (1 st. dev.)"):
        sd = {
            "growth": st.number_input("Revenue growth (pp)", 0.0, 15.0, 2.0, 0.5) / 100,
            "margin": st.number_input("Target margin (pp)", 0.0, 15.0, 1.5, 0.5) / 100,
            "wacc": st.number_input("WACC (pp)", 0.0, 4.0, 0.75, 0.25) / 100,
            "tg": st.number_input("Terminal growth (pp)", 0.0, 2.0, 0.4, 0.1) / 100,
            "mult_rel": st.number_input("Exit multiple (% of base)", 0.0, 50.0, 10.0, 1.0) / 100,
        }
    if fallback:
        sd["margin"] = 0.0
        sd["mult_rel"] = 0.0

    st.header("6. Peers")
    peers_txt = st.text_area("Peer tickers (comma separated)", key=K("peers"), height=80,
                             value=", ".join(md.DEFAULT_PEERS.get(ticker, [])),
                             help="Used for P/E and EV/EBITDA relative valuation.")

# ==========================================================================
# Company header
# ==========================================================================
currency = info.get("currency", "USD")
cur = {"USD": "$", "INR": "₹", "GBP": "£", "EUR": "€", "JPY": "¥"}.get(currency, currency + " ")
name = info.get("longName") or info.get("shortName") or ticker
beta = float(info.get("beta") or 1.0)
mcap = float(info.get("marketCap") or price * shares)
debt = float(info.get("totalDebt") or 0)
cash = float(info.get("totalCash") or 0)
net_debt = debt - cash
fin_cur = info.get("financialCurrency")

st.subheader(f"{name} ({ticker})")
if fin_cur and fin_cur != currency:
    st.warning(f"Financial statements are in {fin_cur} but the share price is in {currency}. "
               "Convert inputs to one currency before trusting the valuation.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current price", f"{cur}{price:,.2f}")
m2.metric("Market cap", eng.fmt_big(mcap, cur))
m3.metric("Beta", f"{beta:.2f}")
m4.metric("Net debt", eng.fmt_big(net_debt, cur))

# ==========================================================================
# Operating assumptions
# ==========================================================================
with st.expander("Operating assumptions", expanded=True):
    if fallback:
        c1, c2 = st.columns(2)
        base_fcf = float(d.fcf_hist.iloc[-1])
        rev0 = c1.number_input(f"Base free cash flow ({cur}, FY{d.fcf_year})", value=base_fcf,
                               step=max(abs(base_fcf) * 0.01, 1.0), format="%.0f", key=K("fcf0"))
        hist_g = None
        pos = d.fcf_hist[d.fcf_hist > 0]
        if len(pos) >= 2 and pos.index[-1] > pos.index[0]:
            hist_g = (pos.iloc[-1] / pos.iloc[0]) ** (1 / (pos.index[-1] - pos.index[0])) - 1
        g_def = float(np.clip(hist_g if hist_g is not None else 0.08, -0.05, 0.30))
        growth = c2.slider("FCF growth, year 1 (%)", -10.0, 40.0, round(g_def * 100, 1), 0.5, key=K("g")) / 100
        if rev0 <= 0:
            st.warning("Base FCF is zero or negative; a DCF is unreliable. Override with a normalised figure.")
        a = eng.Assumptions(rev0=rev0, growth_start=growth, margin_start=1.0, margin_target=1.0, margin_years=1,
                            tax=0.0, da_pct=0.0, capex_pct=0.0, nwc_pct=0.0, wacc=0.0, terminal_g=terminal_g,
                            exit_multiple=exit_mult, years=years, net_debt=net_debt, shares=float(shares),
                            tv_method="gordon")
    else:
        c1, c2, c3 = st.columns(3)
        rev0 = c1.number_input(f"Base revenue ({cur}, latest FY)", value=float(d.rev0),
                               step=max(abs(d.rev0) * 0.01, 1.0), format="%.0f", key=K("rev0"))
        growth = c1.slider("Revenue growth, year 1 (%)", -10.0, 40.0, round(d.growth * 100, 1), 0.5,
                           key=K("g"), help="Fades linearly to the terminal growth rate.") / 100
        tgt_def = float(np.clip(d.margin_target * 100, -10.0, 60.0))
        margin_target = c2.slider("Target EBIT margin (%)", -10.0, 60.0, round(tgt_def, 1), 0.5, key=K("mt"),
                                  help="Defaults to the 3-year average margin.") / 100
        c2.caption(f"Current EBIT margin: {d.margin_now:.1%}")
        margin_years = c2.slider("Years to reach target margin", 1, 15, min(5, years), key=K("my"))
        da_pct = c3.number_input("D&A (% of revenue)", 0.0, 30.0, round(d.da_pct * 100, 1), 0.1, key=K("da")) / 100
        capex_pct = c3.number_input("Capex (% of revenue)", 0.0, 40.0, round(d.capex_pct * 100, 1), 0.1,
                                    key=K("cx")) / 100
        nwc_pct = c3.number_input("NWC investment (% of revenue change)", 0.0, 50.0, round(d.nwc_pct * 100, 1),
                                  0.5, key=K("nwc")) / 100
        if capex_pct < da_pct:
            st.warning("Capex is below D&A. Final-year FCF (used for terminal value) may be overstated.")
        a = eng.Assumptions(rev0=rev0, growth_start=growth, margin_start=d.margin_now,
                            margin_target=margin_target, margin_years=margin_years, tax=tax, da_pct=da_pct,
                            capex_pct=capex_pct, nwc_pct=nwc_pct, wacc=0.0, terminal_g=terminal_g,
                            exit_multiple=exit_mult, years=years, net_debt=net_debt, shares=float(shares),
                            tv_method=tv_method)

# ---- WACC ---------------------------------------------------------------
if wacc_mode.startswith("Calculate"):
    wacc, ke, kd, we, wd = eng.compute_wacc(rf, beta, erp, cost_debt, tax, mcap, debt)
    with st.expander("WACC build-up"):
        st.write(f"Cost of equity (CAPM): **{ke:.2%}** = {rf:.2%} + {beta:.2f} × {erp:.2%}")
        st.write(f"After-tax cost of debt: **{kd:.2%}**; weights: equity **{we:.1%}**, debt **{wd:.1%}**")
        st.write(f"WACC: **{wacc:.2%}**")
else:
    wacc = manual_wacc
if wacc <= terminal_g:
    st.error("WACC must exceed terminal growth. Adjust the sidebar inputs.")
    st.stop()
a = replace(a, wacc=wacc)

# ==========================================================================
# Base case
# ==========================================================================
base = eng.run_dcf(a)
iv = float(base["per_share"][0])
upside = iv / price - 1
if upside > band:
    verdict, vcolor, vicon, vhex = "UNDERVALUED", "green", "🟢", "#1E8449"
elif upside < -band:
    verdict, vcolor, vicon, vhex = "OVERVALUED", "red", "🔴", "#C0392B"
else:
    verdict, vcolor, vicon, vhex = "FAIRLY VALUED", "orange", "🟡", "#B9770E"

st.divider()
v1, v2, v3 = st.columns([2, 1, 1])
v1.markdown(f"### {vicon} Base-case verdict: :{vcolor}[{verdict}]")
v2.metric("Intrinsic value / share", f"{cur}{iv:,.2f}")
v3.metric("Upside / (downside)", f"{upside:+.1%}", delta=f"{cur}{iv - price:,.2f}")
if iv <= 0:
    st.warning("Intrinsic value is negative; cash flows or leverage make this DCF unreliable.")

tabs = st.tabs(["Valuation", "Forecast", "Scenarios", "Monte Carlo",
                "Relative valuation", "Sensitivity", "Export"])
t_val, t_fc, t_scn, t_mc, t_rel, t_sen, t_exp = tabs
last_fy = int(d.rev_hist.index[-1]) if (not fallback and len(d.rev_hist)) else int(d.fcf_year)
year_labels = [f"{last_fy + i}E" for i in range(1, years + 1)]

# ==========================================================================
# Forecast tab
# ==========================================================================
with t_fc:
    if fallback:
        fc = pd.DataFrame({
            "Year": year_labels, "Growth": base["growth"][0], "FCF": base["fcff"][0],
            "Discount factor": 1 / base["disc"][0], "PV of FCF": base["pv"][0]})
        fc_spec = {"Growth": "pct", "FCF": "big", "Discount factor": "num", "PV of FCF": "big"}
    else:
        fc = pd.DataFrame({
            "Year": year_labels, "Revenue": base["rev"][0], "Growth": base["growth"][0],
            "EBITDA": base["ebitda"][0], "EBIT": base["ebit"][0], "EBIT margin": base["margin"][0],
            "Capex": base["capex"][0], "NWC change": base["d_nwc"][0], "FCFF": base["fcff"][0],
            "PV of FCFF": base["pv"][0]})
        fc_spec = {"Revenue": "big", "Growth": "pct", "EBITDA": "big", "EBIT": "big", "EBIT margin": "pct",
                   "Capex": "big", "NWC change": "big", "FCFF": "big", "PV of FCFF": "big"}
    st.dataframe(ex.format_df(fc, fc_spec, cur), hide_index=True)

    if not fallback:
        fig = go.Figure()
        hx = [str(y) for y in d.rev_hist.index]
        fig.add_bar(x=hx, y=d.rev_hist.values, name="Revenue (reported)", marker_color=GREY)
        fig.add_bar(x=year_labels, y=base["rev"][0], name="Revenue (forecast)", marker_color=BLUE)
        mh = d.margin_hist.reindex(d.rev_hist.index)
        fig.add_scatter(x=hx + year_labels, y=list(mh.values) + list(base["margin"][0]), name="EBIT margin",
                        yaxis="y2", mode="lines+markers", line=dict(color=RED))
        fig.update_layout(height=380, title="Revenue and EBIT margin", yaxis_title=f"Revenue ({currency})",
                          yaxis2=dict(title="EBIT margin", overlaying="y", side="right", tickformat=".0%"),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig)

    fig2 = go.Figure()
    if len(d.fcf_hist):
        fig2.add_bar(x=[str(y) for y in d.fcf_hist.index], y=d.fcf_hist.values, name="FCF (reported)",
                     marker_color=GREY)
    fig2.add_bar(x=year_labels, y=base["fcff"][0], name="FCF (forecast)", marker_color=BLUE)
    fig2.update_layout(height=340, title="Free cash flow: reported vs forecast (sanity check)",
                       yaxis_title=f"FCF ({currency})", barmode="group", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig2)
    if not fallback:
        st.caption("FCFF = EBIT × (1 − tax) + D&A − capex − investment in NWC. If forecast FCF jumps far away "
                   "from reported FCF in year 1, revisit the margin, capex or NWC assumptions.")

# ==========================================================================
# Scenarios tab
# ==========================================================================
with t_scn:
    st.write("Edit the shifts applied to the base case. Probabilities are normalised to 100%.")
    C_G, C_M, C_W, C_T, C_X, C_P = ("Revenue growth Δ (pp)", "Target margin Δ (pp)", "WACC Δ (pp)",
                                    "Terminal g Δ (pp)", "Exit multiple Δ (x)", "Probability (%)")
    default_scn = pd.DataFrame({
        "Scenario": ["Bear", "Base", "Bull"], C_G: [-3.0, 0.0, 3.0], C_M: [-2.0, 0.0, 2.0],
        C_W: [1.0, 0.0, -1.0], C_T: [-0.5, 0.0, 0.5], C_X: [-2.0, 0.0, 2.0], C_P: [25.0, 50.0, 25.0]})
    cc = {c: st.column_config.NumberColumn(c, format="%.1f", step=0.5) for c in default_scn.columns[1:]}
    cc["Scenario"] = st.column_config.TextColumn("Scenario", disabled=True)
    scn_in = st.data_editor(default_scn, hide_index=True, column_config=cc, num_rows="fixed", key=K("scn"))
    if fallback:
        st.caption("Fallback mode: margin and exit-multiple shifts have no effect.")

    rows = []
    for _, r in scn_in.iterrows():
        sa = eng.apply_scenario(a, r[C_G] / 100, 0.0 if fallback else r[C_M] / 100, r[C_W] / 100,
                                r[C_T] / 100, r[C_X])
        rs = eng.run_dcf(sa)
        v = float(rs["per_share"][0])
        rows.append({"Scenario": r["Scenario"], "Probability": float(r[C_P]), "Revenue growth Y1": sa.growth_start,
                     "Target margin": sa.margin_target, "WACC": sa.wacc, "Terminal g": sa.terminal_g,
                     "Exit multiple": sa.exit_multiple, "Value / share": v, "Upside": v / price - 1})
    scn = pd.DataFrame(rows)
    pw = scn["Probability"].where(scn["Value / share"].notna(), 0.0)
    weighted = float((scn["Value / share"].fillna(0) * pw).sum() / pw.sum()) if pw.sum() > 0 else np.nan
    scn["Probability"] = scn["Probability"] / scn["Probability"].sum()
    scn_spec = {"Probability": "pct", "Revenue growth Y1": "pct", "Target margin": "pct", "WACC": "pct",
                "Terminal g": "pct", "Exit multiple": "x", "Value / share": "price", "Upside": "pct"}
    shown = scn.drop(columns=["Target margin", "Exit multiple"]) if fallback else scn
    st.dataframe(ex.format_df(shown, scn_spec, cur), hide_index=True)

    s1, s2 = st.columns(2)
    s1.metric("Probability-weighted value", f"{cur}{weighted:,.2f}" if np.isfinite(weighted) else "n/a",
              delta=f"{weighted / price - 1:+.1%} vs price" if np.isfinite(weighted) else None)
    s2.metric("Current price", f"{cur}{price:,.2f}")
    figs = go.Figure(go.Bar(x=scn["Scenario"], y=scn["Value / share"],
                            marker_color=[RED if (x or 0) < price else BLUE for x in scn["Value / share"]],
                            text=[f"{cur}{x:,.2f}" if np.isfinite(x) else "n/a" for x in scn["Value / share"]],
                            textposition="outside"))
    figs.add_hline(y=price, line_dash="dash", line_color=NAVY, annotation_text=f"Price {cur}{price:,.2f}")
    figs.update_layout(height=360, yaxis_title=f"Value per share ({currency})", showlegend=False)
    st.plotly_chart(figs)

scn_vals = scn["Value / share"].dropna()

# ==========================================================================
# Monte Carlo tab
# ==========================================================================
with t_mc:
    samples = eng.monte_carlo(a, int(n_sims), int(seed), sd)
    if samples.size < 100:
        st.error("Too few valid simulations; widen WACC vs terminal growth headroom.")
        mc = None
    else:
        mc = eng.mc_summary(samples, price, band)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median value / share", f"{cur}{mc['pct'][50]:,.2f}")
        k2.metric("Mean value / share", f"{cur}{mc['mean']:,.2f}")
        k3.metric("P(value > price)", f"{mc['p_above']:.0%}")
        k4.metric(f"P(upside > {band:.0%})", f"{mc['p_above_band']:.0%}")

        lo, hi = np.percentile(samples, [1, 99])
        figm = go.Figure(go.Histogram(x=np.clip(samples, lo, hi), nbinsx=60, marker_color=BLUE))
        figm.add_vline(x=price, line_dash="dash", line_color=RED, annotation_text=f"Price {cur}{price:,.2f}")
        figm.add_vline(x=mc["pct"][50], line_color=NAVY, annotation_text="Median", annotation_position="top left")
        figm.update_layout(height=380, xaxis_title=f"Intrinsic value per share ({currency}), clipped at P1-P99",
                           yaxis_title="Simulations", showlegend=False)
        st.plotly_chart(figm)

        mc_tbl = pd.DataFrame({"Statistic": ["P5", "P10", "P25", "Median", "P75", "P90", "P95", "Mean", "Std. dev."],
                               "Value / share": [mc["pct"][p] for p in (5, 10, 25, 50, 75, 90, 95)]
                               + [mc["mean"], mc["std"]]})
        mc_spec = {"Value / share": "price"}
        st.dataframe(ex.format_df(mc_tbl, mc_spec, cur), hide_index=True)
        st.caption(f"{mc['n']:,} valid simulations. Inputs are drawn independently from normal distributions "
                   "around the base case and clipped to sane bounds; real-world inputs are correlated "
                   "(for example growth and margins), so treat the spread as indicative.")

# ==========================================================================
# Relative valuation tab
# ==========================================================================
peer_list = [p for p in dict.fromkeys(re.split(r"[,\s]+", peers_txt.upper())) if p and p != ticker][:10]
rel_rows, rel_tbl, peers_df = [], None, None
with t_rel:
    if not peer_list:
        st.info("Add peer tickers in the sidebar (section 6) to value the company on peer P/E and EV/EBITDA.")
    else:
        with st.spinner("Fetching peer multiples..."):
            peers_df = md.load_multiples(tuple(peer_list))
        own = pd.DataFrame([{"Ticker": ticker, "Name": name[:28], "Market cap": mcap,
                             "P/E (TTM)": info.get("trailingPE"), "P/E (Fwd)": info.get("forwardPE"),
                             "EV/EBITDA": info.get("enterpriseToEbitda"), "EV/Revenue": info.get("enterpriseToRevenue")}])
        for c in own.columns[2:]:
            own[c] = pd.to_numeric(own[c], errors="coerce")
        peer_spec = {"Market cap": "big", "P/E (TTM)": "x", "P/E (Fwd)": "x", "EV/EBITDA": "x", "EV/Revenue": "x"}
        st.dataframe(ex.format_df(pd.concat([own, peers_df], ignore_index=True), peer_spec, cur), hide_index=True)
        st.caption("First row is the company itself. Negative or missing multiples are ignored in the statistics.")

        def q(col, p):
            s = peers_df[col].replace([np.inf, -np.inf], np.nan)
            s = s[s > 0].dropna()
            return float(np.percentile(s, p)) if len(s) else np.nan

        eps = d.eps
        ebitda = d.ebitda_ttm
        for label, col, kind in (("P/E (TTM)", "P/E (TTM)", "pe"), ("EV/EBITDA", "EV/EBITDA", "ev")):
            lo_m, md_m, hi_m = q(col, 25), q(col, 50), q(col, 75)
            vals = [np.nan] * 3
            if kind == "pe" and np.isfinite(eps) and eps > 0:
                vals = [m * eps for m in (lo_m, md_m, hi_m)]
            if kind == "ev" and np.isfinite(ebitda) and ebitda > 0:
                vals = [(m * ebitda - net_debt) / shares for m in (lo_m, md_m, hi_m)]
            rel_rows.append({"Method": label, "Peer P25": lo_m, "Peer median": md_m, "Peer P75": hi_m,
                             "Value @ P25": vals[0], "Value @ median": vals[1], "Value @ P75": vals[2]})
        rel_tbl = pd.DataFrame(rel_rows)
        rel_spec = {"Peer P25": "x", "Peer median": "x", "Peer P75": "x",
                    "Value @ P25": "price", "Value @ median": "price", "Value @ P75": "price"}
        st.markdown("**Implied value per share from peer multiples**")
        st.dataframe(ex.format_df(rel_tbl, rel_spec, cur), hide_index=True)
        st.caption(f"Company TTM EPS: {eps:,.2f}" if np.isfinite(eps) else "Company EPS unavailable (or negative).")
        if not (np.isfinite(ebitda) and ebitda > 0):
            st.caption("Company EBITDA unavailable or negative, so EV/EBITDA value is not computed.")

# ==========================================================================
# Sensitivity tab
# ==========================================================================
with t_sen:
    axis_label = st.radio("Column axis", ["Terminal growth", "Exit multiple"], horizontal=True,
                          index=1 if tv_method == "exit" else 0, disabled=fallback)
    axis = "tg" if (fallback or axis_label == "Terminal growth") else "mult"
    waccs, cols, grid = eng.sensitivity(a, axis)
    col_names = [f"{c:.1%}" if axis == "tg" else f"{c:.1f}x" for c in cols]
    sens = pd.DataFrame(grid, columns=col_names)
    sens.insert(0, "WACC", waccs)
    sens_spec = {"WACC": "pct", **{c: "price" for c in col_names}}
    if tv_method == "gordon" and axis == "mult":
        st.info("Terminal value uses Gordon growth, so the exit multiple has no effect. Switch the method "
                "or the axis.")
    if tv_method == "exit" and axis == "tg":
        st.info("Terminal value uses the exit multiple, so terminal growth has no effect. Switch the axis.")

    def shade(v):
        if pd.isna(v):
            return ""
        return ("background-color:#d4edda;color:#155724" if v > price else "background-color:#f8d7da;color:#721c24")

    styled = (sens.style.format({c: f"{cur}{{:,.2f}}" for c in col_names} | {"WACC": "{:.1%}"})
              .map(shade, subset=col_names).hide(axis="index"))
    st.dataframe(styled)
    st.caption(f"Value per share. Green = above the current price ({cur}{price:,.2f}), red = below.")

# ==========================================================================
# Valuation tab (needs everything above)
# ==========================================================================
football = []
if len(scn_vals):
    football.append(("DCF: bear to bull", float(scn_vals.min()), iv, float(scn_vals.max())))
if mc is not None:
    football.append(("Monte Carlo P10-P90", mc["pct"][10], mc["pct"][50], mc["pct"][90]))
for r in rel_rows:
    if np.isfinite(r["Value @ P25"]) and np.isfinite(r["Value @ P75"]):
        football.append((f"Peers {r['Method']} (P25-P75)", r["Value @ P25"], r["Value @ median"], r["Value @ P75"]))
lo52, hi52 = info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh")
if lo52 and hi52:
    football.append(("52-week range", float(lo52), price, float(hi52)))

with t_val:
    bridge = pd.DataFrame({
        "Component": ["PV of forecast cash flows", "PV of terminal value", "Enterprise value", "Less: net debt",
                      "Equity value", "Shares outstanding", "Intrinsic value / share"],
        "Value": [base["pv_sum"][0], base["pv_tv"][0], base["ev"][0], -net_debt, base["equity"][0],
                  float(shares), iv]})
    brs = ex.format_df(bridge, {"Value": "big"}, cur)
    brs.loc[brs["Component"] == "Intrinsic value / share", "Value"] = f"{cur}{iv:,.2f}"
    st.markdown("**Enterprise to equity bridge**")
    st.dataframe(brs, hide_index=True)
    st.caption(f"Terminal value is {base['tv_share'][0]:.0%} of enterprise value; a high share means the "
               "result leans heavily on long-run assumptions.")

    xc = None
    if fallback:
        st.info("Exit-multiple cross-check needs EBITDA, so it is available in the revenue & margin model.")
    else:
        rg = eng.run_dcf(replace(a, tv_method="gordon"))
        rx = eng.run_dcf(replace(a, tv_method="exit"))
        vg, vx = float(rg["per_share"][0]), float(rx["per_share"][0])
        xc = pd.DataFrame({
            "Terminal value method": ["Gordon growth", "Exit multiple (EV/EBITDA)"],
            "Value / share": [vg, vx], "vs price": [vg / price - 1, vx / price - 1],
            "TV % of EV": [float(rg["tv_share"][0]), float(rx["tv_share"][0])],
            "Terminal growth": [a.terminal_g, float(rx["implied_g"][0])],
            "Exit EV/EBITDA": [float(rg["implied_mult"][0]), a.exit_multiple]})
        xc_spec = {"Value / share": "price", "vs price": "pct", "TV % of EV": "pct",
                   "Terminal growth": "pct", "Exit EV/EBITDA": "x"}
        st.markdown("**Terminal value cross-check**")
        st.dataframe(ex.format_df(xc, xc_spec, cur), hide_index=True)
        st.caption("Gordon row shows the exit multiple its terminal value implies; the exit-multiple row shows "
                   "the perpetual growth rate its terminal value implies.")
        gap = abs(vx / vg - 1) if vg else np.nan
        ig = float(rx["implied_g"][0])
        if np.isfinite(ig) and ig > min(wacc - 0.01, 0.05):
            st.warning(f"The exit multiple implies perpetual growth of {ig:.1%}, which is aggressive. "
                       "Consider a lower multiple.")
        elif np.isfinite(gap) and gap > 0.25:
            st.warning(f"The two terminal value methods differ by {gap:.0%}. Revisit the exit multiple or "
                       "terminal growth until they are reasonably consistent.")

    if football:
        st.markdown("**Valuation summary (football field)**")
        figf = go.Figure()
        for label, lo, mid, hi in football:
            figf.add_bar(y=[label], x=[hi - lo], base=[lo], orientation="h", marker_color=BLUE, showlegend=False,
                         hovertemplate=f"{label}<br>{cur}{lo:,.2f} to {cur}{hi:,.2f}<extra></extra>")
            figf.add_scatter(x=[mid], y=[label], mode="markers", showlegend=False,
                             marker=dict(symbol="line-ns", size=22, line=dict(width=3, color="white")),
                             hovertemplate=f"Mid {cur}{mid:,.2f}<extra></extra>")
        figf.add_vline(x=price, line_dash="dash", line_color=RED, annotation_text=f"Price {cur}{price:,.2f}")
        figf.update_layout(height=120 + 55 * len(football), xaxis_title=f"Value per share ({currency})",
                           yaxis=dict(autorange="reversed"), bargap=0.45)
        st.plotly_chart(figf)

# ==========================================================================
# Export tab
# ==========================================================================
with t_exp:
    sig = (ticker, round(iv, 4), int(n_sims), int(seed), fallback, tv_method, round(wacc, 5))
    if st.button("Generate Excel and PDF report", type="primary"):
        assump_rows = [("Forecast model", "FCF growth (fallback)" if fallback else "Revenue & margin"),
                       ("Projection years", years), ("Base " + ("FCF" if fallback else "revenue"), eng.fmt_big(a.rev0, cur)),
                       ("Growth, year 1", f"{a.growth_start:.1%}"), ("Terminal growth", f"{a.terminal_g:.1%}"),
                       ("WACC", f"{wacc:.2%}"), ("Tax rate", f"{tax:.1%}")]
        if not fallback:
            assump_rows += [("Current EBIT margin", f"{a.margin_start:.1%}"),
                            ("Target EBIT margin", f"{a.margin_target:.1%} in {a.margin_years} yrs"),
                            ("D&A % revenue", f"{a.da_pct:.1%}"), ("Capex % revenue", f"{a.capex_pct:.1%}"),
                            ("NWC % of revenue change", f"{a.nwc_pct:.1%}"),
                            ("Terminal value method", tv_label), ("Exit EV/EBITDA", f"{a.exit_multiple:.1f}x")]
        assump_rows += [("Net debt", eng.fmt_big(net_debt, cur)), ("Shares outstanding", eng.fmt_big(float(shares))),
                        ("Beta", f"{beta:.2f}"), ("Margin-of-safety band", f"±{band:.0%}")]

        summary = [("Company", f"{name} ({ticker})"), ("Current price", f"{cur}{price:,.2f}"),
                   ("Base-case intrinsic value / share", f"{cur}{iv:,.2f}"), ("Upside / (downside)", f"{upside:+.1%}"),
                   ("Verdict", verdict)]
        if np.isfinite(weighted):
            summary.append(("Probability-weighted scenario value", f"{cur}{weighted:,.2f}"))
        if len(scn_vals):
            summary.append(("Bear / Bull value", f"{cur}{scn_vals.min():,.2f} / {cur}{scn_vals.max():,.2f}"))
        if mc is not None:
            summary += [("Monte Carlo median", f"{cur}{mc['pct'][50]:,.2f}"),
                        ("Monte Carlo P(value > price)", f"{mc['p_above']:.0%}")]
        for r in rel_rows:
            if np.isfinite(r["Value @ median"]):
                summary.append((f"Peer {r['Method']} implied value", f"{cur}{r['Value @ median']:,.2f}"))

        tables = [
            {"name": "Assumptions", "df": pd.DataFrame(assump_rows, columns=["Parameter", "Value"]),
             "spec": {}, "pdf": True},
            {"name": "Projections", "df": fc, "spec": fc_spec, "pdf": True},
            {"name": "Scenarios", "df": scn, "spec": scn_spec, "pdf": True},
        ]
        if xc is not None:
            tables.append({"name": "TV cross-check", "df": xc, "spec": xc_spec, "pdf": True})
        tables.append({"name": "Sensitivity", "df": sens, "spec": sens_spec, "pdf": True})
        if mc is not None:
            tables.append({"name": "Monte Carlo", "df": mc_tbl, "spec": mc_spec, "pdf": True})
        if rel_tbl is not None:
            tables.append({"name": "Relative valuation", "df": rel_tbl, "spec": rel_spec, "pdf": True})
            tables.append({"name": "Peer multiples", "df": peers_df, "spec": peer_spec, "pdf": True})
        if mc is not None:
            tables.append({"name": "MC samples", "df": pd.DataFrame({"Value / share": samples[:5000]}),
                           "spec": {"Value / share": "price"}, "pdf": False})

        report = {"title": "DCF Equity Valuation Report", "ticker": ticker, "name": name, "currency": currency,
                  "date": date.today().isoformat(), "price": float(price), "verdict": verdict,
                  "verdict_color": vhex, "summary": summary, "tables": tables, "football": football,
                  "mc_samples": samples if mc is not None else None}
        with st.spinner("Building files..."):
            st.session_state["report_files"] = {
                "sig": sig, "xlsx": ex.build_excel(report), "pdf": ex.build_pdf(report)}

    files = st.session_state.get("report_files")
    if files is None:
        st.info("Click the button to build the files for the current inputs.")
    else:
        if files["sig"] != sig:
            st.warning("Inputs changed since these files were built. Regenerate to refresh them.")
        e1, e2 = st.columns(2)
        e1.download_button("⬇️ Download Excel (.xlsx)", files["xlsx"], file_name=f"{ticker}_dcf_report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        e2.download_button("⬇️ Download PDF", files["pdf"], file_name=f"{ticker}_dcf_report.pdf",
                           mime="application/pdf")
        st.caption("The Excel file holds values (not live formulas), with one sheet per table.")

st.divider()
st.caption("Data from Yahoo Finance via yfinance; it may be delayed or incomplete. DCF output is only as good "
           "as its assumptions. Use it alongside qualitative research. Not investment advice.")
