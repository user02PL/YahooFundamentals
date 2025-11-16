#!/usr/bin/env python3
"""
Compute BVPS, last EPS (diluted>basic), and last/TTM DPS for a list of tickers using yfinance.

Notes on annualization:
- If a flow comes from a quarterly statement (EPS, Revenue, OCF, FCF), multiply by 4.
- Balance-sheet items (BVPS/TBVPS/Working Capital/Cash) are not annualized.

Output CSV columns (order starts with):
ticker, price, p_target, dividend_yield, p_b, p_e, p_s, p_ocf, p_fcf, ...
"""

from pathlib import Path
from datetime import timedelta, datetime, timezone, date
import math
import time
import numpy as np
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo
# from util import file_name_from_time

def file_name_from_time(prefix=None, suffix=None):
    """Build a timestamped name 'YYYYMMDD_HHMMam/pm' (America/New_York), optionally adding prefix and suffix."""
    date_time_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%I%M%p").lower()
    if prefix:
        date_time_str = prefix + date_time_str
    if suffix:
        date_time_str = date_time_str + suffix
    return date_time_str

def read_tickers(path: str | None) -> list[str]:
    "Read tickers from a text file (one per line), else return a small default list."
    if path and Path(path).is_file():
        txt = Path(path).read_text().splitlines()
        return [ln.strip().upper() for ln in txt if ln.strip() and not ln.strip().startswith("#")]
    return ["AGNC", "NLY", "MFA", "AAPL"]

def _latest_from_frames(frames: tuple[pd.DataFrame | None, ...],
                        keys: tuple[str, ...]) -> tuple[pd.Timestamp | None, float | None, str | None]:
    "Return (most recent column date, value, period) where period is 'quarterly' or 'annual'."
    for idx, df in enumerate(frames):
        try:
            if df is None or df.empty:
                continue
            row = None
            for k in keys:
                if k in df.index:
                    row = df.loc[k]
                    break
            if row is None or row.dropna().empty:
                continue
            col = row.dropna().index[0]  # most-recent column first in yfinance
            period = "quarterly" if idx == 0 else "annual"
            return col, float(row[col]), period
        except Exception:
            continue
    return None, None, None

def latest_equity(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None]:
    "Return the most recent common or total equity value and its date."
    keys = (
        "Common Stock Equity",
        "Total Stockholder Equity",
        "Total Stockholders Equity",
        "Total Equity Gross Minority Interest",
    )
    d, v, _ = _latest_from_frames((tkr.balance_sheet, tkr.quarterly_balance_sheet), keys)
    return d, v

def latest_shares_outstanding(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None]:
    "Return the most recent shares outstanding and its date (point-in-time if available)."
    try:
        s = tkr.get_shares_full()
        if s is not None and not s.dropna().empty:
            s = s.dropna()
            last_date = s.index.max()
            return last_date, float(s.loc[last_date])
    except Exception:
        pass
    try:
        so = tkr.info.get("sharesOutstanding")
        if so:
            return None, float(so)
    except Exception:
        pass
    return None, None

def latest_eps(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None, str | None]:
    "Return the most recent EPS (prefer diluted, else basic) with its date and period."
    return _latest_from_frames((tkr.quarterly_income_stmt, tkr.income_stmt), ("Diluted EPS", "Basic EPS"))

def latest_and_ttm_dividends(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None, float | None]:
    "Return last dividend date, last dividend amount, and sum of dividends over the last 365 days."
    try:
        s = tkr.dividends
        if s is None or s.empty:
            return None, None, None
        s = s.dropna()
        last_date = s.index.max()
        last_amt = float(s.iloc[-1])
        cutoff = last_date - timedelta(days=365)
        ttm = float(s[s.index > cutoff].sum())
        return last_date, last_amt, ttm
    except Exception:
        return None, None, None

def latest_price(tkr: yf.Ticker) -> float | None:
    "Return a recent market price (fast_info, else last close, else regularMarketPrice)."
    try:
        fi = getattr(tkr, "fast_info", None)
        if fi is not None:
            p = getattr(fi, "last_price", None)
            if p is not None:
                return float(p)
    except Exception:
        pass
    try:
        hist = tkr.history(period="5d", interval="1d", auto_adjust=False)
        if hist is not None and not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    try:
        p = tkr.info.get("regularMarketPrice")
        if p is not None:
            return float(p)
    except Exception:
        pass
    return None

def latest_revenue(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None, str | None]:
    "Return the most recent total revenue value with its date and period."
    return _latest_from_frames((tkr.quarterly_income_stmt, tkr.income_stmt), ("Total Revenue", "Operating Revenue", "Revenue"))

def latest_operating_cf(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None, str | None]:
    "Return the most recent operating cash flow value with its date and period."
    return _latest_from_frames((tkr.quarterly_cashflow, tkr.cashflow), ("Operating Cash Flow",))

def latest_free_cf(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None, str | None]:
    "Return the most recent free cash flow value with its date and period."
    return _latest_from_frames((tkr.quarterly_cashflow, tkr.cashflow), ("Free Cash Flow",))

def latest_cash_equiv(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None]:
    "Return the most recent cash and equivalents value and its date."
    d, v, _ = _latest_from_frames(
        (tkr.balance_sheet, tkr.quarterly_balance_sheet),
        ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments"),
    )
    return d, v

def latest_goodwill(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None]:
    "Return the most recent goodwill value and its date."
    d, v, _ = _latest_from_frames((tkr.balance_sheet, tkr.quarterly_balance_sheet), ("Goodwill",))
    return d, v

def latest_intangibles(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None]:
    "Return the most recent intangible assets value and its date."
    d, v, _ = _latest_from_frames((tkr.balance_sheet, tkr.quarterly_balance_sheet), ("Intangible Assets", "Goodwill And Other Intangible Assets"))
    return d, v

def latest_current_assets(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None]:
    "Return the most recent current assets value (direct or derived) and its date."
    d, v, _ = _latest_from_frames((tkr.balance_sheet, tkr.quarterly_balance_sheet), ("Total Current Assets", "Current Assets"))
    if v is not None:
        return d, v
    _, total_assets, _ = _latest_from_frames((tkr.balance_sheet, tkr.quarterly_balance_sheet), ("Total Assets",))
    _, noncurrent_assets, _ = _latest_from_frames(
        (tkr.balance_sheet, tkr.quarterly_balance_sheet),
        ("Total Non Current Assets", "Total Noncurrent Assets", "Non Current Assets", "Noncurrent Assets"),
    )
    if (total_assets is not None) and (noncurrent_assets is not None):
        return None, float(total_assets - noncurrent_assets)
    return None, None

def latest_current_liabilities(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None]:
    "Return the most recent current liabilities value (direct or derived) and its date."
    d, v, _ = _latest_from_frames((tkr.balance_sheet, tkr.quarterly_balance_sheet), ("Total Current Liabilities", "Current Liabilities"))
    if v is not None:
        return d, v
    _, total_liab, _ = _latest_from_frames((tkr.balance_sheet, tkr.quarterly_balance_sheet), ("Total Liabilities Net Minority Interest", "Total Liabilities"))
    _, noncurrent_liab, _ = _latest_from_frames(
        (tkr.balance_sheet, tkr.quarterly_balance_sheet),
        ("Total Non Current Liabilities Net Minority Interest",
         "Total Noncurrent Liabilities Net Minority Interest",
         "Total Non Current Liabilities",
         "Total Noncurrent Liabilities",
         "Non Current Liabilities",
         "Noncurrent Liabilities"),
    )
    if (total_liab is not None) and (noncurrent_liab is not None):
        return None, float(total_liab - noncurrent_liab)
    return None, None

def latest_working_capital(tkr: yf.Ticker) -> tuple[pd.Timestamp | None, float | None]:
    "Return the most recent working capital value if available."
    d, v, _ = _latest_from_frames((tkr.balance_sheet, tkr.quarterly_balance_sheet), ("Working Capital", "Net Working Capital"))
    return d, v

def latest_beta(tkr: yf.Ticker) -> float | None:
    "Return Yahoo beta if available (any horizon)."
    try:
        info = tkr.info
        b = info.get("beta")
        if b is None:
            b = info.get("beta3Year") or info.get("beta5Year")
        return float(b) if b is not None else None
    except Exception:
        return None

def next_earnings_date(tkr: yf.Ticker):
    "Return the next earnings date as a future date, else None."
    today = datetime.now(timezone.utc).date()
    cutoff_epoch = 946684800  # 2000-01-01 UTC in seconds

    def _to_date(x):
        "Parse a variety of timestamp formats into a date; reject bad/ancient values."
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        if isinstance(x, (list, tuple)) and len(x) == 0:
            return None
        try:
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                if x <= 0 or x < cutoff_epoch:
                    return None
                return datetime.fromtimestamp(float(x), tz=timezone.utc).date()
        except Exception:
            pass
        try:
            dt = pd.to_datetime(x, utc=True, errors="coerce")
            if dt is None or pd.isna(dt):
                return None
            d = dt.date()
            if d < date(2000, 1, 1):
                return None
            return d
        except Exception:
            return None

    def _first_future(candidates):
        "Return the earliest candidate date that is today or later."
        cand = sorted({d for d in (_to_date(x) for x in candidates) if d is not None})
        for d in cand:
            if d >= today:
                return d
        return None

    try:
        fi = getattr(tkr, "fast_info", None)
        if fi is not None:
            d = _to_date(getattr(fi, "next_earnings_date", None))
            if d is not None and d >= today:
                return d
    except Exception:
        pass

    try:
        info = tkr.info
        ed = info.get("earningsDate")
        if isinstance(ed, (list, tuple)):
            d = _first_future(ed)
            if d:
                return d
        else:
            d = _to_date(ed)
            if d is not None and d >= today:
                return d
        for k in ("earningsTimestampStart", "earningsTimestampEnd", "earningsTimestamp"):
            d = _to_date(info.get(k))
            if d is not None and d >= today:
                return d
    except Exception:
        pass

    try:
        cal = tkr.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
            vals = cal.loc("Earnings Date") if callable(getattr(cal, "loc", None)) else None
            if vals is None:
                vals = cal.loc["Earnings Date"].tolist()
            else:
                vals = list(vals)
            d = _first_future(vals)
            if d:
                return d
    except Exception:
        pass

    try:
        df = tkr.get_earnings_dates(limit=24)
        if df is not None and not df.empty:
            idx_dates = [ts.date() for ts in df.index.to_list()]
            d = _first_future(idx_dates)
            if d:
                return d
    except Exception:
        pass

    return None

def forward_dividend(tkr: yf.Ticker) -> float | None:
    "Return forward annual dividend per share if available."
    try:
        val = tkr.info.get("dividendRate")
        return float(val) if val is not None else None
    except Exception:
        return None

def target_1y_est(tkr: yf.Ticker) -> float | None:
    "Return the analyst 1y target price estimate if available."
    try:
        info = tkr.info
        v = info.get("targetMeanPrice")
        if v is None:
            v = info.get("targetMean")
        return float(v) if v is not None else None
    except Exception:
        return None

def fundamentals_snapshot(ticker: str) -> dict:
    "Collect a per-ticker snapshot of per-share fundamentals, price, and related fields (flows annualized if quarterly)."
    t = yf.Ticker(ticker)
    d_eq, eq = latest_equity(t)
    d_sh, sh = latest_shares_outstanding(t)
    bvps = float(eq / sh) if (eq is not None and sh not in (None, 0)) else None

    d_eps, eps, eps_per   = latest_eps(t)
    d_div, dps_last, dps_ttm = latest_and_ttm_dividends(t)
    price = latest_price(t)

    _, rev, rev_per   = latest_revenue(t)
    _, ocf, ocf_per   = latest_operating_cf(t)
    _, fcf, fcf_per   = latest_free_cf(t)
    _, cash_eq        = latest_cash_equiv(t)
    _, gw             = latest_goodwill(t)
    _, ia             = latest_intangibles(t)
    _, ca             = latest_current_assets(t)
    _, cl             = latest_current_liabilities(t)
    _, wc_direct      = latest_working_capital(t)

    # annualize flows if they came from a quarterly statement
    def _ann(x, per): return None if x is None else (x * 4.0 if per == "quarterly" else x)
    eps_ann = _ann(eps, eps_per)
    rev_ann = _ann(rev, rev_per)
    ocf_ann = _ann(ocf, ocf_per)
    fcf_ann = _ann(fcf, fcf_per)

    beta = latest_beta(t)
    ned  = next_earnings_date(t)
    fdiv = forward_dividend(t)
    tgt  = target_1y_est(t)

    # profile fields
    try:
        info = t.info
    except Exception:
        info = {}
    company_name = None
    sector = None
    industry = None
    try:
        company_name = info.get("longName") or info.get("shortName") or None
        sector = info.get("sector") or None
        industry = info.get("industry") or None
    except Exception:
        pass

    if sh in (None, 0):
        rev_ps = ocf_ps = fcf_ps = tbvps = wc_ps = cash_ps = None
    else:
        rev_ps  = None if rev_ann  is None else float(rev_ann  / sh)
        ocf_ps  = None if ocf_ann  is None else float(ocf_ann  / sh)
        fcf_ps  = None if fcf_ann  is None else float(fcf_ann  / sh)
        cash_ps = None if cash_eq  is None else float(cash_eq  / sh)
        eq_tangible = None if eq is None else eq - (gw or 0.0) - (ia or 0.0)
        tbvps = None if (eq_tangible is None) else float(eq_tangible / sh)
        if (ca is not None) and (cl is not None):
            wc_val = ca - cl
        else:
            wc_val = wc_direct
        wc_ps = None if (wc_val is None) else float(wc_val / sh)

    return dict(
        ticker=ticker,
        date_equity=None if d_eq is None else pd.to_datetime(d_eq).date(),
        equity=eq,
        date_shares=None if d_sh is None else pd.to_datetime(d_sh).date(),
        shares=sh,
        bvps=bvps,
        eps_date=None if d_eps is None else pd.to_datetime(d_eps).date(),
        eps=eps,                  # raw latest EPS (may be quarterly)
        eps_ann=eps_ann,          # annualized EPS if quarterly (else same)
        dps_date=None if d_div is None else pd.to_datetime(d_div).date(),
        dps_last=dps_last,
        dps_ttm=dps_ttm,
        price=price,
        rev_ps=rev_ps,            # per-share flows are annualized where needed
        ocf_ps=ocf_ps,
        fcf_ps=fcf_ps,
        tbvps=tbvps,
        wc_ps=wc_ps,
        cash_ps=cash_ps,
        beta=beta,
        next_earnings_date=ned,
        forward_dividend=fdiv,
        target_1y_est=tgt,
        company_name=company_name,
        sector=sector,
        industry=industry,
    )

def main():
    "Entry point: load tickers, compute snapshots, add ratios, reorder columns, write CSV, and print a summary."
    start = time.perf_counter()
    print_null_fracs = True
    results_file = file_name_from_time(prefix="yahoo_fundamentals_", suffix=".csv")
    SYMBOLS_FILE = "dia_tickers.txt" # "itot_tickers_20251103.txt"
    MAX_STOCKS = 0
    
    tickers = read_tickers(SYMBOLS_FILE)
    if MAX_STOCKS and MAX_STOCKS > 0:
        tickers = tickers[:MAX_STOCKS]
    bad_symbols = set(["-"])
    tickers[:] = [e for e in tickers if e not in bad_symbols]
    n = len(tickers)
    print("symbols file, max stocks, # stocks:", SYMBOLS_FILE, MAX_STOCKS, n, end="\n\n")
    rows = []
    for i, tkr in enumerate(tickers, 1):
        print(f"[{i}/{n}] {tkr}", flush=True)
        rows.append(fundamentals_snapshot(tkr))

    df = pd.DataFrame(rows)

    # ratios
    df["dividend_yield"] = df["forward_dividend"] / df["price"]

    def _safe_ratio(numer_col: str, denom_col: str, out_col: str):
        a = df[numer_col]
        b = df[denom_col]
        df[out_col] = a / b
        df.loc[(b.isna()) | (b == 0), out_col] = np.nan
        df.replace({out_col: {np.inf: np.nan, -np.inf: np.nan}}, inplace=True)

    _safe_ratio("price", "target_1y_est", "p_target")
    _safe_ratio("price", "bvps",         "p_b")
    _safe_ratio("price", "eps_ann",      "p_e")     # use annualized EPS
    _safe_ratio("price", "rev_ps",       "p_s")     # per-share flows already annualized
    _safe_ratio("price", "ocf_ps",       "p_ocf")
    _safe_ratio("price", "fcf_ps",       "p_fcf")

    # reorder columns: ticker first, then price/ratios, then the rest
    front = ["ticker", "price", "p_target", "dividend_yield", "p_b", "p_e", "p_s", "p_ocf", "p_fcf"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    # ensure company_name, sector, industry are the last 3 columns
    tail = ["company_name", "sector", "industry"]
    cols = [c for c in df.columns if c not in tail] + tail
    df = df[cols]

    df.to_csv(results_file, index=False)
    print(df.to_string(index=False))

    if print_null_fracs:
        null_fracs = (df.drop(columns=["ticker"], errors="ignore").isna().mean()).to_dict()
        print("\nfraction invalid per field:")
        for k in sorted(null_fracs.keys()):
            print(f"  {k}: {null_fracs[k]:.3f}")

    elapsed = time.perf_counter() - start
    print("\nwrote data to", results_file + "\n")
    print("\n# stocks:", n)
    print("time elapsed (s):", f"{elapsed:.3f}")

if __name__ == "__main__":
    main()

