
#!/usr/bin/env python3
"""
FO_FNO_FYERS_CONFLUENCE_EMAIL_FIXED.py

Strict FYERS confluence screener - Production Version

Design rules:
- Anchor_915 is ONLY the 09:15 candle open for stocks. (Falls back to first available intraday candle if illiquid).
- Only rows with Anchor_Source=OPEN_915 are eligible for ranked Long/Short tables.
  Rows where the 09:15 candle was missing and a later candle was used as fallback are
  tagged Anchor_Source=OPEN_915_LATE so they can be excluded/flagged separately.
- Quote fallback rows are sent to a separate watchlist section, anchored to Live Quote Open.
- Index rows use quote-based anchors (Open Price).
- Automated Cron Safeguards: Market hours gating, JSON state tracking to prevent duplicate emails.
- Breach detection excludes gap-open false positives: the first candle of the day is never
  treated as a "crossover", only genuine intraday crosses count.
"""

import os
import sys
import time
import json
import logging
import warnings
import smtplib
from io import StringIO
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from fyers_apiv3 import fyersModel

IST = "Asia/Kolkata"
FYERS_FO_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
MARKET_OPEN = "09:15:00"
MARKET_CLOSE = "15:30:00"
VERSION = "2026-07-10-prod-v2"
STATE_FILE = Path("fyers_state.json")

# Non-equity underlyings that can appear in NSE_FO.csv but have no *-EQ symbol
NON_EQUITY_UNDERLYINGS = {
    "NIFTY", "NIFTY50", "NIFTYNXT50", "BANKNIFTY", "NIFTYBANK",
    "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX", "SENSEX50",
    "NIFTYMID", "NIFTYIT", "NIFTYPSE", "NIFTYINFRA",
}


class Config:
    def __init__(self):
        self.client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        self.access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")

        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
        self.sender_password = os.environ.get("SENDER_PASSWORD", "password")
        self.recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")

        self.lookback_days = int(os.environ.get("LOOKBACK_DAYS", "365"))
        self.history_pause_sec = float(os.environ.get("HISTORY_PAUSE_SEC", "0.25"))
        self.quotes_pause_sec = float(os.environ.get("QUOTES_PAUSE_SEC", "0.20"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "3"))
        self.stock_limit = int(os.environ.get("STOCK_LIMIT", "0"))

        # Basic proactive throttle: max requests/sec to FYERS across the whole run
        self.max_requests_per_sec = float(os.environ.get("MAX_REQ_PER_SEC", "6"))

        self.disable_index_scan = os.environ.get("DISABLE_INDEX_SCAN", "0") == "1"
        self.force_run = os.environ.get("FORCE_RUN", "0") == "1"

        # Include late-anchor (fallback-to-first-candle) rows in ranked tables too
        self.include_late_anchor_in_ranked = os.environ.get("INCLUDE_LATE_ANCHOR", "0") == "1"

        # Always send watchlist email even if no long/short candidates
        self.always_send_watchlist = os.environ.get("ALWAYS_SEND_WATCHLIST", "1") == "1"

        self.index_symbols = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX",
            "BSE:SENSEX-INDEX",
        ]
        self.fallback_stock_symbols = [
            "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ",
            "NSE:INFY-EQ", "NSE:TCS-EQ", "NSE:SBIN-EQ"
        ]


cfg = Config()

EMAIL_DISPLAY_COLS = [
    "Symbol", "Anchor_Source", "% Change", "Conf_Below-3",
    "Conf_Below-2", "Conf_Below-1", "Anchor_915", "LTP",
    "Conf_Above-1", "Conf_Above-2", "Conf_Above-3", "Breach_Time"
]

RESULT_COLS = [
    "Symbol", "Anchor_915", "Anchor_Source", "Calc_Anchor",
    "LTP", "Current_Close", "% Change", "Signal",
    "Conf_Below-3", "Conf_Below-2", "Conf_Below-1",
    "Conf_Above-1", "Conf_Above-2", "Conf_Above-3",
    "Support", "Resistance", "Support_Gap_Pct", "Resistance_Gap_Pct"
]

logger = logging.getLogger("fyers_confluence")
logger.setLevel(logging.INFO)
logger.handlers.clear()
stream = logging.StreamHandler(sys.stdout)
stream.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(stream)
warnings.filterwarnings("ignore")


# --- Simple global rate limiter (proactive, not just reactive backoff) ---
class RateLimiter:
    def __init__(self, max_per_sec):
        self.min_interval = 1.0 / max_per_sec if max_per_sec > 0 else 0.0
        self._last = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


rate_limiter = RateLimiter(cfg.max_requests_per_sec)


# --- State Management for Cron ---
def load_state(session_date):
    date_str = str(session_date)
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if data.get("date") == date_str:
                    return set(data.get("seen", []))
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    return set()


def save_state(session_date, seen_set):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"date": str(session_date), "seen": list(seen_set)}, f)
    except Exception as e:
        logger.error(f"Error saving state: {e}")


# --- Utility Functions ---
def safe_float(val):
    try:
        if val is None or pd.isna(val):
            return np.nan
        return float(val)
    except Exception:
        return np.nan


def format_value(val):
    if pd.isna(val) or val in [float("inf"), float("-inf")]:
        return "-"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    return str(val)


def format_change(val):
    try:
        if pd.isna(val):
            return "-"
        return f"{float(val):.2f}%"
    except Exception:
        return "-"


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def now_ist():
    return pd.Timestamp.now(tz=IST)


def current_session_date():
    now = now_ist()
    d = now.date()
    if now.time() < pd.Timestamp(MARKET_OPEN).time():
        d = d - timedelta(days=1)
    while pd.Timestamp(d).weekday() >= 5:
        d = d - timedelta(days=1)
    return d


# --- FYERS API ---
def init_fyers():
    try:
        if not cfg.client_id or not cfg.access_token:
            logger.error("Missing CLIENT_ID/ACCESS_TOKEN.")
            return None
        return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e:
        logger.error(f"FYERS init failed: {e}")
        return None


def call_with_retries(func, *args, **kwargs):
    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            rate_limiter.wait()
            out = func(*args, **kwargs)
            if isinstance(out, dict) and out.get("s") == "error":
                raise RuntimeError(out.get("message") or out.get("code") or "FYERS API error")
            return out
        except Exception as e:
            last_err = e
            wait = min(2 ** (attempt - 1), 5)
            logger.warning(f"Attempt {attempt}/{cfg.max_retries} failed: {e}")
            if attempt < cfg.max_retries:
                time.sleep(wait)
    raise RuntimeError(last_err)


def normalize_history_df(candles):
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


def get_history(fyers, symbol, resolution, days=None, range_from=None, range_to=None, date_format="1"):
    try:
        if date_format == "1":
            now = pd.Timestamp.now(tz=IST)
            start = (now - timedelta(days=days or cfg.lookback_days)).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")
        else:
            start, end = str(range_from), str(range_to)

        payload = {
            "symbol": symbol, "resolution": str(resolution), "date_format": str(date_format),
            "range_from": start, "range_to": end, "cont_flag": "1"
        }
        res = call_with_retries(fyers.history, data=payload)
        time.sleep(cfg.history_pause_sec)
        candles = res.get("candles", []) if isinstance(res, dict) else []
        if not candles:
            return None
        return normalize_history_df(candles)
    except Exception as e:
        logger.warning(f"History fetch failed for {symbol}: {e}")
        return None


def get_opening_anchor(fyers, symbol, session_date):
    """
    Returns (anchor_value, is_exact_915) so callers can distinguish a true
    09:15 anchor from a fallback-to-first-available-candle anchor.
    """
    try:
        session_start = int(pd.Timestamp(f"{session_date} {MARKET_OPEN}", tz=IST).timestamp())
        session_end = int(pd.Timestamp(f"{session_date} {MARKET_CLOSE}", tz=IST).timestamp())
        intraday = get_history(fyers, symbol, resolution="5", range_from=session_start, range_to=session_end, date_format="0")
        if intraday is None or intraday.empty:
            return np.nan, False

        intraday = intraday[intraday["timestamp"].dt.date == session_date].copy()
        if intraday.empty:
            return np.nan, False

        target = pd.Timestamp(f"{session_date} {MARKET_OPEN}")
        row = intraday.loc[intraday["timestamp"] == target]
        if not row.empty:
            return safe_float(row.iloc[0]["open"]), True

        market_rows = intraday.sort_values("timestamp")
        if not market_rows.empty:
            logger.info(f"09:15 exact candle missing for {symbol}; fallback first intraday candle used.")
            return safe_float(market_rows.iloc[0]["open"]), False

        return np.nan, False
    except Exception:
        return np.nan, False


def extract_quote_data(quote_item):
    if not isinstance(quote_item, dict):
        return {"ltp": np.nan, "open": np.nan}
    nested = quote_item.get("v") if isinstance(quote_item.get("v"), dict) else quote_item
    return {
        "ltp": safe_float(nested.get("lp") or nested.get("last_price")),
        "open": safe_float(nested.get("open_price"))
    }


def get_quotes_map(fyers, symbols):
    out = {}
    for batch in chunked(symbols, 50):
        try:
            res = call_with_retries(fyers.quotes, data={"symbols": ",".join(batch)})
            time.sleep(cfg.quotes_pause_sec)
            items = res.get("d", []) if isinstance(res, dict) else []
            for item in items:
                symbol = item.get("n") or item.get("symbol")
                if symbol:
                    out[symbol] = extract_quote_data(item)
        except Exception:
            for sym in batch:
                out.setdefault(sym, {"ltp": np.nan, "open": np.nan})
    return out


# --- Strategy Math ---
def extract_volume_weighted_nodes(df, bins=120, peak_quantile=0.60):
    if df is None or df.empty:
        return []
    work = df[["high", "low", "close", "volume"]].copy()
    work["volume"] = work["volume"].fillna(0)
    work = work.dropna()
    if work.empty:
        return []

    typical_price = (work["high"] + work["low"] + work["close"]) / 3.0
    min_p, max_p = safe_float(typical_price.min()), safe_float(typical_price.max())
    if pd.isna(min_p) or pd.isna(max_p) or min_p >= max_p:
        return []

    edges = np.linspace(min_p, max_p, bins)
    weights = work["volume"] if work["volume"].sum() > 0 else None
    hist, bin_edges = np.histogram(typical_price, bins=edges, weights=weights)

    threshold = np.nanquantile(hist, peak_quantile)
    nodes = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] >= threshold and hist[i] > 0:
            nodes.append(round((bin_edges[i] + bin_edges[i + 1]) / 2.0, 2))

    if not nodes and peak_quantile > 0.30:
        return extract_volume_weighted_nodes(df, bins, 0.30)
    return sorted(set(nodes))


def nearest_levels(levels, ref_price, count=3):
    levels = sorted(set([x for x in levels if not pd.isna(x)]))
    below = [x for x in levels if x <= ref_price]
    above = [x for x in levels if x >= ref_price]
    below_vals = [np.nan] * max(0, count - len(below[-count:])) + below[-count:]
    above_vals = above[:count] + [np.nan] * max(0, count - len(above[:count]))
    return below_vals, above_vals


def nearest_support_resistance(levels, ref_price):
    levels = sorted(set([x for x in levels if not pd.isna(x)]))
    support = max([x for x in levels if x <= ref_price], default=np.nan)
    resistance = min([x for x in levels if x >= ref_price], default=np.nan)
    return support, resistance


def build_row(symbol, anchor_915, anchor_source, calc_anchor, ltp_price, levels):
    below_vals, above_vals = nearest_levels(levels, calc_anchor, count=3)
    support, resistance = nearest_support_resistance(levels, ltp_price)

    pct_change = ((ltp_price - calc_anchor) / calc_anchor * 100.0) if calc_anchor and not pd.isna(ltp_price) else np.nan
    signal = "-" if pd.isna(pct_change) else ("Long" if pct_change >= 0 else "Short")

    support_gap_pct = ((ltp_price - support) / ltp_price * 100.0) if not pd.isna(support) and ltp_price else np.nan
    res_gap_pct = ((resistance - ltp_price) / ltp_price * 100.0) if not pd.isna(resistance) and ltp_price else np.nan

    return {
        "Symbol": symbol,
        "Anchor_915": round(anchor_915, 2) if not pd.isna(anchor_915) else np.nan,
        "Anchor_Source": anchor_source,
        "Calc_Anchor": round(calc_anchor, 2) if not pd.isna(calc_anchor) else np.nan,
        "LTP": round(ltp_price, 2) if not pd.isna(ltp_price) else np.nan,
        "Current_Close": round(ltp_price, 2) if not pd.isna(ltp_price) else np.nan,
        "% Change": round(pct_change, 2) if not pd.isna(pct_change) else np.nan,
        "Signal": signal,
        "Conf_Below-3": below_vals[0], "Conf_Below-2": below_vals[1], "Conf_Below-1": below_vals[2],
        "Conf_Above-1": above_vals[0], "Conf_Above-2": above_vals[1], "Conf_Above-3": above_vals[2],
        "Support": support, "Resistance": resistance,
        "Support_Gap_Pct": round(support_gap_pct, 2) if not pd.isna(support_gap_pct) else np.nan,
        "Resistance_Gap_Pct": round(res_gap_pct, 2) if not pd.isna(res_gap_pct) else np.nan,
    }


def scan_universe(fyers, symbol_list, session_date, is_index=False):
    rows = []
    if not symbol_list:
        return pd.DataFrame(columns=RESULT_COLS)
    quotes_map = get_quotes_map(fyers, symbol_list)

    for sym in symbol_list:
        try:
            q_data = quotes_map.get(sym, {})
            live_quote, open_quote = q_data.get("ltp"), q_data.get("open")
            daily = get_history(fyers, sym, "D", days=cfg.lookback_days)
            if daily is None or daily.empty:
                continue

            if daily.iloc[-1]["timestamp"].date() >= session_date:
                hist_daily = daily.iloc[:-1].copy()
            else:
                hist_daily = daily.copy()

            levels = extract_volume_weighted_nodes(hist_daily)
            if not levels:
                continue

            prev_close = safe_float(hist_daily["close"].iloc[-1]) if not hist_daily.empty else np.nan

            if is_index:
                anchor_915, anchor_source = np.nan, "QUOTE_INDEX"
                calc_anchor = open_quote if not pd.isna(open_quote) else prev_close
            else:
                anchor_915, is_exact = get_opening_anchor(fyers, sym, session_date)
                if pd.isna(anchor_915):
                    anchor_source = "QUOTE_FALLBACK"
                    calc_anchor = open_quote if not pd.isna(open_quote) else prev_close
                elif is_exact:
                    anchor_source, calc_anchor = "OPEN_915", anchor_915
                else:
                    # 09:15 candle missing; used first available intraday candle instead.
                    # Tagged distinctly so it is NOT silently treated as a true 09:15 anchor.
                    anchor_source, calc_anchor = "OPEN_915_LATE", anchor_915

            ltp = live_quote if not pd.isna(live_quote) else prev_close
            if pd.isna(ltp):
                continue

            rows.append(build_row(sym, anchor_915, anchor_source, calc_anchor, ltp, levels))
        except Exception as e:
            logger.warning(f"Scan failed for {sym}: {e}")

    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=RESULT_COLS)
    for col in RESULT_COLS:
        if col not in out.columns:
            out[col] = np.nan
    return out[RESULT_COLS]


def get_breach_time(fyers, symbol, session_date, level, direction="above"):
    """
    Returns the last genuine intraday crossover of `level`.
    The FIRST candle of the day is never counted as a crossover on its own
    (fixes the gap-open false positive where a stock opening beyond the level
    would otherwise always show a 09:15-09:20 breach time).
    """
    if pd.isna(level):
        return pd.NaT
    try:
        session_start = int(pd.Timestamp(f"{session_date} {MARKET_OPEN}", tz=IST).timestamp())
        session_end = int(pd.Timestamp(f"{session_date} {MARKET_CLOSE}", tz=IST).timestamp())
        intraday = get_history(fyers, symbol, resolution="5", range_from=session_start, range_to=session_end, date_format="0")
        if intraday is None or intraday.empty:
            return pd.NaT
        intraday = intraday[intraday["timestamp"].dt.date == session_date].sort_values("timestamp").reset_index(drop=True)
        if len(intraday) < 2:
            return pd.NaT

        # Drop the first candle from crossover eligibility; only compare candle[i] vs candle[i-1]
        # for i >= 1, using the actual previous close (no fillna(0)/fillna(inf) sentinel hack).
        prev_close = intraday["close"].shift(1)
        curr_close = intraday["close"]

        if direction == "above":
            hits = intraday[(curr_close > level) & (prev_close <= level) & prev_close.notna()]
        else:
            hits = intraday[(curr_close < level) & (prev_close >= level) & prev_close.notna()]

        if hits.empty:
            return pd.NaT
        return hits.iloc[-1]["timestamp"]
    except Exception:
        return pd.NaT


# --- Execution & Rendering ---
def build_html_table(df, title, cols):
    if df is None or df.empty:
        return f"<h3 style='color:#fbbf24; font-family:sans-serif;'>{title}</h3><p style='color:#94a3b8;'>No candidates found.</p>"

    html = f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top:25px;'>{title}</h3><table style='border-collapse:collapse; width:100%; font-family:sans-serif; font-size:13px; text-align:left; background-color:#0f172a;'>"
    html += "<tr style='background-color:#1e293b; color:#f1f5f9;'>" + "".join([f"<th style='padding:10px; border:1px solid #334155;'>{c}</th>" for c in cols]) + "</tr>"

    for i, (_, row) in enumerate(df.iterrows()):
        bg = "#0f172a" if i % 2 == 0 else "#1e293b"
        html += f"<tr style='background-color:{bg}; color:#e2e8f0;'>"
        for c in cols:
            val = row.get(c, "-")
            style = "padding:8px; border:1px solid #334155;"
            if c == "% Change":
                val_num, val_str = safe_float(val), format_change(val)
                if not pd.isna(val_num):
                    style += " color:#4ade80; font-weight:bold;" if val_num > 0 else " color:#f87171; font-weight:bold;"
                html += f"<td style='{style}'>{val_str}</td>"
            elif c == "Breach_Time" and (pd.isna(val) or val == "-"):
                html += f"<td style='{style}'>-</td>"
            elif c == "Breach_Time":
                html += f"<td style='{style}'>{val.strftime('%H:%M')}</td>"
            else:
                html += f"<td style='{style}'>{format_value(val)}</td>"
        html += "</tr>"
    return html + "</table>"


def send_email(index_df, long_df, short_df, fallback_df, session_date):
    have_ranked = not (long_df.empty and short_df.empty)
    have_watchlist = not fallback_df.empty

    if not have_ranked and not (cfg.always_send_watchlist and have_watchlist):
        logger.info("No new actionable candidates and no watchlist to report. Skipping email.")
        return

    recipients = [x.strip() for x in cfg.recipient_email.split(",") if x.strip()]
    if not recipients:
        return

    msg = MIMEMultipart("alternative")
    msg["From"] = cfg.sender_email
    msg["To"] = ", ".join(recipients)
    total_candidates = len(long_df) + len(short_df)
    subject_suffix = f"{total_candidates} New Candidates" if total_candidates > 0 else "Watchlist Update"
    msg["Subject"] = f"FYERS Alert: {subject_suffix} - {datetime.now().strftime('%H:%M')}"

    html = f"<html><body style='background-color:#030712; padding:20px; font-family:sans-serif;'><h2 style='color:#e2e8f0;'>FYERS Confluence</h2>"
    html += f"{build_html_table(index_df, 'Market Index Snapshot', EMAIL_DISPLAY_COLS)}"
    html += f"{build_html_table(long_df, 'New Long Candidates', EMAIL_DISPLAY_COLS)}"
    html += f"{build_html_table(short_df, 'New Short Candidates', EMAIL_DISPLAY_COLS)}"
    html += f"{build_html_table(fallback_df, 'Watchlist (Missing/Late Open)', EMAIL_DISPLAY_COLS)}</body></html>"

    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, recipients, msg.as_string())
        logger.info("Email sent.")
    except Exception as e:
        logger.error(f"Email failed: {e}")


def get_live_fno_symbols():
    try:
        text = requests.get(FYERS_FO_MASTER_URL, timeout=30).text
        raw = pd.read_csv(StringIO(text), header=None)
        underlying = raw[13].astype(str).str.strip().str.upper().dropna().unique()
        symbols = [
            f"NSE:{sym}-EQ" for sym in underlying
            if len(sym) >= 2 and sym not in {"", "SYMBOL"} and sym not in NON_EQUITY_UNDERLYINGS
        ]
        out = sorted(set(symbols))
        return out[:cfg.stock_limit] if cfg.stock_limit > 0 else out
    except Exception:
        return cfg.fallback_stock_symbols


def main():
    now = now_ist()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    fyers = init_fyers()
    if not fyers:
        return

    session_date = current_session_date()
    seen_candidates = load_state(session_date)

    index_df = pd.DataFrame(columns=RESULT_COLS) if cfg.disable_index_scan else scan_universe(fyers, cfg.index_symbols, session_date, is_index=True)
    live_stock_symbols = get_live_fno_symbols()
    stock_df = scan_universe(fyers, live_stock_symbols, session_date)

    long_stocks, short_stocks = pd.DataFrame(columns=RESULT_COLS), pd.DataFrame(columns=RESULT_COLS)
    fallback_df = pd.DataFrame(columns=RESULT_COLS)
    new_seen = set()

    if not stock_df.empty:
        eligible_sources = {"OPEN_915"}
        if cfg.include_late_anchor_in_ranked:
            eligible_sources.add("OPEN_915_LATE")

        valid_df = stock_df[stock_df["Anchor_Source"].isin(eligible_sources)].copy()
        fallback_df = stock_df[~stock_df["Anchor_Source"].isin(eligible_sources)].copy()

        if not valid_df.empty:
            c_a1, c_a2 = valid_df["Conf_Above-1"], valid_df["Conf_Above-2"]
            c_b1, c_b2 = valid_df["Conf_Below-1"], valid_df["Conf_Below-2"]
            ltp = valid_df["LTP"]

            long_mask = (valid_df["Signal"] == "Long") & c_a1.notna() & c_a2.notna() & (ltp > c_a1) & (ltp < c_a2)
            short_mask = (valid_df["Signal"] == "Short") & c_b1.notna() & c_b2.notna() & (ltp < c_b1) & (ltp > c_b2)

            long_candidates = valid_df[long_mask].copy()
            short_candidates = valid_df[short_mask].copy()

            long_candidates = long_candidates[~long_candidates["Symbol"].isin(seen_candidates)]
            short_candidates = short_candidates[~short_candidates["Symbol"].isin(seen_candidates)]

            if not long_candidates.empty:
                long_candidates["Breach_Time"] = long_candidates.apply(
                    lambda r: get_breach_time(fyers, r["Symbol"], session_date, r["Conf_Above-1"], "above"), axis=1
                )
                long_stocks = long_candidates.sort_values(by=["Breach_Time", "% Change"], ascending=[False, False], na_position="last")
                new_seen.update(long_stocks["Symbol"].tolist())

            if not short_candidates.empty:
                short_candidates["Breach_Time"] = short_candidates.apply(
                    lambda r: get_breach_time(fyers, r["Symbol"], session_date, r["Conf_Below-1"], "below"), axis=1
                )
                short_stocks = short_candidates.sort_values(by=["Breach_Time", "% Change"], ascending=[False, True], na_position="last")
                new_seen.update(short_stocks["Symbol"].tolist())

        if not fallback_df.empty:
            fallback_df = fallback_df.sort_values(by=["% Change"], ascending=[False])

    if not index_df.empty:
        index_df = index_df.sort_values(by=["% Change"], ascending=[False])

    send_email(index_df, long_stocks, short_stocks, fallback_df, session_date)

    seen_candidates.update(new_seen)
    save_state(session_date, seen_candidates)


if __name__ == "__main__":
    main()
