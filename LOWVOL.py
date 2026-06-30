#!/usr/bin/env python3
"""
FO_FNO_FYERS_CONFLUENCE_EMAIL_FIXED.py

Production-hardened index and F&O stock screener using FYERS API v3.

Fixes applied:
- 09:15 anchor is selected by exact candle timestamp, not array position.
- Live price comes from Quotes API; daily history is only for node extraction.
- F&O universe parsing is more defensive.
- Retry, throttling, fallback, and logging are improved.
- Email dashboard keeps your original style, but Anchor_915 is shown honestly.
"""

import os
import sys
import time
import logging
import warnings
import smtplib
from io import StringIO
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from fyers_apiv3 import fyersModel

IST = "Asia/Kolkata"
FYERS_FO_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"


class Config:
    def __init__(self):
        self.client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        self.access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")

        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
        self.sender_password = os.environ.get("SENDER_PASSWORD", "password")
        self.recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")

        self.lookback_days = int(os.environ.get("LOOKBACK_DAYS", "253"))
        self.history_pause_sec = float(os.environ.get("HISTORY_PAUSE_SEC", "0.12"))
        self.quotes_pause_sec = float(os.environ.get("QUOTES_PAUSE_SEC", "0.20"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "3"))
        self.stock_limit = int(os.environ.get("STOCK_LIMIT", "0"))

        self.index_symbols = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX",
            "BSE:SENSEX-INDEX",
        ]

        self.fallback_stock_symbols = [
            "NSE:RELIANCE-EQ",
            "NSE:HDFCBANK-EQ",
            "NSE:ICICIBANK-EQ",
            "NSE:INFY-EQ",
            "NSE:TCS-EQ",
            "NSE:SBIN-EQ",
            "NSE:ITC-EQ",
            "NSE:LT-EQ",
            "NSE:AXISBANK-EQ",
            "NSE:KOTAKBANK-EQ",
        ]


cfg = Config()

EMAIL_DISPLAY_COLS = [
    "Symbol",
    "% Change",
    "Conf_Below-3",
    "Conf_Below-2",
    "Conf_Below-1",
    "Anchor_915",
    "Conf_Above-1",
    "Conf_Above-2",
    "Conf_Above-3",
]

RESULT_COLS = [
    "Symbol",
    "Anchor_915",
    "LTP",
    "Current_Close",
    "% Change",
    "Signal",
    "Conf_Below-3",
    "Conf_Below-2",
    "Conf_Below-1",
    "Conf_Above-1",
    "Conf_Above-2",
    "Conf_Above-3",
    "Support",
    "Resistance",
    "Support_Gap_Pct",
    "Resistance_Gap_Pct",
]

logger = logging.getLogger("fyers_confluence")
logger.setLevel(logging.INFO)
logger.handlers.clear()
stream = logging.StreamHandler(sys.stdout)
stream.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(stream)

warnings.filterwarnings("ignore")


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
        return f"{float(val):.2f}%"
    except Exception:
        return "-"


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def now_ist():
    return pd.Timestamp.now(tz=IST)


def init_fyers():
    try:
        if not cfg.client_id or not cfg.access_token:
            logger.error("Missing CLIENT_ID/ACCESS_TOKEN.")
            return None
        return fyersModel.FyersModel(
            client_id=cfg.client_id,
            is_async=False,
            token=cfg.access_token,
            log_path=""
        )
    except Exception as e:
        logger.error(f"FYERS init failed: {e}")
        return None


def call_with_retries(func, *args, **kwargs):
    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
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
            if range_from is None or range_to is None:
                now = datetime.now()
                start = (now - timedelta(days=days or cfg.lookback_days)).strftime("%Y-%m-%d")
                end = now.strftime("%Y-%m-%d")
            else:
                start, end = range_from, range_to
        else:
            start, end = str(range_from), str(range_to)

        payload = {
            "symbol": symbol,
            "resolution": str(resolution),
            "date_format": str(date_format),
            "range_from": start,
            "range_to": end,
            "cont_flag": "1",
        }
        res = call_with_retries(fyers.history, data=payload)
        time.sleep(cfg.history_pause_sec)
        candles = res.get("candles", []) if isinstance(res, dict) else []
        if not candles:
            return None
        return normalize_history_df(candles)
    except Exception as e:
        logger.warning(f"History fetch failed for {symbol} [{resolution}]: {e}")
        return None


def get_opening_anchor(fyers, symbol):
    try:
        today = now_ist().date()
        session_start = int(pd.Timestamp(f"{today} 09:15:00", tz=IST).timestamp())
        session_end = int(pd.Timestamp(f"{today} 15:30:00", tz=IST).timestamp())

        intraday = get_history(
            fyers,
            symbol,
            resolution="5",
            range_from=session_start,
            range_to=session_end,
            date_format="0",
        )
        if intraday is None or intraday.empty:
            return np.nan

        intraday = intraday[intraday["timestamp"].dt.date == today].copy()
        if intraday.empty:
            logger.info(f"No same-day intraday rows for {symbol}")
            return np.nan

        target = pd.Timestamp(f"{today} 09:15:00")
        row = intraday.loc[intraday["timestamp"] == target]
        if not row.empty:
            return safe_float(row.iloc[0]["open"])

        market_rows = intraday[
            (intraday["timestamp"].dt.time >= pd.Timestamp("09:15:00").time())
            & (intraday["timestamp"].dt.time <= pd.Timestamp("15:30:00").time())
        ].copy()
        if not market_rows.empty:
            market_rows = market_rows.sort_values("timestamp")
            first_ts = market_rows.iloc[0]["timestamp"]
            first_open = safe_float(market_rows.iloc[0]["open"])
            logger.info(f"09:15 exact candle missing for {symbol}; fallback first intraday candle={first_ts}")
            return first_open

        logger.info(f"09:15 candle not found for {symbol}; returned={intraday['timestamp'].dt.strftime('%H:%M:%S').tolist()[:10]}")
        return np.nan
    except Exception as e:
        logger.warning(f"Opening anchor fetch failed for {symbol}: {e}")
        return np.nan


def extract_quote_ltp(quote_item):
    if not isinstance(quote_item, dict):
        return np.nan
    direct_keys = ["lp", "ltp", "last_price", "last_traded_price"]
    for key in direct_keys:
        val = safe_float(quote_item.get(key))
        if not pd.isna(val):
            return val
    nested = quote_item.get("v") if isinstance(quote_item.get("v"), dict) else {}
    for key in direct_keys:
        val = safe_float(nested.get(key))
        if not pd.isna(val):
            return val
    return np.nan


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
                    out[symbol] = extract_quote_ltp(item)
        except Exception as e:
            logger.warning(f"Quotes batch failed: {e}")
            for sym in batch:
                out.setdefault(sym, np.nan)
    return out


def fetch_text(url, timeout=30):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.text


def get_live_fno_symbols():
    exclude_exact = {
        "", "SYMBOL", "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
        "NIFTYNXT50", "SENSEX", "BANKEX", "CE", "PE", "XX", "FUT"
    }
    try:
        text = fetch_text(FYERS_FO_MASTER_URL)
        raw = pd.read_csv(StringIO(text), header=None)
        if raw.empty:
            raise ValueError("NSE_FO.csv is empty")
        if raw.shape[1] <= 13:
            raise ValueError(f"Unexpected NSE_FO.csv shape: {raw.shape}")

        # FYERS community example shows underlying at column 13 in NSE_FO.csv.
        underlying = raw[13].astype(str).str.strip().str.upper()

        symbols = set()
        for sym in underlying.dropna().unique():
            if not sym or sym in exclude_exact:
                continue
            if any(x in sym for x in ["NIFTY", "SENSEX", "BANKEX"]):
                continue
            if len(sym) < 2:
                continue
            if not pd.Series([sym]).str.fullmatch(r"[A-Z][A-Z0-9&\-]{1,30}").iloc[0]:
                continue
            symbols.add(f"NSE:{sym}-EQ")

        out = sorted(symbols)
        if cfg.stock_limit > 0:
            out = out[:cfg.stock_limit]
        if out:
            logger.info(f"Fetched live F&O universe from Fyers Master: {len(out)} symbols")
            return out
        raise ValueError("No valid symbols extracted")
    except Exception as e:
        logger.warning(f"Fyers Symbol Master fetch/parse failed: {e}")
        logger.warning("Falling back to configured stock universe.")
        return cfg.fallback_stock_symbols


def extract_volume_weighted_nodes(df, bins=120, peak_quantile=0.60):
    if df is None or df.empty:
        return []

    work = df[["high", "low", "close", "volume"]].dropna().copy()
    if work.empty:
        return []

    typical_price = (work["high"] + work["low"] + work["close"]) / 3.0
    min_p = safe_float(typical_price.min())
    max_p = safe_float(typical_price.max())
    if pd.isna(min_p) or pd.isna(max_p) or min_p >= max_p:
        return []

    edges = np.linspace(min_p, max_p, bins)
    weights = work["volume"] if work["volume"].sum() > 0 else None
    hist, bin_edges = np.histogram(typical_price, bins=edges, weights=weights)
    if len(hist) < 3 or np.nansum(hist) <= 0:
        return []

    threshold = np.nanquantile(hist, peak_quantile)
    nodes = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] >= threshold and hist[i] > 0:
            nodes.append(round((bin_edges[i] + bin_edges[i + 1]) / 2.0, 2))
    return sorted(set(nodes))


def nearest_levels(levels, ref_price, count=3):
    levels = sorted(set([x for x in levels if not pd.isna(x)]))
    below = [x for x in levels if x < ref_price]
    above = [x for x in levels if x > ref_price]
    below_vals = [np.nan] * max(0, count - len(below[-count:])) + below[-count:]
    above_vals = above[:count] + [np.nan] * max(0, count - len(above[:count]))
    return below_vals, above_vals


def nearest_support_resistance(levels, ref_price):
    levels = sorted(set([x for x in levels if not pd.isna(x)]))
    support = max([x for x in levels if x < ref_price], default=np.nan)
    resistance = min([x for x in levels if x > ref_price], default=np.nan)
    return support, resistance


def build_row(symbol, anchor_price, ltp_price, levels):
    below_vals, above_vals = nearest_levels(levels, anchor_price, count=3)
    support, resistance = nearest_support_resistance(levels, anchor_price)

    pct_change = ((ltp_price - anchor_price) / anchor_price * 100.0) if anchor_price and not pd.isna(ltp_price) else np.nan
    signal = "Long" if not pd.isna(pct_change) and pct_change >= 0 else "Short"
    support_gap_pct = ((anchor_price - support) / anchor_price * 100.0) if not pd.isna(support) and anchor_price else np.nan
    resistance_gap_pct = ((resistance - anchor_price) / anchor_price * 100.0) if not pd.isna(resistance) and anchor_price else np.nan

    return {
        "Symbol": symbol,
        "Anchor_915": round(anchor_price, 2),
        "LTP": round(ltp_price, 2) if not pd.isna(ltp_price) else np.nan,
        "Current_Close": round(ltp_price, 2) if not pd.isna(ltp_price) else np.nan,
        "% Change": round(pct_change, 2) if not pd.isna(pct_change) else np.nan,
        "Signal": signal,
        "Conf_Below-3": below_vals[0],
        "Conf_Below-2": below_vals[1],
        "Conf_Below-1": below_vals[2],
        "Conf_Above-1": above_vals[0],
        "Conf_Above-2": above_vals[1],
        "Conf_Above-3": above_vals[2],
        "Support": support,
        "Resistance": resistance,
        "Support_Gap_Pct": round(support_gap_pct, 2) if not pd.isna(support_gap_pct) else np.nan,
        "Resistance_Gap_Pct": round(resistance_gap_pct, 2) if not pd.isna(resistance_gap_pct) else np.nan,
    }


def scan_universe(fyers, symbol_list, use_quotes_only=False):
    rows = []
    if not symbol_list:
        return pd.DataFrame(columns=RESULT_COLS)

    logger.info(f"Fetching quotes for {len(symbol_list)} symbols...")
    quotes_map = get_quotes_map(fyers, symbol_list)

    for sym in symbol_list:
        try:
            if use_quotes_only:
                anchor_price = safe_float(quotes_map.get(sym))
                if pd.isna(anchor_price):
                    logger.info(f"Skipping {sym}: no valid quote anchor.")
                    continue
            else:
                anchor_price = get_opening_anchor(fyers, sym)
                if pd.isna(anchor_price):
                    logger.info(f"Skipping {sym}: no valid 09:15 anchor.")
                    continue

            daily = get_history(fyers, sym, "D", days=cfg.lookback_days)
            if daily is None or daily.empty:
                logger.info(f"Skipping {sym}: no daily history.")
                continue

            levels = extract_volume_weighted_nodes(daily)
            if not levels:
                logger.info(f"Skipping {sym}: no nodes found.")
                continue

            ltp = safe_float(quotes_map.get(sym))
            if pd.isna(ltp):
                ltp = safe_float(daily["close"].iloc[-1])
                logger.info(f"Quote fallback used for {sym}: {ltp}")
            if pd.isna(ltp):
                logger.info(f"Skipping {sym}: invalid live price.")
                continue

            rows.append(build_row(sym, anchor_price, ltp, levels))
        except Exception as e:
            logger.warning(f"Scan failed for {sym}: {e}")

    if not rows:
        return pd.DataFrame(columns=RESULT_COLS)

    out = pd.DataFrame(rows)
    for col in RESULT_COLS:
        if col not in out.columns:
            out[col] = np.nan
    return out[RESULT_COLS]


def build_html_table(df, title, cols):
    if df is None or df.empty:
        return (
            f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top:25px;'>{title}</h3>"
            "<p style='color:#94a3b8; font-family:sans-serif;'>No candidates found today.</p>"
        )

    table_html = (
        f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top:25px;'>{title}</h3>"
        "<table style='border-collapse:collapse; width:100%; font-family:sans-serif; font-size:13px; text-align:left; background-color:#0f172a;'>"
    )
    table_html += (
        "<tr style='background-color:#1e293b; color:#f1f5f9;'>"
        + "".join([f"<th style='padding:10px; border:1px solid #334155;'>{c}</th>" for c in cols])
        + "</tr>"
    )

    for i, (_, row) in enumerate(df.iterrows()):
        bg_row = "#0f172a" if i % 2 == 0 else "#1e293b"
        table_html += f"<tr style='background-color:{bg_row}; color:#e2e8f0;'>"
        for c in cols:
            val = row.get(c, "-")
            style = "padding:8px; border:1px solid #334155;"
            if c == "% Change":
                val_num = safe_float(val)
                val_str = format_change(val_num)
                if not pd.isna(val_num):
                    style += " color:#4ade80; font-weight:bold;" if val_num > 0 else " color:#f87171; font-weight:bold;"
                table_html += f"<td style='{style}'>{val_str}</td>"
            else:
                table_html += f"<td style='{style}'>{format_value(val)}</td>"
        table_html += "</tr>"
    return table_html + "</table>"


def send_email(index_df, long_df, short_df):
    try:
        recipients = [x.strip() for x in cfg.recipient_email.split(",") if x.strip()]
        if not recipients:
            raise ValueError("RECIPIENT_EMAIL is empty.")

        msg = MIMEMultipart("alternative")
        msg["From"] = cfg.sender_email
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = f"FYERS Confluence Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        html = (
            "<html><body style='background-color:#030712; padding:20px; font-family:sans-serif;'>"
            "<h2 style='color:#e2e8f0;'>FYERS Confluence Dashboard</h2>"
            "<p style='color:#94a3b8;'>For stocks, Anchor_915 is the 9:15 AM candle open. For indexes, anchor falls back to current quote because FYERS intraday history for some index symbols can intermittently fail. % Change is computed from anchor to live Quotes API LTP.</p>"
            f"{build_html_table(index_df, 'Market Index Nodes', EMAIL_DISPLAY_COLS)}"
            f"{build_html_table(long_df, 'F&O Long Candidates (Closest Support First)', EMAIL_DISPLAY_COLS)}"
            f"{build_html_table(short_df, 'F&O Short Candidates (Closest Resistance First)', EMAIL_DISPLAY_COLS)}"
            "</body></html>"
        )

        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, recipients, msg.as_string())

        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Email failed: {e}")


def main():
    fyers = init_fyers()
    if not fyers:
        return

    logger.info("Starting index scan...")
    index_df = scan_universe(fyers, cfg.index_symbols, use_quotes_only=True)

    logger.info("Fetching live F&O stock universe from Fyers Master CSV...")
    live_stock_symbols = get_live_fno_symbols()

    logger.info(f"Starting stock scan on {len(live_stock_symbols)} symbols...")
    stock_df = scan_universe(fyers, live_stock_symbols)

    long_stocks = pd.DataFrame(columns=RESULT_COLS)
    short_stocks = pd.DataFrame(columns=RESULT_COLS)

    if not stock_df.empty:
        long_stocks = stock_df[stock_df["Signal"] == "Long"].copy()
        short_stocks = stock_df[stock_df["Signal"] == "Short"].copy()

        if not long_stocks.empty:
            long_stocks = long_stocks.sort_values(
                by=["Support_Gap_Pct", "% Change"],
                ascending=[True, False],
                na_position="last",
            )

        if not short_stocks.empty:
            short_stocks = short_stocks.sort_values(
                by=["Resistance_Gap_Pct", "% Change"],
                ascending=[True, True],
                na_position="last",
            )

    if not index_df.empty:
        index_df = index_df.sort_values(
            by=["% Change", "Symbol"],
            ascending=[False, True],
            na_position="last",
        )

    send_email(index_df, long_stocks, short_stocks)


if __name__ == "__main__":
    main()
