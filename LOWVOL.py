#!/usr/bin/env python3
"""
FO_FNO_FYERS_CONFLUENCE_EMAIL.py
Index & Stock Confluence Screener using HV/LV overlap zones.
LTP is the 9:15 AM candle OPEN.
"""

import os
import sys
import logging
import warnings
import smtplib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class Config:
    def __init__(self):
        self.client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        self.access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")

        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
        self.sender_password = os.environ.get("SENDER_PASSWORD", "password")
        self.recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")

        self.lookback_days = int(os.environ.get("LOOKBACK_DAYS", "150"))
        self.top_n = int(os.environ.get("TOP_N", "60"))
        self.dedupe_pct = float(os.environ.get("DEDUPE_PCT", "0.005"))
        self.match_tolerance = float(os.environ.get("MATCH_TOLERANCE", "0.005"))

        self.index_symbols = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX",
            "BSE:SENSEX-INDEX",
        ]

        self.stock_symbols = [
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
    "LTP",
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

logger = logging.getLogger("confluence")
logger.setLevel(logging.INFO)
logger.handlers.clear()

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)

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
    except (TypeError, ValueError):
        return "-"


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
        logger.warning(f"INIT Failed: {e}")
        return None


def get_history(fyers, symbol, res, days):
    try:
        now = datetime.now()
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        res_data = fyers.history(data={
            "symbol": symbol,
            "resolution": res,
            "date_format": "1",
            "range_from": start,
            "range_to": end,
            "cont_flag": "1",
        })

        if res_data and "candles" in res_data:
            df = pd.DataFrame(
                res_data["candles"],
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.sort_values("timestamp").reset_index(drop=True)

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        return None
    except Exception as e:
        logger.warning(f"History fetch failed for {symbol} [{res}]: {e}")
        return None


def get_opening_anchor(fyers, symbol):
    """
    Uses the OPEN of the first 5-minute candle as 9:15 anchor.
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        data = fyers.history(data={
            "symbol": symbol,
            "resolution": "5",
            "date_format": "1",
            "range_from": today,
            "range_to": today,
            "cont_flag": "1"
        })

        if data and "candles" in data and len(data["candles"]) > 0:
            return float(data["candles"][0][1])  # 9:15 OPEN

        return None
    except Exception as e:
        logger.warning(f"Opening anchor fetch failed for {symbol}: {e}")
        return None


def dedupe_levels(levels, tolerance_pct):
    values = sorted([
        safe_float(x) for x in levels
        if not pd.isna(safe_float(x)) and safe_float(x) > 0
    ])

    if not values:
        return []

    groups = [[values[0]]]
    for val in values[1:]:
        ref = float(np.mean(groups[-1]))
        if ref > 0 and abs(val - ref) / ref <= tolerance_pct:
            groups[-1].append(val)
        else:
            groups.append([val])

    return [round(float(np.mean(group)), 2) for group in groups]


def build_price_levels(df):
    raw_levels = []
    raw_levels.extend(df["high"].dropna().tolist())
    raw_levels.extend(df["low"].dropna().tolist())
    raw_levels.extend(df["close"].dropna().tolist())
    return raw_levels


def extract_confluence_levels(df):
    if df is None or df.empty:
        return []

    work = df.dropna(subset=["high", "low", "close", "volume"]).copy()
    if work.empty:
        return []

    hv_work = work.sort_values(["volume", "high"], ascending=[False, False]).head(cfg.top_n)
    lv_work = work.sort_values(["volume", "high"], ascending=[True, False]).head(cfg.top_n)

    hv_levels = dedupe_levels(build_price_levels(hv_work), cfg.dedupe_pct)
    lv_levels = dedupe_levels(build_price_levels(lv_work), cfg.dedupe_pct)

    confluence = []
    for hv in hv_levels:
        for lv in lv_levels:
            if hv > 0 and abs(hv - lv) / hv <= cfg.match_tolerance:
                confluence.append(round((hv + lv) / 2.0, 2))

    return dedupe_levels(confluence, cfg.dedupe_pct)


def nearest_levels(levels, ref_price, count=3):
    levels = sorted(set(levels))
    below = [x for x in levels if x < ref_price]
    above = [x for x in levels if x > ref_price]

    below_vals = below[-count:]
    above_vals = above[:count]

    below_vals = [np.nan] * (count - len(below_vals)) + below_vals
    above_vals = above_vals + [np.nan] * (count - len(above_vals))

    return below_vals, above_vals


def nearest_support_resistance(levels, ref_price):
    levels = sorted(set(levels))
    support = max([x for x in levels if x < ref_price], default=np.nan)
    resistance = min([x for x in levels if x > ref_price], default=np.nan)
    return support, resistance


def build_row(symbol, anchor_price, current_close, levels):
    display_ltp = anchor_price  # LTP should be 9:15 open
    below_vals, above_vals = nearest_levels(levels, display_ltp, count=3)
    support, resistance = nearest_support_resistance(levels, display_ltp)

    pct_change = ((current_close - anchor_price) / anchor_price * 100.0) if anchor_price else np.nan
    signal = "Long" if not pd.isna(pct_change) and pct_change >= 0 else "Short"

    support_gap_pct = ((display_ltp - support) / display_ltp * 100.0) if not pd.isna(support) and display_ltp else np.nan
    resistance_gap_pct = ((resistance - display_ltp) / display_ltp * 100.0) if not pd.isna(resistance) and display_ltp else np.nan

    return {
        "Symbol": symbol,
        "Anchor_915": round(anchor_price, 2),
        "LTP": round(display_ltp, 2),
        "Current_Close": round(current_close, 2),
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


def scan_universe(fyers, symbol_list, is_stock=False):
    rows = []

    for sym in symbol_list:
        try:
            anchor_price = get_opening_anchor(fyers, sym)
            if anchor_price is None or pd.isna(anchor_price):
                logger.info(f"Skipping {sym}: no 9:15 anchor.")
                continue

            daily = get_history(fyers, sym, "D", cfg.lookback_days)
            if daily is None or daily.empty:
                logger.info(f"Skipping {sym}: no daily history.")
                continue

            valid_daily = daily[daily["volume"].fillna(0) > 0].copy()
            if valid_daily.empty:
                logger.info(f"Skipping {sym}: no valid daily rows.")
                continue

            levels = extract_confluence_levels(valid_daily)
            if not levels:
                logger.info(f"Skipping {sym}: no HV/LV confluence levels.")
                continue

            current_close = safe_float(valid_daily["close"].iloc[-1])
            if pd.isna(current_close):
                logger.info(f"Skipping {sym}: invalid close.")
                continue

            row = build_row(sym, anchor_price, current_close, levels)
            rows.append(row)

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
        "<table style='border-collapse:collapse; width:100%; font-family:sans-serif; "
        "font-size:13px; text-align:left; background-color:#0f172a;'>"
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
        msg["Subject"] = f"Confluence Trade Setups - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        html = (
            "<html><body style='background-color:#030712; padding:20px; font-family:sans-serif;'>"
            "<h2 style='color:#e2e8f0;'>Confluence Dashboard</h2>"
            f"{build_html_table(index_df, 'Market Index Confluence Dashboard', EMAIL_DISPLAY_COLS)}"
            f"{build_html_table(long_df, 'F&O Long Candidates (Ordered by Proximity to Support)', EMAIL_DISPLAY_COLS)}"
            f"{build_html_table(short_df, 'F&O Short Candidates (Ordered by Proximity to Resistance)', EMAIL_DISPLAY_COLS)}"
            "</body></html>"
        )

        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, recipients, msg.as_string())

        logger.info("Confluence Email sent successfully.")
    except Exception as e:
        logger.error(f"Email Failed: {e}")


def main():
    fyers = init_fyers()
    if not fyers:
        return

    logger.info("Starting Index Scan...")
    index_df = scan_universe(fyers, cfg.index_symbols, is_stock=False)

    logger.info("Starting F&O Stock Scan...")
    stock_df = scan_universe(fyers, cfg.stock_symbols, is_stock=True)

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
