
#!/usr/bin/env python3
"""
FO_FNO_FYERS_CONFLUENCE_EMAIL.py
Index & Stock Screener using live NSE F&O universe from Fyers Symbol Master CSV.
Anchor price is the 09:15 candle open.
Support/Resistance levels are derived from a yearly volume-weighted price profile.
"""

import os
import sys
import logging
import warnings
import smtplib
from datetime import datetime, timedelta, time

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
        self.lookback_days = int(os.environ.get("LOOKBACK_DAYS", "253"))
        self.index_symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]
        self.fallback_stock_symbols = [
            "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:INFY-EQ",
            "NSE:TCS-EQ", "NSE:SBIN-EQ", "NSE:ITC-EQ", "NSE:LT-EQ",
            "NSE:AXISBANK-EQ", "NSE:KOTAKBANK-EQ",
        ]


cfg = Config()

EMAIL_DISPLAY_COLS = [
    "Symbol", "% Change", "Conf_Below-3", "Conf_Below-2", "Conf_Below-1",
    "LTP", "Conf_Above-1", "Conf_Above-2", "Conf_Above-3",
]

RESULT_COLS = [
    "Symbol", "Anchor_915", "LTP", "Current_Close", "% Change", "Signal",
    "Conf_Below-3", "Conf_Below-2", "Conf_Below-1", "Conf_Above-1", "Conf_Above-2",
    "Conf_Above-3", "Support", "Resistance", "Support_Gap_Pct", "Resistance_Gap_Pct",
]

logger = logging.getLogger("volume_profile")
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
    if not cfg.client_id or not cfg.access_token:
        logger.error("Missing CLIENT_ID/ACCESS_TOKEN.")
        return None
    try:
        return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
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
        if res_data and "candles" in res_data and res_data["candles"]:
            df = pd.DataFrame(res_data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
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
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        data = fyers.history(data={
            "symbol": symbol,
            "resolution": "5",
            "date_format": "1",
            "range_from": today,
            "range_to": today,
            "cont_flag": "1",
        })
        if not data or "candles" not in data or not data["candles"]:
            return None
        candles = pd.DataFrame(data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
        candles["timestamp"] = pd.to_datetime(candles["timestamp"], unit="s")
        candles["open"] = pd.to_numeric(candles["open"], errors="coerce")
        candles["time"] = candles["timestamp"].dt.time
        match = candles[candles["time"] == time(9, 15)]
        if match.empty:
            logger.info(f"No 09:15 candle for {symbol}")
            return None
        return safe_float(match.iloc[0]["open"])
    except Exception as e:
        logger.warning(f"Opening anchor fetch failed for {symbol}: {e}")
        return None


def get_live_fno_symbols():
    url = "https://public.fyers.in/sym_details/NSE_FO.csv"
    exclude = {"", "SYMBOL", "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX"}
    try:
        df = pd.read_csv(url, header=None, usecols=[13], names=["underlying"])
        unique_syms = df["underlying"].dropna().unique()
        symbols = set()
        for sym in unique_syms:
            sym = str(sym).strip().upper()
            if sym and sym not in exclude and "NIFTY" not in sym and "SENSEX" not in sym and "BANKEX" not in sym:
                symbols.add(f"NSE:{sym}-EQ")
        if symbols:
            out = sorted(symbols)
            logger.info(f"Fetched live F&O universe from Fyers Master: {len(out)} symbols")
            return out
    except Exception as e:
        logger.warning(f"Fyers Symbol Master fetch failed: {e}")
    logger.warning("Falling back to configured stock universe.")
    return cfg.fallback_stock_symbols


def extract_volume_profile_levels(df, bins=120):
    if df is None or df.empty:
        return []
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    price = typical_price.dropna()
    vol = df.loc[price.index, "volume"].fillna(0) if "volume" in df.columns else pd.Series(1, index=price.index)
    if price.empty or price.min() == price.max():
        return []
    edges = np.linspace(price.min(), price.max(), bins)
    hist, bin_edges = np.histogram(price, bins=edges, weights=vol if vol.sum() > 0 else None)
    if len(hist) < 3:
        return []
    mean_vol = np.mean(hist)
    hvns = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > mean_vol:
            hvns.append(round((bin_edges[i] + bin_edges[i + 1]) / 2.0, 2))
    return hvns


def nearest_levels(levels, ref_price, count=3):
    levels = sorted(set(levels))
    below = [x for x in levels if x < ref_price]
    above = [x for x in levels if x > ref_price]
    below_vals = [np.nan] * (count - len(below[-count:])) + below[-count:]
    above_vals = above[:count] + [np.nan] * (count - len(above[:count]))
    return below_vals, above_vals


def nearest_support_resistance(levels, ref_price):
    levels = sorted(set(levels))
    support = max([x for x in levels if x < ref_price], default=np.nan)
    resistance = min([x for x in levels if x > ref_price], default=np.nan)
    return support, resistance


def build_row(symbol, anchor_price, current_close, levels):
    ltp = anchor_price
    below_vals, above_vals = nearest_levels(levels, ltp, count=3)
    support, resistance = nearest_support_resistance(levels, ltp)
    pct_change = ((current_close - anchor_price) / anchor_price * 100.0) if anchor_price else np.nan
    signal = "Long" if not pd.isna(pct_change) and pct_change >= 0 else "Short"
    support_gap_pct = ((ltp - support) / ltp * 100.0) if not pd.isna(support) and ltp else np.nan
    resistance_gap_pct = ((resistance - ltp) / ltp * 100.0) if not pd.isna(resistance) and ltp else np.nan
    return {
        "Symbol": symbol, "Anchor_915": round(anchor_price, 2), "LTP": round(ltp, 2),
        "Current_Close": round(current_close, 2), "% Change": round(pct_change, 2) if not pd.isna(pct_change) else np.nan,
        "Signal": signal, "Conf_Below-3": below_vals[0], "Conf_Below-2": below_vals[1], "Conf_Below-1": below_vals[2],
        "Conf_Above-1": above_vals[0], "Conf_Above-2": above_vals[1], "Conf_Above-3": above_vals[2],
        "Support": support, "Resistance": resistance,
        "Support_Gap_Pct": round(support_gap_pct, 2) if not pd.isna(support_gap_pct) else np.nan,
        "Resistance_Gap_Pct": round(resistance_gap_pct, 2) if not pd.isna(resistance_gap_pct) else np.nan,
    }


def scan_universe(fyers, symbol_list):
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
            levels = extract_volume_profile_levels(daily)
            if not levels:
                logger.info(f"Skipping {sym}: no Volume Profile nodes found.")
                continue
            current_close = safe_float(daily["close"].iloc[-1])
            if pd.isna(current_close):
                logger.info(f"Skipping {sym}: invalid close.")
                continue
            rows.append(build_row(sym, anchor_price, current_close, levels))
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
        return f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top:25px;'>{title}</h3><p style='color:#94a3b8; font-family:sans-serif;'>No candidates found today.</p>"
    table_html = f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top:25px;'>{title}</h3><table style='border-collapse:collapse; width:100%; font-family:sans-serif; font-size:13px; text-align:left; background-color:#0f172a;'>"
    table_html += "<tr style='background-color:#1e293b; color:#f1f5f9;'>" + "".join([f"<th style='padding:10px; border:1px solid #334155;'>{c}</th>" for c in cols]) + "</tr>"
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
        msg["Subject"] = f"Volume Profile Trade Setups - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        html = (
            "<html><body style='background-color:#030712; padding:20px; font-family:sans-serif;'>"
            "<h2 style='color:#e2e8f0;'>Volume Profile Dashboard</h2>"
            f"{build_html_table(index_df, 'Market Index Volume Profile', EMAIL_DISPLAY_COLS)}"
            f"{build_html_table(long_df, 'F&O Long Candidates (Ordered by Proximity to Support)', EMAIL_DISPLAY_COLS)}"
            f"{build_html_table(short_df, 'F&O Short Candidates (Ordered by Proximity to Resistance)', EMAIL_DISPLAY_COLS)}"
            "</body></html>"
        )
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, recipients, msg.as_string())
        logger.info("Volume Profile Email sent successfully.")
    except Exception as e:
        logger.error(f"Email Failed: {e}")


def main():
    fyers = init_fyers()
    if not fyers:
        return
    logger.info("Starting Index Scan...")
    index_df = scan_universe(fyers, cfg.index_symbols)
    logger.info("Fetching live F&O stock universe from Fyers Master CSV...")
    live_stock_symbols = get_live_fno_symbols()
    logger.info(f"Starting F&O Stock Scan on {len(live_stock_symbols)} symbols...")
    stock_df = scan_universe(fyers, live_stock_symbols)
    long_stocks = pd.DataFrame(columns=RESULT_COLS)
    short_stocks = pd.DataFrame(columns=RESULT_COLS)
    if not stock_df.empty:
        long_stocks = stock_df[stock_df["Signal"] == "Long"].copy()
        short_stocks = stock_df[stock_df["Signal"] == "Short"].copy()
        if not long_stocks.empty:
            long_stocks = long_stocks.sort_values(by=["Support_Gap_Pct", "% Change"], ascending=[True, False], na_position="last")
        if not short_stocks.empty:
            short_stocks = short_stocks.sort_values(by=["Resistance_Gap_Pct", "% Change"], ascending=[True, True], na_position="last")
    if not index_df.empty:
        index_df = index_df.sort_values(by=["% Change", "Symbol"], ascending=[False, True], na_position="last")
    send_email(index_df, long_stocks, short_stocks)


if __name__ == "__main__":
    main()
