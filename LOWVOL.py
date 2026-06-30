#!/usr/bin/env python3
"""
FO_FNO_FYERS_CONFLUENCE_EMAIL.py - Structural Confluence Screener (High Volume + Low Volume Overlap)
"""

import os
import sys
import logging
import warnings
import calendar
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

class Config:
    def __init__(self):
        self.client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        self.access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
        self.sender_password = os.environ.get("SENDER_PASSWORD", "password")
        self.recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")
        self.index_symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]

cfg = Config()

EMAIL_DISPLAY_COLS = [
    "Symbol", "% Change",
    "Conf_Below-3", "Conf_Below-2", "Conf_Below-1", 
    "LTP", 
    "Conf_Above-1", "Conf_Above-2", "Conf_Above-3"
]

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)
warnings.filterwarnings("ignore")

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
        return ""

def init_fyers():
    try:
        return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e:
        logger.warning(f"INIT Failed: {e}")
        return None

def get_history(fyers, symbol, res, days):
    try:
        now = datetime.now()
        start, end = (now - timedelta(days=days)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")
        res_data = fyers.history(data={
            "symbol": symbol,
            "resolution": res,
            "date_format": "1",
            "range_from": start,
            "range_to": end,
            "cont_flag": "1",
        })
        if res_data and "candles" in res_data:
            df = pd.DataFrame(res_data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df
        return None
    except Exception:
        return None

def scan_fno_universe(fyers):
    lookback_days = 150
    dedupe_pct = 0.005
    # MODIFIED: Widened match tolerance to 0.5% to capture Nifty 50 and Sensex overlaps
    match_tolerance = 0.005  
    rows = []

    for sym in cfg.index_symbols:
        daily = get_history(fyers, sym, "D", lookback_days)
        if daily is None or daily.empty or len(daily) < 10:
            continue

        ltp = float(daily["close"].iloc[-1])
        prev_close = float(daily["close"].iloc[-2])
        pct_ch = ((ltp - prev_close) / prev_close) * 100

        # Filter out 0-volume data glitches
        valid_daily = daily[daily["volume"] > 0]
        
        # 1. Map High Volume Structure (MODIFIED: Look at top 60 days)
        hv_work = valid_daily.sort_values(["volume", "high"], ascending=[False, False]).head(60)
        
        # 2. Map Low Volume Voids (MODIFIED: Look at quietest 60 days)
        lv_work = valid_daily.sort_values(["volume", "high"], ascending=[True, False]).head(60)

        # Helper to extract clean deduplicated boundary bands
        def extract_bands(work_df):
            candidates = []
            for _, c in work_df.iterrows():
                top_b, bot_b = float(c["high"]), float(c["low"])
                mid_b = (top_b + bot_b) / 2.0
                keep = True
                for s in candidates:
                    s_mid = (s["top"] + s["bottom"]) / 2.0
                    if abs(mid_b - s_mid) / max(abs(mid_b), 1.0) <= dedupe_pct:
                        keep = False
                        break
                if keep:
                    candidates.append({"top": top_b, "bottom": bot_b})
            return candidates

        hv_bands = extract_bands(hv_work)
        lv_bands = extract_bands(lv_work)

        unique_confluences = []

        # 3. Cross-Reference: Find exact overlaps between High Volume and Low Volume boundaries
        for hv in hv_bands:
            for lv in lv_bands:
                matches = [
                    (hv["top"], lv["bottom"]),
                    (hv["top"], lv["top"]),
                    (hv["bottom"], lv["bottom"]),
                    (hv["bottom"], lv["top"])
                ]
                for hv_price, lv_price in matches:
                    diff = abs(hv_price - lv_price) / max(hv_price, 1.0)
                    if diff <= match_tolerance:
                        conf_p = (hv_price + lv_price) / 2.0
                        
                        # Ensure we don't save duplicate Confluence lines sitting next to each other
                        is_duplicate = any(abs(conf_p - u) / max(conf_p, 1.0) <= dedupe_pct for u in unique_confluences)
                        if not is_duplicate:
                            unique_confluences.append(conf_p)

        # 4. Split and Sort based on current Last Traded Price (LTP)
        above_ltp = sorted([p for p in unique_confluences if p > ltp])
        below_ltp = sorted([p for p in unique_confluences if p < ltp], reverse=True)

        row = {
            "Symbol": sym, 
            "LTP": ltp, 
            "% Change": pct_ch,
        }

        # Save closest 3 targets ABOVE price
        for i in range(3):
            row[f"Conf_Above-{i+1}"] = format_value(above_ltp[i]) if i < len(above_ltp) else "-"
            
        # Save closest 3 targets BELOW price
        for i in range(3):
            row[f"Conf_Below-{i+1}"] = format_value(below_ltp[i]) if i < len(below_ltp) else "-"

        rows.append(row)

    return pd.DataFrame(rows)

def build_html_table(df, title, cols):
    if df.empty:
        return f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top: 25px;'>{title}</h3><p style='color:#94a3b8; font-family:sans-serif;'>No candidates.</p>"
    table_html = f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top: 25px;'>{title}</h3><table style='border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 13px; text-align: left; background-color: #0f172a;'>"
    table_html += "<tr style='background-color: #1e293b; color: #f1f5f9;'>" + "".join([f"<th style='padding: 10px; border: 1px solid #334155;'>{c}</th>" for c in cols]) + "</tr>"
    for i, (_, row) in enumerate(df.iterrows()):
        bg_row = "#0f172a" if i % 2 == 0 else "#1e293b"
        table_html += f"<tr style='background-color: {bg_row}; color: #e2e8f0;'>"
        for c in cols:
            val = row.get(c, "-")
            style = "padding: 8px; border: 1px solid #334155;"
            if c == "% Change":
                val_str = format_change(val)
                style += " color: #4ade80; font-weight: bold;" if float(row["% Change"]) > 0 else " color: #f87171; font-weight: bold;"
                table_html += f"<td style='{style}'>{val_str}</td>"
            else:
                table_html += f"<td style='{style}'>{val}</td>"
        table_html += "</tr>"
    return table_html + "</table>"

def send_email(dashboard_df):
    try:
        msg = MIMEMultipart()
        msg["From"] = cfg.sender_email
        msg["To"] = cfg.recipient_email
        msg["Subject"] = f"Index Confluence Pivot Points - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        html = (
            "<body style='background-color: #030712; padding: 20px; font-family: sans-serif;'>"
            + build_html_table(dashboard_df, "Ultimate Confluence Zones (HV/LV Overlap)", EMAIL_DISPLAY_COLS)
            + "</body>"
        )
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, cfg.recipient_email, msg.as_string())
        logger.info("Confluence Email sent successfully.")
    except Exception as e:
        logger.error(f"Email Failed: {e}")

def main():
    fyers = init_fyers()
    if not fyers:
        return
    spot_df = scan_fno_universe(fyers)
    if spot_df.empty:
        logger.warning("No spot data generated.")
        return
    
    send_email(spot_df)

if __name__ == "__main__":
    main()
