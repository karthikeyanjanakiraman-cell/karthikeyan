#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_SCANNER.py - Cumulative Calendar-Based Climax Scanner
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================

# Standardized column names to prevent NameError
DISPLAY_COLS = [
    "Symbol", "LTP", "1-Day (T/B)", "3-Day (T/B)", 
    "1-Week (T/B)", "1-Month (T/B)", "3-Month (T/B)", "6-Month (T/B)"
]

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

# Logger Setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)
warnings.filterwarnings("ignore")

# ==========================================
# HELPERS
# ==========================================

def format_tb_pair(top, bottom):
    if pd.isna(top) or pd.isna(bottom): return "-"
    return f"{float(top):.2f} / {float(bottom):.2f}"

def init_fyers():
    try: 
        return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e: 
        logger.error(f"INIT Failed: {e}")
        return None

def get_history(fyers, symbol, res, days):
    try:
        now = datetime.now()
        start, end = (now - timedelta(days=days)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")
        res_data = fyers.history(data={"symbol": symbol, "resolution": res, "date_format": "1", "range_from": start, "range_to": end, "cont_flag": "1"})
        if res_data and "candles" in res_data:
            df = pd.DataFrame(res_data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df
        return None
    except Exception as e: 
        logger.error(f"History Fetch Failed for {symbol}: {e}")
        return None

# ==========================================
# CORE LOGIC
# ==========================================

def scan_fno_universe(fyers):
    rows = []
    for sym in cfg.index_symbols:
        # Fetch 200 days to ensure enough data
        daily = get_history(fyers, sym, "D", 200) 
        if daily is not None and not daily.empty:
            anchor_date = daily["timestamp"].max()
            ltp = float(daily["close"].iloc[-1])
            
            row = {"Symbol": sym, "LTP": ltp}
            
            # Cumulative Calendar Lookback windows
            windows = [
                ("1-Day", 1), ("3-Day", 3), ("1-Week", 7), 
                ("1-Month", 30), ("3-Month", 90), ("6-Month", 180)
            ]
            
            for label, days_back in windows:
                start_date = anchor_date - timedelta(days=days_back)
                df_s = daily[daily['timestamp'] >= start_date]
                
                if df_s.empty: continue
                # Climax = High Volume or Wide Range
                idx_val = df_s["volume"].idxmax() if (df_s["volume"]>0).any() else (df_s["high"]-df_s["low"]).idxmax()
                c = df_s.loc[idx_val]
                
                row[f"{label} (T/B)"] = format_tb_pair(c["high"], c["low"])
            rows.append(row)
    return pd.DataFrame(rows)

def send_email(dashboard_df):
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = cfg.sender_email, cfg.recipient_email, f"Index Climax Dashboard - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Build Table
        table_html = "<table border='1' style='border-collapse: collapse; width: 100%; font-family: sans-serif;'>"
        table_html += "<tr>" + "".join([f"<th style='padding:8px; background-color:#eee;'>{c}</th>" for c in DISPLAY_COLS]) + "</tr>"
        for _, row in dashboard_df.iterrows():
            table_html += "<tr>" + "".join([f"<td style='padding:8px;'>{row.get(c, '-')}</td>" for c in DISPLAY_COLS]) + "</tr>"
        table_html += "</table>"
        
        msg.attach(MIMEText(f"<html><body>{table_html}</body></html>", "html"))
        
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, cfg.recipient_email, msg.as_string())
        logger.info("Email sent successfully.")
    except Exception as e: 
        logger.error(f"Email Failed: {e}")

def main():
    fyers = init_fyers()
    if not fyers: return
    
    spot_df = scan_fno_universe(fyers)
    if spot_df.empty: 
        logger.warning("No data generated.")
        return
        
    # Ensure all columns exist, even if empty
    for col in DISPLAY_COLS:
        if col not in spot_df.columns: spot_df[col] = "-"
    
    send_email(spot_df)

if __name__ == "__main__": main()
