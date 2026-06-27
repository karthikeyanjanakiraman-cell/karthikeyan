#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py - Calendar-Based Climax Scanner with Email
"""

import os
import sys
import logging
import warnings
import calendar
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ==========================================
# CONFIGURATION
# ==========================================

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
    "Symbol", "LTP", "Trigger_TF", 
    "1-Day (T/B)", "3-Day (T/B)", "1-Week (T/B)", 
    "1-Month (T/B)", "3-Month (T/B)", "6-Month (T/B)"
]

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

def format_value(col, val):
    if pd.isna(val): return ""
    if "(T/B)" in col: return str(val)
    if isinstance(val, (int, float, np.integer, np.floating)): return f"{float(val):.2f}"
    return str(val)

def init_fyers():
    try: return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e: logger.warning(f"INIT Failed: {e}"); return None

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
    except Exception as e: return None

# ==========================================
# SCANNER LOGIC
# ==========================================

def scan_fno_universe(fyers):
    rows = []
    for sym in cfg.index_symbols:
        daily = get_history(fyers, sym, "D", 200) 
        if daily is not None and not daily.empty:
            anchor_date = daily["timestamp"].max()
            ltp = float(daily["close"].iloc[-1])
            prev_close = float(daily["close"].iloc[-2])
            
            row = {"Symbol": sym, "LTP": ltp, "% Change": ((ltp - prev_close)/prev_close*100)}
            windows = [("1D", 1), ("3D", 3), ("1W", 7), ("1M", 30), ("3M", 90), ("6M", 180)]
            
            for label, days_back in windows:
                start_date = anchor_date - timedelta(days=days_back)
                df_s = daily[daily['timestamp'] >= start_date]
                
                if df_s.empty: continue
                idx_val = df_s["volume"].idxmax() if (df_s["volume"]>0).any() else (df_s["high"]-df_s["low"]).idxmax()
                c = df_s.loc[idx_val]
                
                row.update({
                    f"T_{label}": float(c["high"]), 
                    f"B_{label}": float(c["low"]), 
                    f"D_{label}": str(c["timestamp"].date())
                })
            rows.append(row)
    return pd.DataFrame(rows)

def build_html_table(df, title, cols):
    table_html = f"<h3>{title}</h3><table border='1' style='border-collapse: collapse; width: 100%; font-family: sans-serif;'>"
    table_html += "<tr>" + "".join([f"<th style='padding:8px;'>{c}</th>" for c in cols]) + "</tr>"
    for _, row in df.iterrows():
        table_html += "<tr>"
        for c in cols:
            val = row.get(c)
            table_html += f"<td style='padding:8px;'>{format_value(c, val)}</td>"
        table_html += "</tr>"
    return table_html + "</table>"

def send_email(dashboard_df):
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = cfg.sender_email, cfg.recipient_email, f"Index Climax Dashboard - {datetime.now().strftime('%Y-%m-%d')}"
        html = f"<html><body>{build_html_table(dashboard_df, 'Market Dashboard', DISPLAY_COLS)}</body></html>"
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, cfg.recipient_email, msg.as_string())
        logger.info("Email sent successfully.")
    except Exception as e: logger.error(f"Email Failed: {e}")

def main():
    fyers = init_fyers()
    if not fyers: return
    spot_df = scan_fno_universe(fyers)
    if spot_df.empty: return
    
    # Process display
    dashboard_rows = []
    for _, row in spot_df.iterrows():
        r_dict = row.to_dict()
        for tf in ["1D", "3D", "1W", "1M", "3M", "6M"]:
            lbl = f"{tf.replace('D', '-Day').replace('W', '-Week').replace('M', '-Month')}"
            r_dict[f"{lbl} (T/B)"] = format_tb_pair(row.get(f"T_{tf}"), row.get(f"B_{tf}"))
        dashboard_rows.append(r_dict)
    
    dashboard_df = pd.DataFrame(dashboard_rows)
    send_email(dashboard_df)

if __name__ == "__main__": main()
