#!/usr/bin/env python3
"""
OPTIONS_OI.py - Comprehensive Calendar-Based Index & Options Dashboard
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

EMAIL_OPT_COLS = [
    "Symbol", "LTP", "% Change", "Signal_Type", 
    "Climax_Date", "Climax_Range (T/B)", "Breach_Days"
]

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ==========================================
# HELPERS
# ==========================================

def format_tb_pair(top, bottom):
    if pd.isna(top) or pd.isna(bottom): return "-"
    return f"{float(top):.2f} / {float(bottom):.2f}"

def format_value(col, val):
    if pd.isna(val) or val in [float("inf"), float("-inf")]: return ""
    if "(T/B)" in col: return str(val)
    if col == "% Change": return f"{float(val):.2f}%"
    if isinstance(val, (int, float, np.integer, np.floating)): return f"{float(val):.2f}"
    return str(val)

def get_index_meta(symbol):
    if "NIFTY50" in symbol: return "NSE", "NIFTY", 50
    if "NIFTYBANK" in symbol: return "NSE", "BANKNIFTY", 100
    return "BSE", "SENSEX", 100

def get_underlying_spot(opt_symbol):
    if "BANKNIFTY" in opt_symbol: return "NSE:NIFTYBANK-INDEX"
    if "NIFTY" in opt_symbol: return "NSE:NIFTY50-INDEX"
    return "BSE:SENSEX-INDEX"

def get_options_data(symbol, ltp, side):
    exch, base_name, interval = get_index_meta(symbol)
    atm_strike = int(round(ltp / interval) * interval)
    # Basic expiry logic for options
    expiry_str = (datetime.now() + timedelta(days=7)).strftime("%y%m%d") 
    opt_type = "CE" if side == "long" else "PE"
    strikes = [atm_strike + (i * interval) for i in range(-5, 6)]
    return [f"{exch}:{base_name}{expiry_str}{s}{opt_type}" for s in strikes]

# ==========================================
# CORE SCANNING ENGINE
# ==========================================

def init_fyers():
    try: return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e: logging.error(f"INIT Failed: {e}"); return None

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

def scan_fno_universe(fyers):
    rows = []
    for sym in cfg.index_symbols:
        daily = get_history(fyers, sym, "D", 200)
        if daily is not None and not daily.empty:
            anchor_date = daily["timestamp"].max()
            ltp = float(daily["close"].iloc[-1])
            prev_close = float(daily["close"].iloc[-2])
            
            row = {"Symbol": sym, "LTP": ltp, "% Change": ((ltp - prev_close)/prev_close*100)}
            windows = [("1-Day", 1), ("3-Day", 3), ("1-Week", 7), ("1-Month", 30), ("3-Month", 90), ("6-Month", 180)]
            
            for label, days_back in windows:
                start_date = anchor_date - timedelta(days=days_back)
                df_s = daily[daily['timestamp'] >= start_date]
                if df_s.empty: continue
                idx_val = df_s["volume"].idxmax() if (df_s["volume"]>0).any() else (df_s["high"]-df_s["low"]).idxmax()
                c = df_s.loc[idx_val]
                row[f"{label} (T/B)"] = format_tb_pair(c["high"], c["low"])
                row[f"T_{label}"] = float(c["high"])
                row[f"B_{label}"] = float(c["low"])
            rows.append(row)
    return pd.DataFrame(rows)

def scan_options_universe(fyers, symbols):
    rows = []
    for sym in symbols:
        daily = get_history(fyers, sym, "D", 30)
        if daily is not None and not daily.empty:
            ltp = float(daily["close"].iloc[-1])
            max_idx = daily["volume"].idxmax()
            c = daily.loc[max_idx]
            rows.append({"Symbol": sym, "LTP": ltp, "T_LOC": float(c["high"]), "B_LOC": float(c["low"]), "Climax_Date": str(c["timestamp"].date())})
    return pd.DataFrame(rows)

# ==========================================
# OUTPUT & EMAIL
# ==========================================

def build_html_table(df, cols):
    table_html = "<table border='1' style='border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 12px;'>"
    table_html += "<tr style='background-color:#eee;'>" + "".join([f"<th style='padding:5px;'>{c}</th>" for c in cols]) + "</tr>"
    for _, row in df.iterrows():
        table_html += "<tr>" + "".join([f"<td style='padding:5px;'>{format_value(c, row.get(c, '-'))}</td>" for c in cols]) + "</tr>"
    table_html += "</table>"
    return table_html

def send_email(dashboard_df, ce_df, pe_df, csv_file):
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = cfg.sender_email, cfg.recipient_email, f"Index Climax Dashboard - {datetime.now().strftime('%Y-%m-%d')}"
        
        html = f"<html><body><h3>Market Dashboard</h3>{build_html_table(dashboard_df, EMAIL_DISPLAY_COLS)}<h3>CE Climax</h3>{build_html_table(ce_df, EMAIL_OPT_COLS)}<h3>PE Climax</h3>{build_html_table(pe_df, EMAIL_OPT_COLS)}</body></html>"
        msg.attach(MIMEText(html, "html"))
        
        with open(csv_file, "rb") as f:
            part = MIMEBase("application", "octet-stream"); part.set_payload(f.read()); encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_file)}"); msg.attach(part)
            
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls(); s.login(cfg.sender_email, cfg.sender_password); s.sendmail(cfg.sender_email, cfg.recipient_email, msg.as_string())
        logging.info("Email sent.")
    except Exception as e: logging.error(f"Email Failed: {e}")

def main():
    fyers = init_fyers()
    if not fyers: return
    
    # 1. Scan
    spot_df = scan_fno_universe(fyers)
    if spot_df.empty: return
    
    # 2. Mocking Options Check (Replace with your actual filtering logic)
    all_opt = ["NSE:BANKNIFTY26JUL57300CE", "NSE:BANKNIFTY26JUL57300PE"] 
    opt_df = scan_options_universe(fyers, all_opt)
    
    # 3. CSV Output
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_file = f"climax_scan_{ts}.csv"
    spot_df.to_csv(csv_file, index=False)
    
    # 4. Email
    send_email(spot_df, opt_df[opt_df["Symbol"].str.contains("CE")], opt_df[opt_df["Symbol"].str.contains("PE")], csv_file)

if __name__ == "__main__": main()
