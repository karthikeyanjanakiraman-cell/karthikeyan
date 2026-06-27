#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py - Calendar-Based Climax Scanner
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

def format_tb_pair(ltp, top, bottom):
    if pd.isna(top) or pd.isna(bottom): return "-"
    t_str, b_str, ltp_val = f"{float(top):.2f}", f"{float(bottom):.2f}", float(ltp)
    if ltp_val > float(top): t_str = f"<span style='color: #4ade80; font-weight: bold;'>{t_str}</span>"
    if ltp_val < float(bottom): b_str = f"<span style='color: #f87171; font-weight: bold;'>{b_str}</span>"
    return f"{t_str} <span style='color: #64748b;'>/</span> {b_str}"

def format_value(col, val):
    if pd.isna(val) or val in [float("inf"), float("-inf")]: return ""
    if "(T/B)" in col: return str(val)
    if col in ["Trigger_TF", "Signal_Type", "Climax_Date"]: return str(val)
    if col == "Breach_Days": return str(int(val)) if not pd.isna(val) else ""
    if col == "% Change": return f"{float(val):.2f}%"
    if isinstance(val, (int, float, np.integer, np.floating)): return f"{float(val):.2f}"
    return str(val)

def get_index_meta(symbol):
    if "NIFTY50" in symbol: return "NSE", "NIFTY", 50
    if "NIFTYBANK" in symbol: return "NSE", "BANKNIFTY", 100
    return "BSE", "SENSEX", 100

def get_expiry_details(symbol):
    today = datetime.now().date()
    if "NIFTYBANK" in symbol:
        def last_thu(y, m):
            last = calendar.monthrange(y, m)[1]
            d = datetime(y, m, last).date()
            return d - timedelta(days=(d.weekday() - 3) % 7)
        expiry = last_thu(today.year, today.month)
        if today > expiry:
            m = today.month + 1 if today.month < 12 else 1
            y = today.year if today.month < 12 else today.year + 1
            expiry = last_thu(y, m)
        return True, expiry
    days_ahead = (3 if "NIFTY" in symbol else 4) - today.weekday()
    if days_ahead < 0: days_ahead += 7
    return False, today + timedelta(days=days_ahead)

def get_options_data(symbol, ltp, side):
    exch, base_name, interval = get_index_meta(symbol)
    atm_strike = int(round(ltp / interval) * interval)
    is_monthly, expiry = get_expiry_details(symbol)
    yy = expiry.strftime("%y")
    expiry_str = expiry.strftime("%b").upper() if is_monthly else f"{['1','2','3','4','5','6','7','8','9','O','N','D'][expiry.month-1]}{expiry.strftime('%d')}"
    opt_type = "CE" if side == "long" else "PE"
    strikes = [atm_strike + (i * interval) for i in range(-10, 11)]
    return [f"{exch}:{base_name}{yy}{expiry_str}{s}{opt_type}" for s in strikes]

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
            anchor_date = daily["timestamp"].max() # Market-aware anchor
            ltp = float(daily["close"].iloc[-1])
            prev_close = float(daily["close"].iloc[-2])
            
            row = {"Symbol": sym, "LTP": ltp, "% Change": ((ltp - prev_close)/prev_close*100)}
            windows = [("1D", 1), ("3D", 3), ("1W", 7), ("1M", 30), ("3M", 90), ("6M", 180)]
            
            for label, days in windows:
                start_date = anchor_date - timedelta(days=days)
                df_s = daily[daily['timestamp'] >= start_date]
                
                if df_s.empty: continue
                idx_val = df_s["volume"].idxmax() if (df_s["volume"]>0).any() else (df_s["high"]-df_s["low"]).idxmax()
                c = df_s.loc[idx_val]
                
                row.update({f"T_{label}": float(c["high"]), f"B_{label}": float(c["low"]), f"D_{label}": str(c["timestamp"].date())})
                
                # Simple Breach Logic
                bd_l, bd_s = 999, 999
                if ltp > float(c["high"]):
                    breach = daily[daily['close'] <= float(c["high"])]
                    if not breach.empty: bd_l = (anchor_date - breach.iloc[-1]['timestamp']).days
                if ltp < float(c["low"]):
                    breach = daily[daily['close'] >= float(c["low"])]
                    if not breach.empty: bd_s = (anchor_date - breach.iloc[-1]['timestamp']).days
                
                row.update({f"Days_L_{label}": bd_l, f"Days_S_{label}": bd_s})
            rows.append(row)
    return pd.DataFrame(rows)

def build_dashboard_and_candidates(df):
    dashboard_rows, valid_long, valid_short = [], [], []
    for _, row in df.iterrows():
        r_dict = row.to_dict()
        for tf in ["1D", "3D", "1W", "1M", "3M", "6M"]:
            lbl = f"{tf.replace('D', '-Day').replace('W', '-Week').replace('M', '-Month')}"
            r_dict[f"{lbl} (T/B)"] = format_tb_pair(row["LTP"], row.get(f"T_{tf}"), row.get(f"B_{tf}"))
        dashboard_rows.append(r_dict)
        
        for tf in ["1D", "3D", "1W", "1M", "3M", "6M"]:
            t, b, d = row.get(f"T_{tf}"), row.get(f"B_{tf}"), row.get(f"D_{tf}")
            if pd.notna(t) and row["LTP"] > t and row.get(f"Days_L_{tf}", 999) <= 5:
                r_dict.update({"Trigger_TF": tf, "Climax_Date": d, "Target_Options": get_options_data(row["Symbol"], row["LTP"], "long"), "Signal_Type": "Active Trend"})
                valid_long.append(r_dict); break
    return pd.DataFrame(dashboard_rows), pd.DataFrame(valid_long), pd.DataFrame(valid_short)

def send_email(dashboard_df, long_df, short_df, csv_file):
    # (Email logic remains the same, assuming standard structure)
    pass 

def main():
    fyers = init_fyers()
    if not fyers: return
    spot_df = scan_fno_universe(fyers)
    if spot_df.empty: return
    dashboard_df, long_df, short_df = build_dashboard_and_candidates(spot_df)
    # ... Add output logic ...

if __name__ == "__main__": main()
